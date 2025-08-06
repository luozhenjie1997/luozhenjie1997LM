import yaml
import math
import numpy as np
import string
import random
import os
import lmdb
import torch
import torch.nn.functional as F
from easydict import EasyDict
from .chemical import cos_ideal_NCAC, backbone_atom_coordinates, BBHeavyAtom
from openfold.utils import rigid_utils as ru
from model.utils import residue_constants
from model.utils import protein
from torch_scatter import scatter_add, scatter
from torch.utils.data._utils.collate import default_collate


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


# 设置随机种子
def set_seed(seed=2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def read_lmdb(db_path):
    db_conn = lmdb.open(db_path, map_size=32*(1024*1024*1024), create=False, subdir=False,
                        readonly=True, lock=False, readahead=False, meminit=False,
                        )
    return db_conn


# 为不同的网络层设置不同的权重衰减
def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        #if len(param.shape) == 1 or name.endswith(".bias"):
        """
        偏置项通常不应用权重衰减，因为它们的作用是调整激活值的均值，而不是直接参与特征提取。
        BatchNorm、LayerNorm等，这些层的参数（如缩放因子和偏移量）也不应该应用权重衰减。因为这些参数的作用是调整激活值的分布，而不是直接参与特征提取。
        对这些参数应用权重衰减可能会破坏归一化的效果，影响模型的训练和收敛。
        """
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]

def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss

Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def process_dic(state_dict):
    new_state_dict = {}
    for k,v in state_dict.items():
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def apply_mask(self, aatype_diff, aatype_0, diff_mask):
    return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0


def calc_distogram(pos, min_bin, max_bin, num_bins):
    dists_2d = torch.linalg.norm(
        pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def sample_from(c):
    """sample from c"""
    N, L, K = c.size()
    c = c.view(N*L, K) + 1e-8
    x = torch.multinomial(c, 1).view(N, L)
    return x


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


# 在梯度检查点中使用
def create_custom_forward(module, **kwargs):
    def custom_forward(*inputs):
        return module(*inputs, **kwargs)
    return custom_forward


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}

    else:
        return obj


def parse_chain_feats(chain_feats, scale_factor=1.):
    chain_feats['bb_mask'] = chain_feats['atom_mask'][:, BBHeavyAtom.CA]
    bb_pos = chain_feats['atom_positions'][:, BBHeavyAtom.CA]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5)
    centered_pos = chain_feats['atom_positions'] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats['atom_positions'] = scaled_pos * chain_feats['atom_mask'][..., None]
    chain_feats['bb_positions'] = chain_feats['atom_positions'][:, BBHeavyAtom.CA]
    return chain_feats


# 将一个批次中的各个样本进行对齐，使得各个样本中需要填充的序列长度一致
class PaddingCollate(object):
    def __init__(self, length_ref_key='aa', pad_values={}, no_padding=[], eight=True):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.no_padding = no_padding
        self.eight = eight  # 如果为True，则在确定最大长度后，会将长度向上取整为8的倍数（这通常有助于提高 GPU 运算效率或与某些模型结构对齐）

    @staticmethod
    def _pad_last(x, n, value=0):
        """
        将输入x在第一维（通常代表序列长度）上填充到指定长度n
        """
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            # 构造一个形状为[n - x.size(0)] + x.shape[1:]的张量，所有元素都填充为指定的value，并与x在第一维上拼接
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        """
        生成一个长度为n的布尔类型掩码，其中前l个位置为True（代表真实数据），后n - l个位置为False（代表填充部分）
        """
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    @staticmethod
    def _get_common_keys(list_of_dict):
        """
        返回一个批次的数据中共有的键
        """
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])  # 获取当前批次中最大的序列长度
        # 所有样本共有的键。确保后续的填充和收集操作只处理所有样本都存在的数据项，从而避免因某个样本缺失某个键而导致出错或数据不一致的问题
        keys = self._get_common_keys(data_list)

        # 最大长度对齐到为8的倍数
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8

        data_list_padded = []
        for data in data_list:
            # 如果该键不在no_padding列表中，则使用_pad_last方法将对应的值填充到最大长度，否则保持原样
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k)) if k not in self.no_padding else v
                for k, v in data.items()
                if k in keys
            }
            data_padded['res_mask'] = (self._get_pad_mask(data[self.length_ref_key].size(0), max_length))
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)  # 打包成一个批次返回




# 更复杂的版本将误差分为 CA-N 和 CA-C（提供更准确的 CB 位置）
# 它返回从局部帧到全局帧的刚性变换
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):
    # N, Ca, C - [B, L, 3]
    # R - [B, L, 3, 3], det(R)=1, inv(R) = R.T, R 是旋转矩阵
    B, L = N.shape[:2]

    """使用施密特正交化获得旋转矩阵"""
    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)
    u2 = v2 - (torch.einsum('bli, bli -> bl', e1, v2)[..., None] * e1)
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.cat([e1[..., None], e2[..., None], e3[..., None]], axis=-1)  # [B,L,3,3] - 旋转矩阵

    """如果non_ideal为True，将考虑更复杂的旋转修正"""
    if non_ideal:
        v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + eps)  # 将从 Cα 指向 N的向量正则化
        cosref = torch.sum(e1 * v2, dim=-1)  # 当前N-CA-C键角的余弦值
        costgt = cos_ideal_NCAC.item()  # 理想键角的余弦值
        # 修正量cos2del结合了当前与理想角度的信息，利用余弦之间的关系以及两者的正交分量（通过平方根项）来构造一个衰减系数
        cos2del = torch.clamp(cosref * costgt + torch.sqrt((1 - cosref * cosref) * (1 - costgt * costgt) + eps), min=-1.0, max=1.0)
        # 利用半角公式计算出cosdel和sindel，其构成了一个在二维平面上（对应于局部坐标系中的 e1-e2 平面）调整旋转角度的因子
        cosdel = torch.sqrt(0.5 * (1 + cos2del) + eps)
        sindel = torch.sign(costgt - cosref) * torch.sqrt(1 - 0.5 * (1 + cos2del) + eps)
        # 仅在 e1 和 e2 方向上应用该修正
        Rp = torch.eye(3, device=N.device).repeat(B, L, 1, 1)
        Rp[:, :, 0, 0] = cosdel
        Rp[:, :, 0, 1] = -sindel
        Rp[:, :, 1, 0] = sindel
        Rp[:, :, 1, 1] = cosdel

        # “拉回”当前的 N–Cα–C 键角
        R = torch.einsum('blij,bljk->blik', R, Rp)

    # 返回旋转矩阵和平移向量
    return R, Ca


def calc_bb_fape_loss(pred, true, mask_2d, d_clamp=10.0, A=10.0, gamma=1.0, eps=1e-6):
    # 计算蛋白质主链原子（N、Ca、C）之间的相对位移向量。并考虑非理想修正
    def get_t(N, Ca, C, non_ideal=False, eps=1e-5):
        I, B, L = N.shape[:3]
        Rs, Ts = rigid_from_3_points(N.view(I * B, L, 3), Ca.view(I * B, L, 3), C.view(I * B, L, 3), non_ideal=non_ideal, eps=eps)
        Rs = Rs.view(I, B, L, 3, 3)
        Ts = Ts.view(I, B, L, 3)
        t = Ts[:, :, None] - Ts[:, :, :, None]  # t[0,1] = residue 0 -> residue 1 vector
        return torch.einsum('iblkj, iblmk -> iblmj', Rs, t)  # (I,B,L,L,3)

    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]

    true = true.unsqueeze(0)
    # 计算真实结构和预测结构的相对位移向量。实验数据并非理想化的数据，因此non_ideal设置为True
    t_tilde_ij = get_t(true[:, :, :, 0], true[:, :, :, 1], true[:, :, :, 2], non_ideal=True)
    t_ij = get_t(pred[:, :, :, 0], pred[:, :, :, 1], pred[:, :, :, 2])
    # 计算相对位移向量的差异
    difference = torch.sqrt(torch.square(t_tilde_ij - t_ij).sum(dim=-1) + eps)

    if d_clamp != None:
        difference = torch.clamp(difference, max=d_clamp)
    loss = difference / A  # (I, B, L, L)
    loss = (mask_2d[None] * loss).sum(dim=(1, 2, 3)) / (mask_2d.sum() + eps)
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()
    tot_loss = (w_loss * loss).sum()
    return tot_loss


def calc_BB_bond_geom_in_ideal(pred, idx, mask=None, eps=1e-6, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255,
                               sig_len=0.02, sig_ang=0.05):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''

    def cosangle(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)

    def length(a, b):
        return torch.norm(a - b, dim=-1)
    B, L = pred.shape[:2]

    bonded = (idx[:, 1:] - idx[:, :-1]) == 1  # 相邻的残基才会计算键长和键角损失
    if mask is not None:
        valid_mask = mask[:, 1:] * mask[:, :-1]
        final_mask = bonded * valid_mask  # 两个参与键长/键角的残基都要存在才有效
    else:
        final_mask = bonded

    # bond length: N-CA, CA-C, C-N
    blen_CN_pred = length(pred[:, :-1, 2], pred[:, 1:, 0]).reshape(B, L - 1)  # (B, L-1)
    CN_loss = final_mask * torch.clamp(torch.square(blen_CN_pred - ideal_NC) - sig_len ** 2, min=0.0)
    n_viol = (CN_loss > 0.0).sum()
    blen_loss = CN_loss.sum() / (n_viol + eps)

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:, :-1, 1], pred[:, :-1, 2], pred[:, 1:, 0]).reshape(B, L - 1)
    bang_CNCA_pred = cosangle(pred[:, :-1, 2], pred[:, 1:, 0], pred[:, 1:, 1]).reshape(B, L - 1)
    CACN_loss = final_mask * torch.clamp(torch.square(bang_CACN_pred - ideal_CACN) - sig_ang ** 2, min=0.0)
    CNCA_loss = final_mask * torch.clamp(torch.square(bang_CNCA_pred - ideal_CNCA) - sig_ang ** 2, min=0.0)
    bang_loss = CACN_loss + CNCA_loss
    n_viol = (bang_loss > 0.0).sum()
    bang_loss = bang_loss.sum() / (n_viol + eps)

    return blen_loss, bang_loss


def eval_BB_bond_geom_in_ideal(pred, idx, mask=None, eps=1e-6, ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255,
                               sig_len=0.02, sig_ang=0.05):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''

    def cosangle(A, B, C):
        AB = A - B
        BC = C - B
        ABn = torch.sqrt(torch.sum(torch.square(AB), dim=-1) + eps)
        BCn = torch.sqrt(torch.sum(torch.square(BC), dim=-1) + eps)
        return torch.clamp(torch.sum(AB * BC, dim=-1) / (ABn * BCn), -0.999, 0.999)

    def length(a, b):
        return torch.norm(a - b, dim=-1)
    B, L = pred.shape[:2]

    bonded = (idx[:, 1:] - idx[:, :-1]) == 1  # 相邻的残基才会计算键长和键角损失
    if mask is not None:
        valid_mask = mask[:, 1:] * mask[:, :-1]
        final_mask = bonded * valid_mask  # 两个参与键长/键角的残基都要存在才有效
    else:
        final_mask = bonded

    # bond length: N-CA, CA-C, C-N
    blen_CN_pred = length(pred[:, :-1, 2], pred[:, 1:, 0]).reshape(B, L - 1)  # (B, L-1)
    CN_loss = final_mask * torch.clamp(torch.square(blen_CN_pred - ideal_NC), min=0.0)
    n_viol = (CN_loss > 0.0).sum()
    blen_loss = CN_loss.sum() / (n_viol + eps)

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:, :-1, 1], pred[:, :-1, 2], pred[:, 1:, 0]).reshape(B, L - 1)
    bang_CNCA_pred = cosangle(pred[:, :-1, 2], pred[:, 1:, 0], pred[:, 1:, 1]).reshape(B, L - 1)
    CACN_loss = final_mask * torch.clamp(torch.square(bang_CACN_pred - ideal_CACN), min=0.0)
    CNCA_loss = final_mask * torch.clamp(torch.square(bang_CNCA_pred - ideal_CNCA), min=0.0)
    bang_loss = CACN_loss + CNCA_loss
    n_viol = (bang_loss > 0.0).sum()
    bang_loss = bang_loss.sum() / (n_viol + eps)

    return {
        'blen_loss': blen_loss,
        'bang_loss': bang_loss,
        "mean_blen_error": torch.mean(torch.abs(blen_CN_pred - ideal_NC)[final_mask.bool()]),
        "std_blen_error": torch.std(torch.abs(blen_CN_pred - ideal_NC)[final_mask.bool()]),
        "mean_CACN_angle": torch.mean(torch.acos(bang_CACN_pred[final_mask.bool()]) * 180 / np.pi),
        "mean_CNCA_angle": torch.mean(torch.acos(bang_CNCA_pred[final_mask.bool()]) * 180 / np.pi),
        "CACN_angle_error": torch.mean(torch.abs(bang_CACN_pred - ideal_CACN)[final_mask.bool()]),
        "CNCA_angle_error": torch.mean(torch.abs(bang_CNCA_pred - ideal_CNCA)[final_mask.bool()])
    }


def angle(a, b, c, eps=1e-6):
    '''
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    '''
    B, L = a.shape[:2]

    u1 = a - b
    u2 = c - b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B * L, 3)
    u2 = u2.reshape(B * L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(B, L, 1)  # (B,L,1)
    cos_theta = torch.matmul(u1[:, None, :], u2[:, :, None]).reshape(B, L, 1)

    return torch.cat([cos_theta, sin_theta], axis=-1)  # (B, L, 2)


def get_init_xyz(aa):
    # 使用理想坐标进行初始化
    xyz = torch.tensor([[backbone_atom_coordinates[token.item()] for token in tokens] for tokens in aa], device=aa.device)
    return torch.concat([xyz, torch.zeros((xyz.shape[0], xyz.shape[1], 12, 3), device=aa.device)], dim=-2)

to_numpy = lambda x: x.detach().cpu().numpy()

def center_zero(pos: torch.Tensor, batch_indexes: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        pos: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch_indexes: [N] batch index for each atom in sparse batch format.

    Returns:
        pos: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    assert len(pos.shape) == 2 and pos.shape[-1] == 3, "pos must have shape [N, 3]"

    means = scatter(pos, batch_indexes, dim=0, reduce="mean")
    return pos - means[batch_indexes]


@torch.no_grad()
def align_structures(
    batch_positions: torch.Tensor,
    batch_indices: torch.Tensor,
    reference_positions: torch.Tensor,
    broadcast_reference: bool = False,
):
    """
    Align structures in a ChemGraph batch to a reference, e.g. for RMSD computation. This uses the
    sparse formulation of pytorch geometric. If the ChemGraph is composed of a single system, then
    the reference can be given as a single structure and broadcasted. Returns the structure
    coordinates shifted to the geometric center and the batch structures rotated to match the
    reference structures. Uses the Kabsch algorithm (see e.g. [kabsch_align1]_). No permutation of
    atoms is carried out.

    Args:
        batch_positions (Tensor): Batch of structures (e.g. from ChemGraph) which should be aligned
          to a reference.
        batch_indices (Tensor): Index tensor mapping each node / atom in batch to the respective
          system (e.g. batch attribute of ChemGraph batch).
        reference_positions (Tensor): Reference structure. Can either be a batch of structures or a
          single structure. In the second case, broadcasting is possible if the input batch is
          composed exclusively of this structure.
        broadcast_reference (bool, optional): If reference batch contains only a single structure,
          broadcast this structure to match the ChemGraph batch. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing the centered positions of batch
          structures rotated into the reference and the centered reference batch.

    References
    ----------
    .. [kabsch_align1] Lawrence, Bernal, Witzgall:
       A purely algebraic justification of the Kabsch-Umeyama algorithm.
       Journal of research of the National Institute of Standards and Technology, 124, 1. 2019.
    """
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    # batch_positions     -> P [BN x 3]
    # reference_positions -> Q [B / BN x 3]

    if batch_positions.shape[0] != reference_positions.shape[0]:
        if broadcast_reference:
            # Get number of systems in batch and broadcast reference structure.
            # This assumes, all systems in the current batch correspond to the reference system.
            # Typically always the case during evaluation.
            num_molecules = int(torch.max(batch_indices) + 1)
            reference_positions = reference_positions.repeat(num_molecules, 1)
        else:
            raise ValueError("Mismatch in batch dimensions.")

    # Center structures at origin (takes care of translation alignment)
    batch_positions = center_zero(batch_positions, batch_indices)
    reference_positions = center_zero(reference_positions, batch_indices)

    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B x 3 x 3].
    cov = scatter_add(
        batch_positions[:, None, :] * reference_positions[:, :, None], batch_indices, dim=0
    )

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system
    # For comparison with other sources: det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix (R = V @ diag(1, 1, sign_correction) @ U.T).
    rotation_matrices = torch.bmm(v, u_t)

    # Rotate batch positions P to optimal alignment with Q (P @ R)
    batch_positions_rotated = torch.bmm(
        batch_positions[:, None, :],
        rotation_matrices[batch_indices],
    ).squeeze(1)

    return batch_positions_rotated, reference_positions, rotation_matrices


def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64)
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots

def compute_kabsch_aligned_mse_batched(pred_atoms, gt_atoms, mask=None):
    def kabsch_align(P, Q):
        """
        Kabsch alignment of two point clouds: P (pred), Q (gt), shape: [N, 3]
        """
        P_cent = P - P.mean(axis=0)
        Q_cent = Q - Q.mean(axis=0)

        H = P_cent.T @ Q_cent
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        P_aligned = (P_cent @ R) + Q.mean(axis=0)
        return P_aligned
    """
    pred_atoms, gt_atoms: [B, L, 3, 3] (batch, residue, atom, xyz)
    mask: [B, L] or None, indicating valid residues
    """
    aligned_mse_list = []

    B, L, A, _ = pred_atoms.shape
    for i in range(B):
        # [L, 3, 3] -> [L*3, 3]
        P = pred_atoms[i].detach().cpu().numpy().reshape(L * A, 3)
        Q = gt_atoms[i].detach().cpu().numpy().reshape(L * A, 3)

        if mask is not None:
            m = mask[i].bool().detach().cpu().numpy()  # [L]
            m = np.repeat(m, A)  # [L*3]
            P = P[m]
            Q = Q[m]

        P_aligned = kabsch_align(P, Q)
        mse = ((P_aligned - Q) ** 2).mean()
        aligned_mse_list.append(mse)

    return np.mean(aligned_mse_list)
