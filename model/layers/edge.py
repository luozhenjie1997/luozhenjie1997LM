import torch
import torch.nn as nn
import torch.nn.functional as F
from .AngularEncoding import AngularEncoding
from ..utils.geometry import angstrom_to_nm, pairwise_dihedrals
from ..utils.chemical import BBHeavyAtom, AA3Letter
from ..utils.utils import calc_distogram, get_index_embedding


class EdgeEmbedder(nn.Module):
    """
    为蛋白质中残基对之间构造边的嵌入
    """

    def __init__(self, node_embed_dim, feat_dim, index_dim, only_cdr=False, embed_diffuse_mask=False, max_num_atoms=14, max_aa_types=22,
                 num_bins=22, max_relpos=32):
        super().__init__()
        self.feat_dim = feat_dim
        self.index_dim = index_dim
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.max_relpos = max_relpos
        self.num_bins = num_bins
        self.embed_diffuse_mask = embed_diffuse_mask
        self.only_cdr = only_cdr

        self.linear_s_p = nn.Linear(node_embed_dim, self.feat_dim)
        self.linear_relpos = nn.Linear(self.index_dim, self.feat_dim)
        # self.relpos_embed = nn.Embedding(2 * max_relpos + 1, feat_dim)  # 氨基酸相对位置索引嵌入

        self.dihedral_embed = AngularEncoding()

        infeat_dim = feat_dim * 3 + num_bins * 2 + self.dihedral_embed.get_out_dim(2)

        if only_cdr:
            self.aa_pair_embed = nn.Embedding(self.max_aa_types * self.max_aa_types, feat_dim)

            self.aapair_to_distcoef = nn.Embedding(self.max_aa_types ** 2, max_num_atoms ** 2)
            nn.init.zeros_(self.aapair_to_distcoef.weight)

            self.distance_embed = nn.Sequential(
                nn.Linear(max_num_atoms ** 2, feat_dim), nn.ReLU(),
                nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            )

            infeat_dim += feat_dim * 2 + self.dihedral_embed.get_out_dim(2)

        if embed_diffuse_mask:
            infeat_dim += 2

        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, feat_dim * 4), nn.ReLU(),
            nn.Linear(feat_dim * 4, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.LayerNorm(feat_dim)
        )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: [b, n_res]
        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self.index_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, init_node_embed, res_nb, trans_t, sc_trans_t, sc_pos_atoms, mask_atoms, diffuse_mask, aa=None,
                pos_atoms=None, context_mask=None):
        """
        Args:
            aa: 氨基酸索引(N, L).
            res_nb: 残基编号(N, L).
            chain_nb: 链编号(N, L).
            pos_atoms:  每个残基的前A个原子坐标(N, L, A, 3)
            mask_atoms: 原子坐标掩码(N, L, A)
            structure_mask: 残基结构掩码(N, L)
            sequence_mask:  (N, L), 用于屏蔽未知氨基酸

        Returns:
            (N, L, L, feat_dim)
        """
        # Input: [b, n_res, c_s]
        b, num_res, _ = init_node_embed.shape

        # [b, n_res, c_p]
        p_i = self.linear_s_p(init_node_embed)
        cross_node_feats = self._cross_concat(p_i, b, num_res)

        relpos_feats = self.embed_relpos(res_nb)
        # 判断两残基是否属于同一链
        # same_chain = (chain_nb[:, :, None] == chain_nb[:, None, :])
        # # 计算相对位置relpos，即两残基编号的差值，并将其限制在[-max_relpos, max_relpos]的范围内
        # relpos = torch.clamp(
        #     res_nb[:, :, None] - res_nb[:, None, :],
        #     min=-self.max_relpos, max=self.max_relpos,
        # )  # (N, L, L)
        # # 得到两两残基相对位置的嵌入，将不是同一条链的残基对的特征置0
        # relpos_feats = self.relpos_embed(relpos + self.max_relpos) * same_chain[:, :, :, None]

        dist_feats = calc_distogram(trans_t, min_bin=1e-3, max_bin=20.0, num_bins=self.num_bins)
        sc_dist_feats = calc_distogram(sc_trans_t, min_bin=1e-3, max_bin=20.0, num_bins=self.num_bins)

        if sc_pos_atoms.sum() == 0:
            sc_feat_dihed = torch.zeros((b, num_res, num_res, self.dihedral_embed.get_out_dim(2)), device=init_node_embed.device)
        else:
            sc_dihed = pairwise_dihedrals(sc_pos_atoms)  # (N, L, L, 2)
            sc_feat_dihed = self.dihedral_embed(sc_dihed)  # 对二面角进行编码

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_dist_feats, sc_feat_dihed]
        if self.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], b, num_res)
            all_edge_feats.append(diff_feat)

        if self.only_cdr:
            # 移除多余的原子
            pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
            mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
            # 使用CA的掩码来代表该残基是否有效
            mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)
            # 创建对无效的残基对进行屏蔽的掩码
            mask_pair = mask_residue[:, :, None] * mask_residue[:, None, :]
            pair_structure_mask = context_mask[:, :, None] * context_mask[:, None, :]

            aa = torch.where(context_mask, aa, torch.full_like(aa, fill_value=AA3Letter.UNK))
            aa_pair = aa[:, :, None] * self.max_aa_types + aa[:, None, :]  # (N, L, L)
            feat_aapair = self.aa_pair_embed(aa_pair) * mask_pair[..., None]

            """氨基酸距离"""
            # 计算任意两个残基之间所有原子对的欧几里得距离
            a2a_coords = pos_atoms[:, :, None, :, None] - pos_atoms[:, None, :, None, :]
            a2a_dist = torch.linalg.norm(a2a_coords, dim=-1)
            d_flat = angstrom_to_nm(a2a_dist)  # 转换单位
            d = d_flat.reshape(b, num_res, num_res, -1)  # (N, L, L, A*A)
            # 经过softplus激活保证其为正数
            c = F.softplus(self.aapair_to_distcoef(aa_pair))
            # 利用高斯核函数计算d_gauss
            d_gauss = torch.exp(-1. * c * d ** 2)
            # 排除掉两个残基之间原子不存在的结果
            mask2d_aa_pair = (mask_atoms[:, :, None, :, None] * mask_atoms[:, None, :, None, :]).reshape(b, num_res, num_res, -1)
            feat_dist = self.distance_embed(d_gauss * mask2d_aa_pair)
            feat_dist = feat_dist * pair_structure_mask[:, :, :, None]  # 屏蔽无效的残基对

            """二面角"""
            # 计算每对残基之间的两个二面角（ψ角与φ角）
            dihed = pairwise_dihedrals(pos_atoms)  # (N, L, L, 2)
            feat_dihed = self.dihedral_embed(dihed)  # 对二面角进行编码
            feat_dihed = feat_dihed * pair_structure_mask[:, :, :, None]  # 屏蔽掉成对氨基酸中其中一个被屏蔽时的二面角特征

            all_edge_feats.extend([feat_aapair, feat_dist, feat_dihed])

        edge_feats = self.mlp(torch.concat(all_edge_feats, dim=-1))

        return edge_feats
