import torch
import torch.nn as nn
import torch.nn.functional as F
import model.utils.so3.so3_utils as so3_utils
from .layers.edge import EdgeEmbedder
from .layers.node import NodeEmbedder
from .layers.ga import GAEncoder
from .layers.RobertaLMHead import RobertaLMHead
from .utils.chemical import AA, BBHeavyAtom, max_num_heavyatoms, backbone_atom_coordinates, INIT_CRDS
from .utils.geometry import construct_3d_basis
from .utils.so3.dist import uniform_so3
from .utils.all_atom import to_atom37, compute_backbone, atom37_from_trans_rot
from .utils.interpolant import trans_vector_field, plm_embed_vector_field
from .layers.ga import MatchModel
from scipy.optimize import linear_sum_assignment
from .utils.utils import sample_from, batch_align_structures, to_numpy, calc_bb_fape_loss, calc_BB_bond_geom_in_ideal, \
    NM_TO_ANG_SCALE


class FlowModel(nn.Module):
    def __init__(self, cfg, igso3=None):
        super().__init__()
        self._model_cfg = cfg.model.encoder
        self._interpolant_cfg = cfg.model.interpolant
        self.cdr_weight = cfg.train.cdr_weight
        if igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir='.cache')
        self.igso3 = igso3

        self.node_embedder = NodeEmbedder(cfg.model, max_num_atoms=max_num_heavyatoms)  # 构建蛋白质残基的节点特征
        self.edge_embedder = EdgeEmbedder(cfg.model.encoder.node_embed_size, cfg.model.encoder.edge_embed_size, cfg.model.encoder.index_embed_size,
                                          cfg.model.sample_cdr, embed_diffuse_mask=cfg.model.sample_cdr, max_num_atoms=max_num_heavyatoms)  # 构建蛋白质中残基对之间的边特征
        self.ga_encoder = GAEncoder(cfg.model)  # 用于蛋白质结构预测
        # self.plm_denoiser = PLMDenoiser(cfg.model)  # 用于plm嵌入去噪
        self.decoder = RobertaLMHead(
            embed_dim=cfg.model.encoder.seq_emb_size + cfg.model.encoder.node_embed_size,
            output_dim=20,
            weight=nn.Parameter(torch.zeros((20, cfg.model.encoder.seq_emb_size + cfg.model.encoder.node_embed_size)), requires_grad=True)
        )
        self.match_loss = MatchModel(cfg.model)

        self.register_buffer('position_mean', torch.FloatTensor(cfg.model.scale.trans.position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(cfg.model.scale.trans.position_scale).view(1, 1, -1))
        self.register_buffer('bb_atom_scale', torch.FloatTensor(cfg.model.scale.bb_atom).view(1, 1, -1))

    def _get_rigid(self, batch):
        """
        提取相关信息，并获取刚体变化以及掩码
        """
        pos_heavyatom = batch['pos_heavyatom']
        # 计算蛋白质刚体旋转矩阵和平移向量
        rotmats_1 = construct_3d_basis(pos_heavyatom[:, :, BBHeavyAtom.CA],
                                       pos_heavyatom[:, :, BBHeavyAtom.C],
                                       pos_heavyatom[:, :, BBHeavyAtom.N])
        trans_1 = pos_heavyatom[:, :, BBHeavyAtom.CA]
        angles_1 = batch['torsion_angle']
        mask_heavyatom = batch['mask_heavyatom']
        res_nb = batch['res_nb']
        chain_nb = batch['chain_id']
        aa = batch['aa']
        # 将需要生成的部分屏蔽，避免信息泄露
        context_mask = torch.logical_and(mask_heavyatom[:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        return pos_heavyatom, mask_heavyatom, rotmats_1, trans_1, angles_1, aa, res_nb, chain_nb, context_mask

    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5], device=rotmats_1.device),
            num_batch * num_res
        ).to(rotmats_1.device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        so3_schedule = self._interpolant_cfg.rots.train_schedule
        if so3_schedule == 'exp':
            so3_t = 1 - torch.exp(-t * self._rots_cfg.exp_rate)
        elif so3_schedule == 'linear':
            so3_t = t
        else:
            raise ValueError(f'Invalid schedule: {so3_schedule}')
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=rotmats_1.device)
        rotmats_t = (
                rotmats_t * res_mask[..., None, None]
                + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (1 - diffuse_mask[..., None, None])

    def zero_center_part(self, pos, gen_mask, res_mask=None):
        """
        带有掩码的坐标归一化
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[..., None], dim=1) / (torch.sum(gen_mask, dim=-1, keepdim=True) + 1e-8)  # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1)  # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        if res_mask is not None:
            pos = pos * res_mask[..., None]
        return pos, center

    def _centered_gaussian(self, num_batch, num_res, device):
        noise = torch.randn(num_batch, num_res, 3, device=device)
        return noise - torch.mean(noise, dim=-2, keepdims=True)

    def forward(self, batch, only_cdr=False, calc_bb_bond=True, use_checkpoint=False, eval=False):
        num_batch, num_res = batch['aa'].shape
        # generate_mask指示哪一部分不参与生成，对应的部分则换成xx_1
        gen_mask = batch['generate_mask'].long()
        res_mask = batch['res_mask']
        # 合并重链和轻链的相关信息，并获取刚体变化以及掩码
        # trans_1在构建数据集时已经居中
        pos_heavyatom, mask_heavyatom, rotmats_1, trans_1_c, angles_1, seqs_aa, res_nb, chain_nb, context_mask = self._get_rigid(batch)
        plm_1 = batch['antibody_emb']
        cdr_mask = batch['cdr_flag'].bool()
        if eval:
            cdr_weight = torch.ones_like(gen_mask, device=gen_mask.device)  # 评估时不需要放大cdr区域的损失
            bb_atom_scale = 1.
            position_scale = torch.ones((1, 1, 1), device=gen_mask.device)
        else:
            cdr_weight = torch.where(cdr_mask,
                                     torch.full_like(gen_mask, self.cdr_weight, device=gen_mask.device),
                                     torch.ones_like(gen_mask, device=gen_mask.device))
            bb_atom_scale = self.bb_atom_scale
            position_scale = self.position_scale

        """
        流采样不涉及任何可学习的参数。
        注意*_1表示目标数据分布，*_0表示初始分布。
        """
        t = batch['t']
        with torch.no_grad():
            trans_0_c_nm = self._centered_gaussian(num_batch, num_res, gen_mask.device)  # 生成随机平移噪声
            # 保证初始噪声和真实平移向量的尺度一致
            trans_0_c = trans_0_c_nm * NM_TO_ANG_SCALE
            trans_0_c = self._batch_ot(trans_0_c, trans_1_c, batch['res_mask'])
            trans_t_c = (1 - t[..., None]) * trans_0_c + t[..., None] * trans_1_c  # 在trans_0和trans_1之间进行插值
            trans_t_c = torch.where(batch['generate_mask'][..., None], trans_t_c, trans_1_c)  # 还原不需要生成的部分
            # 生成随机旋转矩阵噪声
            rotmats_t = self._corrupt_rotmats(rotmats_1, t, batch['res_mask'].float(), gen_mask)
            rotmats_t = torch.where(batch['generate_mask'][..., None, None], rotmats_t, rotmats_1)
            # 生成随机序列logits噪声
            plm_0 = torch.randn_like(plm_1)
            # 在seqs_0和seqs_1之间插值
            plm_t =  + (1 - t[..., None]) * plm_0 + t[..., None] * plm_1
            plm_t = torch.where(batch['generate_mask'][..., None], plm_t, plm_1)

        """获取初始的蛋白质的节点和边特征"""
        if 'pred_bb_atom' not in batch:
            sc_bb_atom = torch.zeros_like(pos_heavyatom, device=pos_heavyatom.device)[:, :, :3]
        else:
            sc_bb_atom = (batch['pred_bb_atom'][:, :, :3] * gen_mask[..., None, None] + pos_heavyatom[:, :, :3] * (1 - gen_mask[..., None, None]))
        node_embedder_input = [t, res_nb, chain_nb, plm_t, sc_bb_atom, res_mask, mask_heavyatom, gen_mask]
        if only_cdr: node_embedder_input.extend([seqs_aa, pos_heavyatom, context_mask])
        node_embed = self.node_embedder(*node_embedder_input)
        if 'pred_trans_1' not in batch:
            trans_sc = torch.zeros_like(trans_t_c, device=trans_t_c.device)
        else:
            trans_sc = batch['pred_trans_1'] * gen_mask[..., None] + trans_1_c * (1 - gen_mask[..., None])
        edge_embedder_input = [node_embed, res_nb, trans_t_c, trans_sc, sc_bb_atom, mask_heavyatom, gen_mask]
        if only_cdr: edge_embedder_input.extend([seqs_aa, pos_heavyatom, context_mask])
        edge_embed = self.edge_embedder(*edge_embedder_input)

        # 调用GAEncoder去噪
        pred_rotmats_1, pred_trans_1, pred_plm_embed, node_embed = self.ga_encoder(rotmats_t, trans_t_c, node_embed, edge_embed,
                                                                                   res_mask.float(), gen_mask,
                                                                                   use_checkpoint=use_checkpoint)

        if eval:
            norm_scale = torch.ones((num_batch, 1, 1), device=t.device)  # 评估时不需要根据时间步信息放大损失
        else:
            # 根据时间步信息创建的损失缩放。时间越早（小），损失越高
            norm_scale = 1 - torch.min(t[..., None], torch.tensor(self._interpolant_cfg.t_normalization_clip))  # (B, 1, 1)

        gt_bb_atoms = to_atom37(trans_1_c, rotmats_1)[:, :, :3]
        pred_bb_atoms = to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        if calc_bb_bond:
            # 主链键长和键角损失
            bb_len_loss, bb_ang_loss = calc_BB_bond_geom_in_ideal(pred_bb_atoms, res_nb, gen_mask)
        else:
            bb_len_loss = None
            bb_ang_loss = None
        # gt_bb_atoms = compute_backbone(create_rigid(rotmats_1, trans_1_c),
        #                                torch.concat([torch.sin(angles_1[:, :, 0])[..., None],
        #                                              torch.cos(angles_1[:, :, 0])[..., None]], dim=-1))[0][:, :, :5]
        # pred_bb_atoms = compute_backbone(curr_rigids, pred_psi)[0][:, :, :5]
        # kb_mse = compute_kabsch_aligned_mse_batched(pred_bb_atoms, gt_bb_atoms, gen_mask)
        # 主链原子坐标mse损失
        bb_atom_loss = torch.sum((gt_bb_atoms * bb_atom_scale / norm_scale[..., None] - pred_bb_atoms * bb_atom_scale / norm_scale[..., None])
                                 ** 2 * gen_mask[..., None, None], dim=(-1, -2, -3)) / (torch.sum(gen_mask, dim=-1) + 1e-8)
        bb_atom_loss = torch.mean(bb_atom_loss)

        # 平移向量向量场损失
        trans_error = (trans_1_c - pred_trans_1) / norm_scale * position_scale
        trans_loss = torch.sum(trans_error ** 2 * gen_mask[..., None], dim=(-1, -2)) / (torch.sum(gen_mask, dim=-1) + 1e-8)  # (B,)
        trans_loss = torch.mean(trans_loss)

        # 旋转矩阵向量场损失
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_vf_loss = torch.sum(((gt_rot_vf - pred_rot_vf) / norm_scale) ** 2 * gen_mask[..., None], dim=(-1, -2)) / (torch.sum(gen_mask, dim=-1) + 1e-8)  # (B,)
        rot_vf_loss = torch.mean(rot_vf_loss)

        # Pairwise 距离约束，用于防止断链和链冲突
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 3, 3])
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 3, 3])
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)
        flat_loss_mask = torch.tile(gen_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 3])
        flat_res_mask = torch.tile(gen_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 3])
        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]
        proximity_mask = gt_pair_dists <= 6.  # 超过一定距离的残基对不计算损失
        pair_dist_mask = pair_dist_mask * proximity_mask
        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2))
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2))
        dist_mat_loss = torch.mean(dist_mat_loss)

        # plm嵌入损失
        plm_loss = F.mse_loss(pred_plm_embed, plm_1, reduction='none')
        plm_loss = torch.sum(plm_loss * gen_mask[..., None], dim=(-1, -2))  # 忽略掉无需生成的部分
        plm_loss /= (torch.sum(gen_mask, dim=-1) + 1e-8)
        plm_loss = torch.mean(plm_loss)
        # 余弦距离损失，用于防止模型无法拟合正确的方向
        plm_cos_loss = 1 - ((F.cosine_similarity(pred_plm_embed, plm_1, dim=-1) * gen_mask).sum() / gen_mask.sum())
        # 解码器交叉熵损失
        pred_seqs_logits = self.decoder(torch.concat([pred_plm_embed, node_embed], dim=-1))
        seqs_loss = F.cross_entropy(pred_seqs_logits.view(-1, pred_seqs_logits.shape[-1]), torch.clamp(batch['aa'], 0, 19).view(-1), reduction='none')
        seqs_loss = (seqs_loss * gen_mask.float().reshape(-1)).sum() / (gen_mask.sum() + 1e-8)  # 忽略掉无需生成的部分
        # pred_seqs = sample_from(F.softmax(pred_seqs_logits, dim=-1))
        # # 计算氨基酸恢复正确率，只计算有效残基
        # with torch.no_grad():
        #     correct = (pred_seqs == seqs_aa) * gen_mask
        #     pred_aa_acc = correct.sum() / gen_mask.sum()

        contrast_loss = self.match_loss(node_embed, pred_plm_embed, gen_mask, match_type='graph')

        return {
            'pred_trans_1': pred_trans_1,
            'pred_bb_atom': pred_bb_atoms,
            'trans_loss': trans_loss,
            'rot_vf_loss': rot_vf_loss,
            'bb_atom_loss': bb_atom_loss,
            'bb_len_loss': bb_len_loss,
            'bb_ang_loss': bb_ang_loss,
            'pair_loss': dist_mat_loss,
            'plm_emb_loss': plm_loss,
            'plm_cos_loss': plm_cos_loss,
            'seqs_loss': seqs_loss,
            # 'pred_aa_acc': pred_aa_acc,
            'contrast_loss': contrast_loss,
        }

    def sample(self, config, batch, num_steps=100, only_cdr=False):

        num_batch, num_res = batch['generate_mask'].shape
        gen_mask = batch['generate_mask']
        res_mask = batch['res_mask']
        res_nb = batch['res_nb']
        chain_nb = batch['chain_id']
        device = batch['generate_mask'].device
        frames_to_atom37 = lambda x, y: atom37_from_trans_rot(x, y, res_mask).detach().cpu()

        sc_bb_atom = torch.zeros((num_batch, num_res, 3, 3), device=device)[:, :, :3]
        trans_sc = torch.zeros((num_batch, num_res, 3), device=device)

        trans_0_c = self._centered_gaussian(num_batch, num_res, device) * NM_TO_ANG_SCALE
        rotmats_0 = uniform_so3(num_batch, num_res, device=device)
        plm_emb_0 = torch.randn((num_batch, num_res, config.model.encoder.seq_emb_size), device=device)
        context_mask = ~gen_mask

        if only_cdr:
            pos_heavyatom = batch['pos_heavyatom']
            rotmats_1 = construct_3d_basis(pos_heavyatom[:, :, BBHeavyAtom.CA],
                                           pos_heavyatom[:, :, BBHeavyAtom.C],
                                           pos_heavyatom[:, :, BBHeavyAtom.N])
            trans_1_c = pos_heavyatom[:, :, BBHeavyAtom.CA]
            plm_emb_1 = batch['antibody_emb']
            mask_atom = batch['mask_heavyatom']
            seqs_aa = batch['aa']
            # 还原不需要生成的部分
            trans_0_c = torch.where(gen_mask[..., None], trans_0_c, trans_1_c)
            rotmats_0 = torch.where(gen_mask[..., None, None], rotmats_0, rotmats_1)
            plm_emb_0 = torch.where(gen_mask[..., None], plm_emb_0, plm_emb_1)
        else:
            mask_atom = torch.ones((num_batch, num_res, 3), device=device)

        ts = torch.linspace(1.e-2, 1.0, num_steps)  # 采样时间列表
        t_1 = ts[0]
        prot_traj = []  # 用于保存轨迹演化历史
        clean_traj = []  # 用于保存预测的目标结构
        rotmats_t_1, trans_t_1_c, plm_emb_t_1 = rotmats_0, trans_0_c, plm_emb_0
        prot_traj.append(
            {'rotmats': rotmats_t_1.cpu(), 'trans': trans_t_1_c.cpu(),
             'seqs_emb': plm_emb_t_1.cpu(), 'bb_pos': frames_to_atom37(trans_t_1_c, rotmats_t_1)}
        )

        """循环去噪"""
        for t_2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=batch['generate_mask'].device) * t_1
            node_embedder_input = [t, res_nb, chain_nb, plm_emb_t_1, sc_bb_atom, batch['res_mask'], mask_atom, gen_mask]
            if only_cdr: node_embedder_input.extend([seqs_aa, pos_heavyatom, context_mask])
            node_embed = self.node_embedder(*node_embedder_input)
            edge_embedder_input = [node_embed, res_nb, trans_t_1_c, trans_sc, sc_bb_atom, mask_atom, gen_mask]
            if only_cdr: edge_embedder_input.extend([seqs_aa, pos_heavyatom, context_mask])
            edge_embed = self.edge_embedder(*edge_embedder_input)
            pred_rotmats_1, pred_trans_1, pred_plm_embed_1, node_embed = self.ga_encoder(rotmats_t_1, trans_t_1_c, node_embed, edge_embed,
                                                                                         res_mask.float(), gen_mask.long())
            if only_cdr:
                pred_trans_1 = torch.where(gen_mask[..., None], pred_trans_1, trans_1_c)
                pred_rotmats_1 = torch.where(gen_mask[..., None, None], pred_rotmats_1, rotmats_1)
                pred_plm_embed_1 = torch.where(gen_mask[..., None], pred_plm_embed_1, plm_emb_1)
            bb_atom = frames_to_atom37(pred_trans_1, pred_rotmats_1)
            pred_seqs_aa = sample_from(F.softmax(self.decoder(torch.concat([pred_plm_embed_1, node_embed], dim=-1)), dim=-1))
            if only_cdr:
                pred_seqs_aa = torch.where(gen_mask, pred_seqs_aa, seqs_aa)
            # 添加至预测轨迹
            clean_traj.append(
                {'rotmats': pred_rotmats_1.cpu(), 'trans': pred_trans_1.cpu(), 'seqs_emb': pred_plm_embed_1.cpu(),
                 'seqs_aa': pred_seqs_aa.cpu(), 'bb_pos': bb_atom}
            )
            # 自条件
            trans_sc = pred_trans_1
            sc_bb_atom = bb_atom.to(device)

            d_t = (t_2 - t_1) * torch.ones((num_batch, 1), device=batch['generate_mask'].device)
            """欧拉过程"""
            trans_vf = trans_vector_field(t_1, pred_trans_1, trans_t_1_c)  # 平移向量向量场
            trans_t_2_c = trans_t_1_c + trans_vf * d_t[..., None]
            rotmats_t_2 = so3_utils.geodesic_t(10. * d_t[..., None], pred_rotmats_1, rotmats_t_1)
            plm_emb_vf = plm_embed_vector_field(t_1, pred_plm_embed_1, plm_emb_t_1)
            plm_emb_t_2 = plm_emb_t_1 + plm_emb_vf * d_t[..., None]
            if only_cdr:
                trans_t_2_c = torch.where(gen_mask[..., None], trans_t_2_c, trans_1_c)
                rotmats_t_2 = torch.where(gen_mask[..., None, None], rotmats_t_2, rotmats_1)
                plm_emb_t_2 = torch.where(gen_mask[..., None], plm_emb_t_2, plm_emb_1)
            rotmats_t_1, trans_t_1_c, plm_emb_t_1 = rotmats_t_2, trans_t_2_c, plm_emb_t_2
            prot_traj.append(
                {'rotmats': rotmats_t_1.cpu(), 'trans': trans_t_1_c.cpu(), 'seqs_emb': plm_emb_t_1.cpu(),
                 'bb_pos': frames_to_atom37(trans_t_1_c, rotmats_t_1)}
            )
            t_1 = t_2

        # 最后的去噪工作
        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch['generate_mask'].device) * t_1
        node_embedder_input = [t, res_nb, chain_nb, plm_emb_t_1, sc_bb_atom, batch['res_mask'], mask_atom, gen_mask]
        if only_cdr: node_embedder_input.extend([seqs_aa, pos_heavyatom, context_mask])
        node_embed = self.node_embedder(*node_embedder_input)
        edge_embedder_input = [node_embed, res_nb, trans_t_1_c, trans_sc, sc_bb_atom, mask_atom, gen_mask]
        if only_cdr: edge_embedder_input.extend([seqs_aa, pos_heavyatom, context_mask])
        edge_embed = self.edge_embedder(*edge_embedder_input)
        pred_rotmats_1, pred_trans_1, pred_plm_embed_1, node_embed = self.ga_encoder(rotmats_t_1, trans_t_1_c, node_embed, edge_embed,
                                                                                     res_mask.float(), gen_mask.long())
        if only_cdr:
            pred_trans_1 = torch.where(gen_mask[..., None], pred_trans_1, trans_1_c)
            pred_rotmats_1 = torch.where(gen_mask[..., None, None], pred_rotmats_1, rotmats_1)
            pred_plm_embed_1 = torch.where(gen_mask[..., None], pred_plm_embed_1, plm_emb_1)
        pred_seqs_aa = sample_from(F.softmax(self.decoder(torch.concat([pred_plm_embed_1, node_embed], dim=-1)), dim=-1))
        if only_cdr:
            pred_seqs_aa = torch.where(gen_mask, pred_seqs_aa, seqs_aa)
        clean_traj.append(
            {'rotmats': pred_rotmats_1.cpu(), 'trans': pred_trans_1.cpu(), 'seqs_emb': pred_plm_embed_1.cpu(),
             'seqs_aa': pred_seqs_aa.cpu(), 'bb_pos': frames_to_atom37(pred_trans_1, pred_rotmats_1)}
        )
        prot_traj.append(
            {'rotmats': pred_rotmats_1.cpu(), 'trans': pred_trans_1.cpu(), 'seqs_emb': pred_plm_embed_1.cpu(),
             'seqs_aa': pred_seqs_aa.cpu(), 'bb_pos': frames_to_atom37(pred_trans_1, pred_rotmats_1)}
        )
        return prot_traj, clean_traj

