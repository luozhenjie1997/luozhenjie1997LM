import torch
from torch import nn
from ..utils.geometry import construct_3d_basis, global_to_local, get_backbone_dihedral_angles
from .AngularEncoding import AngularEncoding
from ..utils.chemical import BBHeavyAtom, AA3Letter
from ..utils.utils import get_index_embedding, get_time_embedding


class NodeEmbedder(nn.Module):
    """
    根据蛋白质序列和结构信息生成每个残基（节点node）的嵌入表示
    """
    def __init__(self, conf, max_num_atoms=14, max_aa_types=22, pad_aa_value=21, max_chain_types=5, pad_chain_value=-1):
        super().__init__()
        self.max_num_atoms = max_num_atoms
        self.max_aa_types = max_aa_types
        self.feat_dim = conf.encoder.node_embed_size
        self.index_dim = conf.encoder.index_embed_size
        self.only_cdr = conf.sample_cdr

        self.chain_emb = nn.Embedding(max_chain_types, self.feat_dim, padding_idx=pad_chain_value)  # 对链编号进行编码
        self.dihed_embed = AngularEncoding()  # 对二面角（例如φ、ψ、ω）进行编码

        infeat_dim = self.feat_dim + self.index_dim * 4 + self.dihed_embed.get_out_dim(3) + conf.encoder.seq_emb_size + 1

        if self.only_cdr:
            self.aatype_embed = nn.Embedding(self.max_aa_types, self.feat_dim, padding_idx=pad_aa_value)
            infeat_dim += (self.max_aa_types * max_num_atoms * 3) + self.dihed_embed.get_out_dim(3) + self.feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(infeat_dim, self.feat_dim * 4), nn.ReLU(),
            nn.Linear(self.feat_dim * 4, self.feat_dim * 2), nn.ReLU(),
            nn.Linear(self.feat_dim * 2, self.feat_dim), nn.LayerNorm(self.feat_dim)
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.index_dim,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, t, res_nb, chain_nb, plm_t, sc_pos_atom, res_mask, mask_atoms, diffuse_mask, aa=None, pos_atoms=None,
                context_mask=None):
        """
        Args:
            aa:             每个残基的氨基酸索引(N, L)
            res_nb:         残基编号(N, L)
            chain_nb:       链编号(N, L)
            pos_atoms:      每个残基中 A 个原子的三维坐标(N, L, A, 3)
            mask_atoms:     用于表示各个原子是否存在的掩码(N, L, A)
            structure_mask: (N, L), 用于屏蔽未知结构
            sequence_mask:  (N, L), 用于屏蔽未知氨基酸
        """
        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device
        # 使用CA的掩码来代表该残基是否有效
        mask_residue = mask_atoms[:, :, BBHeavyAtom.CA]  # (N, L)

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(res_nb, self.index_dim, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        chain_emb = self.chain_emb(chain_nb)

        if sc_pos_atom.sum() == 0:
            sc_dihed_feat = torch.zeros((b, num_res, self.dihed_embed.get_out_dim(3)), device=device)
        else:
            sc_bb_dihedral, sc_mask_bb_dihed = get_backbone_dihedral_angles(sc_pos_atom, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
            sc_dihed_feat = self.dihed_embed(sc_bb_dihedral[:, :, :, None]) * sc_mask_bb_dihed[:, :, :, None]
            sc_dihed_feat = sc_dihed_feat.reshape(b, num_res, -1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            chain_emb,
            sc_dihed_feat,
            plm_t
        ]

        if self.only_cdr:
            aa = torch.where(context_mask, aa, torch.full_like(aa, fill_value=AA3Letter.UNK))
            aa_feat = self.aatype_embed(aa)

            pos_atoms = pos_atoms[:, :, :self.max_num_atoms]
            mask_atoms = mask_atoms[:, :, :self.max_num_atoms]
            R = construct_3d_basis(
                pos_atoms[:, :, BBHeavyAtom.CA],
                pos_atoms[:, :, BBHeavyAtom.C],
                pos_atoms[:, :, BBHeavyAtom.N]
            )
            trans = pos_atoms[:, :, BBHeavyAtom.CA]
            # 转换到局部坐标
            crd = global_to_local(R, trans, pos_atoms)  # (N, L, A, 3)
            crd_mask = mask_atoms[:, :, :, None].expand_as(crd)
            # 将crd_mask对应位置的坐标置0
            crd = torch.where(crd_mask, crd, torch.zeros_like(crd))
            """坐标特征"""
            aa_expand = aa[:, :, None, None, None].expand(b, num_res, self.max_aa_types, self.max_num_atoms, 3)
            rng_expand = torch.arange(0, self.max_aa_types)[None, None, :, None, None].expand(b, num_res, self.max_aa_types,
                                                                                              self.max_num_atoms, 3).to(aa_expand)
            # 用于挑选出实际氨基酸类型对应的坐标信息
            place_mask = (aa_expand == rng_expand)
            crd_expand = crd[:, :, None, :, :].expand(b, num_res, self.max_aa_types, self.max_num_atoms, 3)
            # 根据place_mask只保留与当前氨基酸匹配的位置，其他位置置零，最后将该特征的形状重塑为 (N, L, max_aa_types * max_num_atoms * 3)，形成 crd_feat
            crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
            crd_feat = crd_expand.reshape(b, num_res, self.max_aa_types * self.max_num_atoms * 3)
            # 屏蔽未知结构的部分（即屏蔽掉需要生成的部分），避免数据泄露
            crd_feat = crd_feat * context_mask[:, :, None]

            """主链二面角特征"""
            # 得到主链的二面角及标记其有效性的掩码
            bb_dihedral, mask_bb_dihed = get_backbone_dihedral_angles(pos_atoms, chain_nb=chain_nb, res_nb=res_nb, mask=mask_residue)
            # 进行角度编码，并屏蔽无效角度
            dihed_feat = self.dihed_embed(bb_dihedral[:, :, :, None]) * mask_bb_dihed[:, :, :, None]  # (N, L, 3, dihed/3)
            dihed_feat = dihed_feat.reshape(b, num_res, -1)
            dihed_mask = torch.logical_and(
                context_mask,
                torch.logical_and(
                    torch.roll(context_mask, shifts=+1, dims=1),
                    torch.roll(context_mask, shifts=-1, dims=1)
                ),
            )
            dihed_feat = dihed_feat * dihed_mask[:, :, None]  # 避免通过锚定残基的二面角发生轻微数据泄露

            input_feats.extend([aa_feat, crd_feat, dihed_feat])

        input_feats.extend([diffuse_mask[..., None], self.embed_t(t, res_mask), self.embed_t(t, res_mask), self.embed_t(t, res_mask)])
        input_feats = torch.cat(input_feats, dim=-1)

        return self.mlp(input_feats)
