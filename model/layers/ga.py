import torch
import torch.nn as nn
import torch.nn.functional as F
import model.layers.ipa_pytorch as ipa_pytorch
import torch.utils.checkpoint as checkpoint
import model.utils.utils as du
from .FiLM import FiLM
from .ResidualBlock import ResidualBlock
from .TorsionAngles import TorsionAngles
from .Attention import SelfAttention, CrossAttentionBlock, get_attn_emb_padded
from ..utils.utils import create_custom_forward, get_time_embedding, get_index_embedding


class GAEncoder(nn.Module):
    """用于蛋白质结构的编码和预测"""
    def __init__(self, conf):
        super().__init__()

        self._ipa_conf = conf.encoder.ipa
        """注意力机制对输入的绝对值非常敏感，如果数值范围波动过大会造成训练不稳定，因此需要对平移向量进行缩放，处理完毕后再解除缩放"""
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)

        # self.torsion_pred = TorsionAngles(self._ipa_conf.c_s, 1)  # 预测psi角

        # 重建plm嵌入
        self.seq_net = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s * 2), nn.GELU(),
            ResidualBlock(self._ipa_conf.c_s * 2),
            ResidualBlock(self._ipa_conf.c_s * 2),
            nn.Linear(self._ipa_conf.c_s * 2, conf.encoder.seq_emb_size)
        )
        # 用于对齐结构预测出的plm与真实plm分布
        self.plm_recon_projector = nn.Sequential(
            nn.Linear(conf.encoder.seq_emb_size, conf.encoder.seq_emb_size), nn.GELU(),
            nn.Linear(conf.encoder.seq_emb_size, conf.encoder.seq_emb_size)
        )

        """IPA蛋白质建模注意力机制"""
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)

            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                activation='relu',
                batch_first=True,
                dropout=self._ipa_conf.dropout,
                norm_first=False,
            )
            self.trunk[f'seq_tfmr_{b}'] = torch.nn.TransformerEncoder(tfmr_layer, self._ipa_conf.seq_tfmr_num_layers,
                                                                      enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)
            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._ipa_conf.c_z
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self._ipa_conf.c_z,
                )

    def forward(self, rotmats_t, trans_t, node_embed, edge_embed, res_mask, gen_mask, use_checkpoint=False):
        num_batch, num_res, _ = node_embed.shape

        node_mask = res_mask
        edge_mask = node_mask[:, None] * node_mask[:, :, None]

        node_embed = node_embed * node_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]

        # 迭代 IPA 计算
        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        for b in range(self._ipa_conf.num_blocks):
            if use_checkpoint:
                ipa_embed = checkpoint.checkpoint(create_custom_forward(self.trunk[f'ipa_{b}']), node_embed, edge_embed,
                                                  curr_rigids, node_mask, use_reentrant=False)
            else:
                ipa_embed = self.trunk[f'ipa_{b}'](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)

            if use_checkpoint:
                def custom_forward(node_embed, node_mask):
                    return self.trunk[f'seq_tfmr_{b}'](node_embed, src_key_padding_mask=(1 - node_mask).bool())
                seq_tfmr_out = checkpoint.checkpoint(custom_forward, node_embed, node_mask, use_reentrant=False)
            else:
                seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](node_embed, src_key_padding_mask=(1 - node_mask).bool())
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = node_embed * node_mask[..., None]

            if use_checkpoint:
                node_embed = checkpoint.checkpoint(create_custom_forward(self.trunk[f'node_transition_{b}']), node_embed,
                                                   use_reentrant=False)
            else:
                node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, (node_mask * gen_mask)[..., None])

            if b < self._ipa_conf.num_blocks - 1:
                if use_checkpoint:
                    edge_embed = checkpoint.checkpoint(create_custom_forward(self.trunk[f'edge_transition_{b}']),
                                                       node_embed, edge_embed, use_reentrant=False)
                else:
                    edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        # 预测的蛋白质主链结构
        pred_trans1 = curr_rigids.get_trans()
        pred_rotmats1 = curr_rigids.get_rots().get_rot_mats()

        # 回归plm嵌入
        pred_plm_proj = self.seq_net(node_embed) * node_mask[:, :, None]
        pred_plm_proj = self.plm_recon_projector(pred_plm_proj)

        # _, psi_pred = self.torsion_pred(node_embed)

        return pred_rotmats1, pred_trans1, pred_plm_proj, node_embed

class MatchModel(nn.Module):
    """用于多模态对齐"""
    def __init__(self, conf, projection_dim=128, n_head=4, temperature=0.1):
        super().__init__()

        self._ipa_conf = conf.encoder.ipa
        self.T = temperature
        self.projection_dim = projection_dim  # 投影维度

        self.struct_attention = SelfAttention(n_emb=self._ipa_conf.c_s, n_head=n_head)
        self.plm_attention = SelfAttention(n_emb=conf.encoder.seq_emb_size, n_head=n_head)
        # 结构表示投影头
        self.struct_projector = nn.Sequential(
            nn.Linear(self._ipa_conf.c_s, self._ipa_conf.c_s), nn.ReLU(),
            nn.Linear(self._ipa_conf.c_s, self.projection_dim)
        )
        # PLM表示投影头
        self.plm_projector = nn.Sequential(
            nn.Linear(conf.encoder.seq_emb_size, conf.encoder.seq_emb_size), nn.ReLU(),
            nn.Linear(conf.encoder.seq_emb_size, self.projection_dim)
        )

    def forward(self, struct_embed, plm_embed, mask, match_type="graph"):
        assert match_type in {"node", "graph"}, print("match_type error")
        device = struct_embed.device
        # 注意力机制池化
        struct_emb, struct_attn = self.struct_attention(struct_embed, mask.bool())
        token_emb, token_attn = self.plm_attention(plm_embed, mask.bool())
        sturct_match_emb = get_attn_emb_padded(struct_emb, struct_attn, mask.bool())
        plm_match_emb = get_attn_emb_padded(token_emb, token_attn, mask.bool())
        # 投影到指定维度
        feature_left = self.struct_projector(sturct_match_emb)
        feature_right = self.plm_projector(plm_match_emb)

        if match_type == "node":
            similarity = F.cosine_similarity(feature_left, feature_right, dim=1).to(device)
            similarity = torch.exp(similarity / self.T)
            loss = torch.mean(-torch.log(similarity))
        else:
            n = len(feature_left)
            similarity = F.cosine_similarity(feature_left.unsqueeze(1), feature_right.unsqueeze(0), dim=2).to(device)
            similarity = torch.exp(similarity / self.T)

            mask_pos = torch.eye(n, n, device=device, dtype=bool)
            sim_pos = torch.masked_select(similarity, mask_pos)

            sim_total_row = torch.sum(similarity, dim=0)
            loss_row = torch.div(sim_pos, sim_total_row)
            loss_row = -torch.log(loss_row)

            sim_total_col = torch.sum(similarity, dim=1)
            loss_col = torch.div(sim_pos, sim_total_col)
            loss_col = -torch.log(loss_col)

            loss = loss_row + loss_col
            loss = torch.sum(loss) / (2 * n)

        return loss
