import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, d_kv=None, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, kdim=d_kv or d_model, vdim=d_kv or d_model,
            num_heads=num_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, query, key_value, key_padding_mask=None):
        out, _ = self.cross_attn(query=query, key=key_value, value=key_value, key_padding_mask=key_padding_mask)
        return self.ln(query + out)


class SelfAttention(nn.Module):
    def __init__(self, n_emb, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()

        self.key = nn.Linear(n_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)

        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        self.proj = nn.Linear(n_emb, n_emb)

        self.n_head = n_head
        self.head_dim = n_emb // n_head

    def forward(self, x, padding_mask=None):
        """
        x: [B, T, C]
        padding_mask: [B, T] — True for valid, False for padding
        """
        B, T, C = x.size()
        H = self.n_head
        D = self.head_dim

        # Linear projections
        k = self.key(x).view(B, T, H, D).transpose(1, 2)   # [B, H, T, D]
        q = self.query(x).view(B, T, H, D).transpose(1, 2)
        v = self.value(x).view(B, T, H, D).transpose(1, 2)

        # Attention logits
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

        if padding_mask is not None:
            # padding_mask: [B, T] → [B, 1, 1, T] for broadcasting
            mask = ~padding_mask[:, None, None, :]  # convert to False for valid, True for pad
            attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)  # [B, H, T, T]
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)  # [B, H, T, D]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.proj(y)
        y = self.resid_drop(y)

        return y, attn.mean(dim=1)  # 平均所有头后返回注意力

def get_attn_emb_padded(seq_emb, seq_attn, padding_mask):
    """
    seq_emb: [B, T, C]
    seq_attn: [B, T, T]
    padding_mask: [B, T] — True 表示有效 token
    """
    B, T, C = seq_emb.shape
    seq_attn_emb_list = []

    for i in range(B):
        valid_mask = padding_mask[i]  # [T]
        emb = seq_emb[i][valid_mask]  # [T_i, C]
        attn = seq_attn[i][valid_mask][:, valid_mask]  # [T_i, T_i]
        attn = attn.mean(dim=0)  # [T_i]

        attn_emb = attn @ emb  # [C]
        seq_attn_emb_list.append(attn_emb.unsqueeze(0))

    return torch.cat(seq_attn_emb_list, dim=0)  # [B, C]
