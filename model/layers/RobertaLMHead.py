import torch
import math
import torch.nn as nn
import torch.nn.functional as F


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)


except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm1 = ESM1bLayerNorm(embed_dim)
        self.dense2 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm2 = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, features):
        x = self.gelu(self.dense1(features))
        x = self.layer_norm1(x)
        x = self.gelu(self.dense2(x))
        x = self.layer_norm2(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class RobertaLMHeadWithGate(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm1 = ESM1bLayerNorm(embed_dim)
        self.dense2 = nn.Linear(embed_dim, embed_dim)
        self.layer_norm2 = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, features):
        x = self.gelu(self.dense1(features))
        x = self.layer_norm1(x)
        x = self.gelu(self.dense2(x))
        x = self.layer_norm2(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x
