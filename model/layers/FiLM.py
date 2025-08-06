import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim, scale_activation='none'):
        super().__init__()
        self.scale_activation = scale_activation
        self.film_gen = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x, cond):
        scale_shift = self.film_gen(cond)
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        if self.scale_activation == 'tanh':
            scale = torch.tanh(scale)
        elif self.scale_activation == 'sigmoid':
            scale = torch.sigmoid(scale)
        return x * (1 + scale) + shift
