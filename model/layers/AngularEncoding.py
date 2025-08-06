import torch
import torch.nn as nn


class AngularEncoding(nn.Module):
    """
    通过多频带正弦/余弦编码对输入数据进行特征增强。类似Transformer的位置编码，这种编码常用于处理角度（如方向、旋转）或周期性数据
    通过引入不同频率的正弦和余弦分量，帮助模型捕捉多维空间中的几何关系
    """
    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        # 创建一组频率波段。例如，若 num_funcs=3，则 freq_bands = [1, 2, 3, 1, 0.5, 0.333]
        self.register_buffer('freq_bands', torch.FloatTensor(
            [i+1 for i in range(num_funcs)] + [1./(i+1) for i in range(num_funcs)]
        ))

    # 动态计算编码后的特征维度
    def get_out_dim(self, in_dim):
        return in_dim * (1 + 2*2*self.num_funcs)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)  # (..., d, 1)
        # 沿最后一个维度拼接原始值、正弦和余弦分量
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


class LearnableAngularEncoding(nn.Module):
    """
    可学习频率的多频带正弦/余弦编码
    频率参数通过训练优化，适应数据分布
    """
    def __init__(self, num_funcs=3):
        super().__init__()
        self.num_funcs = num_funcs
        # 初始化可学习的频率参数：前半部分为正频率，后半部分为倒数频率
        # 初始值设为随机值，范围 [0.1, 5.0]，避免极端值影响训练稳定性
        self.freq_bands = nn.Parameter(torch.rand(2 * num_funcs) * 4.9 + 0.1)

    def get_out_dim(self, in_dim):
        """
        计算编码后的输出维度
        Args:
            in_dim: 输入角度特征的维度（如 2 表示 φ 和 ψ）
        Returns:
            编码后的总维度
        """
        return in_dim * (1 + 2 * 2 * self.num_funcs)  # 原始值 + sin(freq*x) + cos(freq*x)

    def forward(self, x):
        """
        Args:
            x: 输入角度张量，形状为 (..., d)，d 为角度特征维度
        Returns:
            编码后的张量，形状为 (..., d * (1 + 2 * 2 * self.num_funcs))
        """
        shape = list(x.shape[:-1]) + [-1]
        # 扩展维度以适配频率参数
        x = x.unsqueeze(-1)  # (..., d, 1)

        # 分离正频率和倒数频率
        pos_freqs = self.freq_bands[:self.num_funcs]  # 正频率
        inv_freqs = self.freq_bands[self.num_funcs:]  # 倒数频率

        # 计算正弦和余弦分量
        sin_components = torch.sin(x * pos_freqs).flatten(-2)  # (..., d * num_funcs)
        cos_components = torch.cos(x * inv_freqs).flatten(-2)  # (..., d * num_funcs)

        # 拼接原始值、正弦和余弦分量
        code = torch.cat([x.squeeze(-1), sin_components, cos_components], dim=-1)  # (..., d * (1 + 2*num_funcs))
        code = code.reshape(shape)
        return code

class NeuralFourierEncoding(nn.Module):
    def __init__(self, input_dim=5, fourier_dim=64):
        """
        input_dim: 输入角度维度（5个侧链角）
        fourier_dim: 编码维度
        """
        super().__init__()
        self.fourier_dim = fourier_dim
        self.freq = nn.Parameter(torch.randn(input_dim, fourier_dim), requires_grad=True)  # 可学习频率
        self.phase = nn.Parameter(torch.randn(fourier_dim), requires_grad=True)  # 可学习相位

    def get_out_dim(self):
        return self.fourier_dim * 2

    def forward(self, angles):
        """
        angles: Tensor of shape (batch, res, input_dim), 单位为弧度
        return: Tensor of shape (batch, res, fourier_dim * 2)
        """
        # 计算投影 angles @ freq + phase
        projected = torch.einsum('brd,df->brf', angles, self.freq) + self.phase  # (batch, res, fourier_dim)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
