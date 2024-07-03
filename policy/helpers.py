import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    # 定义 forward 方法，实现前向传播的逻辑。x 是输入张量。
    def forward(self, x):
        # 获取输入张量 x 所在的设备（例如 CPU 或 GPU）。
        device = x.device
        # 计算 dim 的一半，并存储在变量 half_dim 中。
        half_dim = self.dim // 2
        # 计算一个常数 emb，它是 math.log(10000) 除以 (half_dim - 1) 的值。这个常数用于生成频率。
        emb = math.log(10000) / (half_dim - 1)
        # 生成一个长度为 half_dim 的张量，值为从 0 到 half_dim-1 的整数，然后乘以 -emb 并取指数。这产生了一组以指数方式变化的频率
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 将输入 x 扩展一个维度，使其形状变为 (batch_size, 1)，然后与频率张量 emb（形状为 (1, half_dim)）相乘，得到一个新的张量 emb，其形状为 (batch_size, half_dim)。
        emb = x[:, None] * emb[None, :]
        # 将 emb 的正弦和余弦值计算出来，并在最后一个维度上进行拼接，得到形状为 (batch_size, dim) 的张量。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#


# extract 函数从张量 a 中提取值，并根据 t 的索引进行选择，最后将结果调整为 x_shape 的形状
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# 生成一个基于余弦函数的 beta 时间表，这是改进的去噪扩散概率模型中的关键部分。
def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


# 生成一个线性变化的 beta 时间表，适用于去噪扩散概率模型（DDPM）
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)


# 生成一个基于变分推理的 beta 时间表，适用于去噪扩散概率模型（DDPM）。
def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

# WeightedLoss 是一个抽象基类，用于计算加权损失。它继承自 torch.nn.Module，并定义了一个 forward 方法，该方法计算加权损失
class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

# WeightedLoss 是一个抽象基类，用于计算加权损失。它继承自 torch.nn.Module，并定义了一个 forward 方法，该方法计算加权损失
class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

# WeightedL2 类继承 WeightedLoss 并实现 _loss 方法，计算 L2 损失（均方误差）。
class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


# Losses 字典将损失类型字符串映射到对应的损失类，方便根据字符串名称动态创建损失实例。
Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


# EMA 类用于实现指数移动平均（Exponential Moving Average），常用于模型参数的平滑更新
class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    #  更新模型平均值的方法，将当前模型参数与移动平均模型参数结合。
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    # 更新平均值的方法。
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new