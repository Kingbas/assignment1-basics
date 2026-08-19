"""对应 tests/test_nn_utils.py:softmax、交叉熵损失、梯度裁剪。"""

from collections.abc import Iterable

import torch
from jaxtyping import Float, Int
from torch import Tensor


def softmax(x, n_dim):
    # 首先把n_dim维的最大值找出来
    x_max = torch.max(x, dim=n_dim, keepdim=True).values
    x = x - x_max
    temp = torch.exp(x)
    x_sum = torch.sum(temp, dim=n_dim, keepdim=True)
    x = temp/ x_sum
    return x


def cross_entropy_loss(inputs: Float[Tensor, " ... seq_len vocab_size"], targets: Int[Tensor, " ... seq_len"]) -> Float[Tensor, ""]:
    # targets中是vocab中对应的index
    # inputs中是未标准化的 [x_0:x_i]的下一个词在vocab中的logits
    # 根据target[0]，可以算出x_0:x_0的交叉熵，以此类推，分子为目标输出的概率
    inputs = inputs.reshape(-1, inputs.size(-1)) # total_seq vocab_size
    targets = targets.reshape(-1) # total_seq
    token_num = inputs.shape[-2]
    # 计算最大值
    logits_max = inputs.max(dim=-1, keepdim=True).values # total_seq 1
    inputs = inputs - logits_max
    # 计算交叉熵的分子
    logits = inputs[torch.arange(token_num, device=inputs.device), targets] # 用total_seq替换被索引的维度 total_seq
    # 计算交叉熵
    p = logits - torch.log(torch.sum(torch.exp(inputs), dim=-1)) # total_seq - total_seq = total_seq
    loss = -torch.mean(p)
    return loss


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6) -> None:
    # 将parameter放进容器中
    parameters = list(parameters)
    # 逐张量计算L2范式，累加并开根
    grad_square_sum = torch.zeros([], device=parameters[0].device) + 0.0
    for p in parameters:
        if p.grad is not None:
            grad_square_sum += p.grad.square().sum()
    g2 = torch.sqrt(grad_square_sum)

    c = torch.clamp(max_l2_norm / (g2 + eps), max=1.0)

    for p in parameters:
        if p.grad is not None:
            p.grad = p.grad * c
