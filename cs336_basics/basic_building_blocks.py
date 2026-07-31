
import torch
from torch import Tensor
from einops import einsum, rearrange
import math
from jaxtyping import Bool, Float, Int
from typing import Any

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2/(in_features+out_features))
        W = torch.nn.Parameter(torch.zeros([out_features, in_features], dtype=dtype, device=device))
        self.weight = torch.nn.init.trunc_normal_(W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
        return x


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        E = torch.nn.Parameter(torch.empty([num_embeddings, embedding_dim], device=device, dtype=dtype))
        self.weight = torch.nn.init.trunc_normal_(E, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones([d_model], dtype=dtype, device=device))

    def forward(self, x: Float[torch.Tensor, 'batch_size sequence_length d_model']) -> Float[torch.Tensor, 'batch_size sequence_length d_model']:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(1/self.d_model * torch.square(x).sum(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight / rms
        return x.to(original_dtype)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if d_ff is None:
            d_ff = int((8/3 * d_model) // 64 * 64)
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        # gate
        x1 = self.w1(x)
        x1 = torch.sigmoid(x1) * x1
        # value
        x2 = self.w3(x)
        # output
        return self.w2(x1 * x2)


if __name__ == '__main__':
    l = Linear(3, 1)
    x = torch.rand([3,3])
    weights = torch.rand([1,3])
    out = l(x)

    d_model = 64
    d_ff = 256
    swiglu = SwiGLU(d_model)
    print(list(swiglu.state_dict().keys()))
    swiglu(torch.rand([1,64]))
    pass

