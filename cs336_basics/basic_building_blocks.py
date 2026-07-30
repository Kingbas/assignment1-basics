
import torch
from torch import Tensor
from einops import einsum, rearrange
import math
from jaxtyping import Bool, Float, Int

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2/(in_features+out_features))
        W = torch.nn.Parameter(torch.zeros([out_features, in_features], dtype=dtype, device=device))
        self.W = torch.nn.init.trunc_normal_(W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')
        return x


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        E = torch.nn.Parameter(torch.rand([num_embeddings, embedding_dim], device=device, dtype=dtype))
        self.E = torch.nn.init.trunc_normal_(E, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.E[token_ids]
        
            

if __name__ == '__main__':
    l = Linear(3, 1)
    x = torch.rand([3,3])
    weights = torch.rand([1,3])
    out = l(x)
    pass

