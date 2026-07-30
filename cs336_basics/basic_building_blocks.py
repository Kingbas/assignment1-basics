
import torch
from einops import einsum
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, weights=None, device=None, dtype=None):
        super().__init__()
        if weights is None:
            std = math.sqrt(2/(in_features+out_features))
            self.W = torch.nn.init.trunc_normal_(torch.zeros([out_features, in_features]), mean=0, std=std, a=-3*std, b=3*std)
        else:
            self.W = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(x, self.W, '... d_in, d_out d_in -> ... d_out')
        return x
        

if __name__ == '__main__':
    l = Linear(3, 1)
    x = torch.rand([3,3])
    out = l(x)
    pass

