
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
    def __init__(self, d_model, d_ff=None, device=None, dtype=None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int((8/3 * d_model) // 64 * 64)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        # gate
        x1 = self.w1(x)
        x1 = torch.sigmoid(x1) * x1
        # value
        x2 = self.w3(x)
        # output
        return self.w2(x1 * x2)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        assert d_k % 2 == 0
        super().__init__()
        # 对于d_k维度的词嵌入，有k对词嵌入元素，k的取值为0，1，……，d_k/2 - 1，算出来1/theta^(2k/d)
        # 踩坑：arange intg生成int，linspace生成float
        k = torch.arange(0, d_k/2, 1) # (d_k/2,)
        # 对于max_seq_len，期望与k外积生成一个max_seq_len, d_k/2的张量
        seq = torch.arange(0, max_seq_len, 1)
        seq = rearrange(seq, 'max_seq_len -> max_seq_len 1')
        # 算出来对于每一个k的频率，频率i/theta^(2k/d)的shape是(max_seq_len, d_k/2)
        theta_seq = theta ** (-2*k/d_k)
        # 先把频率搞定
        cache = seq * theta_seq # max_seq_len 1 , d_k/2 -> max_seq_len d_k/2
        # 再搞定对应的sin cos cache，shape是(max_seq_len, d_k/2, 2, 2)
        cache = torch.stack([torch.cos(cache), -torch.sin(cache), torch.sin(cache), torch.cos(cache)], dim=-1) # (max_seq_len, d_k/2, 4)
        cache = rearrange(cache, '... (i j) -> ... i j', i=2)
        # 对于每一个i，有d_k/2对sin cos的缓存,sin cos缓存cache的维度是 d_k/2 2 2
        self.register_buffer('cache', cache.to(device), persistent=False)

    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:
        # 对于一个输入为'... seq_len d_k'的tensor，先reshape成'... seq_len d_k/2 2'，对于'... seq'的索引，前者为even项，后者为odd项
        # 对于论文中的theta_i k，seq_len维度代表position i，k则是d_k中的第几对词嵌入元素
        # 对于每一个i，有d_k/2对sin cos的缓存,sin cos缓存cache的维度是 d_k/2 2 2
        # 那么input @ cache.T就是结果，再reshape回... seq_len d_k就可以了
        
        #input中tokenpositions代表x中每一个token的position，因此要对每一个position查出来sin cos cache
        # 先将x拆分成奇偶
        x = rearrange(x, '... (pairs pair) -> ... pairs pair', pair=2) # ... sequence_length d_k/2 2
        # 从cache中取出对应position的2*2cache
        R = self.cache[token_positions] # R维度是 ... sequence_length d_k/2 2 2,[token_positions]消费了cache的一维（左对齐），接着拼接cache剩余维度
        # x是... sequence_length d_k/2 2，R是... sequence_length d_k/2 2 2
        # 对于每一个sequence token，对每个pair里面的词嵌入元素，与R中每个sequence token的2 2矩阵相乘，获得旋转后的词嵌入元素
        # x = einsum(x, R, '... sequence_length pairs row, ... sequence_length pairs col row -> ... sequence_length pairs col')
        x = einsum(x, R, '...  pairs row, ...  pairs col row -> ...  pairs col') # alternative impl of the einsum above
        x = rearrange(x, '... pairs out -> ... (pairs out)')
        return x


def softmax(x, n_dim):
    # 首先把n_dim维的最大值找出来
    x_max = torch.max(x, dim=n_dim, keepdim=True).values
    x = x - x_max
    temp = torch.exp(x)
    x_sum = torch.sum(temp, dim=n_dim, keepdim=True)
    x = temp/ x_sum
    return x


def scaledDotProductAttention(Q: Float[Tensor, " ... queries d_k"],
                        K: Float[Tensor, " ... keys d_k"],
                        V: Float[Tensor, " ... keys d_v"],
                        mask: Bool[Tensor, " ... queries keys"] | None = None):
    d_k = Q.shape[-1]
    out = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
    out = out / math.sqrt(d_k)
    # 把mask中的False替换为-torch.inf
    if mask is not None:
        out = out.masked_fill(~mask, -torch.inf)
    # mask = torch.where(self.mask, 0.0, -torch.inf)
    out = softmax(out, -1)
    out = einsum(out, V, '... queries keys, ... keys d_v -> ... queries d_v')
    return out


class CasualMultiHeadSelfAttention(torch.nn.Module):
    '''
        不带rope的
    '''
    def __init__(self, d_model:int, num_heads:int, device=None, dtype=None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.device = device
        self.dtype = dtype
        self.d_head = int(d_model / num_heads)
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)


    def forward(self, x: Float[Tensor, '... seq d_model']):
        x = x.to(self.device)
        Q = self.q_proj(x) # ... seq d_model
        Q = rearrange(Q, '... seq (h d_head) -> ... h seq d_head', d_head=self.d_head)
        K = self.k_proj(x) # ... seq d_model
        K = rearrange(K, '... seq (h d_head) -> ... h seq d_head', d_head=self.d_head)
        V = self.v_proj(x)
        V = rearrange(V, '... seq (h d_head) -> ... h seq d_head', d_head=self.d_head)
        seq = Q.shape[-2]
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).to(self.device)
        # scores = einsum(Q, K, '... h seq_q d_head, ... h seq_k d_head -> ... seq_q seq_k')
        # scores = scores.masked_fill(~mask, -torch.inf)
        scores = scaledDotProductAttention(Q, K ,V, mask) # ... h seq_q d_head
        scores = rearrange(scores, '... h seq_q d_head -> ... seq_q (h d_head)')
        return self.output_proj(scores)



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

    rope = RotaryPositionalEmbedding(10000.0, d_k=8, max_seq_len=16)

    print(list(rope.state_dict().keys()))          # → []
    print([n for n, _ in rope.named_buffers()])    # → ['cache']
    print(rope.cache.shape)                        # → (16, 4, 2, 2)
    print(rope.cache[0])                           # → 4 个单位矩阵
    print(rope.cache[5, 0])
    # → [[ 0.2837,  0.9589],
    #    [-0.9589,  0.2837]]

    x = torch.rand(1, 3, 8)
    pos = torch.tensor([[3, 7, 1]])          # 故意乱序、不从 0 开始
    y = rope(x, pos)

    # 单独把第 1 个 token 按位置 7 算一次
    y_single = rope(x[:, 1:2], torch.tensor([[7]]))
    assert torch.allclose(y[:, 1], y_single[:, 0], atol=1e-6)

    pass

