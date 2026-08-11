
from ast import List
import torch
from torch import Tensor
from einops import einsum, rearrange
import math
from jaxtyping import Bool, Float, Int
from typing import Any
from torch.optim.optimizer import ParamsT
from collections.abc import Callable, Iterable
from typing import Optional

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


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return torch.sigmoid(in_features) * in_features


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = int(math.ceil(8/3 * d_model / 64) * 64)
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, '... d_model']) -> Float[Tensor, '... d_model']:
        # gate
        x1 = self.w1(x)
        x1 = silu(x1)
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
        cache = torch.stack([torch.cos(cache), -torch.sin(cache), torch.sin(cache), torch.cos(cache)], dim=-1) # (max_seq_len, d_k/2) * 4 -stack> (max_seq_len, d_k/2, 4)
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


def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
                        K: Float[Tensor, " ... keys d_k"],
                        V: Float[Tensor, " ... keys d_v"],
                        mask: Bool[Tensor, " ... queries keys"] | None = None):
    d_k = Q.shape[-1]
    scores = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
    scores = scores / math.sqrt(d_k)
    # 把mask中的False替换为-torch.inf
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    # mask = torch.where(self.mask, 0.0, -torch.inf)
    scores = softmax(scores, -1)
    scores = einsum(scores, V, '... queries keys, ... keys d_v -> ... queries d_v')
    return scores


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, theta:float | None = None, max_seq_len:int | None = None, device=None, dtype=None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = int(d_model // num_heads)
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device)
        else:
            self.rope = None


    def forward(self, x: Float[Tensor, '... seq d_model'], token_positions:Int[Tensor, " ... sequence_length"] | None = None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.int)
        token_positions = rearrange(token_positions, '... seq -> ... 1 seq')
        Q = self.q_proj(x) # ... seq d_model
        Q = rearrange(Q, '... seq (h d_head) -> ... h seq d_head', d_head=self.d_head)
        K = self.k_proj(x) # ... seq d_model
        K = rearrange(K, '... seq (h d_head) -> ... h seq d_head', d_head=self.d_head)
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        V = self.v_proj(x)
        V = rearrange(V, '... seq (h d_head) -> ... h seq d_head', d_head=self.d_head)
        seq = Q.shape[-2]
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).to(x.device)
        scores = scaled_dot_product_attention(Q, K ,V, mask) # ... h seq_q d_head
        scores = rearrange(scores, '... h seq_q d_head -> ... seq_q (h d_head)')
        return self.output_proj(scores)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int,
                    num_heads: int,
                    d_ff: int,
                    max_seq_len: int,
                    theta: float,
                    device=None,
                    dtype=None) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        residual = x
        x = self.ln1(x)
        x = self.attn(x, token_positions)
        x = residual + x
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        return residual + x

class TransformerLM(torch.nn.Module):
    def __init__(self,
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float,) -> None:
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta))
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        # 计算可学习参数量
        # vocab_size:  50,257
        # context_length:  1,024
        # num_layers:  48
        # d_model:  1,600
        # num_heads:  25
        # d_ff:  4,288
        # result: 1640452800
        # print(sum(p.numel() for p in self.parameters()))
        
    
    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


def cross_entropy_loss(inputs: Float[Tensor, " ... seq_len vocab_size"], targets: Int[Tensor, " ... seq_len"]) -> Float[Tensor, ""]:
    # targets中是vocab中对应的index
    # inputs中是未标准化的 [x_0:x_i]的下一个词在vocab中的logits
    # 根据target[0]，可以算出x_0:x_0的交叉熵，以此类推，分子为目标输出的概率
    batch_boradcast = torch.ones([1,1,1], dtype=torch.int)
    inputs = inputs * batch_boradcast
    seq_len = inputs.shape[-2]
    batch_size = inputs.shape[-3]
    # 计算最大值
    logits_max = inputs.max(dim=-1, keepdim=True).values # ... seq_len 1
    inputs = inputs - logits_max
    # 计算交叉熵的分子
    logits = inputs[..., torch.arange(seq_len, device=inputs.device), targets] # 用seq_len替换被索引的维度 ... seq_len
    # 计算交叉熵
    p = logits - torch.log(torch.sum(torch.exp(inputs), dim=-1)) # ... seq_len  - ... seq_len = ... seq_len
    loss = -torch.sum(p) / (seq_len * batch_size)
    return loss



class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError("Invalid beta")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                if t == 1:
                    self.state[p]['m'] = torch.zeros_like(p)
                    self.state[p]['v'] = torch.zeros_like(p)
                
                lr_t = lr * math.sqrt(1 - group["beta2"] ** t) / (1 - group["beta1"] ** t) # if t is initiated with 0, the divisor will be zero
                p.data = p.data - lr * group["weight_decay"] * p.data

                self.state[p]['m'] = group["beta1"] * self.state[p]['m'] + (1 - group["beta1"]) * p.grad
                self.state[p]['v'] = group["beta2"] * self.state[p]['v'] + (1 - group["beta2"]) * p.grad ** 2
                p.data = p.data - lr_t * self.state[p]['m'] / (torch.sqrt(self.state[p]['v']) + group["eps"])

                state["t"] = t + 1  # Increment iteration number.
        return loss

def learning_rate_schedule(it: int,
                        max_learning_rate: float,
                        min_learning_rate: float,
                        warmup_iters: int,
                        cosine_cycle_iters: int,):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(((it - warmup_iters)/(cosine_cycle_iters - warmup_iters))*math.pi))*(max_learning_rate - min_learning_rate)
    return min_learning_rate


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
    

    vocab_size=  50257
    context_length= 1024
    num_layers= 48
    d_model= 1600
    num_heads= 25
    d_ff= 4288
    lm = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, 10000)
    # 还真是1640452800

    pass
