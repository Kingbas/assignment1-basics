# CS336 Assignment 1 (basics) — Writeup

> 按 `cs336_assignment1_basics.pdf` 的章节顺序组织。
> 标注说明:📝 = 书面题(在本文件作答);💻 = 代码题(在 `cs336_basics/` 实现,此处只记录结果/备注)。

---

## 2. Byte-Pair Encoding (BPE) Tokenizer

### Problem (unicode1): Understanding Unicode (1 point) 📝

**(a) `chr(0)` 返回什么 Unicode 字符?**

```python
ord('0')  # >> 48
```

**(b) 它的字符串表示(`__repr__()`)与打印出来的表示有何不同?**

以 `chr(0)` 举例,0 是 NUL:

```python
chr(0)             # >> '\x00'
assert len('\x00') == 1
chr(0).__repr__()  # >> "'\\x00'"
assert len("'\\x00'") == 6
```

综上,repr 是产出"如何在代码中写出这个字符"的描述文本,而不是字符本身。

**(c) 当这个字符出现在文本中会发生什么?**

```python
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

接着第二问继续探索,在字符串拼接中,隐式调用了 chr(0) 的 `__repr__`,而 print 中则打印出了字符本身。

### Problem (unicode2): Unicode Encodings (3 points) 📝

**(a) 为什么倾向于在 UTF-8 字节上训练 tokenizer,而不是 UTF-16 / UTF-32?**

答: 一个是基于词汇量的考虑，UTF-16或UTF-32太稀疏。一个是方便管理分词，无论什么字符都统统用byte来分词。

**(b) 给出的 `decode_utf8_bytes_to_str_wrong` 为什么是错的?给一个使它出错的输入示例。**

答: 因为UTF-8是变长的，不能逐字节解码。输入“你好”就不能解码了。

**(c) 给出一个不能解码为任何 Unicode 字符的两字节序列,并解释原因。**

答: 0xE4E4。因为3字节的UTF-8的模式是 1110xxxx 10xxxxxx 10xxxxxx，连续两个1110 xxxx 1110xxxx自然就不能解码了。

### Problem (train_bpe): BPE Tokenizer Training (15 points) 💻

- [ ] 实现完成,`uv run pytest tests/test_train_bpe.py` 通过
- 实现位置: cs336_basics/bpe_tokenizer_trainer.py
- 备注(优化思路 / 踩坑记录):
遍历 bytes 出 int:构造 tuple[bytes,...] 要用 bytes([x]) 或切片,不能直接迭代
加权统计错:pair 计数应 += 词频,你早期写成先 +1 再乘词频
重新绑定 bug:merger 里 pretrained_tokenization = freq_temp 不影响调用方 → 每轮选到同一个 pair(要么 return 接住,要么就地改)
全局状态污染:模块级全局变量不重置,pytest 同进程跨测试泄漏 → 改用局部变量 + 返回值
连续相同元素合并:(a,a,a) 合并要用 while 手动步进(命中 i+=2,否则 i+=1)
平局规则:最高频有多个时按字典序取大,max(pc, key=lambda p:(pc[p], p))
max() 空集合崩:语料 merge 榨干时 pair_count 为空 → 要加"无 pair 可合就停"的出口

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points) 📝

**(a) 训练耗时、内存占用;词表中最长的 token 是什么?合理吗?**

答:

**(b) profiling:训练中最耗时的部分是什么?**

答:

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points) 📝

**(a) OWT 词表中最长的 token?合理吗?**

答:

**(b) 对比 TinyStories 与 OWT 的 tokenizer / 词表差异。**

答:

### Problem (tokenizer): Implementing the tokenizer (15 points) 💻

- [ ] 实现完成,`uv run pytest tests/test_tokenizer.py` 通过
- 实现位置:
踩坑：
遍历 bytes 出 int:构造 tuple[bytes,...] 要用 bytes([x]) 或切片,不能直接迭代
加权统计错:pair 计数应 += 词频,你早期写成先 +1 再乘词频
重新绑定 bug:merger 里 pretrained_tokenization = freq_temp 不影响调用方 → 每轮选到同一个 pair(要么 return 接住,要么就地改)
全局状态污染:模块级全局变量不重置,pytest 同进程跨测试泄漏 → 改用局部变量 + 返回值
连续相同元素合并:(a,a,a) 合并要用 while 手动步进(命中 i+=2,否则 i+=1)
平局规则:最高频有多个时按字典序取大,max(pc, key=lambda p:(pc[p], p))
max() 空集合崩:语料 merge 榨干时 pair_count 为空 → 要加"无 pair 可合就停"的出口

### Problem (tokenizer_experiments): Experiments with tokenizers (4 points) 📝

**(a) 采样文档的压缩比(bytes/token)?**

答:

**(b) 交叉使用 tokenizer(TinyStories tokenizer 编码 OWT 等)会怎样?**

答:

**(c) tokenizer 吞吐量估算;编码 825GB Pile 需要多久?**

答:

**(d) 为什么 uint16 是存储 token ID 的合适选择?**

答:

---

## 3. Transformer Language Model Architecture

### Problem (linear): Implementing the linear module (1 point) 💻

- [ ] `uv run pytest -k test_linear` 通过

### Problem (embedding): Implement the embedding module (1 point) 💻

- [ ] `uv run pytest -k test_embedding` 通过

### Problem (rmsnorm): Root Mean Square Layer Normalization (1 point) 💻

- [ ] `uv run pytest -k test_rmsnorm` 通过

### Problem (positionwise_feedforward): Position-wise feed-forward network (2 points) 💻

- [ ] `uv run pytest -k test_swiglu` 通过

### Problem (rope): Implement RoPE (2 points) 💻

- [ ] `uv run pytest -k test_rope` 通过

### Problem (softmax): Implement softmax (1 point) 💻

- [ ] `uv run pytest -k test_softmax` 通过

### Problem (scaled_dot_product_attention): Scaled dot-product attention (5 points) 💻

- [ ] `uv run pytest -k test_scaled_dot_product_attention` 通过

### Problem (multihead_self_attention): Causal multi-head self-attention (5 points) 💻

- [ ] `uv run pytest -k test_multihead_self_attention` 通过

### Problem (transformer_block): Implement the Transformer block (3 points) 💻

- [ ] `uv run pytest -k test_transformer_block` 通过

### Problem (transformer_lm): Implementing the Transformer LM (3 points) 💻

- [ ] `uv run pytest -k test_transformer_lm` 通过

### Problem (transformer_accounting): Transformer LM resource accounting (5 points) 📝

**(a) GPT-2 XL 配置下的可训练参数量与加载所需内存?**

答:

**GPT-2 XL 配置**

| 超参 | 值 |
|---|---|
| `vocab_size` | 50,257 |
| `context_length` | 1,024 |
| `num_layers` | 48 |
| `d_model` | 1,600 |
| `num_heads` | 25 |
| `d_head` = `d_model / num_heads` | 64 |
| `d_ff` | 4,288(8/3 × 1600 向上取到 64 的最近倍数) |

**参数清单**

| 层级 | 模块 | 单个形状 | 个数 | 小计 |
|---|---|---|---|---|
| 模型级 | `token_embeddings.weight` | `vocab_size × d_model` | 1 | `vocab_size · d_model` |
| 每 block | `ln1.weight` | `d_model` | 1 | `d_model` |
| 每 block | `attn` 的 q / k / v / output_proj | `d_model × d_model` | 4 | `4 · d_model²` |
| 每 block | `ffn` 的 w1 / w3 | `d_ff × d_model` | 2 | `2 · d_model · d_ff` |
| 每 block | `ffn` 的 w2 | `d_model × d_ff` | 1 | `d_model · d_ff` |
| 每 block | `ln2.weight` | `d_model` | 1 | `d_model` |
| 模型级 | `ln_final.weight` | `d_model` | 1 | `d_model` |
| 模型级 | `lm_head.weight` | `vocab_size × d_model` | 1 | `vocab_size · d_model` |

注:q / k / v 的权重本是每个 head 一份 `d_head × d_model`，`num_heads` 份竖着摞起来正好是 `d_model × d_model`，所以参数量与 `num_heads` 无关。

RoPE 无可学习参数(cache 是 buffer),不计入。所有 `Linear` 无 bias。

**求和**

```
params_per_block = 2·d_model + 4·d_model² + 3·d_model·d_ff

N = 2·vocab_size·d_model + d_model + num_layers · params_per_block

代入 = 2×50257×1600 + 1600 + 48×(2×1600 + 4×1600² + 3×1600×4288)
     = 160,822,400 + 1,600 + 48 × 30,825,600
     = 1,640,452,800

memory = 4 · N  bytes(fp32)
```

**结果**

| 项 | 代数 | 值 |
|---|---|---|
| 可训练参数量 | `2·vocab_size·d_model + d_model + num_layers·(2·d_model + 4·d_model² + 3·d_model·d_ff)` | **1,640,452,800 ≈ 1.64 B** |
| 单精度(fp32)每参数 | — | 32 bit = 4 B |
| 加载所需内存 | `4·N` | 6,561,811,200 B = **6.11 GiB**(= 6.56 GB) |

此处仅指加载参数本身,不含梯度、优化器状态与激活值。


**(b) 前向传播需要的矩阵乘及对应 FLOPs?**

答:

输入为 1 条序列、`context_length = 1024` 个 token,embedding 输出 `(1, 1024, 1600)`，即 `(batch, context_length, d_model)`。

**不计入矩阵乘的部分**

| 操作 | 量级 | 理由 |
|---|---|---|
| embedding | — | 查表,不涉及矩阵乘法 |
| RMSNorm | `O(context_length · d_model)` | 严格意义上不算矩阵乘法 |
| RoPE | `O(context_length · d_model)` | rope 在初始化的时候进行矩阵乘法，在正向传播时不纳入计算 |
| SwiGLU 的逐元素相乘 | `O(context_length · d_ff)` | 逐元素,不是矩阵乘 |
| softmax | `O(num_heads · context_length²)` | 逐元素 + 行内规约 |
| 拆 / 合 head | 0 | `rearrange` 只改 shape / stride，切割不影响计算量 |

上述每项都比矩阵乘少一个 `d_model` 或 `d_ff` 因子(约 3 个数量级)，故忽略。

**每个 block 内的矩阵乘(共 9 次)**

| # | 矩阵乘 | 形状 | 次数 | FLOPs |
|---|---|---|---|---|
| 1 | q / k / v_proj | `(context_length, d_model) @ (d_model, d_model)` | 3 | `3 · 2·context_length·d_model²` |
| 2 | Q @ Kᵀ | `(context_length, d_head) @ (d_head, context_length)`，每 head 一次 | 1 | `num_heads · 2·context_length²·d_head` = `2·context_length²·d_model` |
| 3 | probs @ V | `(context_length, context_length) @ (context_length, d_head)`，每 head 一次 | 1 | `num_heads · 2·context_length²·d_head` = `2·context_length²·d_model` |
| 4 | output_proj | `(context_length, d_model) @ (d_model, d_model)` | 1 | `2·context_length·d_model²` |
| 5 | w1 / w3 | `(context_length, d_model) @ (d_model, d_ff)` | 2 | `2 · 2·context_length·d_model·d_ff` |
| 6 | w2 | `(context_length, d_ff) @ (d_ff, d_model)` | 1 | `2·context_length·d_model·d_ff` |

第 2 / 3 行的 `num_heads` 被约掉了:每个 head 的收缩是 `2·context_length²·d_head`，`num_heads` 个 head 相加得 `2·context_length²·num_heads·d_head = 2·context_length²·d_model`。因此 FLOPs 也与 `num_heads` 无关。

```
flops_per_block = 8·context_length·d_model²
                + 4·context_length²·d_model
                + 6·context_length·d_model·d_ff

                = 2·context_length·d_model · (4·d_model + 2·context_length + 3·d_ff)

代入 = 2×1024×1600 × (4×1600 + 2×1024 + 3×4288)
     = 3,276,800 × 21,312
     = 69,835,161,600 FLOPs
```

**模型级的矩阵乘**

| 矩阵乘 | 形状 | FLOPs | 数值 |
|---|---|---|---|
| lm_head | `(context_length, d_model) @ (d_model, vocab_size)` | `2·context_length·d_model·vocab_size` | 164,682,137,600 |

**总计**

```
total_flops = num_layers · flops_per_block + 2·context_length·d_model·vocab_size

            = 2·context_length·d_model
              · [ num_layers·(4·d_model + 2·context_length + 3·d_ff) + vocab_size ]

代入 = 3,276,800 × (48 × 21,312 + 50,257)
     = 3,276,800 × 1,073,233
     = 3,516,769,894,400 FLOPs ≈ 3.5 × 10¹² FLOPs
```

校验:经验公式 `2 · N · context_length = 2 × 1,640,452,800 × 1024 ≈ 3.36 × 10¹²`，与精确值相差 4.7%。偏高的原因:attention 的两次收缩没有参数却有 FLOPs，超过了 embedding 有参数但 0 FLOPs 省下的部分。


**(c) 哪些部分占 FLOPs 最多?**

答: 48个block

**(d) GPT-2 small / medium / large 的 FLOPs 分布对比,随规模变化的趋势?**

答:

由 `cs336_basics/calculations.py` 在 `meta` 设备上逐个构造模型统计得出（`context_length = 1024`）:

规模:

| model | d_model | layers | heads | d_ff | params | fp32 GiB |
|---|---|---|---|---|---|---|
| GPT-2 small | 768 | 12 | 12 | 2,048 | 162,148,608 | 0.60 |
| GPT-2 medium | 1,024 | 24 | 16 | 2,752 | 406,539,264 | 1.51 |
| GPT-2 large | 1,280 | 36 | 20 | 3,456 | 842,438,400 | 3.14 |
| GPT-2 XL | 1,600 | 48 | 25 | 4,288 | 1,640,452,800 | 6.11 |

FLOPs 分布（单位 TFLOPs，一次前向，`batch = 1`）:

| model | 1 block | blocks 合计 | block % | lm_head | lm_head % | 合计 | 2ND | ratio |
|---|---|---|---|---|---|---|---|---|
| GPT-2 small | 0.0177 | 0.213 | 72.9% | 0.079 | **27.1%** | 0.292 | 0.332 | **0.878** |
| GPT-2 medium | 0.0302 | 0.725 | 87.3% | 0.105 | **12.7%** | 0.830 | 0.833 | **0.997** |
| GPT-2 large | 0.0460 | 1.655 | 92.6% | 0.132 | **7.4%** | 1.787 | 1.725 | **1.036** |
| GPT-2 XL | 0.0698 | 3.352 | 95.3% | 0.165 | **4.7%** | 3.517 | 3.360 | **1.047** |

- `token_embeddings` 是查表，0 FLOPs，所以 `blocks 合计 + lm_head = 合计`（代码里写成了 `assert`）
- `ratio` = 合计 / 经验公式 `2·N·context_length`

对 `ratio` 的解释 —— 把精确值与 `2ND` 同时除以公因子 `2·context_length·d_model` 后相减:

```
精确 / (2·T·d) = num_layers·(4·d_model + 2·context_length + 3·d_ff) + vocab_size
2ND  / (2·T·d) = N / d_model
              = num_layers·(2 + 4·d_model + 3·d_ff) + 2·vocab_size + 1

残差 / (2·T·d) = 2·num_layers·context_length - vocab_size - 2·num_layers - 1
```

`4·d_model` 与 `3·d_ff` 完全抵消 —— 所有“有权重且做矩阵乘”的部分 `2ND` 算得不差。残差只由参数与 FLOPs **不匹配**的部分组成:

| 项 | 来源 | 方向 |
|---|---|---|
| `+ 2·num_layers·context_length` | attention 两次收缩：有 FLOPs 无参数 | 使 `2ND` **低估** |
| `- vocab_size` | `token_embeddings`：有参数 0 FLOPs | 使 `2ND` **高估** |
| `- 2·num_layers - 1` | RMSNorm 的 gain：有参数，FLOPs 可忽略 | 高估（极小） |

XL 代入:`2×48×1024 - 50257 - 96 - 1 = 47,950`，乘回 `3,276,800` 得 `157,122,560,000`，与 `3,516,769,894,400 - 3,359,647,334,400` 精确相等。

临界条件 `ratio = 1` 即 `2·num_layers·context_length ≈ vocab_size`（`context_length = 1024` 时）:

| model | num_layers | `2·num_layers·context_length` | vs `vocab_size = 50,257` | ratio |
|---|---|---|---|---|
| small | 12 | 24,576 | 小于 | 0.878 |
| medium | 24 | 49,152 | 几乎相等 | 0.997 |
| large | 36 | 73,728 | 大于 | 1.036 |
| XL | 48 | 98,304 | 大于 | 1.047 |

注意残差里 **`d_model` 和 `d_ff` 已经消失** —— 起作用的是 `num_layers`、`context_length` 与 `vocab_size` 的竞争，而不是参数量本身。

注:`d_ff` 统一取 `ceil(8/3 · d_model / 64) · 64`（向上取到 64 的倍数）。handout 原词是 "nearest"，两者在 `d_model = 1600` 上重合（均为 4288），但在 `d_model = 1280` 上不同（ceil 得 3456，nearest 得 3392）。此处选 ceil，以保证 `d_ff ≥ 8/3 · d_model`。

结论: 随着参数量变大，ratio变大，表明block的计算量占比变大，llm_head占比变小

**(e) 上下文长度增至 16,384 后,FLOPs 如何变化?**

答:

GPT-2 XL 在两个 `context_length` 下的对比:

| context_length | 1 block | blocks 合计 | block % | lm_head | lm_head % | 合计 | 2ND | ratio |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 0.0698 | 3.352 | 95.3% | 0.165 | 4.7% | 3.517 | 3.360 | **1.047** |
| 16,384 | 2.7280 | 130.943 | 98.0% | 2.635 | 2.0% | **133.578** | 53.754 | **2.485** |

`context_length` 增为 16 倍，精确 FLOPs 增为 `133.578 / 3.517 ≈ 38` 倍（非线性）。

括号内三项的变化（`4·d_model + 2·context_length + 3·d_ff`）:

| 项 | 对应部分 | T = 1,024 | 占比 | T = 16,384 | 占比 |
|---|---|---|---|---|---|
| `4·d_model` | 4 个投影 | 6,400 | 30.0% | 6,400 | 12.3% |
| `2·context_length` | attention 两次收缩 | 2,048 | **9.6%** | 32,768 | **63.0%** |
| `3·d_ff` | SwiGLU 三个矩阵 | 12,864 | 60.4% | 12,864 | 24.7% |
| 合计 | | 21,312 | 100% | 52,032 | 100% |

三项里**只有 `2·context_length` 变了**，从最小项变成最大项。此时 `ratio` 升至 2.485，经验公式 `2ND` 已彻底失效——它只数参数，而这时过半 FLOPs 来自**无参数**的 attention 收缩。

结论: 

---

## 4. Training a Transformer LM

### Problem (cross_entropy): Implement cross-entropy (1 point) 💻

- [ ] `uv run pytest -k test_cross_entropy` 通过

### Problem (learning_rate_tuning): Tuning the learning rate (1 point) 📝

**SGD toy example 中不同学习率下 loss 的行为?**

答: lr越大loss衰减越快，一次更新的梯度就越大

### Problem (adamw): Implement AdamW (2 points) 💻

- [ ] `uv run pytest -k test_adamw` 通过

### Problem (adamw_accounting): Resource accounting for AdamW (2 points) 📝

**(a) 峰值内存的表达式(参数/梯度/优化器状态/激活)?**
# TODO

用batch_size vocab_size, context_length, num_layers, d_model, num_heads回答

答: 在训练中，内存使用分为四大类

参数本身：P

梯度：P

adamW优化器的state：2P

所以本题只用分析激活值

激活值：前向传播过程中为反向传播保存的中间张量

理想状况下

transformer block中的RMSNorm、MHA、SwiGLU

RMSNorm：

    def forward(self, x: Float[torch.Tensor, 'batch_size sequence_length d_model']) -> Float[torch.Tensor, 'batch_size sequence_length d_model']:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(1/self.d_model * torch.square(x).sum(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight / rms
        return x.to(original_dtype)
     

     计算x = x * self.weight / rms这里的内存开销峰值为2*输入

MHA：

     x对QKV的投影各存一份输入x

     sdpa中Q*K.T，这里对Q和K各存一份K和Q，softmax存输出，与V矩阵相乘存一个scores矩阵


SwiGLU：


block外的最终RMSNorm、output embedding、交叉熵

RMSNorm：

同上

output embedding：



交叉熵：



QKV投影是xQ xK xV，对于QKV要保存x， batch seq d_model * 3，以及QKV自身的参数和梯度




**(b) GPT-2 XL 在 80GB 内存下能用的最大 batch size?**

答:

**(c) 一步 AdamW 的 FLOPs?**

答:

**(d) MFU 假设下训练 400K 步需要多少天?**

答:

### Problem (learning_rate_schedule): Cosine LR schedule with warmup 💻

- [ ] `uv run pytest -k test_get_lr_cosine_schedule` 通过

### Problem (gradient_clipping): Implement gradient clipping (1 point) 💻

- [ ] `uv run pytest -k test_gradient_clipping` 通过

---

## 5. Training Loop

### Problem (data_loading): Implement data loading (2 points) 💻

- [ ] `uv run pytest -k test_get_batch` 通过

### Problem (checkpointing): Implement model checkpointing (1 point) 💻

- [ ] `uv run pytest tests/test_serialization.py` 通过

### Problem (training_together): Put it together (4 points) 💻

- [ ] 训练脚本完成(可配置超参、日志、checkpoint)
- 脚本位置:

---

## 6. Generating Text

### Problem (decoding): Decoding (3 points) 💻

- [ ] 实现 temperature scaling + top-p 采样的解码函数
- 实现位置:

---

## 7. Experiments

### Problem (experiment_log): Experiment logging (3 points) 📝

实验日志(每次 run 的配置、曲线、结论):

### Problem (learning_rate): Tune the learning rate (3 points) 📝

**不同 LR 的 loss 曲线与结论;LR 与发散边界的关系?**

答:

### Problem (batch_size_experiment): Batch size variations (1 point) 📝

答:

### Problem (generate): Generate text (1 point) 📝

生成样本与质量评价:

### Problem (layer_norm_ablation): Remove RMSNorm and train (1 point) 📝

答:

### Problem (pre_norm_ablation): Implement post-norm and train (1 point) 📝

答:

### Problem (no_pos_emb): Implement NoPE (1 point) 📝

答:

### Problem (swiglu_ablation): SwiGLU vs. SiLU (1 point) 📝

答:

### Problem (main_experiment): Experiment on OWT (2 points) 📝

答:

### Problem (leaderboard): Leaderboard (6 points) 📝

- 最终 validation loss:
- 配置与技巧记录:

---

## 附录:踩坑记录(个人学习笔记,非评分内容)

按**根因**归类而非时间顺序,因为根因才可迁移。

### 一、einops pattern 当成"表达式"写

| 错误写法 | 问题 |
|---|---|
| `f'... -> {original_shape}'` | 把运行时形状插进 pattern |
| `'max_seq_len d_k/2 4 -> ...'` | 轴名里有 `/`,当成了表达式 |
| `'... (2 2) -> ... 2 2'` | 数字创建匿名轴,匿名轴之间永不相等,无法跨箭头对应 |
| `'... (pairs pair) -> ...', j=2` | kwarg 名与轴名不一致 |
| `'... pairs out -> ... (paris out)'` | 拼写错误 → identifiers 一边有一边没有 |
| `'... (A B) -> ...'` | 拆/合方向写反(合应该是 `... A B -> ... (A B)`) |

**根因**:把 pattern 当成"描述尺寸的代码",实际它是"描述结构的静态标签"。

**规则**:
- pattern 永远是写死的字符串常量,不能拼接
- 轴名是**标签**,必须是合法标识符(不能有 `/`、不能以数字开头)
- 尺寸走 kwargs(`pair=2`),拆维**必须**给尺寸且**不接受 `-1`**
- `1` 是特例(插入/删除单位轴);其他数字是匿名轴,只能用于不跨箭头引用的场合
- 只给需要区别对待的轴起名,其余交给 `...`

### 二、形状的"局部视角 vs 全局视角"

| 想的 | 实际 |
|---|---|
| "把 token_ids 最后一维替换成 embedding 行" | 每个元素**长出**一维,维数 **+1** 不是替换 |
| cache 是 `(d_k/2, 2, 2)` | 漏了 `max_seq_len` 维 |
| "对每一个 k 建立一个 2×2" | 漏了位置维;角度需要 `(i, k)` 两个下标 |
| einsum 里 R 写 3 根命名轴 | R 实际有 4 根(seq, pairs, row, col) |
| `torch.arange(n)` 是 `(n,1)` | 是 **`(n,)`**,所有 arange/linspace 都返回一维 |

**根因**:脑子里想"一个元素/一个矩阵长什么样",但代码需要"整张表什么形状"。

**对策**:动手前先写**形状轨迹表**,每一步只记形状。

```
k        (d_k/2,)
freqs    (d_k/2,)
seq      (max_seq_len,) → (max_seq_len, 1)
angles   (max_seq_len, d_k/2)      ← 外积，位置维在这里进来
cos/sin  (max_seq_len, d_k/2)
R        (max_seq_len, d_k/2, 2, 2)
```

**einsum 数轴自检**:每个操作数满足 `命名轴个数 + ... 覆盖的个数 = 该张量 ndim`。

### 三、重绑定覆盖(变量生命周期)

| 错误 | 后果 |
|---|---|
| `x1 = self.w1(x)` 后 `x1 = torch.sigmoid(x)` | 覆盖了上一行的投影结果 |
| `theta = 角度表`(参数名也叫 `theta`) | 覆盖了超参数 Θ |
| `cache.to(device)` 不接回来 | `.to()` **不是原地操作**,结果被丢弃 |
| `cache` 一个名字表示"角度表/摊平4元组/2×2表" | 导致乘了下标 `k` 而不是频率 `freqs` |

**根因**:一个名字承担多个概念。

**对策**:每个概念一个名字(`freqs` / `angles` / `R`),且**名字之间留足编辑距离**(`pair` vs `pairs` 太近,才会写出 `paris`)。

### 四、形状对但数值错(最危险,不报错)

| 错误 | 为什么形状检查抓不住 |
|---|---|
| `Linear(d_ff, d_model)` 参数顺序反 | 只在 `d_ff != d_model` 时才报错 |
| SwiGLU 的 `w1`/`w3` 装反 | 两者形状**完全相同** |
| `stack(..., dim=-1)` 而非 `dim=-2` | **形状完全相同**,内容互为转置 → 得到 `Rᵗ`,旋转方向反了 |
| 用 `torch.sigmoid` 而非 SiLU | 形状相同,少乘一个 `x` |
| 漏 `keepdim` 且张量是 `(4,4)` | 广播"成立"但语义错(行范数当列除数) |

**根因**:结构性检查(形状、范数、不变量)只能验证"是某类东西",不能验证"是那个东西"。

**对策**:关键处必须**对手算的数值点**。

```python
# 一条就能定住 R vs Rᵗ 的方向
rope.cache[5, 0]  # → [[ 0.2837, 0.9589], [-0.9589, 0.2837]]
                  # k=0 频率为 1，故位置 5 的角度就是 5 弧度
```

⚠️ 范数不变检查**抓不住** `R` vs `Rᵗ`(转置也是正交矩阵),相对位置不变性检查同样抓不住。

### 五、PyTorch API 陷阱

| 陷阱 | 正确做法 |
|---|---|
| `torch.Tensor(512)` | 大写 T 是 legacy 构造器,把整数当**尺寸**,给 512 个未初始化垃圾值;`torch.Tensor(1e-5)` 直接 TypeError。**永远用 `torch.tensor(数据)` 或 `torch.zeros/empty(形状)`** |
| `d_model`/`eps` 包成 `nn.Parameter` | 不需要梯度的量不要用 Parameter,否则会进 state_dict 导致 `load_state_dict` 报 key 错误 |
| `x.to(dtype)` 当原地操作 | 必须接回来:`x = x.to(...)` |
| 4 维张量用 `.T` | `.T` 仅限 2 维;批量矩阵转置用 `.mT` 或 `.transpose(-2,-1)` |
| 漏 `super().__init__()` | 报 `cannot assign buffer/parameter before Module.__init__() call`;必须在第一行,空括号调用 |
| `register_buffer` 漏 `persistent=False` | 默认 `True` → 进 state_dict → 后续 `load_state_dict(weights)` 报 `Missing key(s)` |
| `linspace` 当 `arange` 用 | `linspace(1, d/2, steps=d/2)` 是 `[1..d/2]`,和"k 从 0 开始"矛盾;整数序列用 `arange` |
| `stack` 期待广播 | `stack` **不广播**,所有输入形状必须完全相同 |
| `squeeze()` 不传 dim | 会去掉所有长度 1 的轴,batch=1 时意外降维 |

**存张量的四种方式对照**:

| | in `parameters()` | in `state_dict` | 被 `.to()` 搬走 | 收梯度 |
|---|---|---|---|---|
| `nn.Parameter` | ✓ | ✓ | ✓ | ✓ |
| buffer `persistent=True` | ✗ | ✓ | ✓ | ✗ |
| buffer `persistent=False` | ✗ | ✗ | ✓ | ✗ |
| **普通张量属性** | ✗ | ✗ | **✗** | ✗ |

最后一行是 GPU 上 `Expected all tensors to be on the same device` 的根源,**本地 CPU 永远暴露不出来**。

### 六、契约类(测试抓不到,但后面会炸)

- 属性名 `W` vs 测试期望的 `weight` —— state_dict key 沿属性名拼接,`load_state_dict` 是**递归**的,叶子名不对递归优势全废
- 改了属性名但 `adapters.py` 里的 key 没同步
- `device`/`dtype` 没往子模块透传 —— 本地 CPU 永远发现不了
- 残留 `raise NotImplementedError`
- **凡是会被序列化的名字都是 API,改名是破坏性变更**

### 七、工作方式

1. **先跑再问**:einops 的 pattern 错误在第一次调用时全部暴露,报错信息很具体(会指出哪个标识符只在一边、哪个 kwarg 没用上)。循环应该是 `写 → 跑 → 报错自己修 → 数值不对/想不通原理 → 再问`
2. **注释和代码同步**:forward 注释写着 `input @ cache.T`,实际用的是 einsum,以后会误导自己
3. **测试用不对称尺寸**:`max_seq_len=4, d_k=8` 时 `d_k/2` 也是 4,形状 `(4,4)` 分辨不出轴顺序;改成 `max_seq_len=16, d_k=8` → `(16,4)`,一眼看出方向。**让每个维度长度都不同,形状本身就成为诊断工具**
4. **小到能手算的玩具例子**:`max_seq_len=3, d_k=8` 手算出角度表再对照,比读十遍文档有效

### 八、值得保留的自查清单(RoPE 为例)

```python
# 结构
assert list(rope.state_dict().keys()) == []          # persistent=False 生效
assert [n for n, _ in rope.named_buffers()] == ['cache']
assert rope.cache.shape == (16, 4, 2, 2)

# 数值(抓方向错误,不可替代)
rope.cache[0]      # → d_k/2 个单位矩阵
rope.cache[5, 0]   # → [[0.2837, 0.9589], [-0.9589, 0.2837]]

# 前向
assert y.shape == x.shape
assert torch.allclose(x[:, 0], y[:, 0], atol=1e-6)                 # 位置 0 恒等
assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)   # 旋转保长度

# 契约(最容易被官方测试漏掉)
# 用非连续位置 [[3, 7, 1]] 验证真的在按 token_positions 查表,
# 而不是偷偷假设位置是 0,1,2,...
```

最后一条最重要:如果实现里偷偷假设了位置连续(比如 `cache[:seq_len]`),前面所有检查都会通过,只有非连续位置测试会挂 —— 而这个 bug 到生成式解码时才会炸。

### 九、概念性收获(不是坑,但值得记)

- **RoPE 的 2×2 块不是工程 trick**:任何实正交矩阵在某个正交基下必然是 2×2 旋转块 + `±1` 的块对角形式(实正交矩阵标准形)。要"位置只通过相对差影响注意力" → 变换必须保内积 → 正交 → 标准形就是 2×2 旋转块
- **为什么必须在 `W_q` 投影之后**:相对位置性质 `(R^i)ᵗR^j = R^{j-i}` 是旋转在**点积**下的代数性质;先旋转 `x` 再过任意矩阵会破坏结构
- **为什么只作用 Q/K 不作用 V**:位置信息通过 `q·k` 点积进入分数,V 不参与点积
- **绝不构造 `(d_k, d_k)` 块对角矩阵**:d_k=64 时 4096 个数里 4032 个是 0(32 倍浪费);存 `(S, d/2, 2, 2)` 只有 2 倍冗余,可接受,且**计算量与逐元素版相同**(都是 4 次乘法/对)
- **`d_ff = 8/3 · d_model` 的来由**:传统 FFN 参数量 `2·d·4d = 8d²`;SwiGLU 三个矩阵 `3·d·d_ff = 8d²` → `d_ff = 8d/3`。取整到 64 倍数是为对齐 tensor core
- **theta 的直觉**:k=0 频率 1,转得最快,编码紧邻;k=d/2-1 频率约 1/Θ,转得极慢,编码长程。一组不同尺度的旋转同时编码远近距离(同傅里叶展开思想)。**调大 theta → 最慢的平面更慢 → 能表达更长距离,这是 NTK scaling 的核心**
- **指数除以 `d` 是归一化**:让频率范围恒为 `[1, 1/Θ]`,与 `d_k` 解耦,改模型宽度时位置编码的"尺度感"不漂移

### 十、Attention 阶段(softmax / SDPA / MHA)

#### einsum 静默错误(同一类根因,又踩了几次)

| 错误 | 后果 | 怎么早发现 |
|---|---|---|
| `'... q d, ... k d -> q k'` 输出漏 `...` | batch/head 被**求和**成一张图,再靠下一步广播回去,形状居然对 | batch 独立性检查;同函数内两个 einsum 一个有 `...` 一个没有 = 信号 |
| `'... seq d, ... seq d -> ... seq seq'` q/k 轴同名 | 同名=同轴,输出重复标识符,直接报错 | 语义不同的轴必须不同名(`queries`/`keys`),即使长度相等 |
| 轴名撒谎:拆完 head 还叫 `d_model` | 不报错,但误导后续 | 名字要反映**当前**含义(拆后是 `d_head`) |

**数轴自检**:每个操作数 `命名轴个数 + ...覆盖数 = ndim`。反过来用:想让 `...` 吃几根,剩下的必须恰好是命名的那几根。

**通用读法**:einsum 报"某下标两处尺寸不一致",几乎总是**形状定义反了或接错层**,先查形状不要改 pattern。(SwiGLU 参数顺序、RoPE-MHA 的 `d_k` 都是这么被抓到的)

#### softmax

- **分子分母必须用同一个移位后的张量**:分子减了 max、分母没减 → 结果整体差一个 `e^{-max}` 因子。诊断法:每行比值是常数 → 分母错;`ln(因子)` ≈ 每行最大值即可确认
- **平移不变性**是最强自查:`softmax(x) == softmax(x + 100)`,不依赖参考答案,且能同时抓住"没减 max"
- **`dim` 是运行时整数 → 不能用 einops**(pattern 是静态字符串)。这是 einops **不适用**的典型场景,必须用 torch 的 `dim=` 风格
- `torch.max(x, dim=...)` 返回 **namedtuple**,要 `.values`,或直接用 `torch.amax`
- 别 `torch.exp(x)` 算两遍(分子分母),存下来复用 —— 分数矩阵是 `seq²` 量级

#### mask

- **涉及 inf 永远用 `where`/`masked_fill`,不用算术**:`0 * inf = nan` 是定时炸弹
- `masked_fill` **不是原地**,必须接回来(`masked_fill_` 才是原地)—— 这是"返回张量的独立一行没接收 = bug"规律的第三次出现
- 屏蔽位置填 **`-inf` 不是 0**:`exp(-inf)=0` 权重才真的为 0;填 0 得到 `exp(0)=1` 反而有权重
- mask 必须加在 **softmax 之前、缩放之后**
- `~` 只对 bool/整型有效;对 float 是按位取反给乱码。务必让 mask 生成时就是 bool
- **整行全屏蔽 → softmax 分母 0 → nan**(causal 下不会,padding mask 下会;届时用 `torch.finfo(dtype).min` 代替 `-inf`)

#### causal mask

- **方向**:`key 下标 j ≤ query 下标 i`(只能看自己和过去),下三角含对角线为"保留"。记反了(`i ≤ j`)= 只能看未来 = 训练时抄答案,loss 异常低但生成全是垃圾
- **对角线必须包含**:漏了 → 位置 0 无 key 可看 → 整行 `-inf` → nan
- **一横一竖广播**:`i[:,None]`(query 竖)`<= j[None,:]`(key 横)。方向对调得到转置 = 反因果。记语义(query 是行)不记符号
- **`device=x.device`**、**不要 expand 到 batch/head**(靠广播)、**别注册成 persistent buffer**(否则 load_state_dict 报 missing key)
- 自查:`mask.sum(-1) == [1,2,3,...,seq]`,一个向量区分方向错(得递减)和漏对角线(得 `[0,1,2,...]`)

#### MHA / 多头

- **沿 feature 维(-1)切 head,不沿 seq**;`-1` 是行向量布局的结论不是原因。feature 从头到尾在最后一维,**唯一例外是 scores(最后两维都是位置)**
- **`h` 必须排在 `seq` 前面**:`(..., h, seq, d_head)`,让 `...` 把 `h` 当 batch 吃掉 → 每个 head 自动独立,SDPA 一行不用改。停在 `(..., seq, h, d_head)` 会把 h 当 queries 轴
- **拆分顺序 `(h d)` vs `(d h)`**:权重按行连续分块 = `(h d)`(h 慢变)。拆错形状仍合法、数值错、无报错 —— 和 RoPE 的"相邻/对半配对"同类。合回来的 pattern 必须是拆的严格反写
- **复用 `Linear` 别手写 einsum**:属性名 `q_proj/k_proj/v_proj/output_proj` + 内部 `weight` → 自动拼出 `attn.q_proj.weight`,`load_state_dict` 一次装完。手写 einsum 只得到两段 key
- **`self.heads` 存成了 `d_head` 却当 `num_heads` 传给 rearrange 的 `h`** = 真 bug(拆成了 16 个 4 维头而非 4 个 16 维头),形状合法数值错。命名要诚实
- **RoPE 的 `d_k` = `d_model // num_heads`(d_head)不是 d_model**;RoPE 在拆 head **之后**、SDPA **之前**、只作用 Q/K。建 cache 时传错 d_model 会报 `subscript size 32 does not broadcast with 8`
- **`d_model // num_heads` 用 `//`**:`/` 得 float,当 rearrange kwarg / reshape 尺寸会报错
- **缩放用 `√d_head` 不是 `√d_model`**(差 `√h` 倍):从 `Q.shape[-1]` 取、且在拆 head 之后调 SDPA,天然正确

#### device 反模式

- **不要 `self.device = device` + forward 里 `x.to(self.device)`**:方向反了(应让参数跟数据走,`Linear` 权重已带 device);`device=None` 时 `x.to(None)` 靠巧合是无操作;mask 应 `device=x.device` 而非 `self.device`

#### 设计

- **无状态操作(SDPA/softmax)写成函数不写成 Module**:无 Parameter/buffer 时 Module 只是空壳,还会污染 state_dict 层级。期望的 9 个 key 里 `attn` 下只有 4 个 proj,本身就说明 SDPA/softmax 不是子模块
- **数据(Q/K/V)放 forward 局部变量,不放 `__init__`/`self`**:放 self 会导致 forward 无入参、每批新建对象、且普通张量属性不被 `.to()` 搬走
- **RoPE 开关用 `self.rope = None` 不用新类/继承**:两版本只差"拆 head 后 Q/K 过不过 RoPE"一处,抽两个类=几十行重复;RoPE 在中间,继承没法 `super().forward()` 复用。YAGNI,别为假想需求造抽象基类

#### 贯穿性洞察

- **整个 Transformer 只有 attention 跨位置混合信息**(Linear/Embedding/RMSNorm/SwiGLU/RoPE 全是逐位置独立)。所以因果性只需在 attention 一处保证,一个 causal mask 就够
- **`softmax(ΣQK) ≠ Σsoftmax(QK)`**:单头分数 = 所有 head 分数之和;因 softmax 非线性,多头 ≠ 单头。`(QKᵀ)V` 纯线性会退化成一个线性层,softmax 是 attention 唯一非线性来源,不能省
- **多头是免费的结构收益**:参数量/FLOPs 与单头几乎相同(`d_head = d_model/h`),换来 `h` 张独立注意力图;唯一代价是 scores 显存 ×h、每头表示宽度降到 d_head(太小成瓶颈,故 64/128 经验值)
- **验证"reshape 不是没用"**:同权重同输入只改 `num_heads`(1 vs 4),输出**不同** —— 证明"在哪切开、哪些项相加"这个结构假设才是起作用的东西
- **`d_k`/`d_v` 分开命名有意义**:V 不参与点积,`d_v` 可以 ≠ `d_k`(T5 就是);attention 硬约束只有两条(q/k 输出维相等、`W_o` 输出 = d_model),其余维度都自由

#### 长度类参数的三个层级

容易混成一团,实际上是三个不同层级的东西:

| 量 | 是什么 | 由谁定 | 代码里的体现 |
|---|---|---|---|
| `context_length` | **模型规格**:一次最多能处理多少 token | 超参 | `__init__` 参数,往下传 |
| `max_seq_len` | **实现细节**:RoPE 预计算了多少个位置 | 由 context_length 推出 | `cache` 的第 0 维 |
| `sequence_length` | **运行时实际长度** | 输入张量 | `x.shape[-2]` |

约束:`sequence_length ≤ context_length`,且**用到的位置下标必须 < max_seq_len**(否则 `cache[token_positions]` 索引越界)。这个作业里三者数值上相等,但职责不同。

- **mask 必须用运行时 `seq` 不是 `max_seq_len`**:否则 `(max, max)` 和分数矩阵 `(..., seq, seq)` 对不上。真实实现常见"建 `max` 大小的 buffer + forward 里切 `[:seq,:seq]`",那是为了省重复构造
- **`seq <= context_length` 值得加 assert**:否则只会得到 `cache[...]` 一个难懂的索引越界错
- **一个来源往下传**,别在每层各存一份再自己算 —— 避免两处不一致
- **RoPE 下 `context_length` 是软限制,学习式位置编码下是硬限制**:RoPE 无可学习位置参数(key 清单里只有 `token_embeddings.weight`,**没有 `position_embeddings.weight`**),改 `max_seq_len` 只是重算 cache、不动任何参数 —— 这就是长度外推/NTK scaling 可行的基础。而学习式绝对位置编码(原版 BERT/GPT-2)直接决定一张 `(context_length, d_model)` **参数表**的大小,想变长只能重训

### 十一、TransformerBlock 与 TransformerLM(组装阶段)

这两层几乎不含新算法,全是**接线**。但踩的坑和前面几节性质不同:前面是"算错了",这里是"接错了"。

#### 组装层的共同特征

- **自己不持有任何参数**:TransformerBlock 只有 `attn`/`ln1`/`ln2`/`ffn` 四个子模块,TransformerLM 只有 `token_embeddings`/`layers`/`ln_final`/`lm_head`。一旦发现自己在组装层里写 `nn.Parameter`,多半是理解错了
- **属性名 = state_dict 前缀**,递归拼接。`self.attn = MHA(...)` 且 MHA 内 `self.q_proj = Linear(...)` → `attn.q_proj.weight`。**两层属性名都对,`load_state_dict(strict=True)` 一次装完九个 key**,这是前面几节坚持命名规范的全部回报
- **先打印 key 再写 forward**:`print(list(m.state_dict().keys()))` 对着 adapter 文档里的清单逐条核对。10 秒,能省掉一整轮调试

#### pre-norm 的接线

正确形式是 `x + Sublayer(Norm(x))` —— **norm 在子层之前**,残差从 norm **之前**分叉。

踩的坑:第一个子层写对了(`ln1` 在 `attn` 前),第二个却把 `ln2` 放到了 `ffn` **后面**。原因是 forward 里 `x` 被反复重绑定,视觉上难以核对。

- **对策**:写完把每一行的语义读一遍(`residual = ?` / `x = ?`),或者干脆用不同变量名,别都叫 `x`
- 这是**会挂测试但不报错**的错误,静态阅读容易漏
- pre-norm vs post-norm 不是风格问题:post-norm 深层需要 learning-rate warmup 才能稳定,pre-norm 不需要 —— 这是现代 LLM 普遍用 pre-norm 的原因

#### `token_positions` 的责任归属

`run_transformer_block` 的签名**没有** `token_positions` 参数 → 默认情况必须能自动工作 → **MHA 必须在 `None` 时自造 `arange(seq)`**。

而 `_with_rope` 的 adapter 会显式传 `(batch, seq)` → 两条路都要通。

- **踩的坑**:一开始在 MHA 里写了 `assert token_positions is not None`,而 TransformerBlock 的 `theta`/`max_seq_len` 是必填 → `self.rope` 永远存在 → assert 必然触发
- **判断依据**:看 adapter 签名有没有那个参数,就知道"谁该负责兜底"

#### head 轴错位:被 size-1 广播掩盖的 bug ⭐

RoPE 在拆 head **之后**调用,此时:

```
Q                (batch, h, seq, d_head)     → einsum 的 ... = (batch, h, seq)
token_positions  (batch,    seq)             → R 的 ...      = (batch, seq)
                        右对齐 ↓
                 batch 撞上了 h
```

后果分两种:`batch != h` 报 `does not broadcast`;**`batch == h` 静默算错**(每个 head 拿到错误 batch 的位置编码)。

- **修法**:在 MHA 里把位置张量变成 `(batch, 1, seq)`(倒数第二维插 1)。**不要改 RoPE** —— head 轴是 MHA 引入的复杂度,该由 MHA 消化
- **统一两条路的技巧**:不管传进来几维,一律在倒数第二维插 1。`(seq,)` → `(1, seq)`(无害),`(batch, seq)` → `(batch, 1, seq)`(正是所需),不用分支
- **⚠️ 最值得记的一点:测试通过了,但没验证这个修复**。fixture 里 `token_positions` 是 `(1, 12)`,batch 维恰好是 1,size-1 广播让「插不插轴」结果相同。**"绿了"不等于"那个 bug 不存在"**
- **真正能验证的检查**:构造 `batch == num_heads`(如 4 和 4)、显式传 `(4, seq)` 位置,做 batch 独立性检查(整批喂 vs 逐样本喂再拼,必须逐位相同)。注释掉插轴那行应该挂,加回来应该过 —— **能区分对错的检查,才是有价值的检查**

#### Embedding 不是 Linear

`self.token_embeddings = Linear(d_model, vocab_size)` —— 复制粘贴 `lm_head` 那行忘了改。

| | 输入 | 操作 | 权重形状 | 构造参数顺序 |
|---|---|---|---|---|
| `Linear` | float `(..., d_in)` | 矩阵乘(收缩 d_in) | `(d_out, d_in)` | `(in, out)` |
| `Embedding` | **int** `(...)` | **查表**(索引) | `(vocab_size, d_model)` | `(vocab_size, d_model)` |

- **`Embedding` 是唯一增加维数的模块**:`(batch, seq)` 进,`(batch, seq, d_model)` 出 —— 因为它把一个整数换成一个向量
- **两者的构造参数顺序相反**(`Linear` 是 in 在前,`Embedding` 是 vocab 在前),而且 `lm_head` 和 `token_embeddings` 的权重形状**都是 `(vocab_size, d_model)`** —— 形状检查抓不住把 Embedding 写成 Linear
- **报错读法**:traceback 显示 `token_embeddings(x)` 掉进了 `Linear.forward`,这一条就定案了。**看 traceback 掉进了哪个类,比看报错信息本身更快**

#### 模型输出 logits,不输出概率

在 `lm_head` 之后多做了一次 `softmax` → 100% 元素不匹配。

**为什么 softmax 不属于模型**:

1. **契约**:snapshot 存的是 logits;`F.cross_entropy` 也接受 logits 而非概率,整个生态约定如此
2. **下游需求各不相同**:训练要 `log_softmax`、采样要 `softmax(logits/T)` + top-k/top-p、预测只要 `argmax`(softmax 严格单调,不改排序,**对 argmax 纯属浪费**)。提前归一化等于替所有下游做了决定,且丢掉了调温度的能力
3. **数值有害** ⭐:`log(softmax(x))` 里的 `exp` 会永久销毁信息。logit 比最大值低 20 时,fp16 下 `exp(-20)` 直接下溢为 0 → `log(0) = -inf` → loss `inf` → 梯度 NaN。而 `log_softmax` 用 `log p_i = (x_i - m) - log Σ e^{x_j - m}`,代数上化简掉了 exp,结果就是个普通的 `-20`
4. **梯度**:softmax+CE 融合后梯度是 `p - y` 一个减法;拆开则要走 softmax 的完整雅可比 `diag(p) - ppᵀ`

一句话:**softmax 属于损失函数或采样器,不属于模型。**

#### 诊断手法:同行内比值 ⭐

第二次用这招定位问题了(第一次是 softmax 分母没减 max)。

**规律**:`Mismatched elements: 100%` 但 ACTUAL 和 DESIRED 之间存在**简洁函数关系**时,说明主体计算是对的,只是最后套了一层(或漏了一层)变换。

步骤:

1. **看取值范围**:全正且 <1 → 概率;有正有负、量级几 → logits;行和为 1 → 确定过了 softmax
2. **算同行两元素比值**,和 DESIRED 之差取 `exp` 对比
3. 相等 → 就是多/少了一个 softmax

本次实例:`8.76e-3 / 1.4957e-5 = 585.7`,而 `e^{3.587 - (-2.786)} = 585.6` —— 精确相等,证明 embedding、所有 block、RoPE、causal mask、`ln_final`、`lm_head` **全部正确**,唯一多余的就是最后那个 softmax。

**比逐层排查快得多。**

#### 小的

- **`nn.ModuleList` 比 `nn.Sequential` 贴切**:key 完全相同(`layers.0.attn...`),但 `Sequential` 承诺"依次调用且单输入单输出",而组装层是手写循环。将来 block 需要多参数(如 `token_positions`)时 `Sequential` 就用不了
- **调试用的 `print` 不要留在库代码里**,想看 key 就在 `__main__` 里打
- **`Linear` 权重是 `(out, in)` 但构造参数是 `(in, out)`** —— 两个顺序相反,最容易错。`Linear(3, 5).weight.shape` 应是 `(5, 3)`,一行确认,以后不用怀疑
- **死字段要删**:`self.device`/`self.dtype` 在删掉 `x.to(self.device)` 之后就无人使用了,留着只会误用
- **变量名要诚实**:`scores = scaledDotProductAttention(...)` 存的是加权后的 V 不是分数;`self.heads` 存的是 `d_head` 不是头数(后者是真 bug)

#### 组装阶段的推进顺序

1. **底下全绿再往上盖**:`test_linear` / `test_embedding` / `test_rmsnorm` / `test_swiglu` / `test_multihead_self_attention{,_with_rope}` 有一个挂着,上层测试的失败原因就会混成一团
2. `print(state_dict().keys())` 对照 adapter 的 key 清单
3. `load_state_dict(strict=True)` —— **绝不设 `strict=False`**,它会静默跳过不匹配的 key,权重根本没装进去却毫无提示
4. 跑测试

### 十二、资源账本(参数量 / FLOPs 手算阶段)

这一节的坑和前面几节不同:**没有一个是报错**。手算题算错了不会有任何提示,只能靠自己设计校验手段。

#### 手算对不上时:先提公因子

FLOPs 的量级是 10¹²,盯着 12 位数看是看不出问题的。

**方法:把公因子 `2·context_length·d_model` 提出来,除掉它。** 剩下的残差是个由超参组成的小整数,肉眼就能核对。

本次实例:算出的"一个 block"是 `234,517,299,200`,除以 `3,276,800` 得 `71,569` —— **整除,说明加法没错,是多加了一项**。再做减法 `71,569 - 21,312 = 50,257` —— 正好是 `vocab_size`,于是定位到:`lm_head` 被错误地算进了 block 内部(它在 48 个 block 之外,只做一次)。

一旦提出公因子,`21,312` 还能继续拆成三项校验:

```
4·d_model        = 6,400    ← 4 个投影
2·context_length = 2,048    ← attention 两次收缩
3·d_ff           = 12,864   ← SwiGLU 三个矩阵
                   ------
                   21,312
```

**这三个数同时也是 (c) 的答案**,占比一除就出来。

#### 用外部锚点验证量级

算出 GPT-2 XL 是 1.64B 之后,和公开数字对一下:真实 GPT-2 XL 约 **1.56B**,我们略大。方向说得通(SwiGLU 三个矩阵 vs 原版两个、无 bias、RoPE 省掉了 `1024 × 1600` 的位置表)。

**落在 1.5~1.7B 就说明结构没漏项。** 量级检查抓不出"少乘一个 1600"这类错(那会差 3 个数量级,一眼就能看出),但能抓出整块模块漏掉。

#### 用代码对账(决定性)

```
sum(p.numel() for p in model.parameters())
```

必须**精确等于**手算值,一位不差。量级检查会漏掉的错误,这里全暴露。

两个要点:

- **`model.parameters()` 只返回 `nn.Parameter`,不含 buffer** → RoPE cache 自动被排除,正好对应题目要的 "trainable"。这也反证了当初把 cache 注册成 buffer 是对的
- **`with torch.device('meta'):`** 构造模型只有张量元信息、不分配存储。XL 有 6.11 GiB 权重,四个模型一起跑要 11 GiB —— 用 meta 就是 0。**而且不需要给 `TransformerLM.__init__` 加 `device` 参数**,这个上下文管理器是全局生效的

#### 经验公式 `2·N·context_length` 与它的两个偏差源

前向 FLOPs ≈ `2 · 参数量 · token 数`(训练是 `6ND`,因为反向约为前向的 2 倍)。

这个粗估在 XL 上偏高 4.7%,偏差来自两股方向相反的力:

| 来源 | 方向 |
|---|---|
| `token_embeddings` 有参数但 **0 FLOPs**(查表) | 让 `2ND` **高估** |
| attention 两次收缩有 FLOPs 但 **无参数** | 让 `2ND` **低估** |

⚠️ 注意 `token_embeddings.weight` 和 `lm_head.weight` **形状相同、参数量相同,但只有后者做矩阵乘**。scaling law 文献里的 `6ND` 通常指 **non-embedding** 参数,就是为了绕开这一点。

这个比值随模型变大而上升(small 0.878 → XL 1.047),因为 embedding 占参数量的比重在萎缩(small 里近一半,XL 里只剩 10%)。

#### 哪些操作不算矩阵乘 —— 判据

**看有没有一根轴被两个操作数共享、且被求和掉**(就是 einsum "轴名不在输出就求和"那条规则)。

- RMSNorm 的 `g` 共享 `d_model`,但 `d_model` **在输出里保留** → 逐元素,不是收缩。内部的平方和是**一个张量自己**的规约
- `probs @ V` 共享 `keys` 轴且它**不在输出** → 是收缩,**必须算**

量级对比(XL, `context_length=1024`):

| 操作 | 量级 | 数值 |
|---|---|---|
| RMSNorm | `O(context_length · d_model)` | ~1.6 M |
| softmax | `O(num_heads · context_length²)` | ~26 M |
| 一次权重投影 | `O(context_length · d_model²)` | ~2.6 **G** |

**差 1600 倍,正好是 `d_model`。** handout 只让数矩阵乘,不是因为其他操作免费,而是因为它们小 3 个数量级。

⚠️ RoPE 也属于这一类:cache 在 `__init__` 预算,但那个旋转**每次前向都在跑**,只是量级 `O(context_length · d_model)` 可忽略。写"前向不参与计算"是不准确的,应该写"参与了但量级可忽略"。

#### attention 两次收缩的特殊性

`QKᵀ` 和 `probs @ V` 这一对**两边都是激活,没有权重参与**。三个后果:

1. 在参数量里**根本不出现** → 数 FLOPs 时最容易漏
2. 量级是 `O(context_length² · d_model)`,随 `context_length` **平方**增长,而所有权重投影只是线性
3. 漏了它,(e) 会答成"FLOPs 线性增长" —— 直接错

实测:`context_length` 从 1024 涨到 16384(16 倍),FLOPs 涨 **38 倍**;`2·context_length` 这一项在括号里从 9.6% 变成 63%,从最小项变成最大项。**这就是 FlashAttention 和各种长上下文方案存在的直接原因。**

#### 多头不影响参数量,也不影响 FLOPs

- **参数量**:q/k/v 的权重本是每 head 一份 `d_head × d_model`,`num_heads` 份竖着摞起来正好是 `d_model × d_model`
- **FLOPs**:每 head 的收缩是 `2·context_length²·d_head`,`num_heads` 份相加得 `2·context_length²·num_heads·d_head` = `2·context_length²·d_model`

**拆 head 本身零 FLOPs** —— `rearrange` 只改 shape / stride,不动 storage。所以四个投影按 `d_model × d_model` 整块数就行,不必按 head 拆。

`num_heads` 在整道题里**只在一处真正起作用**:attention 那两次收缩要乘 head 数(然后又被约掉)。

#### 单位:三对容易混的

| 混淆 | 区别 | 后果 |
|---|---|---|
| `GiB` vs `GB` | `1024³` vs `1000³`,差 **7%** | 显卡标称的 "80GB" 是十进制值,混用会偏 7%,可能刚好跨过一个整数 batch |
| `FLOPs` vs `FLOPS` | 操作**次数** vs 每秒操作数(**吞吐率**) | `时间 = 总FLOPs ÷ (硬件FLOP/s × MFU)`,分子分母混了得到量纲错误的答案,而且两个数都"以 T 开头",很难发现 |
| "加载" vs "训练" | 权重一份 vs 权重+梯度+AdamW 的 m/v+激活 | 6.11 GiB 只是权重。24GB 的 4090 勉强推理,但 AdamW 全参微调装不下(≈26 GiB 还没算激活)—— 这就是 LoRA / 量化 / ZeRO 存在的直接原因 |

**两个单位都写出来**最省事。

#### 张量的三层轴:`batch` 就是"有几条 sequence"

`(batch, context_length, d_model)` = `(1, 1024, 1600)`,主干上**永远 3 轴**。

一句话记法:**每个 token 一个向量,每条序列一串 token,每个 batch 一叠序列。**

⚠️ "1 个 batch、1 个 sequence"是**同一件事说了两遍** —— `batch size` 的定义就是"有几条 sequence",不是两个层级。把它当两层就会写出 `(1, 1, context_length, d_model)` 这种多一根轴的形状。

唯一的第 4 轴只在 MHA 内部拆完 head 后出现,长度是 `num_heads` 而不是 1,`output_proj` 之后就消失。

另外:**权重和激活必须分开叫**。`token_embeddings.weight` 是 `(vocab_size, d_model)`、全模型一份;embedding 的**输出**是 `(batch, context_length, d_model)`、每次前向一份。两个都叫"embedding 矩阵"必然出错 —— 和之前 `q_proj` vs `Q` 是同一个坑。

同理,attention 里连着三个中间张量都叫"V 矩阵"也是错的:`probs @ V` 的输出、`output_proj` 的输出、SwiGLU 的输入是**三个不同的张量**。

#### `context_length` 与参数量无关

`context_length` 是**输入的属性,不是权重的属性**。它绝不该出现在参数量公式里。

推论 —— 写 `count_forward_flops(model, ???)` 时的签名设计:`vocab_size` / `d_model` / `d_ff` 能从权重 `.shape` 读出,`num_layers` 能从 `ModuleList` 的 `len()` 读出,但 **`context_length` 只能作为参数传进来**。(从 RoPE cache 长度读也不对 —— 那是"最大支持长度",不是"这次实际输入长度"。)

FLOPs 对 `batch` 严格线性,参数量与 `batch` 完全无关。

#### 取整到某个粒度:`ceil` 该套在哪一层

```python
# 错:ceil 作用在乘积上,/64*64 互相抵消,等于什么都没做
d_ff = int(math.ceil(8/3 * d_model) / 64 * 64)        # 1600 -> 4267

# 对:ceil 作用在"是 64 的几倍"上
d_ff = int(math.ceil(8/3 * d_model / 64) * 64)        # 1600 -> 4288
```

要向上取整的对象不是 `8/3 · d_model` 这个数,而是**它是 64 的几倍**。

**必须加的断言:结果能被 64 整除。** 这类错误的可怕之处在于数字看起来"差不多对"(4267 vs 4288 只差 0.5%),肉眼极难发现,但一个 `% 64 == 0` 就能拦住。

另外 `d_ff` 有两处独立来源(`SwiGLU` 的默认值、脚本里手写的 config),**要互相对账**。对不上说明其中一个错了。

顺带一个规则歧义:handout 说 "nearest multiple of 64",而 `ceil` 和 "nearest" 在 `d_model = 1600` 上恰好都给 4288,在 `d_model = 1280` 上却不同(3456 vs 3392)。**这种情况要在 writeup 里写明自己用的是哪个规则**,否则批改的人分不清是有意选择还是算错。

#### Python 包机制:`attempted relative import with no known parent package`

直接执行包内文件时,Python 把 `__name__` 设为 `"__main__"`、`__package__` 设为 `None`。相对导入的 `.` 是**相对 `__package__` 解析**的,没有父包 → 报错。

**一个文件是"包的一部分"还是"脚本",取决于怎么启动它,而不是它放在哪个目录。** 同一个文件被 `import` 时相对导入正常,被直接执行就炸。

| 方案 | 代价 |
|---|---|
| 改绝对导入 `from cs336_basics.xxx import` | 无。`import` 和直接执行都能用,且和 `tests/adapters.py` 的写法一致 |
| `python -m cs336_basics.calculations` | 能用,但绑定了启动方式和工作目录 |

用 `uv run python ...` 而不是手敲 `.venv/bin/python` 全路径 —— `uv run` 会自己确保依赖同步。

#### 报告脚本的职责划分

**计算函数只返回数字,绝不 `print`;打印函数只负责格式化,绝不计算。**

理由很实在:数字可以写 `assert`,`print` 不能。一旦在 `count_parameters` 里塞了 `print`,就没法在测试里干净地用它。

其他几条:

- **千分位 `f"{n:,}"`** —— 12 位数不加千分位读不出来。前面那个多算的 `lm_head` 就是因为盯着裸数字看不出来
- **占比列加一行合计**,应当正好 100% —— 这一行本身就是断言,能查出分项漏了一项
- **中文在终端占两格而 `len()` 只算一格**,表格对齐要用 `unicodedata.east_asian_width` 判宽
- **不要硬编码超参**,全部从 `model` 读 —— 这样对比 small/medium/large 只是换 config,不用手算三遍
- **表格函数接受一组 config 而不是一个** —— (d) 是一次调用输出四行,(e) 是同一个 config 换 `context_length` 跑两次

⚠️ **脚本是验算,不是答案来源。** writeup 里该留的是推导,不是脚本输出。如果脚本和手算不一致,**先怀疑脚本** —— 手算已经用提公因子的方法独立验过一遍了,脚本是新写的。

### 十三、损失函数与训练工具(cross entropy / 梯度裁剪)

#### `cross_entropy_loss`:测试自己压平了输入

测试在调用前就做了 `inputs.view(-1, inputs.size(-1))` 和 `targets.view(-1)`,所以**只要二维路径对就能全绿**。这个函数改了三次才真正正确,三次 `pytest -k cross_entropy` 都是绿的。

自建的形状矩阵才把问题逼出来:

| 输入 | 第二版 | 第三版 |
|---|---|---|
| 2D `(8,5)` | OK | OK |
| 3D `(2,4,5)` | IndexError | OK |
| 4D `(2,3,4,5)` | IndexError | OK |
| 1D `(5,)` | IndexError | OK |
| 非连续切片 `[:, :-1]` | RuntimeError | OK |
| 数值溢出 `1000×` | OK | OK |

**签名承诺了 `...`(任意前导维),测试只喂了一种维数 → 覆盖缺口正好落在承诺最宽的地方。**

#### 把 `inputs` 的 reshape 模式照抄到 `targets`

两者语义不同,而 `reshape` 只检查元素总数、不检查语义:

| | 形状 | 最后一维是什么 |
|---|---|---|
| `inputs` | `... seq_len vocab_size` | vocab 维,**要保留** |
| `targets` | `... seq_len` | 就是 seq 本身,**没有 vocab 维** |

`targets.view(-1, targets.size(-1))` 不报错,只静默改形:`(8,)→(1,8)`、`(2,4)→(2,4)` **完全没展平**、`(2,3,4)→(6,4)`。而 `(8,)→(1,8)` 因为元素个数和求和结果都没变,**又一次侥幸通过了测试**。

⚠️ 附带后果:`(5,)` 的 targets 是 0 维标量,`size(-1)` 直接 IndexError —— 这一版把上一版本来能跑的 case 弄坏了。

#### `view` vs `reshape`:对外函数一律用 `reshape`

训练循环里最常见的错位写法 `logits[:, :-1]` 切片后 `stride[0]` 仍是 `T*V` 而连续性要求 `(T-1)*V` → **不连续** → `view` 直接抛 `RuntimeError: view size is not compatible with input tensor's size and stride`。

| | 连续时 | 不连续时 |
|---|---|---|
| `view` | 返回 view,免费 | **抛异常** |
| `reshape` | 完全相同 | 自动拷贝,成功 |

判据和 `Iterable` 那条一样:**你无法控制调用方传进来的 stride**。`view` 留给自己确信连续、且希望不连续时报错的内部代码。

#### 用乘法改形状是最贵的做法

`inputs * torch.ones([1,1,1])` 干了两件坏事:分配一份和 `inputs` 等大的新张量,再往计算图里加一个 `mul` 节点(**乘法保存两个操作数**)又留一份。

而 `inputs` 的形状是 `batch · seq · vocab_size` —— **整个模型里最大的那个张量**。GPT-2 XL、`context_length=1024`、`batch=1` 时它是 206 MB,这一行让它变成 412 MB。`reshape` 是 0。

#### "补维"和"归约"的区别

乘 `(1,1,1)` 只保证**至少 3 维**,不保证恰好 3 维:4D 输入完全不受影响,`shape[-3]` 拿到的是中间那一维,分母漏掉最外层因子 → 4D 实测偏差 **11.5 倍**。

`reshape(-1, size(-1))` 是**归约**:

| 输入 | `shape[:-1]` | 乘积 | 展平后 |
|---|---|---|---|
| `(2,3,4,5)` | `(2,3,4)` | 24 | `(24, 5)` |
| `(8,5)` | `(8,)` | 8 | `(8, 5)` |
| `(5,)` | `()` | **1**(空乘积) | `(1, 5)` |

**归约没有上下界问题,补维两头都有。** 空乘积等于 1 这一条让 1D 也免特判。

#### 展平之后名字会骗人

`inputs.shape[-2]` 展平后拿到的是**总 token 数**(`batch × seq`),不再是序列长度。值对、名错 —— 下次读代码会照着名字误用。而且既然已经归约成一维,直接 `.mean()` 就不需要手工分母,少一处出错机会。

同理:高级索引 `inputs[..., arange(n), targets]` 在存在剩余前导轴时会**额外插入一根轴**(3D 实测得到 `(2,2,4)` 而不是 `(2,4)`),只在二维下侥幸正确。展平后 `...` 也变成匹配空,留着会让读者以为还支持前导维。

#### 唯一还会静默给错数的输入组合

`inputs (5,)` + `targets (5,)`(调用方误以为传的是 5 个位置)→ 一路走通,输出一个没有意义的数,**不报错**。

入口断言 `inputs.shape[:-1] == targets.shape` 一行拦住。展平之后这个信息就永久丢失了,所以断言必须在展平**之前**。

#### `gradient_clipping`:范数是梯度的,不是参数的

函数签名给的是 `parameters`,要裁剪的是 `p.grad`。参数本身从头到尾不动。

`‖concat(v₁..vₙ)‖₂ = √(Σᵢ‖vᵢ‖₂²)`,因为 L2 的定义就是所有元素平方和开根,拼接不改变元素集合 → 逐张量算平方范数、加标量、**最后开一次根**,不需要 `torch.cat` 出一个 1.6B 的巨型向量。

⚠️ **`sqrt` 不能开在循环里**:范数不可加,`√a + √b ≠ √(a+b)`,只有**平方范数**可加。

#### `Iterable` + 生成器只能遍历一次

裁剪需要两遍(算全局范数 → 算系数 → 逐个乘同一系数),而 `model.parameters()` 是生成器,第一个 `for` 就把它耗尽了,第二个 `for` **一次都不执行**。

| 调用方传什么 | 结果 |
|---|---|
| list(测试常这么写) | ✓ 正常 |
| `model.parameters()`(真实训练) | ✗ **裁剪静默失效** |

不报错。表现是 loss 偶尔 spike 到 NaN,你会去怀疑学习率、数据、初始化,绕一大圈才想到裁剪根本没生效。

> **只要一个函数需要对同一个 `Iterable` 参数遍历两次以上,进函数就先物化。** 反之只遍历一次就不要物化 —— 那会剥夺调用方用惰性生成器省内存的能力。判据就是遍历次数。

PyTorch 官方 `clip_grad_norm_` 第一行做的就是这件事。另外两个循环的 `p.grad is not None` 过滤条件必须一致,只在第一个里写会在第二个里撞 `AttributeError`。

#### `torch.zeros(0)` 是空张量,不是标量零

`torch.zeros` 的参数是**形状**,不是维数:

| 写法 | 形状 | 元素个数 |
|---|---|---|
| `torch.zeros(0)` | `(0,)` | **0** |
| `torch.zeros(1)` | `(1,)` | 1 |
| `torch.zeros(())` | `()` | 1(真正的 0 维标量) |

空张量最烦人的地方是**它对几乎所有运算都合法,只是结果继续是空的**:`empty += scalar` 累加静默失效 → `sqrt(empty)` 空 → `clamp(scalar/empty)` 空 → 一直到 `(5,5) * (0,)` 才广播失败。**报错行离病根隔了 10 行。**

一句 `numel() == 1` 的断言就能让它在源头暴露。和 `d_ff` 的 `% 64 == 0` 同一个套路:**在最接近错误源头的地方检查最容易被搞错的那个属性。**

#### 凭空造张量必须指定 device

`ones` / `zeros` / `arange` / `eye` / `full` / `rand` 默认建在 CPU 上。同一个坑在本阶段出现了**三次**(`arange`、`zeros`、`ones`),因为一直在 CPU 上跑测试所以全绿,上 GPU 会一次性炸出一片 `Expected all tensors to be on the same device`。

要么显式传 `device=`,要么从已有张量派生(`torch.ones_like` / `x.new_ones`)。

#### 别在梯度循环里 `.item()`

`.item()` 强制一次 GPU→CPU 同步。遍历上百个参数张量就是每步上百次同步,把流水线打得七零八落 —— 梯度裁剪本身很便宜,这么写能让它变成训练循环里的显著开销。

让累加量始终是一个 **0 维 tensor 留在 device 上**:`torch.sqrt`、加减乘除、`p.grad.mul_(scale)` 对 0 维 tensor 都正常工作,全程不需要变成 Python float。

`if norm > M` 这个分支也会同步,而它可以被算术吸收:`c = min(1, M/(‖g‖+ε))`,范数小于阈值时 `M/‖g‖ > 1` 被截成 1,乘上去什么都不变。用 `torch.clamp(..., max=1.0)` 保持在 device 上,零同步。

> **能用算术表达的条件,别用 Python 分支。**

⚠️ 但要分清"条件缩放"和"归一化":只在**超过**阈值时缩放才是裁剪,无条件缩放到阈值是另一个算法。

另外 `p.grad = p.grad * c` 会为每个参数**新分配**一份等大张量(XL 上又是一个 6.5 GB),用原地乘法。

#### 裁剪的三条自查

1. 范数**超**阈值:裁剪后重算应**恰好等于**阈值(差在 `eps` 量级)
2. 范数**未超**:梯度应**逐元素完全不变** —— 用 `torch.equal`,不是 `allclose`
3. **方向保持**:任取两个梯度元素,比值不变 —— 这条能抓到"逐张量各自缩放"的错误实现

第 2 条在生成器被耗尽时会"通过"而第 1 条会失败,两条一起看就能定位。

### 十四、数据加载(`get_batch`)

#### off-by-one 的落点

`y` 的最后一个元素在 `dataset` 里的下标是 `start + context_length`,必须 `≤ len-1` → 起点最大值 `len - ctx - 1`,可能的起点**个数**是 `len - ctx`。

而 `torch.randint` / `np.random.integers` 的 `high` 都是**开区间**。

> **巧合值得记住:开区间上界恰好等于"从 0 开始数的可取值个数"。** 所以 `high` 直接填个数,不容易错。

填成 `len - ctx - 1` 会让最大起点少一个,永远采不到最后一个窗口 —— 测试的 `max(starting_indices) == num_possible - 1` 专门抓这个。

#### 顺序决定了能不能用 memmap

`torch.tensor(dataset, device=device)` 把**整个数据集**读进内存再拷一份到显存。真实数据是 `np.memmap`、几亿 token,而每步只需要 `batch_size × context_length` 个元素。

正确顺序:

```
1. numpy 侧采起点 → 广播出下标矩阵 → 索引 dataset     (只碰需要的元素)
2. 转 torch,显式 int64
3. 一次性 .to(device)
```

实测对 memmap 做 fancy indexing 返回的是**普通 ndarray**,只包含要的那 `B×L` 个元素 —— 磁盘上只有相关的页被读进来。这就是"先在 numpy 侧索引"的全部价值。

⚠️ 下标张量必须留在 CPU:CUDA 张量不能隐式转 numpy,会报 `can't convert cuda:0 device type tensor to numpy`。

#### 广播升维与 fancy indexing

`(B,1) + (L,)` → `(B,L)`。忘了给起点补轴时 `(B,) + (L,)` 报 `ValueError`(友好),但 **`batch_size == context_length` 时它是合法的逐元素相加,静默给出 `(B,)`**。

一维数组的 fancy indexing 规则:**输出形状 = 下标数组的形状**。所以下标矩阵是 `(B,L)`,取出来直接就是 `(B,L)`,不用 reshape。

`torch.gather` 不是这里的工具:它要求下标与源**维数相同**,一维源要产二维输出得先 expand,反而更绕。

#### uint16 不报错,延迟到 embedding 才炸

实测 torch 2.11:

```
torch.from_numpy(uint16数组)          -> torch.uint16      不报错
torch.tensor(uint16数组)              -> torch.uint16      也不报错
embedding(uint16 张量)                -> RuntimeError: Expected tensor for argument #1 'indices' ...
```

转换那一步**静默通过**,直到喂进 `Embedding` 才炸。而 `get_batch` 的测试用 `arange(100)`(本来就是 int64),完全暴露不出来 —— 这个 bug 会一路潜伏到跑训练循环。转 torch 时显式指定 int64,不要依赖默认推断。

#### `y == x + 1` 是测试构造出的假象

测试里 `dataset = np.arange(0, 100)`,所以"在数据流里往后挪一个位置"和"数值加一"**碰巧等价**。一个真的去写 `y = x + 1` 的实现能完整通过这个测试,然后在真实数据上产生完全错误的标签。

自查必须用**非 arange** 的数据集,断言 `y[:, :-1] == x[:, 1:]` —— 这是"偏移一位"的直接表达,与数据内容无关。实测同时确认 `x + 1 == y` 为 **False**,才算排除了假实现。

#### 死代码与 `device` 的隐式耦合

那行没用的 `torch.tensor(dataset, device=device)` 一度是**唯一碰到 `device` 的地方**,`cuda:99` 必须抛异常那条断言全靠它满足。

所以顺序很重要:**先补上真正的 `.to(device)`,再删死代码。** 反过来做会让那条断言挂掉,然后你会去怀疑别的地方。

#### 测试为什么用 μ ± 5σ:多重比较

对某个固定起点,被抽中的次数服从 `Binomial(N, 1/K)`,`N = num_iters × batch_size = 32000`,`K = len - ctx = 93`。于是 `μ = Np = 344.09`,`σ = √(Np(1-p)) = 18.45`。

| z | 单个下标越界概率 | 93 个全不越界 | **测试误报率** |
|---|---|---|---|
| 3 | 2.70e-3 | 77.8% | **22.2%** |
| 4 | 6.33e-5 | 99.41% | 0.59% |
| 5 | 5.73e-7 | 99.9947% | 0.0053% |

**单个检查的 3σ 看起来很稳,重复 93 次之后误报率变成 22%。** 整体误报率 ≈ `K ×` 单次尾概率,要把整体控制在 `α`,单次就得控制在 `α/K`。

严格说 93 个计数服从多项分布(和恒为 `N`、彼此负相关),测试用的是"边缘分布是二项"+union bound,是保守的标准做法。

⚠️ 二项模型本身假设每次抽样独立,也就是**有放回**。实现成"一个 batch 内起点不重复"虽然大概也能过 5σ,但模型上不匹配。

写随机化测试的通用套路:把"随机性是否正确"翻译成一个可数的统计量 → 求它在零假设下的分布 → 边界按 `α / 检查次数` 定,宁宽勿窄。

### 十五、工具链

#### Jupyter / IPython 里 `torch.` 补全为空

实测对 `torch.ra`:

```
use_jedi=True   ->  []                                      0 项
use_jedi=False  ->  ['.rand', '.randint', '.randn', ...]    12 项
```

而 `'rand' in dir(torch)` 是 `True`、`len(dir(torch))` 是 1488。**jedi 不是漏了几个,是整个放弃了。**

原因:`torch.rand` 是 `builtin_function_or_method`,来自 C 扩展 `torch._C._VariableFunctions`,由 `torch/__init__.py` 用循环动态塞进命名空间。jedi 是纯静态分析器,既看不到 C 扩展也无法执行那个循环。numpy 2.x 同理(模块级 `__getattr__` 惰性加载)。

| 场景 | 谁在补全 | torch |
|---|---|---|
| 普通 `.py` 文件 | Pylance(读 `.pyi` stub) | ✓ |
| 交互窗口,jedi 开 | jedi(静态,忽略 stub) | ✗ 空 |
| 交互窗口,jedi 关 | IPython 运行时 `dir()` | ✓ |

"在 `.py` 里有提示、到交互窗口就没了"就是这个问题的典型症状。

临时:`%config Completer.use_jedi = False`。永久:`<IPYTHONDIR>/profile_default/ipython_config.py` 里写 `c.Completer.use_jedi = False`。

⚠️ 实测过一个反直觉的点:`IPKernelApp.config_file_name` 是 `ipython_kernel_config.py`,但 IPython 的应用基类会先把 `ipython_config.py` 作为**所有 IPython 应用的公共基础配置**加载 → **一个文件同时覆盖终端 IPython 和 Jupyter kernel**,不需要建两个。配置搜索路径还包含 `<venv>/etc/ipython`,放那里可做成只对本项目生效。

另外:numpy 2.0 真的删了 `np.float` / `np.int` / `np.NaN` / `np.alltrue`,找这些时没提示是**正确行为**。判据是直接求值看是否 `AttributeError`。

#### pytest 调试

| 命令 | 效果 |
|---|---|
| `--pdb` | 失败/异常时进入(事后调试,**不用改代码**) |
| `--pdb -x --lf` | 只重跑上次失败的,第一个失败就停并进调试器 |
| `breakpoint()` + 必要时 `-s` | 在指定位置停。`-s` 即 `--capture=no` |
| `--trace` | 每个测试第一行就停 |
| `PYTHONBREAKPOINT=0` | 临时禁用所有 `breakpoint()`,不用删代码 |

pdb 里 `interact` 进完整 REPL,随手敲 `p.grad.shape`、`torch.sqrt(acc)` 比一条条 `p` 舒服。

调试张量时**先看元信息再看数值** —— 本阶段的 bug(device 写死、`(0,)` 空张量、uint16、ndarray vs Tensor)全部在 `shape/dtype/device/type` 这一层,看数值反而绕远。

⚠️ 用 `len(list(生成器))` 诊断生成器是否被耗尽,**这个操作本身会耗尽它**。观察改变了被观察对象,调试惰性对象时是常态。

顺带:pytest 会重写 `assert` 并展开两边的值,`assert_allclose` 还会打出不匹配元素数和最大偏差。**先认真读失败输出**,调试器留给"输出看不出原因"的情况。

### 十六、贯穿本阶段的一条主线

`§3`–`§5` 一共踩了五个"测试绿但实现错"的坑,共性完全一致:

| 函数 | 测试用的输入 | 真实调用的输入 | 缺口 |
|---|---|---|---|
| `cross_entropy_loss` | 自己压平成 2D | 任意前导维、非连续切片 | 形状 |
| `gradient_clipping` | list | `model.parameters()` 生成器 | 类型 |
| `gradient_clipping` | CPU | GPU | device |
| `get_batch` | `arange(100)` | 任意 token 序列 | **数据内容让两种语义等价** |
| `get_batch` | int64 | uint16 memmap | dtype |

> **签名/契约允许的输入范围 > 测试用到的范围,而缺口正好落在承诺最宽的地方。**

因为测试是为了"能跑"而写的,作者会挑最省事的构造方式 —— 而最省事的构造方式往往恰好让某个语义区分**退化**(压平消掉了前导维、`arange` 让位移和加一等价、list 让生成器问题消失)。

对策不是多读几遍代码,而是写完一个对外函数就花两分钟列**形状 / 类型 / device / dtype 矩阵**逐项对账。本阶段每一个 bug 都是这么找到的,而且都是在 pytest 全绿的状态下找到的。

⚠️ 反过来也要警惕:这五个坑里有三个"侥幸通过"是因为**错误恰好不改变元素个数或求和结果**(`(8,)→(1,8)`、`view` 套壳、逐元素相加)。**形状错了但标量结果对**是最难发现的一档 —— 所以断言要断在**形状**上,不要只断在最终数值上。
