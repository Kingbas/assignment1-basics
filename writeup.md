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

**(e) 上下文长度增至 16,384 后,FLOPs 如何变化?**

答:

---

## 4. Training a Transformer LM

### Problem (cross_entropy): Implement cross-entropy (1 point) 💻

- [ ] `uv run pytest -k test_cross_entropy` 通过

### Problem (learning_rate_tuning): Tuning the learning rate (1 point) 📝

**SGD toy example 中不同学习率下 loss 的行为?**

答:

### Problem (adamw): Implement AdamW (2 points) 💻

- [ ] `uv run pytest -k test_adamw` 通过

### Problem (adamw_accounting): Resource accounting for AdamW (2 points) 📝

**(a) 峰值内存的表达式(参数/梯度/优化器状态/激活)?**

答:

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
