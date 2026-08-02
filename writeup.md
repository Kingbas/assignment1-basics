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

**(b) 前向传播需要的矩阵乘及对应 FLOPs?**

答:

**(c) 哪些部分占 FLOPs 最多?**

答:

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
- 尺寸走 kwargs(`pair=2`),拆维**必须**给尺寸且**不接受 `-1``
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
