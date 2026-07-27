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
- 实现位置:
- 备注(优化思路 / 踩坑记录):

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
