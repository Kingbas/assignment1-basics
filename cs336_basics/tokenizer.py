

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
import json
import regex as re


class Tokenizer(ABC):
    """抽象基类：定义 tokenizer 的最小接口，不持有任何状态。"""

    @classmethod
    @abstractmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens: list[str] | None = None) -> "Tokenizer":
        """从序列化的 vocab/merges 文件构造 tokenizer（工厂方法）。"""
        raise NotImplementedError
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        raise NotImplementedError
    
    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

class BPETokenizer(Tokenizer):
    """基于字节级 BPE 的 tokenizer。

    vocab: token ID -> token 字节序列
    merges: 按训练时产生顺序排列的合并规则列表
    special_tokens: 需要原样保留、不参与 BPE 合并的特殊 token
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        # 优先最长匹配
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=lambda token: (len(token), token), reverse=True)
        # 若 special_tokens 不在 vocab 中，追加到 vocab 末尾
        for token in self.special_tokens:
            # 踩坑：注意这里要encode
            if token.encode() not in vocab.values():
                self.vocab[max(self.vocab.keys()) + 1] = token.encode()
        # encoding时需要用token反查int
        self.vocab_encoding = {value: key for key, value in self.vocab.items()}
        # 初始化regex pattern
        self.gpt2_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 踩坑，如果没有special token的话就会是一个空的捕获组，因此
        # print(repr(self.special_token_pattern))   # 期待看到 '()'
        # print(re.split(self.special_token_pattern, 'the cat')[:8])   # 期待看到被碎成单字符
        self.special_token_pattern = '(' + '|'.join(re.escape(special_token) for special_token in self.special_tokens) + ')' if special_tokens else None

        # 构建 merge 优先级表 (bytes, bytes) -> rank（encode 时按 rank 选合并对）
        self.merges_rank = {}
        for i in range(len(self.merges)):
            self.merges_rank[self.merges[i]] = i


    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        # 读取并反序列化 vocab/merges（格式与 trainer 的输出对齐），再调用 cls(...)
        with open(vocab_filepath) as f:
            vocab = json.load(f)
        merges = []
        with open(merges_filepath) as f:
            while True:
                line = f.readline().strip().split(' ')
                if len(line) < 2:
                    break
                merges.append((line[0].encode(), line[1].encode()))
                
        return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


    def encode(self, text: str) -> list[int]:
        encoded = []
        # 1) 按 special_tokens 切分文本（注意长 token 优先，避免重叠 token 误切）
        # 因为encode阶段特殊词也算一个词，不同于训练阶段，这里要将其捕获
        if self.special_token_pattern:
            texts = re.split(self.special_token_pattern, text)
        else:
            # 踩坑，这里texts要[text]
            texts = [text]
        # 2) 非特殊段用 GPT-2 PAT 预分词
        for text in texts:
            # 如果遇到了special tokens则跳过合并
            if text in self.special_tokens:
                encoded += [self.vocab_encoding[text.encode()]]
                continue
            # 获取pre-tokens
            tokens = re.findall(self.gpt2_pattern, text)
            for token in tokens:
                token_bytes = list(bytes([x]) for x in token.encode('UTF-8'))
                # 对于每一个相邻的byte pair在merges中找是否有对应的merge
                while True:
                    i = 0
                    pair_start = -1
                    min_rank = len(self.merges) + 1
                    # 遍历一遍找rank最小的match
                    while i < len(token_bytes) - 1:
                        pair = (token_bytes[i], token_bytes[i + 1])
                        if pair in self.merges_rank:
                            if self.merges_rank[pair] < min_rank:
                                pair_start = i
                                min_rank = self.merges_rank[pair]
                        i += 1
                    # 合并完成跳出
                    if pair_start == -1:
                        break
                    token_bytes[pair_start] = token_bytes[pair_start] + token_bytes[pair_start + 1]
                    token_bytes.pop(pair_start+1)

                # 合并完成后将该token的encoding送进encoded
                token_encoded = []
                for word in token_bytes:
                    token_encoded.append(self.vocab_encoding[word])
                encoded += token_encoded
        return encoded


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 惰性逐块 encode 并 yield，不得一次性读入全部内容（内存受限测试会检查）
        for iter in iterable:
            encoded = self.encode(iter)
            yield from encoded


    def decode(self, ids: list[int]) -> str:
        # 逐个查 vocab 取 bytes -> 全部拼接 -> 最后一次性 decode(errors="replace")
        decoded = []
        for id in ids:
            decoded.append(self.vocab[id])
        return b''.join(decoded).decode(errors='replace')
        
