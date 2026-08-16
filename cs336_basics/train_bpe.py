
import os
import json
import regex as re
import psutil
import resource
from concurrent.futures import ProcessPoolExecutor

import time
from contextlib import contextmanager

from cs336_basics.common import find_chunk_boundaries, gpt2_bytes_to_unicode

@contextmanager
def timed(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - t0:.3f}s rss:{psutil.Process().memory_info().rss/1024/1024:.3f}MB")
"""

    首先要对corpus进行pre_tokenization
    利用
    BPE算法是初始化词汇表为ASCII码 外加一个特殊字符 <|endoftext|>
    
"""

# regex pattern used in gpt-2 pre-tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

@timed('bpe init vocab')
def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    i = 0
    for special_token in special_tokens:
        vocab[i] = special_token.encode('UTF-8')
        i = i + 1
    for j in range(256):
        vocab[i] = bytes([j])
        i = i + 1
    return vocab


@timed('bpe pre-tokenizing')
def bpe_tokenizer_worker(file_path: str, special_tokens: list[str], offset: tuple[int, int]):
    special_token_pattern = '|'.join(re.escape(special_token) for special_token in special_tokens)
    token_count: dict[tuple[bytes], int] = {}
    with open(file_path, 'rb') as f:
        f.seek(offset[0])
        with timed(f'bpe tokenizer: prepare chunk {offset[0]/1024/1024:.3f}MB:{offset[1]/1024/1024:.3f}MB'):
            texts = f.read(offset[1] - offset[0]).decode("utf-8", errors="ignore")
            texts = re.split(special_token_pattern, texts)
        with timed('bpe tokenizer: process chunk'):
            for text in texts:
                tokens = re.findall(PAT, text)
                for token in tokens:
                    # 初版实现
                    # key = tuple(bytes([x]) for x in token.encode('UTF-8'))
                    # pretrained_tokenization[key] = pretrained_tokenization.get(key, 0) + 1
                    token_count[token] = token_count.get(token, 0) + 1
    # 交由父进程进行
    # token_count = {tuple(bytes([x]) for x in k.encode('UTF-8')):v for k, v in token_count.items()}
    return token_count


def bpe_tokenizer_merger(vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        pretrained_tokenization: dict[tuple[bytes, ...], int],
        pair_token: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
        pair_count: dict[tuple[bytes, bytes], int]):
    # 如果没得合并了
    if len(pair_count) == 0:
        return
    # 找到频次最高的pairs，对pairs进行字典序排序，将字典序最大者添加进vocab
    best_pair = max(pair_count, key=lambda p: (pair_count[p], p))
    # 更新vocab和merges
    merged_pair = b''.join(best_pair)
    vocab[len(vocab)] = merged_pair
    merges.append(best_pair)

    # 对于这个best_pair，要更新
    # 更新pair_count
    for token in pair_token[best_pair].copy():
        # 词频
        freq = pretrained_tokenization[token]
        # 构造出merge后的token
        i = 0
        merging: list[bytes] = []
        while i < len(token) - 1:
            pair = (token[i], token[i+1])
            if pair == best_pair:
                merging.append(merged_pair)
                i += 1
            else:
                merging.append(token[i])
            i += 1
        if i == len(token) - 1:
            merging.append(token[i])
        merged_token: tuple[bytes, ...] = tuple(merging)

        assert b''.join(merged_token) == b''.join(token)
        assert len(merged_token) < len(token)

        # 更新pretrained_tokenization
        # token变成merged_token
        pretrained_tokenization[merged_token] = pretrained_tokenization[token]
        del pretrained_tokenization[token]

        # 更新pair_token 更新pair_count
        # best_pair将不复存在
        # 删除所有原token中pair的键值，将merged_token中所有pair的键值换为pair related_tokens
        # 把原token中每个pair的计数减去词频
        # 把merged_token中每个pair加上词频
        i = 0
        while i < len(token) - 1:
            pair = (token[i], token[i+1])
            pair_count[pair] -= freq
            # 坑：具体例子：token = (a, a, a)，pair (a,a) 在 i=0 和 i=1 各出现一次。
            # i=0：pair_token[(a,a)].remove(token) → 移除成功
            # i=1：pair_token[(a,a)].remove(token) → KeyError，它已经不在里面了
            # 因此要用discard
            pair_token[pair].discard(token)
            if pair_count[pair] == 0:
                del pair_count[pair]
                del pair_token[pair]
            i += 1
        i = 0
        while i < len(merged_token) - 1:
            pair = (merged_token[i], merged_token[i+1])
            pair_count[pair] = pair_count.get(pair, 0) + freq
            pair_token[pair] = pair_token.get(pair, set())
            pair_token[pair].add(merged_token)
            i += 1
    
    assert best_pair not in pair_token
    assert best_pair not in pair_count

def bpe_serializer(vocab: dict[int, bytes], merges, out_path='data'):
    gpt2_encoder = gpt2_bytes_to_unicode()
    vocab_out = { ''.join([gpt2_encoder[v] for v in val]): key for key, val in vocab.items()}
    assert len(vocab_out) == len(vocab)
    merges_out = '\n'.join([ ''.join([gpt2_encoder[b] for b in merge[0]]) + ' ' + ''.join([gpt2_encoder[b] for b in merge[1]]) for merge in merges])
    with open(os.path.join(out_path, 'vocab.json'), 'w', encoding='utf-8') as f:
        print(f'dumping vocab.json into {out_path}')
        json.dump(vocab_out, f, ensure_ascii=False)
    with open(os.path.join(out_path, 'merges.txt'), 'w', encoding='utf-8') as f:
        print(f'dumping merges.txt into {out_path}')
        f.writelines(merges_out)
    

def bpe_tokenizer_main(input_path: str, vocab_size: int, special_tokens: list[str], chunk_num=8, out_path=None, max_workers=4):
    with timed('bpe main time'):
        vocab: dict[int, bytes] = init_vocab(special_tokens)
        pretrained_tokenization: dict[tuple, int] = {}
        merges: list[tuple[bytes, bytes]] = []

        with open(input_path, 'rb') as f:
            boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")
        offsets = list(zip(boundaries[:-1], boundaries[1:]))
        
        task_num = len(offsets)
        with timed('pre tokenizing total time'), ProcessPoolExecutor(max_workers) as ex:
            results = list(ex.map(bpe_tokenizer_worker, [input_path] * task_num, [special_tokens] * task_num, [offsets[i] for i in range(task_num)]))

        token_count: dict[str, int] = {}
        for result in results:
            for k, v in result.items():
                token_count[k] = token_count.get(k, 0) + v
        with timed('bpe pre-tokenizer: convert str to bytes'):
            pretrained_tokenization = {tuple(bytes([x]) for x in k.encode('UTF-8')):v for k, v in token_count.items()}

        with timed('bpe setting up pair:tokens dict'):
            # 建立 pair: set(tokens) 的字典
            pair_token: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
            for token in pretrained_tokenization.keys():
                for i in range(len(token) - 1):
                    pair = (token[i], token[i+1])
                    # 如果每次都新建一个set的话开销是100倍，以tinystoriesV2-train为例
                    # before
                    #bpe setting up pair:tokens dict: 4.253s rss:79.328MB
                    # after
                    # bpe setting up pair:tokens dict: 0.054s rss:79.516MB
                    pair_token[pair] = pair_token.get(pair, set())
                    pair_token[pair].add(token)
        with timed('bpe setting up pair:freq dict'):
            # 建立 pair: freq
            pair_count: dict[tuple[bytes, bytes], int] = {}
            for key in pretrained_tokenization.keys():
                for i in range(len(key)-1):
                    pair = (key[i], key[i+1])
                    pair_count[pair] = pair_count.get(pair, 0) + pretrained_tokenization[key]
        
        t0 = time.perf_counter()
        print(f'bpe merger: vocab size:{len(vocab)} elapsed time:{time.perf_counter() - t0:.3f}s rss:{psutil.Process().memory_info().rss/1024/1024:.3f}MB')
        with timed('bpe main merger'):
            while len(vocab) < vocab_size:
                before_vocab_size = len(vocab)
                bpe_tokenizer_merger(vocab, merges, pretrained_tokenization, pair_token, pair_count)
                after_vocab_size = len(vocab)
                if before_vocab_size == after_vocab_size:
                    break
                if len(vocab) % 200 == 0:
                    print(f'bpe merger: vocab size:{len(vocab)} elapsed time:{time.perf_counter() - t0:.3f}s rss:{psutil.Process().memory_info().rss/1024/1024:.3f}MB')
        print(f'bpe merger: vocab size:{len(vocab)} elapsed time:{time.perf_counter() - t0:.3f}s rss:{psutil.Process().memory_info().rss/1024/1024:.3f}MB')
    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        with timed('bpe serilizing time'):
            bpe_serializer(vocab, merges, out_path)
    

    vocab_vals = [f'[{v.decode('utf-8', errors='replace')}]' for v in list(vocab.values())]
    vocab_vals = sorted(vocab_vals, key=lambda p: (len(p), p), reverse=True)
    print(f'top 20 longest tokens are \n{'\n'.join(vocab_vals[:20])}')
    print(f'maxrss is {resource.getrusage(resource.RUSAGE_SELF ).ru_maxrss/1024/1024:.3f}MB')
    return vocab, merges
