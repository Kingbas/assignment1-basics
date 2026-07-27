

import regex as re

"""

    首先要对corpus进行pre_tokenization
    利用
    BPE算法是初始化词汇表为ASCII码 外加一个特殊字符 <|endoftext|>
    
"""

# regex pattern used in gpt-2 pre-tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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


def bpe_tokenizer_worker(file_path: str, special_tokens: list[str], pretrained_tokenization: dict[tuple, int]):
    with open(file=file_path) as f:
        texts = f.read()
    special_token_pattern = '|'.join(re.escape(special_token) for special_token in special_tokens)
    texts = re.split(special_token_pattern, texts)

    
    for text in texts:
        tokens = re.findall(PAT, text)
        for token in tokens:
            key = tuple(bytes([x]) for x in token.encode('UTF-8'))
            pretrained_tokenization[key] = pretrained_tokenization.get(key, 0) + 1
    pass


def bpe_tokenizer_merger(vocab, merges, pretrained_tokenization) -> dict[tuple, int]:
    pair_count: dict[tuple[bytes, bytes], int] = {}
    # 通过词频统计pair频率
    for key in pretrained_tokenization.keys():
        for i in range(len(key)-1):
            pair = (key[i], key[i+1])
            pair_count[pair] = pair_count.get(pair, 0) + pretrained_tokenization[key]
    # 找到频次最高的pairs，对pairs进行字典序排序，将字典序最大者添加进vocab
    best_pair = max(pair_count, key=lambda p: (pair_count[p], p))
    vocab[len(vocab)] = b''.join(best_pair)
    merges.append(best_pair)
    # 再次遍历pretrained_tokenization，在每一个key上应用刚才的merge
    freq_temp: dict[tuple, int] = {}
    for token, freq in pretrained_tokenization.items():
        temp_list = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and tuple([token[i], token[i+1]]) == best_pair:
                temp_list.append(token[i] + token[i+1])
                i = i + 2
            else:
                temp_list.append(token[i])
                i = i + 1
        freq_temp[tuple(temp_list)] = freq
    return freq_temp


def bpe_tokenizer_main(input_path: str, vocab_size: int, special_tokens: list[str]):
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    pretrained_tokenization: dict[tuple, int] = {}
    merges: list[tuple[bytes, bytes]] = []

    bpe_tokenizer_worker(input_path, special_tokens, pretrained_tokenization)
    while len(vocab) < vocab_size:
        pretrained_tokenization = bpe_tokenizer_merger(vocab, merges, pretrained_tokenization)
    return vocab, merges

if __name__ == '__main__':
    special_tokens = [r'<|endoftext|>', r'<|padding|>']
    bpe_tokenizer_main('data/toy_corpus.txt', 512, special_tokens)
    pass
