
from cs336_basics.train_bpe import bpe_tokenizer_main

def train_bpe():
    pass
    # bpe_tokenizer_main('data/toy_corpus.txt', 10000, ['<|endoftext|>'], 2, 'data/toy_corpus')
    bpe_tokenizer_main('data/TinyStoriesV2-GPT4-valid.txt', 10000, ['<|endoftext|>'], 3, 'data/tinystoriesV2_valid')
    # bpe_tokenizer_main('data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'], 10, 'data/tinystoriesV2_train', 10)
    # bpe_tokenizer_main('data/owt_train.txt', 32000, ['<|endoftext|>'], 100, 'data/owt_train', 10)


if __name__ == '__main__':
    train_bpe()


