
from itertools import islice
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.common import find_chunk_boundaries
from contextlib import contextmanager

import numpy as np
import os

import time
import psutil

@contextmanager
def timed(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - t0:.3f}s rss:{psutil.Process().memory_info().rss/1024/1024:.3f}MB")

special_tokens = ['<|endoftext|>']

def encode_data(vocab_path, merges_path, input_path, output_path, special_tokens = ['<|endoftext|>']):
    print(f'using vocab: {vocab_path}')
    print(f'using merges: {merges_path}')
    print(f'encoding: {input_path}')
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    t0 = time.perf_counter()
    res = []
    with open(input_path) as f:
        text = f.read()
        encoded = tokenizer.encode(text)
        res += encoded
    t1 = time.perf_counter()
    total_time = t1 - t0

    if output_path:
        with timed(f'writing numpy array to {output_path}'):
            with open(output_path, 'wb') as f:
                np.array(res, dtype=np.uint16).tofile(output_path)

    print(f'total text size is {len(text)}bytes({len(text)/1000/1000:.3f}MB). the text has been compressed into {len(res)} tokens')
    print(f'compress ratio is {len(text)/len(res):.3f}bytes/token, total process time is {total_time:.3f}s, throughput is {(len(text) / total_time):.3f}bytes/s')
    return res


def encode_data_by_offset(vocab_path, merges_path, input_path, offset: tuple[int, int], special_tokens = ['<|endoftext|>']):
    print(f'using vocab: {vocab_path}')
    print(f'using merges: {merges_path}')
    print(f'encoding: {input_path}')
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    t0 = time.perf_counter()
    res = []
    with open(input_path, 'rb') as f:
        f.seek(offset[0])
        text = f.read(offset[1] - offset[0])
        encoded = tokenizer.encode(text.decode(encoding='utf-8'))
        res += encoded
    t1 = time.perf_counter()
    total_time = t1 - t0

    print(f'total text size is {len(text)}bytes({len(text)/1000/1000:.3f}MB). the text has been compressed into {len(res)} tokens')
    print(f'compress ratio is {len(text)/len(res):.3f}bytes/token, total process time is {total_time:.3f}s, throughput is {(len(text) / total_time):.3f}bytes/s')
    return res


def encode_data_parallel(vocab_path, merges_path, input_path, num_chunks, num_workers, special_tokens = ['<|endoftext|>']):
    # TODO: 实现并行encoder
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens)
    





def encode_data_v2(vocab_path, merges_path, input_path, output_path, special_tokens = ['<|endoftext|>']):
    print(f'using vocab: {vocab_path}')
    print(f'using merges: {merges_path}')
    print(f'encoding: {input_path}')
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens)

    file_size = os.path.getsize(input_path)

    t0 = time.perf_counter()
    token_count = 0
    with open(input_path) as f, open(output_path, 'wb') as f1:
        gen = tokenizer.encode_iterable(f)
        while True:
            buf = list(islice(gen, 1000000))
            if not buf:
                break
            np.array(buf, dtype=np.uint16).tofile(f1)
            token_count += len(buf)
            if token_count % 100000 == 0:
                print(f'processing: {token_count} tokens, time elapsed: {(time.perf_counter() - t0):.3f}s')
    t1 = time.perf_counter()
    total_time = t1 - t0


    print(f'total text size is {file_size}bytes({file_size/1000/1000:.3f}MB). the text has been compressed into {token_count} tokens')
    print(f'compress ratio is {file_size/token_count:.3f}bytes/token, total process time is {total_time:.3f}s, throughput is {(file_size / total_time):.3f}bytes/s')





# encode_data('data/tinystoriesV2_train/vocab.json', 'data/tinystoriesV2_train/merges.txt', 'data/TinyStoriesV2-GPT4-train.txt', 'data/tinystoriesV2_train/TinyStoriesV2-GPT4-train.npy')
# encode_data('data/tinystoriesV2_train/vocab.json', 'data/tinystoriesV2_train/merges.txt', 'data/TinyStoriesV2-GPT4-valid.txt', 'data/tinystoriesV2_train/TinyStoriesV2-GPT4-valid.npy')

# encode_data('data/owt_train/vocab.json', 'data/owt_train/merges.txt', 'data/owt_valid.txt', 'data/owt_valid.npy')
# encode_data('data/owt_train/vocab.json', 'data/owt_train/merges.txt', 'data/owt_train.txt', 'data/owt_train.npy')

encode_data_v2('data/tinystoriesV2_train/vocab.json', 'data/tinystoriesV2_train/merges.txt', 'data/TinyStoriesV2-GPT4-valid.txt', 'data/tinystoriesV2_train/TinyStoriesV2-GPT4-valid_v2.npy')
encode_data_v2('data/tinystoriesV2_train/vocab.json', 'data/tinystoriesV2_train/merges.txt', 'data/TinyStoriesV2-GPT4-train.txt', 'data/tinystoriesV2_train/TinyStoriesV2-GPT4-train_v2.npy')
encode_data_v2('data/owt_train/vocab.json', 'data/owt_train/merges.txt', 'data/owt_valid.txt', 'data/owt_valid.bin')
encode_data_v2('data/owt_train/vocab.json', 'data/owt_train/merges.txt', 'data/owt_train.txt', 'data/owt_train.bin')

