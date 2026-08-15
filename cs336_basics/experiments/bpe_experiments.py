"""BPE 训练实验的入口。

算法本身全部在 ``cs336_basics.train_bpe`` 里，这个模块只负责一件事：
把一个语料名字变成一次可复现、有记录的 ``bpe_tokenizer_main`` 调用。
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass

from cs336_basics.train_bpe import bpe_tokenizer_main


@dataclass(frozen=True)
class Preset:
    """一个语料对应的默认参数。

    chunk_num / max_workers 是跟语料规模绑定的，不是自由组合，所以按语料成组给出，
    而不是让调用方每次手拼六个参数。
    """

    input_path: str
    vocab_size: int
    chunk_num: int
    max_workers: int
    out_path: str


PRESETS: dict[str, Preset] = {
    'cat':      Preset('data/the_cat_ate.txt',              10_000,   2,  4, 'data/the_cat_ate'),
    'toy':      Preset('data/toy_corpus.txt',               10_000,   2,  4, 'data/toy_corpus'),
    'ts-valid': Preset('data/TinyStoriesV2-GPT4-valid.txt', 10_000,   3,  4, 'data/tinystoriesV2_valid'),
    'ts-train': Preset('data/TinyStoriesV2-GPT4-train.txt', 10_000,  10, 10, 'data/tinystoriesV2_train'),
    'owt':      Preset('data/owt_train.txt',                32_000, 100, 10, 'data/owt_train'),
}

DEFAULT_SPECIAL_TOKENS = ['<|endoftext|>']


def peak_rss_mb(who: int) -> float:
    """进程的峰值 RSS，单位 MiB。

    ``ru_maxrss`` 在 macOS 上以字节计，在 Linux 上以 KiB 计 —— 单位搞错会得到差 1024 倍
    的荒谬数字，所以这里按平台分开处理。
    """
    peak = resource.getrusage(who).ru_maxrss
    return peak / 1024 / 1024 if sys.platform == 'darwin' else peak / 1024


def git_revision() -> str:
    def run(*args: str) -> str:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()

    try:
        rev = run('git', 'rev-parse', '--short', 'HEAD')
        return f'{rev}-dirty' if run('git', 'status', '--porcelain') else rev
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return 'unknown'


def print_run_header(corpus: str, params: dict) -> None:
    """在日志开头记录环境和参数。

    没有这段的话，一周后回看 train.log 无法确定它是哪个版本的代码跑出来的；而
    train_bpe.py 正在被频繁修改，测量结果一旦失去版本归属就作废了。

    ``assertions`` 一项记录 ``__debug__``：用 ``python -O`` 跑时所有 assert 会被剥掉，
    这对耗时有实际影响，必须和测量结果记在一起。
    """
    print('=' * 72)
    print(f'corpus     : {corpus}')
    print(f'git        : {git_revision()}')
    print(f'python     : {platform.python_version()} ({sys.implementation.name})')
    print(f'platform   : {platform.platform()}')
    print(f'cpu count  : {os.cpu_count()}')
    print(f'assertions : {__debug__}')
    print(f'started    : {time.strftime("%Y-%m-%d %H:%M:%S")}')
    for key, value in params.items():
        print(f'  {key:<14}= {value!r}')
    print('=' * 72, flush=True)


def summarize(vocab: dict[int, bytes], merges: list, wall_seconds: float) -> dict:
    longest = max(vocab.values(), key=len)
    return {
        'wall_seconds': round(wall_seconds, 3),
        'vocab_size': len(vocab),
        'num_merges': len(merges),
        'longest_token_len': len(longest),
        'longest_token': longest.decode('utf-8', errors='replace'),
        'peak_rss_self_mb': round(peak_rss_mb(resource.RUSAGE_SELF), 3),
        'peak_rss_children_mb': round(peak_rss_mb(resource.RUSAGE_CHILDREN), 3),
        'assertions_enabled': __debug__,
        'git': git_revision(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train a BPE tokenizer on one of the assignment corpora.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--corpus', choices=sorted(PRESETS), default='toy',
                        help='which corpus preset to run')
    # 下面几个默认 None，表示"沿用 preset 的值"；只有显式传了才覆盖。
    parser.add_argument('--input-path', help='override the preset corpus path')
    parser.add_argument('--vocab-size', type=int, help='override the preset vocab size')
    parser.add_argument('--chunk-num', type=int, help='override the preset chunk count')
    parser.add_argument('--max-workers', type=int, help='override the preset worker count')
    parser.add_argument('--out-path', help='override where vocab.json / merges.txt go')
    parser.add_argument('--special-token', action='append', dest='special_tokens',
                        help='may be repeated; defaults to <|endoftext|>')
    parser.add_argument('--no-serialize', action='store_true',
                        help='skip writing vocab.json / merges.txt (useful while profiling)')
    parser.add_argument('--summary-json', metavar='PATH',
                        help='also write the run summary to this file as JSON')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = PRESETS[args.corpus]

    # 用 `is None` 而不是 `or`：显式传 --chunk-num 0 也应该被当成"传了"，`or` 会把它
    # 当作缺省而静默换成 preset 的值。
    def pick(override, fallback):
        return fallback if override is None else override

    input_path = pick(args.input_path, preset.input_path)
    out_path = None if args.no_serialize else pick(args.out_path, preset.out_path)
    params = {
        'input_path': input_path,
        'vocab_size': pick(args.vocab_size, preset.vocab_size),
        'special_tokens': pick(args.special_tokens, DEFAULT_SPECIAL_TOKENS),
        'chunk_num': pick(args.chunk_num, preset.chunk_num),
        'out_path': out_path,
        'max_workers': pick(args.max_workers, preset.max_workers),
    }

    # 早失败：语料不存在时立刻报错，而不是等 ProcessPoolExecutor 里的 worker 抛出来
    # 再被 .result() 重新抛一遍 —— 那种 traceback 读起来要绕一大圈。
    if not os.path.isfile(input_path):
        print(f'corpus not found: {input_path}', file=sys.stderr)
        return 1

    print_run_header(args.corpus, params)

    t0 = time.perf_counter()
    vocab, merges = bpe_tokenizer_main(**params)
    summary = summarize(vocab, merges, time.perf_counter() - t0)

    print('-' * 72)
    for key, value in summary.items():
        print(f'{key:<22}: {value!r}')
    print('-' * 72, flush=True)

    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'wrote summary to {args.summary_json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
