
import unicodedata

import torch

from cs336_basics.basic_building_blocks import TransformerBlock, TransformerLM


def count_parameters(model) -> int:
    # 计算可学习参数量
    # N = 2·vocab_size·d_model + d_model + num_layers · params_per_block
    return sum(p.numel() for p in model.parameters())


def count_forward_flops(model: TransformerLM, context_length) -> int:
    # 前向传播FLOPs精确值
    # flops_per_block = 2·context_length·d_model · (4·d_model + 2·context_length + 3·d_ff)
    # total_flops = num_layers · flops_per_block + 2·context_length·d_model·vocab_size
    #         = 2·context_length·d_model
    #           · [ num_layers·(4·d_model + 2·context_length + 3·d_ff) + vocab_size ]
    num_layers = len(model.layers)
    block: TransformerBlock = model.layers[0]
    d_ff = block.ffn.w1.weight.shape[0]
    d_model = model.token_embeddings.weight.shape[1]
    vocab_size = model.token_embeddings.weight.shape[0]
    return 2 * context_length * d_model * (num_layers * (4 * d_model + 2 * context_length + 3 * d_ff) + vocab_size)


def count_flops_per_block(model: TransformerBlock, context_length) -> int:
    # flops_per_block = 2·context_length·d_model · (4·d_model + 2·context_length + 3·d_ff)
    d_ff = model.ffn.w1.weight.shape[0]
    d_model = model.ffn.w1.weight.shape[1]
    return 2 * context_length * d_model * (4 * d_model + 2 * context_length + 3 * d_ff)


def count_flops_llm_head(model: TransformerLM, context_length) -> int:
    # 2·context_length·d_model·vocab_size
    d_model = model.token_embeddings.weight.shape[1]
    vocab_size = model.token_embeddings.weight.shape[0]
    return 2 * context_length * d_model * vocab_size


def estimate_forward_flops(model, context_length) -> int:
    # 估算参数量经验公式
    # FLOPs ≈ 2 * N_params * T_tokens
    return 2 * count_parameters(model) * context_length


def _display_width(text: str) -> int:
    # 中文字符在终端占两格，len() 只算一格，直接用 len 对齐会错位
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in text)


def _pad(text: str, width: int, align_right: bool) -> str:
    fill = ' ' * max(0, width - _display_width(text))
    return fill + text if align_right else text + fill


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    # 除第一列（名称）外全部右对齐，方便比较数字位数
    widths = [
        max(_display_width(headers[i]), *(_display_width(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    rule = '-+-'.join('-' * w for w in widths)
    print(' | '.join(_pad(h, widths[i], i > 0) for i, h in enumerate(headers)))
    print(rule)
    for row in rows:
        print(' | '.join(_pad(c, widths[i], i > 0) for i, c in enumerate(row)))


def print_report_table(configs: dict[str, dict], context_length: int) -> None:
    """对比多个模型配置的参数量与一次前向传播的 FLOPs。

    configs: {名称: {vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta}}
             rope_theta 可缺省，默认 10000。
    context_length: 本次前向的序列长度。注意它既做模型的 max_seq_len，
                    也做 FLOPs 公式里的 T。

    模型在 meta 设备上构造，只有张量元信息、不分配存储。
    """
    scale_headers = ['model', 'd_model', 'layers', 'heads', 'd_ff', 'params', 'B', 'fp32 GiB']
    flops_headers = [
        'model', '1 block', 'blocks 合计', 'block %',
        'lm_head', 'lm_head %', '合计 TFLOPs', '2ND', 'ratio',
    ]
    scale_rows, flops_rows = [], []
    for name, cfg in configs.items():
        with torch.device('meta'):
            model = TransformerLM(
                cfg['vocab_size'],
                context_length,
                cfg['d_model'],
                cfg['num_layers'],
                cfg['num_heads'],
                cfg['d_ff'],
                cfg.get('rope_theta', 10000),
            )
        params = count_parameters(model)
        exact = count_forward_flops(model, context_length)
        approx = estimate_forward_flops(model, context_length)

        one_block = count_flops_per_block(model.layers[0], context_length)
        all_blocks = one_block * cfg['num_layers']
        head = count_flops_llm_head(model, context_length)
        # 分项必须加回总数：少一项 / 多算一层都会在这里被拦住
        assert all_blocks + head == exact, (all_blocks, head, exact)

        scale_rows.append([
            name,
            f'{cfg["d_model"]:,}',
            f'{cfg["num_layers"]:,}',
            f'{cfg["num_heads"]:,}',
            f'{cfg["d_ff"]:,}',
            f'{params:,}',
            f'{params / 1e9:.3f}',
            f'{params * 4 / 1024 ** 3:.2f}',
        ])
        flops_rows.append([
            name,
            f'{one_block / 1e12:.4f}',
            f'{all_blocks / 1e12:.3f}',
            f'{all_blocks / exact * 100:.1f}%',
            f'{head / 1e12:.3f}',
            f'{head / exact * 100:.1f}%',
            f'{exact / 1e12:.3f}',
            f'{approx / 1e12:.3f}',
            f'{exact / approx:.3f}',
        ])

    print(f'context_length = {context_length:,}\n')
    _print_table(scale_headers, scale_rows)
    print(
        '\nparams 为可训练参数量（不含 buffer）；B = params / 1e9\n'
        'fp32 GiB = params * 4 / 1024^3，仅权重，不含梯度 / 优化器状态 / 激活\n'
    )
    _print_table(flops_headers, flops_rows)
    print(
        '\nFLOPs 均为 TFLOPs（一次前向，batch = 1）；token_embeddings 是查表，0 FLOPs\n'
        'block % + lm_head % 应为 100%（舍入误差除外）\n'
        '2ND 为经验公式 2 * params * context_length；ratio = 合计 / 2ND'
    )




# d_ff 取 8/3 × d_model 向上取到 64 的倍数
GPT2_CONFIGS = {
    'GPT-2 small': dict(vocab_size=50257, d_model=768, num_layers=12, num_heads=12, d_ff=2048),
    'GPT-2 medium': dict(vocab_size=50257, d_model=1024, num_layers=24, num_heads=16, d_ff=2752),
    'GPT-2 large': dict(vocab_size=50257, d_model=1280, num_layers=36, num_heads=20, d_ff=3456),
    'GPT-2 XL': dict(vocab_size=50257, d_model=1600, num_layers=48, num_heads=25, d_ff=4288),
}


if __name__ == '__main__':
    print('=== (a)(b)(c)(d)  context_length = 1024 ===\n')
    print_report_table(GPT2_CONFIGS, 1024)

    print('\n\n=== (e)  GPT-2 XL, context_length = 16384 ===\n')
    print_report_table({'GPT-2 XL': GPT2_CONFIGS['GPT-2 XL']}, 16384)


