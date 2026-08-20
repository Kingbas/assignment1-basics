

import torch
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.nn_utils import softmax


@torch.no_grad()
def decode(model: TransformerLM, tokenizer:BPETokenizer, max_tokens, temperature, top_k, top_p, prompt: str, end_token='<|endoftext|>') -> str:
    if temperature == 0:
        raise ValueError('temperature cannot be zero')
    if top_k > model.vocab_size:
        raise ValueError('top_k must be lower than vocab_size')

    max_ctx_len = model.context_length
    prompt_token_ids = tokenizer.encode(prompt)
    if len(prompt_token_ids) >= max_ctx_len:
        raise ValueError('prompt too long!')
    output_token_ids: list[int] = []
    end_token_id = tokenizer.encode(end_token)[0]

    next_token_id = -1
    while next_token_id != end_token_id:
        x_token_ids = prompt_token_ids + output_token_ids
        if len(x_token_ids) >= max_ctx_len:
            break
        if len(output_token_ids) >= max_tokens:
            break
        # 每次拼接提示词和输出
        x = torch.tensor(x_token_ids, dtype=torch.int, device=model.device)
        # get logits seq_len vocab_size
        x = model.forward(x)
        logits = x[-1]
        # 先温度
        logits = logits / temperature
        # 再softmax
        logits = softmax(logits, -1)
        # 再topk topp
        topk_res = torch.topk(logits, top_k)
        sorted_probs = topk_res.values
        prob_cumsum = torch.cumsum(sorted_probs, dim=0)
        mask = ~(prob_cumsum - sorted_probs < top_p)
        sorted_probs = torch.masked_fill(sorted_probs, mask, 0)
        # 归一化
        sorted_probs = sorted_probs / torch.sum(sorted_probs)
        assert abs(torch.sum(sorted_probs) - 1) < 1e-5

        prob_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token_id = int(topk_res.indices[prob_idx])
        output_token_ids.append(next_token_id)

    return tokenizer.decode(output_token_ids) if output_token_ids[-1] != end_token_id else tokenizer.decode(output_token_ids[:-1])


def sanity_check():
    top_k = 5
    top_p = 1
    logits = torch.tensor([0.45, 0.4, 0.08, 0.05, 0.02])
    # # 先温度
    # logits = logits / 0.1
    # # 再softmax
    # logits = softmax(logits, -1)
    # 再topk topp
    topk_res = torch.topk(logits, top_k)
    sorted_probs = topk_res.values
    prob_cumsum = torch.cumsum(sorted_probs, dim=0)
    mask = ~(prob_cumsum - sorted_probs < top_p)
    sorted_probs = torch.masked_fill(sorted_probs, mask, 0)
    # 归一化
    sorted_probs = sorted_probs / torch.sum(sorted_probs)

if __name__ == '__main__':
    hyperparas = {
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'num_layers': 4,
        'num_heads': 16,
        'd_ff': 1344,
        'rope_theta': 10000,
        'device': 'mps'
    }
    model = TransformerLM(**hyperparas)
    model.load_state_dict(torch.load('data/exp_log/ckpt-1')['model_weights'])
    model.eval()
    tokenizer = BPETokenizer.from_files('data/tinystoriesV2_train/vocab.json', 'data/tinystoriesV2_train/merges.txt', ['<|endoftext|>'])

    max_tokens = 256
    temperature = 1
    top_k = 30
    top_p = 0.9
    while True:
        prompt = input('plz input anything you would like me to expand\ninput: ')
        out = decode(model, tokenizer, max_tokens, temperature, top_k, top_p, prompt)
        print(out)


