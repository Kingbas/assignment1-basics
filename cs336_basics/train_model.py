
import argparse
import math
from cs336_basics.model import TransformerLM
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy_loss, gradient_clipping
import numpy as np
import torch

from cs336_basics.optimizer import AdamW, learning_rate_schedule

GPT2_CONFIGS = {
    'GPT-2 small': dict(vocab_size=50257, d_model=768, num_layers=12, num_heads=12, d_ff=2048),
    'GPT-2 medium': dict(vocab_size=50257, d_model=1024, num_layers=24, num_heads=16, d_ff=2752),
    'GPT-2 large': dict(vocab_size=50257, d_model=1280, num_layers=36, num_heads=20, d_ff=3456),
    'GPT-2 XL': dict(vocab_size=50257, d_model=1600, num_layers=48, num_heads=25, d_ff=4288),
}

preset = GPT2_CONFIGS['GPT-2 small']


device = 'mps'
hyperparas = {
    'vocab_size': 10000,
    'context_length': 256,
    'd_model': 512,
    'num_layers': 4,
    'num_heads': 16,
    'd_ff': 1344,
    'rope_theta': 10000,
    'device': device
}

torch.manual_seed(42)

lm = TransformerLM(**hyperparas)

max_lr = 5e-4
weight_decay= 0.01
def init_opt(model, max_lr, weight_decay):
    return AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)

opt = init_opt(lm, max_lr, weight_decay)

input_path = 'data/TinyStoriesV2-GPT4-valid.bin'
input = np.memmap(input_path, dtype=np.uint16, mode='r')

step = 100
batch_size = 16
warmup_iters = step * 0.02

def sanity_check():
    # 初始loss为ln(vocab_size)
    # 过拟合单个batch
    # lr曲线 warmup 线性升、余弦降到 min_lr、之后恒定
    batch = None
    for i in range(step):
        lr = learning_rate_schedule(it=i+1, max_learning_rate=max_lr, min_learning_rate=max_lr / 10, warmup_iters=warmup_iters, cosine_cycle_iters=step)
        opt.param_groups[0]['lr'] = lr
        opt.zero_grad()
        if batch is None:
            batch = get_batch(dataset=input, batch_size=batch_size, context_length=hyperparas['context_length'], device=device)
        out = lm(batch[0])
        loss = cross_entropy_loss(out, batch[1])
        loss.backward()
        # gradient clipping
        gradient_clipping(lm.parameters(), max_l2_norm=1.0)
        opt.step()
        if i % 1 == 0:
            print(f'training step {i}, loss: {loss}, adamW lr: {opt.param_groups[0]['lr']}, ')

def main():
    for i in range(step):
        lr = learning_rate_schedule(it=i+1, max_learning_rate=max_lr, min_learning_rate=max_lr / 10, warmup_iters=warmup_iters, cosine_cycle_iters=step)
        opt.param_groups[0]['lr'] = lr
        opt.zero_grad()
        batch = get_batch(dataset=input, batch_size=batch_size, context_length=hyperparas['context_length'], device=device)
        out = lm(batch[0])
        loss = cross_entropy_loss(out, batch[1])
        loss.backward()
        # gradient clipping
        gradient_clipping(lm.parameters(), max_l2_norm=1.0)
        opt.step()
        if i % 1 == 0:
            print(f'training step {i}, loss: {loss}, adamW lr: {opt.param_groups[0]['lr']}, ')

main()


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        prog="train",
        description="出现在 --help 顶部",
        epilog="出现在 --help 底部",
    )
    # model
    m = p.add_argument_group('model', '模型超参数')
    m.add_argument('--vocab-size', type=int, help='词汇量')


    # optim
    o = p.add_argument_group('optim', 'AdamW优化器超参数')


    # sched
    s = p.add_argument_group('sched', 'lr warm-up')

    # train
    t = p.add_argument_group('train', '训练时超参数')


    # data
    d = p.add_argument_group('data', '采样训练数据')


    # ckpt
    c = p.add_argument_group('ckpt', '检查点')
