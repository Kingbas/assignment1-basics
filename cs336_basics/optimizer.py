"""对应 tests/test_optimizer.py:AdamW 与学习率调度。"""

import math
from collections.abc import Callable
from typing import Optional

import torch
from torch.optim.optimizer import ParamsT


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr, weight_decay, betas=(0.9, 0.999), eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0 or betas[1] < 0:
            raise ValueError("Invalid beta")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                if t == 1:
                    self.state[p]['m'] = torch.zeros_like(p)
                    self.state[p]['v'] = torch.zeros_like(p)
                
                lr_t = lr * math.sqrt(1 - group["beta2"] ** t) / (1 - group["beta1"] ** t) # if t is initiated with 0, the divisor will be zero
                p.data = p.data - lr * group["weight_decay"] * p.data

                self.state[p]['m'] = group["beta1"] * self.state[p]['m'] + (1 - group["beta1"]) * p.grad
                self.state[p]['v'] = group["beta2"] * self.state[p]['v'] + (1 - group["beta2"]) * p.grad ** 2
                p.data = p.data - lr_t * self.state[p]['m'] / (torch.sqrt(self.state[p]['v']) + group["eps"])

                state["t"] = t + 1  # Increment iteration number.
        return loss


def learning_rate_schedule(it: int,
                        max_learning_rate: float,
                        min_learning_rate: float,
                        warmup_iters: int,
                        cosine_cycle_iters: int,):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(((it - warmup_iters)/(cosine_cycle_iters - warmup_iters))*math.pi))*(max_learning_rate - min_learning_rate)
    return min_learning_rate
