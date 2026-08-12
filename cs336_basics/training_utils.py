import os
from typing import IO, BinaryIO

from einops import rearrange
import numpy as np
import numpy.typing as npt
import torch

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # 起点范围为[0, len(dataset) - context_length - 1]
    # 因为右边极端情况下out的最后一个下标为 len(dataset) - 1
    # 起点是out最后一个下标往左边数ctx个，故起点的最大值为 len(dataset) - ctx - 1
    # 起点可能的个数为len(dataset) - ctx
    start_idx = torch.randint(0, len(dataset) - context_length, (batch_size,))
    start_idx = rearrange(start_idx, 'batch_size -> batch_size 1')
    offset = torch.arange(0, context_length)
    input_idx = start_idx + offset
    # int64避免到embedding代码炸了
    input_token_ids = torch.tensor(dataset[input_idx], dtype=torch.int64, device=device)
    output_idx = input_idx + 1
    output_token_ids = torch.tensor(dataset[output_idx], dtype=torch.int64, device=device)

    return (input_token_ids, output_token_ids)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_weights = model.state_dict()
    opt_weights = optimizer.state_dict()
    checkpoint = {
        'model_weights': model_weights,
        'opt_weights': opt_weights,
        'it': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_weights'])
    optimizer.load_state_dict(checkpoint['opt_weights'])
    return checkpoint['it'] 


if __name__ == '__main__':
    a = np.zeros((50))
    pass
