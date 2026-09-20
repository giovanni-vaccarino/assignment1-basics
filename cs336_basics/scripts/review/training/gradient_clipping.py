from typing import Iterable
import torch
import math

def gradient_clipping(params: Iterable[torch.nn.Parameter], M: float, eps:float = 1e-6 ) -> Iterable[torch.nn.Parameter]:
    # 1. Compute l2_norm
    # 2. If l2_norm >= M rescale in place

    l2_norm = 0

    for param in params:
        if param.grad is None:
            continue
        l2_norm += param.grad.data.norm()**2

    l2_norm = torch.sqrt(l2_norm)

    if l2_norm < M:
        return params

    rescale_factor = M / (l2_norm + eps)

    for param in params:
        if param.grad is None:
            continue
        param.grad = param.grad * rescale_factor

    return params