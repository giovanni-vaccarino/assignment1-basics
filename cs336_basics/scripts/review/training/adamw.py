from torch import optim, nn
from collections.abc import Callable
from typing import Optional, List
import math
import torch

class AdamW(optim.Optimizer):
    def __init__(self, 
                 params,
                 lr: float=1e-3,
                 betas: List[float] = [0.9, 0.999],
                 weight_decay: float = 1e-1,
                 eps: float = 1e-5):
        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "lambda": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            lambd = group["lambda"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)

                p.data -= lr * lambd * p.data # weight decay

                adaptive_lr = lr * (math.sqrt(1 - beta_2**t)) / (1 - beta_1**t)
                m = beta_1 * m + (1 - beta_1) * p.grad
                v = beta_2 * v + (1 - beta_2) * (p.grad**2)

                p.data -= adaptive_lr * m / (v**0.5 + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
                

        return loss

# Quick Testing

weights = nn.Parameter(5 * torch.randn((10, 10)))
opt = AdamW([weights], lr=1)

for t in range(100):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()
