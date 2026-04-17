import torch
from collections.abc import Callable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid LR: {lr}")
        defaults = {"lr": lr, "beta_1": betas[0], "beta_2": betas[1], "eps": eps, "ld": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            ld = group["ld"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                adjusted_lr = lr * ((1 - beta_2**t)**0.5 / (1 - beta_1**t))
                p.data -= lr*ld*p.data
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad)**2
                p.data -= adjusted_lr * (m / ((v)**0.5 + eps))
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        
        return loss
