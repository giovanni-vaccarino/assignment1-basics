from typing import Optional
from collections.abc import Callable
from torch import optim, nn
import torch
import math

class SGD(optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        # Init the optimizer. Params is the list of params to optimize. Can also take additional args
        # that we can pass to the super init call via the dict
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        # make one update of parameters. This will be called after the backward (after computing the gradients)
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= (lr / math.sqrt(t + 1)) * grad
                state["t"] = t + 1

        return loss

# Quick Testing

weights = nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=100)

for t in range(100):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()