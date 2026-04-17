import torch
from collections.abc import Callable
from typing import Optional

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid LR: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                # get iteration number
                state = self.state[p] # get state associated with p
                t = state.get("t", 0)
                d_lr = lr / (t + 1)**0.5
                grad = p.grad.data # get the gradient of the loss w.r.t. p
                p.data -= d_lr * grad
                state["t"] = t + 1

        return loss

# Example usage
weights = torch.nn.ParameterList([torch.nn.Parameter(5 * torch.randn((10,10)))])
opt = SGD(weights, lr=1e2)

for _ in range(10):
    opt.zero_grad() # Reset the gradients for all the learnable params
    loss = (weights[0]**2).mean() # Compute a scalar loss value
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes the gradients
    opt.step() # Run optimizer step