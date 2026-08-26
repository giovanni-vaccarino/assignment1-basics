import torch
from torch import nn
from cs336_basics.scripts.review.linear_module import Linear

class SwiGLU(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_ff: int = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        if d_ff:
            self.d_ff = d_ff
        else:
            self.d_ff = round(((8 / 3) * d_model / 64)) * 64
        
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_proj = self.w1(x)
        w3_proj = self.w3(x)

        return self.w2(self.silu(w1_proj) * w3_proj)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
