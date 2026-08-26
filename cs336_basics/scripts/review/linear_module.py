import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        W = torch.empty((out_features, in_features), dtype=dtype, device=device)
        std_dev = (2 / (in_features + out_features))**0.5
        nn.init.trunc_normal_(W, 0, std_dev, a= -3 * std_dev, b= 3 * std_dev)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "out_f in_f, ... in_f -> ... out_f")
