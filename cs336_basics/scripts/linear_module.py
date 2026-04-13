from torch import nn
import torch
from einops import einsum

class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        std_dev = (2 / (in_features + out_features)) ** 0.5
        init_weights = torch.empty(out_features, in_features, device=device, dtype=dtype) # following math convention
        nn.init.trunc_normal_(init_weights, 0, std_dev, std_dev * -3, std_dev * 3)
        self.W = nn.Parameter(data=init_weights)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")
