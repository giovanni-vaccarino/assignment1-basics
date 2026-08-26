import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape (..., s, d_model)
        in_dype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.mean(x**2, dim=-1, keepdim=True) + self.eps) ** 0.5

        return ((self.weight * x ) / rms).to(in_dype)

# c = RMSNorm(5)

# x = torch.randn((2, 3, 5))
# print(x.shape)
# print(x)

# print(c.forward(x))