import torch
from torch import nn
from einops import reduce

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model,
                 eps,
                 device=None,
                 dtype=None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(data=torch.ones(d_model, device=device, dtype=dtype))

    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(reduce(x**2, "... d_model -> ... 1", "mean") + self.eps)
        
        normalized_x = (x / rms) * self.g

        return normalized_x.to(in_dtype)
