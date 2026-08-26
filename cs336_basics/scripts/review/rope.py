import torch
from torch import nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super().__init__()
        positions = torch.arange(max_seq_len, device=device)
        freqs = 1.0 / (theta ** ((2 * torch.arange(1, d_k // 2 + 1, device=device) - 2) / d_k))
        angles = torch.outer(positions, freqs)
        self.register_buffer(name="sin", tensor=torch.sin(angles), persistent=False)
        self.register_buffer(name="cos", tensor=torch.cos(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x.shape -> (..., s, d) token_pos.shape -> (..., s)

        x = rearrange(x, "... s (d c) -> ... s d c", c = 2)
        x1 = x[..., 0] # (..., s, d/2)
        x2 = x[..., 1]

        sin = self.get_buffer("sin")[token_positions] # (..., s)
        cos = self.get_buffer("cos")[token_positions]

        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rearrange(rotated, "... s d c -> ... s (d c)", c = 2)

        

# x = torch.randn((2, 3, 4, 2))

# print(x)

# x1 = x[..., 0]
# x2 = x[..., 1]
# print(x1.shape)
# print(x1)
# print(x2)
