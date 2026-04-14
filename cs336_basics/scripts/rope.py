import torch
from torch import nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self,
                 theta,
                 d_k,
                 max_seq_len,
                 device=None):
        super().__init__()
        k = torch.arange(d_k // 2)
        positions = torch.arange(max_seq_len)
        freqs = 1.0 / (theta **((2*k)/d_k))
        angles = torch.outer(positions, freqs)
        self.register_buffer('cos_vals', torch.cos(angles), persistent=False)
        self.register_buffer('sin_vals', torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos_vals[token_positions]
        sin = self.sin_vals[token_positions]
        x_pairs = rearrange(x, "... (d_k two) -> ... d_k two", two=2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        rotated = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        return rearrange(rotated, "... d_k two -> ... (d_k two)")
