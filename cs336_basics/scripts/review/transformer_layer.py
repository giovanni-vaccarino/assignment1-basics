import torch
from torch import nn
from cs336_basics.scripts.review.rms_norm import RMSNorm
from cs336_basics.scripts.review.multi_head_sa import MHA
from cs336_basics.scripts.review.swiglu import SwiGLU

class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 eps: float,
                 theta: float,
                 max_seq_len: int,
                 device=None,
                 dtype=None):
        super().__init__()

        self.norm_mha = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.norm_ffn = RMSNorm(d_model, eps, device=device, dtype=dtype)

        self.mha = MHA(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.norm_mha(x))

        return x + self.ffn(self.norm_ffn(x))
        