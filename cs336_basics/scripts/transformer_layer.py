import torch
from torch import nn
from cs336_basics.scripts.rms_norm import RMSNorm
from cs336_basics.scripts.multi_head_sa import MultiHeadSelfAttention
from cs336_basics.scripts.swiglu import SwiGLU

class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_ff,
                 eps,
                 theta,
                 max_seq_len,
                 is_rope,
                 device=None,
                 dtype=None):
        super().__init__()
        # Init the 2 RMSNorm
        self.rms_norm_mha = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.rms_norm_ffn = RMSNorm(d_model, eps, device=device, dtype=dtype)

        # Init the MHA
        self.mha = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len, rope=is_rope, device=device, dtype=dtype)

        # Init the SwiGLU (FFN)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.mha(self.rms_norm_mha(x))
        x = x + self.ffn(self.rms_norm_ffn(x))

        return x
