import torch
from torch import nn
from einops import rearrange
from cs336_basics.scripts.review.dot_produc_attention import attention
from cs336_basics.scripts.review.rope import RoPE
from cs336_basics.scripts.review.linear_module import Linear

class MHA(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model={d_model} not divisible by num_heads={num_heads}"
        assert (theta is None) == (max_seq_len is None), "theta and max_seq_len must be given together"
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.num_heads = num_heads
        self.device = device
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device)
        # in GQA or MQA you can't do this, remove the value in that case
        self.qkv_proj = Linear(d_model, 3* self.d_k * num_heads, device=device, dtype=dtype)
        self.w_o = Linear(self.d_v * num_heads, d_model, device=device, dtype=dtype)        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q, K, V = torch.chunk(self.qkv_proj(x), 3, dim=-1)

        Q = rearrange(Q, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        K = rearrange(K, "... T (h d_k) -> ... h T d_k", h=self.num_heads)
        V = rearrange(V, "... T (h d_v) -> ... h T d_v", h=self.num_heads)

        T = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0, Q.shape[-2])
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.triu(torch.ones((T, T), device=self.device), diagonal=1) == 0
        #mask = torch.tril(torch.ones(T, T, dtype=bool)) Equivalent
        attn = attention(Q, K, V, mask=mask)

        attn = rearrange(attn, "... h T d_v -> ... T (h d_v)")

        return self.w_o(attn)

# a = torch.triu(torch.randn((5, 5)))
# print(a)

# print(a == 0)



