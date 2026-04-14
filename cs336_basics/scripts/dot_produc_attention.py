import torch
from einops import einsum
from cs336_basics.scripts.softmax import softmax

def attention(Q, K, V, mask=None):
    Z = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    Z = Z / (Q.shape[-1])**0.5
    if mask is not None:
        Z = torch.where(mask, Z, float('-inf'))
    attn_weights = softmax(Z, dim=-1)
    return einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

# Q = torch.randn(4, 2)
# K = torch.randn(4, 2)
# V = torch.randn(4, 3)

# print(attention(Q, K, V))
