import torch
from einops import einsum
from cs336_basics.scripts.review.softmax import softmax

def attention(Q: torch.Tensor, 
              K: torch.Tensor, 
              V: torch.Tensor,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    # 1. Q K^T
    z = einsum(Q, K, "... N d_k, ... M d_k -> ... N M")
    # 2. Normalize
    norm_factor = (Q.shape[-1])**0.5
    z = z / norm_factor
    # Optional. Set based on the mask -inf in the masked positions
    if mask is not None:
        z = torch.where(condition=mask, input=z, other=float('-inf'))
    # 3. Softmax
    z_softmax = softmax(z, -1)
    # 4. V
    return einsum(z_softmax, V, "... N M, ... M d_v -> ... N d_v")
