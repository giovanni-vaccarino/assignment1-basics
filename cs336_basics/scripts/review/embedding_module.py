import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Vocab size or model dimension are invalid: less than zero")
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
