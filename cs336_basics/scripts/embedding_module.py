import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 device=None,
                 dtype=None):
        super().__init__()
        init_weights = nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 0, 1, -3, 3)
        self.emb_matrix = nn.Parameter(data=init_weights)

    def forward(self, token_ids):
        return self.emb_matrix[token_ids]
