from torch import nn
import torch
from cs336_basics.scripts.review.transformer_layer import TransformerLayer
from cs336_basics.scripts.review.embedding_module import Embedding
from cs336_basics.scripts.review.linear_module import Linear
from cs336_basics.scripts.review.rms_norm import RMSNorm

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 eps: float,
                 theta: float,
                 device=None,
                 dtype=None):
        super().__init__()

        self.emb = Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, eps, theta, context_length, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.norm_head = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.linear_head = Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)

        for _, l in enumerate(self.transformer_layers):
            x = l(x)
        
        x = self.norm_head(x)
        x = self.linear_head(x)

        return x