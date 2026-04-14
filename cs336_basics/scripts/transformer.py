from torch import nn
from cs336_basics.scripts.transformer_layer import TransformerLayer
from cs336_basics.scripts.embedding_module import Embedding
from cs336_basics.scripts.linear_module import Linear
from cs336_basics.scripts.rms_norm import RMSNorm

class Transformer(nn.Module):
    def __init__(self,
                 vocab_size,
                 context_length,
                 num_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 eps,
                 theta,
                 is_rope,
                 device=None,
                 dtype=None):
        super().__init__()
        # Init embedding layer
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        # Init transformer layers
        self.transformer_layers = nn.ModuleList([TransformerLayer(
            d_model,
            num_heads,
            d_ff,
            eps,
            theta,
            context_length,
            is_rope,
            device=device,
            dtype=dtype
        ) for _ in range(num_layers)])

        # Init RMSNorm
        self.rms_norm = RMSNorm(d_model, eps, device=device, dtype=dtype)

        # Init linear layer for LM head
        self.linear = Linear(d_model, vocab_size)


    def forward(self, ids):
        x = self.embedding(ids)

        for _, l in enumerate(self.transformer_layers):
            x = l(x)
        
        x = self.rms_norm(x)
        x = self.linear(x)

        return x
