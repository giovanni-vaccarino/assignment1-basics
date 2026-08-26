import torch
from torch import nn
from einops import einsum, reduce, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = (2 / (in_features + out_features)) ** 0.5
        weights = torch.ones(in_features, out_features, device=device, dtype=dtype)
        nn.init.trunc_normal_(weights, 0, sigma, -3*sigma, 3*sigma)
        self.weight = nn.Parameter(data=weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_in d_out -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(w, 0, 1, -3, 3)
        self.weights = nn.Parameter(data=w)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.g = torch.ones((d_model), device=device, dtype=dtype)
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is batch seq d_model 
        # RMS(a_i) = a_i / RMS(a) * g_i

        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Calculate RMS(a) -> one per el of the seq
        mean = reduce(x**2, "... d_model -> ... 1", 'mean')
        rms = (mean + self.eps) **0.5

        normalized = (x / rms) * self.g

        return normalized.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = (8/3) * d_model
        d_ff = round(d_ff / 64) * 64
        init_weights_w1 = torch.empty(d_ff, d_model, device=device, dtype=dtype) # following math convention
        init_weights_w2 = torch.empty(d_model, d_ff, device=device, dtype=dtype) # following math convention
        init_weights_w3 = torch.empty(d_ff, d_model, device=device, dtype=dtype) # following math convention
        std_dev = (2 / (d_model + d_ff)) ** 0.5
        nn.init.trunc_normal_(init_weights_w1, 0, std_dev, std_dev * -3, std_dev * 3)
        nn.init.trunc_normal_(init_weights_w2, 0, std_dev, std_dev * -3, std_dev * 3)
        nn.init.trunc_normal_(init_weights_w3, 0, std_dev, std_dev * -3, std_dev * 3)
        self.w1 = nn.Parameter(data=init_weights_w1)
        self.w2 = nn.Parameter(data=init_weights_w2)
        self.w3 = nn.Parameter(data=init_weights_w3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x -> W2(SILU(W1 x) x W3 x)
        activation_w1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        activation_w3 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        x_w2 = torch.mul(self.silu(activation_w1), activation_w3)
        return einsum(x_w2, self.w2, "... d_ff, d_model d_ff -> ... d_model")

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class RoPE(nn.Module):
    def __init__(self, theta, d_model, max_seq_len, device=None):
        super().__init__()
        k = torch.arange(d_model // 2)
        i = torch.arange(max_seq_len)
        freqs = 1.0 / (theta)**((2*k - 2) / d_model)
        angles = torch.outer(i, freqs)
        self.register_buffer('sin', torch.sin(angles), persistent=False)
        self.register_buffer('cos', torch.cos(angles), persistent=False)

    def forward(self, x:torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        x_pairs = rearrange(x, "... (d_model pair) -> ... d_model pair", pair=2)
        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        rotated = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)

        return rearrange(rotated, "... d_model pair -> ... d_model pair", pair=2)


def softmax(x: torch.Tensor, dim: int):
    # s(x) = exp(x_i) / sum(exp(x_i))
    max_val = torch.max(x, dim=dim, keepdim=True)
    x = x - max_val.values

    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k")
    normalized_qk = qk / (d_k) ** 0.5

    if mask is not None:
        normalized_qk = torch.where(mask, normalized_qk, float("-inf"))

    return einsum(softmax(normalized_qk, dim=-1), V, "... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v")

class MultiHeadSelfAttention(nn.Moudule):
    def __init__(self,
                 d_model,
                 num_heads,
                 theta=0.0,
                 max_seq_len=0,
                 rope=False,
                 device=None,
                 dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.is_rope = rope
        if rope:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
            pass
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (b s d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = rearrange(Q, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq (num_heads d_k) -> ... num_heads seq d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq (num_heads d_v) -> ... num_heads seq d_v", num_heads=self.num_heads)
        
        seq_len = x.shape[-2]
        if self.is_rope:
            Q = self.rope(Q, torch.arange(seq_len))
            K = self.rope(K, torch.arange(seq_len))
        
        lower_triangle = torch.tril(torch.ones((seq_len, seq_len)), device=x.device)
        mask = lower_triangle > 0

        attn_per_head = attention(Q, K, V, mask=mask)
        attn_per_head = rearrange(attn_per_head, "... num_heads seq d_v -> ... seq (num_heads d_v)", num_heads=self.num_heads)

        return self.W_o(attn_per_head)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.rms_norm_mha(x))
        x = x + self.ffn(self.rms_norm_ffn(x))

        return x

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
    
    def forward(self, token_ids: list[int]):
        x = self.embedding(token_ids)

        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.rms_norm(x)

        return self.linear(x)
