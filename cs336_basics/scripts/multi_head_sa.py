import torch
from torch import nn
from einops import rearrange
from cs336_basics.scripts.linear_module import Linear
from cs336_basics.scripts.dot_produc_attention import attention
from cs336_basics.scripts.rope import RoPE

class MultiHeadSelfAttention(nn.Module):
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
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        # one could also have a unique matrix for Q, K, V
        # self.w_qkv = Linear(d_model, 3*d_model)
        # QKV = seld.w_qkv(x)
        # QKV = rearrange(QKV, "... seq (c d_model) -> ... c seq d_model", c=3)
        # Q = torch.select(QKV, dim=-3, index=0)
        # K = torch.select(QKV, dim=-3, index=1)
        # V = torch.select(QKV, dim=-3, index=2)
        # but loading the weights with load_state_dict would be a bit problematic
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask > 0
        # less efficient way of doing that
        # attn_results = []
        # for i in range(self.num_heads):
        #     index = torch.arange(i, i + self.d_model // self.num_heads)
        #     attn_results.append(attention(
        #         torch.index_select(Q, -1, index),
        #         torch.index_select(K, -1, index),
        #         torch.index_select(V, -1, index),
        #         mask
        #     ))
        # concat_result = torch.cat(attn_results, dim=-1)
        Q = rearrange(Q, "... seq_len (h d_q) -> ... h seq_len d_q", h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)
        
        if self.is_rope:
            if token_positions is None:
                token_positions = torch.arange(0, Q.shape[-2])
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        attention_result = attention(Q, K, V, mask) # ... h seq_len d_v
        attention_result = rearrange(attention_result, "... h seq_len d_v -> ... seq_len (h d_v)", h=self.num_heads)

        return self.W_o(attention_result)
