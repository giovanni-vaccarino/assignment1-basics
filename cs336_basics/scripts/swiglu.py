from torch import nn
import torch
from einops import einsum

class SwiGLU(nn.Module):
    "Implementation of the position-wise FFN."
    def __init__(self,
                 d_model,
                 d_ff=None,
                 device=None,
                 dtype=None):
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

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        w1_x = einsum(self.w1, x, "d_out d_in , ... d_in -> ... d_out")
        w3_x = einsum(self.w3, x, "d_out d_in , ... d_in -> ... d_out")
        z = self.silu(w1_x) * w3_x
        return einsum(self.w2, z, "d_in d_out , ... d_out -> ... d_in")
