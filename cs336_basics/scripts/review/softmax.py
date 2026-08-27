import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # x.shape -> (..., v)
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e = torch.exp(x - x_max)

    # softmax_wm = torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)
    # print(f"Softmax without max: \n {softmax_wm}")
    # assert(torch.allclose(softmax_wm, softmax))

    return e / torch.sum(e, dim=dim, keepdim=True)


# x = torch.randn((1, 3, 4))
# print(x)

# x     -> (1, 3, 4)
# x_max -> (1, 3, 1)
# x - x_max -> (1, 3, 4)
# ...

# print(softmax(x))
