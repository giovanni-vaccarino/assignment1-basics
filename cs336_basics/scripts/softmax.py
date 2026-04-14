import torch

def softmax(x, dim):
    "Applies softmax to tensor x along dimension dim"
    max_val = torch.max(x, dim=dim, keepdim=True).values

    return torch.exp(x - max_val) / torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True)
