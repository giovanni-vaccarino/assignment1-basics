import numpy as np
import torch
from einops import repeat

def data_loader(x: np.array, batch_size: int, context_length: int, device: str):
    # start_indexes = torch.randint(0, x.shape[0] - context_length, tuple([batch_size]))
    # indexes = torch.stack([torch.arange(index, index + context_length, 1) for index in start_indexes])
    # indexes_targets = torch.stack([torch.arange(index + 1, index + context_length + 1, 1) for index in start_indexes])
    # x = repeat(x, "n -> b n", b=batch_size)
    # x_tokens = torch.gather(torch.tensor(x), -1, indexes).to(device)
    # y_tokens = torch.gather(torch.tensor(x), -1, indexes_targets).to(device)
    # return x_tokens, y_tokens

    # more efficient version with broadcasting
    start_indexes = np.random.randint(0, len(x) - context_length, batch_size)
    # shape: (batch_size, context_length)
    indexes = start_indexes[:, None] + np.arange(context_length)[None, :]
    x_tokens = torch.tensor(x[indexes], dtype=torch.long).to(device)
    y_tokens = torch.tensor(x[indexes + 1], dtype=torch.long).to(device)
    return x_tokens, y_tokens

x = np.array([10, 12, 43, 31, 53, 75])

x, y = data_loader(x, 3, 2, "cpu")
print(x)
print(y)
