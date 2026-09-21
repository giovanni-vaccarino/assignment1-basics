import numpy as np
from typing import Tuple
import torch

def data_loader(token_ids: np.array, batch_size: int, context_length: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # returns [Tensor(sampled input sequences), Tensor(next target tokens)] both with shape (batch_size, context_length)

    # Optional checks for valid inputs
    start_indexes = np.random.randint(0, len(token_ids) - context_length, batch_size) # (B)
    # (B, T)
    indexes = start_indexes[:, None] + np.arange(context_length)[None, :]
    x = torch.tensor(token_ids[indexes]).to(device)
    y = torch.tensor(token_ids[indexes + 1]).to(device)

    return x, y

# token_ids = np.random.randint(0, 100, 10)
# print(f"Token_ids: \n {token_ids}")
# start_indexes = np.random.randint(0, len(token_ids) - 3, 4)
# indexes = start_indexes[:, None] + np.arange(3)
# print(f"Indexes: \n {indexes}")

# x = token_ids[indexes]

# print(x.shape)
# print(f"X: \n {x}")
