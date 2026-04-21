import numpy as np
import torch


def data_loader(
    dataset: np.ndarray | str,
    batch_size: int,
    context_length: int,
    device: str,
    dtype: np.dtype = np.uint16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from a token array or a memory-mapped .npy file.

    Args:
        dataset: Either a numpy array already in memory, or a path to a .npy
                 file that will be loaded via mmap (no RAM copy of the full file).
        dtype:   Must match the dtype used when the .npy file was saved (uint16
                 for vocab sizes ≤ 65535).  Ignored when dataset is already an array.
    """
    if isinstance(dataset, str):
        x = np.load(dataset, mmap_mode="r").view(dtype)
        assert x.max() < np.iinfo(dtype).max, (
            f"Token values exceed dtype {dtype} range — dtype mismatch?"
        )
    else:
        x = dataset

    start_indexes = np.random.randint(0, len(x) - context_length, batch_size)
    indexes = start_indexes[:, None] + np.arange(context_length)[None, :]
    x_tokens = torch.tensor(x[indexes].astype(np.int64), dtype=torch.long).to(device)
    y_tokens = torch.tensor(x[indexes + 1].astype(np.int64), dtype=torch.long).to(device)
    return x_tokens, y_tokens
