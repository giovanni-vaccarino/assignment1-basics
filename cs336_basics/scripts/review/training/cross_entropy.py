import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits.shape  -> (B, T, V)
    # targets.shape -> (B, T)
    assert logits.shape[:-1] == targets.shape, f"{logits.shape=} incompatible with {targets.shape=}"
    assert not targets.is_floating_point(), "targets must be integer class indices"

    max_logits = torch.max(logits, dim=-1, keepdim=True).values # (B, T, 1)
    shifted = logits - max_logits # (B, T, V)

    log_exp_sum = torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True)) # (B, T, 1)
    target_logits = torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return torch.mean(log_exp_sum - target_logits)
