from einops import reduce
import torch

def cross_entropy(logits, targets):
    """logits have a shape (..., seq_len, vocab_size)"""
    logits = logits - reduce(logits, "... vocab_size -> ... 1", "max")
    # l - log(sum(exp(logit)))
    log_sum = torch.log(reduce(torch.exp(logits), "... vocab_size -> ... 1", "sum"))
    log_logits = logits - log_sum
    
    loss = torch.gather(log_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # the result without squeeze has the same result of index targets.unsqueeze(-1)
    # thus we squeeze
    return -torch.mean(loss)


logits =  torch.tensor([[10.0, 1.0, 0.5], [10.0, 1.0, 0.5]])
targets = torch.tensor([1, 0])

print(cross_entropy(logits, targets))

logits = torch.tensor([[10.0, 1.0, 0.5], [0.5, 1.0, 10.0]])
targets = torch.tensor([0, 2])  # first should be low loss, second too
print(cross_entropy(logits, targets)) 