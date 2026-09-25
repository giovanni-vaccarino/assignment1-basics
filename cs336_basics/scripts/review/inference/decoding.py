import torch
from cs336_basics.scripts.review.softmax import softmax

@torch.no_grad()
def generate(tokenizer,
           model: torch.nn.Module, 
           prompt: str,
           temperature: float = 1.0,
           max_tokens: int = 256,
           top_p: float | None = None,
           device: torch.device | None = None,
    ):
    # checks on temperature, max_tokens, max_p ...
    generated_tokens = 0
    end_of_text_token = tokenizer.encode("<|endoftext|>")[0]
    token_ids = tokenizer.encode(prompt)
    len_prompt_ids = len(token_ids)

    model.eval()
    while generated_tokens < max_tokens:
        # can also stop the generation
        max_context_length = model.max_context_length
        if len(token_ids) > max_context_length:
            logits = model(torch.tensor(token_ids[(len(token_ids) - max_context_length):], device=device).unsqueeze(0))
        else:
            logits = model(torch.tensor(token_ids, device=device).unsqueeze(0)) # returns shape of x -> (..., seq_len, vocab_size)
        output = softmax(logits[..., -1, :] / temperature, -1) # (..., vocab_size)
        if top_p is not None:
            sampled = apply_top_p(output, top_p)
        else:
            sampled = torch.multinomial(output, 1) # (..., 1)
        if sampled == end_of_text_token:
            break
        token_ids.append(sampled.item())
        generated_tokens += 1

    return tokenizer.decode(token_ids[len_prompt_ids:])

def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indexes = torch.sort(probs, descending=True)
    cum_sum = torch.cumsum(sorted_probs, dim=-1) # [0.5, 0.6, 0.8, 0.9...]
    mask =  (cum_sum - top_p) < sorted_probs
    sorted_probs = sorted_probs[mask]
    sorted_indexes = sorted_indexes[mask]
    sorted_probs = sorted_probs / sorted_probs.sum()
    return sorted_indexes[torch.multinomial(sorted_probs, 1)]

# sorted_probs = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1])
# cum_sum = torch.tensor([0.5, 0.7, 0.8, 0.9, 1.0])
# mask = cum_sum <= 0.8
# sorted_probs = sorted_probs[mask]
# normalized_probs = sorted_probs / sorted_probs.sum()
# print(normalized_probs)
# sorted_probs = softmax(sorted_probs, 0)
# print(sorted_probs)
# sampled = torch.multinomial(sorted_probs, 1)

# x = torch.tensor([[20, 5]])
# logits = torch.tensor([
#     [0.2, 0.6, 0.2],
#     [1.2, 5.1, 0.2]
# ]).unsqueeze(0) # (1, 2, 3)
# temperature = 1.0
# output = softmax(logits[..., -1, :] / temperature, -1) # (1, 3)
# sampled = torch.multinomial(output, 1) # (1, 1)
# x = torch.cat((x, sampled), -1)
# print(x)