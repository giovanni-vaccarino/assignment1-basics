import torch
from cs336_basics.scripts.review.softmax import softmax

def decode(tokenizer,
           model: torch.nn.Module, 
           prompts: torch.Tensor,
           temperature: float = 1.0,
           max_tokens: int = 256,
           top_p: float | None = None
    ):
    # checks on temperature, max_tokens, max_p ...
    generated_tokens = 0
    end_of_text_token = tokenizer.encode("<|endoftext|>")[0]
    token_ids = tokenizer.encode(prompts)

    while generated_tokens <= max_tokens:
        logits = model(token_ids) # returns shape of x -> (..., seq_len, vocab_size)
        output = softmax(logits[..., -1, :] / temperature, -1) # (..., vocab_size)
        if top_p is not None:
            output = apply_top_p(output, top_p)
        else:
            sampled = torch.multinomial(output, 1) # (..., 1)
        token_ids = torch.cat((token_ids, sampled), -1)

        

def get_end_of_text_token():
    pass

def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, indexes = torch.sort(probs, descending=True)
    cum_sum = torch.cumsum(sorted_probs, dim=-1) # [0.5, 0.6, 0.9, ...]
    mask = (cum_sum - sorted_probs) < top_p
    updated_probs = softmax(cum_sum)




x = torch.tensor([[20, 5]])
logits = torch.tensor([
    [0.2, 0.6, 0.2],
    [1.2, 5.1, 0.2]
]).unsqueeze(0) # (1, 2, 3)
temperature = 1.0
output = softmax(logits[..., -1, :] / temperature, -1) # (1, 3)
sampled = torch.multinomial(output, 1) # (1, 1)
x = torch.cat((x, sampled), -1)
print(x)