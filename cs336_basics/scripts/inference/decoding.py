import numpy as np
import torch
from cs336_basics.scripts.transformer import Transformer
from cs336_basics.scripts.tokenizer import Tokenizer
from cs336_basics.scripts.softmax import softmax

def generate(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    device: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    generated = []
    eot_id = tokenizer.encode("<|endoftext|>")[0]
    ids = tokenizer.encode(prompt)
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor(ids, device=device).unsqueeze(0)
            logits = model(x)[0, -1]
            probs = softmax(logits / temperature, dim=0)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                mask = (cumsum - sorted_probs) < top_p
                sorted_probs = sorted_probs[mask]
                sorted_indices = sorted_indices[mask]
                probs = sorted_probs / sorted_probs.sum()
                token = sorted_indices[torch.multinomial(probs, 1)].item()
            else:
                token = torch.multinomial(probs, 1).item()

            ids.append(token)
            generated.append(token)
            if token == eot_id:
                break
    return tokenizer.decode(generated)

# x = torch.tensor([0.3, 0.1, 0.1, 0.5])
# p, idxs = torch.sort(x, descending=True)
# cumsum = torch.cumsum(p, dim=0)
# mask = (cumsum - p) < 0.6
# sorted_probs = p[mask]
# sorted_indices = idxs[mask]
# print(sorted_indices)