from einops import reduce

def gradient_clipping(params, max_norm, eps=1e-6):
    total_norm = 0.0

    for p in params:
        if p.grad is None:
            continue
        total_norm += p.grad.data.norm()**2
    total_norm = total_norm**0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= scale
    
    return params