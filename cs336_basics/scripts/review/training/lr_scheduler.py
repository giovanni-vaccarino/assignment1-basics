import math

def cosine_annealing_lr_scheduler(t: int, max_lr: float, min_lr: float, T_w: int, T_c: int) -> float:
    # T_w -> # warmup iterations
    # T_c -> final iteration of cosine annealing
    # returns the learning rate at time t

    if t < T_w:
        return max_lr * (t / T_w)

    if t > T_c:
        return min_lr

    return min_lr + 0.5*(1 + math.cos(((t - T_w) / (T_c - T_w)) * math.pi))*(max_lr - min_lr)
