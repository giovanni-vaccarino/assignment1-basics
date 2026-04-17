import numpy as np

def cosine_lr_scheduler(t, lr_max, lr_min, T_w, T_c):
    if t < T_w:
        return (t/T_w)*lr_max

    if t > T_c:
        return lr_min
    cos_expression = np.cos(((t-T_w)*np.pi) / (T_c - T_w))

    return lr_min + 0.5*(1 + cos_expression) * (lr_max - lr_min)
