import numpy as np
import torch
import time
from cs336_basics.scripts.transformer import Transformer
from cs336_basics.scripts.training.adamw import AdamW
from cs336_basics.scripts.training.cross_entropy import cross_entropy
from cs336_basics.scripts.training.data_loading import data_loader
from cs336_basics.scripts.training.lr_scheduler import cosine_lr_scheduler
from cs336_basics.scripts.training.gradient_clipping import gradient_clipping
from cs336_basics.scripts.training.checkpointing import save_checkpoint

def training(x_train: np.array,
             x_val: np.array,
             vocab_size,
            context_length,
            num_layers,
            d_model,
            num_heads,
            d_ff,
            eps,
            theta,
            is_rope,
            lr,
            betas,
            eps_adam,
            weight_decay,
            lr_max,
            lr_min,
            T_w,
            T_c,
            max_norm_gc,
            total_steps,
            num_val_batches,
            eval_interval,
            checkpoint_interval,
            out,
            batch_size,
            batch_context_length,
            device):
    
    model = Transformer(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        eps,
        theta,
        is_rope,
        device=device
    )
    optimizer = AdamW(model.parameters(), lr, betas, eps_adam, weight_decay)

    start_time = time.time()
    for step in range(total_steps):
        lr = cosine_lr_scheduler(step, lr_max, lr_min, T_w, T_c)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data_loader(x_train, batch_size, batch_context_length, device)
        # 0. Reset grad
        model.zero_grad()

        # 1. Forward pass
        y_hat = model(x)

        # 2. Loss
        loss = cross_entropy(y_hat, y)

        # 3. Backward pass
        loss.backward()

        # 3.1 Gradient clipping
        gradient_clipping(model.parameters(), max_norm_gc)

        # 4. Optimizer step
        optimizer.step()

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(num_val_batches):  # e.g. 20 batches
                    x_v, y_v = data_loader(x_val, batch_size, batch_context_length, device)
                    val_logits = model(x_v)
                    val_losses.append(cross_entropy(val_logits, y_v).item())
                val_loss = np.mean(val_losses)
            model.train()
            passed_time = time.time() - start_time
            print(f"Step {step} | train loss: {loss.item():.4f} | val loss: {val_loss:.4f} | time: {passed_time}")

        if step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, out)
        
