"""This module handles the training loop.

It tokenizes the train/val text (caching token ids as uint16 .npy files so it only
happens once), trains the Transformer and logs everything to a log file.

Example:
    uv run python -m cs336_basics.scripts.review.training.training \
        --train_text data/TinyStoriesV2-GPT4-train.txt \
        --val_text data/tiny_stories_val.txt \
        --tokenizer gpt2 --vocab_size 50257 \
        --context_length 256 --num_layers 4 --d_model 512 --num_heads 16 --d_ff 1344 \
        --batch_size 32 --total_steps 5000 --out_dir runs/tinystories
"""
import argparse
import array
import json
import logging
import os
import pickle
import time

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.scripts.review.transformer import Transformer
from cs336_basics.scripts.review.training.adamw import AdamW
from cs336_basics.scripts.review.training.cross_entropy import cross_entropy
from cs336_basics.scripts.review.training.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.scripts.review.training.data_loading import data_loader
from cs336_basics.scripts.review.training.gradient_clipping import gradient_clipping
from cs336_basics.scripts.review.training.lr_scheduler import cosine_annealing_lr_scheduler

logger = logging.getLogger("training")

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


# ── Logging ──────────────────────────────────────────────────────────────────
def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    logger.setLevel(logging.INFO)
    for handler in (logging.FileHandler(log_file), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def log_metrics(**metrics) -> None:
    # One JSON object per line after "METRICS " -> easy to grep/parse into loss curves
    logger.info("METRICS %s", json.dumps(metrics))


# ── Tokenization ─────────────────────────────────────────────────────────────
def get_encode_fn(args):
    """Returns (encode_fn, tokenizer_vocab_size). encode_fn: str -> list[int]."""
    if args.tokenizer == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        special = set(args.special_tokens)
        return (lambda text: enc.encode(text, allowed_special=special)), enc.n_vocab

    from cs336_basics.scripts.tokenizer import Tokenizer
    if os.path.exists(args.vocab_path) and os.path.exists(args.merges_path):
        with open(args.vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(args.merges_path, "rb") as f:
            merges = pickle.load(f)
        logger.info("Loaded BPE tokenizer from %s / %s", args.vocab_path, args.merges_path)
    else:
        from cs336_basics.scripts.train_tok import train_tokenizer
        logger.info("Training BPE tokenizer (vocab_size=%d) on %s", args.vocab_size, args.train_text)
        vocab, merges = train_tokenizer(args.train_text, args.vocab_size, args.special_tokens)
        os.makedirs(os.path.dirname(args.vocab_path) or ".", exist_ok=True)
        with open(args.vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        with open(args.merges_path, "wb") as f:
            pickle.dump(merges, f)
        logger.info("Saved BPE tokenizer to %s / %s", args.vocab_path, args.merges_path)

    tokenizer = Tokenizer(vocab, merges, special_tokens=args.special_tokens)
    return tokenizer.encode, len(vocab)


def encode_to_npy(encode_fn, text_path: str, out_path: str) -> None:
    ids = array.array("H")  # uint16, much cheaper than a list of Python ints
    with open(text_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"tokenizing {os.path.basename(text_path)}", unit=" lines"):
            ids.extend(encode_fn(line))
    arr = np.frombuffer(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.save(out_path, arr)
    logger.info("Saved %s tokens -> %s", f"{len(arr):,}", out_path)


def load_tokens(args) -> tuple[np.ndarray, np.ndarray]:
    """Tokenizes train/val once (cached as .npy) and memory-maps the token ids."""
    paths = {
        split: os.path.join(args.data_dir, f"{os.path.splitext(os.path.basename(text))[0]}_{args.tokenizer}.npy")
        for split, text in (("train", args.train_text), ("val", args.val_text))
    }
    missing = [s for s, p in paths.items() if not os.path.exists(p)]
    if missing:
        encode_fn, tok_vocab_size = get_encode_fn(args)
        if tok_vocab_size > np.iinfo(np.uint16).max + 1:
            raise ValueError(f"Tokenizer vocab ({tok_vocab_size}) does not fit in uint16")
        if tok_vocab_size > args.vocab_size:
            raise ValueError(f"--vocab_size {args.vocab_size} < tokenizer vocab size {tok_vocab_size}")
        for split in missing:
            text = args.train_text if split == "train" else args.val_text
            encode_to_npy(encode_fn, text, paths[split])

    train = np.load(paths["train"], mmap_mode="r")
    val = np.load(paths["val"], mmap_mode="r")
    logger.info("Train tokens: %s | Val tokens: %s", f"{len(train):,}", f"{len(val):,}")

    max_id = int(max(train.max(), val.max()))
    if max_id >= args.vocab_size:
        raise ValueError(f"Found token id {max_id} >= --vocab_size {args.vocab_size}")
    return train, val


# ── Training ─────────────────────────────────────────────────────────────────
def get_batch(tokens: np.ndarray, args) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = data_loader(tokens, args.batch_size, args.context_length, args.device)
    # token ids are stored as uint16; indexing/gather need int64
    return x.long(), y.long()


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_tokens: np.ndarray, args) -> float:
    model.eval()
    losses = []
    for _ in range(args.num_val_batches):
        x, y = get_batch(val_tokens, args)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    return float(np.mean(losses))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.log_file or os.path.join(args.out_dir, "train.log"))
    logger.info("Config: %s", json.dumps(vars(args)))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 0. Data
    train_tokens, val_tokens = load_tokens(args)

    # 1. Init model and optimizer
    model = Transformer(
        args.vocab_size,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.eps,
        args.theta,
        device=args.device,
        dtype=DTYPES[args.dtype]
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{num_params:,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=[args.beta1, args.beta2],
        weight_decay=args.weight_decay,
        eps=args.eps_adam
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer) + 1
        logger.info("Resumed from %s at step %d", args.resume, start_step)

    warmup_steps = args.warmup_steps
    cosine_steps = args.cosine_steps or args.total_steps
    tokens_per_step = args.batch_size * args.context_length

    model.train()
    start_time = time.time()
    last_log_time = start_time
    for t in range(start_step, args.total_steps):
        # a) lr schedule
        lr = cosine_annealing_lr_scheduler(t, args.lr_max, args.lr_min, warmup_steps, cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # b) sample batch
        x, y = get_batch(train_tokens, args)

        # c) zero_grad
        optimizer.zero_grad()

        # d) forward
        y_hat = model(x)

        # e) Loss + backward
        loss = cross_entropy(y_hat, y)
        loss.backward()

        # f) clip + optim
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        # g) logging
        if t % args.log_interval == 0:
            now = time.time()
            steps_done = args.log_interval if t > start_step else 1
            log_metrics(
                step=t,
                train_loss=round(loss.item(), 4),
                lr=lr,
                tokens_per_sec=round(steps_done * tokens_per_step / max(now - last_log_time, 1e-9)),
                elapsed_sec=round(now - start_time, 1),
            )
            last_log_time = now

        # h) eval
        if t % args.eval_interval == 0 or t == args.total_steps - 1:
            val_loss = evaluate(model, val_tokens, args)
            log_metrics(step=t, val_loss=round(val_loss, 4), elapsed_sec=round(time.time() - start_time, 1))

        # i) save checkpoint
        if t > start_step and t % args.checkpoint_interval == 0:
            path = os.path.join(args.out_dir, f"ckpt_{t}.pt")
            save_checkpoint(model, optimizer, t, path)
            logger.info("Saved checkpoint -> %s", path)

    final_path = os.path.join(args.out_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.total_steps - 1, final_path)
    logger.info("Training done in %.1fs. Final checkpoint -> %s", time.time() - start_time, final_path)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    # Data / tokenizer
    parser.add_argument("--train_text", type=str, default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--val_text", type=str, default="data/tiny_stories_val.txt")
    parser.add_argument("--data_dir", type=str, default="data/tokenizer", help="Where tokenized .npy files are cached")
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "gpt2"], default="bpe",
                        help="bpe: your own tokenizer (trained if vocab/merges are missing), gpt2: tiktoken")
    parser.add_argument("--vocab_path", type=str, default="data/tokenizer/vocab.pkl")
    parser.add_argument("--merges_path", type=str, default="data/tokenizer/merges.pkl")
    parser.add_argument("--special_tokens", type=str, nargs="*", default=["<|endoftext|>"])

    # Model
    parser.add_argument("--vocab_size", type=int, default=10_000, help="Vocab size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--num_layers", type=int, default=4, help="Num layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=16, help="Num Heads")
    parser.add_argument("--d_ff", type=int, default=1344, help="Dimension FFN")
    parser.add_argument("--eps", type=float, default=1e-5, help="RMSNorm epsilon")
    parser.add_argument("--theta", type=float, default=10_000.0, help="RoPE theta")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--dtype", type=str, choices=list(DTYPES), default="float32")

    # Optimizer / schedule
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--cosine_steps", type=int, default=None, help="End of cosine decay (default: total_steps)")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps_adam", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training loop
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--num_val_batches", type=int, default=20)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="runs/default")
    parser.add_argument("--log_file", type=str, default=None, help="Default: <out_dir>/train.log")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(args)
