"""
Trains BPE tokenizer on TinyStories and encodes train/val splits to numpy arrays.
Outputs saved to data/tokenizer/:
  - vocab.pkl        : dict[int, bytes]
  - merges.pkl       : list[tuple[bytes, bytes]]
  - tinystories_train.npy  : uint16 token IDs
  - tinystories_val.npy    : uint16 token IDs
"""

import array
import os
import pickle

import numpy as np

from cs336_basics.scripts.tokenizer import Tokenizer
from cs336_basics.scripts.train_tok import train_tokenizer

TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VAL_PATH = "data/tiny_stories_val.txt"
TOKENIZER_DIR = "data/tokenizer"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]


def encode_to_npy(tokenizer: Tokenizer, text_path: str, out_path: str) -> None:
    # array.array('H') stores uint16 natively — much cheaper than a Python int list
    ids: array.array = array.array("H")
    with open(text_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
    arr = np.frombuffer(ids, dtype=np.uint16)
    np.save(out_path, arr)
    print(f"  saved {len(arr):,} tokens → {out_path}")


def main() -> None:
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    # ── 1. Train tokenizer ────────────────────────────────────────────────────
    print(f"Training BPE tokenizer (vocab_size={VOCAB_SIZE}) on {TRAIN_PATH} …")
    vocab, merges = train_tokenizer(TRAIN_PATH, VOCAB_SIZE, SPECIAL_TOKENS)

    vocab_path = os.path.join(TOKENIZER_DIR, "vocab.pkl")
    merges_path = os.path.join(TOKENIZER_DIR, "merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"  saved vocab  → {vocab_path}  ({len(vocab):,} tokens)")
    print(f"  saved merges → {merges_path}  ({len(merges):,} merges)")

    # ── 2. Encode train + val splits ──────────────────────────────────────────
    tokenizer = Tokenizer(vocab, merges, special_tokens=SPECIAL_TOKENS)

    print(f"Encoding train split ({TRAIN_PATH}) …")
    encode_to_npy(tokenizer, TRAIN_PATH, os.path.join(TOKENIZER_DIR, "tinystories_train.npy"))

    print(f"Encoding val split ({VAL_PATH}) …")
    encode_to_npy(tokenizer, VAL_PATH, os.path.join(TOKENIZER_DIR, "tinystories_val.npy"))

    print("Done.")


if __name__ == "__main__":
    main()
