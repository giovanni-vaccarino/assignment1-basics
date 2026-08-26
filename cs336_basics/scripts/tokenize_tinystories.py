"""
Encode TinyStories train/val splits with the GPT-2 tiktoken tokenizer.
Outputs uint16 NumPy arrays to data/tokenizer/:
  - tinystories_train_gpt2.npy
  - tinystories_val_gpt2.npy
"""

import array
import os

import numpy as np
import tiktoken
from tqdm import tqdm

TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VAL_PATH = "data/tiny_stories_val.txt"
OUT_DIR = "data/tokenizer"

# GPT-2 max token id is 50256, fits safely in uint16 (max 65535)
assert tiktoken.get_encoding("gpt2").n_vocab <= 65535


def encode_file(enc: tiktoken.Encoding, text_path: str, out_path: str) -> None:
    ids: array.array = array.array("H")  # uint16
    with open(text_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=os.path.basename(text_path), unit=" lines"):
            ids.extend(enc.encode_ordinary(line))
    arr = np.frombuffer(ids, dtype=np.uint16)
    np.save(out_path, arr)
    print(f"  saved {len(arr):,} tokens → {out_path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    print(f"Using GPT-2 tokenizer (vocab size: {enc.n_vocab})")

    print(f"\nEncoding train split: {TRAIN_PATH}")
    encode_file(enc, TRAIN_PATH, os.path.join(OUT_DIR, "tinystories_train_gpt2.npy"))

    print(f"\nEncoding val split: {VAL_PATH}")
    encode_file(enc, VAL_PATH, os.path.join(OUT_DIR, "tinystories_val_gpt2.npy"))

    print("\nDone.")


if __name__ == "__main__":
    main()
