import os
import sys
from typing import BinaryIO
from multiprocessing import Pool
import regex as re
import cProfile
import pstats
from tqdm import tqdm

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize(corpus: str, special_tokens: list[str]) -> dict[tuple, int]:
    """Pre-tokenizes a corpus"""
    regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 1. First split on special tokens
    delim = "|".join(re.escape(tok) for tok in special_tokens)
    corpus_splits = re.split(delim, corpus)

    # 2. Apply the tokenization to each piece separately
    unique_pre_tokens = {}
    for split in corpus_splits:
        pre_tokenized_corpus = re.finditer(regex, split)

        # Useful to have a mapping of unique pre-tokenized tokens and occs
        for pre_token in pre_tokenized_corpus:
            bytes_tuple = tuple(bytes([b]) for b in (pre_token.group()).encode('utf-8'))
            unique_pre_tokens[bytes_tuple] = unique_pre_tokens.get(bytes_tuple, 0) + 1
        
    return unique_pre_tokens


def train_tokenizer(input_path:str, vocab_size: int, special_tokens: list[str]):
    """
    Given an input path to a corpus, it trains a BPE tokenizer

    Returns:
        - vocab: dict[int, bytes] The tokenizer vocabulary
        - merges: list[tuple[bytes, bytes]] The list of merges applied. Has to be ordered by creation
    """
    num_workers = 5
    num_merges = vocab_size - 256 - len(special_tokens)

    # 1. Initialize vocab with also special tokens
    vocab = {i:bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    
    # 2. Extract the corpus from input path file
    with open(input_path, "rb") as f:
        data = f.read()

    # 3. Pre-tokenize (in parallel)    

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, special_tokens[0].encode("utf-8"))
    
    # boundaries have to be used as byte offsets like below
    # print(repr(data[boundaries[1]:boundaries[1]+30]))
    chunks = [data[boundaries[i]:boundaries[i+1]].decode("utf-8") for i in range(len(boundaries) - 1)]
    args = [(chunk, special_tokens) for chunk in chunks]

    with Pool(num_workers) as p:
        result = p.starmap(pre_tokenize, args)
    pre_tokens = {}
    for dic in result:
        for k, v in dic.items():
            pre_tokens[k] = pre_tokens.get(k, 0) + v
    print(f"Pre-tokenization done: {len(pre_tokens):,} unique pre-tokens", flush=True)

    pair_occs = {} # key: tuple of the pair ; value: occurrencies of that pair

    for pre_token in pre_tokens:
        for i_c in range(len(pre_token) - 1):
            pair_occs[(pre_token[i_c], pre_token[i_c + 1])] = pair_occs.get((pre_token[i_c], pre_token[i_c + 1]), 0) + pre_tokens[pre_token]
    
    merges = []
    # Compute BPE merges
    for _ in tqdm(range(num_merges), desc="BPE merges", unit="merge", file=sys.stdout, dynamic_ncols=False):
        # 1. Identify the most frequent pair   
        max_pair = max(pair_occs, key=lambda x: (pair_occs[x], x))

        # 2. Add the pair to the merges, and add into the vocabulary

        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]

        # 3. Merge the pair
        merged = max_pair[0] + max_pair[1]

        updated_pre_tokens = {}
        for pre_token, occs in pre_tokens.items():
            updated_tuple = []
            i = 0
            while i < len(pre_token) - 1:
                if max_pair[0] == pre_token[i] and max_pair[1] == pre_token[i + 1]:
                    # Update neighbor pairs
                    if i > 0:
                        pair_occs[(pre_token[i-1], max_pair[0])] = pair_occs.get((pre_token[i-1], max_pair[0]), 0) - occs
                        pair_occs[(pre_token[i-1], merged)] = pair_occs.get((pre_token[i-1], merged), 0) + occs
                    if i + 2 < len(pre_token):
                        pair_occs[(max_pair[1], pre_token[i+2])] = pair_occs.get((max_pair[1], pre_token[i+2]), 0) - occs
                        pair_occs[(merged, pre_token[i+2])] = pair_occs.get((merged, pre_token[i+2]), 0) + occs
                    pair_occs[max_pair] = pair_occs.get(max_pair, 0) - occs
                    updated_tuple.append(merged)
                    i += 2
                else:
                    updated_tuple.append(pre_token[i])
                    i += 1

            if i < len(pre_token):
                updated_tuple.append(pre_token[i])

            updated_pre_tokens[tuple(updated_tuple)] = occs
        
        pre_tokens = updated_pre_tokens
    
    return vocab, merges




if __name__ == "__main__":
    with cProfile.Profile() as pr:
        vocab, merges = train_tokenizer("data/tiny_stories_val.txt", 500, ["<|endoftext|>"])
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)  # or SortKey.TIME for self-time
    stats.print_stats(20)  # print top 20 functions