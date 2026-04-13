import regex as re
from typing import Iterable, Iterator

class Tokenizer:
    """Tokenizer class that supports encoding and decoding"""

    def __init__(
            self,
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]],
            special_tokens: list[str] | None = None
        ):
        self.vocabulary = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocabulary = {v:k for k, v in vocab.items()}
    

    def _apply_merges(self, bytes_seq) -> list[int]:
        ids = []
        for toks in bytes_seq:
            if isinstance(toks, int):
                ids.append(toks)
            else:
                for el1, el2 in self.merges:
                    upd_tok = []
                    i = 0
                    while i < len(toks) - 1:
                        if toks[i] == el1 and toks[i + 1] == el2:
                            upd_tok.append(el1 + el2)
                            i += 2
                        else:
                            upd_tok.append(toks[i])
                            i += 1
                    if i < len(toks):
                        upd_tok.append(toks[i])
                    toks = upd_tok
                ids.extend(self.reverse_vocabulary[t] for t in toks)
        return ids


    def encode(self, text: str) -> list[int]:
        # 1. Pre-tokenize (handle special tokens)
        regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            ordered_special_tokens = sorted(self.special_tokens, reverse=True)
            delimiter = "|".join(re.escape(st) for st in ordered_special_tokens)
            splitted_corpus = re.split(f"({delimiter})", text)
            pre_tokenized = []
            for chunk in splitted_corpus:
                if chunk in self.special_tokens:
                    pre_tokenized.append(self.reverse_vocabulary[chunk.encode("utf-8")])
                else:
                    pre_tokens = re.finditer(regex, chunk)

                    for tok in pre_tokens:
                        bytes_seq = [bytes([c]) for c in tok.group().encode("utf-8")]
                        pre_tokenized.append(bytes_seq)
        else:
            pre_tokenized = []
            pre_tokens = re.finditer(regex, text)

            for tok in pre_tokens:
                bytes_seq = [bytes([c]) for c in tok.group().encode("utf-8")]
                pre_tokenized.append(bytes_seq)

        # 2. Apply merge rules progressively (can be done with multiprocessing)

        return self._apply_merges(pre_tokenized)

    def decode(self, ids: list[int]) -> str:
        all_bytes = b"".join(self.vocabulary[id] for id in ids)

        return all_bytes.decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            ids = self.encode(line)
            for id in ids:
                yield id
