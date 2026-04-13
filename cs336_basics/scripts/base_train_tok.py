import regex as re

def train_tokenizer(corpus: str, num_merges: int):
    """Basic and inefficient way of training a tokenizer"""

    # Initialize the vocabulary
    initial_vocab = {i:bytes([i]) for i in range(256)}
    # print(len(initial_vocab)) -> 256 

    # Pre-tokenize the corpus
    regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_corpus = re.finditer(regex, corpus)

    # Useful to have a mapping of unique pre-tokenized tokens and occs
    unique_pre_tokens = {}
    for pre_token in pre_tokenized_corpus:
        bytes_tuple = tuple(bytes([b]) for b in (pre_token.group()).encode('utf-8'))
        unique_pre_tokens[bytes_tuple] = unique_pre_tokens.get(bytes_tuple, 0) + 1
    merges = []
    # Compute BPE merges
    for _ in range(num_merges):
        # 1. Identify the most frequent pair
        pair_occs = {} # key: tuple of the pair ; value: occurrencies of that pair

        for pre_token in unique_pre_tokens:
            for i_c in range(len(pre_token) - 1):
                pair_occs[(pre_token[i_c], pre_token[i_c + 1])] = pair_occs.get((pre_token[i_c], pre_token[i_c + 1]), 0) + unique_pre_tokens[pre_token]
        
        max_pair = max(pair_occs, key=lambda x: (pair_occs[x], x))

        # 2. Add the pair to the merges, and add into the vocabulary

        merges.append(max_pair)
        initial_vocab[len(initial_vocab)] = max_pair[0] + max_pair[1]

        # 3. Merge the pair
        updated_unique_pre_tokens = {}
        for pre_token, occs in unique_pre_tokens.items():
            updated_tuple = []
            i = 0
            while i < len(pre_token) - 1:
                if max_pair[0] == pre_token[i] and max_pair[1] == pre_token[i + 1]:
                    updated_tuple.append(max_pair[0] + max_pair[1])
                    i += 2
                else:
                    updated_tuple.append(pre_token[i])
                    i += 1

            if i < len(pre_token):
                updated_tuple.append(pre_token[i])

            updated_unique_pre_tokens[tuple(updated_tuple)] = occs
        
        print(updated_unique_pre_tokens)
        unique_pre_tokens = updated_unique_pre_tokens
    
    return initial_vocab, merges


test_corpus = "the cat is on the table table table on"
num_merges = 4
train_tokenizer(test_corpus, 10)
