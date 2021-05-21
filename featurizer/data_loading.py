from typing import List, Tuple

import numpy as np


def aggregate_parts(parts):
    res = [parts[0][1:]]
    for i in range(1, len(parts)):
        res.append(parts[i][0].capitalize() + parts[i][1:])
    return ''.join(res)


def string_to_token_parts(subtokens):
    token_parts = []
    for subtoken in subtokens:
        if subtoken.startswith('_'):
            token_parts.append([subtoken])
        else:
            token_parts[-1].append(subtoken)
    return token_parts


def read_vec_file(vec_file: str) -> Tuple[List[str], np.ndarray]:
    with open(vec_file, 'r') as fin:
        n, dim = map(int, fin.readline().split())
        vectors = np.zeros((n, dim))
        tokens = []
        for i, line in enumerate(fin):
            token, *values = line.split()
            tokens.append(token)
            vectors[i, :] = list(map(float, values))

    return tokens, vectors


class PartsVectorizer:

    def __init__(self, vectors_file: str):
        self.subtokens, self.vectors = read_vec_file(vectors_file)
        self.subtoken_to_ind = {
            subtoken: i
            for i, subtoken in enumerate(self.subtokens)
        }
        self.tokens = {}
        self.dim = self.vectors.shape[1]

    def get_vector_from_parts(self, token_parts):
        v = np.zeros(self.dim)
        updated = False
        for token_part in token_parts:
            # print(token_part)
            if token_part in self.subtoken_to_ind:
                v += self.vectors[self.subtoken_to_ind[token_part]]
                updated = True
            # else:
            #     print(f"Missed {token_part}")
        return v, updated

    def add_string(self, subtokens):
        for token_parts in string_to_token_parts(subtokens):
            # print(token_parts)
            full_token = aggregate_parts(token_parts)
            # print(token_parts)
            if full_token not in self.tokens:
                vec, updated = self.get_vector_from_parts(token_parts)
                if updated:
                    self.tokens[full_token] = vec

    def dump(self, tokens_file: str, vectors_file: str):
        vectors = []
        tokens = []
        for tok, vec in self.tokens.items():
            tokens.append(tok)
            vectors.append(vec)
        vectors = np.concatenate(vectors)
        np.save(vectors_file, vectors)
        with open(tokens_file, 'w') as tok_out:
            tok_out.write('\n'.join(tokens))
