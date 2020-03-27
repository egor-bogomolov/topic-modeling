import numpy as np
import pickle

from typing import *
from pathlib import Path
from tqdm import tqdm

from spherecluster import SphericalKMeans

__all__ = [
    'read_vec_file', 'save_tokens', 'read_tokens', 'save_vectors', 'read_vectors', 'load_clustering_model',
    'save_clusters', 'read_clusters', 'save_reference_tokens', 'save_readable_ref_tokens',
    'read_reference_tokens', 'read_doc_freq', 'read_term_freq', 'read_idf'
]


def __drop_npy(path: str):
    return path.replace('.npy', '')


def read_vec_file(vec_file: Path, n_tokens: int, dim: int) -> Tuple[List, np.ndarray]:
    with vec_file.open('r') as fin:
        fin.readline()
        tokens = [''] * n_tokens
        vectors = np.empty((n_tokens, dim), np.float)

        for i in tqdm(range(n_tokens)):
            items = fin.readline().strip().split()
            tokens[i] = items[0]
            vectors[i, :] = list(map(np.float, items[1:]))

        return tokens, vectors


def save_tokens(tokens_file: Path, tokens: List) -> None:
    tokens_file.open('w').write('\n'.join(tokens))


def read_tokens(tokens_file: Path) -> List:
    tokens = list(map(lambda s: s.strip(), tokens_file.open('r').readlines()))
    return tokens


def save_vectors(vectors_file: Path, vectors: np.ndarray) -> None:
    np.save(__drop_npy(str(vectors_file)), vectors)


def read_vectors(vectors_file: Path) -> np.ndarray:
    vectors = np.load(str(vectors_file), allow_pickle=True)
    return vectors


def load_clustering_model(model_file: Path) -> SphericalKMeans:
    return pickle.load(model_file.open('rb'))


def save_clusters(clusters_file: Path, clusters: np.ndarray) -> None:
    np.save(__drop_npy(str(clusters_file)), clusters)


def read_clusters(clusters_file: Path) -> np.ndarray:
    clusters = np.load(str(clusters_file), allow_pickle=True)
    return clusters


def save_reference_tokens(reference_tokens_file: Path, reference_tokens: List[List]) -> None:
    pickle.dump(reference_tokens, reference_tokens_file.open('wb'))


def save_readable_ref_tokens(
        readable_reference_tokens_path: Path, reference_tokens: List[List], tokens: List, n_tokens: int
) -> None:
    with readable_reference_tokens_path.open('w') as fout:
        for cluster, ref_tokens in enumerate(reference_tokens):
            fout.write('-----------------------------\n')
            fout.write(f'Cluster {cluster}:\n')
            for distance, ind in ref_tokens[:n_tokens]:
                fout.write(f'{ind} | {tokens[ind]} | {distance}\n')
            fout.write('-----------------------------\n')
            fout.write('\n')


def read_reference_tokens(reference_tokens_file: Path) -> List[List]:
    return pickle.load(reference_tokens_file.open('rb'))


def read_doc_freq(doc_freq_file: Path) -> Counter:
    return pickle.load(doc_freq_file.open('rb'))


def read_term_freq(term_freq_file: Path) -> Counter:
    return pickle.load(term_freq_file.open('rb'))


def read_idf(idf_file: Path) -> Counter:
    return pickle.load(idf_file.open('rb'))
