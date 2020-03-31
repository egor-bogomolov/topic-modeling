import os
import numpy as np

from typing import *
from pathlib import Path

from scripts.data_processing.data_loading import *


__all__ = ['ModelFolder']


class ModelFolder:

    def __init__(self, folder: Path, data_folder: Path):
        if not folder.is_dir():
            raise ValueError(f'{folder} is not a directory')

        self.folder = folder
        self.model_bin_file = folder / 'model.bin'
        self.model_bin = None
        self.model_vec_file = folder / 'model.vec'
        self.vectors_file = folder / 'vectors.npy'
        self.vectors = None
        self.tokens_file = folder / 'tokens.txt'
        self.tokens = None
        self.clustering_models_folder = folder / 'clustering_models'
        self.clustering_models = None

        with self.model_vec_file.open('r') as fin:
            self.n_tokens, self.dim = map(int, fin.readline().strip().split())

        print(self.n_tokens, self.dim)

        self.data_folder = data_folder
        self.doc_freq_file = data_folder / 'stats' / 'doc_frequency.pkl'
        self.doc_freq = None
        self.term_freq_file = data_folder / 'stats' / 'term_frequency.pkl'
        self.term_freq = None
        self.idf_file = data_folder / 'stats' / 'idf.pkl'
        self.idf = None

        self.oov = folder / 'oov.txt'

    def __parse_vec_file(self) -> None:
        self.tokens, self.vectors = read_vec_file(self.model_vec_file, self.n_tokens, self.dim)
        save_tokens(self.tokens_file, self.tokens)
        save_vectors(self.vectors_file, self.vectors)

    def get_tokens(self) -> List:
        if not self.tokens_file.exists():
            self.__parse_vec_file()
        if self.tokens is None:
            self.tokens = read_tokens(self.tokens_file)
        return self.tokens

    def get_vectors(self) -> np.ndarray:
        if not self.vectors_file.exists():
            self.__parse_vec_file()
        if self.vectors is None:
            self.vectors = read_vectors(self.vectors_file)
        return self.vectors

    def list_clustering_models(self) -> None:
        print(os.listdir(self.clustering_models_folder))

    def get_clustering_models_folders(self) -> List[Path]:
        return [
            self.clustering_models_folder / f
            for f in os.listdir(self.clustering_models_folder)
        ]

    def get_doc_freq(self) -> Counter:
        if self.doc_freq is None:
            self.doc_freq = read_doc_freq(self.doc_freq_file)
        return self.doc_freq

    def get_term_freq(self) -> Counter:
        if self.term_freq is None:
            self.term_freq = read_term_freq(self.term_freq_file)
        return self.term_freq

    def get_idf(self) -> Counter:
        if self.idf is None:
            self.idf = read_idf(self.idf_file)
        return self.idf

    def save_oov(self) -> None:
        if not self.oov.exists():
            tf = self.get_term_freq()
            tokens = set(self.get_tokens())
            with self.oov.open('w') as fout:
                fout.write('\n'.join(token for token in tf if token not in tokens))
