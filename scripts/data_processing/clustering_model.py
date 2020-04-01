import os
import numpy as np

from typing import *
from pathlib import Path

from scipy.spatial.distance import cosine
from tqdm import tqdm

from scripts.data_processing.data_loading import *
from scripts.data_processing.model_folder import ModelFolder
from spherecluster import SphericalKMeans


__all__ = ['ClusteringModel']


class ClusteringModel:

    def __init__(self, folder: Path, model_folder: ModelFolder, min_repo_count: int = 10):
        if not folder.is_dir():
            raise ValueError(f'{folder} is not a directory')

        self.folder = folder
        self.name = self.folder.name
        self.model_folder = model_folder
        self.min_repo_count = min_repo_count

        self.model_file = folder / 'model.pkl'
        self.model = None
        if not self.model_file.exists():
            raise ValueError(f'Model file {str(self.model_file)} does not exist')

        self.clusters_file = folder / 'clusters.npy'
        self.clusters = None
        self.reference_tokens_file = folder / 'reference_tokens.pkl'
        self.readable_tokens_file = folder / 'readable.txt'
        self.reference_tokens = None

        self.__n_clusters = None
        self.__wetc = None

    def get_model(self) -> SphericalKMeans:
        if self.model is None:
            self.model = load_clustering_model(self.model_file)
        return self.model

    def n_clusters(self) -> int:
        if self.__n_clusters is None:
            model = self.get_model()
            self.__n_clusters = model.n_clusters
        return self.__n_clusters

    def __extract_clusters(self) -> None:
        model = self.get_model()
        self.clusters = model.predict(self.model_folder.get_vectors())
        save_clusters(self.clusters_file, self.clusters)

    def get_clusters(self) -> np.ndarray:
        if not self.clusters_file.exists():
            self.__extract_clusters()
        if self.clusters is None:
            self.clusters = read_clusters(self.clusters_file)
        return self.clusters

    def __extract_reference_tokens(self) -> None:
        model = self.get_model()
        clusters = self.get_clusters()
        vectors = self.model_folder.get_vectors()
        tokens = self.model_folder.get_tokens()
        doc_freq = self.model_folder.get_doc_freq()
        n_clusters = self.n_clusters()

        self.reference_tokens = [[] for _ in range(n_clusters)]

        for i, (cluster, vector, token) in enumerate(zip(clusters, vectors, tokens)):
            if doc_freq[token] < self.min_repo_count:
                continue
            self.reference_tokens[cluster].append((cosine(model.cluster_centers_[cluster], vector), i))

        for cluster in range(n_clusters):
            self.reference_tokens[cluster].sort()

        save_reference_tokens(self.reference_tokens_file, self.reference_tokens)
        save_readable_ref_tokens(self.readable_tokens_file, self.reference_tokens, tokens, 30)

    def get_reference_tokens(self) -> List[List]:
        if not self.reference_tokens_file.exists():
            self.__extract_reference_tokens()
        if self.reference_tokens is None:
            self.reference_tokens = read_reference_tokens(self.reference_tokens_file)
        return self.reference_tokens

    def __compute_wetc(self) -> np.ndarray:
        model = self.get_model()
        clusters = self.get_clusters()
        vectors = self.model_folder.get_vectors()
        doc_freq = self.model_folder.get_doc_freq()
        tokens = self.model_folder.get_tokens()
        n_clusters = self.n_clusters()

        wetc = np.zeros(n_clusters)
        counts = np.zeros(n_clusters, np.int32)

        for cluster, vector, token in tqdm(zip(clusters, vectors, tokens), total=len(tokens)):
            if doc_freq[token] >= self.min_repo_count:
                wetc[cluster] += cosine(model.cluster_centers_[cluster], vector)
                counts[cluster] += 1

        for cluster, count in enumerate(counts):
            wetc[cluster] /= count

        return wetc

    def wetc(self) -> np.ndarray:
        if self.__wetc is None:
            self.__wetc = self.__compute_wetc()
        return self.__wetc

    def mean_wetc(self) -> np.float:
        return self.wetc().mean()
