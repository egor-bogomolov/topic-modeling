import os
import re
import numpy as np

from pathlib import Path
from joblib import Parallel, cpu_count, delayed
from typing import *

from scripts.data_processing.clustering_model import ClusteringModel
from scripts.data_processing.model_folder import ModelFolder
from scripts.data_processing.data_loading import *

T = TypeVar('T')


class ModelRepo:

    def __init__(self, data_folder: Path, n_jobs=cpu_count()):
        self.data_folder = data_folder
        self.repos_data_folder = data_folder / 'bow'
        self.repos_data_files = [
            data_folder / 'bow' / f
            for f in os.listdir(self.repos_data_folder)
            if not f.endswith('.lzo') and not f.endswith('.lzo.index') and f != '.gitkeep'
        ]

        self.repos_names_file = data_folder / 'stats' / 'repos_names.txt'
        self.names = None
        self.rev_index = None

        self.n_jobs = n_jobs
        self.__pattern_repo = "\A\(('|\")(.*)('|\"),\s\["
        self.__pattern_words = "\(('|\")(\w+)('|\"), (\d+)\)"

        self.cluster_embeddings = {}

    def __process_repos_chunk(self, repos_chunk: List[str], f: Callable[[str, List[str], List[int]], T]) -> List[T]:
        result = [None] * len(repos_chunk)
        for i, repo in enumerate(repos_chunk):
            repo = repo.strip()
            repo_name = re.findall(self.__pattern_repo, repo)[0]
            word_counts = re.findall(self.__pattern_words, repo)
            words = [word for _, word, _, _ in word_counts]
            counts = [int(count) for _, _, _, count in word_counts]
            result[i] = f(repo_name, words, counts)
        return result

    def __aggregate(self, f: Callable[[str, List[str], List[int]], T]) -> List[T]:
        """Apply function to all repositories in parallel.
        """
        results = []
        with Parallel(self.n_jobs) as pool:
            for file in self.repos_data_files:
                print(f'Processing repos in {file.name}')
                repos = file.open('r').readlines()
                chunk_size = (len(repos) + self.n_jobs - 1) // self.n_jobs
                print(f'Found {len(repos)} repos, chunk size = {chunk_size}')
                results += pool([
                    delayed(self.__process_repos_chunk)(repos[start:start + chunk_size], f)
                    for start in range(0, len(repos), chunk_size)
                ])
        return results

    def repos_names(self) -> List[str]:
        if not self.repos_names_file.exists():
            self.names = self.__aggregate(lambda names, words, counts: names)
            self.repos_names_file.open('w').write('\n'.join(self.names))
        if self.names is None:
            self.names = [line.strip() for line in self.repos_names_file.open('r').readlines()]
        return self.names

    def repos_rev_index(self) -> Dict[str, int]:
        if self.rev_index is None:
            self.rev_index = {name: i for i, name in enumerate(self.repos_names())}
        return self.rev_index

    def __repos_cluster_embeddings(self, model_folder: ModelFolder, clustering_model: ClusteringModel) -> np.ndarray:
        n_clusters = clustering_model.n_clusters()
        token_rev_index = model_folder.tokens_rev_index()
        clusters = clustering_model.get_clusters()

        def repo_to_cluster_embedding(name: str, words: List[str], counts: List[int]) -> np.ndarray:
            embedding = np.zeros(n_clusters, dtype=np.int32)
            for word, count in zip(words, counts):
                embedding[clusters[token_rev_index[word]]] += count
            return embedding

        cluster_embeddings = np.array(self.__aggregate(repo_to_cluster_embedding))
        return cluster_embeddings

    def __cluster_embeddings_file(self, clustering_model_name: str):
        folder = self.data_folder / 'stats' / clustering_model_name
        if not folder.exists():
            folder.mkdir()
        return folder / 'repo_vectors.npy'

    def repos_cluster_embeddings(self, model_folder: ModelFolder, clustering_model_name: str) -> np.ndarray:
        clustering_embeddings_file = self.__cluster_embeddings_file(clustering_model_name)
        if not clustering_embeddings_file.exists():
            clustering_model = ClusteringModel(Path(clustering_model_name), model_folder)
            cluster_embeddings = self.__repos_cluster_embeddings(model_folder, clustering_model)
            save_vectors(self.__cluster_embeddings_file(clustering_model_name), cluster_embeddings)
            self.cluster_embeddings[clustering_model_name] = cluster_embeddings
        if clustering_model_name not in self.cluster_embeddings:
            self.cluster_embeddings[clustering_model_name] = \
                read_vectors(self.__cluster_embeddings_file(clustering_model_name))
        return self.cluster_embeddings[clustering_model_name]
