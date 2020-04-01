import os
import re

import numpy as np

from pathlib import Path
from joblib import Parallel, cpu_count, delayed
from typing import *

from tqdm import tqdm

from scripts.data_processing.clustering_model import ClusteringModel
from scripts.data_processing.model_folder import ModelFolder
from scripts.data_processing.data_loading import *


class RepoWordCount(NamedTuple):
    name: str
    word: str
    count: int


class ModelRepo:

    def __init__(self, data_folder: Path, n_jobs=cpu_count()):
        self.data_folder = data_folder
        self.repos_data_folder = data_folder / 'bow'
        self.repos_data_files = [
            data_folder / 'bow' / f
            for f in os.listdir(self.repos_data_folder)
            if not f.endswith('.lzo') and not f.endswith('.lzo.index') and f != '.gitkeep'
        ]

        self.repo_word_counts = None

        self.repos_names_file = data_folder / 'stats' / 'repos_names.txt'
        self.names = None
        self.rev_index = None

        self.n_jobs = n_jobs
        self.__pattern_repo = "\A\(('|\")(.*)('|\"),\s\["
        self.__pattern_words = "\(('|\")(\w+)('|\"), (\d+)\)"

        self.cluster_embeddings = {}

    @staticmethod
    def __process_repos_chunk(repos_chunk: List[str], pattern_repo, pattern_words) -> List[RepoWordCount]:
        result = []
        for i, repo in enumerate(repos_chunk):
            repo = repo.strip()
            repo_name = re.findall(pattern_repo, repo)[0][1]
            word_counts = re.findall(pattern_words, repo)
            for _, word, _, count in word_counts:
                result.append(RepoWordCount(repo_name, word, int(count)))
        return result

    def __extract_repo_data(self) -> List[RepoWordCount]:
        print('Extracting repo data')
        if self.repo_word_counts is None:
            results = []
            with Parallel(self.n_jobs) as pool:
                n_files = len(self.repos_data_files)
                file_batch_size = 10
                for i in range(0, n_files, file_batch_size):
                    file_batch = self.repos_data_files[i:i + file_batch_size]
                    print(f'Loading {len(file_batch)} files')
                    file_data = [
                        file.open('r').readlines()
                        for file in tqdm(file_batch)
                    ]
                    total_len = sum(map(len, file_data))
                    chunk_size = total_len // (self.n_jobs * n_files)
                    print(f'Found {total_len} repos, chunk size = {chunk_size}')
                    chunks = [
                        repos[start:start + chunk_size]
                        for repos in file_data
                        for start in range(0, len(repos), chunk_size)
                    ]
                    print(f'Processing {len(chunks)} chunks')

                    chunk_results = pool([
                        delayed(self.__process_repos_chunk)(chunk, self.__pattern_repo, self.__pattern_words)
                        for chunk in chunks
                    ])

                for chunk_result in chunk_results:
                    results += chunk_result

            self.repo_word_counts = results
        print('Repo word counts extracted')
        return self.repo_word_counts

    def repos_names(self) -> List[str]:
        if not self.repos_names_file.exists():
            self.names = list(set(rwc.name for rwc in self.__extract_repo_data()))
            self.repos_names_file.open('w').write('\n'.join(self.names))
        if self.names is None:
            self.names = [line.strip() for line in self.repos_names_file.open('r').readlines()]
        return self.names

    def repos_rev_index(self) -> Dict[str, int]:
        if self.rev_index is None:
            self.rev_index = {name: i for i, name in enumerate(self.repos_names())}
        return self.rev_index

    def __repos_cluster_embeddings(self, model_folder: ModelFolder, clustering_model: ClusteringModel) -> np.ndarray:
        token_rev_index = model_folder.tokens_rev_index()
        repos_rev_index = self.repos_rev_index()
        n_repos = len(repos_rev_index)
        clusters = clustering_model.get_clusters()
        n_clusters = clustering_model.n_clusters()

        repo_word_counts = self.__extract_repo_data()
        print('Sorting repo word counts')
        repo_word_counts.sort(key=lambda rwc: rwc.word)

        cluster_embeddings = np.zeros((n_repos, n_clusters), dtype=np.int32)
        last_word = None
        cur_cluster = -1
        for rwc in repo_word_counts:
            if rwc.word != last_word:
                cur_cluster = clusters[token_rev_index[rwc.word]]
                last_word = rwc.word
            repo_index = repos_rev_index[rwc.name]
            cluster_embeddings[repo_index][cur_cluster] += rwc.count

        return cluster_embeddings

    @staticmethod
    def __cluster_embeddings_file(clustering_model: ClusteringModel):
        return clustering_model.folder / 'repos_cluster_embeddings.npy'

    def repos_cluster_embeddings(self, model_folder: ModelFolder, clustering_model: ClusteringModel) -> np.ndarray:
        clustering_embeddings_file = self.__cluster_embeddings_file(clustering_model)
        if not clustering_embeddings_file.exists():
            cluster_embeddings = self.__repos_cluster_embeddings(model_folder, clustering_model)
            self.cluster_embeddings[clustering_model.name] = cluster_embeddings
            save_vectors(self.__cluster_embeddings_file(clustering_model), cluster_embeddings)
        if clustering_model.name not in self.cluster_embeddings:
            self.cluster_embeddings[clustering_model.name] = \
                read_vectors(self.__cluster_embeddings_file(clustering_model.name))
        return self.cluster_embeddings[clustering_model.name]
