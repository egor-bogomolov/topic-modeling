import os
import pickle
import re
import time
import sys
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
    words: str
    count: np.ndarray


class ModelRepo:

    def __init__(self, data_folder: Path, n_jobs=cpu_count()):
        self.data_folder = data_folder
        self.repos_data_folder = data_folder / 'bow'
        self.repos_data_files = [
            data_folder / 'bow' / f
            for f in os.listdir(self.repos_data_folder)
            if not f.endswith('.lzo') and not f.endswith('.lzo.index') and f != '.gitkeep'
        ]

        self.repo_word_counts_file = data_folder / 'stats' / 'rwc.pkl'
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
            words = '|'.join(wc[1] for wc in word_counts)
            counts = np.array([int(wc[3]) for wc in word_counts], dtype=np.int32)
            result.append(RepoWordCount(repo_name, words, counts))
        return result

    def __extract_from_file_batch(self, file_batch) -> List[RepoWordCount]:
        print(f'Loading {len(file_batch)} files')
        file_data = [
            file.open('r').readlines()
            for file in tqdm(file_batch)
        ]
        total_len = sum(map(len, file_data))
        chunk_size = total_len // (self.n_jobs * len(file_batch))
        print(f'Found {total_len} repos, chunk size = {chunk_size}')
        chunks = [
            repos[start:start + chunk_size]
            for repos in file_data
            for start in range(0, len(repos), chunk_size)
        ]
        print(f'Processing {len(chunks)} chunks')
        start_time = time.time()
        with Parallel(self.n_jobs) as pool:
            chunk_results = pool([
                delayed(self.__process_repos_chunk)(chunk, self.__pattern_repo, self.__pattern_words)
                for chunk in chunks
            ])
        end_time = time.time()
        print(f'Extracted repo word counts in {end_time - start_time} sec')
        results = []
        for chunk_result in chunk_results:
            results += chunk_result
        return results

    def __extract_repo_data(self) -> List[RepoWordCount]:
        if not self.repo_word_counts_file.exists():
            print('Extracting repo data')
            n_files = len(self.repos_data_files)
            file_batch_size = 10
            result = []
            for ind, i in enumerate(range(0, n_files, file_batch_size)):
                file_batch = self.repos_data_files[i:i + file_batch_size]
                rwc_batch = self.__extract_from_file_batch(file_batch)
                result += rwc_batch
            print('Repo word counts extracted')
            print(f'Result takes {sys.getsizeof(result) // 2**20} MB')
            print(f'Dumping extracted data')
            pickle.dump(result, self.repo_word_counts_file.open('wb'))
            print(f'Dumped')
            self.repo_word_counts = result
        if self.repo_word_counts is None:
            print('Loading pickled rwc data')
            self.repo_word_counts = pickle.load(self.repo_word_counts_file.open('rb'))
        return self.repo_word_counts

    def repos_names(self) -> List[str]:
        if not self.repos_names_file.exists():
            self.names = [rwc.name for rwc in self.__extract_repo_data()]
            self.repos_names_file.open('w').write('\n'.join(self.names))
        if self.names is None:
            self.names = [line.strip() for line in self.repos_names_file.open('r').readlines()]
        return self.names

    def data_stats(self) -> None:
        count_repos = 0
        count_wc = 0
        total_counts = 0
        for rwc in tqdm(self.__extract_repo_data()):
            count_repos += 1
            count_wc += len(rwc.count)
            total_counts += sum(rwc.count)
        print(f'Found:\n{count_repos} repos\n{count_wc} word-count pairs\nTotal count: {total_counts}')

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
        cluster_embeddings = np.zeros((n_repos, n_clusters), dtype=np.int32)
        repo_word_counts = self.__extract_repo_data()
        batch_size = len(repo_word_counts) // 100

        for i, start in enumerate(range(0, len(repo_word_counts), batch_size)):
            step_start_time = time.time()
            print(f'Step {i + 1} out of {len(repo_word_counts) // batch_size}')
            print('Collecting repo word counts into list')
            rwc_list = []
            for rwc in tqdm(repo_word_counts[start:start + batch_size]):
                repo_index = repos_rev_index[rwc.name]
                for word, count in zip(rwc.words.split('|'), rwc.count):
                    rwc_list.append((word, repo_index, count))

            print(f'Sorting list of len {len(rwc_list)}')
            start_time = time.time()
            rwc_list.sort()
            end_time = time.time()
            print(f'Sorted in {end_time - start_time} sec')

            print(f'Aggregating information from sorted rwc list')
            start_time = time.time()
            last_word = None
            cur_cluster = -1
            missing = 0
            saved = 0
            for rwc in tqdm(rwc_list):
                try:
                    if rwc[0] != last_word:
                        cur_cluster = clusters[token_rev_index[rwc[0]]]
                        last_word = rwc[0]
                    else:
                        saved += 1
                    cluster_embeddings[rwc[1]][cur_cluster] += rwc[2]
                except KeyError:
                    missing += 1
            end_time = time.time()
            print(f'Aggregated in {end_time - start_time} sec')
            print(f'Missed {missing} values, saved {saved} calls to dictionary')
            step_end_time = time.time()
            print(f'Step finished in {step_end_time - step_start_time:.1f} sec')

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
