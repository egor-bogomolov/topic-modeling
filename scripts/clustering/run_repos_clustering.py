import numpy as np

from argparse import ArgumentParser
from pathlib import Path
from typing import List

from scripts.data_processing.clustering_model import ClusteringModel
from scripts.data_processing.model_repo import ModelRepo
from spherecluster import SphericalKMeans
from joblib import Parallel, delayed, cpu_count

from scripts.data_processing.model_folder import ModelFolder
from scripts.data_processing.data_loading import save_clustering_model


def __run_clustering(vectors: np.ndarray, models_folder: Path, n_clusters: int, n_init: int, max_iter: int, init: str,
                     n_jobs: int, random_state: int) -> SphericalKMeans:
    print(f'thread = {n_jobs}')
    kmeans = SphericalKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, n_jobs=n_jobs,
                             verbose=1, random_state=random_state, init=init)
    kmeans.fit(vectors)
    save_clustering_model(models_folder / f'repo-kmeans-{n_clusters}', kmeans)


def run_clustering(
        model_folder: ModelFolder, clustering_model_name: str, data_folder: Path, ns_clusters: List[int], n_init: int,
        max_iter: int, init: str = 'random', n_jobs: int = -1, random_state: int = 42
) -> List[SphericalKMeans]:

    clustering_model = ClusteringModel(model_folder.clustering_models_folder / clustering_model_name, model_folder)
    model_repo = ModelRepo(data_folder)
    vectors = model_repo.repos_cluster_embeddings(model_folder, clustering_model)
    with Parallel(cpu_count() - 1) as pool:
        models = pool([
            delayed(__run_clustering)(vectors, model_folder.clustering_models_folder, n_clusters, n_init, max_iter,
                                      init, n_jobs, random_state)
            for n_clusters in ns_clusters
        ])
    return models


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', help='Folder with model information', required=True, type=str)
    parser.add_argument('--clustering_model', help='Name of clustering model', required=True, type=str)
    parser.add_argument('--data_folder', help='Folder with dataset info', required=True, type=str)
    parser.add_argument('--n_clusters', help='Number of clusters', required=True, type=int, nargs='+')
    parser.add_argument('--n_init', help='Number of initializations to try', default=10, type=int)
    parser.add_argument('--max_iter', help='Maximum number of iterations', default=300, type=int)
    parser.add_argument('--init', help='Type of initialization (k-means++ or random)', default='random', type=str)
    parser.add_argument('--n_jobs', help='Number of processes to use when optimizing', default=-1, type=int)
    parser.add_argument('--random_state', help='Random state', default=42, type=int)
    args = parser.parse_args()

    run_clustering(ModelFolder(Path(args.folder), Path(args.data_folder)), args.clustering_model,
                   Path(args.data_folder), args.n_clusters,
                   args.n_init, args.max_iter, args.init, args.n_jobs, args.random_state)
