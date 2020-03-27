from argparse import ArgumentParser
from pathlib import Path
from spherecluster import SphericalKMeans

from scripts.data_processing.model_folder import ModelFolder
from scripts.data_processing.data_loading import save_clustering_model


def run_clustering(model_folder: ModelFolder, n_clusters: int, n_init: int, max_iter: int, init: str = 'random',
                   n_jobs: int = -1, random_state: int = 42) -> SphericalKMeans:

    vectors = model_folder.get_vectors()
    kmeans = SphericalKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, n_jobs=-n_jobs,
                             verbose=1, random_state=random_state, init=init)
    kmeans.fit(vectors)

    save_clustering_model(model_folder.clustering_models_folder / f'kmeans-{n_clusters}', kmeans)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', help='Folder with model information', required=True, type=str)
    parser.add_argument('--data_folder', help='Folder with dataset info', required=True, type=str)
    parser.add_argument('--n_clusters', help='Number of clusters', required=True, type=int)
    parser.add_argument('--n_init', help='Number of initializations to try', default=10, type=int)
    parser.add_argument('--max_iter', help='Maximum number of iterations', default=300, type=int)
    parser.add_argument('--init', help='Type of initialization (k-means++ or random)', default='random', type=str)
    parser.add_argument('--n_jobs', help='Number of processes to use when optimizing', default=-1, type=int)
    parser.add_argument('--random_state', help='Random state', default=42, type=int)
    args = parser.parse_args()

    run_clustering(ModelFolder(Path(args.folder), Path(args.data_folder)), args.n_clusters, args.n_init, args.max_iter,
                   args.init, args.n_jobs, args.random_state)
