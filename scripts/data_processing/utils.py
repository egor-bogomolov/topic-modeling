from typing import *

from scripts.data_processing.clustering_model import ClusteringModel
from scripts.data_processing.model_folder import ModelFolder


def get_clustering_models(model_folder: ModelFolder, min_repo_count: int = 10) -> List[ClusteringModel]:
    return [
        ClusteringModel(f, model_folder, min_repo_count)
        for f in model_folder.get_clustering_models_folders()
    ]
