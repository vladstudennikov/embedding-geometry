import numpy as np

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def count_clusters(labels):

    unique = set(labels)

    if -1 in unique:
        unique.remove(-1)

    return len(unique)


def compute_noise_ratio(labels):

    noise_points = np.sum(labels == -1)

    return noise_points / len(labels)


def safe_clustering_metrics(
    X,
    labels
):

    cluster_count = count_clusters(labels)

    if cluster_count < 2:

        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
            "cluster_count": cluster_count,
            "noise_ratio": compute_noise_ratio(labels)
        }

    metrics = {
        "silhouette": silhouette_score(
            X,
            labels
        ),

        "davies_bouldin": davies_bouldin_score(
            X,
            labels
        ),

        "calinski_harabasz": calinski_harabasz_score(
            X,
            labels
        ),

        "cluster_count": cluster_count,

        "noise_ratio": compute_noise_ratio(labels)
    }

    return metrics