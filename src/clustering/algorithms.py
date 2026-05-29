from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

import hdbscan
from scipy.cluster.hierarchy import linkage, fcluster


class ClusteringFactory:

    @staticmethod
    def create(
        algorithm_name,
        parameters
    ):

        if algorithm_name == "kmeans":

            return KMeans(
                n_clusters=parameters["n_clusters"],
                random_state=42,
                n_init="auto"
            )

        elif algorithm_name == "dbscan":

            return DBSCAN(
                eps=parameters["eps"],
                min_samples=parameters["min_samples"]
            )

        elif algorithm_name == "hdbscan":

            return hdbscan.HDBSCAN(
                min_cluster_size=parameters[
                    "min_cluster_size"
                ]
            )

        elif algorithm_name == "agglomerative":

            linkage_type = parameters["linkage"]

            kwargs = {
                "n_clusters": parameters["n_clusters"],
                "linkage": linkage_type
            }

            if linkage_type != "ward":
                kwargs["metric"] = "cosine"

            return AgglomerativeClustering(
                **kwargs
            )

        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}"
            )


class HierarchicalClustering:
    """
    Wrapper for Scipy hierarchical clustering to allow for linkage tree 
    and multi-level cuts.
    """

    @staticmethod
    def compute_linkage(embeddings, method='single', metric='cosine'):
        """
        Computes the linkage matrix.
        """
        return linkage(embeddings, method=method, metric=metric)

    @staticmethod
    def get_clusters(linkage_matrix, k):
        """
        Extracts cluster assignments for a given number of clusters k.
        """
        return fcluster(linkage_matrix, t=k, criterion='maxclust')