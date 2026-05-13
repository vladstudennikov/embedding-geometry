from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

import hdbscan


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

            linkage = parameters["linkage"]

            kwargs = {
                "n_clusters": parameters["n_clusters"],
                "linkage": linkage
            }

            if linkage != "ward":
                kwargs["metric"] = "cosine"

            return AgglomerativeClustering(
                **kwargs
            )

        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm_name}"
            )