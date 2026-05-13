from src.metrics.clustering_metrics import (
    safe_clustering_metrics
)

from src.metrics.preservation_metrics import (
    compute_trustworthiness,
    knn_preservation
)


class ClusteringEvaluator:

    def evaluate(
        self,
        original_embeddings,
        reduced_embeddings,
        labels
    ):

        clustering_metrics = (
            safe_clustering_metrics(
                reduced_embeddings,
                labels
            )
        )

        preservation_metrics = {

            "trustworthiness":
                compute_trustworthiness(
                    original_embeddings,
                    reduced_embeddings
                ),

            "knn_preservation":
                knn_preservation(
                    original_embeddings,
                    reduced_embeddings
                )
        }

        return {
            **clustering_metrics,
            **preservation_metrics
        }