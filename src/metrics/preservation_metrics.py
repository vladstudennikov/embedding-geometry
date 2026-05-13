import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def compute_trustworthiness(
    original_embeddings,
    reduced_embeddings,
    n_neighbors=15
):

    return trustworthiness(
        original_embeddings,
        reduced_embeddings,
        n_neighbors=n_neighbors
    )


def knn_preservation(
    original_embeddings,
    reduced_embeddings,
    k=15
):

    original_nn = NearestNeighbors(
        n_neighbors=k
    )

    reduced_nn = NearestNeighbors(
        n_neighbors=k
    )

    original_nn.fit(original_embeddings)

    reduced_nn.fit(reduced_embeddings)

    original_indices = original_nn.kneighbors(
        return_distance=False
    )

    reduced_indices = reduced_nn.kneighbors(
        return_distance=False
    )

    overlaps = []

    for original, reduced in zip(
        original_indices,
        reduced_indices
    ):

        overlap = len(
            set(original).intersection(
                set(reduced)
            )
        )

        overlaps.append(overlap / k)

    return np.mean(overlaps)