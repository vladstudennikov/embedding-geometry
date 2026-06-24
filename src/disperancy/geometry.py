from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist


def normalize_to_sphere(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero vector encountered.")

    return X / norms


def pairwise_distance_matrix(X: np.ndarray) -> np.ndarray:
    return cdist(X, X, metric="euclidean")


def stolarsky_distance_term(X: np.ndarray) -> float:
    D = pairwise_distance_matrix(X)
    N = D.shape[0]

    return float(D.sum() / (N * N))

if __name__ == "__main__":
    # === Test 1 ===
    X = np.random.randn(5, 3)
    X = normalize_to_sphere(X)

    D = pairwise_distance_matrix(X)

    print(D)
    print()

    print(np.allclose(D, D.T))
    print(np.allclose(np.diag(D), 0))

    print(stolarsky_distance_term(X))

    # === Test 2 ===
    mean1 = pdist(X).mean()
    mean2 = stolarsky_distance_term(X)

    N = len(X)

    print(mean1)
    print(mean2)
    print((N-1)/N * mean1)