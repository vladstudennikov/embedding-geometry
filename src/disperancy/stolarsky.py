from __future__ import annotations

import numpy as np

from src.disperancy.constants import C_d, I_d
from src.disperancy.geometry import stolarsky_distance_term


def discrepancy_squared(X: np.ndarray) -> float:
    N, embedding_dim = X.shape
    d = embedding_dim - 1
    pairwise = stolarsky_distance_term(X)

    return C_d(d) * I_d(d) - C_d(d) * pairwise

def discrepancy(X: np.ndarray) -> float:
    value = discrepancy_squared(X)

    if value < -1e-12:
        raise RuntimeError(
            f"Negative discrepancy² = {value}"
        )

    return np.sqrt(max(value, 0.0))