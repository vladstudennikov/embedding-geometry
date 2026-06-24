from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def sphere_dimension(embedding_dimension: int) -> int:
    if embedding_dimension < 2:
        raise ValueError("Embedding dimension must be >=2")

    return embedding_dimension - 1


def C_d(d: int) -> float:
    log_value = (
        -np.log(d)
        + gammaln((d + 1) / 2)
        - 0.5 * np.log(np.pi)
        - gammaln(d / 2)
    )

    return float(np.exp(log_value))


def I_d(d: int) -> float:
    log_value = (
        d * np.log(2.0)
        + 2 * gammaln((d + 1) / 2)
        - 0.5 * np.log(np.pi)
        - gammaln(d + 0.5)
    )

    return float(np.exp(log_value))


def initial_discrepancy_squared(d: int) -> float:
    return 1.0 - C_d(d) * I_d(d)


if __name__ == "__main__":
    d = sphere_dimension(384)

    print(d)
    print(C_d(d))
    print(I_d(d))
    print(initial_discrepancy_squared(d))