import numpy as np

from src.disperancy.geometry import normalize_to_sphere
from src.disperancy.stolarsky import discrepancy, discrepancy_squared


def random_points(n: int, embedding_dim: int) -> np.ndarray:
    X = np.random.randn(n, embedding_dim)
    return normalize_to_sphere(X)


def main():
    np.random.seed(42)

    N = 1024
    embedding_dim = 384

    X = random_points(N, embedding_dim)

    print(f"L2² = {discrepancy_squared(X)}")
    print(f"L2  = {discrepancy(X)}")


if __name__ == "__main__":
    main()