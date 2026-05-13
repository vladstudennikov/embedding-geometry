from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.paths import (
    project_path
)


class ProjectionPlotter:

    def __init__(
        self,
        figures_dir
    ):

        self.figures_dir = project_path(
            figures_dir
        )

        self.figures_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def plot_projection(
        self,
        embeddings_2d,
        labels,
        algorithm_name,
        dimension
    ):

        plt.figure(figsize=(8, 6))

        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            s=10
        )

        plt.title(
            f"{algorithm_name} "
            f"(dim={dimension})"
        )

        plt.tight_layout()

        output_path = (
            self.figures_dir
            / f"{algorithm_name}_{dimension}d_projection.png"
        )

        plt.savefig(output_path)

        plt.close()