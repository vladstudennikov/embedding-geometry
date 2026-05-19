import seaborn as sns
import pandas as pd

from src.utils.paths import (
    experiment_path,
    ensure_directory
)


class StabilityPlotter:

    def __init__(
        self,
        experiment_name="semantic_stability"
    ):

        self.figures_dir = ensure_directory(
            experiment_path(
                experiment_name,
                "figures"
            )
        )

        sns.set_theme(style="white")

    def plot_heatmap(
        self,
        matrix_df,
        method_name,
        matrix_type="similarity"
    ):

        cluster_grid = sns.clustermap(
            matrix_df,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            figsize=(16, 16),
            xticklabels=True,
            yticklabels=True
        )

        cluster_grid.fig.suptitle(
            f"{method_name} {matrix_type.capitalize()} Clustermap",
            y=1.02
        )

        output_path = (
            self.figures_dir
            / f"{method_name}_{matrix_type}_clustermap.png"
        )

        cluster_grid.savefig(
            output_path,
            dpi=300
        )

        return output_path

    def plot_all_heatmaps(
        self,
        stability_results
    ):

        plots = {}

        for method, data in stability_results.items():

            sim_df = pd.DataFrame(
                data["matrix"],
                index=data["file_names"],
                columns=data["file_names"]
            )

            cov_df = pd.DataFrame(
                data["coverage_matrix"],
                index=data["file_names"],
                columns=data["file_names"]
            )

            sim_plot = self.plot_heatmap(
                sim_df,
                method,
                "similarity"
            )

            cov_plot = self.plot_heatmap(
                cov_df,
                method,
                "coverage"
            )

            plots[method] = {
                "similarity_plot": sim_plot,
                "coverage_plot": cov_plot
            }

        return plots