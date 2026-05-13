from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.paths import (
    project_path
)


class MetricPlotter:

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

    def plot_metric_by_dimension(
        self,
        results_df,
        metric_name,
        algorithm_name
    ):

        subset = results_df[
            results_df["algorithm"]
            == algorithm_name
        ]

        plt.figure(figsize=(8, 5))

        for column in subset.columns:

            if column in [
                "algorithm",
                "dimension",
                metric_name,
                "trustworthiness",
                "knn_preservation",
                "cluster_count",
                "noise_ratio"
            ]:
                continue

        grouped = (
            subset.groupby("dimension")[
                metric_name
            ]
            .mean()
            .reset_index()
        )

        plt.plot(
            grouped["dimension"],
            grouped[metric_name],
            marker="o"
        )

        plt.xscale("log")

        plt.xlabel(
            "Reduced Dimension"
        )

        plt.ylabel(metric_name)

        plt.title(
            f"{algorithm_name}: "
            f"{metric_name} vs Dimension"
        )

        plt.grid(True)

        plt.tight_layout()

        output_path = (
            self.figures_dir
            / f"{algorithm_name}_{metric_name}.png"
        )

        plt.savefig(output_path)

        plt.close()

    def plot_preservation_metrics(
        self,
        results_df
    ):

        metrics = [
            "trustworthiness",
            "knn_preservation"
        ]

        for metric in metrics:

            grouped = (
                results_df.groupby("dimension")[
                    metric
                ]
                .mean()
                .reset_index()
            )

            plt.figure(figsize=(8, 5))

            plt.plot(
                grouped["dimension"],
                grouped[metric],
                marker="o"
            )

            plt.xscale("log")

            plt.xlabel(
                "Reduced Dimension"
            )

            plt.ylabel(metric)

            plt.title(
                f"{metric} vs Dimension"
            )

            plt.grid(True)

            plt.tight_layout()

            output_path = (
                self.figures_dir
                / f"{metric}.png"
            )

            plt.savefig(output_path)

            plt.close()