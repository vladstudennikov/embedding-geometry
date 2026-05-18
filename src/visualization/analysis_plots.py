import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.paths import project_path


class AnalysisPlotter:
    """
    Clean plotting utilities for UMAP clustering experiments.
    """

    def __init__(self, figures_dir):
        self.figures_dir = project_path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="whitegrid")

    def _prepare_data(self, summary_df):
        """
        Cleans and aggregates dataframe.
        """

        plot_df = (
            summary_df
            .dropna(subset=["dimension"])
            .copy()
        )

        plot_df["dimension"] = plot_df["dimension"].astype(int)

        return plot_df

    def plot_metric_vs_dimension(self, summary_df, metric="n_clusters"):
        """
        Plots aggregated metric vs UMAP dimension.
        """

        plot_df = self._prepare_data(summary_df)

        agg_df = (
            plot_df
            .groupby(["algorithm", "dimension"])[metric]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=agg_df,
            x="dimension",
            y=metric,
            hue="algorithm",
            marker="o"
        )

        plt.xscale("log", base=2)

        dims = sorted(agg_df["dimension"].unique())
        plt.xticks(dims, dims)

        plt.title(f"{metric.replace('_', ' ').title()} vs UMAP Dimension")
        plt.xlabel("UMAP Dimension")
        plt.ylabel(metric.replace("_", " ").title())

        plt.tight_layout()

        output_path = self.figures_dir / f"{metric}_vs_dimension.png"

        plt.savefig(output_path)
        plt.close()

    def plot_cluster_size_mean(self, summary_df):
        """
        Plots average cluster size vs dimension.
        """

        plot_df = self._prepare_data(summary_df)

        agg_df = (
            plot_df
            .groupby(["algorithm", "dimension"])["cluster_size_mean"]
            .mean()
            .reset_index()
        )

        g = sns.FacetGrid(
            agg_df,
            col="algorithm",
            col_wrap=2,
            height=4,
            sharey=False
        )

        g.map_dataframe(
            sns.lineplot,
            x="dimension",
            y="cluster_size_mean",
            marker="o"
        )

        for ax in g.axes.flatten():

            ax.set_xscale("log", base=2)

            dims = sorted(agg_df["dimension"].unique())

            ax.set_xticks(dims)
            ax.set_xticklabels(dims)

            ax.set_xlabel("UMAP Dimension")
            ax.set_ylabel("Mean Cluster Size")

        g.set_titles("{col_name}")

        plt.tight_layout()

        output_path = self.figures_dir / "cluster_size_mean_vs_dimension.png"

        plt.savefig(output_path)
        plt.close()

    def plot_cluster_size_max(self, summary_df):
        """
        Plots maximum cluster size vs dimension.
        """

        plot_df = self._prepare_data(summary_df)

        agg_df = (
            plot_df
            .groupby(["algorithm", "dimension"])["cluster_size_max"]
            .mean()
            .reset_index()
        )

        g = sns.FacetGrid(
            agg_df,
            col="algorithm",
            col_wrap=2,
            height=4,
            sharey=False
        )

        g.map_dataframe(
            sns.lineplot,
            x="dimension",
            y="cluster_size_max",
            marker="o"
        )

        for ax in g.axes.flatten():

            ax.set_xscale("log", base=2)

            dims = sorted(agg_df["dimension"].unique())

            ax.set_xticks(dims)
            ax.set_xticklabels(dims)

            ax.set_xlabel("UMAP Dimension")
            ax.set_ylabel("Max Cluster Size")

        g.set_titles("{col_name}")

        plt.tight_layout()

        output_path = self.figures_dir / "cluster_size_max_vs_dimension.png"

        plt.savefig(output_path)
        plt.close()

    def plot_noise_ratio(self, summary_df):
        """
        Plots noise ratio vs dimension.
        Only for DBSCAN/HDBSCAN.
        """

        plot_df = self._prepare_data(summary_df)

        noise_df = plot_df[
            plot_df["algorithm"].isin(["dbscan", "hdbscan"])
        ]

        if noise_df.empty:
            return

        agg_df = (
            noise_df
            .groupby(["algorithm", "dimension"])["noise_ratio"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=agg_df,
            x="dimension",
            y="noise_ratio",
            hue="algorithm",
            marker="o"
        )

        plt.xscale("log", base=2)

        dims = sorted(agg_df["dimension"].unique())
        plt.xticks(dims, dims)

        plt.title("Noise Ratio vs UMAP Dimension")
        plt.xlabel("UMAP Dimension")
        plt.ylabel("Noise Ratio")

        plt.tight_layout()

        output_path = self.figures_dir / "noise_ratio_vs_dimension.png"

        plt.savefig(output_path)
        plt.close()