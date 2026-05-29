import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from src.utils.paths import project_path


class HierarchicalPlotter:
    """
    Plotting utilities for hierarchical clustering experiments.
    """

    def __init__(self, figures_dir):
        self.figures_dir = project_path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="white")

    def plot_dendrogram(self, linkage_matrix, truncate_mode='lastp', p=30, title="Hierarchical Clustering Dendrogram"):
        """
        Plots a truncated dendrogram.
        """
        plt.figure(figsize=(12, 6))
        dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
        plt.title(title)
        plt.xlabel("Cluster Size (or index if not truncated)")
        plt.ylabel("Distance")
        
        output_path = self.figures_dir / "dendrogram.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def plot_cluster_flow(self, df, level1_col, level2_col, title="Cluster Split Flow"):
        """
        Creates a heatmap showing how clusters from level 1 split into level 2.
        This serves as a 'flow' visualization.
        """
        flow_counts = pd.crosstab(df[level1_col], df[level2_col])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(flow_counts, annot=True, fmt='d', cmap="YlGnBu")
        plt.title(title)
        plt.xlabel(f"Level: {level2_col}")
        plt.ylabel(f"Level: {level1_col}")
        
        output_path = self.figures_dir / f"flow_{level1_col}_to_{level2_col}.png"
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def plot_parent_child_similarity(self, stability_df, parent_col='parent_id', child_col='child_id', metric_col='jaccard_similarity', title="Parent-Child Semantic Similarity", suffix=""):
        """
        Plots a heatmap of semantic similarity between parent and child clusters.
        """
        pivot_df = stability_df.pivot(index=parent_col, columns=child_col, values=metric_col)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap="viridis", vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel("Child Cluster ID")
        plt.ylabel("Parent Cluster ID")
        
        filename = f"similarity_{parent_col}_{child_col}_{suffix}.png" if suffix else f"similarity_{parent_col}_{child_col}.png"
        output_path = self.figures_dir / filename
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
