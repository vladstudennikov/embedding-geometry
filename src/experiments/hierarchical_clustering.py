import pandas as pd
import numpy as np
import logging
from .base_experiment import BaseExperiment
from src.clustering.algorithms import HierarchicalClustering
from src.clustering.semantic_analysis import ClusterSemanticAnalyzer
from src.visualization.hierarchical_plots import HierarchicalPlotter
from src.utils.paths import experiment_path, ensure_directory, project_path
from src.embeddings.storage import load_embeddings
from src.data.loaders import load_product_dataset

class HierarchicalExperiment(BaseExperiment):
    """
    Experiment to run hierarchical clustering, extract semantic content,
    and analyze hierarchy stability.
    """

    def __init__(self, 
                 embeddings_path="data/embeddings/product_embeddings.npy",
                 dataset_path="data/raw/some_product_names.xlsx",
                 text_column="Name",
                 k_values=[10, 25, 50],
                 experiment_name="hierarchical_clustering"):
        
        self.embeddings_path = embeddings_path
        self.dataset_path = dataset_path
        self.text_column = text_column
        self.k_values = k_values
        self.experiment_name = experiment_name
        
        # Paths
        self.results_dir = ensure_directory(experiment_path(self.experiment_name, "results"))
        self.figures_dir = ensure_directory(experiment_path(self.experiment_name, "figures"))
        self.logs_dir = ensure_directory(experiment_path(self.experiment_name, "logs"))
        
        self._setup_logging()
        self.analyzer = ClusterSemanticAnalyzer(top_n=10)
        self.plotter = HierarchicalPlotter(self.figures_dir)

    def _setup_logging(self):
        log_file = self.logs_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.info("Starting Hierarchical Clustering Experiment")
        
        # 1. Load Data
        self.logger.info("Loading embeddings and dataset...")
        embeddings = load_embeddings(self.embeddings_path)
        df = load_product_dataset(self.dataset_path, self.text_column)
        
        # Ensure they match
        if len(embeddings) != len(df):
            self.logger.warning(f"Embedding count ({len(embeddings)}) != Dataset size ({len(df)}). Truncating.")
            min_len = min(len(embeddings), len(df))
            embeddings = embeddings[:min_len]
            df = df.iloc[:min_len]

        # 2. Compute Linkage
        self.logger.info("Computing linkage matrix (complete linkage)...")
        linkage_matrix = HierarchicalClustering.compute_linkage(embeddings, method='complete', metric='cosine')
        
        # 3. Generate Cuts and Analyze Semantics
        level_results = {}
        assignments_df = df.copy()
        
        for k in self.k_values:
            self.logger.info(f"Extracting clusters for k={k}...")
            col_name = f"cluster_k{k}"
            assignments_df[col_name] = HierarchicalClustering.get_clusters(linkage_matrix, k)
            
            # Analyze semantics
            self.logger.info(f"Analyzing semantics for k={k}...")
            level_results[k] = self.analyzer.analyze_clusters(assignments_df, self.text_column, cluster_column=col_name)
            
            # Save top terms for this level
            level_df = pd.DataFrame.from_dict(level_results[k], orient='index')
            level_df.to_csv(self.results_dir / f"top_terms_k{k}.csv", index=False)

        # 4. Compute Stability/Overlap between levels
        stability_data = {}
        for i in range(len(self.k_values) - 1):
            k_parent = self.k_values[i]
            k_child = self.k_values[i+1]
            
            self.logger.info(f"Computing stability between k={k_parent} and k={k_child}...")
            stability_df = self.analyzer.compute_hierarchy_stability(
                level_results[k_parent], 
                level_results[k_child], 
                assignments_df, 
                f"cluster_k{k_parent}", 
                f"cluster_k{k_child}"
            )
            
            stability_df.to_csv(self.results_dir / f"stability_k{k_parent}_to_k{k_child}.csv", index=False)
            stability_data[(k_parent, k_child)] = stability_df

        # 5. Visualizations
        self.logger.info("Generating visualizations...")
        self.plotter.plot_dendrogram(linkage_matrix)
        
        for i in range(len(self.k_values) - 1):
            k_parent = self.k_values[i]
            k_child = self.k_values[i+1]
            
            self.plotter.plot_cluster_flow(assignments_df, f"cluster_k{k_parent}", f"cluster_k{k_child}")
            self.plotter.plot_parent_child_similarity(
                stability_data[(k_parent, k_child)], 
                parent_col='parent_id', 
                child_col='child_id',
                suffix=f"k{k_parent}_to_k{k_child}"
            )

        # Save final assignments
        assignments_df.to_csv(self.results_dir / "hierarchical_assignments.csv", index=False)
        self.logger.info(f"Experiment complete. Results saved in {self.results_dir}")
        
        return assignments_df, linkage_matrix, level_results, stability_data
