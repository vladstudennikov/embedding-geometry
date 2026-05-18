import pandas as pd
import logging
from pathlib import Path
from src.reduction.umap_reducer import UMAPReducer
from src.clustering.algorithms import ClusteringFactory
from src.clustering.parameter_grids import generate_parameter_grid
from src.visualization.projection_plots import ProjectionPlotter
from .base_experiment import BaseExperiment
from src.utils.paths import experiment_path, ensure_directory

class ClusterAssignmentExperiment(BaseExperiment):
    def __init__(self, config, experiment_name="cluster_assignment_sweep"):
        self.config = config
        self.experiment_name = experiment_name
        self.reducer = UMAPReducer(**config["umap"])
        
        self.results_dir = ensure_directory(experiment_path(self.experiment_name, "results"))
        self.figures_dir = ensure_directory(experiment_path(self.experiment_name, "figures"))
        self.logs_dir = ensure_directory(experiment_path(self.experiment_name, "logs"))
        
        self._setup_logging()
        self.plotter = ProjectionPlotter(self.figures_dir)

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

    def run(self, embeddings, metadata=None):
        dimensions = self.config["dimension_sweep"]["dimensions"]
        clustering_configs = {
            key: value
            for key, value in self.config.items()
            if isinstance(value, dict) and value.get("enabled", False)
        }

        self.logger.info(f"Starting ClusterAssignmentExperiment: {self.experiment_name}")

        for dimension in dimensions:
            self.logger.info(f"Processing dimension: {dimension}")
            reduced_embeddings = self.reducer.reduce(embeddings, n_components=dimension)
            
            for algorithm_name, algorithm_config in clustering_configs.items():
                parameter_grid = generate_parameter_grid(algorithm_config["parameters"])
                
                for params in parameter_grid:
                    clusterer = ClusteringFactory.create(algorithm_name, params)
                    labels = clusterer.fit_predict(reduced_embeddings)
                    
                    assignment_df = pd.DataFrame({"cluster": labels})
                    if metadata is not None:
                        metadata_reset = metadata.reset_index(drop=True)
                        assignment_df = pd.concat([metadata_reset, assignment_df], axis=1)
                    
                    params_str = "_".join([f"{k}-{v}" for k, v in params.items()])
                    filename = f"{algorithm_name}_d{dimension}_{params_str}.csv"
                    save_path = self.results_dir / filename
                    assignment_df.to_csv(save_path, index=False)
                    self.logger.info(f"  - Saved assignment: {filename}")
                    
        self.logger.info("Experiment completed successfully.")
