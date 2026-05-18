import pandas as pd
import logging
import re
from pathlib import Path
from .base_experiment import BaseExperiment
from src.clustering.analysis import calculate_cluster_statistics
from src.utils.paths import experiment_path, ensure_directory

class ClusterAnalysisExperiment(BaseExperiment):
    """
    Experiment to analyze cluster assignment results and aggregate statistics.
    """

    def __init__(self, source_experiment="cluster_assignment_sweep", experiment_name="cluster_analysis"):
        self.source_experiment = source_experiment
        self.experiment_name = experiment_name
        
        # Paths
        self.source_dir = experiment_path(self.source_experiment, "results")
        self.results_dir = ensure_directory(experiment_path(self.experiment_name, "results"))
        self.figures_dir = ensure_directory(experiment_path(self.experiment_name, "figures"))
        self.logs_dir = ensure_directory(experiment_path(self.experiment_name, "logs"))
        
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.logs_dir / "analysis.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _parse_filename(self, filename):
        """
        Parses metadata from filename: algorithm_d<dim>_p1-v1_p2-v2.csv
        """
        # Remove extension
        name = Path(filename).stem
        parts = name.split("_")
        
        if len(parts) < 2:
            return {}
            
        algorithm = parts[0]
        dimension_str = parts[1]
        
        # Extract dimension number
        dim_match = re.search(r'd(\d+)', dimension_str)
        dimension = int(dim_match.group(1)) if dim_match else None
        
        metadata = {
            "algorithm": algorithm,
            "dimension": dimension,
            "filename": filename
        }
        
        # Extract remaining parameters
        for part in parts[2:]:
            if "-" in part:
                k, v = part.split("-", 1)
                # Try to convert to float/int if possible
                try:
                    if "." in v:
                        v = float(v)
                    else:
                        v = int(v)
                except ValueError:
                    pass
                metadata[k] = v
                
        return metadata

    def run(self):
        """
        Scans source results, analyzes each, and saves a summary.
        """
        self.logger.info(f"Starting ClusterAnalysisExperiment for source: {self.source_experiment}")
        
        if not self.source_dir.exists():
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            return None
            
        all_results = []
        csv_files = list(self.source_dir.glob("*.csv"))
        
        if not csv_files:
            self.logger.warning(f"No CSV files found in {self.source_dir}")
            return None

        for csv_file in csv_files:
            self.logger.info(f"Analyzing {csv_file.name}...")
            
            # Parse metadata
            metadata = self._parse_filename(csv_file.name)
            
            # Load and analyze
            try:
                df = pd.read_csv(csv_file)
                if "cluster" not in df.columns:
                    self.logger.warning(f"  - No 'cluster' column in {csv_file.name}. Skipping.")
                    continue
                    
                stats = calculate_cluster_statistics(df["cluster"])
                
                # Combine metadata and stats
                row = {**metadata, **stats}
                all_results.append(row)
            except Exception as e:
                self.logger.error(f"  - Error analyzing {csv_file.name}: {e}")

        summary_df = pd.DataFrame(all_results)
        
        # Save summary
        save_path = self.results_dir / "summary.csv"
        summary_df.to_csv(save_path, index=False)
        self.logger.info(f"Analysis complete. Summary saved to {save_path}")
        
        return summary_df
