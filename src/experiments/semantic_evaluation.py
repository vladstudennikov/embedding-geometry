import pandas as pd
import json
import logging
from pathlib import Path
from .base_experiment import BaseExperiment
from src.clustering.semantic_analysis import ClusterSemanticAnalyzer
from src.utils.paths import experiment_path, ensure_directory

class SemanticEvaluationExperiment(BaseExperiment):
    """
    Experiment to evaluate the semantic content of clusters.
    """

    def __init__(self, source_experiment="cluster_assignment_sweep", experiment_name="semantic_evaluation"):
        self.source_experiment = source_experiment
        self.experiment_name = experiment_name
        
        # Paths
        self.source_dir = experiment_path(self.source_experiment, "results")
        self.results_dir = ensure_directory(experiment_path(self.experiment_name, "results"))
        self.logs_dir = ensure_directory(experiment_path(self.experiment_name, "logs"))
        
        self.analyzer = ClusterSemanticAnalyzer(top_n=15)
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.logs_dir / "evaluation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run(self, text_column):
        """
        Analyzes semantic content for all files in the source experiment.
        """
        self.logger.info(f"Starting SemanticEvaluationExperiment for source: {self.source_experiment}")
        
        if not self.source_dir.exists():
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            return
            
        csv_files = list(self.source_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            self.logger.info(f"Analyzing semantics for {csv_file.name}...")
            
            try:
                df = pd.read_csv(csv_file)
                if text_column not in df.columns or "cluster" not in df.columns:
                    self.logger.warning(f"  - Missing columns in {csv_file.name}. Skipping.")
                    continue
                
                # Perform analysis
                results = self.analyzer.analyze_clusters(df, text_column)
                
                # Add "strange words" (unique to this cluster across the current analysis)
                all_top_words = []
                for res in results.values():
                    all_top_words.extend(res["top_keywords"])
                
                word_counts = pd.Series(all_top_words).value_counts()
                unique_words = word_counts[word_counts == 1].index.tolist()
                
                for res in results.values():
                    res["unique_keywords"] = [w for w in res["top_keywords"] if w in unique_words]

                # Save individual report
                report_filename = f"{csv_file.stem}_semantics.json"
                with open(self.results_dir / report_filename, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                
                self.logger.info(f"  - Saved semantic report: {report_filename}")
                
            except Exception as e:
                self.logger.error(f"  - Error evaluating {csv_file.name}: {e}")

        self.logger.info("Semantic evaluation complete.")
