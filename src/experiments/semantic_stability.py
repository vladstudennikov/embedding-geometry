import json
import logging
import pandas as pd
import numpy as np

from tqdm import tqdm

from .base_experiment import BaseExperiment
from src.clustering.matching import ClusterMatcher
from src.utils.paths import (
    experiment_path,
    ensure_directory
)


class SemanticStabilityExperiment(BaseExperiment):
    def __init__(
        self,
        source_experiment="semantic_evaluation",
        experiment_name="semantic_stability"
    ):

        self.source_experiment = source_experiment
        self.experiment_name = experiment_name

        self.source_dir = experiment_path(
            self.source_experiment,
            "results"
        )

        self.results_dir = ensure_directory(
            experiment_path(
                self.experiment_name,
                "results"
            )
        )

        self.logs_dir = ensure_directory(
            experiment_path(
                self.experiment_name,
                "logs"
            )
        )

        self.matcher = ClusterMatcher()
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.logs_dir / "stability.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def _group_files(self):
        json_files = list(
            self.source_dir.glob("*.json")
        )

        groups = {}
        for f in json_files:
            method = f.name.split("_")[0]

            if method not in groups:
                groups[method] = []

            groups[method].append(f)
        return groups

    def _load_clusters(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            clusters = json.load(f)

        clusters = {
            k: v for k, v in clusters.items()
            if str(k) != "-1"
        }

        return clusters

    def run(self):
        self.logger.info(
            "Starting Semantic Stability Experiment"
        )

        groups = self._group_files()
        all_results = {}
        for method, files in groups.items():
            self.logger.info(
                f"Processing method: {method}"
            )

            files = sorted(files)
            n_files = len(files)
            similarity_matrix = np.zeros(
                (n_files, n_files)
            )

            coverage_matrix = np.zeros(
                (n_files, n_files)
            )

            file_names = [
                f.name.replace(
                    "_semantics.json",
                    ""
                )
                for f in files
            ]

            pairwise_results = []
            for i in tqdm(
                range(n_files),
                desc=f"Comparing {method}"
            ):
                clusters1 = self._load_clusters(
                    files[i]
                )
                similarity_matrix[i, i] = 1.0
                coverage_matrix[i, i] = 1.0
                
                for j in range(i + 1, n_files):
                    clusters2 = self._load_clusters(
                        files[j]
                    )
                    result = self.matcher.match(
                        clusters1,
                        clusters2
                    )
                    similarity_matrix[i, j] = (
                        result["mean_similarity"]
                    )
                    similarity_matrix[j, i] = (
                        result["mean_similarity"]
                    )
                    coverage_matrix[i, j] = (
                        result["coverage_ratio"]
                    )
                    coverage_matrix[j, i] = (
                        result["coverage_ratio"]
                    )

                    pairwise_results.append({
                        "config_1": file_names[i],
                        "config_2": file_names[j],
                        "mean_similarity":
                            result["mean_similarity"],
                        "min_similarity":
                            result["min_similarity"],
                        "max_similarity":
                            result["max_similarity"],
                        "std_similarity":
                            result["std_similarity"],
                        "coverage_ratio":
                            result["coverage_ratio"]
                    })

            df_sim = pd.DataFrame(
                similarity_matrix,
                index=file_names,
                columns=file_names
            )

            sim_path = (
                self.results_dir
                / f"{method}_similarity_matrix.csv"
            )

            df_sim.to_csv(sim_path)
            df_cov = pd.DataFrame(
                coverage_matrix,
                index=file_names,
                columns=file_names
            )

            cov_path = (
                self.results_dir
                / f"{method}_coverage_matrix.csv"
            )

            df_cov.to_csv(cov_path)
            df_pairs = pd.DataFrame(
                pairwise_results
            )

            pairs_path = (
                self.results_dir
                / f"{method}_pairwise_stats.csv"
            )

            df_pairs.to_csv(
                pairs_path,
                index=False
            )

            all_results[method] = {
                "matrix": similarity_matrix.tolist(),
                "coverage_matrix":
                    coverage_matrix.tolist(),
                "file_names": file_names
            }

            self.logger.info(
                f"Saved results for {method}"
            )

        summary_path = (
            self.results_dir / "summary.json"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                all_results,
                f,
                indent=4
            )

        self.logger.info(
            "Semantic Stability Experiment Complete"
        )

        return all_results