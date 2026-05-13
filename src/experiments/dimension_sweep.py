import pandas as pd

from src.reduction.umap_reducer import (
    UMAPReducer
)

from src.clustering.algorithms import (
    ClusteringFactory
)

from src.clustering.parameter_grids import (
    generate_parameter_grid
)

from src.clustering.evaluation import (
    ClusteringEvaluator
)

from .base_experiment import (
    BaseExperiment
)


class DimensionSweepExperiment(
    BaseExperiment
):

    def __init__(
        self,
        config
    ):

        self.config = config

        self.reducer = UMAPReducer(
            **config["umap"]
        )

        self.evaluator = (
            ClusteringEvaluator()
        )

    def run(
        self,
        embeddings
    ):

        dimensions = (
            self.config["dimension_sweep"][
                "dimensions"
            ]
        )

        clustering_configs = {
            key: value
            for key, value in self.config.items()
            if isinstance(value, dict)
            and value.get("enabled", False)
        }

        results = []

        for dimension in dimensions:

            reduced_embeddings = (
                self.reducer.reduce(
                    embeddings,
                    n_components=dimension
                )
            )

            for (
                algorithm_name,
                algorithm_config
            ) in clustering_configs.items():

                parameter_grid = (
                    generate_parameter_grid(
                        algorithm_config[
                            "parameters"
                        ]
                    )
                )

                for params in parameter_grid:

                    clusterer = (
                        ClusteringFactory.create(
                            algorithm_name,
                            params
                        )
                    )

                    labels = (
                        clusterer.fit_predict(
                            reduced_embeddings
                        )
                    )

                    metrics = (
                        self.evaluator.evaluate(
                            embeddings,
                            reduced_embeddings,
                            labels
                        )
                    )

                    row = {
                        "algorithm":
                            algorithm_name,

                        "dimension":
                            dimension,

                        **params,

                        **metrics
                    }

                    results.append(row)

        return pd.DataFrame(results)