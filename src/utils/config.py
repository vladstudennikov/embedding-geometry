from pathlib import Path

import yaml


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path):

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge_dicts(base, update):

    result = base.copy()

    for key, value in update.items():

        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):

            result[key] = deep_merge_dicts(
                result[key],
                value
            )

        else:
            result[key] = value

    return result


def load_default_config():

    return load_yaml(
        CONFIG_DIR / "default.yaml"
    )


def load_umap_config():

    return load_yaml(
        CONFIG_DIR / "umap.yaml"
    )


def load_clustering_config():

    return load_yaml(
        CONFIG_DIR / "clustering.yaml"
    )


def load_experiment_config(
    experiment_name
):

    return load_yaml(
        CONFIG_DIR
        / "experiments"
        / f"{experiment_name}.yaml"
    )


def load_all_configs(
    experiment_name
):

    configs = [

        load_default_config(),

        load_umap_config(),

        load_clustering_config(),

        load_experiment_config(
            experiment_name
        )
    ]

    merged = {}

    for config in configs:

        merged = deep_merge_dicts(
            merged,
            config
        )

    return merged