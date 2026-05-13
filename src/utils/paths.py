from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


def project_path(*parts):

    return PROJECT_ROOT.joinpath(*parts)


def ensure_directory(path):

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    return path

def experiment_path(
    experiment_name,
    *parts
):

    return project_path(
        "experiments",
        experiment_name,
        *parts
    )