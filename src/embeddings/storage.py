from pathlib import Path
import numpy as np
from src.utils.paths import project_path


def save_embeddings(
    embeddings,
    output_path
):

    full_path = project_path(output_path)

    full_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(full_path, embeddings)


def load_embeddings(
    path
):

    full_path = project_path(path)

    return np.load(full_path)