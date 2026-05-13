import pandas as pd
from .preprocessing import preprocess_text
from src.utils.paths import project_path


def load_product_dataset(
    dataset_path,
    text_column
):

    full_path = project_path(dataset_path)
    df = pd.read_excel(full_path)
    df = df.dropna(subset=[text_column])

    df["processed_text"] = (
        df[text_column]
        .astype(str)
        .apply(preprocess_text)
    )

    return df