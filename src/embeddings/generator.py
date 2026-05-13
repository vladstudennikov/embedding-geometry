from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:

    def __init__(
        self,
        model_name
    ):

        self.model = SentenceTransformer(
            model_name
        )

    def generate(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True
    ):

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings
        )

        return embeddings