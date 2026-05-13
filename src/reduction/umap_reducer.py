import umap.umap_ as umap


class UMAPReducer:

    def __init__(
        self,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    ):

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state

    def reduce(
        self,
        embeddings,
        n_components
    ):

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )

        reduced = reducer.fit_transform(
            embeddings
        )

        return reduced