import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_keyword_similarity(keywords1, keywords2):
    """
    Computes similarity between two keyword sets using
    normalized keyword overlap.
    """

    set1 = set(keywords1)
    set2 = set(keywords2)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1.intersection(set2))
    similarity = intersection / np.sqrt(len(set1) * len(set2))

    return float(similarity)


class ClusterMatcher:
    def __init__(self, similarity_func=compute_keyword_similarity):
        self.similarity_func = similarity_func

    def get_similarity_matrix(self, clusters1, clusters2):

        ids1 = sorted(clusters1.keys())
        ids2 = sorted(clusters2.keys())
        matrix = np.zeros((len(ids1), len(ids2)))

        for i, id1 in enumerate(ids1):
            for j, id2 in enumerate(ids2):
                keywords1 = clusters1[id1].get(
                    "unique_keywords",
                    []
                )
                keywords2 = clusters2[id2].get(
                    "unique_keywords",
                    []
                )

                matrix[i, j] = self.similarity_func(
                    keywords1,
                    keywords2
                )

        return matrix, ids1, ids2

    def match(self, clusters1, clusters2):
        if not clusters1 or not clusters2:
            return {
                "mean_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
                "std_similarity": 0.0,
                "coverage_ratio": 0.0,
                "matches": []
            }

        matrix, ids1, ids2 = self.get_similarity_matrix(
            clusters1,
            clusters2
        )

        row_ind, col_ind = linear_sum_assignment(-matrix)
        matched_similarities = matrix[row_ind, col_ind]
        coverage_ratio = (
            len(matched_similarities)
            / max(len(ids1), len(ids2))
        )

        matches = []
        for r, c in zip(row_ind, col_ind):
            matches.append({
                "cluster_id1": ids1[r],
                "cluster_id2": ids2[c],
                "similarity": float(matrix[r, c])
            })

        return {
            "mean_similarity": float(
                matched_similarities.mean()
            ),
            "min_similarity": float(
                matched_similarities.min()
            ),
            "max_similarity": float(
                matched_similarities.max()
            ),
            "std_similarity": float(
                matched_similarities.std()
            ),
            "coverage_ratio": float(
                coverage_ratio
            ),
            "matches": matches
        }