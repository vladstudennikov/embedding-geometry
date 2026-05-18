import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class ClusterSemanticAnalyzer:
    """
    Analyzes the semantic content of clusters using TF-IDF.
    """

    def __init__(self, stop_words="english", top_n=10):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.top_n = top_n

    def analyze_clusters(self, df, text_column, cluster_column="cluster"):
        """
        Extracts top keywords and statistics for each cluster.
        """
        # Group text by cluster
        cluster_docs = df.groupby(cluster_column)[text_column].apply(lambda x: " ".join(x.astype(str)))
        
        # Fit-transform
        tfidf_matrix = self.vectorizer.fit_transform(cluster_docs)
        feature_names = self.vectorizer.get_feature_names_out()
        
        results = {}
        cluster_ids = cluster_docs.index.tolist()
        
        for i, cluster_id in enumerate(cluster_ids):
            # Get tf-idf scores for this cluster
            row = tfidf_matrix.getrow(i).toarray().flatten()
            
            # Get top indices
            top_indices = row.argsort()[-self.top_n:][::-1]
            
            # Filter out zero-score words
            top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            
            # Basic stats
            n_points = len(df[df[cluster_column] == cluster_id])
            
            results[int(cluster_id)] = {
                "n_points": n_points,
                "top_keywords": top_words,
                "cluster_id": int(cluster_id)
            }
            
        return results
