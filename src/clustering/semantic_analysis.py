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

    @staticmethod
    def compute_jaccard_overlap(set1, set2):
        """
        Computes Jaccard similarity between two sets of keywords.
        """
        s1 = set(set1)
        s2 = set(set2)
        if not s1 and not s2:
            return 0.0
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def compute_hierarchy_stability(self, level_results_parent, level_results_child, mapping_df, parent_col, child_col):
        """
        Computes similarity between parent and child clusters in a hierarchy.
        """
        stability_results = []
        
        # Get unique parent-child pairs from mapping
        pairs = mapping_df[[parent_col, child_col]].drop_duplicates()
        
        for _, row in pairs.iterrows():
            p_id = int(row[parent_col])
            c_id = int(row[child_col])
            
            if p_id in level_results_parent and c_id in level_results_child:
                p_keywords = level_results_parent[p_id]["top_keywords"]
                c_keywords = level_results_child[c_id]["top_keywords"]
                
                jaccard = self.compute_jaccard_overlap(p_keywords, c_keywords)
                
                stability_results.append({
                    "parent_id": p_id,
                    "child_id": c_id,
                    "jaccard_similarity": jaccard
                })
                
        return pd.DataFrame(stability_results)
