import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_cluster_statistics(labels):
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    is_noise = (labels == -1)
    n_noise = np.sum(is_noise)
    noise_ratio = n_noise / len(labels)
    
    cluster_labels = unique_labels[unique_labels != -1]
    n_clusters = len(cluster_labels)
    
    if n_clusters > 0:
        counts = pd.Series(labels[~is_noise]).value_counts()
        cluster_size_mean = counts.mean()
        cluster_size_std = counts.std()
        cluster_size_min = counts.min()
        cluster_size_max = counts.max()
        
        probs = counts / counts.sum()
        ent = entropy(probs)
        max_ent = np.log(n_clusters) if n_clusters > 1 else 1
        normalized_entropy = ent / max_ent if n_clusters > 1 else 0
    else:
        cluster_size_mean = 0
        cluster_size_std = 0
        cluster_size_min = 0
        cluster_size_max = 0
        normalized_entropy = 0

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": noise_ratio,
        "cluster_size_mean": cluster_size_mean,
        "cluster_size_std": cluster_size_std,
        "cluster_size_min": cluster_size_min,
        "cluster_size_max": cluster_size_max,
        "normalized_entropy": normalized_entropy
    }
