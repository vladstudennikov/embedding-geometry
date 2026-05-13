# RESEARCH ROADMAP

Project: UMAP Spurious Clusters

Focus:
Investigating how UMAP dimensionality reduction influences geometric distortion, clustering behavior, and artificial cluster emergence in text embedding spaces.

---

# Core Research Hypothesis

Aggressive dimensionality reduction and certain UMAP parameter configurations may artificially increase cluster separability and clustering metrics without preserving meaningful semantic structure from the original embedding space.

---

# Core Experiments

## 1. Dimensionality Compression Analysis
Status: In Progress

Goal:
Analyze how clustering behavior changes under increasingly aggressive dimensionality reduction.

Main questions:
- Do clustering metrics improve artificially at low dimensions?
- At which dimensionality does geometric distortion become significant?
- How much neighborhood structure is preserved?

Main variables:
- target dimensionality
- clustering metrics
- neighborhood preservation

---

## 2. Hyperparameter Sensitivity Study
Status: Planned

Goal:
Measure how UMAP hyperparameters influence geometric structure and cluster formation.

Parameters:
- n_neighbors
- min_dist

Main questions:
- Which parameter regions produce strongest separation effects?
- Which settings produce unstable or misleading structures?
- How sensitive are clustering outcomes to parameter changes?

---

## 3. Clustering Before vs After UMAP
Status: Planned

Goal:
Compare clustering quality in original and reduced embedding spaces.

Main questions:
- Does UMAP improve semantic clustering quality?
- Does geometric separation correspond to meaningful structure?
- Which clustering characteristics change after reduction?

---

## 4. Neighborhood Preservation Analysis
Status: Planned

Goal:
Measure preservation of local relationships after dimensionality reduction.

Metrics:
- trustworthiness
- k-nearest-neighbor overlap
- continuity

Main questions:
- Which local structures remain stable?
- How strongly are neighborhoods distorted?
- How does preservation change with dimensionality?

---

# Secondary Experiments

## 5. Random Seed Stability
Status: Planned

Goal:
Evaluate reproducibility of UMAP projections and clustering results.

Main questions:
- How stable are cluster structures across runs?
- Which parameter settings maximize stability?
- How strongly do stochastic effects influence results?

---

## 6. Artificial Cluster Emergence
Status: Planned

Goal:
Determine whether UMAP can produce convincing cluster structures in weakly structured or ambiguous data.

Possible approaches:
- noisy embeddings
- partially shuffled data
- semantically overlapping datasets

Main questions:
- Can visually separated clusters emerge without strong semantic structure?
- Does dimensionality reduction exaggerate weak patterns?

---

# Future Work

## Distance Metric Analysis
Compare cosine, euclidean, and manhattan distance metrics.

## Cross-Dataset Validation
Evaluate whether observed effects generalize across multiple text datasets.

## Embedding Model Comparison
Analyze whether different embedding architectures influence distortion behavior.

## Topological Structure Analysis
Investigate structural consistency using topological data analysis methods.

---

# Long-Term Objective

Develop a systematic understanding of when UMAP:
- preserves meaningful semantic structure,
- improves clustering reliability,
- or introduces misleading geometric artifacts in text embedding spaces.