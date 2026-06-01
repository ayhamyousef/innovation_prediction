# Innovation Prediction from Patents

Pipeline for predicting technology trajectory patterns from USPTO patent data.
Extends Chen et al. (2025, Scientometrics) by adding unsupervised feature
selection (FAE) and modern tabular classifiers.

## What this project does

A technology is defined as a pairwise IPC-code combination at the 6-digit
level. A novel technology is a first-time combination. Its cumulative reuse
over a 20-year window forms a trajectory. We cluster technologies into a
small number of growth-pattern groups based on 7 features computed at the
technology's emergence year, then train classifiers to predict the cluster
from those features.

The pipeline has four steps:

1. FAE feature selection. Pick K of the 7 features unsupervised.
2. Cluster on the selected features with k-means.
3. Stratified 60/40 train/test split, 90/10 train/val within train.
4. Classify with one or more tabular models.

## How this differs from Chen et al. (2025)

| Aspect | Chen et al. | This project |
|---|---|---|
| Data source | PATSTAT (EPO) | USPTO via ODP API |
| Time range | 1995 to 2020 | 2002 to 2022 |
| Feature selection | none | FAE (Wu and Cheng, AAAI 2021) |
| Clustering | DTW k-means, k=4 | Euclidean k-means on selected features |
| Classification | GBDT | TabNet, TabM, FT-Transformer, plus GBDT and ExtraTrees for comparison |
| Evaluation | ROC-AUC, cross-entropy | Accuracy, macro-F1, plus reconstruction MSE for FAE eval |

## The 7 features

From Chen et al. (Eqs. 3-9):

| Feature | Description |
|---|---|
| ACCESS_SIZE | Sum of component patent counts in 5 years before emergence |
| ACCESS_TREND | Growth ratio of component accessibility |
| SIM_ACCESS | Absolute difference in component cumulative counts |
| SIM_TECH | Weighted IPC distance at section, class, and subclass levels |
| INVENT_DIVER | Entropy of IPC sections in early inventions |
| INVENT_APPL | Average IPC codes per early patent |
| ATTENT_SIZE | Number of early inventions |

SIM_TECH ends up with only 4 unique values because IPC codes are hierarchical
(weights are 0.5, 0.3, 0.2). This is the correct output of Chen et al.'s
formula, not a bug.

## Quick start

```bash
pip install -r requirements.txt
```

Run the pipeline end to end:

```bash
# Fetch USPTO patents
python scripts/01_fetch_data.py --start-year 2002 --end-year 2022

# Extract technologies, compute features, build 20-year trajectories
python scripts/02_build_trajectories.py

# FAE feature selection (K=3, 4, 5 by default)
python scripts/04_feature_selection.py

# Cluster on FAE K=3 features into 3 clusters
python scripts/04b_recluster.py --fae-k 3 --cluster-k 3

# Classify with multiple models
python scripts/05_classify.py \
    --labeled-csv results/clustering/technologies_labeled_fae_k3_k3.csv \
    --features SIM_TECH ACCESS_SIZE SIM_ACCESS \
    --models tabm ft_transformer gbdt
```

To run the full 4 x 3 x 3 = 36-experiment matrix at once:

```bash
python scripts/run_experiments.py
```

## Scripts

| Script | Purpose |
|---|---|
| 01_fetch_data.py | Download patents from USPTO ODP API |
| 02_build_trajectories.py | Extract tech pairs, compute features, build trajectories |
| 04_feature_selection.py | FAE feature selection (K = 3, 4, 5) |
| 04b_recluster.py | Re-cluster on selected features |
| 05_classify.py | Train and evaluate a single classifier |
| 06_ablation_subsets.py | Exhaustive feature subset ablation by silhouette |
| 07_visualize_clusters.py | Cluster figures (PCA, boxplots, trajectories) |
| 08_fae_stability.py | Bootstrap stability test for FAE selection |
| 09_multi_metric_ablation.py | Silhouette + Davies-Bouldin + Calinski-Harabasz |
| 10_classify_top_subsets.py | Classification on top size-3 subsets across 5 models |
| 11_paper_eval.py | FAE evaluation matched to Wu and Cheng (AAAI 2021) |
| 12_training_fraction_sweep.py | DL vs GBDT under reduced training data |
| run_experiments.py | Full 36-experiment matrix orchestrator |

## Configuration

All defaults live in `config/default.yaml`. Key settings:

- `data`: API config, year range, observation window
- `features`: SIM_TECH weights [0.5, 0.3, 0.2], access window 5 years
- `clustering`: Euclidean k-means with StandardScaler, n_init=10
- `training`: seed 42, 60/40 test ratio, 90/10 val ratio, early stopping on val macro-F1

## References

Chen, W., Ma, Y., Ba, Z., and Li, G. (2025). Predicting reuse patterns of novel
technologies: the impact of technology components and early inventions on
technology trajectories. *Scientometrics*.
https://doi.org/10.1007/s11192-025-05449-1

Wu, X., and Cheng, Q. (2021). Fractal autoencoders for feature selection.
*Proceedings of the AAAI Conference on Artificial Intelligence*.
