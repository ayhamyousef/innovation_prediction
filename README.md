# Innovation Prediction from Patent Text (Transformers)

Predicting technology reuse patterns (S-shaped, Fleeting, Linear, Exponential) from USPTO patent text and metadata using a BERT-based Transformer classifier. Inspired by [Chen et al. (2025)](https://doi.org/10.1007/s11192-025-05449-1) — *Scientometrics*.

## Overview

Technologies are defined as **pairwise IPC-code combinations** (6-digit level). Novel technologies are first-time combinations, and their cumulative reuse over a 20-year window forms a trajectory. We cluster trajectories into **4 growth patterns** via DTW + k-means, then train a Transformer to predict which pattern a novel technology will follow — using only its patent abstract/title and 7 early-stage metadata features available at emergence time.

### Key Differences from the Original Paper
| Aspect | Chen et al. (2025) | This Implementation |
|---|---|---|
| Data source | PATSTAT (EPO) | **USPTO via ODP API** |
| Time range | 1995–2020 | **2002–2022** |
| Best model | GBDT on 7 metadata features | **BERT + metadata fusion** |
| Text features | None | **Patent title + abstract** |
| Evaluation | ROC-AUC, cross-entropy (10-fold CV) | **Macro-F1, accuracy, ROC-AUC** (time-based split) |

## Project Structure

```
innovation_prediction/
├── config/
│   └── default.yaml            # All hyperparameters & pipeline settings
├── src/
│   ├── data/
│   │   ├── odp_client.py       # USPTO PatentsView API client (w/ caching)
│   │   ├── feature_extraction.py  # 7 paper features (Eqs. 3-9) + text extractor
│   │   ├── clustering.py       # DTW distance matrix, k-means, trajectory builder
│   │   └── dataset.py          # PyTorch Dataset, time-based splits, samplers
│   ├── models/
│   │   ├── transformer.py      # BERT + MetadataEncoder + fusion (concat/gated/cross-attn)
│   │   └── baselines.py        # TF-IDF+LR, 7 metadata-only algos, combined
│   ├── training/
│   │   ├── trainer.py          # FP16 training loop, early stopping, LR schedulers
│   │   └── evaluate.py         # Metrics, confusion matrix, training curves, ablation
│   └── utils/
│       └── helpers.py          # Config loading, seed, device, logging
├── scripts/
│   ├── 01_fetch_data.py        # Download patents from PatentsView
│   ├── 02_build_trajectories.py # Extract tech pairs, compute features, build trajectories
│   ├── 03_cluster_patterns.py  # DTW + k-means → 4 cluster labels
│   ├── 05_train_transformer.py # Train BERT classifier
│   ├── 06_train_baselines.py   # Train all baselines
│   └── 07_run_ablations.py     # Ablation study runner
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 7 Predictive Features (from the paper)

| # | Feature | Equation | Description |
|---|---------|----------|-------------|
| 1 | `ACCESS_SIZE` | Eq. 3 | Sum of component patent counts in 5 years before emergence |
| 2 | `ACCESS_TREND` | Eq. 4 | Growth ratio of component accessibility |
| 3 | `SIM_ACCESS` | Eq. 5 | Absolute difference in component cumulative counts |
| 4 | `SIM_TECH` | Eq. 6 | Weighted IPC distance (section/class/subclass) |
| 5 | `INVENT_DIVER` | Eq. 7 | Entropy of IPC sections in early inventions |
| 6 | `INVENT_APPL` | Eq. 8 | Average IPC codes per early patent |
| 7 | `ATTENT_SIZE` | Eq. 9 | Number of early inventions |

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
export ODP_API_KEY="your_key_here"   # optional — public endpoint works without key
```

### 2. Run the Pipeline

```bash
# Fetch USPTO patents (2002–2022)
python scripts/01_fetch_data.py --start-year 2002 --end-year 2022 --max-pages 5

# Extract technologies, compute features, build 20-year reuse trajectories
python scripts/02_build_trajectories.py

# Cluster trajectories → 4 patterns (S-shaped, Fleeting, Linear, Exponential)
python scripts/03_cluster_patterns.py

# Train BERT-based classifier
python scripts/05_train_transformer.py --epochs 20 --lr 2e-5

# Train baselines for comparison
python scripts/06_train_baselines.py

# Run ablation studies
python scripts/07_run_ablations.py --variants text_only,metadata_text,length_128
```

### 3. Docker (Reproducible)

```bash
export ODP_API_KEY="your_key"
docker compose run fetch-data
docker compose run build-trajectories
docker compose run cluster
docker compose run train-transformer
docker compose run train-baselines
```

## Model Architecture

```
Patent Text (title + abstract)
        │
        ▼
  ┌─────────────┐
  │ BERT-base    │  (frozen lower layers optional)
  │ [CLS] token  │──────────────────────┐
  └─────────────┘                       │
                                        ▼
  7 Metadata Features              ┌──────────┐
        │                          │  Fusion   │  concat / gated / cross-attention
        ▼                          └──────────┘
  ┌─────────────┐                       │
  │ 2-layer MLP  │──────────────────────┘
  │ (metadata)   │                      │
  └─────────────┘                       ▼
                                  ┌──────────┐
                                  │ 3-layer  │
                                  │ MLP Head │ → 4 classes
                                  └──────────┘
```

## Data Splits

- **Train**: technologies emerging ≤ 2015
- **Validation**: technologies emerging 2016–2018
- **Test**: technologies emerging > 2018

This simulates real-world deployment: predicting future technology trajectories from historically observed patterns.

## Configuration

All settings live in `config/default.yaml`. Key sections:

- `data`: API settings, year range, observation window, min reuse threshold
- `features`: access window (5 years), SIM_TECH weights [0.5, 0.3, 0.2]
- `clustering`: DTW + k-means, k=4, z-score normalization
- `model`: backbone, hidden_dim, dropout, fusion strategy
- `training`: epochs, batch_size, lr, scheduler, early stopping
- `baselines`: TF-IDF settings, ML algorithms
- `ablations`: variant definitions

## Citation

Based on:
```
Chen, W., Ma, Y., Ba, Z., & Li, G. (2025). Predicting reuse patterns of novel
technologies: the impact of technology components and early inventions on
technology trajectories. Scientometrics. https://doi.org/10.1007/s11192-025-05449-1
```
