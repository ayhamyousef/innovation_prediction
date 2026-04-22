#!/usr/bin/env python3
"""
investigate_ablation.py — Quick diagnostic for two open questions:

1. Is ACCESS_TREND effectively categorical? (would explain silhouette=0.9344
   at size=1)
2. Why did the FAE K=3, k=3 silhouette go from 0.684 (earlier run) to 0.6535
   (ablation script)? Suspect causes: row count changed (201,134 -> 201,710)
   or seed/n_init difference.

Usage:
    python scripts/investigate_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, set_seed


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def investigate_access_trend(tech_df):
    print("\n" + "=" * 60)
    print("Q1: Is ACCESS_TREND effectively categorical?")
    print("=" * 60)
    s = tech_df["ACCESS_TREND"]
    print(f"dtype: {s.dtype}")
    print(f"n_rows: {len(s)}")
    print(f"n_unique: {s.nunique()}")
    print(f"n_null: {s.isna().sum()}")
    print(f"describe:\n{s.describe()}")
    print(f"\ntop 10 value counts:")
    print(s.value_counts(dropna=False).head(10).to_string())
    pct_top_3 = s.value_counts(normalize=True).head(3).sum() * 100
    print(f"\ntop 3 values cover {pct_top_3:.2f}% of all rows")

    print(f"\nAll 7 feature unique-count comparison:")
    for col in FEATURE_COLS:
        if col in tech_df.columns:
            n_uniq = tech_df[col].nunique()
            print(f"  {col}: {n_uniq:,} unique")


def investigate_row_count(tech_df, fae_labeled_path):
    print("\n" + "=" * 60)
    print("Q2: Row count and silhouette reconciliation")
    print("=" * 60)
    print(f"Current technologies.csv rows: {len(tech_df):,}")

    p = Path(fae_labeled_path)
    if not p.exists():
        print(f"FAE labeled CSV not found at {p}")
        return
    labeled = pd.read_csv(p)
    print(f"technologies_labeled_fae_k3_k3.csv rows: {len(labeled):,}")
    print(f"Difference: {len(tech_df) - len(labeled):+,}")

    # Recompute silhouette exactly as 04b_recluster.py would, on BOTH datasets
    # (labeled CSV has 'cluster' column already from the original run)
    sel = ["SIM_TECH", "ACCESS_SIZE", "SIM_ACCESS"]

    print(f"\nRecomputing silhouette on the labeled CSV as-is:")
    X_labeled = StandardScaler().fit_transform(
        labeled[sel].values.astype(np.float64)
    )
    labels_old = labeled["cluster"].values
    sil_old = silhouette_score(X_labeled, labels_old, random_state=42)
    print(f"  Old labels, old features, n={len(labeled):,}: "
          f"silhouette={sil_old:.4f}")

    print(f"\nRe-clustering current technologies.csv with same params:")
    X_new = StandardScaler().fit_transform(
        tech_df[sel].values.astype(np.float64)
    )
    km = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=42)
    labels_new = km.fit_predict(X_new)
    sil_new = silhouette_score(X_new, labels_new, random_state=42)
    print(f"  Fresh clustering, n={len(tech_df):,}: silhouette={sil_new:.4f}")


def main():
    cfg = load_config("config/default.yaml")
    set_seed(cfg["training"]["seed"])
    tech_path = Path(cfg["data"]["processed_dir"]) / "technologies.csv"
    print(f"Loading {tech_path}")
    tech_df = pd.read_csv(tech_path)
    print(f"Loaded {len(tech_df):,} rows")

    investigate_access_trend(tech_df)
    investigate_row_count(
        tech_df,
        "results/clustering/technologies_labeled_fae_k3_k3.csv",
    )


if __name__ == "__main__":
    main()
