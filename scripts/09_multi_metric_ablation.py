#!/usr/bin/env python3
"""
09_multi_metric_ablation.py — Re-evaluate size-3 subsets with multiple
clustering metrics.

Silhouette is one of several internal validity criteria, and they do not always
agree. If FAE's subset ranks well across all of them the result is robust; if the
criteria disagree, that disagreement is itself worth reporting.

Computes for each subset:
  - Silhouette (sampled, sample_size=10000 by default for speed)
  - Davies-Bouldin Index (lower is better)
  - Calinski-Harabasz Index (higher is better)

Default: all 35 size-3 subsets (the comparison group for FAE K=3).
Optional: extend to other sizes with --sizes.

Usage:
    # Default: size-3 subsets only, k=3 clusters
    python scripts/09_multi_metric_ablation.py

    # Extend to size-4 too
    python scripts/09_multi_metric_ablation.py --sizes 3 4

    # Exact (un-sampled) silhouette (slower)
    python scripts/09_multi_metric_ablation.py --sample-size 0
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def cluster_and_score(X: np.ndarray, k: int, seed: int, n_init: int,
                      sample_size: int) -> dict:
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=300, random_state=seed)
    labels = km.fit_predict(X)
    if len(set(labels)) < 2:
        return {"silhouette": 0.0, "davies_bouldin": float("inf"),
                "calinski_harabasz": 0.0, "inertia": float(km.inertia_)}
    sil_kwargs = {"random_state": seed}
    if sample_size and sample_size < len(X):
        sil_kwargs["sample_size"] = sample_size
    sil = float(silhouette_score(X, labels, **sil_kwargs))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "inertia": float(km.inertia_),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--sizes", nargs="+", type=int, default=[3])
    parser.add_argument("--cluster-k", type=int, default=3)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Silhouette sample size (0 = exact, slower)")
    parser.add_argument("--fae-k3-path",
                        default="results/feature_selection/fae_k3_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    out_dir = Path("results/multi_metric_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))

    tech_path = Path(cfg["data"]["processed_dir"]) / "technologies.csv"
    logger.info(f"Loading {tech_path}")
    tech_df = pd.read_csv(tech_path)
    logger.info(f"Loaded {len(tech_df):,} rows")

    missing = [c for c in FEATURE_COLS if c not in tech_df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)

    subsets = []
    for s in args.sizes:
        if s < 1 or s > len(FEATURE_COLS):
            continue
        for combo in combinations(FEATURE_COLS, s):
            subsets.append(list(combo))
    logger.info(f"Evaluating {len(subsets)} subsets across sizes {args.sizes}, "
                f"cluster k={args.cluster_k}, "
                f"silhouette sample_size={args.sample_size}")

    rows = []
    for i, subset in enumerate(subsets):
        X = tech_df[subset].values.astype(np.float64)
        X = StandardScaler().fit_transform(X)
        res = cluster_and_score(X, args.cluster_k, seed, args.n_init,
                                args.sample_size)
        rows.append({
            "size": len(subset),
            "features": "|".join(subset),
            "silhouette": res["silhouette"],
            "davies_bouldin": res["davies_bouldin"],
            "calinski_harabasz": res["calinski_harabasz"],
        })
        if (i + 1) % 5 == 0 or i == len(subsets) - 1:
            logger.info(f"  [{i+1}/{len(subsets)}] size={len(subset)} "
                        f"sil={res['silhouette']:.4f} "
                        f"DB={res['davies_bouldin']:.4f} "
                        f"CH={res['calinski_harabasz']:.0f} "
                        f"feats={subset}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"subset_metrics_k{args.cluster_k}.csv", index=False)
    logger.info(f"Saved per-subset metrics to {out_dir}/")

    # ============================================================
    # Per-size best-by-metric ranking
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("BEST SUBSET PER SIZE PER METRIC")
    logger.info("=" * 60)

    metric_dirs = {
        "silhouette": "max",
        "calinski_harabasz": "max",
        "davies_bouldin": "min",
    }

    for size in sorted(df["size"].unique()):
        sub = df[df["size"] == size]
        logger.info(f"\nSize {size} ({len(sub)} subsets):")
        for metric, direction in metric_dirs.items():
            if direction == "max":
                best = sub.nlargest(1, metric).iloc[0]
            else:
                best = sub.nsmallest(1, metric).iloc[0]
            logger.info(f"  best {metric:>20s} ({direction}): "
                        f"{best[metric]:>10.4f}  {best['features']}")

    # ============================================================
    # FAE K=3 rank under each metric
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("FAE K=3 RANK UNDER EACH METRIC")
    logger.info("=" * 60)

    fae_path = Path(args.fae_k3_path)
    fae_rank_info = None
    if fae_path.exists() and 3 in args.sizes:
        with open(fae_path) as f:
            fae_sel = json.load(f)["selection"]["selected_names"]
        fae_key = "|".join(sorted(fae_sel))
        size3 = df[df["size"] == 3].copy()
        size3["features_sorted"] = size3["features"].apply(
            lambda s: "|".join(sorted(s.split("|")))
        )
        match = size3[size3["features_sorted"] == fae_key]
        if len(match) == 1:
            fae_row = match.iloc[0]
            fae_rank_info = {"fae_features": fae_sel}
            logger.info(f"\nFAE K=3 features: {fae_sel}")
            for metric, direction in metric_dirs.items():
                fae_val = float(fae_row[metric])
                if direction == "max":
                    rank = int((size3[metric] > fae_val).sum()) + 1
                else:
                    rank = int((size3[metric] < fae_val).sum()) + 1
                logger.info(f"  {metric:>20s} = {fae_val:>10.4f}  "
                            f"rank {rank} of {len(size3)}")
                fae_rank_info[metric] = {
                    "value": fae_val,
                    "rank": rank,
                    "n_total": len(size3),
                }
        else:
            logger.warning(f"Could not match FAE subset {fae_sel} in size-3 results")

    save_json({
        "cluster_k": args.cluster_k,
        "subset_sizes": args.sizes,
        "sample_size": args.sample_size,
        "fae_k3_rank": fae_rank_info,
    }, str(out_dir / f"summary_k{args.cluster_k}.json"))

    logger.info("\nInterpretation:")
    logger.info("  Higher silhouette = tighter, better-separated clusters")
    logger.info("  Lower Davies-Bouldin = better cluster separation per scatter")
    logger.info("  Higher Calinski-Harabasz = better between/within variance ratio")
    logger.info("\nIf FAE ranks well in 2 of 3 metrics, claim is robust.")


if __name__ == "__main__":
    main()
