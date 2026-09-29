#!/usr/bin/env python3
"""
06_ablation_subsets.py: Exhaustive feature-subset ablation.

For every non-empty subset of the 7 features (2^7 - 1 = 127 subsets, or
a restricted size range), cluster with k-means and record silhouette.

Answers three questions at once:
  1. How does the FAE K=3 subset rank among all 3-feature subsets?
     -> rank FAE's subset silhouette among all C(7,3)=35 size-3 subsets.
  2. What is the marginal effect of each feature?
     -> for each feature, compare avg silhouette of subsets containing it
        vs subsets excluding it (the "marginal effect" of that feature).
  3. How does clustering quality vary with subset size?
     -> best silhouette at each subset size.

Usage:
    # Default: all sizes 1..6, k=3 clusters
    python scripts/06_ablation_subsets.py

    # Restrict to sizes 2..4
    python scripts/06_ablation_subsets.py --sizes 2 3 4

    # Different cluster count
    python scripts/06_ablation_subsets.py --cluster-k 4
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def cluster_and_score(X: np.ndarray, k: int, seed: int, n_init: int) -> dict:
    """Cluster a feature matrix with k-means, return silhouette + sizes."""
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=300, random_state=seed)
    labels = km.fit_predict(X)
    if len(set(labels)) < 2:
        return {"silhouette": 0.0, "inertia": float(km.inertia_),
                "cluster_sizes": {0: len(labels)}}
    sil = float(silhouette_score(X, labels))
    sizes = {int(c): int((labels == c).sum()) for c in sorted(set(labels))}
    min_pct = min(sizes.values()) / len(labels) * 100
    return {
        "silhouette": sil,
        "inertia": float(km.inertia_),
        "cluster_sizes": sizes,
        "min_cluster_pct": min_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Subset sizes to evaluate (default 1..6)")
    parser.add_argument("--cluster-k", type=int, default=3,
                        help="Number of clusters for each subset")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--fae-k3-path",
                        default="results/feature_selection/fae_k3_results.json",
                        help="Path to FAE K=3 results for comparison")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    out_dir = Path("results/ablation_subsets")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))

    # Load tech data
    tech_path = Path(cfg["data"]["processed_dir"]) / "technologies.csv"
    logger.info(f"Loading {tech_path}")
    tech_df = pd.read_csv(tech_path)
    logger.info(f"Loaded {len(tech_df)} technologies")

    # Validate feature columns exist
    missing = [c for c in FEATURE_COLS if c not in tech_df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)

    # Enumerate subsets
    all_subsets = []
    for s in args.sizes:
        if s < 1 or s > len(FEATURE_COLS):
            continue
        for combo in combinations(FEATURE_COLS, s):
            all_subsets.append(list(combo))
    logger.info(f"Evaluating {len(all_subsets)} subsets across sizes {args.sizes}")
    logger.info(f"Clustering with k={args.cluster_k}")

    # Run clustering on each subset
    rows = []
    for i, subset in enumerate(all_subsets):
        X = tech_df[subset].values.astype(np.float64)
        X = StandardScaler().fit_transform(X)
        # Silhouette with k=1 feature is undefined meaningfully but sklearn handles it
        res = cluster_and_score(X, args.cluster_k, seed, args.n_init)
        rows.append({
            "size": len(subset),
            "features": "|".join(subset),
            "silhouette": res["silhouette"],
            "inertia": res["inertia"],
            "min_cluster_pct": res.get("min_cluster_pct", 0.0),
        })
        if (i + 1) % 10 == 0 or i == len(all_subsets) - 1:
            logger.info(f"  [{i+1}/{len(all_subsets)}] "
                        f"size={len(subset)} sil={res['silhouette']:.4f} "
                        f"feats={subset}")

    df = pd.DataFrame(rows).sort_values(
        ["size", "silhouette"], ascending=[True, False]
    ).reset_index(drop=True)
    df_path = out_dir / f"subset_silhouettes_k{args.cluster_k}.csv"
    df.to_csv(df_path, index=False)
    logger.info(f"\nSaved per-subset results to {df_path}")

    # ============================================================
    # Analysis 1: best per size (fewer-features sweep)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 1: Best silhouette by subset size")
    logger.info("=" * 60)
    best_per_size_rows = []
    for size in sorted(df["size"].unique()):
        sub = df[df["size"] == size]
        best = sub.iloc[0]
        best_per_size_rows.append({
            "size": int(size),
            "n_subsets": len(sub),
            "best_silhouette": float(best["silhouette"]),
            "best_features": best["features"],
            "mean_silhouette": float(sub["silhouette"].mean()),
            "worst_silhouette": float(sub["silhouette"].min()),
        })
        logger.info(f"  size={size}: best={best['silhouette']:.4f} "
                    f"({best['features']}), mean={sub['silhouette'].mean():.4f}")
    best_per_size = pd.DataFrame(best_per_size_rows)
    best_per_size.to_csv(out_dir / f"best_per_size_k{args.cluster_k}.csv",
                         index=False)

    # ============================================================
    # Analysis 2: FAE K=3 rank among size-3 subsets
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 2: FAE K=3 vs all size-3 subsets")
    logger.info("=" * 60)
    fae_rank_info = None
    fae_path = Path(args.fae_k3_path)
    if fae_path.exists():
        with open(fae_path) as f:
            fae_sel = json.load(f)["selection"]["selected_names"]
        fae_key = "|".join(sorted(fae_sel))
        size3 = df[df["size"] == 3].copy().reset_index(drop=True)
        size3["features_sorted"] = size3["features"].apply(
            lambda s: "|".join(sorted(s.split("|")))
        )
        match = size3[size3["features_sorted"] == fae_key]
        if len(match) == 1:
            fae_row = match.iloc[0]
            fae_sil = float(fae_row["silhouette"])
            # Rank: how many size-3 subsets match or exceed FAE?
            better = int((size3["silhouette"] > fae_sil).sum())
            equal = int((size3["silhouette"] == fae_sil).sum())
            rank = better + 1  # 1 = best
            logger.info(f"  FAE K=3 features: {fae_sel}")
            logger.info(f"  FAE silhouette: {fae_sil:.4f}")
            logger.info(f"  Rank: {rank} of {len(size3)} size-3 subsets")
            logger.info(f"  Subsets beating FAE: {better}")
            logger.info(f"  Mean size-3 silhouette: {size3['silhouette'].mean():.4f}")
            logger.info(f"  Top 5 size-3 subsets:")
            for _, r in size3.head(5).iterrows():
                marker = "  <- FAE" if r["features_sorted"] == fae_key else ""
                logger.info(f"    {r['silhouette']:.4f}  {r['features']}{marker}")
            fae_rank_info = {
                "fae_features": fae_sel,
                "fae_silhouette": fae_sil,
                "rank_of_35": rank,
                "subsets_beating_fae": better,
                "subsets_tying_fae": equal - 1,  # exclude FAE itself
                "mean_size3_silhouette": float(size3["silhouette"].mean()),
                "percentile": 100.0 * (len(size3) - rank + 1) / len(size3),
            }
        else:
            logger.warning(f"Could not match FAE subset {fae_sel} in size-3 results")
    else:
        logger.warning(f"FAE K=3 results not found at {fae_path}")

    # ============================================================
    # Analysis 3: marginal effect of each feature
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS 3: Marginal effect of each feature on silhouette")
    logger.info("(avg silhouette of subsets containing feature vs not)")
    logger.info("=" * 60)
    marginal_rows = []
    for feat in FEATURE_COLS:
        contains = df[df["features"].apply(lambda s: feat in s.split("|"))]
        without = df[~df["features"].apply(lambda s: feat in s.split("|"))]
        if len(contains) == 0 or len(without) == 0:
            continue
        marginal_rows.append({
            "feature": feat,
            "n_subsets_with": len(contains),
            "n_subsets_without": len(without),
            "mean_sil_with": float(contains["silhouette"].mean()),
            "mean_sil_without": float(without["silhouette"].mean()),
            "effect": float(contains["silhouette"].mean() -
                            without["silhouette"].mean()),
        })
    marginal = pd.DataFrame(marginal_rows).sort_values("effect", ascending=False)
    logger.info("\n" + marginal.to_string(index=False,
                                           float_format=lambda x: f"{x:.4f}"))
    marginal.to_csv(out_dir / f"feature_marginal_effect_k{args.cluster_k}.csv",
                    index=False)
    logger.info("\n  Positive effect = feature helps clustering when included.")
    logger.info("  Negative effect = feature hurts clustering when included.")

    # ============================================================
    # Save everything
    # ============================================================
    summary = {
        "cluster_k": args.cluster_k,
        "subset_sizes": args.sizes,
        "n_subsets_evaluated": len(all_subsets),
        "best_per_size": best_per_size_rows,
        "fae_k3_comparison": fae_rank_info,
        "marginal_effects": marginal.to_dict(orient="records"),
    }
    save_json(summary, str(out_dir / f"ablation_summary_k{args.cluster_k}.json"))
    logger.info(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
