#!/usr/bin/env python3
"""
04b_recluster.py — Re-cluster technology trajectories using FAE-selected
features and variable cluster counts.

Takes FAE output (selected feature indices) and runs DTW k-means
with k=3, k=4, k=5. Reports clustering quality metrics.

Usage:
    python scripts/04b_recluster.py [--config config/default.yaml]
                                     [--fae-k 3]
                                     [--cluster-k 3 4 5]
                                     [--method dtw_kmeans]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clustering import (
    cluster_trajectories, characterize_clusters,
    compute_clustering_quality, CLUSTER_NAMES,
)
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fae-k", type=int, default=None,
                        help="FAE K value to use (loads from results/feature_selection/). "
                             "If None, uses all 7 features (no selection).")
    parser.add_argument("--cluster-k", nargs="+", type=int, default=[3, 4, 5],
                        help="Number of clusters to try")
    parser.add_argument("--method", default="dtw_kmeans",
                        choices=["dtw_kmeans", "euclidean_kmeans"])
    parser.add_argument("--n-init", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    clust_cfg = cfg["clustering"]

    results_dir = Path("results/clustering")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(results_dir))

    # Load trajectories and tech data
    processed_dir = Path(cfg["data"]["processed_dir"])
    trajectories = np.load(processed_dir / "trajectories.npy")
    tech_df = pd.read_csv(processed_dir / "technologies.csv")

    logger.info(f"Loaded {len(tech_df)} technologies, "
                f"trajectories shape: {trajectories.shape}")

    # Determine which features to use
    if args.fae_k is not None:
        fae_results_path = Path(f"results/feature_selection/fae_k{args.fae_k}_results.json")
        if not fae_results_path.exists():
            logger.error(f"FAE results not found: {fae_results_path}")
            logger.error("Run scripts/04_feature_selection.py first.")
            sys.exit(1)

        with open(fae_results_path) as f:
            fae_results = json.load(f)
        selected_names = fae_results["selection"]["selected_names"]
        feature_tag = f"fae_k{args.fae_k}"
        logger.info(f"Using FAE-selected features (K={args.fae_k}): {selected_names}")
    else:
        selected_names = FEATURE_COLS
        feature_tag = "all7"
        logger.info(f"Using all 7 features (no FAE selection)")

    # Run clustering for each k
    all_results = {}

    for k in args.cluster_k:
        logger.info("=" * 60)
        logger.info(f"Clustering with k={k}, features={selected_names}, "
                     f"method={args.method}")
        logger.info("=" * 60)

        labels, info = cluster_trajectories(
            trajectories,
            n_clusters=k,
            method=args.method,
            normalize=clust_cfg["z_score_normalize"],
            n_init=args.n_init,
            max_iter=clust_cfg["max_iter"],
            random_state=clust_cfg["random_state"],
            dtw_window=clust_cfg["dtw_window"],
        )

        # Quality metrics
        centers = info.get("centers")
        quality = compute_clustering_quality(trajectories, labels, centers)

        # Characterize clusters
        char_df = characterize_clusters(tech_df, labels, trajectories)
        logger.info(f"\nCluster characteristics (k={k}):")
        logger.info(char_df.to_string(index=False))

        # Label distribution
        logger.info(f"\nLabel distribution:")
        for c in sorted(set(labels)):
            n = int((labels == c).sum())
            pct = n / len(labels) * 100
            logger.info(f"  Cluster {c}: {n:,} ({pct:.1f}%)")

        logger.info(f"Silhouette score: {quality['silhouette']:.4f}")
        logger.info(f"Min cluster: {quality['min_cluster_pct']:.1f}%")
        if "mean_centroid_distance" in quality:
            logger.info(f"Mean centroid distance: {quality['mean_centroid_distance']:.4f}")

        # Save results for this k
        run_key = f"{feature_tag}_k{k}"
        run_results = {
            "feature_tag": feature_tag,
            "selected_features": selected_names,
            "n_clusters": k,
            "method": args.method,
            "quality": quality,
            "cluster_sizes": quality["cluster_sizes"],
            "inertia": info.get("inertia"),
            "silhouette": quality["silhouette"],
        }
        all_results[run_key] = run_results

        # Save labeled tech_df for this configuration
        labeled_df = tech_df.copy()
        labeled_df["cluster"] = labels

        # Convert list/dict columns for CSV
        for col in labeled_df.columns:
            if labeled_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                labeled_df[col] = labeled_df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )

        out_path = results_dir / f"technologies_labeled_{run_key}.csv"
        labeled_df.to_csv(out_path, index=False)
        logger.info(f"Saved labeled data to {out_path}")

        # Save cluster centers
        if centers is not None:
            np.save(results_dir / f"cluster_centers_{run_key}.npy", centers)

        # Save characteristics
        char_df.to_csv(results_dir / f"cluster_chars_{run_key}.csv", index=False)

    # Summary comparison across k values
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTERING COMPARISON SUMMARY")
    logger.info("=" * 60)

    summary_rows = []
    for key, r in all_results.items():
        summary_rows.append({
            "config": key,
            "features": ", ".join(r["selected_features"]),
            "n_clusters": r["n_clusters"],
            "silhouette": r["silhouette"],
            "inertia": r["inertia"],
            "min_cluster_pct": r["quality"]["min_cluster_pct"],
        })

    summary_df = pd.DataFrame(summary_rows)
    logger.info("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(results_dir / f"clustering_summary_{feature_tag}.csv", index=False)

    save_json(all_results, str(results_dir / f"clustering_results_{feature_tag}.json"))
    logger.info(f"\nAll results saved to {results_dir}/")


if __name__ == "__main__":
    main()
