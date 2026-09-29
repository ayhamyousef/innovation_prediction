#!/usr/bin/env python3
"""
03_cluster_patterns.py: Optional replication of the trajectory clustering of
Chen et al. (2025), DTW k-means with k=4 on the z-normalized reuse trajectories.

No reported result depends on this stage. The trajectory-shape labels in the
paper use Euclidean k-means on the same z-normalized trajectories (scripts 14,
18 and 27), which Chen et al. report agrees with the DTW clustering on 92.85%
of technologies.

Usage:
    python scripts/03_cluster_patterns.py [--config config/default.yaml]
                                           [--find-k]  # run elbow analysis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clustering import (
    cluster_trajectories, find_optimal_k, characterize_clusters,
    z_score_normalize, CLUSTER_NAMES
)
from src.utils.helpers import load_config, setup_logging, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--find-k", action="store_true",
                        help="Run elbow analysis to find optimal k")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg["training"]["output_dir"])
    clust_cfg = cfg["clustering"]
    processed_dir = Path(cfg["data"]["processed_dir"])

    # Load data
    tech_df = pd.read_csv(processed_dir / "technologies.csv")
    trajectories = np.load(processed_dir / "trajectories.npy")
    logger.info(f"Loaded {len(tech_df)} technologies, "
                f"trajectories shape: {trajectories.shape}")

    # Optional: find optimal k
    if args.find_k:
        logger.info("Running elbow analysis...")
        k_results = find_optimal_k(
            trajectories, k_range=range(2, 11),
            method=clust_cfg["method"],
            random_state=clust_cfg["random_state"]
        )
        save_json(k_results, str(processed_dir / "elbow_analysis.json"))
        logger.info("Elbow analysis saved. SSE values:")
        for k, sse in zip(k_results["k"], k_results["sse"]):
            logger.info(f"  k={k}: SSE={sse:.2f}")

    # Cluster with k=4 (or configured value)
    labels, cluster_info = cluster_trajectories(
        trajectories,
        n_clusters=clust_cfg["n_clusters"],
        method=clust_cfg["method"],
        normalize=clust_cfg["z_score_normalize"],
        n_init=clust_cfg["n_init"],
        max_iter=clust_cfg["max_iter"],
        random_state=clust_cfg["random_state"],
        dtw_window=clust_cfg["dtw_window"],
    )

    # Add labels to DataFrame
    tech_df["cluster"] = labels

    # Characterize clusters
    char_df = characterize_clusters(tech_df, labels, trajectories)
    logger.info("\nCluster characteristics:")
    logger.info(char_df.to_string(index=False))
    char_df.to_csv(processed_dir / "cluster_characteristics.csv", index=False)

    # Save cluster centers
    if "centers" in cluster_info:
        np.save(processed_dir / "cluster_centers.npy", cluster_info["centers"])

    # Label distribution
    logger.info("\nLabel distribution:")
    for c in sorted(set(labels)):
        n = (labels == c).sum()
        name = CLUSTER_NAMES.get(c, f"Cluster_{c}")
        logger.info(f"  {name} (cluster {c}): {n} ({n/len(labels)*100:.1f}%)")

    # Save updated tech_df with labels
    save_df = tech_df.copy()
    for col in save_df.columns:
        if save_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            save_df[col] = save_df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    save_df.to_csv(processed_dir / "technologies_labeled.csv", index=False)
    logger.info(f"Saved labeled technologies to {processed_dir}/technologies_labeled.csv")


if __name__ == "__main__":
    main()
