#!/usr/bin/env python3
"""
run_experiments.py — Run the full experiment matrix:
  FAE K={3,4,5,all7} × Cluster k={3,4,5} × Model={TabNet,TabM,FT-Transformer}

That's 4 × 3 × 3 = 36 experiments total (27 with FAE + 9 baseline).

Usage:
    # Full matrix (36 experiments):
    python scripts/run_experiments.py --config config/default.yaml

    # Quick test (1 experiment):
    python scripts/run_experiments.py --fae-k 3 --cluster-k 3 --models ft_transformer

    # Just the FAE-selected runs:
    python scripts/run_experiments.py --fae-k 3 4 5 --cluster-k 3 4 5

    # Skip clustering (use pre-computed labeled CSVs):
    python scripts/run_experiments.py --skip-fae --skip-clustering
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]

MODELS = ["tabnet", "tabm", "ft_transformer"]


def run_command(cmd, logger):
    """Run a subprocess command, log output."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed (exit {result.returncode}):")
        logger.error(result.stderr[-2000:] if result.stderr else "no stderr")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fae-k", nargs="+", type=int, default=[3, 4, 5],
                        help="FAE K values to run")
    parser.add_argument("--cluster-k", nargs="+", type=int, default=[3, 4, 5],
                        help="Cluster k values to run")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        choices=MODELS)
    parser.add_argument("--include-baseline", action="store_true", default=True,
                        help="Include no-FAE baseline (all 7 features)")
    parser.add_argument("--skip-fae", action="store_true",
                        help="Skip FAE step (use existing results)")
    parser.add_argument("--skip-clustering", action="store_true",
                        help="Skip clustering step (use existing labeled CSVs)")
    parser.add_argument("--clustering-method", default="euclidean_kmeans",
                        choices=["dtw_kmeans", "euclidean_kmeans"],
                        help="Clustering method (euclidean_kmeans is much faster)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    logger = setup_logging(str(results_dir))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 60)
    logger.info("EXPERIMENT RUNNER")
    logger.info(f"  FAE K values: {args.fae_k}")
    logger.info(f"  Cluster k values: {args.cluster_k}")
    logger.info(f"  Models: {args.models}")
    logger.info(f"  Include baseline: {args.include_baseline}")
    logger.info(f"  Clustering method: {args.clustering_method}")
    logger.info("=" * 60)

    # ============================================================
    # Step 1: FAE Feature Selection
    # ============================================================
    if not args.skip_fae:
        logger.info("\n>>> STEP 1: FAE Feature Selection")
        k_str = " ".join(str(k) for k in args.fae_k)
        cmd = [
            sys.executable, "scripts/04_feature_selection.py",
            "--config", args.config,
            "--k-values", *[str(k) for k in args.fae_k],
        ]
        if not run_command(cmd, logger):
            logger.error("FAE feature selection failed. Aborting.")
            sys.exit(1)
    else:
        logger.info("\n>>> STEP 1: Skipped (--skip-fae)")

    # ============================================================
    # Step 2: Re-cluster with each FAE K + all7 baseline
    # ============================================================
    feature_configs = []  # list of (tag, feature_list, labeled_csv_pattern)

    if not args.skip_clustering:
        logger.info("\n>>> STEP 2: Re-clustering")

        # FAE-selected feature sets
        for fae_k in args.fae_k:
            fae_path = Path(f"results/feature_selection/fae_k{fae_k}_results.json")
            if not fae_path.exists():
                logger.warning(f"FAE K={fae_k} results not found, skipping")
                continue

            cmd = [
                sys.executable, "scripts/04b_recluster.py",
                "--config", args.config,
                "--fae-k", str(fae_k),
                "--cluster-k", *[str(k) for k in args.cluster_k],
                "--method", args.clustering_method,
            ]
            if not run_command(cmd, logger):
                logger.warning(f"Clustering with FAE K={fae_k} failed")
                continue

            # Load selected features for classification
            with open(fae_path) as f:
                sel = json.load(f)["selection"]["selected_names"]
            for ck in args.cluster_k:
                feature_configs.append((
                    f"fae_k{fae_k}_k{ck}",
                    sel,
                    f"results/clustering/technologies_labeled_fae_k{fae_k}_k{ck}.csv"
                ))

        # Baseline: all 7 features
        if args.include_baseline:
            cmd = [
                sys.executable, "scripts/04b_recluster.py",
                "--config", args.config,
                "--cluster-k", *[str(k) for k in args.cluster_k],
                "--method", args.clustering_method,
            ]
            if run_command(cmd, logger):
                for ck in args.cluster_k:
                    feature_configs.append((
                        f"all7_k{ck}",
                        FEATURE_COLS,
                        f"results/clustering/technologies_labeled_all7_k{ck}.csv"
                    ))
    else:
        logger.info("\n>>> STEP 2: Skipped (--skip-clustering)")
        # Build feature_configs from existing files
        for fae_k in args.fae_k:
            fae_path = Path(f"results/feature_selection/fae_k{fae_k}_results.json")
            if fae_path.exists():
                with open(fae_path) as f:
                    sel = json.load(f)["selection"]["selected_names"]
                for ck in args.cluster_k:
                    csv_path = f"results/clustering/technologies_labeled_fae_k{fae_k}_k{ck}.csv"
                    if Path(csv_path).exists():
                        feature_configs.append((f"fae_k{fae_k}_k{ck}", sel, csv_path))

        if args.include_baseline:
            for ck in args.cluster_k:
                csv_path = f"results/clustering/technologies_labeled_all7_k{ck}.csv"
                if Path(csv_path).exists():
                    feature_configs.append((f"all7_k{ck}", FEATURE_COLS, csv_path))

    logger.info(f"\n>>> Experiment configurations: {len(feature_configs)}")
    for tag, feats, csv_path in feature_configs:
        logger.info(f"  {tag}: {feats} -> {csv_path}")

    # ============================================================
    # Step 3: Classification
    # ============================================================
    logger.info("\n>>> STEP 3: Classification")

    all_experiment_results = []

    for tag, feats, csv_path in feature_configs:
        if not Path(csv_path).exists():
            logger.warning(f"Labeled CSV not found: {csv_path}, skipping {tag}")
            continue

        for model_name in args.models:
            experiment_id = f"{tag}_{model_name}"
            logger.info(f"\n--- Experiment: {experiment_id} ---")

            cmd = [
                sys.executable, "scripts/05_classify.py",
                "--config", args.config,
                "--labeled-csv", csv_path,
                "--features", *feats,
                "--models", model_name,
                "--output-dir", f"results/classification/{experiment_id}",
            ]

            success = run_command(cmd, logger)

            # Load results
            results_path = Path(f"results/classification/{experiment_id}/{model_name}_results.json")
            if success and results_path.exists():
                with open(results_path) as f:
                    metrics = json.load(f)
                all_experiment_results.append({
                    "experiment_id": experiment_id,
                    "feature_config": tag,
                    "model": model_name,
                    "n_features": len(feats),
                    "features": ", ".join(feats),
                    "accuracy": metrics.get("accuracy", 0),
                    "macro_f1": metrics.get("macro_f1", 0),
                    "macro_precision": metrics.get("macro_precision", 0),
                    "macro_recall": metrics.get("macro_recall", 0),
                })
            else:
                all_experiment_results.append({
                    "experiment_id": experiment_id,
                    "feature_config": tag,
                    "model": model_name,
                    "n_features": len(feats),
                    "features": ", ".join(feats),
                    "accuracy": "FAILED",
                    "macro_f1": "FAILED",
                    "macro_precision": "FAILED",
                    "macro_recall": "FAILED",
                })

    # ============================================================
    # Final Summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("FULL EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    summary_df = pd.DataFrame(all_experiment_results)
    if not summary_df.empty:
        logger.info("\n" + summary_df.to_string(index=False))
        summary_path = results_dir / f"experiment_summary_{timestamp}.csv"
        summary_df.to_csv(summary_path, index=False)
        # Also save as latest
        summary_df.to_csv(results_dir / "experiment_summary_latest.csv", index=False)
        save_json(all_experiment_results,
                  str(results_dir / "experiment_summary_latest.json"))
        logger.info(f"\nSummary saved to {summary_path}")
    else:
        logger.warning("No experiment results collected!")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
