#!/usr/bin/env python3
"""
04_feature_selection.py — Run FAE unsupervised feature selection on the
7 Chen et al. technology features.

Tests K=3, K=4, K=5 (number of features to select out of 7).
Saves selected features, importance weights, and reconstruction errors.

Usage:
    python scripts/04_feature_selection.py [--config config/default.yaml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fae import train_fae
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 4, 5],
                        help="Values of K (features to select)")
    parser.add_argument("--lambda1", type=float, default=2.0)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    results_dir = Path("results/feature_selection")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(results_dir))

    # Load technology data (from script 02 output)
    processed_dir = Path(cfg["data"]["processed_dir"])
    tech_path = processed_dir / "technologies.csv"

    logger.info(f"Loading technology data from {tech_path}")
    tech_df = pd.read_csv(tech_path)
    logger.info(f"Loaded {len(tech_df)} technologies")

    # Extract the 7 features
    existing_cols = [c for c in FEATURE_COLS if c in tech_df.columns]
    if len(existing_cols) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(existing_cols)
        logger.warning(f"Missing feature columns: {missing}")

    X_raw = tech_df[existing_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # Z-score normalize for FAE input
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std == 0] = 1.0
    X = (X_raw - X_mean) / X_std

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {existing_cols}")
    logger.info(f"Feature means: {dict(zip(existing_cols, X_mean.tolist()))}")
    logger.info(f"Feature stds:  {dict(zip(existing_cols, X_std.tolist()))}")

    # Run FAE for each K value
    all_results = {}

    for k in args.k_values:
        if k >= len(existing_cols):
            logger.warning(f"K={k} >= num features ({len(existing_cols)}), skipping")
            continue

        logger.info("=" * 60)
        logger.info(f"Running FAE with K={k} (select {k} out of {len(existing_cols)})")
        logger.info("=" * 60)

        model, results = train_fae(
            X=X,
            k=k,
            feature_names=existing_cols,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device="cpu",  # 7 features, CPU is fine
            verbose=True,
        )

        all_results[f"K={k}"] = results

        # Save per-K results
        save_json(results, str(results_dir / f"fae_k{k}_results.json"))

    # Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("FAE FEATURE SELECTION SUMMARY")
    logger.info("=" * 60)

    summary_rows = []
    for k in args.k_values:
        key = f"K={k}"
        if key not in all_results:
            continue
        r = all_results[key]
        sel = r["selection"]
        summary_rows.append({
            "K": k,
            "selected_features": ", ".join(sel.get("selected_names", [])),
            "recon_global": r["final_recon_global"],
            "recon_sub": r["final_recon_sub"],
            "total_loss": r["final_loss"],
            "epochs": r["epochs_trained"],
        })
        logger.info(f"\nK={k}: {sel.get('selected_names', sel['selected_indices'])}")
        logger.info(f"  Recon (global): {r['final_recon_global']:.4f}")
        logger.info(f"  Recon (sub-NN): {r['final_recon_sub']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "fae_summary.csv", index=False)

    # Save all results
    save_json(all_results, str(results_dir / "fae_all_results.json"))

    logger.info(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
