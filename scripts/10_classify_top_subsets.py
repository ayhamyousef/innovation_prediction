#!/usr/bin/env python3
"""
10_classify_top_subsets.py — Classification on top-silhouette subsets and FAE.

Per Dr. Cheng (2026-04-22): "use downstream prediction/classification results
to speak. This is more certain, because for classification/prediction we have
ground truth."

For each subset in the candidate list:
  1. Cluster (KMeans) on the standardized feature matrix to produce labels
  2. Train TabNet, TabM, FT-Transformer
  3. Report test accuracy and macro-F1

Default candidates: top 5 size-3 subsets by silhouette + FAE K=3 + all-7 baseline.

Usage:
    # Default (uses top-5 from results/ablation_subsets/subset_silhouettes_k3.csv
    # plus FAE K=3 and all-7 baseline)
    python scripts/10_classify_top_subsets.py

    # Specify candidates manually
    python scripts/10_classify_top_subsets.py --subsets \\
        "ACCESS_SIZE,ACCESS_TREND,SIM_ACCESS" \\
        "SIM_TECH,ACCESS_SIZE,SIM_ACCESS"

    # Just one model
    python scripts/10_classify_top_subsets.py --models tabm
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]
MODELS = ["tabnet", "tabm", "ft_transformer"]


def load_top_subsets(silhouette_csv: Path, top_n: int) -> list:
    """Load top-N size-3 subsets from the ablation output CSV."""
    if not silhouette_csv.exists():
        return []
    df = pd.read_csv(silhouette_csv)
    df3 = df[df["size"] == 3].sort_values("silhouette", ascending=False)
    return [row["features"].split("|") for _, row in df3.head(top_n).iterrows()]


def cluster_subset(tech_df, features, k, seed):
    """Cluster on the given feature subset; return labels and the labeled DataFrame."""
    X = tech_df[features].values.astype(np.float64)
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=seed)
    labels = km.fit_predict(X)
    labeled = tech_df.copy()
    labeled["cluster"] = labels
    return labeled, km.inertia_


def run_classify(labeled_csv: str, features: list, models: list,
                 output_dir: str, config: str, logger):
    """Invoke 05_classify.py via subprocess."""
    cmd = [
        sys.executable, "scripts/05_classify.py",
        "--config", config,
        "--labeled-csv", labeled_csv,
        "--features", *features,
        "--models", *models,
        "--output-dir", output_dir,
    ]
    logger.info(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  classify failed (exit {result.returncode})")
        logger.error(result.stderr[-1500:] if result.stderr else "no stderr")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--cluster-k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top size-3 subsets to evaluate")
    parser.add_argument("--silhouette-csv",
                        default="results/ablation_subsets/subset_silhouettes_k3.csv")
    parser.add_argument("--fae-k3-path",
                        default="results/feature_selection/fae_k3_results.json")
    parser.add_argument("--include-baseline", action="store_true", default=True,
                        help="Include all-7-features baseline")
    parser.add_argument("--subsets", nargs="+", default=None,
                        help="Manual list of subsets, comma-separated. "
                             "Overrides --top-n and FAE/baseline auto-selection.")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--output-root", default="results/top_subsets_classify")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_root))

    # Build candidate list
    candidates = []  # list of (tag, features)

    if args.subsets is not None:
        for i, s in enumerate(args.subsets):
            feats = [f.strip() for f in s.split(",")]
            candidates.append((f"manual_{i:02d}", feats))
    else:
        top = load_top_subsets(Path(args.silhouette_csv), args.top_n)
        for i, feats in enumerate(top, start=1):
            candidates.append((f"silhouette_top{i}", feats))

        fae_path = Path(args.fae_k3_path)
        if fae_path.exists():
            with open(fae_path) as f:
                fae_sel = json.load(f)["selection"]["selected_names"]
            fae_key = "|".join(sorted(fae_sel))
            already_present = any("|".join(sorted(c[1])) == fae_key
                                  for c in candidates)
            if not already_present:
                candidates.append(("fae_k3", fae_sel))
            else:
                logger.info("FAE K=3 subset is already in the top-N; "
                            "no separate fae_k3 entry added.")

        if args.include_baseline:
            candidates.append(("baseline_all7", FEATURE_COLS))

    logger.info(f"Evaluating {len(candidates)} candidate subsets")
    for tag, feats in candidates:
        logger.info(f"  {tag}: {feats}")

    # Load tech data once
    tech_path = Path(cfg["data"]["processed_dir"]) / "technologies.csv"
    tech_df = pd.read_csv(tech_path)

    # Output directory for clustered CSVs
    cluster_dir = out_root / "clustered_csvs"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for tag, features in candidates:
        logger.info("\n" + "=" * 60)
        logger.info(f"CANDIDATE: {tag}")
        logger.info(f"Features: {features}")
        logger.info("=" * 60)

        # Cluster
        labeled, inertia = cluster_subset(tech_df, features,
                                          args.cluster_k, seed)
        labeled_csv = cluster_dir / f"labeled_{tag}.csv"
        # Drop list/dict columns before writing (matches 04b_recluster behavior)
        for col in labeled.columns:
            if labeled[col].apply(lambda x: isinstance(x, (list, dict))).any():
                labeled[col] = labeled[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )
        labeled.to_csv(labeled_csv, index=False)
        logger.info(f"  saved labeled CSV: {labeled_csv}")

        # Classify
        classify_out = out_root / f"classify_{tag}"
        success = run_classify(
            str(labeled_csv), features, args.models,
            str(classify_out), args.config, logger,
        )
        if not success:
            all_results.append({
                "tag": tag,
                "features": ", ".join(features),
                "n_features": len(features),
                "status": "FAILED",
            })
            continue

        # Read per-model JSON outputs
        for model_name in args.models:
            res_path = classify_out / f"{model_name}_results.json"
            if not res_path.exists():
                all_results.append({
                    "tag": tag,
                    "features": ", ".join(features),
                    "n_features": len(features),
                    "model": model_name,
                    "status": "MISSING",
                })
                continue
            with open(res_path) as f:
                m = json.load(f)
            all_results.append({
                "tag": tag,
                "features": ", ".join(features),
                "n_features": len(features),
                "model": model_name,
                "accuracy": m.get("accuracy"),
                "macro_f1": m.get("macro_f1"),
                "macro_precision": m.get("macro_precision"),
                "macro_recall": m.get("macro_recall"),
                "status": "OK",
            })

    # ============================================================
    # Summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("TOP-SUBSET CLASSIFICATION SUMMARY")
    logger.info("=" * 60)

    summary = pd.DataFrame(all_results)
    summary_path = out_root / f"summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(out_root / "summary_latest.csv", index=False)
    logger.info("\n" + summary.to_string(index=False))
    logger.info(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
