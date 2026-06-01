#!/usr/bin/env python3
"""
12_training_fraction_sweep.py — Test whether DL holds up better than GBDT
under reduced training data.

Motivated by Dr. Cheng's question (2026-04-29): if GBDT already achieves
near-100% accuracy on FAE-selected features, why use DL? Cheng's option 3 was
to test under harder conditions where DL might genuinely help.

For each training fraction in {0.01, 0.05, 0.10, 0.25, 0.50, 1.00} and each
model in {gbdt, tabm, ft_transformer, extra_trees, tabnet}:
  - Subsample the train set stratified by class
  - Train the model
  - Evaluate on the FULL test set
  - Record accuracy, macro-F1, train-test gap

Default config: FAE K=3 selected features, k=3 clusters (the headline config).

Usage:
    # Default sweep
    python scripts/12_training_fraction_sweep.py

    # Quick test
    python scripts/12_training_fraction_sweep.py \\
        --fractions 0.05 0.5 1.0 \\
        --models gbdt tabm

    # All-7 features instead of FAE K=3 (to compare)
    python scripts/12_training_fraction_sweep.py \\
        --features ACCESS_SIZE ACCESS_TREND SIM_ACCESS SIM_TECH INVENT_DIVER INVENT_APPL ATTENT_SIZE \\
        --tag all7
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tabular_models import build_tabular_model
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]
DEFAULT_MODELS = ["gbdt", "tabm", "ft_transformer", "extra_trees", "tabnet"]
DEFAULT_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]


def fae_k3_features():
    """Load FAE K=3 selected features from the previous FAE run."""
    p = Path("results/feature_selection/fae_k3_results.json")
    if not p.exists():
        return ["SIM_TECH", "ACCESS_SIZE", "SIM_ACCESS"]  # fallback
    with open(p) as f:
        return json.load(f)["selection"]["selected_names"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Features to use. Default: FAE K=3 selection.")
    parser.add_argument("--tag", default="fae_k3",
                        help="Tag for output dir (e.g. 'fae_k3' or 'all7')")
    parser.add_argument("--cluster-k", type=int, default=3)
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=DEFAULT_FRACTIONS,
                        help="Training-set fractions to sweep")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=["tabnet", "tabm", "ft_transformer",
                                 "extra_trees", "gbdt"])
    parser.add_argument("--test-ratio", type=float, default=0.4)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds per (fraction, model) combo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    # Features
    if args.features is None:
        features = fae_k3_features()
    else:
        features = list(args.features)

    out_dir = Path(f"results/training_fraction_sweep/{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))
    logger.info(f"Features ({args.tag}): {features}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Fractions: {args.fractions}")
    logger.info(f"Seeds per combo: {args.n_seeds}")

    # Load data
    tech_path = Path(cfg["data"]["processed_dir"]) / "technologies.csv"
    tech_df = pd.read_csv(tech_path)
    logger.info(f"Loaded {len(tech_df):,} rows")

    # Build feature matrix from raw features
    X_raw = tech_df[features].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # Step 1: cluster on full-data-standardized features to generate labels.
    # Clustering is unsupervised so using all data here is not classification
    # leakage; it is just label generation.
    X_for_clustering = StandardScaler().fit_transform(X_raw)
    km = KMeans(n_clusters=args.cluster_k, n_init=10, max_iter=300,
                random_state=seed)
    y_full = km.fit_predict(X_for_clustering)
    n_classes = args.cluster_k
    logger.info(f"Cluster sizes: "
                f"{dict((int(c), int((y_full == c).sum())) for c in range(n_classes))}")

    # Step 2: stratified 60/40 split on RAW features (test set fixed across runs)
    X_train_raw, X_test_raw, y_train_full, y_test = train_test_split(
        X_raw, y_full, test_size=args.test_ratio,
        random_state=seed, stratify=y_full,
    )
    logger.info(f"Full train n={len(X_train_raw):,} | Test n={len(X_test_raw):,}")

    # Detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    logger.info(f"Device for DL models: {device}")

    rows = []
    total = len(args.fractions) * len(args.models) * args.n_seeds
    run_i = 0

    for frac in args.fractions:
        n_sub = max(int(len(X_train_raw) * frac), n_classes * 10)
        for seed_offset in range(args.n_seeds):
            sub_seed = seed + seed_offset * 1000
            # Stratified subsample of the train set (still raw)
            if frac < 1.0:
                X_sub_raw, _, y_sub, _ = train_test_split(
                    X_train_raw, y_train_full, train_size=n_sub,
                    random_state=sub_seed, stratify=y_train_full,
                )
            else:
                X_sub_raw, y_sub = X_train_raw, y_train_full

            # 90/10 train/val within the subsample, still raw
            X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
                X_sub_raw, y_sub, test_size=0.1,
                random_state=sub_seed, stratify=y_sub,
            )

            # Refit StandardScaler on the actual training subset only.
            # This is what a real reduced-data setting would have access to.
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr_raw)
            X_val = sc.transform(X_val_raw)
            X_test = sc.transform(X_test_raw)

            for model_name in args.models:
                run_i += 1
                logger.info("=" * 60)
                logger.info(f"[{run_i}/{total}] "
                            f"frac={frac} model={model_name} "
                            f"seed_offset={seed_offset} n_train={len(X_tr):,}")
                logger.info("=" * 60)
                try:
                    model = build_tabular_model(
                        model_name=model_name,
                        n_features=len(features),
                        n_classes=n_classes,
                        device=device,
                        seed=sub_seed,
                    )
                    model.fit(X_tr, y_tr, X_val, y_val)
                    y_pred = model.predict(X_test)
                    y_train_pred = model.predict(X_tr)
                    test_acc = float(accuracy_score(y_test, y_pred))
                    test_f1 = float(f1_score(y_test, y_pred, average="macro"))
                    train_acc = float(accuracy_score(y_tr, y_train_pred))
                    rows.append({
                        "fraction": frac,
                        "n_train": len(X_tr),
                        "model": model_name,
                        "seed_offset": seed_offset,
                        "test_accuracy": test_acc,
                        "test_macro_f1": test_f1,
                        "train_accuracy": train_acc,
                        "train_test_gap": train_acc - test_acc,
                        "status": "OK",
                    })
                    logger.info(f"  test acc={test_acc:.4f} | "
                                f"macro-F1={test_f1:.4f} | "
                                f"train-test gap={train_acc - test_acc:+.4f}")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    rows.append({
                        "fraction": frac, "n_train": len(X_tr),
                        "model": model_name, "seed_offset": seed_offset,
                        "test_accuracy": None, "test_macro_f1": None,
                        "train_accuracy": None, "train_test_gap": None,
                        "status": f"FAILED: {e}",
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fraction_sweep.csv", index=False)
    logger.info(f"\nSaved per-run results to {out_dir}/")

    # ============================================================
    # Aggregated per-(fraction, model) summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING-FRACTION SWEEP SUMMARY")
    logger.info("=" * 60)

    # Pivot for easy reading
    if args.n_seeds == 1:
        summary = df.pivot(index="fraction", columns="model",
                           values="test_accuracy")
        summary_f1 = df.pivot(index="fraction", columns="model",
                              values="test_macro_f1")
    else:
        summary = df.groupby(["fraction", "model"])["test_accuracy"].agg(
            ["mean", "std"]
        ).unstack("model")
        summary_f1 = df.groupby(["fraction", "model"])["test_macro_f1"].agg(
            ["mean", "std"]
        ).unstack("model")

    logger.info("\nTest accuracy by (fraction, model):")
    logger.info("\n" + summary.to_string())
    logger.info("\nTest macro-F1 by (fraction, model):")
    logger.info("\n" + summary_f1.to_string())

    summary.to_csv(out_dir / "summary_accuracy.csv")
    summary_f1.to_csv(out_dir / "summary_macro_f1.csv")

    save_json({
        "tag": args.tag,
        "features": features,
        "cluster_k": args.cluster_k,
        "fractions": args.fractions,
        "models": args.models,
        "n_seeds": args.n_seeds,
        "test_size": len(X_test),
        "train_size_full": len(X_train_full),
    }, str(out_dir / "config.json"))

    # ============================================================
    # Per-model robustness diagnostic
    # ============================================================
    logger.info("\nDoes DL hold up better than GBDT under shrinking data?")
    logger.info("(test accuracy at smallest vs largest fraction)")
    smallest = min(args.fractions)
    largest = max(args.fractions)
    for m in args.models:
        try:
            small_acc = df[(df["fraction"] == smallest) &
                           (df["model"] == m)]["test_accuracy"].mean()
            large_acc = df[(df["fraction"] == largest) &
                           (df["model"] == m)]["test_accuracy"].mean()
            drop = large_acc - small_acc
            logger.info(f"  {m:>16}: {large_acc:.4f} (frac={largest}) -> "
                        f"{small_acc:.4f} (frac={smallest})  "
                        f"drop={drop:+.4f}")
        except Exception:
            pass

    logger.info(f"\nAll outputs: {out_dir}/")


if __name__ == "__main__":
    main()
