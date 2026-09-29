#!/usr/bin/env python3
"""
05_classify.py: Train the tabular classifiers on clustered technology data.

The classification stage of the pipeline:
  - Load pre-clustered data (from 04b_recluster.py output)
  - Split: 60% train / 40% test, within train 90% train / 10% val
  - Train any of the seven classifiers (--models; default TabNet, TabM, FT-Transformer)
  - Report accuracy, macro F1, per-class metrics

Usage:
    python scripts/05_classify.py --labeled-csv results/clustering/technologies_labeled_fae_k3_k3.csv
                                   --features ACCESS_SIZE ACCESS_TREND SIM_TECH
                                   [--models tabnet tabm ft_transformer]
                                   [--config config/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tabular_models import build_tabular_model
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def compute_classification_metrics(y_true, y_pred, y_proba=None, n_classes=None):
    """Compute classification metrics. y_proba enables cross-entropy and ROC-AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "per_class": classification_report(y_true, y_pred, output_dict=True),
    }
    if y_proba is not None:
        labels = list(range(n_classes)) if n_classes else None
        try:
            metrics["cross_entropy"] = float(log_loss(y_true, y_proba, labels=labels))
        except Exception as e:
            metrics["cross_entropy"] = None
            metrics["cross_entropy_error"] = str(e)
        try:
            if n_classes and n_classes > 2:
                metrics["roc_auc_ovr_macro"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro",
                    labels=labels,
                ))
                metrics["roc_auc_ovo_macro"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovo", average="macro",
                    labels=labels,
                ))
            else:
                # binary: use positive-class probability
                pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, pos))
        except Exception as e:
            metrics["roc_auc_error"] = str(e)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--labeled-csv", required=True,
                        help="Path to labeled technologies CSV "
                             "(output from 04b_recluster.py)")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Feature columns to use. If None, uses all 7.")
    parser.add_argument("--models", nargs="+",
                        default=["tabnet", "tabm", "ft_transformer"],
                        choices=["tabnet", "tabm", "ft_transformer",
                                 "extra_trees", "gbdt",
                                 "tabkan", "tabmixer"])
    parser.add_argument("--test-ratio", type=float, default=0.4,
                        help="Fraction for test set (default 0.4)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of train set for validation (default 0.1)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if None)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    # Determine features
    feature_cols = args.features if args.features else FEATURE_COLS

    # Auto-generate output dir from labeled CSV name
    if args.output_dir is None:
        csv_stem = Path(args.labeled_csv).stem  # e.g. "technologies_labeled_fae_k3_k3"
        args.output_dir = f"results/classification/{csv_stem}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir))

    # Load data
    logger.info(f"Loading labeled data from {args.labeled_csv}")
    tech_df = pd.read_csv(args.labeled_csv)
    logger.info(f"Loaded {len(tech_df)} technologies")

    # Verify columns exist
    missing = [c for c in feature_cols if c not in tech_df.columns]
    if missing:
        logger.error(f"Missing feature columns: {missing}")
        logger.error(f"Available columns: {list(tech_df.columns)}")
        sys.exit(1)

    if "cluster" not in tech_df.columns:
        logger.error("No 'cluster' column found. Run 04b_recluster.py first.")
        sys.exit(1)

    n_classes = tech_df["cluster"].nunique()
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Classes: {n_classes}")
    logger.info(f"Class distribution:")
    for c in sorted(tech_df["cluster"].unique()):
        n = (tech_df["cluster"] == c).sum()
        logger.info(f"  Cluster {c}: {n:,} ({n/len(tech_df)*100:.1f}%)")

    # Extract features and labels
    X = tech_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = tech_df["cluster"].values.astype(np.int64)

    # Split: 60% train / 40% test (stratified)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_ratio,
        stratify=y, random_state=seed
    )

    # Within train: 90% train / 10% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_ratio,
        stratify=y_train_full, random_state=seed
    )

    logger.info(f"Split sizes, train: {len(y_train)}, "
                f"val: {len(y_val)}, test: {len(y_test)}")

    # Normalize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    logger.info(f"Device: {device}")

    # Train and evaluate each model
    all_results = {}

    for model_name in args.models:
        logger.info("=" * 60)
        logger.info(f"Training: {model_name}")
        logger.info("=" * 60)

        try:
            model = build_tabular_model(
                model_name=model_name,
                n_features=len(feature_cols),
                n_classes=n_classes,
                device=device,
                seed=seed,
            )

            model.fit(X_train, y_train, X_val, y_val)

            # Evaluate on test set
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            metrics = compute_classification_metrics(
                y_test, y_pred, y_proba, n_classes=n_classes
            )

            # Train accuracy (overfitting diagnostic)
            try:
                y_train_pred = model.predict(X_train)
                metrics["train_accuracy"] = float(accuracy_score(y_train, y_train_pred))
                metrics["train_test_gap"] = (
                    metrics["train_accuracy"] - metrics["accuracy"]
                )
            except Exception as e:
                metrics["train_accuracy_error"] = str(e)

            all_results[model_name] = metrics

            logger.info(f"\n{model_name} TEST RESULTS:")
            logger.info(f"  Accuracy:        {metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1:        {metrics['macro_f1']:.4f}")
            logger.info(f"  Macro Precision: {metrics['macro_precision']:.4f}")
            logger.info(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
            if "cross_entropy" in metrics and metrics["cross_entropy"] is not None:
                logger.info(f"  Cross-entropy:   {metrics['cross_entropy']:.6f}")
            if "roc_auc_ovr_macro" in metrics:
                logger.info(f"  ROC-AUC (OvR):   {metrics['roc_auc_ovr_macro']:.4f}")
            if "roc_auc_ovo_macro" in metrics:
                logger.info(f"  ROC-AUC (OvO):   {metrics['roc_auc_ovo_macro']:.4f}")
            if "train_accuracy" in metrics:
                logger.info(f"  Train accuracy:  {metrics['train_accuracy']:.4f}")
                logger.info(f"  Train-test gap:  {metrics['train_test_gap']:+.4f}")
            logger.info(f"\n  Confusion matrix:")
            for row in metrics["confusion_matrix"]:
                logger.info(f"    {row}")

            # Save per-model results
            save_json(metrics, str(output_dir / f"{model_name}_results.json"))

        except Exception as e:
            logger.error(f"  {model_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = {"error": str(e)}

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("CLASSIFICATION RESULTS SUMMARY")
    logger.info("=" * 60)

    summary_rows = []
    for model_name, metrics in all_results.items():
        if "error" in metrics:
            summary_rows.append({
                "model": model_name,
                "accuracy": "FAILED",
                "macro_f1": "FAILED",
            })
        else:
            summary_rows.append({
                "model": model_name,
                "accuracy": f"{metrics['accuracy']:.4f}",
                "macro_f1": f"{metrics['macro_f1']:.4f}",
                "macro_precision": f"{metrics['macro_precision']:.4f}",
                "macro_recall": f"{metrics['macro_recall']:.4f}",
            })

    summary_df = pd.DataFrame(summary_rows)
    logger.info("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(output_dir / "classification_summary.csv", index=False)

    # Save experiment config
    experiment_config = {
        "labeled_csv": str(args.labeled_csv),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "n_classes": n_classes,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "split_sizes": {
            "train": len(y_train),
            "val": len(y_val),
            "test": len(y_test),
        },
        "seed": seed,
        "device": device,
        "models": args.models,
    }
    save_json(experiment_config, str(output_dir / "experiment_config.json"))
    save_json(all_results, str(output_dir / "all_results.json"))

    logger.info(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
