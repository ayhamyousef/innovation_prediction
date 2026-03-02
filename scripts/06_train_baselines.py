#!/usr/bin/env python3
"""
06_train_baselines.py — Train TF-IDF + LR and metadata-only baselines.

Replicates the 10 ML algorithms from the Chen et al. paper as baselines,
plus a TF-IDF + LR text baseline and a combined TF-IDF + metadata baseline.

Usage:
    python scripts/06_train_baselines.py [--config config/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_extraction import PatentTextExtractor
from src.models.baselines import (
    TFIDFBaseline, MetadataBaseline, CombinedBaseline, compute_metrics
)
from src.training.evaluate import format_results_table, per_class_analysis
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["training"]["output_dir"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir))
    set_seed(cfg["training"]["seed"])

    logger.info("=" * 60)
    logger.info("Innovation Prediction — Baseline Training")
    logger.info("=" * 60)

    # Load data
    processed_dir = Path(cfg["data"]["processed_dir"])
    tech_df = pd.read_csv(processed_dir / "technologies_labeled.csv")

    with open(processed_dir / "patent_lookup.json", "r") as f:
        patent_lookup = json.load(f)

    # Parse JSON columns
    for col in ["early_patent_numbers", "year_counts"]:
        if col in tech_df.columns:
            tech_df[col] = tech_df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

    logger.info(f"Loaded {len(tech_df)} labeled technologies")

    # Split data
    split_cfg = cfg["splits"]
    if split_cfg["method"] == "time_based":
        train_df = tech_df[tech_df["emergence_year"] <= split_cfg["train_end_year"]]
        val_df = tech_df[(tech_df["emergence_year"] > split_cfg["train_end_year"]) &
                         (tech_df["emergence_year"] <= split_cfg["val_end_year"])]
        test_df = tech_df[tech_df["emergence_year"] > split_cfg["val_end_year"]]
    else:
        from sklearn.model_selection import train_test_split
        train_val, test_df = train_test_split(
            tech_df, test_size=0.15, stratify=tech_df["cluster"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_val, test_size=0.176, stratify=train_val["cluster"], random_state=42
        )

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Build text representations
    text_ext = PatentTextExtractor(use_title=True, use_abstract=False)

    def get_texts(df):
        texts = []
        for _, row in df.iterrows():
            early_pats = row.get("early_patent_numbers", [])
            if isinstance(early_pats, str):
                early_pats = json.loads(early_pats)
            parts = [f"CPC: {row.get('ipc1', '')} + {row.get('ipc2', '')}"]
            for pnum in early_pats[:3]:
                pat = patent_lookup.get(pnum, {})
                t = text_ext.get_text(pat)
                if t:
                    parts.append(t)
            texts.append(" ".join(parts))
        return texts

    train_texts = get_texts(train_df)
    test_texts = get_texts(test_df)

    all_results = {}

    # ---- 1. TF-IDF + Logistic Regression ----
    logger.info("\n--- TF-IDF + Logistic Regression ---")
    tfidf_lr = TFIDFBaseline(cfg["baseline"]["tfidf"],
                              cfg["baseline"]["logistic_regression"])
    tfidf_lr.fit(train_texts, train_df["cluster"].values)
    tfidf_lr_results = tfidf_lr.evaluate(test_texts, test_df["cluster"].values)
    all_results["TF-IDF + LR"] = tfidf_lr_results
    logger.info(f"  Accuracy: {tfidf_lr_results['accuracy']:.4f}")
    logger.info(f"  Macro F1: {tfidf_lr_results['macro_f1']:.4f}")

    # ---- 2. Metadata-only classifiers (paper's 10 models) ----
    metadata_algorithms = ["lr", "rf", "gbdt", "knn", "svm", "mlp", "gnb"]

    for algo in metadata_algorithms:
        logger.info(f"\n--- Metadata-only: {algo.upper()} ---")
        try:
            baseline = MetadataBaseline(algorithm=algo)
            baseline.fit(train_df)
            results = baseline.evaluate(test_df)
            all_results[f"Metadata-{algo.upper()}"] = results
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  Macro F1: {results['macro_f1']:.4f}")
            logger.info(f"  ROC-AUC:  {results.get('roc_auc', 0):.4f}")
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    # ---- 3. Combined TF-IDF + Metadata ----
    logger.info("\n--- Combined TF-IDF + Metadata ---")
    combined = CombinedBaseline(cfg["baseline"]["tfidf"],
                                 cfg["baseline"]["logistic_regression"])
    combined.fit(train_texts, train_df)
    combined_results = combined.evaluate(test_texts, test_df)
    all_results["TF-IDF + Metadata + LR"] = combined_results
    logger.info(f"  Accuracy: {combined_results['accuracy']:.4f}")
    logger.info(f"  Macro F1: {combined_results['macro_f1']:.4f}")

    # ---- Summary ----
    summary = format_results_table(all_results)
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("\n" + summary.to_string(index=False))

    summary.to_csv(output_dir / "baseline_results.csv", index=False)

    # Save detailed results
    serializable_results = {}
    for name, res in all_results.items():
        serializable_results[name] = {
            k: v for k, v in res.items()
            if k != "classification_report"
        }
    save_json(serializable_results, str(output_dir / "baseline_results.json"))

    logger.info(f"\nAll baseline results saved to {output_dir}")


if __name__ == "__main__":
    main()
