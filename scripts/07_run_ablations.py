#!/usr/bin/env python3
"""
07_run_ablations.py — Run ablation studies comparing:
  - Text-only vs metadata+text
  - Different context lengths (128, 256, 512)
  - Different model sizes (tiny, small, base)

Usage:
    python scripts/07_run_ablations.py [--config config/default.yaml]
                                        [--variants text_only,metadata_text]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import PatentDataset, prepare_splits, create_dataloaders
from src.models.transformer import PatentTransformerClassifier, build_model
from src.training.trainer import Trainer
from src.training.evaluate import format_results_table, ablation_summary
from src.utils.helpers import (
    load_config, setup_logging, set_seed, get_device, save_json, count_parameters
)


def run_variant(variant_name: str, variant_cfg: dict, base_cfg: dict,
                tech_df: pd.DataFrame, patent_lookup: dict,
                device, output_dir: Path, logger) -> dict:
    """Train and evaluate a single ablation variant."""
    logger.info(f"\n{'='*50}")
    logger.info(f"ABLATION: {variant_name}")
    logger.info(f"{'='*50}")

    # Override base config with variant settings
    cfg = json.loads(json.dumps(base_cfg))  # deep copy
    cfg["model"]["use_metadata"] = variant_cfg.get("use_metadata",
                                                     cfg["model"]["use_metadata"])
    cfg["model"]["backbone"] = variant_cfg.get("backbone",
                                                 cfg["model"]["backbone"])
    cfg["tokenizer"]["max_length"] = variant_cfg.get("max_length",
                                                       cfg["tokenizer"]["max_length"])
    cfg["tokenizer"]["name"] = variant_cfg.get("backbone",
                                                 cfg["tokenizer"]["name"])

    variant_dir = output_dir / variant_name
    cfg["training"]["output_dir"] = str(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["training"]["seed"])

    # Prepare data
    train_ds, val_ds, test_ds = prepare_splits(tech_df, patent_lookup, cfg)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=cfg["training"]["batch_size"],
        use_weighted_sampler=cfg["training"]["use_class_weights"],
    )

    # Update metadata count
    if cfg["model"]["use_metadata"]:
        cfg["model"]["metadata_features_count"] = len(train_ds.metadata_cols)
    else:
        cfg["model"]["metadata_features_count"] = 0

    # Build and train
    model = build_model(cfg)
    n_params = count_parameters(model)
    logger.info(f"  Params: {n_params:,}")
    logger.info(f"  Backbone: {cfg['model']['backbone']}")
    logger.info(f"  Max length: {cfg['tokenizer']['max_length']}")
    logger.info(f"  Use metadata: {cfg['model']['use_metadata']}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        class_weights=train_ds.get_class_weights(),
    )

    results = trainer.train()
    test_metrics = results.get("test_metrics", {})
    test_metrics["n_params"] = n_params
    test_metrics["variant"] = variant_name

    save_json(test_metrics, str(variant_dir / "test_results.json"))
    save_json(cfg, str(variant_dir / "config_used.json"))

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--variants", default=None,
                        help="Comma-separated list of variant names to run")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["training"]["output_dir"]) / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(output_dir))
    device = get_device()

    logger.info("=" * 60)
    logger.info("Innovation Prediction — Ablation Studies")
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

    # Determine which variants to run
    all_variants = {v["name"]: v for v in cfg["ablations"]["variants"]}
    if args.variants:
        selected = args.variants.split(",")
        variants = {k: all_variants[k] for k in selected if k in all_variants}
    else:
        variants = all_variants

    logger.info(f"Running {len(variants)} ablation variants: "
                f"{list(variants.keys())}")

    # Run ablations
    all_results = {}
    for name, variant_cfg in variants.items():
        try:
            metrics = run_variant(
                name, variant_cfg, cfg, tech_df, patent_lookup,
                device, output_dir, logger
            )
            all_results[name] = metrics
        except Exception as e:
            logger.error(f"Variant {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_results:
        summary = ablation_summary(all_results)
        logger.info("\n" + "=" * 60)
        logger.info("ABLATION RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info("\n" + summary.to_string(index=False))

        summary.to_csv(output_dir / "ablation_summary.csv", index=False)
        save_json(
            {k: {kk: vv for kk, vv in v.items()
                  if not isinstance(vv, (np.ndarray, list))}
             for k, v in all_results.items()},
            str(output_dir / "ablation_results.json")
        )

    logger.info(f"\nAll ablation results saved to {output_dir}")


if __name__ == "__main__":
    main()
