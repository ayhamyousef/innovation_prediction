#!/usr/bin/env python3
"""
05_train_transformer.py — Train the Transformer patent classifier.

Usage:
    python scripts/05_train_transformer.py [--config config/default.yaml]
                                            [--backbone bert-base-uncased]
                                            [--epochs 20]
                                            [--batch-size 32]
                                            [--lr 2e-5]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import PatentDataset, prepare_splits, create_dataloaders
from src.models.transformer import build_model, PatentTransformerClassifier
from src.training.trainer import Trainer
from src.training.evaluate import (
    per_class_analysis, plot_confusion_matrix, plot_training_curves
)
from src.utils.helpers import (
    load_config, setup_logging, set_seed, get_device, save_json, count_parameters
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output-tag", default="transformer",
                        help="Tag for output directory")
    args = parser.parse_args()

    # Load and override config
    cfg = load_config(args.config)
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.no_metadata:
        cfg["model"]["use_metadata"] = False
    if args.max_length:
        cfg["tokenizer"]["max_length"] = args.max_length

    output_dir = Path(cfg["training"]["output_dir"]) / args.output_tag
    cfg["training"]["output_dir"] = str(output_dir)

    logger = setup_logging(str(output_dir))
    set_seed(cfg["training"]["seed"])
    device = get_device()

    logger.info("=" * 60)
    logger.info("Innovation Prediction — Transformer Training")
    logger.info("=" * 60)
    logger.info(f"Config: backbone={cfg['model']['backbone']}, "
                f"use_metadata={cfg['model']['use_metadata']}, "
                f"fusion={cfg['model']['fusion_strategy']}")

    # Load labeled data
    processed_dir = Path(cfg["data"]["processed_dir"])
    tech_df = pd.read_csv(processed_dir / "technologies_labeled.csv")

    # Load patent lookup
    with open(processed_dir / "patent_lookup.json", "r") as f:
        patent_lookup = json.load(f)

    logger.info(f"Loaded {len(tech_df)} labeled technologies")

    # Parse JSON columns
    for col in ["early_patent_numbers", "year_counts"]:
        if col in tech_df.columns:
            tech_df[col] = tech_df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

    # Create splits
    train_ds, val_ds, test_ds = prepare_splits(tech_df, patent_lookup, cfg)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=cfg["training"]["batch_size"],
        use_weighted_sampler=cfg["training"]["use_class_weights"],
    )

    # Build model
    # Adjust metadata_features_count based on actual dataset
    if cfg["model"]["use_metadata"]:
        cfg["model"]["metadata_features_count"] = len(train_ds.metadata_cols)

    model = build_model(cfg)
    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params:,} trainable")

    # Get class weights
    class_weights = train_ds.get_class_weights()
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        class_weights=class_weights,
    )

    results = trainer.train()

    # Per-class analysis
    if test_loader is not None:
        import numpy as np
        import torch

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    metadata=batch.get("metadata"),
                )
                logits = outputs["logits"]
                all_preds.append(logits.argmax(-1).cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
                all_probs.append(torch.softmax(logits, -1).cpu().numpy())

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)

        # Per-class analysis
        pca = per_class_analysis(labels, preds, probs)
        logger.info("\nPer-class analysis:")
        logger.info(pca.to_string(index=False))
        pca.to_csv(output_dir / "per_class_analysis.csv", index=False)

        # Confusion matrix
        plot_confusion_matrix(labels, preds,
                               save_path=str(output_dir / "confusion_matrix.png"))

    # Training curves
    if "train_losses" in results:
        plot_training_curves(
            results["train_losses"],
            results.get("val_metrics", []),
            save_path=str(output_dir / "training_curves.png")
        )

    # Save all results
    save_json(results.get("test_metrics", {}),
              str(output_dir / "test_results.json"))
    save_json(cfg, str(output_dir / "config_used.json"))

    logger.info(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
