#!/usr/bin/env python3
"""
02_build_trajectories.py — Extract technologies, compute features, and
build reuse trajectories from patent data.

Usage:
    python scripts/02_build_trajectories.py [--config config/default.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_extraction import TechnologyExtractor
from src.data.clustering import build_reuse_trajectories
from src.utils.helpers import load_config, setup_logging, ensure_dir, save_json
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg["training"]["output_dir"])
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    # Load patents
    patents_path = Path(data_cfg["raw_dir"]) / "patents_all.jsonl"
    logger.info(f"Loading patents from {patents_path}")
    patents = []
    with open(patents_path, "r") as f:
        for line in f:
            if line.strip():
                patents.append(json.loads(line))
    logger.info(f"Loaded {len(patents)} patents")

    # Extract technologies
    extractor = TechnologyExtractor(
        ipc_digits=data_cfg["ipc_code_digits"],
        access_window=feat_cfg["access_window_years"],
        sim_tech_weights=feat_cfg["sim_tech_weights"],
    )

    tech_df, component_counts = extractor.extract_technologies(patents)

    # Compute features
    tech_df = extractor.compute_features(tech_df, component_counts, patents)

    # Build reuse trajectories
    filtered_df, trajectories = build_reuse_trajectories(
        tech_df,
        observation_window=data_cfg["observation_window_years"],
        min_reuse=data_cfg["min_reuse_frequency"],
    )

    # Save outputs
    processed_dir = Path(data_cfg["processed_dir"])
    ensure_dir(str(processed_dir))

    # Save tech DataFrame (convert lists to JSON strings for CSV compatibility)
    save_df = filtered_df.copy()
    for col in save_df.columns:
        if save_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            save_df[col] = save_df[col].apply(json.dumps)
    save_df.to_csv(processed_dir / "technologies.csv", index=False)

    # Save trajectories
    np.save(processed_dir / "trajectories.npy", trajectories)

    # Save patent lookup for dataset
    patent_lookup = {}
    for p in patents:
        pnum = p.get("patent_number", "")
        if pnum:
            patent_lookup[pnum] = p
    with open(processed_dir / "patent_lookup.json", "w") as f:
        json.dump(patent_lookup, f, default=str)

    # Summary statistics
    logger.info(f"Technologies with sufficient reuse: {len(filtered_df)}")
    logger.info(f"Trajectory shape: {trajectories.shape}")
    logger.info(f"Emergence year range: "
                f"{filtered_df['emergence_year'].min()} - "
                f"{filtered_df['emergence_year'].max()}")

    # Feature statistics
    feature_cols = ["ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS",
                    "SIM_TECH", "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE"]
    for col in feature_cols:
        if col in filtered_df.columns:
            logger.info(f"  {col}: mean={filtered_df[col].mean():.2f}, "
                        f"std={filtered_df[col].std():.2f}")


if __name__ == "__main__":
    main()
