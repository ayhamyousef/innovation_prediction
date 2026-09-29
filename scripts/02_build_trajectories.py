#!/usr/bin/env python3
"""
02_build_trajectories.py: Extract technologies, compute features, and
build reuse trajectories from patent data.

Memory-optimized: streams JSONL in two passes to avoid loading all 4.9M
full patent records into memory at once.

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

    patents_path = Path(data_cfg["raw_dir"]) / "patents_all.jsonl"

    # ----------------------------------------------------------------
    # Pass 1: Stream patents, keep only fields needed for technology
    # extraction (patent_number, patent_year, ipc_codes_6digit).
    # This uses ~1GB instead of ~4-5GB for full records.
    # ----------------------------------------------------------------
    logger.info(f"Pass 1: Loading minimal patent data from {patents_path}")
    patents_minimal = []
    count = 0
    with open(patents_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            pat = json.loads(line)
            patents_minimal.append({
                "patent_number": pat.get("patent_number", ""),
                "patent_year": pat.get("patent_year"),
                "ipc_codes_6digit": pat.get("ipc_codes_6digit", []),
            })
            count += 1
            if count % 1_000_000 == 0:
                logger.info(f"  Read {count:,} patents...")

    logger.info(f"Loaded {count:,} patents (minimal fields only)")

    # Extract technologies & compute features
    extractor = TechnologyExtractor(
        ipc_digits=data_cfg.get("classification_digits", data_cfg.get("ipc_code_digits", 6)),
        access_window=feat_cfg["access_window_years"],
        sim_tech_weights=feat_cfg["sim_tech_weights"],
    )

    tech_df, component_counts = extractor.extract_technologies(patents_minimal)
    tech_df = extractor.compute_features(tech_df, component_counts, patents_minimal)

    # Free the minimal patent list; it is no longer needed
    del patents_minimal

    # Build reuse trajectories & filter
    filtered_df, trajectories = build_reuse_trajectories(
        tech_df,
        observation_window=data_cfg["observation_window_years"],
        min_reuse=data_cfg["min_reuse_frequency"],
    )

    # Free unfiltered tech_df
    del tech_df

    # Save outputs
    processed_dir = Path(data_cfg["processed_dir"])
    ensure_dir(str(processed_dir))

    # Save tech DataFrame
    save_df = filtered_df.copy()
    for col in save_df.columns:
        if save_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            save_df[col] = save_df[col].apply(json.dumps)
    save_df.to_csv(processed_dir / "technologies.csv", index=False)
    del save_df

    # Save trajectories
    np.save(processed_dir / "trajectories.npy", trajectories)

    # ----------------------------------------------------------------
    # Pass 2: Build patent_lookup with ONLY patents referenced by
    # technologies (early_patent_numbers). Write as streaming JSONL.
    # ----------------------------------------------------------------
    needed_patents = set()
    for _, row in filtered_df.iterrows():
        pnums = row.get("early_patent_numbers", [])
        if isinstance(pnums, str):
            pnums = json.loads(pnums)
        if isinstance(pnums, list):
            needed_patents.update(pnums)

    logger.info(f"Pass 2: Extracting {len(needed_patents):,} patents needed "
                f"for lookup (out of {count:,} total)")

    patent_lookup = {}
    with open(patents_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            pat = json.loads(line)
            pnum = pat.get("patent_number", "")
            if pnum in needed_patents:
                patent_lookup[pnum] = pat
                if len(patent_lookup) >= len(needed_patents):
                    break  # found all we need

    logger.info(f"Built patent lookup with {len(patent_lookup):,} entries")

    with open(processed_dir / "patent_lookup.json", "w") as f:
        json.dump(patent_lookup, f, default=str)

    # Summary statistics
    logger.info(f"Technologies with sufficient reuse: {len(filtered_df)}")
    logger.info(f"Trajectory shape: {trajectories.shape}")
    logger.info(f"Emergence year range: "
                f"{filtered_df['emergence_year'].min()} - "
                f"{filtered_df['emergence_year'].max()}")

    feature_cols = ["ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS",
                    "SIM_TECH", "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE"]
    for col in feature_cols:
        if col in filtered_df.columns:
            logger.info(f"  {col}: mean={filtered_df[col].mean():.2f}, "
                        f"std={filtered_df[col].std():.2f}")


if __name__ == "__main__":
    main()
