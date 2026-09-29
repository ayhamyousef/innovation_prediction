#!/usr/bin/env python3
"""
01_fetch_data.py: Fetch patent data from USPTO Open Data Portal API.

Downloads granted utility patent metadata (title, CPC codes, inventors)
via the ODP API, month by month, with per-month caching.

Usage:
    python scripts/01_fetch_data.py [--config config/default.yaml]
                                     [--start-year 2002]
                                     [--end-year 2022]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.data.odp_client import ODPClient
from src.utils.helpers import load_config, setup_logging, ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Fetch USPTO patent data")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg["training"]["output_dir"])

    data_cfg = cfg["data"]
    start_year = args.start_year or data_cfg["start_year"]
    end_year = args.end_year or data_cfg["end_year"]

    ensure_dir(data_cfg["raw_dir"])
    ensure_dir(data_cfg["cache_dir"])

    logger.info(f"Fetching patents from {start_year} to {end_year}")

    client = ODPClient(
        cache_dir=data_cfg["cache_dir"],
        delay_sec=data_cfg.get("request_delay_sec", 1.0),
    )

    all_patents = client.fetch_patents_range(
        start_year=start_year,
        end_year=end_year,
    )

    # Save consolidated output
    output_path = Path(data_cfg["raw_dir"]) / "patents_all.jsonl"
    with open(output_path, "w") as f:
        for pat in all_patents:
            f.write(json.dumps(pat, default=str) + "\n")

    logger.info(f"Saved {len(all_patents)} patents to {output_path}")

    # Print summary
    years = [p.get("patent_year") for p in all_patents if p.get("patent_year")]
    if years:
        from collections import Counter
        year_dist = Counter(years)
        logger.info("Year distribution:")
        for y in sorted(year_dist.keys()):
            logger.info(f"  {y}: {year_dist[y]} patents")


if __name__ == "__main__":
    main()
