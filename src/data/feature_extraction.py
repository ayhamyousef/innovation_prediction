"""
Feature extraction for technology components and early inventions.

Implements the 7 features from Chen et al. (2025):
  Technology components:  ACCESS_SIZE, ACCESS_TREND, SIM_ACCESS, SIM_TECH
  Early inventions:       INVENT_DIVER, INVENT_APPL, ATTENT_SIZE
"""

import logging
import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("innovation_prediction.features")


class TechnologyExtractor:
    """
    Extracts 'technologies' from patent data.

    A technology = pairwise combination of 6-digit CPC/IPC codes.
    A novel technology = first occurrence of that pair.
    Early inventions = patents applying that technology in year of emergence.

    Note: ODP API provides CPC codes (successor to IPC, same hierarchy).
    These are mapped to ipc_codes_6digit for downstream compatibility.
    """

    def __init__(self, ipc_digits: int = 6, access_window: int = 5,
                 sim_tech_weights: List[float] = None):
        self.ipc_digits = ipc_digits
        self.access_window = access_window
        self.sim_tech_weights = sim_tech_weights or [0.5, 0.3, 0.2]

    def extract_technologies(self, patents: List[Dict]) -> pd.DataFrame:
        """
        From raw patents, extract all pairwise IPC combinations as technologies.

        Returns DataFrame with columns:
          tech_id, ipc1, ipc2, emergence_year, patent_numbers (list),
          num_early_inventions
        """
        logger.info("Extracting technologies from patents...")

        # Build mapping: year -> list of (patent_number, ipc_codes_6digit)
        year_patents = defaultdict(list)
        for pat in patents:
            year = pat.get("patent_year")
            ipcs = pat.get("ipc_codes_6digit", [])
            pnum = pat.get("patent_number", "")
            if year and ipcs and pnum:
                year_patents[year].append((pnum, ipcs))

        # Track first appearance of each IPC pair
        tech_first_seen = {}       # (ipc1, ipc2) -> year
        tech_patents = defaultdict(list)  # (ipc1, ipc2) -> [(year, patent_number)]
        # Per-component patent counts by year
        component_year_count = defaultdict(lambda: defaultdict(int))

        for year in sorted(year_patents.keys()):
            for pnum, ipcs in year_patents[year]:
                # Count each component
                for ipc in ipcs:
                    component_year_count[ipc][year] += 1

                # Generate all pairwise combinations
                unique_ipcs = sorted(set(ipcs))
                for ipc_a, ipc_b in combinations(unique_ipcs, 2):
                    tech_key = (ipc_a, ipc_b)
                    tech_patents[tech_key].append((year, pnum))
                    if tech_key not in tech_first_seen:
                        tech_first_seen[tech_key] = year

        logger.info(f"Found {len(tech_first_seen)} unique technologies")

        # Build technology records
        records = []
        for (ipc1, ipc2), emergence_year in tech_first_seen.items():
            # Early inventions = patents in emergence year
            early_patents = [
                pnum for (y, pnum) in tech_patents[(ipc1, ipc2)]
                if y == emergence_year
            ]
            # All patent occurrences by year
            year_counts = Counter(
                y for (y, _) in tech_patents[(ipc1, ipc2)]
            )

            records.append({
                "tech_id": f"{ipc1}_{ipc2}",
                "ipc1": ipc1,
                "ipc2": ipc2,
                "emergence_year": emergence_year,
                "early_patent_numbers": early_patents,
                "num_early_inventions": len(early_patents),
                "year_counts": dict(year_counts),
                "total_reuse_count": sum(year_counts.values()),
            })

        df = pd.DataFrame(records)
        logger.info(f"Technologies DataFrame shape: {df.shape}")
        return df, component_year_count

    def compute_features(self, tech_df: pd.DataFrame,
                         component_year_count: Dict,
                         patents: List[Dict]) -> pd.DataFrame:
        """
        Compute the 7 features from the paper for each technology.

        Args:
            tech_df: DataFrame from extract_technologies
            component_year_count: {ipc_code: {year: count}}
            patents: original patent list (for early invention features)
        Returns:
            tech_df with added feature columns
        """
        logger.info("Computing technology component & early invention features...")

        # Build patent lookup for early invention features
        patent_lookup = {}
        for pat in patents:
            pnum = pat.get("patent_number", "")
            if pnum:
                patent_lookup[pnum] = pat

        features = []
        for _, row in tech_df.iterrows():
            ipc1 = row["ipc1"]
            ipc2 = row["ipc2"]
            year = row["emergence_year"]
            early_pats = row["early_patent_numbers"]

            feat = self._compute_component_features(
                ipc1, ipc2, year, component_year_count
            )
            feat.update(self._compute_early_invention_features(
                early_pats, patent_lookup
            ))
            features.append(feat)

        feat_df = pd.DataFrame(features)
        tech_df = pd.concat([tech_df.reset_index(drop=True),
                             feat_df.reset_index(drop=True)], axis=1)

        logger.info("Feature computation complete.")
        return tech_df

    def _compute_component_features(self, ipc1: str, ipc2: str, year: int,
                                     comp_counts: Dict) -> Dict[str, float]:
        """Compute ACCESS_SIZE, ACCESS_TREND, SIM_ACCESS, SIM_TECH."""

        # --- ACCESS_SIZE: sum of patent counts for both components
        #     in the 5 years before emergence (Eq. 3) ---
        access_size = 0.0
        for y in range(year - self.access_window, year):
            access_size += comp_counts.get(ipc1, {}).get(y, 0)
            access_size += comp_counts.get(ipc2, {}).get(y, 0)

        # --- ACCESS_TREND: ratio of last year to first year minus 1 (Eq. 4) ---
        tc_last = (comp_counts.get(ipc1, {}).get(year - 1, 0) +
                   comp_counts.get(ipc2, {}).get(year - 1, 0))
        tc_first = (comp_counts.get(ipc1, {}).get(year - self.access_window, 0) +
                    comp_counts.get(ipc2, {}).get(year - self.access_window, 0))
        if tc_first > 0:
            access_trend = (tc_last / tc_first) - 1.0
        else:
            access_trend = 0.0

        # --- SIM_ACCESS: absolute difference in cumulative counts (Eq. 5) ---
        sum_tc1 = sum(comp_counts.get(ipc1, {}).get(y, 0)
                      for y in range(year - self.access_window, year))
        sum_tc2 = sum(comp_counts.get(ipc2, {}).get(y, 0)
                      for y in range(year - self.access_window, year))
        sim_access = abs(sum_tc1 - sum_tc2)

        # --- SIM_TECH: weighted IPC distance (Eq. 6) ---
        w1, w2, w3 = self.sim_tech_weights
        ipc1_dist = 1.0 if ipc1[0] != ipc2[0] else 0.0        # section (1st digit)
        ipc3_dist = 1.0 if ipc1[:3] != ipc2[:3] else 0.0       # class (3 digits)
        ipc4_dist = 1.0 if ipc1[:4] != ipc2[:4] else 0.0       # subclass (4 digits)
        sim_tech = ipc1_dist * w1 + ipc3_dist * w2 + ipc4_dist * w3

        return {
            "ACCESS_SIZE": access_size,
            "ACCESS_TREND": access_trend,
            "SIM_ACCESS": sim_access,
            "SIM_TECH": sim_tech,
        }

    def _compute_early_invention_features(self, early_patent_nums: List[str],
                                           patent_lookup: Dict) -> Dict[str, float]:
        """Compute INVENT_DIVER, INVENT_APPL, ATTENT_SIZE (Eqs. 7-9)."""

        num_pat = len(early_patent_nums)

        # --- ATTENT_SIZE = number of early inventions (Eq. 9) ---
        attent_size = float(num_pat)

        if num_pat == 0:
            return {
                "INVENT_DIVER": 0.0,
                "INVENT_APPL": 0.0,
                "ATTENT_SIZE": 0.0,
            }

        # Collect IPC section distribution and IPC code counts
        section_counts = Counter()
        total_ipc_count = 0

        for pnum in early_patent_nums:
            pat = patent_lookup.get(pnum, {})
            ipcs = pat.get("ipc_codes_6digit", [])
            total_ipc_count += len(ipcs)
            for ipc in ipcs:
                if ipc:
                    section_counts[ipc[0]] += 1  # first char = section

        # --- INVENT_DIVER: entropy of IPC sections (Eq. 7) ---
        invent_diver = 0.0
        total_section_sum = sum(section_counts.values())
        if total_section_sum > 0:
            for section, cnt in section_counts.items():
                p = cnt / total_section_sum
                if p > 0:
                    invent_diver += p * math.log(1.0 / p)

        # --- INVENT_APPL: avg number of IPC codes per patent (Eq. 8) ---
        invent_appl = total_ipc_count / num_pat if num_pat > 0 else 0.0

        return {
            "INVENT_DIVER": invent_diver,
            "INVENT_APPL": invent_appl,
            "ATTENT_SIZE": attent_size,
        }
