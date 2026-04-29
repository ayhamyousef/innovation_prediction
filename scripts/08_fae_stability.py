#!/usr/bin/env python3
"""
08_fae_stability.py — Test FAE selection stability under bootstrap subsampling.

This is the central test of FAE's claimed contribution per Dr. Cheng (2026-04-22):
"FAE is designed to have algorithmic stability... if one or some of examples
are added or deleted, the results would not change much."

For each (K, fraction) combination, run FAE multiple times on independently
bootstrapped subsamples of the data and record which features were selected.

Reports:
  - Per-feature selection frequency (fraction of trials picking each feature)
  - Mode subset (most common selected set)
  - Pairwise Jaccard similarity across trials (mean +/- std)
  - Subset agreement matrix (% of trials that picked the exact same set)

Usage:
    # Default: K=3,4,5; fractions 0.5, 0.75, 0.9; 5 trials each
    python scripts/08_fae_stability.py

    # Quicker test: just K=3, 3 trials per fraction
    python scripts/08_fae_stability.py --k-values 3 --n-trials 3

    # On GPU
    python scripts/08_fae_stability.py --device cuda
"""

import argparse
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fae import train_fae
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.5, 0.75, 0.9])
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Bootstrap replicates per (K, fraction)")
    parser.add_argument("--lambda1", type=float, default=2.0)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_seed = cfg["training"]["seed"]
    set_seed(base_seed)

    out_dir = Path("results/fae_stability")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))

    tech_path = Path(cfg["data"]["processed_dir"]) / "technologies.csv"
    logger.info(f"Loading {tech_path}")
    tech_df = pd.read_csv(tech_path)
    n_total = len(tech_df)
    logger.info(f"Loaded {n_total:,} rows")

    existing_cols = [c for c in FEATURE_COLS if c in tech_df.columns]
    X_full = tech_df[existing_cols].values.astype(np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0)

    # z-score on full population (fixed normalization across trials)
    mu = X_full.mean(axis=0)
    sd = X_full.std(axis=0)
    sd[sd == 0] = 1.0
    X_full_z = (X_full - mu) / sd

    rng = np.random.RandomState(base_seed)
    all_trial_records = []

    total_runs = len(args.k_values) * len(args.fractions) * args.n_trials
    run_i = 0

    for k in args.k_values:
        for frac in args.fractions:
            sample_size = int(n_total * frac)
            for trial in range(args.n_trials):
                run_i += 1
                trial_seed = base_seed + 1000 * trial + int(frac * 100) + 10 * k
                idx = rng.choice(n_total, sample_size, replace=False)
                X_sub = X_full_z[idx]

                logger.info(f"[{run_i}/{total_runs}] K={k} frac={frac} "
                            f"trial={trial+1}/{args.n_trials} "
                            f"n={sample_size:,} seed={trial_seed}")
                set_seed(trial_seed)
                _, results = train_fae(
                    X=X_sub, k=k, feature_names=existing_cols,
                    lambda1=args.lambda1, lambda2=args.lambda2,
                    lr=args.lr, max_epochs=args.max_epochs,
                    patience=args.patience, device=args.device,
                    verbose=False,
                )
                selected = sorted(results["selection"]["selected_names"])
                logger.info(f"  selected: {selected}  "
                            f"recon_global={results['final_recon_global']:.2f}")
                all_trial_records.append({
                    "K": k,
                    "fraction": frac,
                    "trial": trial,
                    "n_samples": sample_size,
                    "seed": trial_seed,
                    "selected": "|".join(selected),
                    "recon_global": results["final_recon_global"],
                    "recon_sub": results["final_recon_sub"],
                    "epochs_trained": results["epochs_trained"],
                })

    trials_df = pd.DataFrame(all_trial_records)
    trials_df.to_csv(out_dir / "trials.csv", index=False)
    logger.info(f"Saved trial records to {out_dir / 'trials.csv'}")

    # ============================================================
    # Stability analysis
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("STABILITY ANALYSIS")
    logger.info("=" * 60)

    summary_rows = []

    for k in args.k_values:
        for frac in args.fractions:
            group = trials_df[(trials_df["K"] == k) &
                              (trials_df["fraction"] == frac)]
            sel_strs = group["selected"].tolist()
            sets = [set(s.split("|")) for s in sel_strs]

            # Mode subset and its rate
            counter = Counter(sel_strs)
            mode_subset, mode_count = counter.most_common(1)[0]
            mode_rate = mode_count / len(sets)

            # Per-feature selection frequency
            feat_freq = {feat: 0 for feat in existing_cols}
            for s in sets:
                for f in s:
                    feat_freq[f] += 1
            feat_freq_pct = {f: feat_freq[f] / len(sets)
                             for f in existing_cols}

            # Pairwise Jaccard
            jaccs = [jaccard(a, b) for a, b in combinations(sets, 2)]
            jacc_mean = float(np.mean(jaccs)) if jaccs else 1.0
            jacc_std = float(np.std(jaccs)) if jaccs else 0.0

            logger.info(f"\nK={k}, fraction={frac}, n_trials={len(sets)}")
            logger.info(f"  Mode subset: {mode_subset}")
            logger.info(f"  Mode rate (exact agreement): "
                        f"{mode_rate*100:.1f}% "
                        f"({mode_count}/{len(sets)} trials)")
            logger.info(f"  Pairwise Jaccard: "
                        f"{jacc_mean:.3f} +/- {jacc_std:.3f}")
            logger.info(f"  Per-feature selection frequency:")
            for f, p in sorted(feat_freq_pct.items(),
                               key=lambda kv: -kv[1]):
                logger.info(f"    {f:14s} {p*100:5.1f}%")

            summary_rows.append({
                "K": k,
                "fraction": frac,
                "n_trials": len(sets),
                "mode_subset": mode_subset,
                "mode_rate": mode_rate,
                "jaccard_mean": jacc_mean,
                "jaccard_std": jacc_std,
                **{f"freq_{f}": feat_freq_pct[f] for f in existing_cols},
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "stability_summary.csv", index=False)
    logger.info(f"\nSaved stability summary to {out_dir / 'stability_summary.csv'}")

    # JSON snapshot
    save_json({
        "k_values": args.k_values,
        "fractions": args.fractions,
        "n_trials": args.n_trials,
        "summary": summary_rows,
    }, str(out_dir / "stability_summary.json"))

    logger.info("\nInterpretation guide:")
    logger.info("  Jaccard near 1.0  -> identical subsets across trials (stable)")
    logger.info("  Jaccard 0.5-0.9   -> partial agreement (most features stable)")
    logger.info("  Jaccard < 0.5     -> unstable selection")
    logger.info("  Mode rate near 1.0 -> exact same subset chosen most trials")


if __name__ == "__main__":
    main()
