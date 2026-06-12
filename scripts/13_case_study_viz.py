#!/usr/bin/env python3
"""
13_case_study_viz.py — Case study visualization (Dr. Cheng validation 1).

Randomly sample technologies from the TEST set, plot their actual cumulative
reuse trajectories, and annotate each with:
  - the "given label" (the unsupervised k-means cluster assignment), and
  - the "predicted label" (the trained supervised classifier's prediction).

This is the "seeing is believing" visual sanity check Dr. Cheng requested
(2026-06-02 meeting), analogous to showing correctly-classified cat/dog
examples in an image classification paper.

The script reproduces the exact 60/40 train/test split used in 05_classify.py
(same seed, same stratification) so the test set is identical, then trains the
chosen classifier and visualizes its predictions on sampled test examples.

Usage:
    python scripts/13_case_study_viz.py \
        --labeled-csv results/clustering/technologies_labeled_fae_k3_k3.csv \
        --features SIM_TECH ACCESS_SIZE SIM_ACCESS \
        --model tabm \
        --n-per-cluster 8

    # Also surface any misclassified examples (rare, given saturation)
    python scripts/13_case_study_viz.py --show-errors
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tabular_models import build_tabular_model
from src.utils.helpers import load_config, setup_logging, set_seed


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
           "#56B4E9", "#E69F00"]


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def fae_k3_features():
    p = Path("results/feature_selection/fae_k3_results.json")
    if p.exists():
        return json.load(open(p))["selection"]["selected_names"]
    return ["SIM_TECH", "ACCESS_SIZE", "SIM_ACCESS"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--labeled-csv",
                        default="results/clustering/technologies_labeled_fae_k3_k3.csv")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Features used for clustering/classification. "
                             "Default: FAE K=3 selection.")
    parser.add_argument("--model", default="tabm",
                        choices=["tabnet", "tabm", "ft_transformer",
                                 "extra_trees", "gbdt", "tabkan", "tabmixer"])
    parser.add_argument("--n-per-cluster", type=int, default=8,
                        help="Number of test examples to plot per cluster")
    parser.add_argument("--show-errors", action="store_true",
                        help="Also produce a panel of misclassified test examples")
    parser.add_argument("--test-ratio", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-dir", default="results/case_study")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)
    set_paper_style()

    features = args.features if args.features else fae_k3_features()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))
    logger.info(f"Labeled CSV: {args.labeled_csv}")
    logger.info(f"Features: {features}")
    logger.info(f"Model: {args.model}")

    # ---- Load labeled data and aligned trajectories ----
    tech_df = pd.read_csv(args.labeled_csv)
    n = len(tech_df)
    logger.info(f"Loaded {n:,} technologies")

    processed_dir = Path(cfg["data"]["processed_dir"])
    trajectories = np.load(processed_dir / "trajectories.npy")
    if len(trajectories) != n:
        logger.error(f"Trajectory count ({len(trajectories)}) != tech rows ({n}). "
                     "Cannot align; aborting.")
        sys.exit(1)

    X = np.nan_to_num(tech_df[features].values.astype(np.float32), nan=0.0)
    y = tech_df["cluster"].values.astype(np.int64)
    n_classes = int(tech_df["cluster"].nunique())

    # ---- Reproduce the EXACT split from 05_classify.py via indices ----
    idx = np.arange(n)
    idx_train_full, idx_test = train_test_split(
        idx, test_size=args.test_ratio, stratify=y, random_state=seed
    )
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=args.val_ratio,
        stratify=y[idx_train_full], random_state=seed
    )
    logger.info(f"Split sizes — train: {len(idx_train)}, "
                f"val: {len(idx_val)}, test: {len(idx_test)}")

    # ---- Scale (fit on train only) and train the classifier ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[idx_train]).astype(np.float32)
    X_val = scaler.transform(X[idx_val]).astype(np.float32)
    X_test = scaler.transform(X[idx_test]).astype(np.float32)

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    logger.info(f"Device: {device}")

    logger.info(f"Training {args.model} ...")
    model = build_tabular_model(
        model_name=args.model, n_features=len(features),
        n_classes=n_classes, device=device, seed=seed,
    )
    model.fit(X_train, y[idx_train], X_val, y[idx_val])

    y_pred_test = model.predict(X_test)
    given_test = y[idx_test]
    test_acc = float((y_pred_test == given_test).mean())
    logger.info(f"Test accuracy: {test_acc:.5f}")

    # ---- Sample examples per cluster and plot ----
    window = trajectories.shape[1]
    years = np.arange(1, window + 1)
    rng = np.random.RandomState(seed)

    clusters = sorted(np.unique(given_test))
    fig, axes = plt.subplots(
        len(clusters), args.n_per_cluster,
        figsize=(2.0 * args.n_per_cluster, 2.2 * len(clusters)),
        squeeze=False,
    )

    for row, c in enumerate(clusters):
        # test rows whose GIVEN label is c
        pool = np.where(given_test == c)[0]
        pick = rng.choice(pool, min(args.n_per_cluster, len(pool)), replace=False)
        for col in range(args.n_per_cluster):
            ax = axes[row][col]
            if col >= len(pick):
                ax.set_visible(False)
                continue
            local_i = pick[col]
            global_i = idx_test[local_i]
            traj = trajectories[global_i]
            given = int(given_test[local_i])
            pred = int(y_pred_test[local_i])
            correct = (given == pred)
            color = PALETTE[given % len(PALETTE)]
            ax.plot(years, traj, color=color, lw=1.8)
            ax.fill_between(years, 0, traj, color=color, alpha=0.12)
            title = f"given {given} / pred {pred}"
            ax.set_title(title, fontsize=8,
                         color="black" if correct else "red")
            ax.set_xticks([])
            ax.set_yticks([])
            if not correct:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(1.5)
        axes[row][0].set_ylabel(f"Cluster {c}", fontsize=11)

    fig.suptitle(
        f"Case study: test-set trajectories with given (k-means) vs "
        f"predicted ({args.model}) labels\n"
        f"test accuracy {test_acc*100:.2f}% — red title/border = misclassified",
        y=1.02, fontsize=12,
    )
    out_path = out_dir / f"case_study_{args.model}"
    fig.savefig(out_path.with_suffix(".png"))
    if not args.no_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info(f"Saved case study figure to {out_path}.png")

    # ---- Optional: panel of misclassified examples ----
    if args.show_errors:
        err_local = np.where(y_pred_test != given_test)[0]
        logger.info(f"Misclassified test examples: {len(err_local)} "
                    f"({len(err_local)/len(given_test)*100:.3f}%)")
        if len(err_local) > 0:
            k = min(len(err_local), 12)
            pick = rng.choice(err_local, k, replace=False)
            cols = min(k, 4)
            rows = (k + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols,
                                     figsize=(3.2 * cols, 2.4 * rows),
                                     squeeze=False)
            for j in range(rows * cols):
                ax = axes[j // cols][j % cols]
                if j >= k:
                    ax.set_visible(False)
                    continue
                li = pick[j]
                gi = idx_test[li]
                given = int(given_test[li])
                pred = int(y_pred_test[li])
                ax.plot(years, trajectories[gi],
                        color=PALETTE[given % len(PALETTE)], lw=1.8)
                ax.set_title(f"given {given} / pred {pred}", fontsize=9, color="red")
                ax.set_xlabel("years since emergence", fontsize=8)
            fig.suptitle("Misclassified test examples", y=1.02)
            ep = out_dir / f"case_study_errors_{args.model}"
            fig.savefig(ep.with_suffix(".png"))
            if not args.no_pdf:
                fig.savefig(ep.with_suffix(".pdf"))
            plt.close(fig)
            logger.info(f"Saved error panel to {ep}.png")
        else:
            logger.info("No misclassified test examples to plot.")

    # ---- Save a small summary ----
    summary = {
        "model": args.model,
        "features": features,
        "n_classes": n_classes,
        "test_accuracy": test_acc,
        "n_test": int(len(idx_test)),
        "n_per_cluster": args.n_per_cluster,
        "seed": seed,
    }
    with open(out_dir / f"case_study_summary_{args.model}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
