#!/usr/bin/env python3
"""
13_case_study_viz.py — Case study visualization (Dr. Cheng validation 1).

Randomly sample technologies from the TEST set, plot their actual cumulative
reuse trajectories, and annotate each with the "given label" (unsupervised
k-means cluster) and the "predicted label" (trained classifier). Each panel
overlays the cluster mean trajectory as a reference. A second figure shows all
misclassified test examples.

Reproduces the exact 60/40 split of 05_classify.py (same seed/stratification)
so the test set is identical, then trains the chosen classifier.

Usage:
    python scripts/13_case_study_viz.py --model tabmixer --n-per-cluster 5 --show-errors
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
from src.utils.plotting import (
    set_paper_style, save_fig, CLUSTER_COLORS, REF_COLOR, REF_DARK,
    ERROR_COLOR, ARROW,
)


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def fae_k3_features():
    p = Path("results/feature_selection/fae_k3_results.json")
    if p.exists():
        return json.load(open(p))["selection"]["selected_names"]
    return ["SIM_TECH", "ACCESS_SIZE", "SIM_ACCESS"]


def _supxlabel(fig, text, **kw):
    """Figure-level x label, with fallback for matplotlib < 3.4."""
    fn = getattr(fig, "supxlabel", None)
    if fn is not None:
        fn(text, **kw)
    else:
        fig.text(0.5, 0.005, text, ha="center", **kw)


def _supylabel(fig, text, **kw):
    fn = getattr(fig, "supylabel", None)
    if fn is not None:
        fn(text, **kw)
    else:
        fig.text(0.005, 0.5, text, va="center", rotation="vertical", **kw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--labeled-csv",
                        default="results/clustering/technologies_labeled_fae_k3_k3.csv")
    parser.add_argument("--features", nargs="+", default=None)
    parser.add_argument("--model", default="tabmixer",
                        choices=["tabnet", "tabm", "ft_transformer",
                                 "extra_trees", "gbdt", "tabkan", "tabmixer"])
    parser.add_argument("--n-per-cluster", type=int, default=5,
                        help="Test examples to plot per cluster")
    parser.add_argument("--show-errors", action="store_true")
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
    logger.info(f"Features: {features} | model: {args.model}")

    # ---- Load labeled data + aligned trajectories ----
    tech_df = pd.read_csv(args.labeled_csv)
    n = len(tech_df)
    trajectories = np.load(Path(cfg["data"]["processed_dir"]) / "trajectories.npy")
    if len(trajectories) != n:
        logger.error("Trajectory/row count mismatch; aborting.")
        sys.exit(1)

    X = np.nan_to_num(tech_df[features].values.astype(np.float32), nan=0.0)
    y = tech_df["cluster"].values.astype(np.int64)
    n_classes = int(tech_df["cluster"].nunique())
    window = trajectories.shape[1]
    years = np.arange(1, window + 1)

    # Cluster MEDIAN trajectory (typical member; robust to the right-skew that
    # makes the mean unrepresentative) + a robust per-cluster y-limit so a few
    # extreme outliers do not blow up the shared scale.
    cluster_median = {c: np.median(trajectories[y == c], axis=0)
                      for c in range(n_classes)}
    cluster_ylim = {c: float(np.percentile(trajectories[y == c][:, -1], 95)) * 1.1
                    for c in range(n_classes)}

    # ---- Reproduce 05_classify split via indices ----
    idx = np.arange(n)
    idx_train_full, idx_test = train_test_split(
        idx, test_size=args.test_ratio, stratify=y, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=args.val_ratio,
        stratify=y[idx_train_full], random_state=seed)

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

    logger.info(f"Training {args.model} (device={device}) ...")
    model = build_tabular_model(model_name=args.model, n_features=len(features),
                                n_classes=n_classes, device=device, seed=seed)
    model.fit(X_train, y[idx_train], X_val, y[idx_val])
    y_pred_test = model.predict(X_test)
    given_test = y[idx_test]
    test_acc = float((y_pred_test == given_test).mean())
    n_err = int((y_pred_test != given_test).sum())
    logger.info(f"Test accuracy {test_acc:.5f} | errors {n_err}/{len(given_test)}")

    # ================================================================
    # Main figure: rows = clusters, cols = sampled correct examples
    # ================================================================
    rng = np.random.RandomState(seed)
    clusters = sorted(np.unique(given_test))
    ncol = args.n_per_cluster
    fig, axes = plt.subplots(
        len(clusters), ncol, figsize=(1.7 * ncol, 1.7 * len(clusters)),
        sharex=True, sharey=False, constrained_layout=True, squeeze=False)

    for r, c in enumerate(clusters):
        pool = np.where(given_test == c)[0]
        pick = rng.choice(pool, min(ncol, len(pool)), replace=False)
        cmed = cluster_median[c]
        color = CLUSTER_COLORS[int(c)]
        ytop = cluster_ylim[c]
        for j in range(ncol):
            ax = axes[r][j]
            if j >= len(pick):
                ax.set_visible(False)
                continue
            li = pick[j]
            gi = idx_test[li]
            given, pred = int(given_test[li]), int(y_pred_test[li])
            correct = given == pred
            # cluster-median reference (typical member)
            ax.plot(years, cmed, ls=(0, (5, 2)), color=REF_COLOR, lw=1.1, zorder=1,
                    label="cluster median" if (r == 0 and j == 0) else None)
            # the example (solid, semantic cluster color, no area fill)
            ax.plot(years, trajectories[gi], color=color, lw=1.8, zorder=2)
            ax.set_ylim(0, ytop)            # shared, robust scale within the row
            # error cue: red + BOLD title (bold survives greyscale; the cross
            # glyph is absent in Liberation Sans so weight carries the signal)
            ax.set_title(f"C{given}{ARROW}C{pred}", fontsize=8.5, pad=2,
                         color="black" if correct else ERROR_COLOR,
                         fontweight="normal" if correct else "bold")
            ax.tick_params(labelsize=7)
            if not correct:
                for s in ax.spines.values():
                    s.set_edgecolor(ERROR_COLOR); s.set_linewidth(1.5)
        axes[r][0].set_ylabel(f"Cluster {c}", fontsize=9, fontweight="bold")

    axes[0][0].legend(loc="upper left", fontsize=7)
    _supxlabel(fig, "Years since emergence", fontsize=10)
    _supylabel(fig, "Cumulative reuse count", fontsize=10)
    save_fig(fig, out_dir / f"case_study_{args.model}",
             formats=("pdf", "png") if not args.no_pdf else ("png",))
    logger.info(f"Saved main figure: case_study_{args.model}")

    # ================================================================
    # Error figure: all misclassified test examples
    # ================================================================
    if args.show_errors and n_err > 0:
        err = np.where(y_pred_test != given_test)[0]
        k = min(len(err), 12)
        pick = err if len(err) <= 12 else rng.choice(err, 12, replace=False)
        cols = min(k, 4)
        rows = (k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.0 * rows),
                                 sharex=True, constrained_layout=True,
                                 squeeze=False)
        for jj in range(rows * cols):
            ax = axes[jj // cols][jj % cols]
            if jj >= k:
                ax.set_visible(False)
                continue
            li = pick[jj]
            gi = idx_test[li]
            given, pred = int(given_test[li]), int(y_pred_test[li])
            # assigned-cluster median = dashed REF grey; predicted-cluster
            # median = dotted dark grey (two greys separated by value + dash)
            ax.plot(years, cluster_median[given], ls=(0, (5, 2)), color=REF_COLOR,
                    lw=1.1, label=f"C{given} median (assigned)")
            ax.plot(years, cluster_median[pred], ls=":", color=REF_DARK, lw=1.1,
                    label=f"C{pred} median (predicted)")
            ax.plot(years, trajectories[gi], color=CLUSTER_COLORS[given],
                    lw=1.8, label="this technology")
            ax.set_title(f"C{given}{ARROW}C{pred}", fontsize=8.5, pad=2,
                         color=ERROR_COLOR, fontweight="bold")
            ax.tick_params(labelsize=7)
            if jj == 0:
                ax.legend(fontsize=6.5, loc="lower right")
        _supxlabel(fig, "Years since emergence", fontsize=10)
        _supylabel(fig, "Cumulative reuse count", fontsize=10)
        # (no in-figure title; the "{n_err} of {n_test} misclassified" statement
        #  and the dashed/dotted legend belong in the LaTeX caption)
        save_fig(fig, out_dir / f"case_study_errors_{args.model}",
                 formats=("pdf", "png") if not args.no_pdf else ("png",))
        logger.info(f"Saved error figure: case_study_errors_{args.model}")
    elif args.show_errors:
        logger.info("No misclassified test examples.")

    with open(out_dir / f"case_study_summary_{args.model}.json", "w") as f:
        json.dump({"model": args.model, "features": features,
                   "n_classes": n_classes, "test_accuracy": test_acc,
                   "n_test": int(len(idx_test)), "n_errors": n_err,
                   "n_per_cluster": ncol, "seed": seed}, f, indent=2)
    logger.info(f"Done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
