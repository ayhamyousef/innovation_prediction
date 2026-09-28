#!/usr/bin/env python3
"""
13_case_study_viz.py -- case study visualization.

Randomly sample technologies from the TEST set, plot their actual cumulative
reuse trajectories, and annotate each with the assigned label (unsupervised
k-means cluster) and the predicted label (trained classifier). Each panel
overlays the cluster mean trajectory as a reference. A second figure shows all
misclassified test examples.

Reproduces the exact 60/40 split of 05_classify.py (same seed and stratification)
so the test set is identical, then trains the chosen classifier.

Usage:
    python scripts/13_case_study_viz.py --model tabmixer --n-per-cluster 5 --show-errors

Note on reproducibility
-----------------------
This script retrains the classifier, and the error count is sensitive to the
numerical environment rather than to the seed alone. On the classification task the
model misclassifies a handful of the 80,684 test technologies, and BLAS threading
order can move which ones, so a rerun on different hardware may report a slightly
different count at the same seed=42. The published figures come from the reference
run recorded in results/case_study/case_study_summary_tabmixer.json; check that file
before replacing them, so that the figure and the reported count agree.

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
    traj_npy = Path(cfg["data"]["processed_dir"]) / "trajectories.npy"
    if traj_npy.exists():
        trajectories = np.load(traj_npy)
    else:
        # data/processed/ is only populated on the machine that ran scripts 1-3.
        # The labeled CSV carries year_counts, so the same cumulative
        # trajectories can be rebuilt row-aligned with tech_df. Verified against
        # the published cluster means (129.6 / 253.1 / 50.1 at year 20).
        logger.info(f"{traj_npy} absent; rebuilding trajectories from year_counts")
        W_ = int(cfg["data"].get("window_years", 20))
        _yc = tech_df["year_counts"].values
        _y0 = tech_df["emergence_year"].values.astype(int)
        trajectories = np.zeros((len(tech_df), W_), dtype=np.float64)
        for _i in range(len(tech_df)):
            _d = json.loads(_yc[_i])
            trajectories[_i] = np.cumsum(
                [_d.get(str(_y0[_i] + _t), 0) for _t in range(W_)])
    if len(trajectories) != n:
        logger.error("Trajectory/row count mismatch; aborting.")
        sys.exit(1)

    X = np.nan_to_num(tech_df[features].values.astype(np.float32), nan=0.0)
    y = tech_df["cluster"].values.astype(np.int64)
    n_classes = int(tech_df["cluster"].nunique())
    window = trajectories.shape[1]
    years = np.arange(1, window + 1)

    # Cluster MEDIAN trajectory (typical member; robust to the right-skew that
    # makes the mean unrepresentative). Panel y-limits are set per panel from
    # the displayed example + its median (below), so every small multiple fills
    # its axes. A row-shared 95th-percentile scale flattened most panels in a
    # row whenever the random sample happened to include one fast-growth member.
    cluster_median = {c: np.median(trajectories[y == c], axis=0)
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
        for j in range(ncol):
            ax = axes[r][j]
            if j >= len(pick):
                ax.set_visible(False)
                continue
            li = pick[j]
            gi = idx_test[li]
            traj = trajectories[gi]
            given, pred = int(given_test[li]), int(y_pred_test[li])
            correct = given == pred
            # cluster-median reference (typical member)
            ax.plot(years, cmed, ls=(0, (5, 2)), color=REF_COLOR, lw=1.1, zorder=1,
                    label="cluster median" if (r == 0 and j == 0) else None)
            # the example (solid, semantic cluster color, no area fill)
            ax.plot(years, traj, color=color, lw=1.8, zorder=2)
            # per-panel y-limit from this example + its median, so the SHAPE is
            # always legible. Clusters differ in magnitude by design; the median
            # reference inside each panel carries the scale comparison.
            # Cap the panel relative to the cluster median so BOTH the example
            # and the median reference stay legible. A rare member can exceed
            # its cluster median by an order of magnitude; clipping that line
            # and stating its true year-20 value keeps the shape comparison
            # readable without hiding the magnitude.
            cap = 1.15 * max(float(traj.max()), float(cmed.max()))
            ceiling = 4.0 * float(cmed.max())
            if float(traj.max()) > ceiling:
                cap = 1.15 * ceiling
                ax.annotate(f"reaches {traj[-1]:,.0f}", xy=(0.96, 0.95),
                            xycoords="axes fraction", ha="right", va="top",
                            fontsize=6.5, color=color)
            ax.set_ylim(0, cap)
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
        # Up to five errors fit on one row; a 4+1 grid left three dead cells
        # and, with sharex, stripped the x tick labels from the top row.
        cols = k if k <= 5 else 4
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
            ax.tick_params(labelsize=7, labelbottom=True)
        # One figure-level legend below the row, instead of one inside panel 0
        # sitting on top of that panel's trajectory.
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="outside lower center", ncols=3, fontsize=7,
                   frameon=False)
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
