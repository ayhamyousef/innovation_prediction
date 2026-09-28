#!/usr/bin/env python3
"""
15_paper_figures.py — Publication figures from already-computed numeric results
(runs locally; no trajectories needed).

  1. fraction_sweep         - macro-F1 vs training fraction, per model.
  2. feature_budget_trajectory - trajectory ROC-AUC vs FAE budget (INVENT_APPL).
  3. fae_vs_all7_trajectory - FAE-K3 vs all-7 ROC-AUC across classifiers.

Design follows src/utils/plotting.py (named colors, marker/dash disambiguation,
shared ROC-AUC y-limits, greyscale-safe encodings).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.plotting import (
    set_paper_style, save_fig, MODEL_STYLE, MODEL_LABEL,
    REF_COLOR, BAND_ALPHA, CLUSTER_COLORS,
)

OUT = Path("results/figures/paper")
CHEN_AUC = 0.728
# Shared limits for the two trajectory-ROC-AUC figures so they read side by side.
ROC_YLIM = (0.70, 0.86)
ROC_TICK = 0.02


# ------------------------------------------------------------------
# Figure 1: training-fraction sweep
# ------------------------------------------------------------------
def fig_fraction_sweep():
    csv = Path("results/training_fraction_sweep/fae_k3/fraction_sweep.csv")
    if not csv.exists():
        print(f"  skip fraction sweep: {csv} not found")
        return
    df = pd.read_csv(csv)
    df = df[df["status"] == "OK"]
    agg = (df.groupby(["model", "fraction"])["test_macro_f1"]
             .agg(["mean", "std"]).reset_index())

    order = ["tabnet", "gbdt", "extra_trees", "ft_transformer", "tabm"]
    order = [m for m in order if m in agg["model"].unique()]

    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    for m in order:
        st = MODEL_STYLE[m]
        d = agg[agg["model"] == m].sort_values("fraction")
        ax.plot(d["fraction"], d["mean"], color=st["color"], marker=st["marker"],
                linestyle=st["ls"], markersize=4.5, label=MODEL_LABEL[m])
        lo = d["mean"] - d["std"].fillna(0)
        hi = d["mean"] + d["std"].fillna(0)
        ax.fill_between(d["fraction"], lo, hi, color=st["color"],
                        alpha=BAND_ALPHA, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Training-set fraction")
    ax.set_ylabel("Test macro-$F_1$")
    ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["1%", "5%", "10%", "25%", "50%", "100%"])
    ax.set_ylim(0.955, 1.002)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              frameon=False, handlelength=2.2)
    save_fig(fig, OUT / "fraction_sweep")
    print("  wrote fraction_sweep")


# ------------------------------------------------------------------
# Figure 2: feature-budget trajectory sweep (GBDT)
# ------------------------------------------------------------------
def _auc(path):
    return json.load(open(path))["retrain"]["roc_auc_ovr_macro"] if path.exists() else None


def fig_feature_budget():
    base = Path("results/label_transfer")
    pts = [
        ("FAE $K{=}3$", _auc(base / "label_transfer_gbdt_retrain_k4.json"), False),
        ("FAE $K{=}4$", _auc(base / "label_transfer_gbdt_retrain_k4_faeK4.json"), True),
        ("FAE $K{=}5$", _auc(base / "label_transfer_gbdt_retrain_k4_faeK5.json"), True),
        ("All 7",       _auc(base / "label_transfer_gbdt_retrain_k4_all7.json"), True),
    ]
    pts = [p for p in pts if p[1] is not None]
    if not pts:
        print("  skip feature budget: no GBDT retrain JSONs")
        return
    labels = [p[0] for p in pts]
    aucs = [p[1] for p in pts]
    has = [p[2] for p in pts]
    x = np.arange(len(pts))

    fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")
    ax.plot(x, aucs, "-", color=REF_COLOR, lw=1.2, zorder=1)
    # binary encoding OFF the cluster palette: present = blue filled circle,
    # absent = grey open square (shape carries the signal in greyscale too).
    for xi, auc, h in zip(x, aucs, has):
        if h:
            ax.plot(xi, auc, "o", color="#0072B2", markersize=8, zorder=3)
        else:
            ax.plot(xi, auc, "s", markerfacecolor="white", markeredgecolor=REF_COLOR,
                    markeredgewidth=1.3, markersize=8, zorder=3)
        # 9pt cleared the marker centre but not its 8pt glyph; 13pt clears
        # both the marker and the connecting line.
        ax.annotate(f"{auc:.3f}", (xi, auc), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=7)

    ax.axhline(CHEN_AUC, ls=(0, (5, 2)), color=REF_COLOR, lw=1.0)
    # 0.99 put the string flush against the right spine.
    ax.text(0.965, CHEN_AUC + 0.004, "Chen et al. (2025): 0.728",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=7.5, color=REF_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(pts) - 0.6)
    ax.set_ylabel("Trajectory-task ROC-AUC")
    ax.set_ylim(*ROC_YLIM)
    ax.yaxis.set_major_locator(MultipleLocator(ROC_TICK))
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
               markersize=8, label="with INVENT_APPL"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor=REF_COLOR, markeredgewidth=1.3, markersize=8,
               label="without INVENT_APPL"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    save_fig(fig, OUT / "feature_budget_trajectory")
    print("  wrote feature_budget_trajectory")


# ------------------------------------------------------------------
# Figure 3: FAE K=3 vs all-7 across models (trajectory task)
# ------------------------------------------------------------------
def fig_fae_vs_all7():
    base = Path("results/label_transfer")
    models = ["gbdt", "ft_transformer", "tabmixer", "tabm"]
    present, fae, allf = [], [], []
    for m in models:
        a = _auc(base / f"label_transfer_{m}_retrain_k4.json")
        b = _auc(base / f"label_transfer_{m}_retrain_k4_all7.json")
        if a is None or b is None:
            continue
        present.append(m); fae.append(a); allf.append(b)
    if not present:
        print("  skip FAE-vs-all7: missing JSONs")
        return

    x = np.arange(len(present))
    w = 0.38
    fig, ax = plt.subplots(figsize=(3.6, 2.9), layout="constrained")
    b1 = ax.bar(x - w / 2, fae, w, color="#0072B2", label="FAE $K{=}3$ (3 features)",
                edgecolor="white", linewidth=0.4)
    b2 = ax.bar(x + w / 2, allf, w, color="#E69F00", label="All 7 features",
                hatch="//", edgecolor="white", linewidth=0.4)
    chen = ax.axhline(CHEN_AUC, ls=(0, (5, 2)), color=REF_COLOR, lw=1.0,
                      label="Chen et al. (2025): 0.728")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in present])
    ax.set_ylabel("Trajectory-task ROC-AUC")
    ax.set_ylim(*ROC_YLIM)
    ax.yaxis.set_major_locator(MultipleLocator(ROC_TICK))
    # Chen baseline goes in the legend (every x has a bar at y=0.728, so an
    # in-plot annotation would overlap the bars).
    ax.legend([b1, b2, chen],
              ["FAE $K{=}3$ (3 features)", "All 7 features",
               "Chen et al. (2025): 0.728"],
              loc="upper left", frameon=False, ncol=1)
    save_fig(fig, OUT / "fae_vs_all7_trajectory")
    print("  wrote fae_vs_all7_trajectory")


def main():
    set_paper_style()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Generating paper figures from local numeric results:")
    fig_fraction_sweep()
    fig_feature_budget()
    fig_fae_vs_all7()
    print(f"Done. Figures in {OUT}/")


if __name__ == "__main__":
    main()
