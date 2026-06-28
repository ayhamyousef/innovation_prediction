#!/usr/bin/env python3
"""
15_paper_figures.py — Publication-quality figures from already-computed numeric
results (no trajectories needed; runs locally).

Produces:
  1. Training-fraction sweep: macro-F1 vs training fraction, per model, with
     standard-deviation bands over seeds.
  2. Feature-budget trajectory sweep: ROC-AUC on the trajectory-shape task as
     the FAE budget grows (K=3 -> K=4 -> K=5 -> all 7), with Chen et al.'s
     baseline marked. Shows the jump when INVENT_APPL enters.
  3. FAE K=3 vs all-7 across classifiers on the trajectory task (grouped bars).

Usage:
    python scripts/15_paper_figures.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.plotting import set_paper_style, save_fig, PALETTE

OUT = Path("results/figures/paper")
MODEL_LABEL = {
    "gbdt": "GBDT", "extra_trees": "ExtraTrees", "tabnet": "TabNet",
    "tabm": "TabM", "ft_transformer": "FT-Transformer",
    "tabmixer": "TabMixer", "tabkan": "TabKAN",
}
CHEN_AUC = 0.728


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

    # order models by their full-data F1 (best last so it draws on top)
    order = ["tabnet", "gbdt", "extra_trees", "ft_transformer", "tabm"]
    order = [m for m in order if m in agg["model"].unique()]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, m in enumerate(order):
        d = agg[agg["model"] == m].sort_values("fraction")
        c = PALETTE[i % len(PALETTE)]
        ax.plot(d["fraction"], d["mean"], marker="o", color=c,
                label=MODEL_LABEL.get(m, m))
        lo = d["mean"] - d["std"].fillna(0)
        hi = d["mean"] + d["std"].fillna(0)
        ax.fill_between(d["fraction"], lo, hi, color=c, alpha=0.15, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Training-set fraction")
    ax.set_ylabel("Test macro-$F_1$")
    ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5, 1.0])
    ax.set_xticklabels(["1%", "5%", "10%", "25%", "50%", "100%"])
    ax.set_ylim(0.95, 1.001)
    ax.legend(loc="lower right", ncol=2)
    save_fig(fig, OUT / "fraction_sweep")
    print("  wrote fraction_sweep.pdf/png")


# ------------------------------------------------------------------
# Figure 2: feature-budget trajectory sweep (GBDT)
# ------------------------------------------------------------------
def _load_auc(path):
    if not path.exists():
        return None
    return json.load(open(path))["retrain"]["roc_auc_ovr_macro"]


def fig_feature_budget():
    base = Path("results/label_transfer")
    points = [
        ("FAE $K{=}3$", _load_auc(base / "label_transfer_gbdt_retrain_k4.json"), False),
        ("FAE $K{=}4$", _load_auc(base / "label_transfer_gbdt_retrain_k4_faeK4.json"), True),
        ("FAE $K{=}5$", _load_auc(base / "label_transfer_gbdt_retrain_k4_faeK5.json"), True),
        ("All 7",       _load_auc(base / "label_transfer_gbdt_retrain_k4_all7.json"), True),
    ]
    points = [(lbl, auc, has) for (lbl, auc, has) in points if auc is not None]
    if not points:
        print("  skip feature budget: no GBDT retrain JSONs found")
        return

    labels = [p[0] for p in points]
    aucs = [p[1] for p in points]
    has_invent = [p[2] for p in points]
    x = np.arange(len(points))

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.plot(x, aucs, "-", color="#444444", zorder=1, linewidth=1.4)
    for xi, auc, has in zip(x, aucs, has_invent):
        c = PALETTE[2] if has else PALETTE[1]  # green if has INVENT_APPL else vermillion
        ax.plot(xi, auc, "o", color=c, markersize=8, zorder=3)
        ax.annotate(f"{auc:.3f}", (xi, auc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ax.axhline(CHEN_AUC, ls="--", color="#888888", linewidth=1.2)
    ax.annotate("Chen et al. (0.728)", (0, CHEN_AUC),
                textcoords="offset points", xytext=(2, 5), ha="left",
                fontsize=8, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Trajectory-task ROC-AUC")
    ax.set_ylim(0.70, 0.85)
    # legend proxies
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[1],
               markersize=8, label="without INVENT_APPL"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[2],
               markersize=8, label="with INVENT_APPL"),
    ]
    ax.legend(handles=handles, loc="center right")
    save_fig(fig, OUT / "feature_budget_trajectory")
    print("  wrote feature_budget_trajectory.pdf/png")


# ------------------------------------------------------------------
# Figure 3: FAE K=3 vs all-7 across models (trajectory task)
# ------------------------------------------------------------------
def fig_fae_vs_all7():
    base = Path("results/label_transfer")
    models = ["gbdt", "ft_transformer", "tabmixer", "tabm"]
    fae, allf = [], []
    present = []
    for m in models:
        a = _load_auc(base / f"label_transfer_{m}_retrain_k4.json")
        b = _load_auc(base / f"label_transfer_{m}_retrain_k4_all7.json")
        if a is None or b is None:
            continue
        present.append(m)
        fae.append(a)
        allf.append(b)
    if not present:
        print("  skip FAE-vs-all7: missing JSONs")
        return

    x = np.arange(len(present))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(x - w / 2, fae, w, color=PALETTE[0], label="FAE $K{=}3$ (3 feat.)")
    ax.bar(x + w / 2, allf, w, color=PALETTE[1], label="All 7 features")
    ax.axhline(CHEN_AUC, ls="--", color="#888888", linewidth=1.2)
    ax.annotate("Chen et al. (0.728)", (-0.45, CHEN_AUC),
                textcoords="offset points", xytext=(0, 4), ha="left",
                fontsize=8, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in present], rotation=15)
    ax.set_ylabel("Trajectory-task ROC-AUC")
    ax.set_ylim(0.70, 0.86)
    ax.legend(loc="upper right")
    save_fig(fig, OUT / "fae_vs_all7_trajectory")
    print("  wrote fae_vs_all7_trajectory.pdf/png")


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
