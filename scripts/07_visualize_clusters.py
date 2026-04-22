#!/usr/bin/env python3
"""
07_visualize_clusters.py — Publication-quality cluster visualizations.

Produces four plot types for a chosen feature config + cluster count:
  1. 2D PCA scatter of technologies colored by cluster
  2. Feature-distribution violin/box plots per cluster
  3. Cluster-mean trajectories (cumulative reuse over time)
  4. Representative trajectory examples (closest-to-centroid) per cluster

Default target: FAE K=3 features, k=3 clusters — the winning config.

Usage:
    # Defaults (fae_k3_k3, the best config)
    python scripts/07_visualize_clusters.py

    # Different config
    python scripts/07_visualize_clusters.py --labeled-csv \\
        results/clustering/technologies_labeled_fae_k4_k4.csv

    # PNG only (skip PDF)
    python scripts/07_visualize_clusters.py --no-pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config, setup_logging, set_seed


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]

# Colorblind-friendly palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
           "#56B4E9", "#E69F00"]


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_fig(fig, out_path: Path, also_pdf: bool = True):
    fig.savefig(out_path.with_suffix(".png"))
    if also_pdf:
        fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_pca_scatter(X_scaled, labels, feature_names, out_path, also_pdf=True,
                     sample_n: int = 20000):
    """2D PCA projection colored by cluster."""
    rng = np.random.RandomState(42)
    if len(X_scaled) > sample_n:
        idx = rng.choice(len(X_scaled), sample_n, replace=False)
        X_plot, lab_plot = X_scaled[idx], labels[idx]
    else:
        X_plot, lab_plot = X_scaled, labels

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X_plot)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in sorted(np.unique(lab_plot)):
        mask = lab_plot == c
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.45,
                   color=PALETTE[c % len(PALETTE)],
                   label=f"Cluster {c} (n={int(mask.sum())})",
                   linewidths=0)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(f"Clusters in feature space\nfeatures: {', '.join(feature_names)}")
    leg = ax.legend(markerscale=2, frameon=True)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    save_fig(fig, out_path, also_pdf)


def plot_feature_boxplots(tech_df, feature_names, out_path, also_pdf=True):
    """Per-feature boxplot broken down by cluster."""
    n = len(feature_names)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows),
                             squeeze=False)
    clusters = sorted(tech_df["cluster"].unique())
    box_colors = [PALETTE[c % len(PALETTE)] for c in clusters]

    for i, feat in enumerate(feature_names):
        ax = axes[i // cols][i % cols]
        data = [tech_df.loc[tech_df["cluster"] == c, feat].values
                for c in clusters]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        widths=0.6, medianprops={"color": "black"})
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels([f"C{c}" for c in clusters])
        ax.set_title(feat)
        ax.set_ylabel("value")
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Feature distributions by cluster", y=1.02)
    save_fig(fig, out_path, also_pdf)


def plot_mean_trajectories(tech_df, trajectories, valid_idx, out_path,
                           also_pdf=True):
    """Mean cumulative-reuse curve per cluster with 25-75 percentile band."""
    clusters = sorted(tech_df["cluster"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    window = trajectories.shape[1]
    years = np.arange(1, window + 1)

    # Map tech_df index -> row in trajectories (valid_idx is positional)
    traj = trajectories
    for c in clusters:
        mask = (tech_df["cluster"].values == c)
        traj_c = traj[mask]
        if len(traj_c) == 0:
            continue
        mean = traj_c.mean(axis=0)
        p25 = np.percentile(traj_c, 25, axis=0)
        p75 = np.percentile(traj_c, 75, axis=0)
        color = PALETTE[c % len(PALETTE)]
        ax.fill_between(years, p25, p75, alpha=0.18, color=color)
        ax.plot(years, mean, color=color, lw=2.2,
                label=f"Cluster {c} (n={len(traj_c):,})")
    ax.set_xlabel("Years since emergence")
    ax.set_ylabel("Cumulative reuse count")
    ax.set_title("Cluster-mean reuse trajectories (shaded: 25-75 percentile)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.3)
    save_fig(fig, out_path, also_pdf)


def plot_example_trajectories(tech_df, trajectories, feature_names, out_path,
                              also_pdf=True, n_examples: int = 6):
    """Representative trajectories per cluster (closest to cluster centroid in feature space)."""
    clusters = sorted(tech_df["cluster"].unique())
    X = StandardScaler().fit_transform(tech_df[feature_names].values)
    window = trajectories.shape[1]
    years = np.arange(1, window + 1)

    fig, axes = plt.subplots(1, len(clusters),
                             figsize=(3.8 * len(clusters), 3.8), sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    for ax, c in zip(axes, clusters):
        mask = tech_df["cluster"].values == c
        if not mask.any():
            continue
        X_c = X[mask]
        traj_c = trajectories[mask]
        centroid = X_c.mean(axis=0)
        dists = np.linalg.norm(X_c - centroid, axis=1)
        nearest = np.argsort(dists)[:n_examples]
        color = PALETTE[c % len(PALETTE)]
        for i in nearest:
            ax.plot(years, traj_c[i], color=color, alpha=0.55, lw=1.3)
        # Overlay the cluster mean
        ax.plot(years, traj_c.mean(axis=0), color="black", lw=2.0,
                linestyle="--", label="cluster mean")
        ax.set_title(f"Cluster {c}\n(n={int(mask.sum()):,})")
        ax.set_xlabel("Years since emergence")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Cumulative reuse count")
    fig.suptitle(f"Representative trajectories (n={n_examples} nearest to centroid)",
                 y=1.05)
    save_fig(fig, out_path, also_pdf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--labeled-csv",
                        default="results/clustering/technologies_labeled_fae_k3_k3.csv",
                        help="Labeled CSV from 04b_recluster.py")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Features used for clustering. If None, inferred "
                             "from filename (fae_k{K}_...) or uses all 7.")
    parser.add_argument("--fae-k", type=int, default=None,
                        help="FAE K value (for feature inference from disk)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (auto-generated if None)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF output (PNG only)")
    parser.add_argument("--n-examples", type=int, default=6,
                        help="Representative trajectories per cluster")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])
    set_paper_style()

    csv_path = Path(args.labeled_csv)
    if args.out_dir is None:
        args.out_dir = f"results/figures/{csv_path.stem}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))
    logger.info(f"Loading {csv_path}")
    tech_df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(tech_df)} technologies, "
                f"{tech_df['cluster'].nunique()} clusters")

    # Determine features
    if args.features is not None:
        features = list(args.features)
    else:
        # Try to infer from filename or sidecar JSON
        import json as _json
        if args.fae_k is not None:
            fae_path = Path(f"results/feature_selection/fae_k{args.fae_k}_results.json")
            if fae_path.exists():
                features = _json.load(open(fae_path))["selection"]["selected_names"]
            else:
                features = FEATURE_COLS
        else:
            # parse filename: technologies_labeled_fae_k3_k3.csv -> fae_k3
            stem = csv_path.stem
            if "fae_k" in stem:
                fae_k = int(stem.split("fae_k")[1].split("_")[0])
                fae_path = Path(f"results/feature_selection/fae_k{fae_k}_results.json")
                if fae_path.exists():
                    features = _json.load(open(fae_path))["selection"]["selected_names"]
                else:
                    features = FEATURE_COLS
            else:
                features = FEATURE_COLS
    logger.info(f"Using features: {features}")

    # Standardize for PCA
    X = tech_df[features].values.astype(np.float64)
    X_scaled = StandardScaler().fit_transform(X)
    labels = tech_df["cluster"].values.astype(int)

    # Load trajectories (aligned to tech_df by row order — same source in 04b_recluster)
    processed_dir = Path(cfg["data"]["processed_dir"])
    traj_path = processed_dir / "trajectories.npy"
    trajectories = None
    if traj_path.exists():
        trajectories = np.load(traj_path)
        if len(trajectories) != len(tech_df):
            logger.warning(
                f"Trajectory count ({len(trajectories)}) != tech_df rows "
                f"({len(tech_df)}). Trajectory plots will be skipped."
            )
            trajectories = None
    else:
        logger.warning(f"No trajectories.npy at {traj_path} — skipping trajectory plots")

    # Plot 1: PCA scatter
    logger.info("Plot 1/4: PCA scatter")
    plot_pca_scatter(X_scaled, labels, features,
                     out_dir / "pca_scatter", also_pdf=not args.no_pdf)

    # Plot 2: feature boxplots
    logger.info("Plot 2/4: feature distributions")
    plot_feature_boxplots(tech_df, features,
                          out_dir / "feature_boxplots", also_pdf=not args.no_pdf)

    # Plots 3 & 4 need trajectories
    if trajectories is not None:
        logger.info("Plot 3/4: cluster-mean trajectories")
        plot_mean_trajectories(tech_df, trajectories, None,
                               out_dir / "mean_trajectories",
                               also_pdf=not args.no_pdf)

        logger.info("Plot 4/4: representative example trajectories")
        plot_example_trajectories(tech_df, trajectories, features,
                                  out_dir / "example_trajectories",
                                  also_pdf=not args.no_pdf,
                                  n_examples=args.n_examples)
    else:
        logger.info("Skipping trajectory plots (no trajectories.npy)")

    logger.info(f"\nFigures saved to {out_dir}/")


if __name__ == "__main__":
    main()
