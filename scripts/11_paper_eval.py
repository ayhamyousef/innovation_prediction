#!/usr/bin/env python3
"""
11_paper_eval.py: Evaluate FAE using the metrics from Wu & Cheng (AAAI 2021).

The original FAE paper uses two metrics (Section "Design of Experiments"):
  1. Linear reconstruction error: train LinearRegression (no regularization) on
     the selected features to reconstruct the original features, report MSE.
  2. Classification accuracy: pass selected features to an
     ExtremelyRandomizedTrees classifier (sklearn ExtraTreesClassifier),
     report accuracy.

The paper's "stability" claim refers to smooth performance curves as K varies,
not feature-selection consistency under data perturbation (which is what
08_fae_stability.py tests). This script reproduces the paper's evaluation
methodology.

Outputs per K = 1..6:
  - FAE-selected features
  - Linear reconstruction MSE on test data
  - ExtraTrees classification accuracy on test data
  - For comparison: same metrics on random size-K subsets

Usage:
    python scripts/11_paper_eval.py
    python scripts/11_paper_eval.py --k-values 1 2 3 4 5 6 --n-random 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.fae import train_fae
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]


def linear_recon_mse(X_train: np.ndarray, X_test: np.ndarray,
                     selected_idx: list) -> float:
    """Train LinearRegression on selected features to reconstruct all features.
    Returns mean squared error on the test set.
    Mirrors the paper's "linear reconstruction error" metric.
    """
    X_train_sel = X_train[:, selected_idx]
    X_test_sel = X_test[:, selected_idx]
    lr = LinearRegression()
    lr.fit(X_train_sel, X_train)
    X_test_hat = lr.predict(X_test_sel)
    return float(mean_squared_error(X_test, X_test_hat))


def extra_trees_accuracy(X_train, y_train, X_test, y_test,
                          selected_idx: list, seed: int) -> dict:
    """ExtraTreesClassifier on selected features. Returns dict of metrics."""
    clf = ExtraTreesClassifier(n_estimators=100, random_state=seed,
                                n_jobs=-1)
    clf.fit(X_train[:, selected_idx], y_train)
    y_pred = clf.predict(X_test[:, selected_idx])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--k-values", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--cluster-k", type=int, default=3,
                        help="Number of clusters for label generation")
    parser.add_argument("--lambda1", type=float, default=2.0)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-random", type=int, default=10,
                        help="Number of random subsets for comparison per K")
    parser.add_argument("--test-ratio", type=float, default=0.4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    out_dir = Path("results/paper_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))

    tech_path = Path(cfg["data"]["processed_dir"]) / "technologies.csv"
    logger.info(f"Loading {tech_path}")
    tech_df = pd.read_csv(tech_path)
    n_total = len(tech_df)
    logger.info(f"Loaded {n_total:,} rows")

    existing_cols = [c for c in FEATURE_COLS if c in tech_df.columns]
    feat_index = {c: i for i, c in enumerate(existing_cols)}

    X_raw = tech_df[existing_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0)

    # Step 1: generate stratification labels via KMeans on full-data-standardized
    # features. Clustering is unsupervised so using all data here is acceptable
    # and is not the same as classification leakage.
    X_for_clustering = StandardScaler().fit_transform(X_raw)
    km_full = KMeans(n_clusters=args.cluster_k, n_init=10, max_iter=300,
                     random_state=seed)
    full_labels = km_full.fit_predict(X_for_clustering)

    # Step 2: stratified split on RAW features so we can refit the scaler on
    # train only (proper standardization for the downstream classifier).
    X_train_raw, X_test_raw, y_train_full, y_test_full = train_test_split(
        X_raw, full_labels, test_size=args.test_ratio, random_state=seed,
        stratify=full_labels,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    logger.info(f"Train n={len(X_train):,} | Test n={len(X_test):,}")

    rng = np.random.RandomState(seed)
    rows = []

    for k in args.k_values:
        if k >= len(existing_cols):
            logger.info(f"Skipping K={k} (must be < {len(existing_cols)})")
            continue
        logger.info("=" * 60)
        logger.info(f"K = {k}")
        logger.info("=" * 60)

        # ---------- FAE ----------
        logger.info("Running FAE...")
        set_seed(seed)
        _, fae_results = train_fae(
            X=X_train, k=k, feature_names=existing_cols,
            lambda1=args.lambda1, lambda2=args.lambda2,
            lr=args.lr, max_epochs=args.max_epochs,
            patience=args.patience, device=args.device, verbose=False,
        )
        fae_selected = fae_results["selection"]["selected_names"]
        fae_idx = [feat_index[c] for c in fae_selected]
        logger.info(f"  FAE selected: {fae_selected}")

        # Per-K labels: cluster on FAE-selected features in train,
        # then assign train and test using the trained KMeans centroids.
        km_k = KMeans(n_clusters=args.cluster_k, n_init=10, max_iter=300,
                      random_state=seed)
        y_train = km_k.fit_predict(X_train[:, fae_idx])
        y_test = km_k.predict(X_test[:, fae_idx])

        # FAE metrics
        fae_recon = linear_recon_mse(X_train, X_test, fae_idx)
        fae_clf = extra_trees_accuracy(X_train, y_train, X_test, y_test,
                                        fae_idx, seed)
        logger.info(f"  FAE recon MSE: {fae_recon:.6f}")
        logger.info(f"  FAE ExtraTrees acc: {fae_clf['accuracy']:.4f} "
                    f"(macro-F1 {fae_clf['macro_f1']:.4f})")
        rows.append({
            "K": k, "method": "FAE",
            "features": "|".join(fae_selected),
            "recon_mse": fae_recon,
            "accuracy": fae_clf["accuracy"],
            "macro_f1": fae_clf["macro_f1"],
        })

        # ---------- Random subsets (baseline) ----------
        # Use the SAME y_train/y_test as FAE for fair comparison
        for r in range(args.n_random):
            rand_idx = sorted(rng.choice(len(existing_cols), k, replace=False))
            rand_features = [existing_cols[i] for i in rand_idx]
            rand_recon = linear_recon_mse(X_train, X_test, rand_idx)
            rand_clf = extra_trees_accuracy(X_train, y_train, X_test, y_test,
                                            rand_idx, seed)
            rows.append({
                "K": k, "method": f"random_{r:02d}",
                "features": "|".join(rand_features),
                "recon_mse": rand_recon,
                "accuracy": rand_clf["accuracy"],
                "macro_f1": rand_clf["macro_f1"],
            })

        # Quick summary at this K
        random_for_k = [r for r in rows if r["K"] == k and r["method"] != "FAE"]
        rec_arr = np.array([r["recon_mse"] for r in random_for_k])
        acc_arr = np.array([r["accuracy"] for r in random_for_k])
        logger.info(f"  Random (n={len(random_for_k)}): "
                    f"recon MSE {rec_arr.mean():.6f} +/- {rec_arr.std():.6f} | "
                    f"acc {acc_arr.mean():.4f} +/- {acc_arr.std():.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"paper_eval_clusterk{args.cluster_k}.csv", index=False)
    logger.info(f"\nSaved per-row results to {out_dir}/")

    # ============================================================
    # Per-K summary table
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PAPER-STYLE EVAL SUMMARY")
    logger.info("=" * 60)

    summary_rows = []
    for k in sorted(df["K"].unique()):
        sub = df[df["K"] == k]
        fae = sub[sub["method"] == "FAE"].iloc[0]
        rand = sub[sub["method"] != "FAE"]
        summary_rows.append({
            "K": k,
            "fae_features": fae["features"],
            "fae_recon_mse": fae["recon_mse"],
            "fae_accuracy": fae["accuracy"],
            "fae_macro_f1": fae["macro_f1"],
            "random_recon_mse_mean": float(rand["recon_mse"].mean()),
            "random_recon_mse_std": float(rand["recon_mse"].std()),
            "random_accuracy_mean": float(rand["accuracy"].mean()),
            "random_accuracy_std": float(rand["accuracy"].std()),
            "fae_recon_better_than_random": int(
                (fae["recon_mse"] < rand["recon_mse"]).sum()
            ),
            "fae_accuracy_better_than_random": int(
                (fae["accuracy"] > rand["accuracy"]).sum()
            ),
            "n_random": len(rand),
        })
        logger.info(f"\nK = {k}: {fae['features']}")
        logger.info(f"  FAE:    recon MSE = {fae['recon_mse']:.6f}, "
                    f"acc = {fae['accuracy']:.4f}, F1 = {fae['macro_f1']:.4f}")
        logger.info(f"  Random: recon MSE = {rand['recon_mse'].mean():.6f} "
                    f"+/- {rand['recon_mse'].std():.6f}, "
                    f"acc = {rand['accuracy'].mean():.4f} "
                    f"+/- {rand['accuracy'].std():.4f}")
        logger.info(f"  FAE better than random, recon: "
                    f"{summary_rows[-1]['fae_recon_better_than_random']}/{len(rand)}")
        logger.info(f"  FAE better than random, acc:   "
                    f"{summary_rows[-1]['fae_accuracy_better_than_random']}/{len(rand)}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"paper_eval_summary_clusterk{args.cluster_k}.csv",
                   index=False)
    save_json({
        "cluster_k": args.cluster_k,
        "k_values": args.k_values,
        "n_random_per_k": args.n_random,
        "summary": summary_rows,
    }, str(out_dir / f"paper_eval_summary_clusterk{args.cluster_k}.json"))

    logger.info("\nInterpretation:")
    logger.info("  This evaluates FAE the same way Wu & Cheng (2021) did:")
    logger.info("    metric 1: linear reconstruction MSE (lower is better)")
    logger.info("    metric 2: ExtraTrees classification accuracy (higher is better)")
    logger.info("  'Stability' in the paper means smooth curves across K values.")
    logger.info("  Compare FAE's curves to the random-subset bands to assess.")


if __name__ == "__main__":
    main()
