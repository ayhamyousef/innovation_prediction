#!/usr/bin/env python3
"""
14_label_transfer_validation.py — Validation against an independently constructed labeling.

Addresses the concern that our unsupervised labeling (k-means on feature
vectors) shares inputs with the supervised classifier. The test:

  1. Train the classifier on OUR k-means feature-cluster labels (the main
     pipeline), reproducing the exact 60/40 split from 05_classify.py.
  2. Independently generate Chen et al.-style labels by clustering the
     reuse TRAJECTORIES (not features) with k-means, following Chen et al.'s
     shape-based clustering. This is a fully independent labeling scheme.
  3. WITHOUT retraining, apply the trained classifier to the test set and
     measure how well its predictions agree with the trajectory-shape labels.

If agreement is high, the classifier has learned structure that generalizes
across labeling schemes, not just memorized the k-means partition. That
directly answers the labeling concern.

Label-space note: our classifier outputs k=3 feature-cluster predictions.
Trajectory clustering can use a different k (Chen et al. used 4). To compare:
  - Agreement metrics (Adjusted Rand Index, Normalized Mutual Info) need NO
    label matching and work across different cluster counts. These are the
    primary, most rigorous numbers.
  - When k matches (default 3), we also Hungarian-match the labels and report
    accuracy / macro-F1 / ROC-AUC, the "performance on new labels" framing
    a target external to the features, alongside the reference GBDT ROC-AUC of 0.728
    reported by Chen et al. (2025) on a different corpus.

Backup mode (--retrain): if transfer is weak, retrain the classifier from
scratch on the trajectory-shape labels (supports k=4) and evaluate normally.

Usage:
    # Primary transfer test (k=3 trajectory clusters to match our model)
    python scripts/14_label_transfer_validation.py \
        --labeled-csv results/clustering/technologies_labeled_fae_k3_k3.csv \
        --features SIM_TECH ACCESS_SIZE SIM_ACCESS \
        --model tabm --traj-k 3

    # Agreement-only at Chen et al.'s k=4 (no matched accuracy possible)
    python scripts/14_label_transfer_validation.py --traj-k 4

    # Backup: retrain on trajectory-shape labels (k=4) and evaluate normally
    python scripts/14_label_transfer_validation.py --traj-k 4 --retrain
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    adjusted_rand_score, normalized_mutual_info_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clustering import cluster_trajectories
from src.models.tabular_models import build_tabular_model
from src.utils.helpers import load_config, setup_logging, set_seed, save_json


FEATURE_COLS = [
    "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
    "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
]
CHEN_GBDT_ROC_AUC = 0.728  # Chen et al. (2025) best classifier, for reference


def fae_k3_features():
    p = Path("results/feature_selection/fae_k3_results.json")
    if p.exists():
        return json.load(open(p))["selection"]["selected_names"]
    return ["SIM_TECH", "ACCESS_SIZE", "SIM_ACCESS"]


def hungarian_remap(pred, true, k):
    """Optimally relabel `pred` to match `true` (both in 0..k-1) by maximizing
    overlap. Returns (remapped_pred, mapping dict pred_label -> true_label)."""
    cm = confusion_matrix(true, pred, labels=list(range(k)))  # rows=true, cols=pred
    row_ind, col_ind = linear_sum_assignment(-cm)             # maximize overlap
    mapping = {int(c): int(r) for r, c in zip(row_ind, col_ind)}
    remapped = np.array([mapping.get(int(p), int(p)) for p in pred])
    return remapped, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--labeled-csv",
                        default="results/clustering/technologies_labeled_fae_k3_k3.csv")
    parser.add_argument("--features", nargs="+", default=None,
                        help="Features for classification. Default: FAE K=3.")
    parser.add_argument("--model", default="tabm",
                        choices=["tabnet", "tabm", "ft_transformer",
                                 "extra_trees", "gbdt", "tabkan", "tabmixer"])
    parser.add_argument("--traj-k", type=int, default=3,
                        help="Number of trajectory-shape clusters (Chen et al. "
                             "used 4; use 3 to match our model for matched metrics)")
    parser.add_argument("--traj-method", default="euclidean_kmeans",
                        choices=["euclidean_kmeans", "dtw_kmeans"],
                        help="Trajectory clustering method. Euclidean is fast and "
                             "couples 92.85%% with DTW per Chen et al. Appendix A.")
    parser.add_argument("--retrain", action="store_true",
                        help="Backup mode: retrain classifier on trajectory-shape "
                             "labels and evaluate normally.")
    parser.add_argument("--tag", default=None,
                        help="Optional suffix for the output filename, to avoid "
                             "overwriting (e.g. 'all7' for the all-7-feature run).")
    parser.add_argument("--test-ratio", type=float, default=0.4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--out-dir", default="results/label_transfer")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["seed"]
    set_seed(seed)

    features = args.features if args.features else fae_k3_features()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir))
    logger.info(f"Labeled CSV: {args.labeled_csv}")
    logger.info(f"Features: {features}")
    logger.info(f"Model: {args.model} | trajectory clustering: "
                f"{args.traj_method} k={args.traj_k} | retrain={args.retrain}")

    # ---- Load data + aligned trajectories ----
    tech_df = pd.read_csv(args.labeled_csv)
    n = len(tech_df)
    processed_dir = Path(cfg["data"]["processed_dir"])
    trajectories = np.load(processed_dir / "trajectories.npy")
    if len(trajectories) != n:
        logger.error(f"Trajectory count ({len(trajectories)}) != tech rows ({n}).")
        sys.exit(1)
    logger.info(f"Loaded {n:,} technologies and aligned trajectories")

    X = np.nan_to_num(tech_df[features].values.astype(np.float32), nan=0.0)
    y_feat = tech_df["cluster"].values.astype(np.int64)   # our k-means feature labels
    feat_k = int(tech_df["cluster"].nunique())

    # ---- Independent trajectory-shape labels for ALL technologies ----
    # Cluster the whole population (matches Chen et al.'s approach), then index
    # out the test subset. Trajectory clustering never sees the feature labels.
    logger.info(f"Clustering ALL {n:,} trajectories "
                f"({args.traj_method}, k={args.traj_k}) ...")
    traj_labels_all, traj_info = cluster_trajectories(
        trajectories, n_clusters=args.traj_k, method=args.traj_method,
        normalize=True, random_state=seed,
    )
    logger.info(f"Trajectory clustering done. silhouette="
                f"{traj_info.get('silhouette', float('nan')):.4f}")

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    logger.info(f"Device: {device}")

    results = {
        "model": args.model,
        "features": features,
        "feature_clusters_k": feat_k,
        "trajectory_clusters_k": args.traj_k,
        "trajectory_method": args.traj_method,
        "trajectory_silhouette": float(traj_info.get("silhouette", float("nan"))),
        "chen_gbdt_roc_auc_reference": CHEN_GBDT_ROC_AUC,
    }

    # ================================================================
    # MODE A: TRANSFER TEST (default) — train on feature labels, test
    #         agreement with trajectory labels, NO retraining.
    # ================================================================
    if not args.retrain:
        # Reproduce the exact 05_classify split via indices
        idx = np.arange(n)
        idx_train_full, idx_test = train_test_split(
            idx, test_size=args.test_ratio, stratify=y_feat, random_state=seed
        )
        idx_train, idx_val = train_test_split(
            idx_train_full, test_size=args.val_ratio,
            stratify=y_feat[idx_train_full], random_state=seed
        )
        logger.info(f"Split — train {len(idx_train)}, val {len(idx_val)}, "
                    f"test {len(idx_test)}")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[idx_train]).astype(np.float32)
        X_val = scaler.transform(X[idx_val]).astype(np.float32)
        X_test = scaler.transform(X[idx_test]).astype(np.float32)

        logger.info(f"Training {args.model} on OUR feature-cluster labels ...")
        model = build_tabular_model(
            model_name=args.model, n_features=len(features),
            n_classes=feat_k, device=device, seed=seed,
        )
        model.fit(X_train, y_feat[idx_train], X_val, y_feat[idx_val])

        pred_test = model.predict(X_test)            # feature-cluster predictions
        proba_test = model.predict_proba(X_test)
        given_feat_test = y_feat[idx_test]           # our labels on test
        traj_test = traj_labels_all[idx_test]        # trajectory labels on test

        # Sanity: classifier vs its own training labels
        own_acc = float(accuracy_score(given_feat_test, pred_test))
        logger.info(f"Classifier accuracy on OUR labels (sanity): {own_acc:.5f}")

        # --- Matching-free agreement (works for any traj_k) ---
        ari_pred_traj = float(adjusted_rand_score(traj_test, pred_test))
        nmi_pred_traj = float(normalized_mutual_info_score(traj_test, pred_test))
        # How much the two LABELING schemes agree, independent of the classifier:
        ari_feat_traj = float(adjusted_rand_score(traj_test, given_feat_test))
        nmi_feat_traj = float(normalized_mutual_info_score(traj_test, given_feat_test))

        logger.info("=" * 60)
        logger.info("AGREEMENT (matching-free, primary metrics)")
        logger.info("=" * 60)
        logger.info(f"  ARI  predicted vs trajectory labels : {ari_pred_traj:.4f}")
        logger.info(f"  NMI  predicted vs trajectory labels : {nmi_pred_traj:.4f}")
        logger.info(f"  ARI  our-labels vs trajectory labels: {ari_feat_traj:.4f}")
        logger.info(f"  NMI  our-labels vs trajectory labels: {nmi_feat_traj:.4f}")
        logger.info("  (ARI/NMI of 1.0 = identical partitions; 0.0 = random)")

        results["transfer"] = {
            "classifier_accuracy_on_own_labels": own_acc,
            "ari_predicted_vs_trajectory": ari_pred_traj,
            "nmi_predicted_vs_trajectory": nmi_pred_traj,
            "ari_ourlabels_vs_trajectory": ari_feat_traj,
            "nmi_ourlabels_vs_trajectory": nmi_feat_traj,
        }

        # --- Matched metrics (only when label counts match) ---
        if args.traj_k == feat_k:
            remapped_pred, mapping = hungarian_remap(pred_test, traj_test, feat_k)
            acc = float(accuracy_score(traj_test, remapped_pred))
            f1 = float(f1_score(traj_test, remapped_pred, average="macro"))
            # permute proba columns to trajectory-label space, then ROC-AUC
            proba_remap = np.zeros_like(proba_test)
            for j in range(feat_k):
                proba_remap[:, mapping[j]] = proba_test[:, j]
            try:
                auc = float(roc_auc_score(
                    traj_test, proba_remap, multi_class="ovr",
                    average="macro", labels=list(range(feat_k)),
                ))
            except Exception as e:
                auc = None
                logger.warning(f"ROC-AUC failed: {e}")

            logger.info("=" * 60)
            logger.info("MATCHED METRICS (Hungarian label alignment)")
            logger.info("=" * 60)
            logger.info(f"  pred->traj label mapping: {mapping}")
            logger.info(f"  Accuracy vs trajectory labels : {acc:.4f}")
            logger.info(f"  Macro-F1 vs trajectory labels : {f1:.4f}")
            if auc is not None:
                logger.info(f"  ROC-AUC  vs trajectory labels : {auc:.4f}  "
                            f"(Chen et al. GBDT = {CHEN_GBDT_ROC_AUC})")
            results["transfer"]["matched"] = {
                "label_mapping": mapping,
                "accuracy": acc,
                "macro_f1": f1,
                "roc_auc_ovr_macro": auc,
            }
        else:
            logger.info(f"traj_k ({args.traj_k}) != feature_k ({feat_k}); "
                        "skipping matched accuracy/F1/AUC. ARI/NMI above are the "
                        "valid cross-k comparison.")

        # --- Interpretation hint ---
        logger.info("\nInterpretation:")
        logger.info("  If ARI(predicted, trajectory) is close to "
                    "ARI(our-labels, trajectory), the classifier reproduces the "
                    "feature-cluster structure, and the gap to trajectory labels "
                    "is a property of the two labeling schemes, not the model.")

    # ================================================================
    # MODE B: RETRAIN backup — train directly on trajectory labels.
    # ================================================================
    else:
        logger.info("RETRAIN mode: training classifier on trajectory-shape labels")
        y_traj = traj_labels_all.astype(np.int64)
        traj_k = args.traj_k

        Xtr_full, Xte, ytr_full, yte = train_test_split(
            X, y_traj, test_size=args.test_ratio,
            stratify=y_traj, random_state=seed,
        )
        Xtr, Xva, ytr, yva = train_test_split(
            Xtr_full, ytr_full, test_size=args.val_ratio,
            stratify=ytr_full, random_state=seed,
        )
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr).astype(np.float32)
        Xva = scaler.transform(Xva).astype(np.float32)
        Xte = scaler.transform(Xte).astype(np.float32)
        logger.info(f"Split — train {len(ytr)}, val {len(yva)}, test {len(yte)}")
        logger.info(f"Trajectory-label distribution (train): "
                    f"{dict((int(c), int((ytr==c).sum())) for c in range(traj_k))}")

        model = build_tabular_model(
            model_name=args.model, n_features=len(features),
            n_classes=traj_k, device=device, seed=seed,
        )
        model.fit(Xtr, ytr, Xva, yva)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)

        acc = float(accuracy_score(yte, pred))
        f1 = float(f1_score(yte, pred, average="macro"))
        try:
            auc = float(roc_auc_score(yte, proba, multi_class="ovr",
                                      average="macro", labels=list(range(traj_k))))
        except Exception as e:
            auc = None
            logger.warning(f"ROC-AUC failed: {e}")

        logger.info("=" * 60)
        logger.info("RETRAIN ON TRAJECTORY LABELS — TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Accuracy : {acc:.4f}")
        logger.info(f"  Macro-F1 : {f1:.4f}")
        if auc is not None:
            logger.info(f"  ROC-AUC  : {auc:.4f}  "
                        f"(Chen et al. GBDT = {CHEN_GBDT_ROC_AUC})")
            if auc > CHEN_GBDT_ROC_AUC:
                logger.info("  -> Beats Chen et al.'s GBDT baseline.")
            else:
                logger.info("  -> Does not beat Chen et al.'s GBDT baseline.")

        results["retrain"] = {
            "accuracy": acc, "macro_f1": f1, "roc_auc_ovr_macro": auc,
            "beats_chen_baseline": (auc is not None and auc > CHEN_GBDT_ROC_AUC),
        }

    # ---- Save ----
    mode = "retrain" if args.retrain else "transfer"
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"label_transfer_{args.model}_{mode}_k{args.traj_k}{suffix}.json"
    save_json(results, str(out_path))
    logger.info(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
