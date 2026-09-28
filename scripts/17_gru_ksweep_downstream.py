#!/usr/bin/env python3
"""
17_gru_ksweep_downstream.py -- cluster-count sweep and downstream prediction for the
GRU sequence clustering.

Two questions:
  (1) Sweep k in {3,4,5} on the trained GRU latent Z and check whether more clusters
      give something interpretable. The k=3 run left a ~69% catch-all cluster, so 4-5
      groups may split it into recognizable shapes.
  (2) Downstream classification as an INDEPENDENT validation of the sequence clustering
      (on top of silhouette). The GRU clusters are derived from the reuse trajectories,
      not from the 7 emergence-time features, so we test how well those features can
      *predict* the GRU cluster label. Good predictability means the trajectory clusters
      correspond to structure that is recoverable at emergence time; poor predictability
      means the sequence clustering found something the features do not encode. We run
      this for each k as a second, external check beyond silhouette.

No GPU / no GRU retrain: the GRU is already trained and its latent Z is saved. This
script only re-clusters Z (k-means) and trains tabular classifiers on the 7 features.

Alignment: rows of the labeled feature CSV are in the same order as latent_Z. Verified
by reproducing ARI(GRU k=3 labels, feature clusters) = 0.0425 from the original run;
the script re-checks this and warns if it drifts.

Run (laptop, CPU):
    ./venv/bin/python scripts/17_gru_ksweep_downstream.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.tabular_models import build_tabular_model

# The seven emergence-time features. FAE3 is the K=3 subset FAE selected, kept so that
# the all-seven (0.831) versus FAE-3 (0.740) contrast from the trajectory-shape task can
# be mirrored on this target.
FEATURES = ["ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
            "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE"]
FAE3 = ["SIM_TECH", "ACCESS_SIZE", "SIM_ACCESS"]
EXPECTED_ARI_K3 = 0.0425   # from the reference run recorded in metrics.json


def reconstruct_traj(df, W=20):
    """Cumulative reuse over a W-year window from year_counts. Used only for the
    cluster-shape figures; the downstream classification uses the tabular features."""
    ycs = df["year_counts"].values
    y0s = df["emergence_year"].values.astype(int)
    out = np.zeros((len(df), W), dtype=np.float64)
    for i in range(len(df)):
        yc = json.loads(ycs[i])
        y0 = y0s[i]
        counts = [yc.get(str(y0 + t), 0) for t in range(W)]
        out[i] = np.cumsum(counts)
    return out


def plot_shapes(traj, labels, k, path):
    """Median (+IQR band) raw trajectory per learned cluster -- the interpretability
    do 4 or 5 clusters correspond to distinct, recognizable shapes?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    years = np.arange(1, traj.shape[1] + 1)
    # order clusters by year-20 median so the legend reads small -> large
    present = [c for c in range(k) if (labels == c).any()]
    order = sorted(present, key=lambda c: np.median(traj[labels == c][:, -1]))
    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")
    for c in order:
        m = traj[labels == c]
        med = np.median(m, axis=0)
        q1, q3 = np.percentile(m, [25, 75], axis=0)
        ln, = ax.plot(years, med, lw=2,
                      label=f"cluster {c}  (n={len(m):,}, {100*len(m)/len(traj):.1f}%)")
        ax.fill_between(years, q1, q3, color=ln.get_color(), alpha=0.15, linewidth=0)
    ax.set_xlabel("Years since emergence")
    ax.set_ylabel("Cumulative reuse count")
    ax.set_title(f"GRU sequence clusters (k={k}): median trajectory, IQR band")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"    saved shape figure: {path}")


def downstream(X, y, k, feat_name, models, test_ratio=0.4, seed=42):
    """Predict the GRU cluster label from tabular features. Same protocol as the paper's
    trajectory-task retrain (stratified 60/40, standardize on train, OvR macro ROC-AUC)."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    rows = []
    for mname in models:
        model = build_tabular_model(mname, n_features=X.shape[1], n_classes=k,
                                    device="cpu", seed=seed)
        model.fit(Xtr, ytr, Xte, yte)          # sklearn wrappers ignore the val args
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)
        acc = float(accuracy_score(yte, pred))
        f1 = float(f1_score(yte, pred, average="macro"))
        try:
            auc = float(roc_auc_score(yte, proba, multi_class="ovr",
                                      average="macro", labels=list(range(k))))
        except Exception as e:
            auc = float("nan")
            print(f"    ROC-AUC failed (k={k}, {mname}): {e}")
        rows.append({"k": k, "features": feat_name, "model": mname,
                     "n_features": X.shape[1], "accuracy": round(acc, 4),
                     "macro_f1": round(f1, 4), "roc_auc_ovr_macro": round(auc, 4)})
        print(f"    k={k} {feat_name:5s} {mname:12s} "
              f"acc={acc:.4f}  macroF1={f1:.4f}  ROC-AUC(OvR)={auc:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", default="results/seq_autoencoder/latent_Z.npy")
    ap.add_argument("--csv",
                    default="results/clustering/technologies_labeled_fae_k3_k3.csv")
    ap.add_argument("--saved-k3-labels",
                    default="results/seq_autoencoder/cluster_labels.npy")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--out-dir", default="results/seq_autoencoder/ksweep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = np.load(args.latent)
    df = pd.read_csv(args.csv)
    print(f"latent Z: {Z.shape} | CSV rows: {len(df)}")
    if len(df) != len(Z):
        raise SystemExit("ERROR: CSV/latent length mismatch -- cannot align.")

    # ---- alignment check: reproduce ARI(saved GRU k3, feature clusters) ----
    saved_k3 = np.load(args.saved_k3_labels)
    ari = round(float(adjusted_rand_score(saved_k3, df["cluster"].values)), 4)
    tag = "OK" if abs(ari - EXPECTED_ARI_K3) < 0.005 else "WARNING: drifted"
    print(f"alignment check: ARI(GRU k3, feature clusters) = {ari} "
          f"(expected {EXPECTED_ARI_K3}) -> {tag}")
    if tag != "OK":
        print("  proceeding, but verify CSV order matches trajectories.npy order.")

    traj = reconstruct_traj(df)
    X_all7 = df[FEATURES].to_numpy(dtype=np.float64)
    X_fae3 = df[FAE3].to_numpy(dtype=np.float64)

    ksweep_rows, downstream_rows = [], []
    for k in args.ks:
        print(f"\n=== k = {k} ===")
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(Z)
        labels = km.labels_
        sil = float(silhouette_score(Z, labels, sample_size=min(10000, len(Z)),
                                     random_state=args.seed))
        sizes = {int(c): int((labels == c).sum()) for c in range(k)}
        np.save(out_dir / f"labels_k{k}.npy", labels)
        yr20 = {int(c): round(float(np.median(traj[labels == c][:, -1])), 1)
                for c in range(k)}
        print(f"  silhouette(Z) = {sil:.4f}")
        print(f"  cluster sizes = {sizes}")
        print(f"  cluster yr20 median reuse = {yr20}")
        ksweep_rows.append({"k": k, "silhouette_Z": round(sil, 4),
                            "sizes": json.dumps(sizes),
                            "yr20_median_reuse": json.dumps(yr20)})
        plot_shapes(traj, labels, k, out_dir / f"cluster_shapes_k{k}.png")

        # downstream: features -> GRU label. all-7 (GBDT + ExtraTrees) and FAE-3 (GBDT).
        print("  downstream classification (features -> GRU cluster label):")
        downstream_rows += downstream(X_all7, labels, k, "all7",
                                      ["gbdt", "extra_trees"], seed=args.seed)
        downstream_rows += downstream(X_fae3, labels, k, "fae3",
                                      ["gbdt"], seed=args.seed)

    pd.DataFrame(ksweep_rows).to_csv(out_dir / "ksweep_summary.csv", index=False)
    pd.DataFrame(downstream_rows).to_csv(out_dir / "downstream_summary.csv", index=False)
    json.dump({"ksweep": ksweep_rows, "downstream": downstream_rows},
              open(out_dir / "results.json", "w"), indent=2)
    print(f"\nWrote ksweep_summary.csv, downstream_summary.csv, results.json to {out_dir}/")


if __name__ == "__main__":
    main()
