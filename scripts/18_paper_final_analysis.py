#!/usr/bin/env python3
"""
18_paper_final_analysis.py -- supporting analyses for the reported results.

Each item below closes a specific gap:
  A. Right-censoring: emergence-year cohort distribution; how many technologies have a
     full 20-year window; per-cluster emergence-year composition (are late, censored
     cohorts concentrated in particular clusters?).
  B. Cluster-count justification: k-means on the FAE K=3 features for k in 2..6 with
     silhouette / Calinski-Harabasz / Davies-Bouldin (the paper currently claims k=3 is
     favoured "across multiple criteria" but never shows the sweep).
  C. Cluster characterization: per-cluster feature centroids (raw units) and IPC-section
     composition -- what a scientometrics reader should understand each cluster to be.
  D. Matched-k agreement: ARI/NMI between feature clusters and trajectory-shape clusters
     at matched k (3v3, 4v4) plus the full 3v4 contingency table. Removes the "different
     numbers of clusters" excuse from the independence claim.
  E. Controlled INVENT_APPL ablation on the trajectory task: reproduce the published
     all-7 GBDT run (fingerprint: ROC-AUC 0.8306), then run FAE3+INVENT_APPL (4 feats)
     and all7-INVENT_APPL (6 feats) with the identical protocol.
  F. GRU-latent elbow (k vs inertia, k=2..10), the cluster-count selection criterion.

Trajectories are rebuilt from year_counts exactly as src/data/clustering.py
build_reuse_trajectories does (zeros beyond the data horizon, cumsum), so labels
reproduce the pipeline's. Protocol constants mirror scripts/14 (seed 42, 60/40 split
then 90/10 train/val, scaler fit on train, GBDT fit on the 54% train portion).

Run:  ./venv/bin/python scripts/18_paper_final_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.tabular_models import build_tabular_model

SEED = 42
W = 20
FEATURES = ["ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
            "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE"]
FAE3 = ["SIM_TECH", "ACCESS_SIZE", "SIM_ACCESS"]
OUT = Path("results/paper_final_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def rebuild_trajectories(df):
    ycs = df["year_counts"].values
    y0s = df["emergence_year"].values.astype(int)
    out = np.zeros((len(df), W), dtype=np.float64)
    for i in range(len(df)):
        yc = json.loads(ycs[i])
        y0 = y0s[i]
        out[i] = np.cumsum([yc.get(str(y0 + t), 0) for t in range(W)])
    return out


def znorm(traj):
    mu = traj.mean(axis=1, keepdims=True)
    sd = traj.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (traj - mu) / sd


def traj_task_gbdt(X, y, k, tag):
    """Mirror scripts/14 retrain protocol exactly (seed 42, 60/40, then 90/10,
    scaler on train only, GBDT fit on the 54% train portion)."""
    Xtr_full, Xte, ytr_full, yte = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(
        Xtr_full, ytr_full, test_size=0.1, stratify=ytr_full, random_state=SEED)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    m = build_tabular_model("gbdt", n_features=X.shape[1], n_classes=k,
                            device="cpu", seed=SEED)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    auc = float(roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                              labels=list(range(k))))
    acc = float((m.predict(Xte) == yte).mean())
    log(f"  [{tag}] n_feat={X.shape[1]}  acc={acc:.4f}  ROC-AUC(OvR)={auc:.4f}")
    return {"tag": tag, "n_features": X.shape[1], "accuracy": round(acc, 4),
            "roc_auc_ovr_macro": round(auc, 4)}


def main():
    df = pd.read_csv("results/clustering/technologies_labeled_fae_k3_k3.csv")
    log(f"corpus: {len(df)} technologies")
    feat_k3 = df["cluster"].values.astype(int)
    traj = rebuild_trajectories(df)

    results = {}

    # ---------- A. censoring / cohorts ----------
    log("\n[A] emergence-year cohorts and censoring")
    yr = df["emergence_year"].astype(int)
    cohort = yr.value_counts().sort_index()
    results["cohorts"] = {int(y): int(n) for y, n in cohort.items()}
    latest_full = 2022 - (W - 1)          # emergence <= 2003 has a full 20-yr window
    full_n = int((yr <= latest_full).sum())
    results["censoring"] = {
        "year_min": int(yr.min()), "year_max": int(yr.max()),
        "full_window_cutoff": latest_full,
        "n_full_window": full_n,
        "pct_full_window": round(100 * full_n / len(df), 2),
        "median_observed_years": float(np.median(np.minimum(2022 - yr + 1, W))),
        "mean_observed_years": round(float(np.minimum(2022 - yr + 1, W).mean()), 2),
    }
    log(f"  emergence {yr.min()}-{yr.max()}; full 20-yr window (<= {latest_full}): "
        f"{full_n} ({100*full_n/len(df):.1f}%)")
    # per feature-cluster year composition
    comp = df.groupby("cluster")["emergence_year"].agg(["mean", "median"]).round(2)
    late = df.groupby("cluster")["emergence_year"].apply(
        lambda s: round(100 * (s >= 2013).mean(), 1))
    comp["pct_2013_plus"] = late
    results["cluster_year_composition"] = comp.to_dict("index")
    log(f"  per-cluster emergence year:\n{comp.to_string()}")

    # ---------- B. cluster-count sweep on FAE-3 features ----------
    log("\n[B] k sweep on FAE K=3 features (k=2..6)")
    scaler = StandardScaler()
    Xf3 = scaler.fit_transform(df[FAE3].to_numpy(float))
    ksweep = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xf3)
        lab = km.labels_
        sil = float(silhouette_score(Xf3, lab, sample_size=10000, random_state=SEED))
        ch = float(calinski_harabasz_score(Xf3, lab))
        db = float(davies_bouldin_score(Xf3, lab))
        row = {"k": k, "silhouette": round(sil, 4), "calinski_harabasz": round(ch, 0),
               "davies_bouldin": round(db, 4), "inertia": round(float(km.inertia_), 0),
               "min_cluster_pct": round(100 * np.bincount(lab).min() / len(lab), 2)}
        ksweep.append(row)
        log(f"  k={k}: sil={sil:.4f}  CH={ch:,.0f}  DB={db:.4f}  "
            f"min-cluster={row['min_cluster_pct']}%")
        if k == 3:
            match = float(adjusted_rand_score(lab, feat_k3))
            log(f"    (reproduction check vs CSV labels: ARI={match:.4f})")
            results["k3_reproduction_ari"] = round(match, 4)
        if k == 4:
            feat_k4 = lab.copy()
    results["cluster_k_sweep"] = ksweep

    # ---------- C. cluster characterization ----------
    log("\n[C] cluster characterization (feature centroids + IPC sections)")
    cent = df.groupby("cluster")[FEATURES].mean().round(3)
    results["cluster_feature_means"] = cent.to_dict("index")
    log(f"  feature means per cluster:\n{cent.to_string()}")
    sec = pd.concat([
        df[["cluster"]].assign(section=df["ipc1"].str[0]),
        df[["cluster"]].assign(section=df["ipc2"].str[0]),
    ])
    sec_tab = (sec.groupby(["cluster", "section"]).size()
               .unstack(fill_value=0))
    sec_pct = (100 * sec_tab.div(sec_tab.sum(axis=1), axis=0)).round(1)
    results["ipc_section_pct_by_cluster"] = sec_pct.to_dict("index")
    log(f"  IPC section % per cluster:\n{sec_pct.to_string()}")
    overall = (100 * sec_tab.sum(axis=0) / sec_tab.values.sum()).round(1)
    results["ipc_section_pct_overall"] = overall.to_dict()

    # ---------- D. trajectory labels + matched-k agreement ----------
    log("\n[D] trajectory-shape labels (reproduce) + matched-k ARI")
    Zt = znorm(traj)
    traj_k4 = KMeans(n_clusters=4, n_init=10, max_iter=300,
                     random_state=SEED).fit_predict(Zt)
    traj_k3 = KMeans(n_clusters=3, n_init=10, max_iter=300,
                     random_state=SEED).fit_predict(Zt)
    sil4 = float(silhouette_score(Zt, traj_k4, sample_size=20000, random_state=SEED))
    log(f"  traj k=4 sizes={np.bincount(traj_k4).tolist()}  "
        f"sampled sil={sil4:.4f} (paper full-data value 0.3243)")
    np.save(OUT / "traj_labels_k4.npy", traj_k4)
    np.save(OUT / "traj_labels_k3.npy", traj_k3)
    results["traj_k4_sampled_silhouette"] = round(sil4, 4)
    agree = {
        "feat3_vs_traj4_ARI": adjusted_rand_score(feat_k3, traj_k4),
        "feat3_vs_traj4_NMI": normalized_mutual_info_score(feat_k3, traj_k4),
        "feat3_vs_traj3_ARI": adjusted_rand_score(feat_k3, traj_k3),
        "feat3_vs_traj3_NMI": normalized_mutual_info_score(feat_k3, traj_k3),
        "feat4_vs_traj4_ARI": adjusted_rand_score(feat_k4, traj_k4),
        "feat4_vs_traj4_NMI": normalized_mutual_info_score(feat_k4, traj_k4),
    }
    results["matched_k_agreement"] = {k: round(float(v), 4) for k, v in agree.items()}
    for k, v in results["matched_k_agreement"].items():
        log(f"  {k} = {v}")
    ct = confusion_matrix(feat_k3, traj_k4)
    results["contingency_feat3_traj4"] = ct.tolist()
    log(f"  contingency (rows=feature clusters, cols=trajectory clusters):\n{ct}")

    # ---------- E. controlled INVENT_APPL ablation ----------
    log("\n[E] trajectory-task GBDT: reproduction fingerprints + controlled ablation")
    y = traj_k4.astype(np.int64)
    fits = []
    fits.append(traj_task_gbdt(df[FEATURES].to_numpy(float), y, 4,
                               "all7_reproduction (expect ROC-AUC ~0.8306)"))
    fits.append(traj_task_gbdt(df[FAE3].to_numpy(float), y, 4,
                               "fae3_reproduction (expect ROC-AUC ~0.7400)"))
    fits.append(traj_task_gbdt(df[FAE3 + ["INVENT_APPL"]].to_numpy(float), y, 4,
                               "fae3_plus_INVENT_APPL"))
    six = [f for f in FEATURES if f != "INVENT_APPL"]
    fits.append(traj_task_gbdt(df[six].to_numpy(float), y, 4,
                               "all7_minus_INVENT_APPL"))
    results["trajectory_task_fits"] = fits

    # ---------- F. GRU latent elbow ----------
    log("\n[F] GRU latent elbow (k=2..10)")
    try:
        Z = np.load("results/seq_autoencoder/latent_Z.npy")
        elbow = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
            sil = float(silhouette_score(Z, km.labels_, sample_size=10000,
                                         random_state=SEED))
            elbow.append({"k": k, "inertia": round(float(km.inertia_), 1),
                          "silhouette": round(sil, 4)})
            log(f"  k={k}: inertia={km.inertia_:,.0f}  sil={sil:.4f}")
        results["gru_elbow"] = elbow
        pd.DataFrame(elbow).to_csv(OUT / "gru_elbow.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e = pd.DataFrame(elbow)
        fig, ax1 = plt.subplots(figsize=(6, 4), layout="constrained")
        ax1.plot(e["k"], e["inertia"], "o-", color="tab:blue")
        ax1.set_xlabel("k"); ax1.set_ylabel("k-means inertia (cost)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(e["k"], e["silhouette"], "s--", color="tab:orange")
        ax2.set_ylabel("silhouette (sampled)", color="tab:orange")
        ax1.set_title("GRU latent: k vs cost (elbow) and silhouette")
        fig.savefig(OUT / "gru_elbow.png", dpi=200)
        plt.close(fig)
    except FileNotFoundError:
        log("  latent_Z.npy not found; skipping")

    json.dump(results, open(OUT / "analysis.json", "w"), indent=2, default=float)
    log(f"\nAll results written to {OUT}/analysis.json")


if __name__ == "__main__":
    main()
