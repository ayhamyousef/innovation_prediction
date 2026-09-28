#!/usr/bin/env python3
"""
21_seed_stability_and_k2.py -- two artifact gaps found by verification.

A. Persist the k=3 downstream seed-stability result: GRU k=3 labels held
   FIXED (ksweep labels_k3.npy), GBDT all-7 refit under three train/test
   split seeds. (Previously computed in a scratch log only.)
B. k=2 row for the sequence-labeling sweep table: k-means (k=2) on the GRU
   latent, sampled silhouette, smallest-cluster share, and GBDT all-7
   downstream ROC-AUC under the standard protocol.

Run:  ./venv/bin/python scripts/21_seed_stability_and_k2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.tabular_models import build_tabular_model

FEATS = ["ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
         "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE"]
OUT = Path("results/paper_final_analysis")


def gbdt_downstream(X, y, k, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y,
                                          random_state=seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    m = build_tabular_model("gbdt", n_features=X.shape[1], n_classes=k,
                            device="cpu", seed=seed)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte)
    pred = m.predict(Xte)
    if k == 2:
        auc = roc_auc_score(yte, proba[:, 1])
    else:
        auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                            labels=list(range(k)))
    return {"seed": seed,
            "accuracy": round(float((pred == yte).mean()), 4),
            "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
            "roc_auc_ovr_macro": round(float(auc), 4)}


def main():
    df = pd.read_csv("results/clustering/technologies_labeled_fae_k3_k3.csv")
    X = df[FEATS].to_numpy(float)
    Z = np.load("results/seq_autoencoder/latent_Z.npy")
    y3 = np.load("results/seq_autoencoder/ksweep/labels_k3.npy").astype(np.int64)

    out = {}
    print("[A] k=3 seed stability (labels fixed, split seed varies)", flush=True)
    out["k3_split_seed_stability"] = []
    for s in (42, 7, 123):
        r = gbdt_downstream(X, y3, 3, s)
        out["k3_split_seed_stability"].append(r)
        print(f"  seed {s}: AUC {r['roc_auc_ovr_macro']}", flush=True)

    print("[B] k=2 sweep row", flush=True)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Z)
    y2 = km.labels_.astype(np.int64)
    sizes = np.bincount(y2)
    sil = float(silhouette_score(Z, y2, sample_size=10000, random_state=42))
    r2 = gbdt_downstream(X, y2, 2, 42)
    out["k2_row"] = {"silhouette": round(sil, 4),
                     "sizes": sizes.tolist(),
                     "smallest_cluster_pct": round(100 * sizes.min() / len(y2), 1),
                     "gbdt_all7": r2}
    print(f"  sil {sil:.4f} sizes {sizes.tolist()} AUC {r2['roc_auc_ovr_macro']}",
          flush=True)

    json.dump(out, open(OUT / "gru_downstream_seeds_and_k2.json", "w"), indent=2)
    print(f"-> {OUT}/gru_downstream_seeds_and_k2.json", flush=True)


if __name__ == "__main__":
    main()
