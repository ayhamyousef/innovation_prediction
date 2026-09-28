"""Censoring-free robustness check for the sequence-based labeling.

The corpus runs 2002-2022, so a technology emerging in 2017 is observed for only
five of the twenty years and its trajectory is flat thereafter by construction.
scripts/25 showed the consequence: emergence year alone predicts the k=3
sequence labels almost as well as the seven features, because cluster membership
is partly a function of how much of the window was observed.

This script removes the confound instead of measuring it. Restricting to
technologies that emerged in 2012 or earlier and truncating every trajectory to
its first ten years gives a subcohort in which EVERY technology is observed for
exactly the same length of time. Right-censoring is then absent by construction,
so if the year shortcut is censoring it should collapse here, and whatever
predictability survives is attributable to the features rather than the horizon.

Everything else follows the paper: trajectories rebuilt from year_counts,
per-sequence z-normalization, the GRUSeqAutoencoder of scripts/16 with the same
hyperparameters (1 layer, hidden 64, latent 8, l1 0.01, Adam 1e-3, batch 512,
40 epochs), k-means k=3 on the latent, then GBDT on a stratified 60/40 split.

Reported for the censoring-free subcohort:
  all7        the seven emergence-time features
  year_only   emergence year as the single predictor
  no_access   dropping the two left-truncated features
Compare against the full-corpus figures in results/paper_final_analysis/
recency_shortcut_check.json (all7 0.9143, year_only 0.8736, no_access 0.8498).
"""
import json
import os

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ALL7 = ['ACCESS_SIZE', 'ACCESS_TREND', 'SIM_ACCESS', 'SIM_TECH',
        'INVENT_DIVER', 'INVENT_APPL', 'ATTENT_SIZE']
TRUNCATED = ['ACCESS_SIZE', 'ACCESS_TREND']
CSV = 'results/clustering/technologies_labeled_all7_k3.csv'
OUT = 'results/paper_final_analysis/censoring_free_robustness.json'
SEED, K, HORIZON = 42, 3, 10
# 2012 = every technology fully observed for HORIZON years (censoring-free arm).
# 2022 = all cohorts, so 2013+ are censored within the same HORIZON (control arm).
MAX_COHORT = int(os.environ.get('MAX_COHORT', 2012))
OUT = OUT.replace('.json', f'_cohort{MAX_COHORT}.json')


class GRUSeqAutoencoder(torch.nn.Module):
    """Identical to scripts/16 with cond='repeat'."""

    def __init__(self, latent_dim=8, hidden=64, num_layers=1):
        super().__init__()
        self.enc = torch.nn.GRU(1, hidden, num_layers=num_layers, batch_first=True)
        self.enc_to_z = torch.nn.Linear(hidden, latent_dim)
        self.dec = torch.nn.GRU(latent_dim, hidden, num_layers=num_layers, batch_first=True)
        self.dec_to_out = torch.nn.Linear(hidden, 1)

    def encode(self, x):
        _, h = self.enc(x)
        return self.enc_to_z(h[-1])

    def forward(self, x):
        z = self.encode(x)
        dec_in = z.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.dec(dec_in)
        return self.dec_to_out(out), z


def build_trajectories(df, horizon):
    """Cumulative reuse over `horizon` years starting at the emergence year."""
    traj = np.zeros((len(df), horizon), dtype=np.float32)
    for i, (yc, y0) in enumerate(zip(df.year_counts.values, df.emergence_year.values)):
        counts = json.loads(yc)
        row = np.zeros(horizon, dtype=np.float32)
        for ys, c in counts.items():
            off = int(ys) - int(y0)
            if 0 <= off < horizon:
                row[off] = c
        traj[i] = np.cumsum(row)
    return traj


def znorm(traj):
    mu = traj.mean(axis=1, keepdims=True)
    sd = traj.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (traj - mu) / sd


def fit_autoencoder(Xn, seed, epochs=40, l1=0.01):
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  training GRU autoencoder on {dev}, {Xn.shape[0]:,} sequences of length {Xn.shape[1]}")
    X = torch.tensor(Xn, dtype=torch.float32, device=dev).unsqueeze(-1)
    model = GRUSeqAutoencoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()
    n, bs = X.size(0), 512
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        model.train()
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = X[idx]
            xhat, z = model(xb)
            loss = mse(xhat, xb) + l1 * z.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{epochs}  loss {tot / n:.5f}")
    model.eval()
    with torch.no_grad():
        Z = torch.cat([model.encode(X[s:s + 4096]) for s in range(0, n, 4096)]).cpu().numpy()
    return Z


def downstream(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    m = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                   learning_rate=0.1, random_state=SEED).fit(Xtr, ytr)
    pred, proba = m.predict(Xte), m.predict_proba(Xte)
    r = {'n_features': X.shape[1],
         'accuracy': round(float(accuracy_score(yte, pred)), 4),
         'macro_f1': round(float(f1_score(yte, pred, average='macro')), 4),
         'roc_auc_ovr_macro': round(float(roc_auc_score(yte, proba, multi_class='ovr',
                                                        average='macro', labels=list(range(K)))), 4)}
    print(f"  {name:12s} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  "
          f"ROC-AUC={r['roc_auc_ovr_macro']:.4f}")
    return r


def main():
    full = pd.read_csv(CSV)
    sub = full[full.emergence_year <= MAX_COHORT].reset_index(drop=True)
    print(f"censoring-free subcohort: emerged <= {MAX_COHORT}, observed {HORIZON} full years")
    print(f"  {len(sub):,} of {len(full):,} technologies ({100 * len(sub) / len(full):.1f}%)")

    traj = build_trajectories(sub, HORIZON)
    print(f"  mean year-{HORIZON} cumulative reuse: {traj[:, -1].mean():.1f}")

    Z = fit_autoencoder(znorm(traj), SEED)
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
    labels = km.labels_
    sizes = np.bincount(labels)
    rng = np.random.RandomState(SEED)
    samp = rng.choice(len(Z), size=min(10000, len(Z)), replace=False)
    sil = float(silhouette_score(Z[samp], labels[samp]))
    print(f"  k={K} cluster sizes {sizes.tolist()}  silhouette {sil:.4f}")

    out = {'horizon_years': HORIZON, 'max_cohort': MAX_COHORT,
           'n': int(len(sub)), 'n_full_corpus': int(len(full)),
           'pct_of_corpus': round(100 * len(sub) / len(full), 1),
           'cluster_sizes': sizes.tolist(), 'silhouette_latent': round(sil, 4),
           'cluster_median_year': {int(c): int(sub.emergence_year[labels == c].median())
                                   for c in range(K)},
           'downstream': {}}

    print("\ndownstream prediction of the censoring-free labels:")
    out['downstream']['all7'] = downstream(sub[ALL7].values.astype(np.float64), labels, 'all7')
    out['downstream']['year_only'] = downstream(
        sub[['emergence_year']].values.astype(np.float64), labels, 'year_only')
    out['downstream']['no_access'] = downstream(
        sub[[c for c in ALL7 if c not in TRUNCATED]].values.astype(np.float64), labels, 'no_access')

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, 'w'), indent=2)

    a = out['downstream']['all7']['roc_auc_ovr_macro']
    y = out['downstream']['year_only']['roc_auc_ovr_macro']
    print(f"\n  censoring-free: all7 {a}, year_only {y}, gap {round(a - y, 4)}")
    print(f"  full corpus   : all7 0.9143, year_only 0.8736, gap 0.0407")
    print(f"\n  -> {'year shortcut SHRINKS without censoring' if (a - y) > 0.0407 else 'year shortcut PERSISTS without censoring'}")


if __name__ == '__main__':
    main()
