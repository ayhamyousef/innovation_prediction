#!/usr/bin/env python3
"""
16_sequence_autoencoder.py -- GRU autoencoder for sequence-based clustering.

This implements the approach Dr. Ye specified (2026-07-15 meeting): learn a latent
representation Z of each reuse trajectory with a plain GRU autoencoder, optionally push
Z toward a sparse (cluster-revealing) structure with an L1 penalty, then cluster Z.

Design decisions from that meeting, encoded here:
  - Plain GRU, NOT NC-GRU. Sequences are length ~20, so the orthogonal-matrix machinery
    for long-term dependencies is unnecessary.
  - Architecture = encoder (sequence -> latent vector Z) + decoder (Z -> sequence),
    trained on reconstruction. This is the paper's AutoEncoder box; the Molecular
    Properties Consistency Network (which regularized the latent against 7 external
    properties) is DELETED. We do NOT regularize Z against the 7 features -- that would
    re-create the circularity from paper 1. The only regularizer on Z is the L1 penalty.
  - Cluster the latent Z (k-means), not the raw sequence. Two Z's that are close should
    share a trajectory pattern.
  - Grid: {1 layer, 3 layers} x {L1 off/on} x {latent dim 8, 16, 32 (and optionally 3)}.
    Start simple: 1 layer, dim 16, no L1.

Usage:
    # quick end-to-end self-test on synthetic archetypes (no real data needed):
    python scripts/16_sequence_autoencoder.py --synthetic

    # single real run (on speedy3, where trajectories.npy exists):
    python scripts/16_sequence_autoencoder.py --latent-dim 16 --layers 1

    # the full grid Ye asked for:
    python scripts/16_sequence_autoencoder.py --grid
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class GRUSeqAutoencoder(nn.Module):
    """Encoder GRU maps a length-T scalar sequence to a latent Z; decoder GRU maps Z
    back to the sequence. Real-valued input (one channel), MSE reconstruction."""

    def __init__(self, latent_dim=16, hidden=64, num_layers=1, cond="repeat"):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.num_layers = num_layers
        self.cond = cond  # "repeat" = feed Z at every decoder step; "h0" = Z as initial state

        self.enc = nn.GRU(1, hidden, num_layers=num_layers, batch_first=True)
        self.enc_to_z = nn.Linear(hidden, latent_dim)

        dec_in = latent_dim if cond == "repeat" else 1
        self.z_to_h0 = nn.Linear(latent_dim, hidden) if cond == "h0" else None
        self.dec = nn.GRU(dec_in, hidden, num_layers=num_layers, batch_first=True)
        self.dec_to_out = nn.Linear(hidden, 1)

    def encode(self, x):                       # x: (B, T, 1)
        _, h = self.enc(x)                     # h: (num_layers, B, hidden)
        return self.enc_to_z(h[-1])            # Z: (B, latent_dim)

    def decode(self, z, T):
        B = z.size(0)
        if self.cond == "repeat":
            dec_in = z.unsqueeze(1).repeat(1, T, 1)      # (B, T, latent_dim)
            out, _ = self.dec(dec_in)
        else:  # "h0": Z initializes the decoder hidden state; inputs are zeros
            h0 = torch.tanh(self.z_to_h0(z)).unsqueeze(0).repeat(self.num_layers, 1, 1)
            dec_in = torch.zeros(B, T, 1, device=z.device)
            out, _ = self.dec(dec_in, h0.contiguous())
        return self.dec_to_out(out)            # (B, T, 1)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z, x.size(1)), z


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def normalize_trajectories(traj, mode):
    """traj: (n, T). Shape-focused normalization so clustering sees pattern not scale."""
    traj = traj.astype(np.float32)
    if mode == "none":
        return traj
    if mode == "per-seq-max":
        m = traj.max(axis=1, keepdims=True)
        m[m == 0] = 1.0
        return traj / m
    # default: per-sequence z-normalization (matches paper 1's Validation 2)
    mu = traj.mean(axis=1, keepdims=True)
    sd = traj.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (traj - mu) / sd


def make_synthetic(n=3000, T=20, seed=42):
    """Three archetypes matching the real clusters: moderate S-growth, fast growth,
    early plateau. Returns (traj, true_labels). Lets us verify the pipeline recovers
    known structure end-to-end without the real data."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, T)
    traj, labels = [], []
    for i in range(n):
        c = i % 3
        if c == 0:      # moderate steady growth (gentle S-curve)
            base = 130 / (1 + np.exp(-8 * (t - 0.55)))
        elif c == 1:    # fast growth (accelerating, high total)
            base = 250 * (t ** 2.2)
        else:           # early plateau (rise fast then flatten)
            base = 50 * (1 - np.exp(-6 * t))
        noise = rng.normal(0, 0.06 * (base.max() + 1e-6), size=T)
        seq = np.maximum.accumulate(np.clip(base + noise, 0, None))  # keep monotone-ish
        traj.append(seq)
        labels.append(c)
    return np.array(traj, dtype=np.float32), np.array(labels)


def load_real(cfg_processed_dir, labeled_csv, max_n=None, seed=42):
    traj = np.load(Path(cfg_processed_dir) / "trajectories.npy")
    ref = None
    p = Path(labeled_csv)
    if p.exists():
        import pandas as pd
        ref = pd.read_csv(p)["cluster"].values.astype(np.int64)
        if len(ref) != len(traj):
            print(f"  warning: label count {len(ref)} != trajectory count {len(traj)}; "
                  "ignoring reference labels")
            ref = None
    if max_n and max_n < len(traj):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(traj), max_n, replace=False)
        traj = traj[idx]
        ref = ref[idx] if ref is not None else None
    return traj, ref


# ----------------------------------------------------------------------
# Train + evaluate one configuration
# ----------------------------------------------------------------------
def run_one(traj_norm, ref_labels, latent_dim, num_layers, l1, k,
            hidden, epochs, cond, device, seed=42, quiet=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(traj_norm[:, :, None], dtype=torch.float32, device=device)
    model = GRUSeqAutoencoder(latent_dim, hidden, num_layers, cond).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    n = X.size(0)
    bs = min(512, n)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
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
        if not quiet and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep+1:3d}  loss {tot/n:.4f}")

    # extract latent + reconstruction error
    model.eval()
    with torch.no_grad():
        xhat, Z = model(X)
        recon = mse(xhat, X).item()
    Z = Z.cpu().numpy()

    # cluster the latent
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    pred = km.labels_

    # sparsity of Z (only meaningful with L1): fraction of entries near zero
    thr = 0.05 * (np.abs(Z).max() + 1e-9)
    sparsity = float((np.abs(Z) < thr).mean())

    sil = float(silhouette_score(Z, pred)) if k < len(Z) else float("nan")
    out = {
        "latent_dim": latent_dim, "layers": num_layers, "l1": l1, "k": k,
        "recon_mse": round(recon, 5), "silhouette_Z": round(sil, 4),
        "sparsity_frac": round(sparsity, 3),
    }
    if ref_labels is not None:
        out["ARI_vs_ref"] = round(float(adjusted_rand_score(ref_labels, pred)), 4)
        out["NMI_vs_ref"] = round(float(normalized_mutual_info_score(ref_labels, pred)), 4)
    return out, Z, pred


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run on synthetic 3-archetype data (self-test, no real data)")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--labeled-csv",
                    default="results/clustering/technologies_labeled_fae_k3_k3.csv")
    ap.add_argument("--normalize", default="per-seq-z",
                    choices=["per-seq-z", "per-seq-max", "none"])
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--l1", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--cond", default="repeat", choices=["repeat", "h0"])
    ap.add_argument("--max-n", type=int, default=None,
                    help="subsample this many trajectories (for quick runs)")
    ap.add_argument("--grid", action="store_true",
                    help="run {1,3 layers} x {L1 0, 1e-2} x {dim 8,16,32}")
    ap.add_argument("--out-dir", default="results/seq_autoencoder")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    if args.synthetic:
        traj, ref = make_synthetic(n=3000, T=20)
        print(f"Synthetic data: {traj.shape} with 3 known archetypes | device={device}")
    else:
        import yaml
        cfg = yaml.safe_load(open(args.config))
        traj, ref = load_real(cfg["data"]["processed_dir"], args.labeled_csv,
                              max_n=args.max_n)
        print(f"Real data: {traj.shape} | ref labels: "
              f"{'yes' if ref is not None else 'no'} | device={device}")

    traj_norm = normalize_trajectories(traj, args.normalize)

    # ---- run ----
    if args.grid:
        configs = list(product([1, 3], [0.0, 1e-2], [8, 16, 32]))
        rows = []
        print(f"Running grid of {len(configs)} configs "
              "({{1,3}} layers x {{L1 off,on}} x {{dim 8,16,32}})")
        for layers, l1, dim in configs:
            print(f"  layers={layers} l1={l1} dim={dim}")
            m, _, _ = run_one(traj_norm, ref, dim, layers, l1, args.k,
                              args.hidden, args.epochs, args.cond, device, quiet=True)
            rows.append(m)
            print(f"    -> recon {m['recon_mse']:.4f}  sil {m['silhouette_Z']:.3f}  "
                  f"sparsity {m['sparsity_frac']:.2f}"
                  + (f"  ARI {m.get('ARI_vs_ref')}" if ref is not None else ""))
        json.dump(rows, open(out_dir / "grid_results.json", "w"), indent=2)
        # also a compact CSV
        keys = list(rows[0].keys())
        with open(out_dir / "grid_results.csv", "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"\nGrid written to {out_dir}/grid_results.[json,csv]")
    else:
        print(f"Single run: dim={args.latent_dim} layers={args.layers} "
              f"l1={args.l1} k={args.k}")
        m, Z, pred = run_one(traj_norm, ref, args.latent_dim, args.layers, args.l1,
                             args.k, args.hidden, args.epochs, args.cond, device)
        print("\nResult:", json.dumps(m, indent=2))
        np.save(out_dir / "latent_Z.npy", Z)
        np.save(out_dir / "cluster_labels.npy", pred)
        json.dump(m, open(out_dir / "metrics.json", "w"), indent=2)
        print(f"Saved latent + labels + metrics to {out_dir}/")


if __name__ == "__main__":
    main()
