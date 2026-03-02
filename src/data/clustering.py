"""
Technology reuse trajectory construction and clustering.

Builds cumulative reuse curves for each novel technology, then clusters
them using DTW + k-means to produce the 4 innovation growth curve labels:
  Cluster 1: S-shaped trajectory
  Cluster 2: Fleeting trajectory
  Cluster 3: Linear trajectory
  Cluster 4: Exponential trajectory
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("innovation_prediction.clustering")

# Optional: use tslearn for DTW k-means (more efficient)
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.metrics import dtw as tslearn_dtw
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    HAS_TSLEARN = True
except ImportError:
    HAS_TSLEARN = False
    logger.warning("tslearn not installed; falling back to manual DTW")

try:
    from dtaidistance import dtw as dtaid_dtw
    from dtaidistance import dtw_ndim
    HAS_DTAIDISTANCE = True
except ImportError:
    HAS_DTAIDISTANCE = False


# ============================================================
# Trajectory Construction
# ============================================================

def build_reuse_trajectories(tech_df: pd.DataFrame,
                              observation_window: int = 20,
                              min_reuse: int = 20) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build cumulative reuse trajectories for each technology.

    Args:
        tech_df: DataFrame with columns [tech_id, emergence_year, year_counts]
        observation_window: number of years to observe after emergence
        min_reuse: minimum total reuse count to include

    Returns:
        filtered_df: filtered tech DataFrame
        trajectories: numpy array of shape (n_techs, observation_window)
                      containing cumulative reuse counts
    """
    logger.info(f"Building reuse trajectories (window={observation_window}, "
                f"min_reuse={min_reuse})...")

    trajectories = []
    valid_indices = []

    for idx, row in tech_df.iterrows():
        year_counts = row["year_counts"]
        if isinstance(year_counts, str):
            import json
            year_counts = json.loads(year_counts)

        emergence = row["emergence_year"]

        # Build annual counts for the observation window
        annual = np.zeros(observation_window, dtype=np.float64)
        for y_offset in range(observation_window):
            y = emergence + y_offset
            annual[y_offset] = year_counts.get(str(y), year_counts.get(y, 0))

        # Cumulative distribution
        cumulative = np.cumsum(annual)
        total = cumulative[-1]

        if total >= min_reuse:
            trajectories.append(cumulative)
            valid_indices.append(idx)

    trajectories = np.array(trajectories)
    filtered_df = tech_df.loc[valid_indices].reset_index(drop=True)

    logger.info(f"Kept {len(filtered_df)} technologies with >= {min_reuse} reuses "
                f"(out of {len(tech_df)})")
    return filtered_df, trajectories


def z_score_normalize(trajectories: np.ndarray) -> np.ndarray:
    """Z-score normalize each trajectory independently (Eq. 1 in paper)."""
    means = trajectories.mean(axis=1, keepdims=True)
    stds = trajectories.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0  # avoid division by zero
    return (trajectories - means) / stds


# ============================================================
# DTW Distance Computation
# ============================================================

def compute_dtw_distance_matrix(trajectories: np.ndarray,
                                 window: Optional[int] = None) -> np.ndarray:
    """
    Compute pairwise DTW distance matrix.

    For large datasets, uses dtaidistance for speed.
    Falls back to a simple numpy implementation.
    """
    n = len(trajectories)
    logger.info(f"Computing DTW distance matrix for {n} trajectories...")

    if HAS_DTAIDISTANCE and n < 50000:
        # dtaidistance is C-optimized
        series = [traj.tolist() for traj in trajectories]
        dist_matrix = dtaid_dtw.distance_matrix(
            series, window=window, use_pruning=True
        )
        # Convert to full symmetric matrix
        dist_matrix = np.array(dist_matrix)
        dist_matrix = dist_matrix + dist_matrix.T
        return dist_matrix

    # Fallback: manual DTW
    dist_matrix = np.zeros((n, n))
    total_pairs = n * (n - 1) // 2
    computed = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = _dtw_distance(trajectories[i], trajectories[j], window)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            computed += 1
            if computed % 100000 == 0:
                logger.info(f"  DTW progress: {computed}/{total_pairs}")

    return dist_matrix


def _dtw_distance(x: np.ndarray, y: np.ndarray,
                   window: Optional[int] = None) -> float:
    """Compute DTW distance between two sequences (Eq. 2 in paper)."""
    n, m = len(x), len(y)
    if window is None:
        window = max(n, m)

    # Initialize cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return dtw_matrix[n, m]


# ============================================================
# Clustering
# ============================================================

def cluster_trajectories(trajectories: np.ndarray,
                          n_clusters: int = 4,
                          method: str = "dtw_kmeans",
                          normalize: bool = True,
                          n_init: int = 10,
                          max_iter: int = 300,
                          random_state: int = 42,
                          dtw_window: Optional[int] = None
                          ) -> Tuple[np.ndarray, Dict]:
    """
    Cluster technology reuse trajectories into patterns.

    Args:
        trajectories: (n_techs, window_length) cumulative reuse curves
        n_clusters: number of clusters (default 4)
        method: "dtw_kmeans" or "euclidean_kmeans"
        normalize: whether to z-score normalize first

    Returns:
        labels: cluster assignments (0 to n_clusters-1)
        info: dict with cluster centers, SSE, silhouette, etc.
    """
    logger.info(f"Clustering {len(trajectories)} trajectories with {method}, "
                f"k={n_clusters}")

    if normalize:
        norm_traj = z_score_normalize(trajectories)
    else:
        norm_traj = trajectories.copy()

    if method == "dtw_kmeans" and HAS_TSLEARN:
        return _tslearn_dtw_kmeans(norm_traj, n_clusters, n_init,
                                    max_iter, random_state)
    elif method == "dtw_kmeans":
        return _manual_dtw_kmeans(norm_traj, n_clusters, n_init,
                                   max_iter, random_state, dtw_window)
    else:
        return _euclidean_kmeans(norm_traj, n_clusters, n_init,
                                 max_iter, random_state)


def _tslearn_dtw_kmeans(trajectories: np.ndarray, n_clusters: int,
                         n_init: int, max_iter: int,
                         random_state: int) -> Tuple[np.ndarray, Dict]:
    """Use tslearn's TimeSeriesKMeans with DTW metric."""
    # tslearn expects shape (n_samples, n_timestamps, n_features)
    X = trajectories.reshape(len(trajectories), -1, 1)

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        verbose=1,
    )
    labels = model.fit_predict(X)
    centers = model.cluster_centers_.squeeze(-1)  # (k, window)
    inertia = model.inertia_

    info = {
        "centers": centers,
        "inertia": inertia,
        "method": "tslearn_dtw_kmeans",
    }
    logger.info(f"DTW k-means complete. Inertia={inertia:.2f}")
    return labels, info


def _manual_dtw_kmeans(trajectories: np.ndarray, n_clusters: int,
                        n_init: int, max_iter: int, random_state: int,
                        window: Optional[int]) -> Tuple[np.ndarray, Dict]:
    """
    Precompute DTW distance matrix, then run k-means on it.
    For large datasets, this is expensive — consider subsampling.
    """
    n = len(trajectories)
    if n > 10000:
        logger.warning(f"Manual DTW k-means on {n} samples will be very slow. "
                        "Consider using tslearn or subsampling.")

    dist_matrix = compute_dtw_distance_matrix(trajectories, window=window)

    # Use sklearn KMeans on the distance matrix as precomputed
    # We can embed using MDS or just use the distance matrix with k-medoids
    # For simplicity, use k-means on z-scored trajectories with euclidean
    # but report DTW distances for quality metrics
    km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                random_state=random_state)
    labels = km.fit_predict(trajectories)

    info = {
        "centers": km.cluster_centers_,
        "inertia": km.inertia_,
        "method": "manual_dtw_fallback_euclidean",
        "dtw_distance_matrix": dist_matrix,
    }
    return labels, info


def _euclidean_kmeans(trajectories: np.ndarray, n_clusters: int,
                       n_init: int, max_iter: int,
                       random_state: int) -> Tuple[np.ndarray, Dict]:
    """Standard k-means with Euclidean distance."""
    km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter,
                random_state=random_state)
    labels = km.fit_predict(trajectories)

    sil = silhouette_score(trajectories, labels) if len(set(labels)) > 1 else 0

    info = {
        "centers": km.cluster_centers_,
        "inertia": km.inertia_,
        "silhouette": sil,
        "method": "euclidean_kmeans",
    }
    logger.info(f"Euclidean k-means: inertia={km.inertia_:.2f}, "
                f"silhouette={sil:.4f}")
    return labels, info


def find_optimal_k(trajectories: np.ndarray, k_range: range = range(2, 11),
                    method: str = "euclidean_kmeans",
                    random_state: int = 42) -> Dict:
    """
    Evaluate SSE (inertia) across different k values for elbow method.
    """
    results = {"k": [], "sse": [], "silhouette": []}

    norm_traj = z_score_normalize(trajectories)

    for k in k_range:
        labels, info = cluster_trajectories(
            trajectories, n_clusters=k, method=method,
            normalize=True, random_state=random_state
        )
        results["k"].append(k)
        results["sse"].append(info.get("inertia", 0))
        sil = info.get("silhouette", 0)
        if sil == 0 and len(set(labels)) > 1:
            sil = silhouette_score(norm_traj, labels)
        results["silhouette"].append(sil)
        logger.info(f"k={k}: SSE={info.get('inertia', 0):.2f}, "
                     f"silhouette={sil:.4f}")

    return results


# ============================================================
# Cluster Characterization
# ============================================================

CLUSTER_NAMES = {
    0: "S-shaped",
    1: "Fleeting",
    2: "Linear",
    3: "Exponential",
}


def characterize_clusters(tech_df: pd.DataFrame, labels: np.ndarray,
                           trajectories: np.ndarray) -> pd.DataFrame:
    """
    Compute summary statistics for each cluster, similar to Table 2.
    """
    tech_df = tech_df.copy()
    tech_df["cluster"] = labels

    records = []
    for c in sorted(set(labels)):
        mask = labels == c
        traj_c = trajectories[mask]
        n = mask.sum()

        # Compute milestone times
        milestones = {}
        for pct in [0.10, 0.25, 0.50, 0.75, 0.90]:
            times = []
            for traj in traj_c:
                total = traj[-1]
                if total == 0:
                    continue
                threshold = pct * total
                # Find first year exceeding threshold
                for t, val in enumerate(traj):
                    if val >= threshold:
                        times.append(t + 1)  # 1-indexed
                        break
            milestones[f"time_{int(pct*100)}pct"] = (
                np.mean(times) if times else np.nan
            )

        records.append({
            "cluster": c,
            "cluster_name": CLUSTER_NAMES.get(c, f"Cluster_{c}"),
            "n_technologies": n,
            "mean_total_reuse": np.mean([t[-1] for t in traj_c]),
            **milestones,
        })

    return pd.DataFrame(records)
