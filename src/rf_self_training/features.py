"""
features.py
-----------
Multi-scale PCA geometric feature extraction and RGB-derived colour features.

Each point is described at every requested radius by:
  - PCA eigenvalues and derived shape descriptors
    (anisotropy, planarity, linearity, omnivariance, sphericity,
     surface variation, eigenentropy)
  - Verticality (alignment of the minor eigenvector with the Z axis)
  - Roughness (std-dev of residuals from the local tangent plane)
  - Curvature (ratio of smallest eigenvalue to eigenvalue sum)
  - rel_mean_z = local mean Z − local ground Z  (replaces raw mean Z)
  - std_z, skew_z, first_order_moment

All column names are prefixed with the radius string, e.g. ``r0p1_anisotropy``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.neighbors import KDTree
from tqdm import tqdm
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# PCA helpers                                                                  #
# --------------------------------------------------------------------------- #

def _pca_eigvals_and_vectors(points: np.ndarray):
    """Return eigenvalues (descending) and eigenvectors of the covariance matrix."""
    if points.shape[0] < 3:
        return np.zeros(3), np.eye(3)
    cov = np.cov(points.T)
    w, v = eigh(cov)          # eigenvalues in ascending order
    w = np.clip(w, 0, None)
    order = np.argsort(w)[::-1]
    return w[order], v[:, order]


# --------------------------------------------------------------------------- #
# Per-point feature computation                                                #
# --------------------------------------------------------------------------- #

def compute_local_features(
    i: int,
    xyz: np.ndarray,
    tree: KDTree,
    radius: float,
    min_neighbors: int = 8,
) -> Optional[Dict[str, float]]:
    """
    Compute geometric features for point ``i`` within ``radius``.

    Returns ``None`` when fewer than ``min_neighbors`` neighbours are found.
    """
    idx = tree.query_radius(xyz[i : i + 1], r=radius)[0]
    if idx.size < min_neighbors:
        return None

    neigh = xyz[idx]
    eigvals, eigvecs = _pca_eigvals_and_vectors(neigh)
    lam1, lam2, lam3 = eigvals
    s = lam1 + lam2 + lam3 if (lam1 + lam2 + lam3) > 0 else 1e-12

    # Normalised eigenvalue probabilities (for entropy)
    p = np.where(eigvals / s <= 0, 1e-12, eigvals / s)
    eigenentropy = float(-np.sum(p * np.log(p)))

    anisotropy     = float((lam1 - lam3) / lam1) if lam1 > 0 else 0.0
    planarity      = float((lam2 - lam3) / lam1) if lam1 > 0 else 0.0
    linearity      = float((lam1 - lam2) / lam1) if lam1 > 0 else 0.0
    omnivariance   = float((lam1 * lam2 * lam3) ** (1.0 / 3.0)) if (lam1 * lam2 * lam3) > 0 else 0.0
    sphericity     = float(lam3 / lam1) if lam1 > 0 else 0.0
    surface_variation = float(lam3 / s)

    normal         = eigvecs[:, 2]
    verticality    = float(abs(normal[2]))

    centroid       = neigh.mean(axis=0)
    residuals      = (neigh - centroid) @ normal
    roughness      = float(np.std(residuals))
    curvature      = float(lam3 / s)

    z              = neigh[:, 2]
    mean_z         = float(z.mean())
    std_z          = float(z.std())
    skew_z         = float(np.mean(((z - mean_z) / std_z) ** 3)) if std_z > 0 else 0.0
    first_order_moment = float(np.mean(np.linalg.norm(neigh - centroid, axis=1)))

    return {
        "lam1": float(lam1),
        "lam2": float(lam2),
        "lam3": float(lam3),
        "sum_eigs": float(s),
        "eigenentropy": eigenentropy,
        "anisotropy": anisotropy,
        "planarity": planarity,
        "linearity": linearity,
        "omnivariance": omnivariance,
        "sphericity": sphericity,
        "surface_variation": surface_variation,
        "verticality": verticality,
        "roughness": roughness,
        "curvature": curvature,
        "mean_z": mean_z,        # will be converted to rel_mean_z in build_feature_matrix
        "std_z": std_z,
        "skew_z": skew_z,
        "first_order_moment": first_order_moment,
    }


# --------------------------------------------------------------------------- #
# Full feature matrix construction                                             #
# --------------------------------------------------------------------------- #

def build_feature_matrix(
    cloud: pd.DataFrame,
    radii: List[float],
    ground_z: np.ndarray,
    min_neighbors: int = 8,
) -> pd.DataFrame:
    """
    Build the multi-scale feature matrix for all points in ``cloud``.

    Parameters
    ----------
    cloud:
        Point cloud DataFrame with columns ``x``, ``y``, ``z``, ``r``, ``g``, ``b``.
    radii:
        List of neighbourhood radii for feature extraction.
    ground_z:
        Per-point local ground elevation (see ``ground.estimate_ground_z``).
    min_neighbors:
        Points with fewer neighbours than this at a given radius receive NaN features.

    Returns
    -------
    pd.DataFrame
        Feature matrix (one row per point).  Absolute ``mean_z`` columns are
        replaced by ``rel_mean_z = mean_z − ground_z``.
    """
    xyz  = cloud[["x", "y", "z"]].values
    n    = xyz.shape[0]
    tree = KDTree(xyz)

    parts: List[pd.DataFrame] = []
    for r in radii:
        print(f"  Extracting features at radius {r} …")
        feats = []
        for i in tqdm(range(n), leave=False):
            f = compute_local_features(i, xyz, tree, r, min_neighbors=min_neighbors)
            if f is None:
                feats.append({})
            else:
                f["rel_mean_z"] = f.pop("mean_z") - ground_z[i]
                feats.append(f)

        prefix = f"r{str(r).replace('.', 'p')}_"
        parts.append(pd.DataFrame(feats).add_prefix(prefix))

    feats_all = pd.concat(parts, axis=1).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # RGB-derived colour features                                          #
    # ------------------------------------------------------------------ #
    rgb     = cloud[["r", "g", "b"]].astype(float).reset_index(drop=True)
    rgb_sum = (rgb["r"] + rgb["g"] + rgb["b"]).replace(0, 1e-6)

    feats_all["g_norm"]   = rgb["g"] / rgb_sum
    feats_all["r_norm"]   = rgb["r"] / rgb_sum
    feats_all["b_norm"]   = rgb["b"] / rgb_sum
    feats_all["rgb_mean"] = rgb_sum / 3.0
    feats_all["rg_ratio"] = rgb["r"] / (rgb["g"] + 1e-6)
    feats_all["ground_z"] = ground_z

    # Drop any residual absolute mean_z columns (safety net)
    abs_mean_cols = [
        c for c in feats_all.columns
        if c.endswith("mean_z") and not c.endswith("rel_mean_z")
    ]
    if abs_mean_cols:
        feats_all.drop(columns=abs_mean_cols, inplace=True)

    print(f"  Feature matrix shape: {feats_all.shape}")
    return feats_all
