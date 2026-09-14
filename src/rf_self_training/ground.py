"""
ground.py
---------
Local ground-elevation estimation from a point cloud.

For each point the ground Z is defined as the ``ground_percentile``-th
percentile of Z values within a sphere of radius ``ground_radius``.  This
avoids the need for a separate DTM or ground-classification pass and works
well for forest / urban scenes where genuine ground points are present.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm


def estimate_ground_z(
    xyz: np.ndarray,
    ground_radius: float = 1.0,
    ground_percentile: int = 5,
) -> np.ndarray:
    """
    Estimate per-point local ground elevation.

    Parameters
    ----------
    xyz:
        (N, 3) array of point coordinates.
    ground_radius:
        Search radius (same units as ``xyz``) used to gather neighbours
        for the local percentile calculation.
    ground_percentile:
        Percentile of Z values within the neighbourhood that represents
        the local ground level.  5 works well for dense urban/forest scans.

    Returns
    -------
    np.ndarray
        Shape (N,) array of per-point ground-Z estimates.
    """
    n      = xyz.shape[0]
    tree   = KDTree(xyz)
    global_min_z = float(xyz[:, 2].min())
    ground_z     = np.empty(n, dtype=float)

    for i in tqdm(range(n), desc="Estimating ground Z"):
        idx = tree.query_radius(xyz[i : i + 1], r=ground_radius)[0]
        if idx.size == 0:
            ground_z[i] = global_min_z
        else:
            ground_z[i] = float(np.percentile(xyz[idx, 2], ground_percentile))

    return ground_z
