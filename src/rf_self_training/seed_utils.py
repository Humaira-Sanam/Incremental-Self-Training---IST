"""
seed_utils.py
-------------
Utilities for mapping labelled training points onto the full point cloud,
cleaning noisy seeds via spatial label consensus, and performing a
spatially-blocked train / test split.

Spatial block holdout (why it matters)
---------------------------------------
A naive random 80/20 split of seed points suffers from spatial
autocorrelation: in a dense 3D point cloud, test points end up immediately
adjacent to training points, so the classifier is essentially interpolating
rather than generalising.  This inflates reported metrics (the near-perfect
F1 = 0.99 observed with random splitting is a direct consequence).

The fix is a *spatial block holdout*: the XY extent of the seed set is
divided into a regular grid of blocks, and whole blocks are reserved for
testing.  Because the test block(s) are spatially separated from every
training point by at least one block width, the evaluation reflects genuine
generalisation across space rather than local interpolation.

Two splitting strategies are provided:

spatial_block_seed_split  (recommended, default)
    Divides the seed footprint into an n_blocks × n_blocks grid.
    The block(s) whose combined area is closest to ``test_size`` are held
    out.  All points inside those blocks go to test; all others go to train.
    The split boundary is saved to a CSV for reporting in the paper.

stratified_seed_split  (legacy, kept for backward compatibility)
    Original random stratified split — produces autocorrelation-inflated
    metrics.  Use only when seed density is too low for block splitting.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Mapping                                                                      #
# --------------------------------------------------------------------------- #

def map_train_to_cloud(
    cloud: pd.DataFrame,
    train_csv: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap each labelled training point to its nearest cloud point.

    When multiple training points map to the same cloud index the first label
    (by input order) is kept.

    Parameters
    ----------
    cloud:
        Full point cloud with columns ``x``, ``y``, ``z``.
    train_csv:
        Labelled seed points with columns ``x``, ``y``, ``z``, ``class``.

    Returns
    -------
    agg_indices : np.ndarray
        Cloud indices of the mapped seed points (one per unique cloud point).
    agg_labels : np.ndarray
        Integer class labels remapped to {1, 2}.
    """
    tree = KDTree(cloud[["x", "y", "z"]].values)
    _, idxs = tree.query(train_csv[["x", "y", "z"]].values, k=1)
    mapped_idx    = idxs.flatten()
    mapped_labels = train_csv["class"].values.astype(int)

    agg = (
        pd.DataFrame({"cloud_idx": mapped_idx, "label": mapped_labels})
        .groupby("cloud_idx", as_index=False)
        .first()
    )
    agg_indices = agg["cloud_idx"].values
    agg_labels  = agg["label"].values

    # Remap to {1, 2}
    unique_labels = np.unique(agg_labels)
    if set(unique_labels) != {1, 2}:
        mapping = {lab: i + 1 for i, lab in enumerate(sorted(unique_labels))}
        agg_labels = np.array([mapping[l] for l in agg_labels])
        print("Remapped labels:", mapping)

    print(f"Mapped {agg_indices.size} unique labelled cloud points.")
    return agg_indices, agg_labels


# --------------------------------------------------------------------------- #
# Seed cleaning                                                                #
# --------------------------------------------------------------------------- #

def clean_seed_set(
    agg_indices: np.ndarray,
    agg_labels: np.ndarray,
    cloud_xyz: np.ndarray,
    k: int = 10,
    frac: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove noisy seed points whose label disagrees with their neighbours.

    A seed point is *kept* only when at least ``frac`` of its ``k`` nearest
    labelled neighbours share the same class label.

    Parameters
    ----------
    agg_indices : np.ndarray
        Indices into ``cloud_xyz`` of the seed points.
    agg_labels : np.ndarray
        Class labels for the seed points.
    cloud_xyz : np.ndarray
        (N, 3) full point cloud coordinates.
    k : int
        Number of nearest labelled neighbours to examine.
    frac : float
        Required fraction of agreeing neighbours [0, 1].

    Returns
    -------
    keep_idx : np.ndarray
        Filtered seed indices.
    keep_lab : np.ndarray
        Corresponding class labels.
    """
    if agg_indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    seed_xyz = cloud_xyz[agg_indices]
    agg_tree = KDTree(seed_xyz)

    keep_idx, keep_lab = [], []
    for i, idx in enumerate(agg_indices):
        kq = min(k, agg_indices.size)
        _, locs = agg_tree.query([cloud_xyz[idx]], k=kq)
        neighbor_indices = agg_indices[locs[0]]

        neighbor_labels = []
        for ni in neighbor_indices:
            pos = np.where(agg_indices == ni)[0]
            if pos.size > 0:
                neighbor_labels.append(agg_labels[pos[0]])

        if not neighbor_labels:
            continue

        same      = sum(1 for v in neighbor_labels if v == agg_labels[i])
        frac_same = same / len(neighbor_labels)
        if frac_same >= frac:
            keep_idx.append(idx)
            keep_lab.append(agg_labels[i])

    return np.array(keep_idx, dtype=int), np.array(keep_lab, dtype=int)


# --------------------------------------------------------------------------- #
# Spatial block holdout split  (primary, addresses autocorrelation)           #
# --------------------------------------------------------------------------- #

def spatial_block_seed_split(
    cleaned_indices: np.ndarray,
    cleaned_labels: np.ndarray,
    cloud_xyz: np.ndarray,
    n_blocks: int = 4,
    test_size: float = 0.20,
    random_state: int = 42,
    output_dir: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split seed points into spatially separated train and test sets.

    The XY footprint of the seed points is divided into an
    ``n_blocks × n_blocks`` regular grid.  Blocks are sorted by their
    distance from the plot centroid and the outermost block(s) whose
    combined point count is closest to ``test_size`` fraction are reserved
    for the test set.  All remaining blocks form the training set.

    This ensures that every test point is at least one block-width away
    from every training point, eliminating the spatial autocorrelation
    that inflates metrics in a random split.

    Parameters
    ----------
    cleaned_indices : np.ndarray
        Cleaned seed indices into ``cloud_xyz``.
    cleaned_labels : np.ndarray
        Class labels for the cleaned seeds.
    cloud_xyz : np.ndarray
        (N, 3) full point cloud coordinates (used to recover XY positions).
    n_blocks : int
        Number of grid divisions along each axis (default 4 → 16 blocks).
        Increase if the seed set is very large and spatially dense.
    test_size : float
        Target fraction of seed points to hold out for testing [0, 1].
    random_state : int
        Seed used only as a tiebreaker when two block selections are
        equally close to ``test_size``.
    output_dir : str or None
        If provided, saves ``spatial_split_map.csv`` describing block
        assignments — useful for reporting in the paper.

    Returns
    -------
    seed_indices, seed_labels, test_indices, test_labels : np.ndarray

    Notes
    -----
    Falls back to :func:`stratified_seed_split` with a warning when fewer
    than 4 blocks contain labelled points (e.g. a very sparse seed set).
    """
    seed_xy = cloud_xyz[cleaned_indices, :2]   # (M, 2)
    n_seeds = cleaned_indices.size

    x_min, x_max = seed_xy[:, 0].min(), seed_xy[:, 0].max()
    y_min, y_max = seed_xy[:, 1].min(), seed_xy[:, 1].max()

    # Guard against degenerate extents (all seeds in a line / point)
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span == 0 or y_span == 0:
        print("WARNING: seed set has zero spatial extent on one axis; "
              "falling back to random stratified split.")
        return stratified_seed_split(
            cleaned_indices, cleaned_labels,
            test_size=test_size, random_state=random_state,
        )

    # Assign each seed to a grid cell (row, col)
    col = np.floor(
        (seed_xy[:, 0] - x_min) / (x_span + 1e-9) * n_blocks
    ).astype(int).clip(0, n_blocks - 1)
    row = np.floor(
        (seed_xy[:, 1] - y_min) / (y_span + 1e-9) * n_blocks
    ).astype(int).clip(0, n_blocks - 1)
    block_id = row * n_blocks + col          # integer block identifier

    unique_blocks = np.unique(block_id)
    n_occupied    = unique_blocks.size

    if n_occupied < 4:
        print(f"WARNING: only {n_occupied} spatial blocks contain seeds; "
              "falling back to random stratified split to avoid empty splits.")
        return stratified_seed_split(
            cleaned_indices, cleaned_labels,
            test_size=test_size, random_state=random_state,
        )

    # Count seeds per block
    block_counts = {b: int(np.sum(block_id == b)) for b in unique_blocks}

    # Compute centroid of each occupied block (in normalised grid coords)
    # and sort by distance from the plot centre so that corner/edge blocks
    # are preferred for the test holdout (maximising spatial separation).
    cx = n_blocks / 2.0
    cy = n_blocks / 2.0
    block_meta = []
    for b in unique_blocks:
        r_b = b // n_blocks
        c_b = b %  n_blocks
        dist_from_centre = np.sqrt((r_b + 0.5 - cy) ** 2 + (c_b + 0.5 - cx) ** 2)
        block_meta.append((b, block_counts[b], dist_from_centre))
    # Sort farthest-first; tiebreak by block id (deterministic)
    block_meta.sort(key=lambda t: (-t[2], t[0]))

    # Greedily select blocks for the test set until we are closest to test_size
    target_n_test = test_size * n_seeds
    selected_test_blocks = []
    running_n = 0
    for b, cnt, _ in block_meta:
        if abs(running_n + cnt - target_n_test) <= abs(running_n - target_n_test):
            selected_test_blocks.append(b)
            running_n += cnt
        if running_n >= target_n_test:
            break

    test_block_set = set(selected_test_blocks)
    test_mask  = np.array([b in test_block_set for b in block_id])
    train_mask = ~test_mask

    # Safety: ensure both classes appear in each split
    for split_name, mask in [("train", train_mask), ("test", test_mask)]:
        present = np.unique(cleaned_labels[mask])
        if present.size < 2:
            print(f"WARNING: spatial block split leaves only class(es) {present} "
                  f"in the {split_name} set; falling back to random stratified split.")
            return stratified_seed_split(
                cleaned_indices, cleaned_labels,
                test_size=test_size, random_state=random_state,
            )

    seed_indices  = cleaned_indices[train_mask]
    seed_labels   = cleaned_labels[train_mask]
    test_indices  = cleaned_indices[test_mask]
    test_labels   = cleaned_labels[test_mask]

    actual_test_frac = test_mask.sum() / n_seeds
    print(
        f"Spatial block split: {n_blocks}×{n_blocks} grid, "
        f"{len(selected_test_blocks)} test block(s), "
        f"train={seed_indices.size}, test={test_indices.size} "
        f"({actual_test_frac:.1%} of seeds)."
    )

    # Optional: save block map for paper reporting
    if output_dir is not None:
        split_map = pd.DataFrame({
            "cloud_idx":  cleaned_indices,
            "label":      cleaned_labels,
            "x":          cloud_xyz[cleaned_indices, 0],
            "y":          cloud_xyz[cleaned_indices, 1],
            "z":          cloud_xyz[cleaned_indices, 2],
            "block_row":  block_id // n_blocks,
            "block_col":  block_id %  n_blocks,
            "block_id":   block_id,
            "split":      np.where(test_mask, "test", "train"),
        })
        out_path = os.path.join(output_dir, "spatial_split_map.csv")
        split_map.to_csv(out_path, index=False)
        print(f"Saved spatial split map → {out_path}")

    return seed_indices, seed_labels, test_indices, test_labels


# --------------------------------------------------------------------------- #
# Legacy random split  (kept for backward compatibility / sparse seed sets)   #
# --------------------------------------------------------------------------- #

def stratified_seed_split(
    cleaned_indices: np.ndarray,
    cleaned_labels: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split cleaned seeds into a training set and a held-out test set using
    random stratified sampling.

    .. warning::
        This split is subject to spatial autocorrelation: test points are
        likely adjacent to training points, inflating reported metrics.
        Prefer :func:`spatial_block_seed_split` for publication-quality
        evaluation.

    Parameters
    ----------
    cleaned_indices, cleaned_labels:
        Cleaned seed arrays from :func:`clean_seed_set`.
    test_size:
        Fraction of seeds reserved for evaluation.
    random_state:
        Reproducibility seed.

    Returns
    -------
    seed_indices, seed_labels, test_indices, test_labels : np.ndarray
    """
    rel_train, rel_test = train_test_split(
        np.arange(cleaned_indices.size),
        stratify=cleaned_labels,
        test_size=test_size,
        random_state=random_state,
    )
    return (
        cleaned_indices[rel_train],
        cleaned_labels[rel_train],
        cleaned_indices[rel_test],
        cleaned_labels[rel_test],
    )
