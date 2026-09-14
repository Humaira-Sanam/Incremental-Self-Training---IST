"""
self_training.py
----------------
Conservative self-training loop for the RF point cloud classifier.

Each iteration:
1. Trains a Random Forest on the current labelled set.
2. Builds a candidate pool from the k-nearest neighbours of labelled points.
3. Gates candidates through three filters (probability, Mahalanobis, consensus).
4. Adds only those that pass all three filters — provided enough new points
   would be added to warrant continuing.

Stopping criteria
-----------------
- No candidates remain in the pool.
- All candidates fail at least one filter.
- The number of newly labelled points falls below ``min_allowed`` (derived
  from the larger of ``min_new_absolute`` and a capped fraction of the
  remaining unlabelled set).  This prevents label drift in late iterations
  when the pool is exhausted and the classifier is extrapolating.
- No net increase in labelled points after an iteration.
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KDTree


# --------------------------------------------------------------------------- #
# Mahalanobis helpers                                                          #
# --------------------------------------------------------------------------- #

def _regularized_cov_inv(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    cov = np.cov(X.T) + np.eye(X.shape[1]) * reg
    return np.linalg.inv(cov)


def _mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    d = x - mean
    return np.sum((d @ cov_inv) * d, axis=1)


# --------------------------------------------------------------------------- #
# Self-training                                                                #
# --------------------------------------------------------------------------- #

def self_training_conservative(
    features_df: pd.DataFrame,
    cloud_xyz: np.ndarray,
    seed_indices: np.ndarray,
    seed_labels: np.ndarray,
    *,
    pool_k: int = 50,
    threshold: float = 0.92,
    chi2_conf: float = 0.90,
    max_iters: int = 20,
    k_consensus: int = 10,
    consensus_frac: float = 0.6,
    min_new_absolute: int = 50,
    min_new_fraction: float = 0.001,
    min_new_cap: int = 1000,
    n_estimators_iter: int = 200,
    n_estimators_final: int = 500,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """
    Run the conservative self-training loop and return the final classifier.

    Parameters
    ----------
    features_df:
        Scaled feature matrix (N × F).
    cloud_xyz:
        (N, 3) raw point coordinates, used for spatial operations.
    seed_indices, seed_labels:
        Initial labelled set (indices into the N-point arrays).
    pool_k:
        Neighbourhood size for building the candidate pool each iteration.
    threshold:
        Minimum predicted probability for a candidate to pass the RF gate.
    chi2_conf:
        Confidence level for the Mahalanobis χ² gate.
    max_iters:
        Hard cap on the number of self-training iterations.
    k_consensus:
        Neighbours used for the spatial consensus check.
    consensus_frac:
        Minimum fraction of neighbours that must agree on a label.
    min_new_absolute:
        Absolute floor on the number of new points required per iteration.
    min_new_fraction:
        Fraction of the unlabelled pool that sets a dynamic minimum.
    min_new_cap:
        Cap applied to the fraction-based dynamic minimum.
    n_estimators_iter:
        Trees in the RF trained at each iteration.
    n_estimators_final:
        Trees in the final RF trained after the loop completes.
    n_jobs:
        Parallel jobs for the final RF (-1 = all cores).
    random_state:
        Random seed.
    verbose:
        Print per-iteration diagnostics.

    Returns
    -------
    final_clf : RandomForestClassifier
        Final fitted classifier.
    final_labels : np.ndarray
        Shape (N,) — class label for every point (-1 for unlabelled).
    labeled_mask : np.ndarray
        Shape (N,) bool — True for every point that received a label.
    """
    X = features_df.values
    n = X.shape[0]

    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[seed_indices] = True
    labels = np.full(n, -1, dtype=int)
    labels[seed_indices] = seed_labels.astype(int)

    tree_all   = KDTree(cloud_xyz)
    prev_count = int(labeled_mask.sum())
    chi2_cutoff = chi2.ppf(chi2_conf, df=X.shape[1])

    for it in range(1, max_iters + 1):
        t0 = time.time()

        # ------------------------------------------------------------------ #
        # 1. Train iteration RF                                               #
        # ------------------------------------------------------------------ #
        clf = RandomForestClassifier(
            n_estimators=n_estimators_iter,
            n_jobs=1,
            random_state=random_state,
        )
        clf.fit(X[labeled_mask], labels[labeled_mask])
        if verbose:
            print(f"[{it}] RF trained on {labeled_mask.sum()} labels in {time.time()-t0:.1f}s")

        # ------------------------------------------------------------------ #
        # 2. Build candidate pool (unlabelled neighbours of labelled points)  #
        # ------------------------------------------------------------------ #
        labeled_idx = np.where(labeled_mask)[0]
        candidate_set: set = set()
        batch_knn = 5_000
        for s in range(0, labeled_idx.size, batch_knn):
            batch = labeled_idx[s : s + batch_knn]
            _, neighs = tree_all.query(cloud_xyz[batch], k=min(pool_k + 1, n))
            for arr in neighs:
                candidate_set.update(arr.tolist())

        candidates = np.array(sorted(candidate_set), dtype=int)
        candidates = candidates[~labeled_mask[candidates]]
        if verbose:
            print(f"[{it}] Candidate pool size: {candidates.size}")
        if candidates.size == 0:
            if verbose:
                print("No candidates → stopping.")
            break

        # ------------------------------------------------------------------ #
        # 3. Gate 1 & 2: probability + Mahalanobis                           #
        # ------------------------------------------------------------------ #
        probs_parts, cls_order = [], None
        for s in range(0, candidates.size, 200_000):
            part = clf.predict_proba(X[candidates[s : s + 200_000]])
            probs_parts.append(part)
            if cls_order is None:
                cls_order = clf.classes_
        probs_all  = np.vstack(probs_parts)
        pred_cand  = cls_order[np.argmax(probs_all, axis=1)]
        max_prob   = np.max(probs_all, axis=1)

        to_add_mask = np.zeros(candidates.size, dtype=bool)
        for cls in np.unique(labels[labeled_mask]):
            cls_mask = labels == cls
            X_cls    = X[cls_mask]
            if X_cls.shape[0] < 5:
                cond = (pred_cand == cls) & (max_prob >= threshold)
            else:
                mean    = X_cls.mean(axis=0)
                cov_inv = _regularized_cov_inv(X_cls)
                dists   = _mahalanobis_distance(X[candidates], mean, cov_inv)
                cond    = (pred_cand == cls) & (max_prob >= threshold) & (dists <= chi2_cutoff)
            to_add_mask |= cond

        shortlist = candidates[to_add_mask]
        if verbose:
            print(f"[{it}] Shortlist after prob + Mahalanobis: {shortlist.size}")
        if shortlist.size == 0:
            if verbose:
                print("Empty shortlist → stopping.")
            break

        # ------------------------------------------------------------------ #
        # 4. Gate 3: spatial consensus                                        #
        # ------------------------------------------------------------------ #
        final_add = []
        for ci in shortlist:
            kq = min(k_consensus, int(labeled_mask.sum()))
            if kq == 0:
                continue
            _, locs = tree_all.query([cloud_xyz[ci]], k=kq)
            neigh_labels = labels[locs[0]]
            valid        = neigh_labels[neigh_labels >= 0]
            if valid.size == 0:
                continue
            pred_label  = int(clf.predict(X[ci : ci + 1])[0])
            frac_same   = np.sum(valid == pred_label) / valid.size
            if frac_same >= consensus_frac:
                final_add.append(ci)

        final_add    = np.array(final_add, dtype=int)
        new_additions = final_add.size

        # ------------------------------------------------------------------ #
        # 5. Stopping check                                                   #
        # ------------------------------------------------------------------ #
        unlabeled_count = n - int(labeled_mask.sum())
        min_from_frac   = math.ceil(min_new_fraction * unlabeled_count)
        min_allowed     = max(min_new_absolute, min(min_new_cap, min_from_frac))

        if verbose:
            print(f"[{it}] final_add={new_additions}, min_allowed={min_allowed}")

        if new_additions == 0:
            if verbose:
                print("No additions → stopping.")
            break
        if new_additions < min_allowed:
            if verbose:
                print("Additions below minimum → stopping to prevent drift.")
            break

        labels[final_add]       = clf.predict(X[final_add])
        labeled_mask[final_add] = True
        if verbose:
            print(f"[{it}] Added {new_additions}; total labelled {labeled_mask.sum()}")

        if int(labeled_mask.sum()) == prev_count:
            if verbose:
                print("No net increase → stopping.")
            break
        prev_count = int(labeled_mask.sum())

    # ---------------------------------------------------------------------- #
    # Final RF on the full expanded training set                              #
    # ---------------------------------------------------------------------- #
    final_clf = RandomForestClassifier(
        n_estimators=n_estimators_final,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    final_clf.fit(X[labeled_mask], labels[labeled_mask])

    # Guarantee labels are in {1, 2}
    final_labels = labels.astype(int)
    unique_final = np.unique(final_labels[final_labels >= 0])
    if set(unique_final) != {1, 2}:
        mapping = {lab: i + 1 for i, lab in enumerate(sorted(unique_final))}
        print("Remapping final classes to {1, 2}:", mapping)
        for k_val, v_val in mapping.items():
            final_labels[final_labels == k_val] = v_val

    return final_clf, final_labels, labeled_mask
