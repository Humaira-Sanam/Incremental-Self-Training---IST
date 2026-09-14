"""
run_pipeline.py
---------------
End-to-end entry point for the RF self-training point cloud classifier.

Steps
-----
1.  Load the full cloud and labelled training CSV.
2.  Map training labels onto the nearest cloud points and clean noisy seeds.
3.  Split seeds into train / test using a **spatial block holdout**
    (eliminates the spatial autocorrelation that inflates metrics in a
    random split — see seed_utils.py for a full explanation).
4.  Estimate per-point local ground elevation (z_rel computation).
5.  Extract multi-scale PCA + colour features.
6.  Impute missing values and standardise.
7.  Run the conservative self-training loop.
8.  Predict final labels for the whole cloud.
9.  Save all outputs (CSV, models, plots, diagnostics).

Configure paths and hyperparameters in
``src/rf_self_training/config.py`` before running.
"""

# %% ── Imports ───────────────────────────────────────────────────────────── #
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from rf_self_training.config import Config
from rf_self_training.ground import estimate_ground_z
from rf_self_training.features import build_feature_matrix
from rf_self_training.seed_utils import (
    map_train_to_cloud,
    clean_seed_set,
    spatial_block_seed_split,
    stratified_seed_split,
)
from rf_self_training.self_training import self_training_conservative
from rf_self_training.analysis import (
    save_metrics,
    save_feature_importances,
    compute_feature_class_summary,
    compute_and_plot_cohens_d,
    save_shap_plots,
)

warnings.simplefilter("ignore", UserWarning)

# %% ── Configuration ─────────────────────────────────────────────────────── #
cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
start_all = time.time()

# %% ── 1. Load data ──────────────────────────────────────────────────────── #
print("=" * 60)
print("Step 1 — Loading data")
print("=" * 60)

COLS = ["x", "y", "z", "r", "g", "b", "class"]
cloud     = pd.read_csv(cfg.cloud_path,  header=None, names=COLS, engine="python")
train_csv = pd.read_csv(cfg.train_path,  header=None, names=COLS, engine="python")
print(f"Cloud: {cloud.shape}   Train: {train_csv.shape}")

# %% ── 2. Seed mapping and cleaning ─────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 2 — Seed mapping and cleaning")
print("=" * 60)

agg_indices, agg_labels = map_train_to_cloud(cloud, train_csv)

cloud_xyz = cloud[["x", "y", "z"]].values
cleaned_indices, cleaned_labels = clean_seed_set(
    agg_indices, agg_labels, cloud_xyz,
    k=cfg.seed_consensus_k,
    frac=cfg.seed_consensus_frac,
)
print(f"Seeds: {agg_indices.size} → {cleaned_indices.size} after cleaning")

if cleaned_indices.size < 10:
    print("Cleaned set too small; reverting to original seeds.")
    cleaned_indices = agg_indices
    cleaned_labels  = agg_labels

# %% ── 3. Train / test split ─────────────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 3 — Train / test split")
print("=" * 60)

if cfg.use_spatial_block_split:
    print(
        f"Using spatial block holdout  "
        f"(grid={cfg.spatial_block_n}×{cfg.spatial_block_n}, "
        f"target test fraction={cfg.test_size:.0%})."
    )
    print(
        "  → Test points are spatially separated from all training points,\n"
        "    removing the autocorrelation that inflates metrics in a random split."
    )
    seed_indices, seed_labels, test_indices, test_labels = spatial_block_seed_split(
        cleaned_indices,
        cleaned_labels,
        cloud_xyz,
        n_blocks=cfg.spatial_block_n,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        output_dir=cfg.output_dir,      # saves spatial_split_map.csv
    )
else:
    print(
        "WARNING: using legacy random stratified split.\n"
        "  Metrics will be inflated by spatial autocorrelation.\n"
        "  Set cfg.use_spatial_block_split = True for publication-quality evaluation."
    )
    seed_indices, seed_labels, test_indices, test_labels = stratified_seed_split(
        cleaned_indices, cleaned_labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

print(f"Seed (train): {seed_indices.size}   Held-out test: {test_indices.size}")

# %% ── 4. Local ground estimation ───────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 4 — Estimating local ground Z")
print("=" * 60)

ground_z = estimate_ground_z(
    cloud_xyz,
    ground_radius=cfg.ground_radius,
    ground_percentile=cfg.ground_percentile,
)

# %% ── 5. Feature extraction ─────────────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 5 — Extracting multi-scale features")
print("=" * 60)

feats_raw = build_feature_matrix(
    cloud,
    radii=cfg.radii,
    ground_z=ground_z,
    min_neighbors=cfg.min_neighbors,
)

# %% ── 6. Imputation and scaling ─────────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 6 — Imputing and scaling")
print("=" * 60)

imputer       = SimpleImputer(strategy="mean")
feats_imputed = pd.DataFrame(
    imputer.fit_transform(feats_raw), columns=feats_raw.columns
)
scaler        = StandardScaler()
feats_scaled  = pd.DataFrame(
    scaler.fit_transform(feats_imputed), columns=feats_imputed.columns
)
print(f"Feature matrix: {feats_imputed.shape}")

# Save feature matrix (sampled for large clouds)
if feats_imputed.shape[0] > cfg.BIG_N:
    feats_imputed.sample(200_000, random_state=42).to_csv(
        os.path.join(cfg.output_dir, "features_imputed_sample.csv"), index=False
    )
    print("Saved features_imputed_sample.csv (200 k sample)")
else:
    feats_imputed.to_csv(
        os.path.join(cfg.output_dir, "features_imputed.csv"), index=False
    )
    print("Saved features_imputed.csv")

# %% ── 7. Self-training ──────────────────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 7 — Self-training")
print("=" * 60)

final_clf, expanded_labels, labeled_mask = self_training_conservative(
    feats_scaled,
    cloud_xyz,
    seed_indices,
    seed_labels,
    pool_k=50,
    threshold=cfg.threshold,
    chi2_conf=cfg.chi2_confidence,
    max_iters=cfg.max_iters,
    k_consensus=cfg.k_consensus,
    consensus_frac=cfg.consensus_frac,
    min_new_absolute=cfg.min_new_absolute,
    min_new_fraction=cfg.min_new_fraction,
    min_new_cap=cfg.min_new_cap,
    n_estimators_iter=cfg.n_estimators_iter,
    n_estimators_final=cfg.n_estimators_final,
    n_jobs=cfg.n_jobs,
    random_state=cfg.random_state,
    verbose=True,
)

# %% ── 8. Final prediction ───────────────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 8 — Predicting labels for the full cloud")
print("=" * 60)

pred_probs_parts = []
for s in range(0, feats_scaled.shape[0], 200_000):
    pred_probs_parts.append(
        final_clf.predict_proba(feats_scaled.values[s : s + 200_000])
    )
pred_probs  = np.vstack(pred_probs_parts)
pred_labels = final_clf.classes_[np.argmax(pred_probs, axis=1)].astype(int)

# Ensure {1, 2}
unique_pred = np.unique(pred_labels)
if set(unique_pred) != {1, 2}:
    mapping = {lab: i + 1 for i, lab in enumerate(sorted(unique_pred))}
    print("Remapping predicted classes to {1, 2}:", mapping)
    for k, v in mapping.items():
        pred_labels[pred_labels == k] = v

# %% ── 9. Save outputs ───────────────────────────────────────────────────── #
print("\n" + "=" * 60)
print("Step 9 — Saving outputs")
print("=" * 60)

# Classified cloud
classified_df          = cloud[["x", "y", "z", "r", "g", "b"]].copy().reset_index(drop=True)
classified_df["class"] = pred_labels
out_classified         = os.path.join(cfg.output_dir, "classified_output.csv")
if cfg.save_classified_with_header:
    classified_df.to_csv(out_classified, index=False)
else:
    classified_df.to_csv(out_classified, index=False, header=False)
print("Saved classified_output.csv")

# Expanded training data
final_training_df = cloud.loc[labeled_mask].copy().reset_index(drop=True)
final_training_df["label"] = expanded_labels[labeled_mask].astype(int)
final_training_df.to_csv(
    os.path.join(cfg.output_dir, "final_training_data.csv"), index=False
)
print("Saved final_training_data.csv")

# Features + predictions
features_with_preds = feats_imputed.copy()
features_with_preds["pred_label"] = pred_labels
for i, cls in enumerate(final_clf.classes_):
    features_with_preds[f"prob_class_{cls}"] = pred_probs[:, i]

if feats_imputed.shape[0] > cfg.BIG_N:
    features_with_preds.sample(200_000, random_state=42).to_csv(
        os.path.join(cfg.output_dir, "features_with_predictions_sample.csv"), index=False
    )
    print("Saved features_with_predictions_sample.csv (200 k sample)")
else:
    features_with_preds.to_csv(
        os.path.join(cfg.output_dir, "features_with_predictions.csv"), index=False
    )
    print("Saved features_with_predictions.csv")

# Metrics  (evaluated on the spatially held-out test block)
split_method = "spatial block holdout" if cfg.use_spatial_block_split else "random stratified split"
print(f"\nMetrics are computed on the held-out test set ({split_method}).")
save_metrics(test_labels, pred_labels[test_indices], cfg.output_dir)

# Feature importances
save_feature_importances(final_clf, list(feats_imputed.columns), cfg.output_dir)

# Per-class statistics
feature_class_summary = compute_feature_class_summary(
    feats_imputed, expanded_labels, labeled_mask, cfg.output_dir
)

# Cohen's d analysis and plots
if feature_class_summary is not None:
    compute_and_plot_cohens_d(
        feature_class_summary, features_with_preds, cfg.output_dir,
        top_n=20, big_n=cfg.BIG_N,
    )

# SHAP
save_shap_plots(
    final_clf, feats_imputed, cfg.output_dir,
    max_shap_points=cfg.MAX_SHAP_POINTS,
    random_state=cfg.random_state,
)

# Serialise model and preprocessing objects
joblib.dump(final_clf, os.path.join(cfg.output_dir, "rf_final_model.joblib"))
joblib.dump(scaler,    os.path.join(cfg.output_dir, "scaler.joblib"))
joblib.dump(imputer,   os.path.join(cfg.output_dir, "imputer.joblib"))
print("Saved rf_final_model.joblib, scaler.joblib, imputer.joblib")

# %% ── Done ──────────────────────────────────────────────────────────────── #
print("\n" + "=" * 60)
print(f"DONE — all outputs written to: {cfg.output_dir}")
print(f"Total runtime: {time.time() - start_all:.1f} s")
print("=" * 60)
