# RF Self-Training Point Cloud Classifier (z_rel)

A self-training Random Forest pipeline for classifying 3D LiDAR / point cloud data into two classes (e.g. wood vs. leaf). The pipeline uses **relative height features** (`z_rel = local_mean_z − local_ground_z`) and iteratively expands a small seed training set via confident, consensus-filtered predictions.

---

## Features

- Multi-scale geometric feature extraction (PCA eigenvalues, surface variation, roughness, curvature, verticality, etc.)
- Relative height (`z_rel`) computed from a local ground-percentile estimate — no separate DTM required
- RGB-derived colour features
- Seed cleaning via spatial label consensus
- Conservative self-training loop with:
  - Probability threshold gate
  - Mahalanobis distance (χ² confidence ellipsoid) gate
  - Spatial neighbourhood consensus gate
  - Configurable minimum-additions criterion to prevent label drift
- Cohen's *d* feature-separation analysis with ranked CSV and plots
- Optional SHAP feature-importance summary
- Big-cloud safety: sampling for debug CSVs, plots, and SHAP when N > 500 k

---

## Repository Layout

```
rf_self_training/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   └── rf_self_training/
│       ├── __init__.py
│       ├── config.py          # All user-tunable parameters
│       ├── features.py        # PCA / geometric / colour feature extraction
│       ├── ground.py          # Local ground-Z estimation
│       ├── seed_utils.py      # Seed mapping, cleaning, and splitting
│       ├── self_training.py   # Conservative self-training loop
│       └── analysis.py        # Metrics, Cohen's d, SHAP, plots
└── notebooks/
    └── run_pipeline.py        # End-to-end entry point (notebook-style script)
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure paths and parameters

Edit `src/rf_self_training/config.py` (or override at runtime):

```python
cloud_path  = "path/to/full_cloud.csv"   # no-header CSV: x,y,z,r,g,b,class
train_path  = "path/to/train.csv"         # labelled seed points, same format
output_dir  = "path/to/output/"
```

### 3. Run

```bash
python notebooks/run_pipeline.py
```

---

## Input Format

Both `cloud_path` and `train_path` are **headerless CSVs** with exactly seven columns:

| x | y | z | r | g | b | class |
|---|---|---|---|---|---|-------|
| float | float | float | int/float | int/float | int/float | int |

Class labels must be integers (any two distinct values; they are remapped to `{1, 2}` automatically).

---

## Outputs

| File | Description |
|------|-------------|
| `classified_output.csv` | Full cloud with predicted class column |
| `final_training_data.csv` | Expanded training set (seed + self-trained labels) |
| `feature_importances.csv / .png` | RF feature importances |
| `features_imputed[_sample].csv` | Imputed feature matrix (sampled if N > 500 k) |
| `features_with_predictions[_sample].csv` | Features + predicted labels + class probabilities |
| `metrics_report.csv` | Held-out test set classification report |
| `confusion_matrix.png` | Confusion matrix plot |
| `feature_class_summary.csv` | Per-class mean / median / std / mode for every feature |
| `feature_difference_ranked_cohend.csv` | Features ranked by \|Cohen's *d*\| |
| `top_features_cohend_bar.png` | Bar chart of top-20 features by Cohen's *d* |
| `top_features_boxplots.png` | Box plots of top-20 features, class-wise |
| `shap_summary_class_*.png` | SHAP summary plots (if `shap` is installed) |
| `rf_final_model.joblib` | Trained Random Forest model |
| `scaler.joblib` | Fitted `StandardScaler` |
| `imputer.joblib` | Fitted `SimpleImputer` |

---

## Parameters

See `src/rf_self_training/config.py` for full documentation. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radii` | `[0.05, 0.1, 0.2]` | Neighbourhood radii for multi-scale features |
| `ground_radius` | `1.0` | Radius for local ground estimation |
| `ground_percentile` | `5` | Percentile used as local ground Z |
| `threshold` | `0.92` | Minimum RF probability to accept a label |
| `chi2_confidence` | `0.90` | Mahalanobis gate (χ² confidence level) |
| `max_iters` | `20` | Maximum self-training iterations |
| `consensus_frac` | `0.6` | Minimum neighbour-agreement fraction |
| `min_new_absolute` | `50` | Minimum new points per iteration |
| `min_new_cap` | `1000` | Cap on fraction-based minimum threshold |

---

## Requirements

- Python ≥ 3.9
- See `requirements.txt`
- Optional: `shap` for SHAP feature explanations

---

## Citation / Acknowledgements

If you use this pipeline in published work, please cite the data source and this repository.
