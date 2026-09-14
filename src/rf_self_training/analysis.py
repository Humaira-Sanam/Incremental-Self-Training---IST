"""
analysis.py
-----------
Post-classification analysis, diagnostics, and visualisations:

  - Held-out test metrics (classification report, confusion matrix)
  - Per-class feature statistics (mean, median, std, mode)
  - Cohen's d feature-separation ranking, bar chart, and box plots
  - Optional SHAP summary plots
  - Feature importance bar chart
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Held-out test metrics                                                        #
# --------------------------------------------------------------------------- #

def save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
) -> None:
    """Save classification report CSV and confusion-matrix PNG."""
    print("Held-out test classification report:")
    print(classification_report(y_true, y_pred))

    pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).T.to_csv(os.path.join(output_dir, "metrics_report.csv"))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print("Saved metrics_report.csv and confusion_matrix.png")


# --------------------------------------------------------------------------- #
# Feature importances                                                          #
# --------------------------------------------------------------------------- #

def save_feature_importances(
    clf: RandomForestClassifier,
    feature_names: list[str],
    output_dir: str,
    top_k: int = 20,
) -> None:
    """Save feature importance CSV and bar chart PNG."""
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    fi.to_csv(os.path.join(output_dir, "feature_importances.csv"), index=False)

    top = fi.head(top_k)
    plt.figure(figsize=(8, 6))
    plt.barh(range(top.shape[0]), top["importance"][::-1])
    plt.yticks(range(top.shape[0]), top["feature"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_k} feature importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=300)
    plt.close()
    print("Saved feature_importances.csv and feature_importances.png")


# --------------------------------------------------------------------------- #
# Per-class feature statistics                                                 #
# --------------------------------------------------------------------------- #

def compute_feature_class_summary(
    feats_imputed: pd.DataFrame,
    expanded_labels: np.ndarray,
    labeled_mask: np.ndarray,
    output_dir: str,
) -> Optional[pd.DataFrame]:
    """
    Compute per-class mean, median, std, and mode for every feature and save to CSV.

    Returns the summary DataFrame, or ``None`` if no valid classes are found.
    """
    print("Computing per-class feature statistics …")
    feature_cols  = feats_imputed.columns
    valid_classes = sorted(
        [c for c in np.unique(expanded_labels[labeled_mask]) if c >= 0]
    )
    summary_list  = []

    for cls in valid_classes:
        cls_mask  = labeled_mask & (expanded_labels == cls)
        data_cls  = feats_imputed.loc[cls_mask, feature_cols]
        if data_cls.shape[0] == 0:
            continue

        desc      = data_cls.agg(["mean", "median", "std"])
        mode_vals = data_cls.mode(dropna=True)
        mode_vals = mode_vals.iloc[0] if mode_vals.shape[0] > 0 else pd.Series(
            [np.nan] * len(feature_cols), index=feature_cols
        )
        summary_list.append(pd.DataFrame({
            "class":    cls,
            "feature":  feature_cols,
            "mean":     desc.loc["mean"].values,
            "median":   desc.loc["median"].values,
            "std":      desc.loc["std"].values,
            "mode":     mode_vals.values,
        }))

    if not summary_list:
        print("No labelled classes found for summary; feature_class_summary.csv not created.")
        return None

    summary = pd.concat(summary_list, ignore_index=True)
    summary.to_csv(os.path.join(output_dir, "feature_class_summary.csv"), index=False)
    print("Saved feature_class_summary.csv")
    return summary


# --------------------------------------------------------------------------- #
# Cohen's d ranking and plots                                                  #
# --------------------------------------------------------------------------- #

def compute_and_plot_cohens_d(
    feature_class_summary: pd.DataFrame,
    features_with_preds: pd.DataFrame,
    output_dir: str,
    top_n: int = 20,
    big_n: int = 500_000,
) -> None:
    """
    Compute Cohen's d for each feature, save a ranked CSV, bar chart, and box plots.

    Parameters
    ----------
    feature_class_summary:
        Output of :func:`compute_feature_class_summary`.
    features_with_preds:
        Feature matrix with a ``pred_label`` column appended.
    output_dir:
        Directory for all output files.
    top_n:
        Number of top features to include in charts.
    big_n:
        Cloud size above which box plots are produced from a random sample.
    """
    print("Computing Cohen's d …")
    wide = feature_class_summary.pivot(
        index="feature", columns="class", values=["mean", "std"]
    )
    wide.columns = [f"{m}_class{int(c)}" for (m, c) in wide.columns]
    wide = wide.reset_index()

    required = {"mean_class1", "mean_class2", "std_class1", "std_class2"}
    if not required.issubset(set(wide.columns)):
        print("Cohen's d skipped: expected class labels {1, 2}.")
        return

    wide["std_pooled"]    = np.sqrt((wide["std_class1"] ** 2 + wide["std_class2"] ** 2) / 2.0)
    wide["cohen_d"]       = (wide["mean_class1"] - wide["mean_class2"]) / wide["std_pooled"]
    wide["relative_diff"] = (
        np.abs(wide["mean_class1"] - wide["mean_class2"])
        / wide[["mean_class1", "mean_class2"]].abs().max(axis=1)
    )

    wide_sorted = wide.sort_values("cohen_d", key=lambda s: s.abs(), ascending=False)
    wide_sorted.to_csv(
        os.path.join(output_dir, "feature_difference_ranked_cohend.csv"), index=False
    )
    print("Saved feature_difference_ranked_cohend.csv")

    # Bar chart
    top = wide_sorted.head(top_n).sort_values("cohen_d", key=lambda s: s.abs())
    plt.figure(figsize=(10, 6))
    y_pos = range(len(top))
    plt.barh(y_pos, top["cohen_d"])
    plt.yticks(y_pos, top["feature"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Cohen's d  (class 1 − class 2 in SD units)")
    plt.title(f"Top {top_n} features by class separation (|Cohen's d|)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_features_cohend_bar.png"), dpi=300)
    plt.close()
    print("Saved top_features_cohend_bar.png")

    # Box plots
    print("Creating box plots for top discriminative features …")
    top_feats = top["feature"].tolist()
    n_feats   = len(top_feats)
    n_cols    = 3
    n_rows    = int(np.ceil(n_feats / n_cols))

    plot_data = (
        features_with_preds.sample(200_000, random_state=42)
        if features_with_preds.shape[0] > big_n
        else features_with_preds
    )

    plt.figure(figsize=(4 * n_cols, 3.5 * n_rows))
    for i, feat in enumerate(top_feats, 1):
        plt.subplot(n_rows, n_cols, i)
        vals1 = plot_data.loc[plot_data["pred_label"] == 1, feat].dropna()
        vals2 = plot_data.loc[plot_data["pred_label"] == 2, feat].dropna()
        plt.boxplot([vals1, vals2], labels=["Class 1", "Class 2"], showfliers=False)
        plt.title(feat, fontsize=8)

    plt.suptitle("Top features by Cohen's d – class-wise distributions", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "top_features_boxplots.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved top_features_boxplots.png")


# --------------------------------------------------------------------------- #
# SHAP                                                                         #
# --------------------------------------------------------------------------- #

def save_shap_plots(
    clf: RandomForestClassifier,
    feats_imputed: pd.DataFrame,
    output_dir: str,
    max_shap_points: int = 20_000,
    random_state: int = 42,
) -> None:
    """
    Compute SHAP values on a sample of points and save one summary plot per class.

    Silently skips if ``shap`` is not installed.
    """
    if not SHAP_AVAILABLE:
        print("shap not installed; SHAP plots skipped.")
        return

    n_points = feats_imputed.shape[0]
    shap_n   = min(max_shap_points, n_points)
    rng      = np.random.RandomState(random_state)
    shap_idx = rng.choice(n_points, size=shap_n, replace=False)
    feats_shap = feats_imputed.iloc[shap_idx]

    print(f"Computing SHAP on {shap_n} of {n_points} points …")
    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(feats_shap)
        for i, cls in enumerate(clf.classes_):
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values[i], feats_shap, show=False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"shap_summary_class_{cls}.png"), dpi=300
            )
            plt.close()
        print("Saved SHAP summary plots.")
    except Exception as exc:
        print(f"SHAP plotting failed: {exc}")
