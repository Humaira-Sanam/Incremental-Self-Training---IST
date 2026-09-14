"""
config.py
---------
All user-tunable parameters for the RF self-training pipeline.
Edit the paths and constants below before running the pipeline.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # I/O paths                                                            #
    # ------------------------------------------------------------------ #
    #: Full point cloud CSV (no header): x, y, z, r, g, b, class
    cloud_path: str = "E:/Vikhroli/Plot1/Classification_paper/Full_plot/plot2/vikhroli2_offG.csv"
    #: Labelled seed CSV (same format as cloud_path)
    train_path: str = "E:/Vikhroli/Plot2/Classification/input/train.csv"
    #: Directory where all outputs are written
    output_dir: str = "E:/Vikhroli/Plot1/Classification_paper/Full_plot/plot2/check_self_roi"

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #
    #: Neighbourhood radii (metres) for multi-scale PCA feature extraction
    radii: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    #: Minimum number of neighbours required to compute features for a point
    min_neighbors: int = 8
    #: Radius used to collect neighbours for local ground estimation
    ground_radius: float = 1.0
    #: Percentile of Z values used as local ground estimate
    ground_percentile: int = 5

    # ------------------------------------------------------------------ #
    # Seed cleaning                                                        #
    # ------------------------------------------------------------------ #
    #: k-nearest labelled neighbours checked during seed consensus cleaning
    seed_consensus_k: int = 10
    #: Fraction of neighbours that must share the same label to keep a seed
    seed_consensus_frac: float = 0.6

    # ------------------------------------------------------------------ #
    # Spatial block holdout split                                          #
    # ------------------------------------------------------------------ #
    #: Number of grid divisions along each XY axis for the spatial block
    #: holdout split.  The plot is divided into n_blocks × n_blocks cells;
    #: outermost cells (maximally separated from the plot centre) are
    #: reserved for testing.  Increase for larger / denser seed sets.
    #: Set use_spatial_block_split = False to revert to the legacy random
    #: stratified split (not recommended for publication).
    spatial_block_n: int = 4
    #: Target fraction of seed points held out for the test set
    test_size: float = 0.20
    #: Use spatial block holdout (True) or legacy random split (False)
    use_spatial_block_split: bool = True

    # ------------------------------------------------------------------ #
    # Self-training loop                                                   #
    # ------------------------------------------------------------------ #
    #: Minimum RF probability for a candidate label to be accepted
    threshold: float = 0.92
    #: χ² confidence level for the Mahalanobis distance gate
    chi2_confidence: float = 0.90
    #: Maximum number of self-training iterations
    max_iters: int = 20
    #: k-nearest labelled neighbours used for spatial consensus check
    k_consensus: int = 10
    #: Minimum fraction of consensus neighbours that must agree on a label
    consensus_frac: float = 0.6
    #: Absolute minimum number of new points required to continue an iteration
    min_new_absolute: int = 50
    #: Fraction of remaining unlabelled points that sets a dynamic minimum
    min_new_fraction: float = 0.001
    #: Upper cap applied to the fraction-based dynamic minimum
    min_new_cap: int = 1000

    # ------------------------------------------------------------------ #
    # Random Forest                                                        #
    # ------------------------------------------------------------------ #
    #: Number of trees in the RF trained at each self-training iteration
    n_estimators_iter: int = 200
    #: Number of trees in the final RF trained after self-training completes
    n_estimators_final: int = 500
    #: Number of parallel jobs (-1 = all available cores)
    n_jobs: int = -1
    #: Random seed for reproducibility
    random_state: int = 42

    # ------------------------------------------------------------------ #
    # Output / safety                                                      #
    # ------------------------------------------------------------------ #
    #: Write a header row to classified_output.csv
    save_classified_with_header: bool = False
    #: Cloud size above which debug CSVs and plots are produced from a sample
    BIG_N: int = 500_000
    #: Maximum number of points used for SHAP explanations
    MAX_SHAP_POINTS: int = 20_000
