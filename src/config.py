"""Authoritative project configuration, feature allowlists, and hyperparameter schemas.

This module is the single source of truth for paths, seeds, feature schemas,
forbidden feature guards, and model configurations across all four models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final
import torch

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT_DIR: Final[Path] = Path(__file__).resolve().parents[1]
DATASET_PATH: Final[Path] = ROOT_DIR / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
ARTIFACTS_DIR: Final[Path] = ROOT_DIR / "artifacts"
SPLITS_DIR: Final[Path] = ARTIFACTS_DIR / "splits"
RUNS_DIR: Final[Path] = ARTIFACTS_DIR / "runs"

# -----------------------------------------------------------------------------
# Environment & Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 0
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Domain Constants
# -----------------------------------------------------------------------------
TARGET_COLUMN: Final[str] = "is_critical"
PROMOTION_FLAG_COLUMN: Final[str] = "was_promoted"
ORACLE_LEAK_COLUMN: Final[str] = "criticality_score"

SENSING_RADIUS: Final[float] = 55.0
ANOMALY_RATIO_THRESHOLD: Final[float] = 3.0

# -----------------------------------------------------------------------------
# Definitive Feature Contract
# -----------------------------------------------------------------------------
# 1. Canonical 22 base features (target-agnostic, shared order across all models)
BASE_FEATURES_22: Final[list[str]] = [
    "speed",
    "accel",
    "heading_rate",
    "flight_path_angle",
    "heading_sin",
    "heading_cos",
    "dist_to_centroid",
    "rank_dist_to_centroid",
    "rel_z",
    "num_neighbors",
    "centroid_vel_alignment",
    "is_isolated",
    "mean_neighbor_dist",
    "min_neighbor_dist",
    "weighted_density",
    "angular_coverage",
    "n_within_r15",
    "n_within_r30",
    "n_within_r45",
    "nn_bearing_entropy",
    "local_density_contrast",
    "degree_centrality_c",
]
# Canonical alias
BASE_FEATURES: Final[list[str]] = BASE_FEATURES_22
BASE_FEATURES_24: Final[list[str]] = BASE_FEATURES_22  # Backward compatibility alias

# 2. Causal lag differences for LightGBM (6 base signals x 5 offsets = 30 features)
LAG_BASE_MAP: Final[dict[str, str]] = {
    "dist": "dist_to_centroid",
    "rank": "rank_dist_to_centroid",
    "nb": "num_neighbors",
    "dens": "weighted_density",
    "ang": "angular_coverage",
    "speed": "speed",
}
LAG_OFFSETS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)

LAG_DIFF_FEATURES_30: Final[list[str]] = [
    f"d{lag}_{tag}"
    for tag in LAG_BASE_MAP.keys()
    for lag in LAG_OFFSETS
]

# LightGBM full tabular feature set (22 + 30 = 52)
LGBM_FEATURES_52: Final[list[str]] = BASE_FEATURES_22 + LAG_DIFF_FEATURES_30
LGBM_FEATURES: Final[list[str]] = LGBM_FEATURES_52
LGBM_FEATURES_54: Final[list[str]] = LGBM_FEATURES_52  # Backward compatibility alias

# 3. Directed Edge Features (7 features per edge src -> dst in receiver frame)
EDGE_FEATURES_7: Final[list[str]] = [
    "dist_norm",
    "rel_speed",
    "closing_rate",
    "bearing_cos",
    "bearing_sin",
    "dheading_cos",
    "dheading_sin",
]

# 4. Centralized Global Swarm Context (6 features per snapshot)
GLOBAL_FEATURES_6: Final[list[str]] = [
    "alive_count",
    "swarm_spread",
    "swarm_centroid_speed",
    "graph_density",
    "n_components",
    "largest_component_frac",
]

# -----------------------------------------------------------------------------
# Forbidden Features Allowlist Guard
# -----------------------------------------------------------------------------
FORBIDDEN_COLUMNS: Final[set[str]] = {
    "criticality_score",
    "was_promoted",
    "is_critical",
    "case_id",
    "agent_id",
    # Target mission information (dropped for target-agnostic / adversarial operation)
    "dist_to_target",
    "rank_dist_to_target",
    # Scenario / environment features (validated no gain & zero-obstacle anomaly)
    "n_obstacles",
    "obstacle_density",
    "mean_obstacle_size",
    "clutter_index",
    "obstacle_clustering",
    "num_agents",
    "attrition_rate",
    # Raw simulator centrality columns (replaced by recomputed degree_centrality_c)
    "degree_centrality",
    "closeness_centrality",
    "betweenness_centrality",
    "eigenvector_centrality",
    # Raw translation-variant coordinates / velocity components (redundant with norms / angles)
    "rel_x",
    "rel_y",
    "vx",
    "vy",
    "vz",
    "heading",
}

# -----------------------------------------------------------------------------
# Model Configurations (Sufficient Hyperparameters for Optimal Convergence)
# -----------------------------------------------------------------------------
LGBM_CONFIG: Final[dict] = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "early_stopping_rounds": 50,
}

GRU_CONFIG: Final[dict] = {
    "input_size": 22,
    "hidden_size": 96,
    "num_layers": 2,
    "dropout": 0.1,
    "window_length": 16,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 12,
    "patience": 4,
    "clip_grad_norm": 1.0,
}

GAT_CONFIG: Final[dict] = {
    "in_channels": 22,
    "hidden_size": 64,
    "edge_dim": 7,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "global_dim": 6,
    "batch_size": 32,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "max_epochs": 8,
    "patience": 4,
    "clip_grad_norm": 1.0,
}

STGAT_CONFIG: Final[dict] = {
    "in_channels": 22,
    "hidden_size": 64,
    "edge_dim": 7,
    "num_heads": 4,
    "num_layers": 2,
    "spatial_dropout": 0.1,
    "temporal_hidden": 64,
    "global_dim": 6,
    "temporal_window": 8,
    "batch_size": 16,
    "train_stride": 4,
    "eval_stride": 4,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "max_epochs": 8,
    "patience": 4,
    "clip_grad_norm": 1.0,
}

