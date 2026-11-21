"""Dataset loader and causal anomaly detection.

Reads the authoritative CSV without modifying it, validates schema integrity,
and computes causal anomaly flags.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    DATASET_PATH,
    ANOMALY_RATIO_THRESHOLD,
)

RAW_DTYPES: dict[str, str] = {
    "case_id": "category",
    "num_agents": "int16",
    "n_obstacles": "int16",
    "obstacle_density": "float32",
    "mean_obstacle_size": "float32",
    "clutter_index": "float32",
    "obstacle_clustering": "float32",
    "attrition_rate": "category",
    "step": "int16",
    "agent_id": "int16",
    "rel_x": "float32",
    "rel_y": "float32",
    "rel_z": "float32",
    "dist_to_centroid": "float32",
    "rank_dist_to_centroid": "float32",
    "dist_to_target": "float32",
    "rank_dist_to_target": "float32",
    "vx": "float32",
    "vy": "float32",
    "vz": "float32",
    "speed": "float32",
    "heading": "float32",
    "flight_path_angle": "float32",
    "accel": "float32",
    "heading_rate": "float32",
    "num_neighbors": "int16",
    "mean_neighbor_dist": "float32",
    "min_neighbor_dist": "float32",
    "weighted_density": "float32",
    "angular_coverage": "float32",
    "criticality_score": "float32",
    "degree_centrality": "float32",
    "closeness_centrality": "float32",
    "betweenness_centrality": "float32",
    "eigenvector_centrality": "float32",
    "is_critical": "bool",
    "was_promoted": "bool",
}


def load_raw_dataset(csv_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the dataset and enforce integrity checks and causal anomaly flagging."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Authoritative dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path, dtype=RAW_DTYPES)

    # 1. Structural integrity assertions
    assert df.shape == (1639166, 37), f"Unexpected dataset shape: {df.shape}"
    assert int(df.isna().sum().sum()) == 0, "Dataset contains unexpected NaN values"
    assert not df.duplicated(["case_id", "step", "agent_id"]).any(), "Duplicate keys found"
    assert (df["step"] % 2 == 0).all(), "Non-even step values found"

    # 2. Sort deterministically: scenario -> agent_id -> step (for causal temporal processing)
    df = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)

    # 3. Causal anomaly detection (displacement inconsistent with velocity over 2 sim steps)
    g = df.groupby(["case_id", "agent_id"], observed=True)
    disp = np.sqrt(
        g["rel_x"].diff() ** 2 +
        g["rel_y"].diff() ** 2 +
        g["rel_z"].diff() ** 2
    )
    vprev = np.sqrt(
        g["vx"].shift() ** 2 +
        g["vy"].shift() ** 2 +
        g["vz"].shift() ** 2
    )
    expected_disp = 2.0 * (df["speed"] + vprev.fillna(df["speed"])) / 2.0
    ratio = disp / expected_disp.replace(0.0, np.nan)

    # Flag anomalies (concentrated in ~520 rows in c002 / c099)
    df["is_anomaly"] = (df["step"] > 0) & (ratio > ANOMALY_RATIO_THRESHOLD)

    return df

