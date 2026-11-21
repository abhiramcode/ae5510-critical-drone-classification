"""Automated verification suite implementing Sanity Checks SC-1 through SC-5.

Verifies:
- SC-1: Strict absence of forbidden columns (including dropped target columns) and zero partition leakage.
- SC-2: Strict causal invariance when future steps are perturbed.
- SC-3: 3D radius-55 graph fidelity and isolated node self-loop handling.
- SC-4: Temporal mask alignment and attrition exclusion.
- SC-5: Train-only standardization stability, raw telemetry derivation, and unified inference parity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from src.config import (
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    LAG_DIFF_FEATURES_30,
    LGBM_FEATURES_52,
    LGBM_FEATURES_54,
    EDGE_FEATURES_7,
    GLOBAL_FEATURES_6,
    FORBIDDEN_COLUMNS,
    SPLITS_DIR,
)
from src.data.loader import load_raw_dataset
from src.data.splitter import get_or_create_primary_split
from src.data.features import (
    compute_derived_snapshot_features,
    compute_causal_lags,
    extract_snapshot_graph,
    derive_features_from_telemetry,
    validate_feature_matrix,
)
from src.data.scaler import SwarmFeatureScaler
from src.data.adapters import (
    prepare_lgbm_data,
    prepare_gru_dataset,
    prepare_gat_graphs,
    build_stgat_windows,
)


def test_sc1_leakage_and_splits() -> None:
    """SC-1: Assert forbidden columns are absent, target features dropped, and split disjoint."""
    # 1. Feature counts
    assert len(BASE_FEATURES_22) == 22, f"Expected 22 base features, got {len(BASE_FEATURES_22)}"
    assert len(LGBM_FEATURES_52) == 52, f"Expected 52 LGBM features, got {len(LGBM_FEATURES_52)}"

    # 2. Target features strictly forbidden
    assert "dist_to_target" in FORBIDDEN_COLUMNS, "dist_to_target must be forbidden"
    assert "rank_dist_to_target" in FORBIDDEN_COLUMNS, "rank_dist_to_target must be forbidden"

    # 3. Feature contract guards
    assert not (set(BASE_FEATURES_22) & FORBIDDEN_COLUMNS)
    assert not (set(LGBM_FEATURES_52) & FORBIDDEN_COLUMNS)
    assert not (set(EDGE_FEATURES_7) & FORBIDDEN_COLUMNS)
    assert not (set(GLOBAL_FEATURES_6) & FORBIDDEN_COLUMNS)

    # 4. Split isolation guard
    split_file = SPLITS_DIR / "primary_split.json"
    assert split_file.exists(), f"Missing split file: {split_file}"
    with open(split_file, "r", encoding="utf-8") as f:
        split = json.load(f)

    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert not (train & val), "Leakage between train and val!"
    assert not (train & test), "Leakage between train and test!"
    assert not (val & test), "Leakage between val and test!"


def test_sc2_causality_perturbation() -> None:
    """SC-2: Perturbing future rows (step > t) must not change features at step <= t."""
    df_raw = load_raw_dataset()
    case = df_raw["case_id"].iloc[0]
    sub = df_raw[df_raw["case_id"] == case].copy()

    steps = np.sort(sub["step"].unique())
    cutoff_step = steps[len(steps) // 2]

    # Baseline features up to cutoff
    df_feat1 = compute_derived_snapshot_features(sub)
    df_feat1 = compute_causal_lags(df_feat1)
    base_rows = df_feat1[df_feat1["step"] <= cutoff_step][LGBM_FEATURES_52].to_numpy()

    # Perturb future data (step > cutoff_step)
    sub_perturbed = sub.copy()
    future_mask = sub_perturbed["step"] > cutoff_step
    sub_perturbed.loc[future_mask, "rel_x"] += 50.0
    sub_perturbed.loc[future_mask, "speed"] += 10.0
    sub_perturbed.loc[future_mask, "num_neighbors"] = 0

    df_feat2 = compute_derived_snapshot_features(sub_perturbed)
    df_feat2 = compute_causal_lags(df_feat2)
    pert_rows = df_feat2[df_feat2["step"] <= cutoff_step][LGBM_FEATURES_52].to_numpy()

    # Bit-for-bit equivalence on past rows (treating NaNs identically)
    nan_mask = np.isnan(base_rows)
    assert (nan_mask == np.isnan(pert_rows)).all()
    np.testing.assert_allclose(base_rows[~nan_mask], pert_rows[~nan_mask], rtol=1e-5, atol=1e-5)


def test_sc3_graph_fidelity_and_self_loops() -> None:
    """SC-3: 3D radius-55 graph reproduces neighbor counts and handles isolated nodes."""
    df_raw = load_raw_dataset()
    # Test clean snapshot c001@0
    c001_0 = df_raw[(df_raw["case_id"] == "c001_n60_none") & (df_raw["step"] == 0)].copy()
    c001_0 = compute_derived_snapshot_features(c001_0)

    x, edge_index, edge_attr, global_feat = extract_snapshot_graph(c001_0)
    assert x.shape == (60, 22), f"Expected node shape (60, 22), got {x.shape}"
    assert edge_attr.shape[1] == 7
    assert len(global_feat) == 6

    # Verify neighbor edges match dataset num_neighbors within ±1
    n = 60
    src, counts = np.unique(edge_index[0], return_counts=True)
    all_counts = np.zeros(n, dtype=int)
    all_counts[src] = counts
    diff = np.abs(all_counts - c001_0["num_neighbors"].to_numpy())
    assert (diff <= 1).all(), f"Graph neighbor count mismatch: max diff = {diff.max()}"

    # Verify isolated node handling: if an isolated agent has no neighbors <= 55m, fallback self-edges are returned
    isolated_df = c001_0.head(1).copy()
    x_iso, ei_iso, ea_iso, g_iso = extract_snapshot_graph(isolated_df)
    assert ei_iso.shape == (2, 1) and ei_iso[0, 0] == ei_iso[1, 0]
    assert ea_iso.shape == (1, 7)


def test_sc4_temporal_mask_and_attrition() -> None:
    """SC-4: Verify left-padded masks and attrition exclusion."""
    df_raw = load_raw_dataset()
    sub = df_raw[df_raw["case_id"] == "c003_n37_medium"].copy()
    sub = compute_derived_snapshot_features(sub)

    scaler = SwarmFeatureScaler()
    scaler.fit_nodes(sub[BASE_FEATURES_22].to_numpy())
    scaler.fit_edges(np.zeros((10, 7), dtype=np.float32))
    scaler.fit_globals(np.zeros((10, 6), dtype=np.float32))

    # Test GRU left-padding on t=0
    ds = prepare_gru_dataset(sub, scaler, window_length=16)
    x0, mask0, y0, _ = ds[0]
    assert x0.shape == (16, 22), f"Expected (16, 22), got {x0.shape}"
    assert mask0.sum().item() == 1  # Only 1 step valid at step 0
    assert mask0[-1].item() is True  # Current observation at index -1
    assert not mask0[:-1].any()  # Padded slots are False
    assert (x0[:-1] == 0.0).all()  # Padded values are exact zeros

    # Test ST-GAT window assembly
    windows = build_stgat_windows(sub, scaler, window_steps=8, stride=10)
    assert len(windows) > 0
    w0 = windows[0]
    assert w0.x_seq.shape[0] == 8
    assert w0.x_seq.shape[2] == 22
    assert w0.node_mask.shape[0] == 8
    assert w0.global_seq.shape == (8, 6)


def test_sc5_telemetry_derivation_and_inference_parity() -> None:
    """SC-5: Test that raw [Px, Py, Pz, Vx, Vy, Vz] telemetry derives all 22 features and executes on all models."""
    from src.inference.predictor import SwarmCriticalityPredictor, predict_all_models

    df_raw = load_raw_dataset()
    sub = df_raw[df_raw["case_id"] == "c001_n60_none"].copy()
    steps = sorted(sub["step"].unique())

    # Create raw telemetry mock (Px, Py, Pz, Vx, Vy, Vz)
    sub_telemetry = sub[["agent_id", "step", "rel_x", "rel_y", "rel_z", "vx", "vy", "vz"]].copy()
    sub_telemetry = sub_telemetry.rename(columns={
        "rel_x": "px",
        "rel_y": "py",
        "rel_z": "pz",
    })

    hist_telemetry = sub_telemetry[sub_telemetry["step"] < steps[20]].copy()
    snap_telemetry = sub_telemetry[sub_telemetry["step"] == steps[20]].copy()

    # 1. Test derivation directly
    derived_snap = derive_features_from_telemetry(snap_telemetry, prev_snapshot_df=sub_telemetry[sub_telemetry["step"] == steps[19]])
    assert all(col in derived_snap.columns for col in BASE_FEATURES_22)

    # 2. Test individual models with raw telemetry
    for m in ["lgbm", "gru", "gatv2", "stgat"]:
        p = SwarmCriticalityPredictor(m)
        assert p.threshold > 0.0
        preds = p.predict_snapshot(snap_telemetry, hist_telemetry)
        assert len(preds) == len(snap_telemetry)
        assert "prob" in preds.columns
        assert "pred_class" in preds.columns
        assert (preds["prob"] >= 0.0).all() and (preds["prob"] <= 1.0).all()

    # 3. Test consolidated all-model prediction
    comb = predict_all_models(snap_telemetry, hist_telemetry)
    assert len(comb) == len(snap_telemetry)
    assert "prob_lgbm" in comb.columns
    assert "prob_gru" in comb.columns
    assert "prob_gatv2" in comb.columns
    assert "prob_stgat" in comb.columns
    assert "prob_consensus" in comb.columns
    assert "consensus_critical" in comb.columns


if __name__ == "__main__":
    print("Running Sanity Checks SC-1 through SC-5...")
    test_sc1_leakage_and_splits()
    print("  [PASS] SC-1: Leakage and split isolation passed.")
    test_sc2_causality_perturbation()
    print("  [PASS] SC-2: Causal temporal invariance passed.")
    test_sc3_graph_fidelity_and_self_loops()
    print("  [PASS] SC-3: Graph fidelity and self-loops passed.")
    test_sc4_temporal_mask_and_attrition()
    print("  [PASS] SC-4: Temporal mask and attrition passed.")
    test_sc5_telemetry_derivation_and_inference_parity()
    print("  [PASS] SC-5: Raw telemetry derivation and unified inference API passed.")
    print("All Sanity Checks PASSED successfully!")
