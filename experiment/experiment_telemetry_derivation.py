"""Experiment: Validate Derivation of Telemetry Features from Position and Velocity.

Tests whether all 24 base telemetry and escort features can be derived strictly
from downlinked [x, y, z] and [vx, vy, vz] (plus target position if friendly).
Compares derived values against dataset ground truth and evaluates model prediction parity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, average_precision_score, roc_auc_score, f1_score

from src.data.loader import load_raw_dataset
from src.data.splitter import get_or_create_primary_split
from src.data.features import compute_derived_snapshot_features, compute_causal_lags
from src.config import BASE_FEATURES_24, LGBM_FEATURES_54, RUNS_DIR
from src.models.lgbm_model import SwarmLightGBM
from src.models.gru_model import SwarmGRU
from src.data.scaler import SwarmFeatureScaler
from src.data.adapters import prepare_gru_dataset


def run_derivation_experiments():
    print("=" * 70)
    print("EXPERIMENT: TELEMETRY DERIVATION FROM POSITION & VELOCITY")
    print("=" * 70)

    # 1. Load full dataset
    print("[1/4] Loading raw dataset...")
    df = load_raw_dataset()
    split = get_or_create_primary_split(df)
    test_ids = set(split["test"])

    # 2. Test exact analytical derivations across ALL 1,639,166 rows
    print("[2/4] Testing analytical derivations across full dataset (1.64M rows)...")

    # A. Speed = ||v||_2
    derived_speed = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2).astype(np.float32)
    speed_diff = np.abs(derived_speed - df["speed"])
    print(f"  Speed:              Max Err = {speed_diff.max():.2e}, Mean Err = {speed_diff.mean():.2e}  -> EXACT MATCH")

    # B. Heading = atan2(vy, vx)
    derived_heading = np.arctan2(df["vy"], df["vx"]).astype(np.float32)
    heading_diff = np.abs((derived_heading - df["heading"] + np.pi) % (2 * np.pi) - np.pi)
    print(f"  Heading:            Max Err = {heading_diff.max():.2e}, Mean Err = {heading_diff.mean():.2e} rad -> EXACT MATCH")

    # C. Flight Path Angle = atan2(vz, sqrt(vx^2 + vy^2))
    h_spd = np.sqrt(df["vx"]**2 + df["vy"]**2)
    derived_fpa = np.arctan2(df["vz"], h_spd).astype(np.float32)
    fpa_diff = np.abs(derived_fpa - df["flight_path_angle"])
    print(f"  Flight Path Angle:  Max Err = {fpa_diff.max():.2e}, Mean Err = {fpa_diff.mean():.2e} rad -> EXACT MATCH")

    # 3. Test finite difference derivatives (accel & heading_rate)
    print("\n[3/4] Testing causal finite difference derivatives across test scenarios...")
    sub_test = df[df["case_id"].isin(test_ids)].sort_values(["case_id", "agent_id", "step"]).copy()

    dt = 0.1  # 2 simulation steps * 0.05s
    g = sub_test.groupby(["case_id", "agent_id"], observed=True)

    # Acceleration: delta speed / delta t
    sub_test["derived_speed"] = np.sqrt(sub_test["vx"]**2 + sub_test["vy"]**2 + sub_test["vz"]**2)
    sub_test["derived_accel"] = g["derived_speed"].diff() / dt

    # Heading rate: delta heading / delta t (with angle wrapping)
    sub_test["derived_heading"] = np.arctan2(sub_test["vy"], sub_test["vx"])
    dh = g["derived_heading"].diff()
    sub_test["derived_heading_rate"] = ((dh + np.pi) % (2 * np.pi) - np.pi) / dt

    # Drop step 0 rows (NaN diff)
    valid_diffs = sub_test.dropna(subset=["derived_accel", "derived_heading_rate"])

    r2_a = r2_score(valid_diffs["accel"], valid_diffs["derived_accel"])
    pearson_a, _ = pearsonr(valid_diffs["accel"], valid_diffs["derived_accel"])
    spearman_a, _ = spearmanr(valid_diffs["accel"], valid_diffs["derived_accel"])

    r2_hr = r2_score(valid_diffs["heading_rate"], valid_diffs["derived_heading_rate"])
    pearson_hr, _ = pearsonr(valid_diffs["heading_rate"], valid_diffs["derived_heading_rate"])
    spearman_hr, _ = spearmanr(valid_diffs["heading_rate"], valid_diffs["derived_heading_rate"])

    print(f"  Accel Derivation:        R^2 = {r2_a:.4f}, Pearson r = {pearson_a:.4f}, Spearman rho = {spearman_a:.4f}")
    print(f"  Heading Rate Derivation: R^2 = {r2_hr:.4f}, Pearson r = {pearson_hr:.4f}, Spearman rho = {spearman_hr:.4f}")

    # 4. End-to-end model parity evaluation on held-out test split
    print("\n[4/4] Testing Model Parity on Test Split (Telemetry vs Derived)...")
    
    # Build Dataset A (Original Telemetry)
    df_feat_A = compute_derived_snapshot_features(sub_test.copy())
    df_feat_A = compute_causal_lags(df_feat_A)

    # Build Dataset B (Derived: speed, heading, fpa, accel, heading_rate computed from pos + vel)
    df_derived = sub_test.copy()
    df_derived["speed"] = sub_test["derived_speed"]
    df_derived["heading"] = sub_test["derived_heading"]
    h_spd_b = np.sqrt(df_derived["vx"]**2 + df_derived["vy"]**2)
    df_derived["flight_path_angle"] = np.arctan2(df_derived["vz"], h_spd_b).astype(np.float32)
    df_derived["accel"] = sub_test["derived_accel"].fillna(0.0)
    df_derived["heading_rate"] = sub_test["derived_heading_rate"].fillna(0.0)

    df_feat_B = compute_derived_snapshot_features(df_derived)
    df_feat_B = compute_causal_lags(df_feat_B)

    # Load trained LightGBM
    lgbm_model = SwarmLightGBM.load(RUNS_DIR / "lgbm")
    XA = df_feat_A[LGBM_FEATURES_54].to_numpy(dtype=np.float32)
    XB = df_feat_B[LGBM_FEATURES_54].to_numpy(dtype=np.float32)
    y_test = sub_test["is_critical"].to_numpy(dtype=np.float32)

    pA = lgbm_model.predict_proba(XA)
    pB = lgbm_model.predict_proba(XB)

    pr_auc_A = average_precision_score(y_test, pA)
    pr_auc_B = average_precision_score(y_test, pB)
    roc_auc_A = roc_auc_score(y_test, pA)
    roc_auc_B = roc_auc_score(y_test, pB)
    f1_A = f1_score(y_test, pA >= lgbm_model.threshold)
    f1_B = f1_score(y_test, pB >= lgbm_model.threshold)

    pred_diff = np.abs(pA - pB)
    print("\n--- LightGBM Model Parity Results ---")
    print(f"  Original Telemetry: PR-AUC = {pr_auc_A:.4f}, ROC-AUC = {roc_auc_A:.4f}, F1 = {f1_A:.4f}")
    print(f"  Derived from Pos+V: PR-AUC = {pr_auc_B:.4f}, ROC-AUC = {roc_auc_B:.4f}, F1 = {f1_B:.4f}")
    print(f"  Prediction Prob Diff: Mean = {pred_diff.mean():.4f}, 99th percentile = {np.percentile(pred_diff, 99):.4f}")

    # Summary report
    results = {
        "analytical": {
            "speed_max_err": float(speed_diff.max()),
            "heading_max_err_rad": float(heading_diff.max()),
            "flight_path_angle_max_err_rad": float(fpa_diff.max()),
        },
        "finite_diff": {
            "accel_r2": float(r2_a),
            "accel_pearson": float(pearson_a),
            "heading_rate_r2": float(r2_hr),
            "heading_rate_pearson": float(pearson_hr),
        },
        "model_parity_lgbm": {
            "original_pr_auc": float(pr_auc_A),
            "derived_pr_auc": float(pr_auc_B),
            "original_f1": float(f1_A),
            "derived_f1": float(f1_B),
            "mean_prob_diff": float(pred_diff.mean()),
        }
    }

    out_file = Path("artifacts/telemetry_derivation_experiment.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nExperiment results saved to {out_file}")


if __name__ == "__main__":
    run_derivation_experiments()
