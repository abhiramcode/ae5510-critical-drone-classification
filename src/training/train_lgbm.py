"""CLI entry point to train and evaluate the LightGBM model.

Runs:
1. Load dataset & extract primary split
2. Compute derived features & causal lags
3. Train LightGBM with validation early stopping
4. Tune operating threshold on validation set
5. Evaluate completely on test set
6. Persist all 7 run artifacts
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import (
    RUNS_DIR,
    LGBM_CONFIG,
    LGBM_FEATURES_52,
    LGBM_FEATURES_54,
    TARGET_COLUMN,
)
from src.data.loader import load_raw_dataset
from src.data.splitter import get_or_create_primary_split
from src.data.features import (
    compute_derived_snapshot_features,
    compute_causal_lags,
)
from src.data.adapters import prepare_lgbm_data
from src.models.lgbm_model import SwarmLightGBM
from src.evaluation.metrics import pick_optimal_threshold
from src.evaluation.evaluate import run_full_evaluation, save_evaluation_artifacts


def train_lgbm_pipeline(run_dir: Path = RUNS_DIR / "lgbm") -> dict:
    """Train and evaluate LightGBM end-to-end."""
    print("=" * 60)
    print("STAGE 6: LightGBM Tabular Ensemble Training")
    print("=" * 60)

    t0 = time.time()
    # 1. Load data & split
    print("[1/5] Loading raw dataset and primary scenario split...")
    df_raw = load_raw_dataset()
    split = get_or_create_primary_split(df_raw)
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])
    print(f"      Scenarios: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # 2. Compute features
    print("[2/5] Computing derived snapshot geometry & causal lag features...")
    df_feat = compute_derived_snapshot_features(df_raw)
    df_feat = compute_causal_lags(df_feat)

    # 3. Prepare data partitions
    print("[3/5] Partitioning feature matrices...")
    X, y, meta = prepare_lgbm_data(df_feat)
    case_ids = meta["case_id"].astype(str).to_numpy()
    anomalies = meta["is_anomaly"].to_numpy()

    # Train mask: train scenarios, excluding anomalous rows
    train_mask = np.isin(case_ids, list(train_ids)) & (~anomalies)
    val_mask = np.isin(case_ids, list(val_ids))
    test_mask = np.isin(case_ids, list(test_ids))

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_val = meta.loc[val_mask].copy()
    meta_test = meta.loc[test_mask].copy()
    meta_val["is_critical"] = y_val
    meta_test["is_critical"] = y_test

    print(f"      Train samples: {len(X_train)} (positive rate: {y_train.mean():.3f})")
    print(f"      Val samples:   {len(X_val)} (positive rate: {y_val.mean():.3f})")
    print(f"      Test samples:  {len(X_test)} (positive rate: {y_test.mean():.3f})")

    # 4. Fit model
    print("[4/5] Training LightGBM...")
    model = SwarmLightGBM(LGBM_CONFIG)
    model.fit(X_train, y_train, X_val, y_val)
    print(f"      Best iteration: {model.best_iteration}")

    # 5. Tune threshold & evaluate
    print("[5/5] Evaluating on Validation and Test splits...")
    p_val = model.predict_proba(X_val)
    threshold = pick_optimal_threshold(y_val, p_val)
    model.threshold = threshold
    print(f"      Optimal validation-tuned threshold: {threshold:.4f}")

    val_eval = run_full_evaluation(meta_val, p_val, threshold, model_name="LightGBM")
    p_test = model.predict_proba(X_test)
    test_eval = run_full_evaluation(meta_test, p_test, threshold, model_name="LightGBM")

    print("\n----- TEST RESULTS -----")
    print(f"  PR-AUC:   {test_eval['overall']['pr_auc']:.4f}")
    print(f"  ROC-AUC:  {test_eval['overall']['roc_auc']:.4f}")
    print(f"  F1:       {test_eval['overall']['f1']:.4f}")
    print(f"  Recall:   {test_eval['overall']['recall']:.4f}")
    print(f"  Precision:{test_eval['overall']['precision']:.4f}")
    print(f"  MCC:      {test_eval['overall']['mcc']:.4f}")
    print(f"  Top-K Rec:{test_eval['top_k']['topk_recall']:.4f}")
    print(f"  Rec@90%P: {test_eval['overall']['recall_at_90_precision']:.4f}")
    print(f"  Agent F1: {test_eval['agent_level']['f1']:.4f}")

    # 6. Save artifacts
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir)
    save_evaluation_artifacts(val_eval, run_dir, tag="val")
    save_evaluation_artifacts(test_eval, run_dir, tag="test")

    # Save training history
    history = {
        "best_iteration": model.best_iteration,
        "train_time_sec": time.time() - t0,
        "n_train_samples": int(len(X_train)),
        "n_val_samples": int(len(X_val)),
        "n_test_samples": int(len(X_test)),
    }
    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save test predictions parquet
    pred_df = meta_test.copy()
    pred_df["prob"] = p_test
    pred_df["pred_class"] = (p_test >= threshold).astype(int)
    pred_df.to_parquet(run_dir / "test_predictions.parquet", index=False)

    print(f"\nAll LightGBM artifacts successfully saved to: {run_dir}")
    print(f"Elapsed time: {time.time() - t0:.1f}s")
    return test_eval


if __name__ == "__main__":
    train_lgbm_pipeline()
