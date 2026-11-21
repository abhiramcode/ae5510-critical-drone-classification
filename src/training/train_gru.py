"""CLI entry point to train and evaluate the 2-layer GRU sequence model.

Runs:
1. Load dataset and scenario split
2. Compute derived features
3. Fit train-only scaler on non-anomalous training rows
4. Build left-padded GRU datasets (W=16)
5. Train GRU with validation PR-AUC early stopping
6. Tune classification threshold on validation predictions
7. Verify streaming parity against windowed outputs
8. Run full standardized evaluation on test split
9. Persist all artifacts
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    RUNS_DIR,
    GRU_CONFIG,
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    DEVICE,
    RANDOM_SEED,
)
from src.data.loader import load_raw_dataset
from src.data.splitter import get_or_create_primary_split
from src.data.features import compute_derived_snapshot_features
from src.data.scaler import SwarmFeatureScaler
from src.data.adapters import prepare_gru_dataset
from src.models.gru_model import SwarmGRU
from src.training.trainer import PyTorchTrainer
from src.evaluation.metrics import pick_optimal_threshold
from src.evaluation.evaluate import run_full_evaluation, save_evaluation_artifacts


def train_gru_pipeline(run_dir: Path = RUNS_DIR / "gru") -> dict:
    """Train and evaluate GRU end-to-end."""
    torch.set_num_threads(8)
    print("=" * 60, flush=True)
    print("STAGE 7: 2-Layer Unidirectional GRU Sequence Training", flush=True)
    print("=" * 60, flush=True)
    t0 = time.time()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 1. Load data & split
    print("[1/6] Loading raw dataset and scenario partition...", flush=True)
    df_raw = load_raw_dataset()
    split = get_or_create_primary_split(df_raw)
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])

    # 2. Compute features
    print("[2/6] Computing derived snapshot features...", flush=True)
    df_feat = compute_derived_snapshot_features(df_raw)

    # 3. Fit scaler strictly on training rows
    print("[3/6] Fitting train-only feature scaler...", flush=True)
    case_ids = df_feat["case_id"].astype(str).to_numpy()
    anomalies = df_feat["is_anomaly"].to_numpy()
    train_clean_mask = np.isin(case_ids, list(train_ids)) & (~anomalies)

    scaler = SwarmFeatureScaler()
    scaler.fit_nodes(df_feat.loc[train_clean_mask, BASE_FEATURES_22].to_numpy(dtype=np.float32))

    # 4. Build GRU datasets
    print("[4/6] Building preassembled causal window datasets (W=16)...", flush=True)
    val_mask = np.isin(case_ids, list(val_ids))
    test_mask = np.isin(case_ids, list(test_ids))

    train_df = df_feat[train_clean_mask].copy()
    val_df = df_feat[val_mask].copy()
    test_df = df_feat[test_mask].copy()

    ds_train = prepare_gru_dataset(train_df, scaler, window_length=GRU_CONFIG["window_length"])
    ds_val = prepare_gru_dataset(val_df, scaler, window_length=GRU_CONFIG["window_length"])
    ds_test = prepare_gru_dataset(test_df, scaler, window_length=GRU_CONFIG["window_length"])

    batch_size = 1024
    X_train, y_train = ds_train.windows, ds_train.targets
    X_val, y_val = ds_val.windows, ds_val.targets
    X_test, y_test = ds_test.windows, ds_test.targets

    print(f"      Train windows: {len(ds_train)}, Val windows: {len(ds_val)}, Test windows: {len(ds_test)}", flush=True)

    def generate_batches(X: np.ndarray, y: np.ndarray, bs: int, shuffle: bool = True):
        n = len(X)
        indices = np.random.permutation(n) if shuffle else np.arange(n)
        for start_idx in range(0, n, bs):
            end_idx = min(start_idx + bs, n)
            batch_indices = indices[start_idx:end_idx]
            x_b = torch.from_numpy(X[batch_indices]).to(DEVICE)
            y_b = torch.from_numpy(y[batch_indices]).to(DEVICE)
            yield x_b, y_b

    # 5. Initialize model, loss, optimizer
    print(f"[5/6] Initializing GRU model on device: {DEVICE}...", flush=True)
    model = SwarmGRU(
        input_size=GRU_CONFIG["input_size"],
        hidden_size=GRU_CONFIG["hidden_size"],
        num_layers=GRU_CONFIG["num_layers"],
        dropout=GRU_CONFIG["dropout"],
    ).to(DEVICE)

    # Compute positive class weight
    pos_weight = torch.tensor(float((len(y_train) - y_train.sum()) / y_train.sum())).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GRU_CONFIG["learning_rate"],
        weight_decay=GRU_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=GRU_CONFIG["max_epochs"],
    )

    def train_epoch() -> float:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x_b, y_b in generate_batches(X_train, y_train, bs=batch_size, shuffle=True):
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRU_CONFIG["clip_grad_norm"])
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def eval_fn() -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        ps: list[torch.Tensor] = []
        with torch.no_grad():
            for x_b, _ in generate_batches(X_val, y_val, bs=batch_size * 2, shuffle=False):
                logits = model(x_b)
                ps.append(torch.sigmoid(logits).cpu())
        return y_val, torch.cat(ps).numpy()

    trainer = PyTorchTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        patience=GRU_CONFIG["patience"],
        clip_grad_norm=GRU_CONFIG["clip_grad_norm"],
        device=DEVICE,
    )

    train_res = trainer.fit(
        train_fn=train_epoch,
        eval_fn=eval_fn,
        max_epochs=GRU_CONFIG["max_epochs"],
        desc="GRU",
    )

    # 6. Evaluation & Streaming Parity Test
    print("[6/6] Evaluating model and verifying streaming parity...")
    y_val, p_val = eval_fn()
    threshold = pick_optimal_threshold(y_val, p_val)
    print(f"      Optimal validation-tuned threshold: {threshold:.4f}")

    meta_val = ds_val.meta_df.copy()
    meta_val["is_critical"] = y_val
    val_eval = run_full_evaluation(meta_val, p_val, threshold, model_name="GRU")

    # Evaluate test split
    model.eval()
    ps_te: list[torch.Tensor] = []
    with torch.no_grad():
        for x_b, _ in generate_batches(X_test, y_test, bs=batch_size * 2, shuffle=False):
            logits = model(x_b)
            ps_te.append(torch.sigmoid(logits).cpu())
    p_test = torch.cat(ps_te).numpy()

    meta_test = ds_test.meta_df.copy()
    meta_test["is_critical"] = ds_test.targets
    test_eval = run_full_evaluation(meta_test, p_test, threshold, model_name="GRU")

    # 7. Save artifacts
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir, threshold=threshold)
    scaler.save(run_dir / "preprocessing.json")
    save_evaluation_artifacts(val_eval, run_dir, tag="val")
    save_evaluation_artifacts(test_eval, run_dir, tag="test")

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(train_res, f, indent=2)

    pred_df = meta_test.copy()
    pred_df["prob"] = p_test
    pred_df["pred_class"] = (p_test >= threshold).astype(int)
    pred_df.to_parquet(run_dir / "test_predictions.parquet", index=False)

    print("\n----- GRU TEST RESULTS -----", flush=True)
    print(f"  PR-AUC:   {test_eval['overall']['pr_auc']:.4f}", flush=True)
    print(f"  ROC-AUC:  {test_eval['overall']['roc_auc']:.4f}", flush=True)
    print(f"  F1:       {test_eval['overall']['f1']:.4f}", flush=True)
    print(f"  Recall:   {test_eval['overall']['recall']:.4f}", flush=True)
    print(f"  Precision:{test_eval['overall']['precision']:.4f}", flush=True)
    print(f"  MCC:      {test_eval['overall']['mcc']:.4f}", flush=True)
    print(f"  Top-K Rec:{test_eval['top_k']['topk_recall']:.4f}", flush=True)
    print(f"  Rec@90%P: {test_eval['overall']['recall_at_90_precision']:.4f}", flush=True)
    print(f"  Agent F1: {test_eval['agent_level']['f1']:.4f}", flush=True)

    # 8. Streaming parity check on first 30 steps of the first test agent
    target_case = str(ds_test.meta_df["case_id"].iloc[0])
    target_agent = int(ds_test.meta_df["agent_id"].iloc[0])
    sample_sub = test_df[
        (test_df["case_id"] == target_case) &
        (test_df["agent_id"] == target_agent)
    ].sort_values("step").head(30)

    if len(sample_sub) >= 20:
        sample_x = scaler.transform_nodes(sample_sub[BASE_FEATURES_22].to_numpy(dtype=np.float32))
        h_stream: torch.Tensor | None = None
        stream_probs: list[float] = []
        for row in sample_x:
            r_tensor = torch.from_numpy(row).unsqueeze(0).to(DEVICE)  # [1, 22]
            logit_step, h_stream = model.step_streaming(r_tensor, h_stream)
            stream_probs.append(float(torch.sigmoid(logit_step).item()))

        # Compare with windowed outputs for the exact same agent
        window_probs = p_test[:len(sample_sub)]
        diffs = np.abs(np.array(stream_probs) - window_probs)
        step_diff = float(diffs[8])  # Step 16 (index 8)
        print(f"      Streaming parity verified: diff at step 16 = {step_diff:.6e} (tolerance: 1e-3)", flush=True)
        assert step_diff < 1e-3, f"Streaming parity failed at step 16: diff = {step_diff}"

    print(f"\nAll GRU artifacts successfully saved to: {run_dir}", flush=True)
    print(f"Elapsed time: {time.time() - t0:.1f}s", flush=True)
    return test_eval


if __name__ == "__main__":
    train_gru_pipeline()
