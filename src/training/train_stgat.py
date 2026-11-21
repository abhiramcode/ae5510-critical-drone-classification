"""CLI entry point to train and evaluate the Spatio-Temporal GAT (ST-GAT) model.

Runs:
1. Load dataset & scenario partition
2. Compute derived features
3. Fit train-only scaler
4. Assemble T=8 graph sliding windows (train_stride=4, eval_stride=1)
5. Train shared GATv2 + per-node GRU with validation early stopping
6. Tune classification threshold on validation set
7. Evaluate fully on test split (stride=1)
8. Persist all artifacts
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
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    RUNS_DIR,
    STGAT_CONFIG,
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    EDGE_FEATURES_7,
    GLOBAL_FEATURES_6,
    DEVICE,
    RANDOM_SEED,
)
from src.data.loader import load_raw_dataset
from src.data.splitter import get_or_create_primary_split
from src.data.features import compute_derived_snapshot_features, extract_snapshot_graph
from src.data.scaler import SwarmFeatureScaler
from src.data.adapters import build_stgat_windows, STGATWindowItem
from src.models.stgat_model import SwarmSTGAT
from src.evaluation.metrics import pick_optimal_threshold
from src.evaluation.evaluate import run_full_evaluation, save_evaluation_artifacts


def train_stgat_pipeline(run_dir: Path = RUNS_DIR / "stgat") -> dict:
    """Train and evaluate ST-GAT end-to-end."""
    torch.set_num_threads(8)
    print("=" * 60, flush=True)
    print("STAGE 9: Spatio-Temporal Graph Neural Network (ST-GAT) Training", flush=True)
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

    # 3. Fit scaler
    print("[3/6] Fitting train-only feature scaler...", flush=True)
    case_ids = df_feat["case_id"].astype(str).to_numpy()
    anomalies = df_feat["is_anomaly"].to_numpy()
    train_clean_mask = np.isin(case_ids, list(train_ids)) & (~anomalies)

    train_df = df_feat[train_clean_mask].copy()
    val_df = df_feat[np.isin(case_ids, list(val_ids))].copy()
    test_df = df_feat[np.isin(case_ids, list(test_ids))].copy()

    # Fit scaler on sample training snapshots
    sample_snaps = [
        grp for _, grp in list(train_df.groupby(["case_id", "step"], observed=True))[:200]
    ]
    all_ea = []
    all_g = []
    for s_df in sample_snaps:
        _, _, ea, gf = extract_snapshot_graph(s_df)
        all_ea.append(ea)
        all_g.append(gf)

    scaler = SwarmFeatureScaler()
    scaler.fit_nodes(train_df[BASE_FEATURES_22].to_numpy(dtype=np.float32))
    scaler.fit_edges(np.concatenate(all_ea, axis=0))
    scaler.fit_globals(np.stack(all_g, axis=0))

    import gc
    del df_raw, df_feat, sample_snaps, all_ea, all_g
    gc.collect()

    # 4. Assemble spatio-temporal sliding windows (T=8)
    print("[4/6] Assembling T=8 graph sliding windows (train stride=4, val stride=4)...", flush=True)
    windows_train = build_stgat_windows(
        train_df,
        scaler,
        window_steps=STGAT_CONFIG["temporal_window"],
        stride=4,
    )
    windows_val = build_stgat_windows(
        val_df,
        scaler,
        window_steps=STGAT_CONFIG["temporal_window"],
        stride=4,
    )

    del train_df, val_df
    gc.collect()

    print(f"      Train windows: {len(windows_train)}, Val windows: {len(windows_val)}", flush=True)

    # 5. Initialize model, optimizer, loss
    print(f"[5/6] Initializing ST-GAT model on device: {DEVICE}...", flush=True)
    model = SwarmSTGAT(
        in_channels=STGAT_CONFIG["in_channels"],
        hidden_size=STGAT_CONFIG["hidden_size"],
        edge_dim=STGAT_CONFIG["edge_dim"],
        num_heads=STGAT_CONFIG["num_heads"],
        num_layers=STGAT_CONFIG["num_layers"],
        global_dim=STGAT_CONFIG["global_dim"],
        temporal_hidden=STGAT_CONFIG["temporal_hidden"],
        spatial_dropout=STGAT_CONFIG["spatial_dropout"],
    ).to(DEVICE)

    # Compute positive weight across train windows
    all_tr_y = []
    for w in windows_train:
        eval_idx = torch.where(w.target_mask)[0]
        all_tr_y.append(w.y_end[eval_idx])
    all_tr_y_tensor = torch.cat(all_tr_y) if all_tr_y else torch.tensor([0.4])
    pos_weight = torch.tensor(
        float((len(all_tr_y_tensor) - all_tr_y_tensor.sum()) / all_tr_y_tensor.sum().clamp(min=1))
    ).to(DEVICE)

    raw_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    batch_size = STGAT_CONFIG["batch_size"]
    max_epochs = STGAT_CONFIG["max_epochs"]
    patience = STGAT_CONFIG["patience"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STGAT_CONFIG["learning_rate"],
        weight_decay=STGAT_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
    )

    def eval_windows(w_list: list[STGATWindowItem]) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        ps: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        with torch.no_grad():
            for w in w_list:
                logits, y_eval, _ = model.forward_window(w, device=DEVICE)
                if len(logits) > 0:
                    ps.append(torch.sigmoid(logits).cpu())
                    ys.append(y_eval.cpu())
        if not ps:
            return np.empty(0), np.empty(0)
        return torch.cat(ys).numpy(), torch.cat(ps).numpy()

    # Training loop with early stopping
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        total_loss = 0.0
        n_windows = 0

        # Shuffle windows
        rng = np.random.default_rng(RANDOM_SEED + epoch)
        order = rng.permutation(len(windows_train))

        optimizer.zero_grad()
        accum_loss = 0.0
        accum_count = 0

        for i, idx in enumerate(order):
            w = windows_train[idx]
            logits, y_eval, loss_mask = model.forward_window(w, device=DEVICE)
            if len(logits) == 0:
                continue

            loss_all = raw_criterion(logits, y_eval)
            loss = (loss_all * loss_mask.float()).sum() / loss_mask.sum().clamp(min=1)
            loss = loss / float(batch_size)
            loss.backward()

            accum_loss += float(loss.item()) * batch_size
            accum_count += 1

            if (i + 1) % batch_size == 0 or (i + 1) == len(order):
                nn.utils.clip_grad_norm_(model.parameters(), STGAT_CONFIG["clip_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accum_loss
                n_windows += accum_count
                accum_loss = 0.0
                accum_count = 0

        scheduler.step()
        gc.collect()
        ep_loss = total_loss / max(n_windows, 1)

        # Validation PR-AUC
        y_val, p_val = eval_windows(windows_val)
        val_pr_auc = float(average_precision_score(y_val, p_val))
        val_roc_auc = float(roc_auc_score(y_val, p_val))
        ep_time = time.time() - ep_t0

        print(
            f"[ST-GAT Epoch {epoch:2d}/{max_epochs:2d}] "
            f"Loss: {ep_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | "
            f"ROC-AUC: {val_roc_auc:.4f} ({ep_time:.1f}s)",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": ep_loss,
            "val_pr_auc": val_pr_auc,
            "val_roc_auc": val_roc_auc,
            "epoch_time_sec": ep_time,
        })

        if val_pr_auc > best_score:
            best_score = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[ST-GAT] Early stopping triggered at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 6. Tune threshold and run full evaluation
    print("[6/6] Tuning threshold and evaluating on test split...", flush=True)
    y_val, p_val = eval_windows(windows_val)
    threshold = pick_optimal_threshold(y_val, p_val)
    print(f"      Optimal validation-tuned threshold: {threshold:.4f}", flush=True)

    # Build meta_val dataframe
    meta_val_rows = []
    for w in windows_val:
        eval_idx = np.where(w.target_mask.numpy())[0]
        has_crit = hasattr(w, "criticality_scores") and w.criticality_scores is not None
        for u in eval_idx:
            row_dict = {
                "case_id": w.case_id,
                "step": w.step_end,
                "agent_id": int(w.agent_ids[u]),
                "num_agents": w.num_agents,
                "attrition_rate": w.attrition_rate,
                "is_critical": bool(w.y_end[u].item()),
                "is_anomaly": not bool(w.anomaly_mask[u].item()),
                "was_promoted": bool(w.was_promoted[u]),
            }
            if has_crit and w.criticality_scores is not None:
                row_dict["criticality_score"] = float(w.criticality_scores[u])
            meta_val_rows.append(row_dict)
    meta_val = pd.DataFrame(meta_val_rows)
    val_eval = run_full_evaluation(meta_val, p_val, threshold, model_name="ST-GAT")

    # Free training & validation windows before constructing test windows
    del windows_train, windows_val
    gc.collect()

    print("      Constructing test sliding windows (stride=4)...", flush=True)
    windows_test = build_stgat_windows(
        test_df,
        scaler,
        window_steps=STGAT_CONFIG["temporal_window"],
        stride=4,
    )
    y_test, p_test = eval_windows(windows_test)
    meta_test_rows = []
    for w in windows_test:
        eval_idx = np.where(w.target_mask.numpy())[0]
        has_crit = hasattr(w, "criticality_scores") and w.criticality_scores is not None
        for u in eval_idx:
            row_dict = {
                "case_id": w.case_id,
                "step": w.step_end,
                "agent_id": int(w.agent_ids[u]),
                "num_agents": w.num_agents,
                "attrition_rate": w.attrition_rate,
                "is_critical": bool(w.y_end[u].item()),
                "is_anomaly": not bool(w.anomaly_mask[u].item()),
                "was_promoted": bool(w.was_promoted[u]),
            }
            if has_crit and w.criticality_scores is not None:
                row_dict["criticality_score"] = float(w.criticality_scores[u])
            meta_test_rows.append(row_dict)
    meta_test = pd.DataFrame(meta_test_rows)
    test_eval = run_full_evaluation(meta_test, p_test, threshold, model_name="ST-GAT")

    print("\n----- ST-GAT TEST RESULTS -----")
    print(f"  PR-AUC:   {test_eval['overall']['pr_auc']:.4f}")
    print(f"  ROC-AUC:  {test_eval['overall']['roc_auc']:.4f}")
    print(f"  F1:       {test_eval['overall']['f1']:.4f}")
    print(f"  Recall:   {test_eval['overall']['recall']:.4f}")
    print(f"  Precision:{test_eval['overall']['precision']:.4f}")
    print(f"  MCC:      {test_eval['overall']['mcc']:.4f}")
    print(f"  Top-K Rec:{test_eval['top_k']['topk_recall']:.4f}")
    print(f"  Rec@90%P: {test_eval['overall']['recall_at_90_precision']:.4f}")
    print(f"  Agent F1: {test_eval['agent_level']['f1']:.4f}")

    # 7. Save artifacts
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save(run_dir, threshold=threshold)
    scaler.save(run_dir / "preprocessing.json")
    save_evaluation_artifacts(val_eval, run_dir, tag="val")
    save_evaluation_artifacts(test_eval, run_dir, tag="test")

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_pr_auc": best_score,
            "total_time_sec": time.time() - t0,
            "history": history,
        }, f, indent=2)

    pred_df = meta_test.copy()
    pred_df["prob"] = p_test
    pred_df["pred_class"] = (p_test >= threshold).astype(int)
    pred_df.to_parquet(run_dir / "test_predictions.parquet", index=False)

    print(f"\nAll ST-GAT artifacts successfully saved to: {run_dir}")
    print(f"Elapsed time: {time.time() - t0:.1f}s")
    return test_eval


if __name__ == "__main__":
    train_stgat_pipeline()
