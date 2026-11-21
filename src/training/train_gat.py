"""CLI entry point to train and evaluate the 3-layer GATv2 spatial graph neural network.

Runs:
1. Load dataset & scenario partition
2. Compute derived features
3. Fit train-only scaler for node, edge, and global features
4. Convert snapshots into PyG graph datasets
5. Train GATv2 with validation PR-AUC early stopping
6. Tune classification threshold on validation set
7. Evaluate fully on test split
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
from torch_geometric.loader import DataLoader

from src.config import (
    RUNS_DIR,
    GAT_CONFIG,
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    EDGE_FEATURES_7,
    GLOBAL_FEATURES_6,
    DEVICE,
    RANDOM_SEED,
)
from src.data.loader import load_raw_dataset
from src.data.splitter import get_or_create_primary_split
from src.data.features import compute_derived_snapshot_features
from src.data.scaler import SwarmFeatureScaler
from src.data.adapters import prepare_gat_graphs
from src.models.gat_model import SwarmGATv2
from src.training.trainer import PyTorchTrainer
from src.evaluation.metrics import pick_optimal_threshold
from src.evaluation.evaluate import run_full_evaluation, save_evaluation_artifacts


def train_gat_pipeline(run_dir: Path = RUNS_DIR / "gatv2") -> dict:
    """Train and evaluate GATv2 end-to-end."""
    torch.set_num_threads(8)
    print("=" * 60, flush=True)
    print("STAGE 8: 3-Layer GATv2 Spatial Graph Neural Network Training", flush=True)
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

    # 3. Partition and fit scaler
    print("[3/6] Fitting train-only feature scaler (nodes, edges, globals)...", flush=True)
    case_ids = df_feat["case_id"].astype(str).to_numpy()
    anomalies = df_feat["is_anomaly"].to_numpy()
    train_clean_mask = np.isin(case_ids, list(train_ids)) & (~anomalies)

    import gc
    train_df = df_feat[train_clean_mask].copy()
    val_df = df_feat[np.isin(case_ids, list(val_ids))].copy()
    test_df = df_feat[np.isin(case_ids, list(test_ids))].copy()

    # Sample snapshots from train to fit edge and global scalers
    sample_snaps = [
        grp for _, grp in list(train_df.groupby(["case_id", "step"], observed=True))[:200]
    ]
    all_ea = []
    all_g = []
    from src.data.features import extract_snapshot_graph
    for s_df in sample_snaps:
        _, _, ea, gf = extract_snapshot_graph(s_df)
        all_ea.append(ea)
        all_g.append(gf)

    scaler = SwarmFeatureScaler()
    scaler.fit_nodes(train_df[BASE_FEATURES_22].to_numpy(dtype=np.float32))
    scaler.fit_edges(np.concatenate(all_ea, axis=0))
    scaler.fit_globals(np.stack(all_g, axis=0))

    # Free raw and intermediate feature matrices to conserve RAM
    del df_raw, df_feat, sample_snaps
    gc.collect()

    # 4. Build PyG graph lists (stride=2 for memory safety and fast training)
    print("[4/6] Constructing PyG snapshot graphs (stride=2 on train/val)...", flush=True)
    train_strided = train_df[train_df["step"] % 4 == 0].copy()
    val_strided = val_df[val_df["step"] % 4 == 0].copy()

    graphs_train = prepare_gat_graphs(train_strided, scaler)
    graphs_val = prepare_gat_graphs(val_strided, scaler)

    del train_df, train_strided, val_strided
    gc.collect()

    batch_size = GAT_CONFIG["batch_size"]
    loader_train = DataLoader(graphs_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(graphs_val, batch_size=batch_size * 2, shuffle=False)

    print(f"      Train graphs: {len(graphs_train)}, Val graphs: {len(graphs_val)}", flush=True)

    # 5. Initialize model, optimizer, loss
    print(f"[5/6] Initializing GATv2 model on device: {DEVICE}...", flush=True)
    model = SwarmGATv2(
        in_channels=GAT_CONFIG["in_channels"],
        hidden_size=GAT_CONFIG["hidden_size"],
        edge_dim=GAT_CONFIG["edge_dim"],
        num_heads=GAT_CONFIG["num_heads"],
        num_layers=GAT_CONFIG["num_layers"],
        global_dim=GAT_CONFIG["global_dim"],
        dropout=GAT_CONFIG["dropout"],
    ).to(DEVICE)

    # Compute positive weight
    all_y = torch.cat([g.y for g in graphs_train])
    pos_weight = torch.tensor(float((len(all_y) - all_y.sum()) / all_y.sum())).to(DEVICE)
    raw_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GAT_CONFIG["learning_rate"],
        weight_decay=GAT_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=GAT_CONFIG["max_epochs"],
    )

    def train_epoch() -> float:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for b in loader_train:
            b = b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(b.x, b.edge_index, b.edge_attr, b.global_feat, b.batch)
            loss_all = raw_criterion(logits, b.y)
            loss = (loss_all * b.loss_mask.float()).sum() / b.loss_mask.sum().clamp(min=1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GAT_CONFIG["clip_grad_norm"])
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        gc.collect()
        return total_loss / max(n_batches, 1)

    def eval_fn() -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        ps: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        with torch.no_grad():
            for b in loader_val:
                b = b.to(DEVICE)
                logits = model(b.x, b.edge_index, b.edge_attr, b.global_feat, b.batch)
                ps.append(torch.sigmoid(logits).cpu())
                ys.append(b.y.cpu())
        return torch.cat(ys).numpy(), torch.cat(ps).numpy()

    trainer = PyTorchTrainer(
        model=model,
        optimizer=optimizer,
        criterion=raw_criterion,
        scheduler=scheduler,
        patience=GAT_CONFIG["patience"],
        clip_grad_norm=GAT_CONFIG["clip_grad_norm"],
        device=DEVICE,
    )

    train_res = trainer.fit(
        train_fn=train_epoch,
        eval_fn=eval_fn,
        max_epochs=GAT_CONFIG["max_epochs"],
        desc="GATv2",
    )

    # 6. Evaluation on Val and Test
    print("[6/6] Evaluating GATv2 model on Validation and Test splits...", flush=True)
    y_val, p_val = eval_fn()
    threshold = pick_optimal_threshold(y_val, p_val)
    print(f"      Optimal validation-tuned threshold: {threshold:.4f}", flush=True)

    meta_val_rows = []
    for g in graphs_val:
        n_g = len(g.y)
        has_crit = hasattr(g, "criticality_score") and g.criticality_score is not None
        for i in range(n_g):
            row_dict = {
                "case_id": g.case_id,
                "step": g.step,
                "agent_id": int(g.agent_id[i].item()),
                "num_agents": g.num_agents,
                "attrition_rate": g.attrition_rate,
                "is_critical": bool(g.y[i].item()),
                "is_anomaly": not bool(g.loss_mask[i].item()),
                "was_promoted": bool(g.was_promoted[i].item()),
            }
            if has_crit:
                row_dict["criticality_score"] = float(g.criticality_score[i].item())
            meta_val_rows.append(row_dict)
    meta_val = pd.DataFrame(meta_val_rows)
    val_eval = run_full_evaluation(meta_val, p_val, threshold, model_name="GATv2")

    # Free training/validation graphs before building test graphs
    del graphs_train, loader_train, graphs_val, loader_val
    gc.collect()

    # Build and evaluate test graphs
    print("      Constructing test snapshot graphs...", flush=True)
    graphs_test = prepare_gat_graphs(test_df, scaler)
    loader_test = DataLoader(graphs_test, batch_size=32, shuffle=False)

    model.eval()
    ps_te: list[torch.Tensor] = []
    with torch.no_grad():
        for b in loader_test:
            b = b.to(DEVICE)
            logits = model(b.x, b.edge_index, b.edge_attr, b.global_feat, b.batch)
            ps_te.append(torch.sigmoid(logits).cpu())
    p_test = torch.cat(ps_te).numpy()

    meta_test_rows = []
    for g in graphs_test:
        n_g = len(g.y)
        has_crit = hasattr(g, "criticality_score") and g.criticality_score is not None
        for i in range(n_g):
            row_dict = {
                "case_id": g.case_id,
                "step": g.step,
                "agent_id": int(g.agent_id[i].item()),
                "num_agents": g.num_agents,
                "attrition_rate": g.attrition_rate,
                "is_critical": bool(g.y[i].item()),
                "is_anomaly": not bool(g.loss_mask[i].item()),
                "was_promoted": bool(g.was_promoted[i].item()),
            }
            if has_crit:
                row_dict["criticality_score"] = float(g.criticality_score[i].item())
            meta_test_rows.append(row_dict)
    meta_test = pd.DataFrame(meta_test_rows)
    test_eval = run_full_evaluation(meta_test, p_test, threshold, model_name="GATv2")

    print("\n----- GATv2 TEST RESULTS -----")
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
        json.dump(train_res, f, indent=2)

    pred_df = meta_test.copy()
    pred_df["prob"] = p_test
    pred_df["pred_class"] = (p_test >= threshold).astype(int)
    pred_df.to_parquet(run_dir / "test_predictions.parquet", index=False)

    print(f"\nAll GATv2 artifacts successfully saved to: {run_dir}")
    print(f"Elapsed time: {time.time() - t0:.1f}s")
    return test_eval


if __name__ == "__main__":
    train_gat_pipeline()
