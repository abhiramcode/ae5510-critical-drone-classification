"""Comprehensive metric computation and threshold selection.

Computes:
- Primary: PR-AUC, ROC-AUC, F1, Precision, Recall, MCC, Balanced Accuracy
- Operational: Recall at >=90% Precision, False Negative Rate, Top-K Precision/Recall
- Validation-based threshold tuning
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_recall_curve,
)


def pick_optimal_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    num_steps: int = 97,
) -> float:
    """Select the operating classification threshold that maximizes F1 on validation predictions."""
    y = np.asarray(y_val, dtype=bool)
    p = np.asarray(p_val, dtype=np.float32)

    thresholds = np.linspace(0.02, 0.98, num_steps)
    f1_scores = np.array([f1_score(y, p >= t, zero_division=0) for t in thresholds])
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def compute_core_metrics(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute primary and operational classification metrics at a fixed operating threshold."""
    y = np.asarray(y_true, dtype=bool)
    p = np.asarray(p_pred, dtype=np.float32)
    n = len(y)

    if n == 0 or y.sum() == 0 or y.sum() == n:
        return {
            "n": float(n),
            "pos_rate": float(y.mean()) if n > 0 else 0.0,
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "threshold": float(threshold),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "mcc": float("nan"),
            "balanced_accuracy": float("nan"),
            "recall_at_90_precision": float("nan"),
            "fnr": float("nan"),
        }

    preds = p >= threshold

    # Recall at >= 90% precision
    prec_curve, rec_curve, _ = precision_recall_curve(y, p)
    qualifying_recalls = rec_curve[prec_curve >= 0.90]
    rec_at_90p = float(np.max(qualifying_recalls)) if len(qualifying_recalls) > 0 else 0.0

    rec = float(recall_score(y, preds, zero_division=0))

    return {
        "n": float(n),
        "pos_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "threshold": float(threshold),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": rec,
        "mcc": float(matthews_corrcoef(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "recall_at_90_precision": rec_at_90p,
        "fnr": 1.0 - rec,
    }


def compute_ranking_metrics(
    df_meta: pd.DataFrame,
    p_pred: np.ndarray,
) -> dict[str, float]:
    """Compute Top-K, NDCG (binary & graded), MAP@K, MRR, and rank correlation per snapshot."""
    eval_dict = {
        "case_id": df_meta["case_id"].to_numpy(),
        "step": df_meta["step"].to_numpy(),
        "y": df_meta["is_critical"].to_numpy(dtype=bool),
        "p": np.asarray(p_pred, dtype=np.float32),
    }
    has_crit_score = "criticality_score" in df_meta.columns
    if has_crit_score:
        eval_dict["c"] = df_meta["criticality_score"].to_numpy(dtype=np.float32)

    df_eval = pd.DataFrame(eval_dict)

    recalls: list[float] = []
    precisions: list[float] = []
    ndcg_bin_list: list[float] = []
    ndcg_grad_list: list[float] = []
    map_list: list[float] = []
    mrr_list: list[float] = []
    spearman_list: list[float] = []

    # Fixed K = 5 operational salvo budget metrics
    k_salvo = 5
    precs_5: list[float] = []
    recalls_5: list[float] = []
    ndcg_bin_5_list: list[float] = []
    ndcg_grad_5_list: list[float] = []

    for _, grp in df_eval.groupby(["case_id", "step"], observed=True):
        y = grp["y"].to_numpy(dtype=bool)
        k = int(y.sum())
        if k == 0 or len(grp) < 2:
            continue

        p = grp["p"].to_numpy(dtype=np.float32)
        order = np.argsort(p)[::-1]
        y_sorted = y[order]

        # 1. Top-K Recall & Precision (set overlap, dynamic K = true count)
        hits = int(y_sorted[:k].sum())
        recalls.append(float(hits) / float(k))
        precisions.append(float(hits) / float(k))

        # Position discount: 1 / log2(rank + 1) for rank = 1..k -> log2(2..k+1)
        discounts = 1.0 / np.log2(np.arange(2, k + 2))

        # 2. Binary NDCG@K (order-sensitive on binary ground truth)
        dcg_bin = float(np.sum((2.0**y_sorted[:k].astype(float) - 1.0) * discounts))
        idcg_bin = float(np.sum((2.0**1.0 - 1.0) * discounts))  # ideal ranking has all 1s at top
        ndcg_bin_list.append(dcg_bin / idcg_bin if idcg_bin > 0 else 1.0)

        # 3. MAP@K (Mean Average Precision at K)
        p_at_i = np.cumsum(y_sorted[:k]) / np.arange(1, k + 1)
        ap = float(np.sum(p_at_i * y_sorted[:k]) / float(k))
        map_list.append(ap)

        # 4. MRR (Mean Reciprocal Rank across whole snapshot)
        pos = np.where(y_sorted)[0]
        if len(pos) > 0:
            mrr_list.append(1.0 / float(pos[0] + 1))

        # 5. Graded NDCG@K & Spearman rho (using continuous criticality_score if present)
        if has_crit_score:
            c = grp["c"].to_numpy(dtype=np.float32)
            c_sorted = c[order]
            dcg_grad = float(np.sum((2.0**c_sorted[:k] - 1.0) * discounts))
            ideal_c = np.sort(c)[::-1][:k]
            idcg_grad = float(np.sum((2.0**ideal_c - 1.0) * discounts))
            ndcg_grad_list.append(dcg_grad / idcg_grad if idcg_grad > 0 else 1.0)

            p_rank = np.argsort(np.argsort(p)).astype(float)
            c_rank = np.argsort(np.argsort(c)).astype(float)
            n_snap = len(grp)
            d = p_rank - c_rank
            rho = 1.0 - (6.0 * np.sum(d**2)) / (n_snap * (n_snap**2 - 1.0))
            spearman_list.append(float(rho))

        # 6. Fixed Salvo Budget K = 5 (Operational firing doctrine)
        if len(grp) >= k_salvo:
            hits_5 = int(y_sorted[:k_salvo].sum())
            precs_5.append(hits_5 / k_salvo)
            recalls_5.append(hits_5 / k)
            discounts_5 = 1.0 / np.log2(np.arange(2, k_salvo + 2))
            dcg_bin_5 = float(np.sum((2.0**y_sorted[:k_salvo].astype(float) - 1.0) * discounts_5))
            ideal_5 = np.sort(y.astype(float))[::-1][:k_salvo]
            idcg_bin_5 = float(np.sum((2.0**ideal_5 - 1.0) * discounts_5))
            ndcg_bin_5_list.append(dcg_bin_5 / idcg_bin_5 if idcg_bin_5 > 0 else 1.0)
            if has_crit_score:
                dcg_grad_5 = float(np.sum((2.0**c_sorted[:k_salvo] - 1.0) * discounts_5))
                ideal_c_5 = np.sort(c)[::-1][:k_salvo]
                idcg_grad_5 = float(np.sum((2.0**ideal_c_5 - 1.0) * discounts_5))
                ndcg_grad_5_list.append(dcg_grad_5 / idcg_grad_5 if idcg_grad_5 > 0 else 1.0)

    out: dict[str, float] = {
        "topk_recall": float(np.mean(recalls)) if recalls else 0.0,
        "topk_precision": float(np.mean(precisions)) if precisions else 0.0,
        "ndcg_binary_at_k": float(np.mean(ndcg_bin_list)) if ndcg_bin_list else 0.0,
        "map_at_k": float(np.mean(map_list)) if map_list else 0.0,
        "mrr": float(np.mean(mrr_list)) if mrr_list else 0.0,
        "precision_at_5": float(np.mean(precs_5)) if precs_5 else 0.0,
        "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
        "ndcg_binary_at_5": float(np.mean(ndcg_bin_5_list)) if ndcg_bin_5_list else 0.0,
    }
    if has_crit_score and ndcg_grad_list:
        out["ndcg_graded_at_k"] = float(np.mean(ndcg_grad_list))
        out["spearman_rho"] = float(np.mean(spearman_list))
        out["ndcg_graded_at_5"] = float(np.mean(ndcg_grad_5_list)) if ndcg_grad_5_list else 0.0

    return out


# Backward-compatible alias
compute_topk_metrics = compute_ranking_metrics

