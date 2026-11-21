"""Unified evaluation harness for all four critical-node identification models.

Generates:
- Primary and operational metrics at validation-tuned threshold
- Full breakdowns: overall, clean-only, per-size, per-attrition, promotion subsets,
  mission tertiles, t=0 cross-section, and agent-level aggregations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_core_metrics,
    compute_topk_metrics,
    compute_ranking_metrics,
    pick_optimal_threshold,
)


def run_full_evaluation(
    df_meta: pd.DataFrame,
    p_pred: np.ndarray,
    threshold: float,
    model_name: str = "model",
) -> dict[str, Any]:
    """Execute complete standardized evaluation and compute all required breakdowns."""
    df_eval = df_meta.copy().reset_index(drop=True)
    df_eval["score"] = np.asarray(p_pred, dtype=np.float32)
    y_all = df_eval["is_critical"].to_numpy(dtype=bool)
    p_all = df_eval["score"].to_numpy(dtype=np.float32)

    results: dict[str, Any] = {
        "model_name": model_name,
        "operating_threshold": float(threshold),
        "overall": compute_core_metrics(y_all, p_all, threshold),
        "overall_clean_only": compute_core_metrics(
            df_eval.loc[~df_eval["is_anomaly"], "is_critical"].to_numpy(),
            df_eval.loc[~df_eval["is_anomaly"], "score"].to_numpy(),
            threshold,
        ),
        "top_k": compute_ranking_metrics(df_eval, p_all),
        "ranking": compute_ranking_metrics(df_eval, p_all),
    }

    # 1. Swarm size breakdown
    by_size: dict[str, Any] = {}
    for sz, grp in df_eval.groupby("num_agents", observed=True):
        by_size[str(sz)] = compute_core_metrics(
            grp["is_critical"].to_numpy(), grp["score"].to_numpy(), threshold
        )
    results["by_swarm_size"] = by_size

    # 2. Attrition breakdown
    by_attr: dict[str, Any] = {}
    for attr, grp in df_eval.groupby("attrition_rate", observed=True):
        by_attr[str(attr)] = compute_core_metrics(
            grp["is_critical"].to_numpy(), grp["score"].to_numpy(), threshold
        )
    results["by_attrition"] = by_attr

    # 3. Promotion breakdown
    # Identify agents that ever experienced a promotion in this scenario
    promo_agent_pairs = set(
        map(
            tuple,
            df_eval.loc[df_eval["was_promoted"], ["case_id", "agent_id"]].to_numpy()
        )
    )
    is_promo_agent = np.array([
        (c, a) in promo_agent_pairs
        for c, a in zip(df_eval["case_id"].to_numpy(), df_eval["agent_id"].to_numpy())
    ])

    df_eval["is_promoted_agent"] = is_promo_agent

    # never_promoted: all rows of agents that never promote
    grp_never = df_eval[~df_eval["is_promoted_agent"]]
    results["promotion_never_promoted"] = compute_core_metrics(
        grp_never["is_critical"].to_numpy(), grp_never["score"].to_numpy(), threshold
    )

    # pre_promotion: rows of promoted agents strictly before promotion (label = 0)
    grp_pre = df_eval[df_eval["is_promoted_agent"] & (~df_eval["was_promoted"])]
    results["promotion_pre_event"] = {
        "n": int(len(grp_pre)),
        "mean_score": float(grp_pre["score"].mean()) if len(grp_pre) > 0 else 0.0,
        "fpr": float((grp_pre["score"] >= threshold).mean()) if len(grp_pre) > 0 else 0.0,
    }

    # post_promotion: rows of promoted agents at/after promotion (label = 1)
    grp_post = df_eval[df_eval["is_promoted_agent"] & df_eval["was_promoted"]]
    results["promotion_post_event"] = {
        "n": int(len(grp_post)),
        "mean_score": float(grp_post["score"].mean()) if len(grp_post) > 0 else 0.0,
        "recall": float((grp_post["score"] >= threshold).mean()) if len(grp_post) > 0 else 0.0,
    }

    # 4. Mission progress tertiles and t=0 cross-section
    # Compute relative progress per scenario [0, 1]
    g_scen = df_eval.groupby("case_id", observed=True)
    step_min = g_scen["step"].transform("min")
    step_max = g_scen["step"].transform("max")
    rel_step = (df_eval["step"] - step_min) / (step_max - step_min).clip(lower=1)

    tertiles = pd.cut(
        rel_step,
        bins=[-0.01, 0.3333, 0.6667, 1.01],
        labels=["early", "mid", "late"],
    )
    df_eval["tertile"] = tertiles

    by_period: dict[str, Any] = {}
    for period, grp in df_eval.groupby("tertile", observed=True):
        by_period[str(period)] = compute_core_metrics(
            grp["is_critical"].to_numpy(), grp["score"].to_numpy(), threshold
        )
    # t = 0 cross-section
    grp_t0 = df_eval[df_eval["step"] == 0]
    by_period["t0_cross_section"] = compute_core_metrics(
        grp_t0["is_critical"].to_numpy(), grp_t0["score"].to_numpy(), threshold
    )
    results["by_mission_period"] = by_period

    # 5. Agent-level aggregation (mean probability over whole scenario per agent)
    agent_df = (
        df_eval.groupby(["case_id", "agent_id"], observed=True)
        .agg(
            score=("score", "mean"),
            is_critical=("is_critical", "mean"),
        )
        .reset_index()
    )
    y_agent = (agent_df["is_critical"] > 0.5).to_numpy(dtype=bool)
    p_agent = agent_df["score"].to_numpy(dtype=np.float32)
    thr_agent = pick_optimal_threshold(y_agent, p_agent)
    results["agent_level"] = compute_core_metrics(y_agent, p_agent, thr_agent)

    return results


def save_evaluation_artifacts(
    eval_results: dict[str, Any],
    out_dir: Path,
    tag: str = "test",
) -> None:
    """Save evaluation metrics to JSON and export readable summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}_metrics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=float)

