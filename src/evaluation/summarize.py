"""Cross-model summary generator for aerial swarm critical-node identification.

Aggregates test metrics across all four models:
- LightGBM
- GRU
- GATv2
- ST-GAT

Outputs:
- artifacts/comparison_summary.csv
- Console Markdown table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import RUNS_DIR, ARTIFACTS_DIR


def generate_comparison_summary() -> pd.DataFrame:
    """Generate cross-model comparison table from run artifacts."""
    model_dirs = [
        ("LightGBM", RUNS_DIR / "lgbm"),
        ("GRU", RUNS_DIR / "gru"),
        ("GATv2", RUNS_DIR / "gatv2"),
        ("ST-GAT", RUNS_DIR / "stgat"),
    ]

    records = []
    for model_name, path in model_dirs:
        test_file = path / "test_metrics.json"
        if not test_file.exists():
            continue

        with open(test_file, "r", encoding="utf-8") as f:
            tm = json.load(f)

        hist_file = path / "training_history.json"
        train_time = None
        if hist_file.exists():
            with open(hist_file, "r", encoding="utf-8") as f:
                hist = json.load(f)
                if isinstance(hist, dict) and "train_time_sec" in hist:
                    train_time = hist["train_time_sec"]
                elif isinstance(hist, dict) and "total_time_sec" in hist:
                    train_time = hist["total_time_sec"]
                elif isinstance(hist, list) and len(hist) > 0:
                    train_time = sum(h.get("epoch_time_sec", 0.0) for h in hist)

        overall = tm.get("overall", {})
        top_k = tm.get("top_k", {})
        agent_lvl = tm.get("agent_level", {})

        records.append({
            "Model": model_name,
            "PR-AUC": overall.get("pr_auc", 0.0),
            "ROC-AUC": overall.get("roc_auc", 0.0),
            "F1": overall.get("f1", 0.0),
            "Recall": overall.get("recall", 0.0),
            "Precision": overall.get("precision", 0.0),
            "MCC": overall.get("mcc", 0.0),
            "Top-K Recall": top_k.get("topk_recall", 0.0),
            "NDCG@K (Bin)": top_k.get("ndcg_binary_at_k", 0.0),
            "NDCG@K (Grad)": top_k.get("ndcg_graded_at_k", 0.0),
            "MAP@K": top_k.get("map_at_k", 0.0),
            "Recall@90%P": overall.get("recall_at_90_precision", 0.0),
            "Agent-Level F1": agent_lvl.get("f1", 0.0),
            "Train Time (s)": round(train_time, 1) if train_time is not None else "N/A",
        })

    df = pd.DataFrame(records)
    if not df.empty:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = ARTIFACTS_DIR / "comparison_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved cross-model comparison summary to: {csv_path}\n")

        # Format markdown table manually to avoid tabulate dependency
        headers = list(df.columns)
        col_formats = []
        for h in headers:
            col_formats.append(h)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            row_vals = []
            for h in headers:
                val = row[h]
                if isinstance(val, float):
                    row_vals.append(f"{val:.4f}")
                else:
                    row_vals.append(str(val))
            lines.append("| " + " | ".join(row_vals) + " |")
        md_table = "\n".join(lines)
        print(md_table)
    else:
        print("No completed model runs found to summarize.")

    return df


if __name__ == "__main__":
    generate_comparison_summary()

