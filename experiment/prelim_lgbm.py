"""Preliminary leak-free baselines + leakage quantification + ablations.

Scenario-grouped split. `criticality_score`, `was_promoted`, `is_critical`, `case_id`
and `agent_id` are never model inputs (except the deliberately leaky reference runs).
The validation threshold is selected on validation only and applied to test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(exist_ok=True)

USE = ["case_id", "num_agents", "n_obstacles", "obstacle_density", "mean_obstacle_size",
       "clutter_index", "obstacle_clustering", "attrition_rate", "step", "agent_id",
       "rel_z", "dist_to_centroid", "rank_dist_to_centroid", "dist_to_target",
       "rank_dist_to_target", "speed", "heading", "flight_path_angle", "accel",
       "heading_rate", "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
       "weighted_density", "angular_coverage", "criticality_score", "degree_centrality",
       "closeness_centrality", "betweenness_centrality", "eigenvector_centrality",
       "is_critical", "was_promoted"]
DT = {"case_id": "category", "attrition_rate": "category", "agent_id": "int16",
      "step": "int16", "num_agents": "int16", "n_obstacles": "int16",
      "num_neighbors": "int16"}
df = pd.read_csv(CSV, usecols=USE, dtype=DT)
df = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)
df["attrition_ord"] = df.attrition_rate.map({"none": 0, "light": 1, "medium": 2, "heavy": 3}).astype("int8")

# ---------- causal derived features ----------
g = df.groupby(["case_id", "agent_id"], observed=True)
df["heading_sin"] = np.sin(df.heading)
df["heading_cos"] = np.cos(df.heading)
df["alive_count"] = df.groupby(["case_id", "step"], observed=True).agent_id.transform("size")
df["neighbor_frac"] = df.num_neighbors / (df.alive_count - 1).clip(lower=1)
df["is_isolated"] = (df.num_neighbors == 0).astype("int8")
for src, dst in [("dist_to_centroid", "d1_dist"), ("rank_dist_to_centroid", "d1_rank"),
                 ("num_neighbors", "d1_nb"), ("weighted_density", "d1_dens"),
                 ("angular_coverage", "d1_ang"), ("speed", "d1_speed")]:
    df[dst] = g[src].diff()

BASE = ["speed", "accel", "heading_rate", "flight_path_angle", "heading_sin", "heading_cos",
        "dist_to_centroid", "rank_dist_to_centroid", "rel_z", "dist_to_target",
        "rank_dist_to_target", "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
        "weighted_density", "angular_coverage", "degree_centrality", "closeness_centrality",
        "betweenness_centrality", "eigenvector_centrality", "neighbor_frac", "is_isolated"]
ENV = ["n_obstacles", "obstacle_density", "mean_obstacle_size", "clutter_index",
       "obstacle_clustering", "num_agents", "attrition_ord"]
DELTAS = ["d1_dist", "d1_rank", "d1_nb", "d1_dens", "d1_ang", "d1_speed"]
print("features:", len(BASE), "base |", len(DELTAS), "deltas |", len(ENV), "env")


def stratified_split(frame, seed=0):
    meta = (frame.groupby("case_id", observed=True)
            .agg(size=("num_agents", "first"), attr=("attrition_rate", "first"))
            .reset_index())
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for _, grp in meta.groupby(["size", "attr"], observed=True):
        ids = grp.case_id.astype(str).tolist()
        rng.shuffle(ids)
        n = len(ids)
        if n >= 5:
            n_test, n_val = max(1, round(0.15 * n)), max(1, round(0.15 * n))
        elif n >= 3:
            n_test, n_val = 1, 1
        else:
            n_test = n_val = 0
        test += ids[:n_test]
        val += ids[n_test:n_test + n_val]
        train += ids[n_test + n_val:]
    return set(train), set(val), set(test)


tr, va, te = stratified_split(df, seed=0)
print(f"scenarios train/val/test = {len(tr)}/{len(va)}/{len(te)}")
trm = df.case_id.astype(str).isin(tr)
vam = df.case_id.astype(str).isin(va)
tem = df.case_id.astype(str).isin(te)
print("train sizes", sorted(df.loc[trm, "num_agents"].unique().tolist()))
print("val sizes  ", sorted(df.loc[vam, "num_agents"].unique().tolist()))
print("test sizes ", sorted(df.loc[tem, "num_agents"].unique().tolist()))


def metrics(y, p, thr, tag=""):
    y = np.asarray(y); p = np.asarray(p)
    out = {"n": int(len(y)), "pos_rate": float(y.mean()),
           "roc_auc": float(roc_auc_score(y, p)),
           "pr_auc": float(average_precision_score(y, p)),
           "threshold": float(thr),
           "f1": float(f1_score(y, p >= thr)),
           "precision": float(precision_score(y, p >= thr, zero_division=0)),
           "recall": float(recall_score(y, p >= thr)),
           "mcc": float(matthews_corrcoef(y, p >= thr))}
    print(f"{tag}n={out['n']} pos={out['pos_rate']:.3f} PR-AUC={out['pr_auc']:.4f} "
          f"ROC-AUC={out['roc_auc']:.4f} thr={thr:.2f} F1={out['f1']:.4f} "
          f"P={out['precision']:.3f} R={out['recall']:.3f} MCC={out['mcc']:.3f}")
    return out


def pick_threshold(y, p):
    ths = np.linspace(0.02, 0.98, 97)
    f1s = np.array([f1_score(y, p >= t) for t in ths])
    return float(ths[int(np.argmax(f1s))])


def topk_metrics(frame, scores):
    d = pd.DataFrame({"case": frame.case_id.astype(str).to_numpy(),
                      "step": frame.step.to_numpy(),
                      "y": frame.is_critical.to_numpy(), "p": scores})
    rec, pre = [], []
    for _, grp in d.groupby(["case", "step"], observed=True):
        k = int(grp.y.sum())
        if k == 0:
            continue
        order = grp.p.to_numpy().argsort()[::-1][:k]
        hits = int(grp.y.to_numpy()[order].sum())
        rec.append(hits / k)
        pre.append(hits / k)
    return float(np.mean(rec)), float(np.mean(pre))


def run(name, cols):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=127,
                           min_child_samples=60, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0,
                           class_weight="balanced", n_jobs=-1, random_state=0,
                           verbose=-1)
    m.fit(df.loc[trm, cols], df.loc[trm, "is_critical"],
          eval_set=[(df.loc[vam, cols], df.loc[vam, "is_critical"])],
          eval_metric="average_precision", callbacks=[lgb.early_stopping(50, verbose=False)])
    p_va = m.predict_proba(df.loc[vam, cols])[:, 1]
    y_va = df.loc[vam, "is_critical"].to_numpy()
    thr = pick_threshold(y_va, p_va)
    p_te = m.predict_proba(df.loc[tem, cols])[:, 1]
    y_te = df.loc[tem, "is_critical"].to_numpy()
    print(f"\n=== {name} | features={len(cols)} best_iter={m.best_iteration_} ===")
    res = {"name": name, "n_features": len(cols), "best_iter": int(m.best_iteration_ or 0),
           "val": metrics(y_va, p_va, thr, "val  "),
           "test": metrics(y_te, p_te, thr, "test ")}
    rec, pre = topk_metrics(df.loc[tem], p_te)
    res["test_topk"] = {"recall": rec, "precision": pre}
    print(f"test Top-K: recall={rec:.4f} precision={pre:.4f}")
    res["importance_gain"] = (pd.Series(m.booster_.feature_importance("gain"), index=cols)
                              .sort_values(ascending=False).round(1).head(20).to_dict())
    return res, m, p_te, thr


results = {}
results["static"], _, _, _ = run("A static (no history, no leak)", BASE + ENV)
results["temporal"], m_temp, p_temp, thr_temp = run("B static + 1-step deltas", BASE + ENV + DELTAS)
results["nocloseness"], _, _, _ = run(
    "C temporal minus closeness_centrality",
    [c for c in BASE if c != "closeness_centrality"] + ENV + DELTAS)
nog = [c for c in BASE if c not in
       ["degree_centrality", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality"]]
results["nograph"], _, _, _ = run("D temporal without graph centralities", nog + ENV + DELTAS)
results["leak"], _, _, _ = run("E LEAK reference (+criticality_score)",
                               BASE + ENV + DELTAS + ["criticality_score"])
results["leak_flag"], _, _, _ = run("F LEAK reference (+was_promoted)",
                                    BASE + ENV + DELTAS + ["was_promoted"])

te_df = df.loc[tem].copy()
te_df["score"] = p_temp
print("\n=== B temporal test by attrition ===")
by_attr = {}
for a, grp in te_df.groupby("attrition_rate", observed=True):
    by_attr[str(a)] = metrics(grp.is_critical.to_numpy(), grp.score.to_numpy(), thr_temp, f"{a:7s} ")
results["temporal_by_attrition"] = by_attr

print("\n=== B temporal test by swarm size ===")
by_size = {}
for a, grp in te_df.groupby("num_agents", observed=True):
    by_size[int(a)] = metrics(grp.is_critical.to_numpy(), grp.score.to_numpy(), thr_temp, f"n={int(a):3d} ")
results["temporal_by_size"] = by_size

promo_keys = set(map(tuple, df.loc[df.was_promoted, ["case_id", "agent_id"]]
                     .astype({"case_id": str}).to_numpy()))
te_df["is_promoted_agent"] = [tuple(x) in promo_keys
                              for x in te_df[["case_id", "agent_id"]]
                              .astype({"case_id": str}).to_numpy()]
print("\n=== B temporal test by promotion subset ===")
for flag, grp in te_df.groupby("is_promoted_agent", observed=True):
    if len(grp) > 10 and grp.is_critical.nunique() > 1:
        results[f"temporal_promoted_{flag}"] = metrics(
            grp.is_critical.to_numpy(), grp.score.to_numpy(), thr_temp, f"promoted={flag} ")
print("promoted test rows:", int(te_df.is_promoted_agent.sum()),
      "agents:", len(promo_keys))

with open(OUT / "prelim_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=float)
print("\nsaved", OUT / "prelim_results.json")
