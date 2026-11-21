"""Third experiment batch: temporal window sensitivity, static vs dynamic, score derivability."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef, r2_score)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
OUT = Path(__file__).resolve().parent / "artifacts"

USE = ["case_id", "num_agents", "n_obstacles", "obstacle_density", "mean_obstacle_size",
       "clutter_index", "obstacle_clustering", "attrition_rate", "step", "agent_id",
       "rel_z", "dist_to_centroid", "rank_dist_to_centroid", "dist_to_target",
       "rank_dist_to_target", "speed", "heading", "flight_path_angle", "accel",
       "heading_rate", "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
       "weighted_density", "angular_coverage", "criticality_score", "degree_centrality",
       "closeness_centrality", "betweenness_centrality", "eigenvector_centrality",
       "is_critical", "was_promoted"]
DT = {"case_id": "category", "attrition_rate": "category", "agent_id": "int16",
      "step": "int16", "num_agents": "int16", "n_obstacles": "int16", "num_neighbors": "int16"}
df = pd.read_csv(CSV, usecols=USE, dtype=DT)
df = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)
df["attrition_ord"] = df.attrition_rate.map({"none": 0, "light": 1, "medium": 2, "heavy": 3}).astype("int8")
g = df.groupby(["case_id", "agent_id"], observed=True)
df["heading_sin"] = np.sin(df.heading); df["heading_cos"] = np.cos(df.heading)
df["alive_count"] = df.groupby(["case_id", "step"], observed=True).agent_id.transform("size")
df["neighbor_frac"] = df.num_neighbors / (df.alive_count - 1).clip(lower=1)
df["is_isolated"] = (df.num_neighbors == 0).astype("int8")
LAG_BASE = {"dist_to_centroid": "dist", "num_neighbors": "nb",
            "weighted_density": "dens", "speed": "speed",
            "angular_coverage": "ang", "rank_dist_to_centroid": "rank"}
for src, tag in LAG_BASE.items():
    for lag in (1, 2, 4, 8, 16):
        df[f"d{lag}_{tag}"] = g[src].diff(lag)

BASE = ["speed", "accel", "heading_rate", "flight_path_angle", "heading_sin", "heading_cos",
        "dist_to_centroid", "rank_dist_to_centroid", "rel_z", "dist_to_target",
        "rank_dist_to_target", "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
        "weighted_density", "angular_coverage", "degree_centrality", "closeness_centrality",
        "betweenness_centrality", "eigenvector_centrality", "neighbor_frac", "is_isolated"]
ENV = ["n_obstacles", "obstacle_density", "mean_obstacle_size", "clutter_index",
       "obstacle_clustering", "num_agents", "attrition_ord"]
D1 = [f"d1_{t}" for t in LAG_BASE.values()]
D2 = D1 + [f"d{lag}_{t}" for lag in (2, 4, 8) for t in LAG_BASE.values()]
D3 = D2 + [f"d16_{t}" for t in LAG_BASE.values()]


def stratified_split(frame, seed=0):
    meta = (frame.groupby("case_id", observed=True)
            .agg(size=("num_agents", "first"), attr=("attrition_rate", "first")).reset_index())
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for _, grp in meta.groupby(["size", "attr"], observed=True):
        ids = grp.case_id.astype(str).tolist(); rng.shuffle(ids); n = len(ids)
        nt = max(1, round(0.15 * n)) if n >= 5 else (1 if n >= 3 else 0)
        nv = max(1, round(0.15 * n)) if n >= 5 else (1 if n >= 3 else 0)
        test += ids[:nt]; val += ids[nt:nt + nv]; train += ids[nt + nv:]
    return set(train), set(val), set(test)


tr, va, te = stratified_split(df)
trm, vam, tem = (df.case_id.astype(str).isin(s) for s in (tr, va, te))


def pick_thr(y, p):
    ths = np.linspace(0.02, 0.98, 97)
    return float(ths[int(np.argmax([f1_score(y, p >= t) for t in ths]))])


def fit(cols, y=None, n=400):
    m = lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=127,
                           min_child_samples=60, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
                           n_jobs=-1, random_state=0, verbose=-1)
    target = df.loc[trm, "is_critical"] if y is None else y[trm]
    m.fit(df.loc[trm, cols], target,
          eval_set=[(df.loc[vam, cols], (df.loc[vam, "is_critical"] if y is None else y[vam]))],
          eval_metric="average_precision" if y is None else "l2",
          callbacks=[lgb.early_stopping(40, verbose=False)])
    return m


def ev(name, y, p, thr):
    r = {"n": int(len(y)), "pr_auc": float(average_precision_score(y, p)),
         "roc_auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= thr)),
         "recall": float(recall_score(y, p >= thr)),
         "precision": float(precision_score(y, p >= thr, zero_division=0)),
         "mcc": float(matthews_corrcoef(y, p >= thr))}
    print(f"{name:30s} PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} "
          f"F1={r['f1']:.4f} R={r['recall']:.3f} P={r['precision']:.3f}")
    return r


print("== temporal window sensitivity (lagged deltas) ==")
win = {}
for name, cols in [("static", BASE + ENV),
                   ("lag1 (W~2)", BASE + ENV + D1),
                   ("lag1+2+4+8 (W~8)", BASE + ENV + D2),
                   ("lag1..16 (W~16)", BASE + ENV + D3)]:
    m = fit(cols)
    thr = pick_thr(df.loc[vam, "is_critical"].to_numpy(),
                   m.predict_proba(df.loc[vam, cols])[:, 1])
    p = m.predict_proba(df.loc[tem, cols])[:, 1]
    win[name] = ev(name, df.loc[tem, "is_critical"].to_numpy(), p, thr)
    win[name]["threshold"] = thr; win[name]["n_features"] = len(cols)
    if name.startswith("lag1 (W"):
        p_static = p
    if name.startswith("lag1..16"):
        p_long = p

# ---- static vs dynamic by scenario progress
print("\n== performance vs scenario progress (lag1..16 model) ==")
prog = {}
for q, (lo, hi) in enumerate([(0, 0), (0.4, 0.6), (0.8, 1.0)]):
    pass
# use step quantile within scenario
qbin = df.groupby("case_id", observed=True).step.transform(
    lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
te_df = df.loc[tem].copy(); te_df["q"] = qbin.loc[tem]
te_df["p"] = p_long
thr_long = win["lag1..16 (W~16)"]["threshold"]
for q, grp in te_df.groupby("q", observed=True):
    prog[int(q)] = ev(f"progress quintile {int(q)}", grp.is_critical.to_numpy(),
                      grp.p.to_numpy(), thr_long)

# ---- t=0 only
print("\n== t=0 cross-section ==")
t0 = te_df[te_df.step == 0]
t0m = ev("test t=0", t0.is_critical.to_numpy(), t0.p.to_numpy(), thr_long)

# ---- agent-level aggregation
print("\n== agent-level aggregation (mean score over trajectory) ==")
ag = (te_df.groupby(["case_id", "agent_id"], observed=True)
      .agg(p=("p", "mean"), y=("is_critical", "mean"), n=("p", "size")).reset_index())
ag["y"] = (ag.y > 0.5).astype(int)
agm = ev("agent-level", ag.y.to_numpy(), ag.p.to_numpy(), pick_thr(ag.y.to_numpy(), ag.p.to_numpy()))

# ---- is criticality_score a function of observable geometry? (regression)
print("\n== derivability of criticality_score from leak-free geometry ==")
REG_COLS = BASE + ENV  # no temporal, no score
reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=127,
                        min_child_samples=60, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.8, n_jobs=-1, random_state=0, verbose=-1)
reg.fit(df.loc[trm, REG_COLS], df.loc[trm, "criticality_score"])
pred = reg.predict(df.loc[tem, REG_COLS])
true = df.loc[tem, "criticality_score"].to_numpy()
reg_res = {"r2": float(r2_score(true, pred)),
           "spearman": float(pd.Series(pred).corr(pd.Series(true), method="spearman")),
           "rmse": float(np.sqrt(np.mean((pred - true) ** 2)))}
print(reg_res)

# ---- feature distributions by label (medians)
print("\n== median feature values by label ==")
cols = ["min_neighbor_dist", "mean_neighbor_dist", "weighted_density", "angular_coverage",
        "dist_to_centroid", "rank_dist_to_centroid", "speed", "accel", "heading_rate",
        "num_neighbors", "degree_centrality", "criticality_score"]
dist = df.groupby("is_critical", observed=True)[cols].median().round(4)
print(dist.to_string())
dist_by_label = {str(k): {c: float(v) for c, v in row.items()} for k, row in dist.iterrows()}

out = {"window": win, "progress": prog, "t0": t0m, "agent_level": agm,
       "score_derivability": reg_res, "median_by_label": dist_by_label}
with open(OUT / "experiments3.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=float)
print("\nsaved", OUT / "experiments3.json")
