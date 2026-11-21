"""Second experiment batch: ablations, promotion behaviour, leave-one-size-out."""
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
LOCAL = ["speed", "accel", "heading_rate", "flight_path_angle", "heading_sin", "heading_cos",
         "rel_z", "dist_to_target", "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
         "weighted_density", "angular_coverage", "neighbor_frac", "is_isolated"]
CENT = ["degree_centrality", "closeness_centrality", "betweenness_centrality", "eigenvector_centrality"]


def stratified_split(frame, seed=0):
    meta = (frame.groupby("case_id", observed=True)
            .agg(size=("num_agents", "first"), attr=("attrition_rate", "first")).reset_index())
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for _, grp in meta.groupby(["size", "attr"], observed=True):
        ids = grp.case_id.astype(str).tolist(); rng.shuffle(ids); n = len(ids)
        n_test = max(1, round(0.15 * n)) if n >= 5 else (1 if n >= 3 else 0)
        n_val = max(1, round(0.15 * n)) if n >= 5 else (1 if n >= 3 else 0)
        test += ids[:n_test]; val += ids[n_test:n_test + n_val]; train += ids[n_test + n_val:]
    return set(train), set(val), set(test)


tr, va, te = stratified_split(df)
trm, vam, tem = (df.case_id.astype(str).isin(s) for s in (tr, va, te))


def fit_predict(cols, n_estimators=400, early=40, train_mask=None, val_mask=None):
    train_mask = trm if train_mask is None else train_mask
    val_mask = vam if val_mask is None else val_mask
    m = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=127,
                           min_child_samples=60, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
                           n_jobs=-1, random_state=0, verbose=-1)
    m.fit(df.loc[train_mask, cols], df.loc[train_mask, "is_critical"],
          eval_set=[(df.loc[val_mask, cols], df.loc[val_mask, "is_critical"])],
          eval_metric="average_precision", callbacks=[lgb.early_stopping(early, verbose=False)])
    return m


def pick_threshold(y, p):
    ths = np.linspace(0.02, 0.98, 97)
    f1s = np.array([f1_score(y, p >= t) for t in ths])
    return float(ths[int(np.argmax(f1s))])


def report(name, y, p, thr):
    print(f"{name:34s} n={len(y):7d} pos={y.mean():.3f} PR-AUC={average_precision_score(y,p):.4f} "
          f"ROC={roc_auc_score(y,p):.4f} F1={f1_score(y,p>=thr):.4f} "
          f"R={recall_score(y,p>=thr):.3f} P={precision_score(y,p>=thr,zero_division=0):.3f} "
          f"MCC={matthews_corrcoef(y,p>=thr):.3f}")
    return {"n": int(len(y)), "pos_rate": float(y.mean()),
            "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "f1": float(f1_score(y, p >= thr)),
            "recall": float(recall_score(y, p >= thr)),
            "precision": float(precision_score(y, p >= thr, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, p >= thr))}


print("== ablations (same split; threshold from val) ==")
abl = {}
runs = {
    "full (temporal)": BASE + DELTAS + ENV,
    "local-only (distributed)": LOCAL + DELTAS + ENV,
    "no env": BASE + DELTAS,
    "scenario prior only": ENV + ["alive_count"],
    "minus min_neighbor_dist": [c for c in BASE if c != "min_neighbor_dist"] + DELTAS + ENV,
    "kinematics only": ["speed", "accel", "heading_rate", "flight_path_angle",
                        "heading_sin", "heading_cos", "rel_z"] + DELTAS + ENV,
    "plus elapsed step": BASE + DELTAS + ENV + ["step"],
    "minus all centralities": [c for c in BASE if c not in CENT] + DELTAS + ENV,
}
preds = {}
for name, cols in runs.items():
    m = fit_predict(cols)
    p_va = m.predict_proba(df.loc[vam, cols])[:, 1]
    thr = pick_threshold(df.loc[vam, "is_critical"].to_numpy(), p_va)
    p_te = m.predict_proba(df.loc[tem, cols])[:, 1]
    abl[name] = report(name, df.loc[tem, "is_critical"].to_numpy(), p_te, thr)
    abl[name]["threshold"] = thr
    abl[name]["n_features"] = len(cols)
    preds[name] = p_te

# ---------------- promotion analysis ----------------
print("\n== promotion behaviour (full temporal vs full static) ==")
m_static = fit_predict(BASE + ENV)
p_static_te = m_static.predict_proba(df.loc[tem, BASE + ENV])[:, 1]
p_temp_te = preds["full (temporal)"]
test_df = df.loc[tem].copy()
test_df["p_static"] = p_static_te
test_df["p_temporal"] = p_temp_te
promo_rows = test_df[test_df.was_promoted]
pre_rows = test_df[(~test_df.was_promoted) & test_df.is_critical == False]
print(f"post-promotion rows in test: {len(promo_rows)} (all label=1), "
      f"promoted agents: {promo_rows.groupby(['case_id', 'agent_id'], observed=True).ngroups}")
thr_temp = abl["full (temporal)"]["threshold"]
thr_static = pick_threshold(
    df.loc[vam, "is_critical"].to_numpy(),
    m_static.predict_proba(df.loc[vam, BASE + ENV])[:, 1])
for tag, thr in [("temporal", thr_temp), ("static", thr_static)]:
    col = f"p_{tag}"
    print(f"  {tag}: post-promo recall={float((promo_rows[col] >= thr).mean()):.3f} "
          f"mean_score={float(promo_rows[col].mean()):.3f} | "
          f"never-critical FPR={float((pre_rows[col] >= thr).mean()):.3f}")
abl["promotion"] = {
    "post_promo_rows": int(len(promo_rows)),
    "temporal_recall": float((promo_rows.p_temporal >= thr_temp).mean()),
    "static_recall": float((promo_rows.p_static >= thr_static).mean()),
    "temporal_mean_score": float(promo_rows.p_temporal.mean()),
    "static_mean_score": float(promo_rows.p_static.mean()),
}

# score trajectory around the promotion step (post-promo rows carry was_promoted=True)
print("\n  score jump at promotion (temporal):")
jumps = []
for (case, agent), grp in test_df[test_df.was_promoted].groupby(["case_id", "agent_id"], observed=True):
    first = grp.step.min()
    before = test_df[(test_df.case_id == case) & (test_df.agent_id == agent) &
                     (test_df.step < first)].tail(1)
    if len(before) == 0:
        continue
    jumps.append({"case": str(case), "agent": int(agent), "step": int(first),
                  "p_before": float(before.p_temporal.iloc[0]),
                  "p_after": float(grp.iloc[0].p_temporal)})
jdf = pd.DataFrame(jumps)
if len(jdf):
    jdf["jump"] = jdf.p_after - jdf.p_before
    print(jdf[["p_before", "p_after", "jump"]].describe().round(3).to_string())
    print("  fraction with positive jump:", float((jdf.jump > 0).mean()))
    abl["promotion_jump"] = jdf[["p_before", "p_after", "jump"]].describe().to_dict()

# ---------------- leave-one-size-out ----------------
print("\n== leave-one-size-out (full temporal features) ==")
loso = {}
sizes = sorted(df.num_agents.unique().tolist())
cols = BASE + DELTAS + ENV
for s in sizes:
    ho = df.num_agents == s
    trn = ~ho
    # inner val: 15% of training scenarios
    inner_meta = (df.loc[trn].groupby("case_id", observed=True)
                  .agg(size=("num_agents", "first"), attr=("attrition_rate", "first")).reset_index())
    rng = np.random.default_rng(0)
    v_ids = []
    for _, grp in inner_meta.groupby(["size", "attr"], observed=True):
        ids = grp.case_id.astype(str).tolist()
        if len(ids) >= 4:
            v_ids += ids[:max(1, round(0.15 * len(ids)))]
    val_mask = df.case_id.astype(str).isin(set(v_ids))
    tr_mask = trn & ~val_mask
    m = fit_predict(cols, n_estimators=300, early=30, train_mask=tr_mask, val_mask=val_mask)
    p_va = m.predict_proba(df.loc[val_mask, cols])[:, 1]
    thr = pick_threshold(df.loc[val_mask, "is_critical"].to_numpy(), p_va)
    p_ho = m.predict_proba(df.loc[ho, cols])[:, 1]
    y_ho = df.loc[ho, "is_critical"].to_numpy()
    loso[int(s)] = report(f"unseen size {int(s):3d}", y_ho, p_ho, thr)
    loso[int(s)]["threshold"] = thr

# leave-one-attrition-out
print("\n== leave-one-attrition-out ==")
lodo = {}
for a in ["none", "light", "medium", "heavy"]:
    ho = df.attrition_rate == a
    trn = ~ho
    val_mask = pd.Series(False, index=df.index)
    inner_meta = (df.loc[trn].groupby("case_id", observed=True)
                  .agg(size=("num_agents", "first"), attr=("attrition_rate", "first")).reset_index())
    rng = np.random.default_rng(0)
    v_ids = []
    for _, grp in inner_meta.groupby(["size", "attr"], observed=True):
        ids = grp.case_id.astype(str).tolist()
        if len(ids) >= 4:
            v_ids += ids[:max(1, round(0.15 * len(ids)))]
    val_mask = df.case_id.astype(str).isin(set(v_ids))
    m = fit_predict(cols, n_estimators=300, early=30, train_mask=trn & ~val_mask, val_mask=val_mask)
    p_va = m.predict_proba(df.loc[val_mask, cols])[:, 1]
    thr = pick_threshold(df.loc[val_mask, "is_critical"].to_numpy(), p_va)
    p_ho = m.predict_proba(df.loc[ho, cols])[:, 1]
    lodo[a] = report(f"unseen attrition {a:6s}", df.loc[ho, "is_critical"].to_numpy(), p_ho, thr)
    lodo[a]["threshold"] = thr

out = {"ablations": abl, "loso": loso, "lodo": lodo}
with open(OUT / "experiments2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=float)
print("\nsaved", OUT / "experiments2.json")
