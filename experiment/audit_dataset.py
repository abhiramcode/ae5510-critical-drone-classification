"""Read-only audit of the swarm criticality dataset.

Writes machine-readable summaries to experiment/artifacts/ and prints a report.
The dataset itself is never modified.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(exist_ok=True)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- load
USECOLS = [
    "case_id", "num_agents", "n_obstacles", "obstacle_density", "mean_obstacle_size",
    "clutter_index", "obstacle_clustering", "attrition_rate", "step", "agent_id",
    "rel_x", "rel_y", "rel_z", "dist_to_centroid", "rank_dist_to_centroid",
    "dist_to_target", "rank_dist_to_target", "vx", "vy", "vz", "speed", "heading",
    "flight_path_angle", "accel", "heading_rate", "num_neighbors", "mean_neighbor_dist",
    "min_neighbor_dist", "weighted_density", "angular_coverage", "criticality_score",
    "degree_centrality", "closeness_centrality", "betweenness_centrality",
    "eigenvector_centrality", "is_critical", "was_promoted",
]
DTYPES = {
    "case_id": "category",
    "attrition_rate": "category",
    "is_critical": "bool",
    "was_promoted": "bool",
    "agent_id": "int16",
    "step": "int16",
    "num_agents": "int16",
    "n_obstacles": "int16",
    "num_neighbors": "int16",
}

log("loading csv")
df = pd.read_csv(CSV, usecols=USECOLS, dtype=DTYPES)
log(f"loaded shape={df.shape}")

report: dict = {}
report["shape"] = list(df.shape)
report["columns"] = list(df.columns)
report["memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 1)

# ---------------------------------------------------------------- integrity
dups = int(df.duplicated(["case_id", "step", "agent_id"]).sum())
report["duplicate_keys"] = dups
report["nan_cells"] = int(df.isna().sum().sum())
report["n_scenarios"] = int(df.case_id.nunique())
report["step_min"] = int(df.step.min())
report["step_max"] = int(df.step.max())
report["step_parity"] = sorted(df.step.unique()[:5].tolist())
report["agents_range"] = [int(df.agent_id.min()), int(df.agent_id.max())]

# static columns must be constant per scenario
static_cols = ["num_agents", "n_obstacles", "obstacle_density", "mean_obstacle_size",
               "clutter_index", "obstacle_clustering", "attrition_rate"]
nun = df.groupby("case_id", observed=True)[static_cols].nunique()
report["static_columns_nonconstant"] = {c: int((nun[c] > 1).sum()) for c in static_cols}

log("per-scenario profile")
prof = (
    df.groupby("case_id", observed=True)
    .agg(rows=("agent_id", "size"),
         n_steps=("step", "nunique"),
         step_max=("step", "max"),
         n_agents=("agent_id", "nunique"),
         pos_rate=("is_critical", "mean"),
         size=("num_agents", "first"),
         attrition=("attrition_rate", "first"),
         obstacles=("n_obstacles", "first"),
         density=("obstacle_density", "first"),
         clutter=("clutter_index", "first"),
         clust=("obstacle_clustering", "first"))
    .reset_index()
)
promo_agent_counts = (df[df.was_promoted].groupby("case_id", observed=True).agent_id.nunique()
                      .rename("promo_agents"))
prof = prof.merge(promo_agent_counts, on="case_id", how="left")
prof["promo_agents"] = prof["promo_agents"].fillna(0).astype(int)
prof["size"] = prof["size"].astype(int)
prof["attrition"] = prof["attrition"].astype(str)
prof["obstacles"] = prof["obstacles"].astype(int)
prof.to_csv(OUT / "scenario_profile.csv", index=False)

report["scenario_size_counts"] = {int(k): int(v) for k, v in
                                  prof["size"].value_counts().sort_index().items()}
report["scenario_attrition_counts"] = {str(k): int(v) for k, v in
                                       prof["attrition"].value_counts().items()}
report["rows_per_scenario"] = {
    "min": int(prof.rows.min()), "median": float(prof.rows.median()),
    "max": int(prof.rows.max()), "mean": float(prof.rows.mean())}
report["steps_per_scenario"] = {
    "min": int(prof.n_steps.min()), "median": float(prof.n_steps.median()),
    "max": int(prof.n_steps.max())}
report["pos_rate_per_scenario"] = {
    "min": float(prof.pos_rate.min()), "median": float(prof.pos_rate.median()),
    "max": float(prof.pos_rate.max())}
report["scenarios_with_zero_obstacles"] = int((prof.obstacles == 0).sum())
report["clustering_nonzero_with_zero_obstacles"] = int(
    ((prof.obstacles == 0) & (prof.clust > 0)).sum())

# ---------------------------------------------------------------- label dynamics
log("label dynamics")
df.sort_values(["case_id", "agent_id", "step"], kind="mergesort", inplace=True)
g = df.groupby(["case_id", "agent_id"], observed=True)
df["prev_label"] = g["is_critical"].shift()
trans = df[df.prev_label.notna() & (df.prev_label != df.is_critical)]
report["label_transition_rows"] = int(len(trans))
report["label_transition_direction"] = (
    trans.assign(kind=np.where(trans.is_critical, "0->1", "1->0"))
    .kind.value_counts().to_dict())
report["agents_with_label_change"] = int(
    trans.groupby(["case_id", "agent_id"], observed=True).ngroups)
promo = df[df.was_promoted]
report["was_promoted_rows"] = int(len(promo))
report["was_promoted_agents"] = int(
    promo.groupby(["case_id", "agent_id"], observed=True).ngroups)
report["was_promoted_scenarios"] = int(promo.case_id.nunique())
report["was_promoted_without_is_critical"] = int((promo.is_critical == False).sum())

# first promotion step per promoted agent + label just before
first_promo = (
    df[df.was_promoted]
    .groupby(["case_id", "agent_id"], observed=True)
    .agg(first_promo_step=("step", "min"), n_promo_steps=("step", "size"),
         size=("num_agents", "first"), attrition=("attrition_rate", "first"))
    .reset_index()
)
# step-2 label before promotion
first_promo = first_promo.merge(
    df[["case_id", "agent_id", "step", "is_critical"]],
    left_on=["case_id", "agent_id", "first_promo_step"],
    right_on=["case_id", "agent_id", "step"], how="left")
first_promo = first_promo.rename(columns={"is_critical": "label_at_promo"})
first_promo["attrition"] = first_promo["attrition"].astype(str)
first_promo.to_csv(OUT / "promotions.csv", index=False)
report["promotion_agents_by_attrition"] = (
    first_promo.groupby("attrition").size().to_dict())
report["promotion_step_stats"] = {
    "min": int(first_promo.first_promo_step.min()),
    "median": float(first_promo.first_promo_step.median()),
    "max": int(first_promo.first_promo_step.max())}
report["promotions_per_scenario"] = {
    int(k): int(v) for k, v in
    first_promo.groupby("case_id", observed=True).size().value_counts().sort_index().items()}

# label constancy among non-promoted agents
nonpromo = df[~df.was_promoted]
lab_std = nonpromo.groupby(["case_id", "agent_id"], observed=True)["is_critical"].nunique()
report["nonpromoted_agents_with_variable_label"] = int((lab_std > 1).sum())

# ---------------------------------------------------------------- class balance
log("class balance")
report["pos_rate_overall"] = float(df.is_critical.mean())
report["pos_rate_by_attrition"] = (
    df.groupby("attrition_rate", observed=True).is_critical.mean().astype(float).to_dict())
report["pos_rate_by_size"] = {
    int(k): float(v) for k, v in
    df.groupby("num_agents", observed=True).is_critical.mean().items()}

# ---------------------------------------------------------------- geometry checks
log("geometry / graph reconstruction")
# (a) sum of relative positions ~ 0 per snapshot
snap_sum = (
    df.groupby(["case_id", "step"], observed=True)[["rel_x", "rel_y", "rel_z"]]
    .sum().abs()
)
report["max_abs_rel_sum"] = float(snap_sum.to_numpy().max())
report["median_abs_rel_sum"] = float(np.median(snap_sum.to_numpy()))

# (b) pairwise distances reproduce dist_to_centroid
d0 = df[df.step == 0]
r = d0[["rel_x", "rel_y", "rel_z"]].to_numpy()
recon = np.linalg.norm(r, axis=1)
err = np.abs(recon - d0.dist_to_centroid.to_numpy())
report["dist_to_centroid_recon_max_err"] = float(err.max())

# (c) radius-55 graph must reproduce num_neighbors: sample snapshots across scenarios
rng = np.random.default_rng(0)
keys = df[["case_id", "step"]].drop_duplicates()
sample_keys = keys.sample(n=min(120, len(keys)), random_state=0)
max_nb_err = 0
centrality_errs = {c: [] for c in ["degree_centrality", "closeness_centrality",
                                   "betweenness_centrality", "eigenvector_centrality"]}
import networkx as nx

for _, row in sample_keys.iterrows():
    sub = df[(df.case_id == row.case_id) & (df.step == row.step)]
    if len(sub) < 4:
        continue
    pts = sub[["rel_x", "rel_y", "rel_z"]].to_numpy()
    ids = sub.agent_id.to_numpy()
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    adj = (D <= 55.0)
    np.fill_diagonal(adj, False)
    nb = adj.sum(1)
    max_nb_err = max(max_nb_err, int(np.abs(nb - sub.num_neighbors.to_numpy()).max()))
    G = nx.from_numpy_array(adj.astype(int))
    mapping = {i: int(ids[i]) for i in range(len(ids))}
    G = nx.relabel_nodes(G, mapping)
    metrics = {}
    for name, fn in [
        ("degree_centrality", lambda: nx.degree_centrality(G)),
        ("closeness_centrality", lambda: nx.closeness_centrality(G)),
        ("betweenness_centrality", lambda: nx.betweenness_centrality(G, normalized=True)),
        ("eigenvector_centrality", lambda: nx.eigenvector_centrality(G, max_iter=1000, tol=1e-8)),
    ]:
        try:
            metrics[name] = fn()
        except Exception:
            pass
    for c, m in metrics.items():
        vals = sub.set_index("agent_id")[c]
        pred = np.array([m[int(a)] for a in ids])
        centrality_errs[c].append(float(np.abs(pred - vals.to_numpy()).max()))

report["num_neighbors_recon_max_err"] = int(max_nb_err)
report["centrality_recon_max_err"] = {c: (float(np.max(v)) if v else None)
                                      for c, v in centrality_errs.items()}
report["centrality_recon_median_err"] = {c: (float(np.median(v)) if v else None)
                                         for c, v in centrality_errs.items()}

# ---------------------------------------------------------------- leakage diagnostics
log("leakage diagnostics")
from sklearn.metrics import roc_auc_score, average_precision_score

y = df.is_critical.to_numpy()
leak = {}
for c in ["criticality_score", "was_promoted", "rank_dist_to_centroid",
          "rank_dist_to_target", "weighted_density", "degree_centrality"]:
    v = df[c].to_numpy(dtype=float)
    leak[c] = {
        "roc_auc": float(roc_auc_score(y, v)),
        "pr_auc": float(average_precision_score(y, v)),
        "point_biserial": float(np.corrcoef(v, y)[0, 1]),
    }
report["single_feature_scores"] = leak

sep = (df[df.step == 0].groupby("is_critical", observed=True)
       .criticality_score.agg(["min", "max", "mean", "count"]))
report["criticality_score_t0"] = {
    str(k): {kk: float(vv) for kk, vv in row.items()} for k, row in sep.iterrows()}

# score behaviour around promotions
promo_rows = []
for _, p in first_promo.iterrows():
    sub = df[(df.case_id == p.case_id) & (df.agent_id == p.agent_id) &
             (df.step <= p.first_promo_step) & (df.step >= p.first_promo_step - 20)]
    promo_rows.append(sub[["case_id", "agent_id", "step", "criticality_score",
                           "num_neighbors", "dist_to_centroid", "is_critical"]])
if promo_rows:
    pr = pd.concat(promo_rows)
    pr.to_csv(OUT / "promotion_trajectories.csv", index=False)

# ---------------------------------------------------------------- feature redundancy (sample)
log("redundancy analysis")
num_cols = ["rel_z", "dist_to_centroid", "rank_dist_to_centroid", "dist_to_target",
            "rank_dist_to_target", "vx", "vy", "vz", "speed", "heading",
            "flight_path_angle", "accel", "heading_rate", "num_neighbors",
            "mean_neighbor_dist", "min_neighbor_dist", "weighted_density",
            "angular_coverage", "degree_centrality", "closeness_centrality",
            "betweenness_centrality", "eigenvector_centrality",
            "n_obstacles", "obstacle_density", "mean_obstacle_size",
            "clutter_index", "obstacle_clustering"]
samp = df.sample(n=min(300_000, len(df)), random_state=0)
corr = samp[num_cols].corr(method="spearman")
corr.to_csv(OUT / "spearman_corr.csv")
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
         .stack().rename("rho").reset_index().rename(columns={"level_0": "a", "level_1": "b"}))
pairs["abs"] = pairs.rho.abs()
report["top_correlated_pairs"] = (
    pairs.sort_values("abs", ascending=False).head(15)
    [["a", "b", "rho"]].round(4).to_dict("records"))

# mutual information with label (sampled)
from sklearn.feature_selection import mutual_info_classif
mi_cols = ["speed", "dist_to_centroid", "rank_dist_to_centroid", "dist_to_target",
           "num_neighbors", "mean_neighbor_dist", "weighted_density",
           "angular_coverage", "degree_centrality", "betweenness_centrality",
           "criticality_score", "rel_z"]
mi = mutual_info_classif(samp[mi_cols].to_numpy(), samp.is_critical.to_numpy(),
                         random_state=0)
report["mutual_info_with_label"] = {c: float(v) for c, v in zip(mi_cols, mi)}

# ---------------------------------------------------------------- agent_id association
log("agent_id association")
samp2 = samp.assign(rel_id=samp.agent_id / samp.num_agents)
report["agent_id_point_biserial"] = float(
    np.corrcoef(samp2.agent_id.to_numpy(dtype=float), samp2.is_critical.to_numpy())[0, 1])
report["rel_id_point_biserial"] = float(
    np.corrcoef(samp2.rel_id.to_numpy(dtype=float), samp2.is_critical.to_numpy())[0, 1])
bins = pd.cut(samp2.rel_id, bins=10, labels=False)
report["pos_rate_by_rel_id_decile"] = {
    int(k): float(v) for k, v in
    samp2.groupby(bins).is_critical.mean().round(4).items()}

# ---------------------------------------------------------------- ranges / sanity
log("ranges")
ranges = {}
for c in ["speed", "accel", "heading", "flight_path_angle", "heading_rate",
          "rel_x", "rel_y", "rel_z", "dist_to_centroid", "dist_to_target",
          "num_neighbors", "weighted_density", "angular_coverage"]:
    s = df[c]
    ranges[c] = {"min": float(s.min()), "max": float(s.max()),
                 "mean": float(s.mean()), "p01": float(s.quantile(0.01)),
                 "p99": float(s.quantile(0.99))}
report["ranges"] = ranges
report["steps_all_even"] = bool((df.step % 2 == 0).all())

# ---------------------------------------------------------------- save
with open(OUT / "audit_summary.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, default=float)

log("done")
print("\n===== SUMMARY =====")
for k in ["shape", "duplicate_keys", "nan_cells", "n_scenarios",
          "duplicate_keys", "scenario_size_counts", "scenario_attrition_counts",
          "rows_per_scenario", "steps_per_scenario", "pos_rate_per_scenario",
          "pos_rate_overall", "pos_rate_by_attrition",
          "label_transition_rows", "label_transition_direction",
          "agents_with_label_change", "was_promoted_rows", "was_promoted_agents",
          "was_promoted_scenarios", "was_promoted_without_is_critical",
          "nonpromoted_agents_with_variable_label",
          "max_abs_rel_sum", "dist_to_centroid_recon_max_err",
          "num_neighbors_recon_max_err", "centrality_recon_max_err",
          "centrality_recon_median_err", "single_feature_scores",
          "criticality_score_t0", "promotion_agents_by_attrition",
          "promotion_step_stats", "promotions_per_scenario",
          "top_correlated_pairs", "mutual_info_with_label",
          "agent_id_point_biserial", "rel_id_point_biserial",
          "pos_rate_by_rel_id_decile", "steps_all_even"]:
    print(f"{k}: {json.dumps(report.get(k), default=float)}")
