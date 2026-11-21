"""Third-stage diagnosis: is the simulator graph a proximity-only graph or occluded?

We cannot rebuild obstacle layouts, but we can test (a) whether provided counts are a
subset of radius-55 counts, (b) whether deficits scale with obstacle/clutter statistics,
and (c) whether including recently-lost agents' last positions helps.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
USE = ["case_id", "step", "agent_id", "rel_x", "rel_y", "rel_z", "num_neighbors",
       "n_obstacles", "obstacle_density", "clutter_index", "obstacle_clustering",
       "dist_to_centroid", "num_agents", "attrition_rate"]
DT = {"case_id": "category", "attrition_rate": "category", "agent_id": "int16",
      "step": "int16", "num_neighbors": "int16", "num_agents": "int16",
      "n_obstacles": "int16"}
df = pd.read_csv(CSV, usecols=USE, dtype=DT)
df = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)

# scenario-level obstacle descriptors
scen = (df.groupby("case_id", observed=True)
        .agg(n_obs=("n_obstacles", "first"), dens=("obstacle_density", "first"),
             clutter=("clutter_index", "first"), clust=("obstacle_clustering", "first"),
             size=("num_agents", "first"), attr=("attrition_rate", "first"))
        .reset_index())
scen["attr"] = scen["attr"].astype(str)

# one snapshot per scenario at ~40% and ~80% of its timeline
rows = []
for case, grp in df.groupby("case_id", observed=True):
    steps = np.sort(grp.step.unique())
    for frac in (0.25, 0.55, 0.85):
        s = steps[min(len(steps) - 1, int(frac * len(steps)))]
        sub = grp[grp.step == s]
        if len(sub) < 5:
            continue
        P = sub[["rel_x", "rel_y", "rel_z"]].to_numpy()
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        cnt = (D <= 55).sum(1)
        d = sub.num_neighbors.to_numpy() - cnt
        rows.append({
            "case_id": case, "step": int(s), "n": len(sub),
            "mean_diff": float(d.mean()), "max_neg": int(d.min()), "max_pos": int(d.max()),
            "mean_abs": float(np.abs(d).mean()),
            "frac_neg": float((d < 0).mean()),
            "all_le_1": bool((d <= 1).all()),
            "all_le_0": bool((d <= 0).all()),
        })
res = pd.DataFrame(rows).merge(scen, on="case_id", how="left")
res.to_csv(Path(__file__).resolve().parent / "artifacts" / "graph_residuals.csv", index=False)

print("snapshots:", len(res))
print("all diffs <= 0 (pure subset):", int(res.all_le_0.sum()))
print("all diffs <= 1:", int(res.all_le_1.sum()))
print("fraction of agents with negative diff:", round(float(res.frac_neg.mean()), 3))
print("\ncorrelation of mean_diff with environment stats:")
for c in ["n_obs", "dens", "clutter", "clust"]:
    print(f"  {c:8s} pearson={res.mean_diff.corr(res[c]):+.3f} "
          f"spearman={res.mean_diff.corr(res[c], method='spearman'):+.3f}")
print("\nby obstacle density bin:")
res["dens_bin"] = pd.cut(res.dens, [-0.01, 0.05, 0.15, 0.30, 1.0],
                         labels=["<0.05", "0.05-0.15", "0.15-0.30", ">0.30"])
print(res.groupby("dens_bin", observed=True)[["mean_diff", "mean_abs", "frac_neg", "all_le_1"]]
      .mean().round(3).to_string())
print("\nby attrition:")
print(res.groupby("attr", observed=True)[["mean_diff", "mean_abs", "frac_neg", "all_le_1"]]
      .mean().round(3).to_string())
print("\nworst mean_diff scenarios:")
print(res.sort_values("mean_diff").head(12)
      [["case_id", "step", "n", "mean_diff", "max_neg", "max_pos", "dens", "clutter", "attr"]]
      .to_string(index=False))
print("\npositive-outlier snapshots (max_pos>2):")
print(res[res.max_pos > 2]
      [["case_id", "step", "n", "mean_diff", "max_neg", "max_pos", "dens", "clutter"]]
      .to_string(index=False))

# ---- does including recently lost agents' frozen positions help?
print("\n== ghost (last-known-position) test on negative-residual snapshots ==")
for case in res.sort_values("mean_diff").head(5).case_id.tolist():
    steps = np.sort(df[df.case_id == case].step.unique())
    s = steps[min(len(steps) - 1, int(0.55 * len(steps)))]
    sub = df[(df.case_id == case) & (df.step == s)]
    alive = set(sub.agent_id)
    prior = df[(df.case_id == case) & (df.step <= s) & (~df.agent_id.isin(alive))]
    if prior.empty:
        print(f"{case}@{s}: no lost agents")
        continue
    last = (prior.sort_values("step").groupby("agent_id", observed=True).tail(1)
            [["agent_id", "rel_x", "rel_y", "rel_z"]])
    allpts = pd.concat([sub[["agent_id", "rel_x", "rel_y", "rel_z"]], last])
    P = allpts[["rel_x", "rel_y", "rel_z"]].to_numpy()
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    cnt = (D <= 55).sum(1)[:len(sub)]
    d0 = sub.num_neighbors.to_numpy() - (cnt - 0)
    Pa = sub[["rel_x", "rel_y", "rel_z"]].to_numpy()
    Da = np.linalg.norm(Pa[:, None, :] - Pa[None, :, :], axis=-1)
    np.fill_diagonal(Da, np.inf)
    ca = (Da <= 55).sum(1)
    print(f"{case}@{s}: lost={len(last)} mean_diff_alive={np.mean(sub.num_neighbors.to_numpy()-ca):.2f} "
          f"mean_diff_with_ghosts={np.mean(sub.num_neighbors.to_numpy()-cnt):.2f}")

# ---- per-agent: positive outliers at c002@154
print("\n== c002@154 per-agent diffs ==")
s = df[(df.case_id == "c002_n30_medium") & (df.step == 154)]
P = s[["rel_x", "rel_y", "rel_z"]].to_numpy()
D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
np.fill_diagonal(D, np.inf)
cnt = (D <= 55).sum(1)
o = s[["agent_id", "num_neighbors", "dist_to_centroid"]].copy()
o["radius"] = cnt
o["diff"] = o.num_neighbors - o.radius
print(o.sort_values("diff").to_string(index=False))
