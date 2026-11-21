"""Diagnose the simulator's neighbor/graph definition from observable geometry."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
USE = ["case_id", "step", "agent_id", "rel_x", "rel_y", "rel_z",
       "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
       "dist_to_centroid", "degree_centrality", "closeness_centrality",
       "betweenness_centrality", "eigenvector_centrality", "is_critical",
       "was_promoted", "num_agents", "attrition_rate"]
DT = {"case_id": "category", "attrition_rate": "category", "agent_id": "int16",
      "step": "int16", "num_neighbors": "int16", "num_agents": "int16"}

df = pd.read_csv(CSV, usecols=USE, dtype=DT)
print("shape", df.shape)

TARGETS = [("c001_n60_none", 0), ("c002_n30_medium", 0), ("c003_n37_medium", 640),
           ("c040_n75_heavy", 550), ("c100_n24_medium", 770), ("c003_n37_medium", 0)]


def snap(case, step):
    s = df[(df.case_id == case) & (df.step == step)]
    return s.reset_index(drop=True)


def counts(s, thr=55.0, dims=3):
    pts = s[["rel_x", "rel_y", "rel_z"]].to_numpy()
    if dims == 2:
        pts = pts[:, :2]
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    return (D <= thr).sum(1), D


print("\n== targeted snapshots ==")
for case, step in TARGETS:
    s = snap(case, step)
    if len(s) == 0:
        print(f"{case}@{step}: MISSING")
        continue
    n3, D3 = counts(s, 55, 3)
    n2, D2 = counts(s, 55, 2)
    err3 = int(np.abs(n3 - s.num_neighbors.to_numpy()).max())
    err2 = int(np.abs(n2 - s.num_neighbors.to_numpy()).max())
    print(f"{case}@{step}: alive={len(s)} num_nb[{s.num_neighbors.min()},{s.num_neighbors.max()}] "
          f"err3D={err3} err2D={err2} maxD={D3[np.isfinite(D3)].max():.1f} "
          f"deg=[{s.degree_centrality.min():.3f},{s.degree_centrality.max():.3f}]")
    # is num_neighbors reproducible by any fixed threshold?
    best = None
    for thr in [30, 40, 45, 50, 55, 60, 65, 70, 80, 90, 110, 150]:
        c, _ = counts(s, thr, 3)
        e = int(np.abs(c - s.num_neighbors.to_numpy()).max())
        if best is None or e < best[1]:
            best = (thr, e)
    print(f"   best 3D threshold={best[0]} maxerr={best[1]}")

print("\n== error distribution over 400 random snapshots ==")
keys = df[["case_id", "step"]].drop_duplicates().sample(n=400, random_state=1)
rows = []
for _, r in keys.iterrows():
    s = snap(r.case_id, r.step)
    if len(s) < 5:
        continue
    n3, _ = counts(s, 55, 3)
    n2, _ = counts(s, 55, 2)
    rows.append({"case_id": str(r.case_id), "step": int(r.step), "n": len(s),
                 "err3": int(np.abs(n3 - s.num_neighbors.to_numpy()).max()),
                 "err2": int(np.abs(n2 - s.num_neighbors.to_numpy()).max()),
                 "med_abs": float(np.abs(n3 - s.num_neighbors.to_numpy()).mean()),
                 "max_nb": int(s.num_neighbors.max()), "alive": len(s)})
e = pd.DataFrame(rows)
print(e[["err3", "err2", "med_abs"]].describe().to_string())
print("snapshots with err3==0:", int((e.err3 == 0).sum()), "/", len(e))
print("snapshots with err2==0:", int((e.err2 == 0).sum()), "/", len(e))
print("\nworst 10:")
print(e.sort_values("err3", ascending=False).head(10).to_string(index=False))
print("\nbest 10:")
print(e.sort_values("err3").head(10).to_string(index=False))
e.to_csv(Path(__file__).resolve().parent / "artifacts" / "graph_err_sample.csv", index=False)
