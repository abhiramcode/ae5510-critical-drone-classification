"""Second-stage graph diagnosis: lag alignment and per-agent error structure."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
USE = ["case_id", "step", "agent_id", "rel_x", "rel_y", "rel_z", "num_neighbors",
       "degree_centrality", "closeness_centrality", "betweenness_centrality",
       "eigenvector_centrality", "is_critical", "dist_to_centroid"]
DT = {"case_id": "category", "agent_id": "int16", "step": "int16", "num_neighbors": "int16"}
df = pd.read_csv(CSV, usecols=USE, dtype=DT)
df = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)


def pos_at(case, step):
    s = df[(df.case_id == case) & (df.step == step)]
    return s.set_index("agent_id")[["rel_x", "rel_y", "rel_z", "num_neighbors",
                                    "dist_to_centroid", "degree_centrality"]]


def radius_counts(r, thr=55.0):
    P = r[["rel_x", "rel_y", "rel_z"]].to_numpy()
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    return (D <= thr).sum(1)


def compare(case, step, other_step):
    a = pos_at(case, step)
    b = pos_at(case, other_step)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    c = radius_counts(b.loc[common])
    return np.abs(c - a.loc[common, "num_neighbors"].to_numpy())


print("== lag test on targeted snapshots (positions from step s+d, neighbors at s) ==")
for case, s in [("c002_n30_medium", 136), ("c066_n30_none", 8), ("c028_n45_heavy", 12),
                ("c002_n30_medium", 246), ("c001_n60_none", 0), ("c040_n75_heavy", 550)]:
    line = [f"{case}@{s}"]
    for d in [-4, -2, 0, 2, 4]:
        e = compare(case, s, s + d)
        line.append(f"d={d:+d}:{'n/a' if e is None else int(e.max())}")
    print("  ".join(line))


print("\n== per-agent structure, c002@136 ==")
s = pos_at("c002_n30_medium", 136).reset_index()
c = radius_counts(s)
s["radius_count"] = c
s["diff"] = s.num_neighbors - c
print(s[["agent_id", "num_neighbors", "radius_count", "diff", "dist_to_centroid"]]
      .sort_values("diff").to_string(index=False))


print("\n== signed error distribution over 300 snapshots, current positions ==")
keys = df[["case_id", "step"]].drop_duplicates().sample(n=300, random_state=3)
rows = []
for _, r in keys.iterrows():
    s = pos_at(r.case_id, r.step).reset_index()
    if len(s) < 5:
        continue
    c = radius_counts(s)
    d = s.num_neighbors.to_numpy() - c
    rows.append({"case": str(r.case_id), "step": int(r.step), "n": len(s),
                 "max_abs": int(np.abs(d).max()),
                 "mean_signed": float(d.mean()),
                 "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum())})
e = pd.DataFrame(rows)
print(e[["max_abs", "mean_signed", "n_pos", "n_neg"]].describe().to_string())
print("max_abs==0:", int((e.max_abs == 0).sum()), "/", len(e))
print("\nrows where provided counts MORE neighbors than radius-55 (positive):")
print(e.sort_values("mean_signed", ascending=False).head(8).to_string(index=False))
print("\nrows where provided counts FEWER neighbors than radius-55 (negative):")
print(e.sort_values("mean_signed").head(8).to_string(index=False))

# degree centrality vs radius graph at current/lagged step
print("\n== degree centrality: radius graph vs provided ==")
import networkx as nx
for case, s in [("c002_n30_medium", 136), ("c003_n37_medium", 640), ("c001_n60_none", 0)]:
    for d in [0, -2, 2]:
        a = pos_at(case, s)
        b = pos_at(case, s + d)
        common = a.index.intersection(b.index)
        P = b.loc[common, ["rel_x", "rel_y", "rel_z"]].to_numpy()
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        A = (D <= 55) & (D > 0)
        G = nx.from_numpy_array(A.astype(int))
        deg = nx.degree_centrality(G)
        pred = np.array([deg[i] for i in range(len(common))])
        prov = a.loc[common, "degree_centrality"].to_numpy()
        print(f"{case}@{s} d={d:+d} deg_maxerr={np.abs(pred-prov).max():.4f}")
