"""Check whether large neighbor-count residuals are explained by per-agent temporal lag."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
USE = ["case_id", "step", "agent_id", "rel_x", "rel_y", "rel_z", "num_neighbors",
       "dist_to_centroid", "speed", "was_promoted", "is_critical"]
DT = {"case_id": "category", "agent_id": "int16", "step": "int16",
      "num_neighbors": "int16"}
df = pd.read_csv(CSV, usecols=USE, dtype=DT)

def at(case, step):
    return df[(df.case_id == case) & (df.step == step)].set_index("agent_id")

def count_for(case, step, lag, agent):
    cur = at(case, step)
    old = at(case, step - lag)
    if agent not in old.index or len(cur) == 0:
        return None
    P = cur[["rel_x", "rel_y", "rel_z"]].to_numpy().copy()
    loc = cur.index.get_loc(agent)
    P[loc] = old.loc[agent, ["rel_x", "rel_y", "rel_z"]].to_numpy(dtype=float)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    return int((D[loc] <= 55).sum())

cases = [("c002_n30_medium", 154, 24), ("c002_n30_medium", 136, 24), ("c002_n30_medium", 136, 6),
         ("c002_n30_medium", 300, 24), ("c002_n30_medium", 154, 6), ("c066_n30_none", 8, None)]
for case, step, agent in cases:
    if agent is None:
        continue
    cur = at(case, step)
    if agent not in cur.index:
        continue
    print(f"\n{case}@{step} agent {agent}: provided={int(cur.loc[agent,'num_neighbors'])} "
          f"dist={cur.loc[agent,'dist_to_centroid']:.1f} promo={bool(cur.loc[agent,'was_promoted'])}")
    for lag in [0, 2, 4, 6, 8, 10, 20, 40]:
        print(f"   lag {lag:2d} steps -> {count_for(case, step, lag, agent)}")

print("\n== trajectory of c002 agent 24 ==")
t = df[(df.case_id == "c002_n30_medium") & (df.agent_id == 24)][
    ["step", "dist_to_centroid", "num_neighbors", "speed", "is_critical", "was_promoted"]]
print(t[(t.step >= 120) & (t.step <= 310)].to_string(index=False))
