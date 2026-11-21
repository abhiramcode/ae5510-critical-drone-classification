"""Check internal consistency of positions, velocities, distances and graph stats."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
USE = ["case_id", "step", "agent_id", "rel_x", "rel_y", "rel_z", "vx", "vy", "vz",
       "speed", "dist_to_centroid", "num_neighbors", "degree_centrality",
       "closeness_centrality", "betweenness_centrality", "eigenvector_centrality",
       "is_critical", "was_promoted", "num_agents"]
DT = {"case_id": "category", "agent_id": "int16", "step": "int16",
      "num_neighbors": "int16", "num_agents": "int16"}
df = pd.read_csv(CSV, usecols=USE, dtype=DT)
df = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)

r = df[["rel_x", "rel_y", "rel_z"]].to_numpy()
v = df[["vx", "vy", "vz"]].to_numpy()
print("== distance vs rel norm ==")
print("max |dist - ||rel|||:", float(np.abs(np.linalg.norm(r, axis=1) - df.dist_to_centroid.to_numpy()).max()))
print("speed vs ||v|| max err:", float(np.abs(np.linalg.norm(v, axis=1) - df.speed.to_numpy()).max()))

# displacement over one sampled step (2 sim steps)
g = df.groupby(["case_id", "agent_id"], observed=True)
df["disp"] = np.sqrt(g.rel_x.diff()**2 + g.rel_y.diff()**2 + g.rel_z.diff()**2)
df["vprev"] = np.sqrt(g.vx.shift()**2 + g.vy.shift()**2 + g.vz.shift()**2)
df["expected"] = 2.0 * (df.speed + df.vprev.fillna(df.speed)) / 2.0
df["ratio"] = df.disp / df.expected.replace(0, np.nan)

ok = df.dropna(subset=["disp"])
print("\n== displacement / (2*speed) ==")
print(ok.ratio.describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]).to_string())
for thr in [1.5, 2, 3, 5, 10]:
    print(f"  fraction ratio > {thr}: {(ok.ratio > thr).mean():.4f}")

# per-scenario anomaly rate
an = ok.assign(bad=ok.ratio > 3).groupby("case_id", observed=True).bad.mean()
print("\nscenario anomaly-rate (ratio>3):", an.describe().round(4).to_dict())
print("worst scenarios:")
print(an.sort_values(ascending=False).head(10).round(4).to_string())

# num_neighbors vs degree centrality internal coherence
print("\n== num_neighbors vs degree_centrality*(alive-1) ==")
alive = df.groupby(["case_id", "step"], observed=True).agent_id.transform("size")
df["deg_n"] = df.degree_centrality * (alive - 1)
print("max |deg_n - num_neighbors|:", float((df.deg_n - df.num_neighbors).abs().max()))
print("median:", float((df.deg_n - df.num_neighbors).abs().median()))
print("fraction |diff|<=0.5:", float(((df.deg_n - df.num_neighbors).abs() <= 0.5).mean()))

# anomaly agents in c002
print("\n== c002 agents with ratio>3 (steps 120-160) ==")
c = ok[(ok.case_id == "c002_n30_medium") & (ok.step.between(120, 160))]
bad = c[c.ratio > 3][["step", "agent_id", "disp", "expected", "ratio", "dist_to_centroid",
                      "num_neighbors", "degree_centrality"]]
print(bad.sort_values(["step", "agent_id"]).to_string(index=False))
print("agents involved:", sorted(bad.agent_id.unique().tolist()))

# Are anomalies persistent per agent or global per step?
ok2 = ok.assign(bad=ok.ratio > 3)
print("\nfraction of rows bad, by agent (top 15):")
print(ok2.groupby(["case_id", "agent_id"], observed=True).bad.mean()
      .sort_values(ascending=False).head(15).round(3).to_string())
print("\nfraction of rows bad, by step (top 10):")
print(ok2.groupby("step").bad.mean().sort_values(ascending=False).head(10).round(3).to_string())
