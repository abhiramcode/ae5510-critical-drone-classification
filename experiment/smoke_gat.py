"""Small scenario-grouped smoke test of the spatio-graph (GATv2) design.

Uses the reconstructed radius-55 graph and leak-free node/edge features.
Compares against a tabular LightGBM on the identical sampled snapshots to isolate
the value of message passing. This is a feasibility check, not full training.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef)
import lightgbm as lgb

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
OUT = Path(__file__).resolve().parent / "artifacts"
SEED = 0
rng = np.random.default_rng(SEED)

USE = ["case_id", "num_agents", "attrition_rate", "step", "agent_id", "rel_x", "rel_y",
       "rel_z", "vx", "vy", "vz", "speed", "heading", "flight_path_angle", "accel",
       "heading_rate", "dist_to_centroid", "rank_dist_to_centroid", "dist_to_target",
       "rank_dist_to_target", "num_neighbors", "mean_neighbor_dist", "min_neighbor_dist",
       "weighted_density", "angular_coverage", "degree_centrality", "closeness_centrality",
       "betweenness_centrality", "eigenvector_centrality", "is_critical"]
DT = {"case_id": "category", "attrition_rate": "category", "agent_id": "int16",
      "step": "int16", "num_agents": "int16", "num_neighbors": "int16"}
print("loading...")
df = pd.read_csv(CSV, usecols=USE, dtype=DT)
df = df.sort_values(["case_id", "step", "agent_id"]).reset_index(drop=True)
df["heading_sin"] = np.sin(df.heading); df["heading_cos"] = np.cos(df.heading)
df["alive_count"] = df.groupby(["case_id", "step"], observed=True).agent_id.transform("size")
df["neighbor_frac"] = df.num_neighbors / (df.alive_count - 1).clip(lower=1)
df["is_isolated"] = (df.num_neighbors == 0).astype("int8")
NFEAT = ["speed", "accel", "heading_rate", "flight_path_angle", "heading_sin", "heading_cos",
         "dist_to_centroid", "rank_dist_to_centroid", "rel_z", "dist_to_target",
         "rank_dist_to_target", "num_neighbors", "neighbor_frac", "mean_neighbor_dist",
         "min_neighbor_dist", "weighted_density", "angular_coverage", "degree_centrality",
         "betweenness_centrality", "eigenvector_centrality", "is_isolated"]
print("node features:", len(NFEAT))


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
print(f"scenarios {len(tr)}/{len(va)}/{len(te)}")

keys = df[["case_id", "step"]].drop_duplicates()
keys["cs"] = keys.case_id.astype(str)
groups = {"train": keys[keys.cs.isin(tr)], "val": keys[keys.cs.isin(va)], "test": keys[keys.cs.isin(te)]}
limits = {"train": 8000, "val": 2500, "test": 2500}
chosen = {}
for split, g in groups.items():
    g = g.sample(n=min(limits[split], len(g)), random_state=SEED)
    chosen[split] = list(zip(g.case_id.astype(str), g.step.astype(int)))
    print(split, "snapshots:", len(chosen[split]))

# group rows for fast lookup
df["cs"] = df.case_id.astype(str)
row_groups = {k: g for k, g in df.groupby(["cs", "step"], observed=True)}


def build(case, step):
    s = row_groups[(case, step)]
    if len(s) < 4:
        return None
    x = s[NFEAT].to_numpy(np.float32)
    pos = s[["rel_x", "rel_y", "rel_z"]].to_numpy(np.float32)
    vel = s[["vx", "vy", "vz"]].to_numpy(np.float32)
    head = s.heading.to_numpy(np.float32)
    y = s.is_critical.to_numpy(np.float32)
    n = len(s)
    D = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    ii, jj = np.where(D <= 55.0)
    if len(ii) == 0:
        ii = np.arange(n); jj = np.arange(n)
    rel = pos[jj] - pos[ii]
    dist = np.linalg.norm(rel, axis=1) + 1e-6
    unit = rel / dist[:, None]
    rv = vel[jj] - vel[ii]
    rel_speed = np.linalg.norm(rv, axis=1)
    closing = np.sum(rv * unit, axis=1)  # >0 approaching
    ang = np.arctan2(rel[:, 1], rel[:, 0]) - head[ii]
    dh = head[jj] - head[ii]
    ea = np.stack([dist / 55.0, rel_speed, closing, np.cos(ang), np.sin(ang),
                   np.cos(dh), np.sin(dh)], axis=1).astype(np.float32)
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(np.stack([ii, jj])),
                edge_attr=torch.from_numpy(ea), y=torch.from_numpy(y))


print("building graphs...")
graphs = {}
for split in chosen:
    gs = []
    for i, (case, step) in enumerate(chosen[split]):
        d = build(case, step)
        if d is not None:
            gs.append(d)
    graphs[split] = gs
    print(split, "usable graphs:", len(gs))

# standardize node features on train
allx = torch.cat([g.x for g in graphs["train"]])
mu, sd = allx.mean(0), allx.std(0).clamp_min(1e-6)
for split in graphs:
    for g in graphs[split]:
        g.x = (g.x - mu) / sd

train_loader = DataLoader(graphs["train"], batch_size=24, shuffle=True)
val_loader = DataLoader(graphs["val"], batch_size=48, shuffle=False)
test_loader = DataLoader(graphs["test"], batch_size=48, shuffle=False)


class GAT(nn.Module):
    def __init__(self, fin, hidden=48, edge_dim=7):
        super().__init__()
        self.inp = nn.Linear(fin, hidden)
        self.c1 = GATv2Conv(hidden, hidden, heads=4, concat=False, edge_dim=edge_dim,
                            dropout=0.1)
        self.c2 = GATv2Conv(hidden, hidden, heads=4, concat=False, edge_dim=edge_dim,
                            dropout=0.1)
        self.n1 = nn.LayerNorm(hidden); self.n2 = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, 1))

    def forward(self, x, ei, ea):
        h = self.inp(x)
        h = h + Fn.gelu(self.n1(self.c1(h, ei, ea)))
        h = h + Fn.gelu(self.n2(self.c2(h, ei, ea)))
        return self.head(h).squeeze(-1)


def evaluate(model, loader):
    model.eval(); ps, ys, sizes = [], [], []
    with torch.no_grad():
        for b in loader:
            out = model(b.x, b.edge_index, b.edge_attr)
            ps.append(torch.sigmoid(out).cpu())
            ys.append(b.y.cpu())
            sizes.append(torch.bincount(b.batch).cpu())
    p = torch.cat(ps).numpy(); y = torch.cat(ys).numpy()
    return y, p


def report(y, p, thr):
    return {"n": int(len(y)), "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= thr)),
            "precision": float(precision_score(y, p >= thr, zero_division=0)),
            "recall": float(recall_score(y, p >= thr)),
            "mcc": float(matthews_corrcoef(y, p >= thr))}


all_y = torch.cat([g.y for g in graphs["train"]])
pos_weight = torch.tensor(float((len(all_y) - all_y.sum()) / all_y.sum()))
print("pos_weight", float(pos_weight))
model = GAT(len(NFEAT))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best = None; t0 = time.time()
for epoch in range(4):
    model.train(); tot = 0.0; nb = 0
    for b in train_loader:
        opt.zero_grad()
        out = model(b.x, b.edge_index, b.edge_attr)
        loss = lossf(out, b.y)
        loss.backward(); opt.step()
        tot += float(loss); nb += 1
    yv, pv = evaluate(model, val_loader)
    va = {"pr_auc": float(average_precision_score(yv, pv)), "roc_auc": float(roc_auc_score(yv, pv))}
    ths = np.linspace(0.02, 0.98, 97)
    thr = float(ths[int(np.argmax([f1_score(yv, pv >= t) for t in ths]))])
    va["f1"] = float(f1_score(yv, pv >= thr))
    print(f"epoch {epoch+1} loss={tot/nb:.4f} val PR-AUC={va['pr_auc']:.4f} "
          f"ROC={va['roc_auc']:.4f} F1={va['f1']:.4f} [{time.time()-t0:.0f}s]")
    if best is None or va["pr_auc"] > best[0]["pr_auc"]:
        best = (va, thr, {k: v.clone() for k, v in model.state_dict().items()})

model.load_state_dict(best[2])
yt, pt = evaluate(model, test_loader)
gat_res = report(yt, pt, best[1])
print("GAT test:", json.dumps(gat_res))

# tabular baseline on the same nodes
print("\ntabular baseline on identical snapshots...")
def stack(split, with_y=True):
    xs = torch.cat([g.x for g in graphs[split]]).numpy()
    ys = torch.cat([g.y for g in graphs[split]]).numpy()
    return xs, ys
Xtr, ytr = stack("train"); Xva, yva = stack("val"); Xte, yte = stack("test")
clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                         min_child_samples=60, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
                         random_state=0, verbose=-1)
clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False)])
pv = clf.predict_proba(Xva)[:, 1]
ths = np.linspace(0.02, 0.98, 97)
thr = float(ths[int(np.argmax([f1_score(yva, pv >= t) for t in ths]))])
pt2 = clf.predict_proba(Xte)[:, 1]
tab_res = report(yte, pt2, thr)
print("LightGBM test:", json.dumps(tab_res))

out = {"gat": gat_res, "lgbm_same_snapshots": tab_res,
       "n_train_graphs": len(graphs["train"]), "n_test_graphs": len(graphs["test"]),
       "node_features": NFEAT, "edge_features": ["dist/55", "rel_speed", "closing",
                                                 "bearing_cos", "bearing_sin",
                                                 "dheading_cos", "dheading_sin"]}
with open(OUT / "smoke_gat.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("saved", OUT / "smoke_gat.json")
