"""Small scenario-grouped smoke test of the temporal (GRU) design.

Windows of W=8 causal samples per agent; compares a GRU against a tabular model on the
flattened window (identical information) to isolate the value of recurrence.
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
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef)
import lightgbm as lgb

torch.set_num_threads(8)
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "dataset" / "swarm_criticality_dataset_100_scenarios.csv"
OUT = Path(__file__).resolve().parent / "artifacts"
SEED = 0
W = 8

USE = ["case_id", "num_agents", "attrition_rate", "step", "agent_id", "rel_z",
       "dist_to_centroid", "rank_dist_to_centroid", "dist_to_target", "rank_dist_to_target",
       "speed", "heading", "flight_path_angle", "accel", "heading_rate", "num_neighbors",
       "mean_neighbor_dist", "min_neighbor_dist", "weighted_density", "angular_coverage",
       "degree_centrality", "closeness_centrality", "betweenness_centrality",
       "eigenvector_centrality", "is_critical"]
DT = {"case_id": "category", "attrition_rate": "category", "agent_id": "int16",
      "step": "int16", "num_agents": "int16", "num_neighbors": "int16"}
print("loading...")
df = pd.read_csv(CSV, usecols=USE, dtype=DT)
df = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)
df["heading_sin"] = np.sin(df.heading); df["heading_cos"] = np.cos(df.heading)
df["alive_count"] = df.groupby(["case_id", "step"], observed=True).agent_id.transform("size")
df["neighbor_frac"] = df.num_neighbors / (df.alive_count - 1).clip(lower=1)
df["is_isolated"] = (df.num_neighbors == 0).astype("int8")
FEAT = ["speed", "accel", "heading_rate", "flight_path_angle", "heading_sin", "heading_cos",
        "dist_to_centroid", "rank_dist_to_centroid", "rel_z", "dist_to_target",
        "rank_dist_to_target", "num_neighbors", "neighbor_frac", "mean_neighbor_dist",
        "min_neighbor_dist", "weighted_density", "angular_coverage", "degree_centrality",
        "betweenness_centrality", "eigenvector_centrality", "is_isolated"]
print("features:", len(FEAT))


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
cs = df.case_id.astype(str).to_numpy()
pos = df.groupby(["case_id", "agent_id"], observed=True).cumcount().to_numpy()
arr = df[FEAT].to_numpy(np.float32)
y_all = df.is_critical.to_numpy(np.int64)
ends_all = np.flatnonzero(pos >= W - 1)
print("valid windows total:", len(ends_all))
rng = np.random.default_rng(SEED)
limits = {"train": 150_000, "val": 40_000, "test": 40_000}
split_ids = {"train": tr, "val": va, "test": te}
data = {}
for split, ids in split_ids.items():
    mask = np.array([c in ids for c in cs[ends_all]])
    cand = ends_all[mask]
    if len(cand) > limits[split]:
        cand = rng.choice(cand, size=limits[split], replace=False)
    X = np.empty((len(cand), W, len(FEAT)), dtype=np.float32)
    for k, e in enumerate(cand):
        X[k] = arr[e - W + 1:e + 1]
    data[split] = (X, y_all[cand])
    print(split, X.shape, "pos", float(y_all[cand].mean()))

mu = data["train"][0].reshape(-1, len(FEAT)).mean(0)
sd = data["train"][0].reshape(-1, len(FEAT)).std(0) + 1e-6
for split in data:
    X, y = data[split]
    data[split] = ((X - mu) / sd, y)


class GRU(nn.Module):
    def __init__(self, fin, hidden=64):
        super().__init__()
        self.gru = nn.GRU(fin, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


def batches(X, y, bs, shuffle):
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield torch.from_numpy(X[j]), torch.from_numpy(y[j]).float()


def evaluate(model, X, y, bs=2048):
    model.eval(); ps = []
    with torch.no_grad():
        for xb, _ in batches(X, y, bs, False):
            ps.append(torch.sigmoid(model(xb)).cpu())
    return torch.cat(ps).numpy()


def report(y, p, thr):
    return {"n": int(len(y)), "pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= thr)),
            "precision": float(precision_score(y, p >= thr, zero_division=0)),
            "recall": float(recall_score(y, p >= thr)),
            "mcc": float(matthews_corrcoef(y, p >= thr))}


pos_weight = torch.tensor(float((len(data["train"][1]) - data["train"][1].sum())
                                / data["train"][1].sum()))
print("pos_weight", float(pos_weight))
model = GRU(len(FEAT))
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
best = None; t0 = time.time()
for epoch in range(5):
    model.train(); tot = 0.0; nb = 0
    for xb, yb in batches(data["train"][0], data["train"][1], 512, True):
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        tot += float(loss.detach()); nb += 1
    pv = evaluate(model, data["val"][0], data["val"][1])
    yv = data["val"][1]
    ths = np.linspace(0.02, 0.98, 97)
    thr = float(ths[int(np.argmax([f1_score(yv, pv >= t) for t in ths]))])
    pr = float(average_precision_score(yv, pv)); roc = float(roc_auc_score(yv, pv))
    print(f"epoch {epoch+1} loss={tot/nb:.4f} val PR-AUC={pr:.4f} ROC={roc:.4f} "
          f"F1={f1_score(yv, pv >= thr):.4f} [{time.time()-t0:.0f}s]")
    if best is None or pr > best[0]:
        best = (pr, thr, {k: v.clone() for k, v in model.state_dict().items()})
model.load_state_dict(best[2])
pt = evaluate(model, data["test"][0], data["test"][1])
gru_res = report(data["test"][1], pt, best[1])
print("GRU test:", json.dumps(gru_res))

# tabular baseline on the flattened window (identical information)
print("\ntabular baseline on flattened windows...")
Xtr = data["train"][0].reshape(len(data["train"][0]), -1)
Xva = data["val"][0].reshape(len(data["val"][0]), -1)
Xte = data["test"][0].reshape(len(data["test"][0]), -1)
clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=127,
                         min_child_samples=60, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, class_weight="balanced", n_jobs=-1,
                         random_state=0, verbose=-1)
clf.fit(Xtr, data["train"][1], eval_set=[(Xva, data["val"][1])],
        eval_metric="average_precision", callbacks=[lgb.early_stopping(40, verbose=False)])
pv = clf.predict_proba(Xva)[:, 1]
ths = np.linspace(0.02, 0.98, 97)
thr = float(ths[int(np.argmax([f1_score(data["val"][1], pv >= t) for t in ths]))])
pt2 = clf.predict_proba(Xte)[:, 1]
tab = report(data["test"][1], pt2, thr)
print("LightGBM flattened:", json.dumps(tab))

out = {"gru": gru_res, "lgbm_flattened": tab, "W": W, "features": FEAT,
       "n_train": int(len(data["train"][1])), "n_test": int(len(data["test"][1]))}
with open(OUT / "smoke_gru.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("saved", OUT / "smoke_gru.json")
