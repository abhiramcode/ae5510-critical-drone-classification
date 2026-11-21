# Critical-Node Four-Model Design — Revised Plan (post-experiment)

**Status:** investigation and preliminary experiments COMPLETE.
**Authoritative deliverable:** `report/Final_Design_Specification.md` (13 sections, Phase-7 feature tables, decision log, experiment log).
**Code/artifacts:** `experiment/` (`audit_dataset.py`, `diagnose_*.py`, `prelim_lgbm.py`, `experiments2.py`, `experiments3.py`, `smoke_gat.py`, `smoke_gru.py`, `artifacts/*.json`).
**Dataset untouched.**

---

## 1. What the experiments established

- 1,639,166 rows × 37 columns; 100 scenarios; **0 duplicates, 0 NaN**; steps sampled every 2; 197–600 sampled steps/scenario.
- Sizes {18,24,30,37,45,52,60,68,75}; attrition none/light/medium/heavy = 24/24/27/25.
- Overall positive rate **0.389** (per-scenario 0.278–0.683).
- `is_critical` is static **except 85 discrete 0→1 promotions in 41 scenarios** (49 heavy, 30 medium, 6 light, 0 none); `was_promoted` flips exactly with the label.
- **`criticality_score` is a near-oracle leak**: ROC-AUC 0.9984; perfect separation at t=0; +0.010 PR-AUC vs leak-free. It is 66% predictable from geometry (R² 0.662) — mostly geometric, partly privileged.
- **Leak-free task is learnable**: LightGBM with kinematics + local escort geometry + lags 1–16 → **PR-AUC 0.9897, ROC 0.9923, F1 0.9463**; agent-level **0.9992**.
- **Local-only** features → 0.9806 (centralized 0.9897): distributed deployment is credible.
- **Environment features add no value**: no-env 0.9834 vs 0.9825.
- **Centralities are redundant**: removing all → 0.9816 vs 0.9825.
- **Temporal window**: static 0.9782 → lag1 0.9830 → lags≤8 0.9883 → lags≤16 **0.9897** → use **W=16**.
- **Small swarms harder**: LOSO n=18 0.9046 → n=68 0.9946.
- **Promotions are the hard tail**: post-promotion recall ≈0.40; promoted-agent PR-AUC 0.838; median score jump +0.004.
- **Graph is 3D radius-55** over alive agents (299/300 snapshots within ±1); reconstruct from `rel_*`; ~0.07% row anomalies (c002/c099).
- **GAT smoke test**: 0.9879 PR-AUC vs 0.9769 for the same node features without graph → **+0.011 from message passing**.
- **GRU smoke test**: 0.9876 vs 0.9901 for a flattened-window GBDT → recurrence is for streaming/deployment, not accuracy.

## 2. Revised decisions vs the original blueprint

| Item | Original plan | Revised after evidence |
|---|---|---|
| Environment/scenario features | included | **drop** (no gain; simplifies deployment) |
| Ensemble graph centralities | degree + closeness + betweenness + eigenvector | **degree only** (others redundant) |
| Temporal window | W=16 proposed | **W=16 confirmed**; W=8 acceptable trade-off |
| Graph model | GATv2 proposed | **GATv2 confirmed** (+0.011 over tabular) |
| Temporal model | LSTM-GRU | **GRU confirmed** (accuracy parity; streaming advantage) |
| ST model | proposed | **design confirmed; full training pending** (compute) |
| `criticality_score` | exclude (assumed) | **exclude (measured: +0.010 PR-AUC only)** |
| `was_promoted` | exclude (assumed) | **exclude (verified exact transition encoding)** |
| Distributed variant | secondary | **upgraded**: local-only is within 0.009 PR-AUC |
| Promotion handling | separate metrics | **must-have**: dedicated failure mode + detachment features |
| Row anomalies | unknown | **flag/exclude 0.07%** inconsistent rows |

## 3. Final four-model inputs (definitive)

1. **Ensemble (LightGBM):** tabular `[N,≈45]` = kinematics + radial + local escort + `degree_centrality_c` + causal lags `d{1,2,4,8,16}` of distance/rank/neighbor/density/angular/speed. No env, no leak features.
2. **GRU:** `[B,16,24]` + mask; local+kinematic+degree per step; optional static head; 2 layers, hidden 96.
3. **GATv2:** `Data(x=[N,24], edge_index=[2,E], edge_attr=[E,7], global=[1,6])`; 3 layers, hidden 96, heads 4; radius-55 reconstructed graph.
4. **ST-GAT:** `[B,8,N_max,24]` + per-step edges (7) + `node_mask` + `global[B,8,6]`; shared GATv2 + per-node GRU(96).

**Never use:** `criticality_score`, `was_promoted`, `is_critical`, `case_id`/`agent_id` as inputs, future steps, post-attrition observations, simulator adjacency, label-containing aggregates.

## 4. Remaining work (implementation phase)

1. Refactor `experiment/` into a package with a shared causal feature builder (snapshot, lags, graph) and anomaly filter.
2. Ensemble: 5-seed 70/15/15 scenario-grouped runs; LOSO/LODO; per-size/attrition/promotion breakdowns; calibration.
3. GRU: W=16 full training; masked batching; streaming-state inference; promotion-subset metrics.
4. GATv2: full training with stride-subsampled timesteps; edge-feature ablation; per-size evaluation.
5. ST-GAT: T=8 full training on GPU; T=4/16 ablation; attrition masking tests; promotion-onset analysis.
6. Shared evaluation harness: PR-AUC, F1@val-threshold, recall@90%P, FN rate, Top-K, row- and agent-level, never/pre/post-promotion, early/mid/late mission.
7. Assemble the final report (`report/Final_Design_Specification.md`) with final numbers replacing preliminary ones; keep the leakage ablation as a separate non-deployable reference.

## 5. Reproduction

```powershell
uv venv experiment/.venv --python 3.12
uv pip install --python experiment/.venv/Scripts/python.exe pandas numpy scikit-learn lightgbm networkx pyarrow torch torch-geometric
experiment\.venv\Scripts\python.exe experiment\audit_dataset.py
experiment\.venv\Scripts\python.exe experiment\prelim_lgbm.py
experiment\.venv\Scripts\python.exe experiment\experiments2.py
experiment\.venv\Scripts\python.exe experiment\experiments3.py
experiment\.venv\Scripts\python.exe experiment\smoke_gat.py
experiment\.venv\Scripts\python.exe experiment\smoke_gru.py
```

Cache note: `uv` stores wheels and ephemeral `--with` envs in `A:\PythonWorkspace\uv_cache` (7.5 GB currently). Reclaim with `uv cache prune` (safe, unreferenced only) or `uv cache clean <package>` (forces re-download later); the project venv is unaffected.
