# Final Minimal Implementation Blueprint: Aerial Swarm Critical-Node Identification

**Target Audience:** Implementation Agent (Gemini Flash)  
**Authoritative Dataset:** `dataset/swarm_criticality_dataset_100_scenarios.csv` (1,639,166 rows × 37 columns, read-only, **must never be modified**)  
**Reference Documents:** `dataset/Dataset_Metadata.md`, `report/Final_Design_Specification.md`, `experiment/*.py`, `experiment/artifacts/*.json`  
**Operational Setting (Locked):** Centralized fused snapshot with causal history $\le t$ (see §4.1).

---

## 1. Project Objective

Implement, train, and evaluate **four** leak-free binary classification models that identify whether each currently alive aerial swarm agent is a **critical asset (`is_critical == True`, class `1`)** or a **decoy (`is_critical == False`, class `0`)** at the current sampled timestep $t$:

- **Task type:** Causal current-state per-agent binary classification at timestep $t$, using only telemetry available at or before $t$.
- **Target:** `is_critical(case_id, step, agent_id) ∈ {0, 1}`.
- **Primary deployment boundary:** Centralized fusion node receiving periodic broadcast telemetry from all currently alive agents (every 2 simulation steps) and retaining a short causal history buffer.

---

## 2. Validated Research Findings (Implementation-Relevant)

All claims below were verified directly against `experiment/artifacts/*.json`, `experiment/*.py`, and dataset inspection:

| Finding | Verified Fact | Implementation Consequence |
|---|---|---|
| Dataset integrity | 1,639,166 rows × 37 cols; 100 scenarios (`c001`–`c100`); 0 NaNs; 0 duplicate `(case_id, step, agent_id)` keys. | No imputation needed for raw columns; key sorting is deterministic. |
| Sampling & continuity | `step` is strictly even (`0, 2, ..., step_max`); 197–600 sampled steps/scenario; every agent starts at `step=0` with `Δstep=2` until permanent attrition (0 gaps, 0 reappearances). | `1 sampled index` = `2 simulation steps`. `W=16` samples = 32 sim steps; `T=8` samples = 16 sim steps. |
| Swarm sizes & attrition | 9 sizes `{18, 24, 30, 37, 45, 52, 60, 68, 75}`; 4 attrition conditions (`none`: 24, `light`: 24, `medium`: 27, `heavy`: 25). Positive rate `0.389` overall (`0.278`–`0.683` per scenario). | Use `class_weight="balanced"` / `BCEWithLogitsLoss(pos_weight)`; support variable $N \le 75$. |
| Dynamic labels | `is_critical` is constant per agent except **85 discrete `0→1` promotions** across 41 scenarios (0 `1→0`). `was_promoted` flips `0→1` identically with those transitions. | Exclude `was_promoted` from all inputs; report `never_promoted`, `pre_promotion`, `post_promotion` breakdowns. |
| `criticality_score` leakage | Single-feature ROC-AUC `0.9984`, PR-AUC `0.9979`; perfect `t=0` separation (decoys $\le 0.0747$, critical $\ge 0.2911$); `R² = 0.662` from geometry. | Strictly exclude from all model inputs. |
| Velocity & rank semantics | `vx, vy, vz` match `speed, heading, flight_path_angle` to `3.6e-15` (world-frame velocity). `rank_dist_to_{centroid,target}` match `(rank_avg - 1) / max(n - 1, 1)` with `0.0` error. | `swarm_centroid_speed = ||mean(v_i)||_2` is exact; ranks are recomputable over alive agents. |
| Isolated agents | 1,895 rows have `num_neighbors == 0`; in all of them `mean_neighbor_dist = min_neighbor_dist = weighted_density = angular_coverage = 0.0`. | `is_isolated` explicitly disambiguates `0.0` neighbor distance from zero separation. |
| Graph definition | 3D Euclidean radius-55 graph over alive agents (`299/300` snapshots within `±1` of `num_neighbors`). 2D fails (`42/400` exact). | Reconstruct undirected 3D radius-55 graph from `(rel_x, rel_y, rel_z)`; never use raw centrality columns. |
| Redundancy & environment | Environment features add `0` gain (`no env` PR-AUC `0.9834` vs `0.9825`); `degree↔closeness` Spearman `0.993`; `obstacle_clustering > 0` even when `n_obstacles = 0`. | Drop all 7 scenario/environment columns and drop closeness/betweenness/eigenvector centralities. |
| Anomalies | `disp / (2 * mean_speed) > 3.0` occurs in 520 rows (`0.0317%`, concentrated in `c002`: `2.37%` and `c099`: `0.23%`); `> 1.5` occurs in 1,212 rows (`0.0739%`). | Flag `is_anomaly = (ratio > 3.0)`; mask/exclude from training loss without breaking the step grid; evaluate full test set. |
| Hardware & environment | `experiment/.venv` has Python 3.12, 16 CPU cores, `torch 2.15.0+cpu`, `pyg 2.8.1`, `lightgbm 4.7.0`. Machine GPU is **RTX 3050 Laptop (4 GB VRAM)** (CUDA PyTorch wheel not yet installed). | Enforce memory-safe batch sizes (`B=8` for ST-GAT), `device="auto"`, and `train_stride=4` for ST-GAT. |

### Discrepancies Resolved Against `report/Final_Design_Specification.md`

1. **Exact feature count (`≈45` vs `54`, `24` GRU vs `23`):** `Final_Design_Specification.md` §5.3 wrote `≈45` while §5.2 listed 24 base + 30 lag-diff features (`54`), and `experiments3.py` ran `59` (including 7 env + 4 raw centralities). Section 6.3 (GRU) listed 23 features by omitting `n_within_r15`, whereas §7.4 (GATv2) mentioned `neighbor_rank_mean`. **Resolution:** Lock one canonical **`BASE_FEATURES_24`** vector shared identically across all four models (see §4.3); LightGBM uses `BASE_FEATURES_24` + `LAG_DIFF_FEATURES_30` = **54 features**; GRU, GATv2, and ST-GAT use **`BASE_FEATURES_24`** (**24 features**).
2. **Primary split size (`66/17/17` vs `70/15/15`):** `stratified_split()` in `prelim_lgbm.py` produced `66/17/17` because 19 of 36 `(size × attrition)` cells have `< 3` scenarios. **Resolution:** Use a deterministic balanced allocation algorithm (see §7.2) to produce an **exact 70 / 15 / 15** scenario split saved once to `artifacts/splits/primary_split.json`.
3. **Edge closing-rate sign & self-loops in `smoke_gat.py`:** `smoke_gat.py` only added self-loops when `len(ii) == 0` (leaving isolated nodes in connected graphs without self-loops) and computed `ang` in sender frame with inverted closing-rate sign. **Resolution:** Explicitly add self-loops `(i, i)` for every alive node, set `add_self_loops=False` in `GATv2Conv`, and define edge vectors in the **receiver (`dst`) frame** (see §6.2).
4. **GRU optional `[B, 3]` "static" head:** §6.4 listed `[swarm_spread, swarm_centroid_speed, graph_density]` as a static head, though all three vary per timestep and were untested in `smoke_gru.py`. **Resolution:** Omit the `[B, 3]` head so GRU strictly takes `[B, 16, 24]` + `mask[B, 16]` and maintains clean O(1) streaming state.
5. **GRU early-mission padding (`k < 15`):** `smoke_gru.py` discarded `k < W - 1`. **Resolution:** Left-pad shorter histories to length 16 with `mask[B, 16]` and use `pack_padded_sequence` inside `forward()` so all `k >= 0` (including `t=0`) are trained and evaluated without padding contamination.

---

## 3. Final Architecture Decisions

Do **not** add LSTM, BiGRU, GCN, GraphSAGE, GIN, ST-GCN, TGN, or Transformer variants.

| Model | Architecture | Key Hyperparameters | Input Shape |
|---|---|---|---|
| **LightGBM** | Binary GBDT (`LGBMClassifier`) | `n_estimators=900`, `lr=0.05`, `num_leaves=127`, `min_child_samples=60`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `class_weight="balanced"` | `[N, 54]` (`float32`, NaNs allowed for early lags) |
| **GRU** | 2-layer unidirectional `nn.GRU` + MLP head | `hidden_size=96`, `num_layers=2`, `dropout=0.1`, `W=16`, left-padded + `pack_padded_sequence` | `x: [B, 16, 24]`, `mask: [B, 16]` |
| **GATv2** | 3-layer `GATv2Conv` + edge MLP + global re-concat + MLP head | `hidden=96`, `heads=4`, `concat=False`, `edge_dim=32` (from 7), `add_self_loops=False`, `dropout=0.1` | `x: [N, 24]`, `edge_index: [2, E]`, `edge_attr: [E, 7]`, `global: [1, 6]` |
| **ST-GAT** | Shared 3-layer `GATv2` spatial encoder + per-node 1-layer `GRU` | Spatial: same 3-layer GATv2 (`96`, `4` heads); Temporal: `GRU(96+6, 96, num_layers=1)`, `T=8` | `x: [B, 8, N_max, 24]`, `node_mask: [B, 8, N_max]`, per-step `(edge_index, edge_attr)`, `global: [B, 8, 6]` |

---

## 4. Shared Data and Feature Pipeline

All four models MUST consume features generated by one shared `features` pipeline (never four separate preprocessing implementations).

### 4.1 Operational Setting & Inference Boundary

- **Primary mode:** Centralized fused snapshot. At sampled step $t$, the pipeline has access to all currently alive agents' telemetry at step $t$ and their causal history at steps $\le t$.
- **Distributed variant (not implemented):** Local-only (`0.9806` vs `0.9897` PR-AUC) is out of scope for the primary codebase; document only as a future extension via allowlist filtering.

### 4.2 Exact Processing Sequence

1. **Load & Validate Raw CSV:** Read `dataset/swarm_criticality_dataset_100_scenarios.csv` with explicit dtypes (`case_id`: str, `step`: int32, `agent_id`: int32, numeric cols: float32/int32, labels: bool). Assert 0 NaNs, 0 duplicate `(case_id, step, agent_id)`, and `step % 2 == 0`.
2. **Deterministic Sort:** Sort strictly by `["case_id", "step", "agent_id"]` (for snapshot/graph iteration) and maintain an index/view sorted by `["case_id", "agent_id", "step"]` (for causal per-agent lag/sequence extraction).
3. **Causal Anomaly Flagging (`is_anomaly`):** For each `(case_id, agent_id)`, compute `disp = ||rel(t) - rel(t-2)||_2` and `expected_disp = 2.0 * 0.5 * (speed(t) + speed(t-2))`. Mark `is_anomaly = (step > 0) & ((disp / np.maximum(expected_disp, 1e-6)) > 3.0)`. Never drop rows from the timeline; pass `is_anomaly` as a training-loss mask only.
4. **Snapshot Graph & Derived Geometry (Per `(case_id, step)`):**
   - Let $N_t$ be alive agents at step $t$ ordered by `agent_id`.
   - Compute pairwise 3D Euclidean distances $D_{ij} = \| \text{rel}_i - \text{rel}_j \|_2$ from `["rel_x", "rel_y", "rel_z"]`.
   - Reconstruct undirected radius-55 adjacency: $A_{ij} = \mathbb{I}[i \neq j \land D_{ij} \le 55.0]$.
   - Compute `degree_centrality_c`, `n_within_r{15,30,45}`, `nn_bearing_entropy`, `local_density_contrast`, `centroid_vel_alignment`, `is_isolated`, `heading_{sin,cos}`, and `GLOBAL_FEATURES_6`.
   - Recompute/verify `rank_dist_to_centroid` and `rank_dist_to_target` over alive agents at $t$ as `(rank_avg - 1) / max(N_t - 1, 1)` (needed at inference time if raw rank columns are not pre-supplied; matches dataset columns to `0.0` error).
5. **Enforce Inference-Time Allowlist:** Select strictly from the allowlisted columns below; raise `AssertionError` if any prohibited column (`criticality_score`, `was_promoted`, `is_critical`, `case_id`, `agent_id`, env cols, raw centralities, `rel_x`, `rel_y`, `vx`, `vy`, `vz`) enters a feature matrix.
6. **Model-Specific Adapters:**
   - `to_lgbm_table()`: computes `LAG_DIFF_FEATURES_30` causally per `(case_id, agent_id)` and concatenates `[BASE_FEATURES_24, LAG_DIFF_FEATURES_30]` $\rightarrow$ `[N, 54]`.
   - `to_gru_sequences()`: slices/left-pads causal windows of length `W=16` over `BASE_FEATURES_24` $\rightarrow$ `(X [N_samples, 16, 24], mask [N_samples, 16])`.
   - `to_gat_graphs()`: builds per-snapshot `Data(x=[N_t, 24], edge_index=[2, E_t], edge_attr=[E_t, 7], global_feat=[1, 6])`.
   - `to_stgat_windows()`: assembles `T=8` consecutive snapshots per window with dynamic `(edge_index_tau, edge_attr_tau)` and `node_mask [T, N_w]`.

### 4.3 Definitive Feature Contract: `BASE_FEATURES_22` (Shared Exact Order)

Used identically as the 22 per-step features for **GRU**, 22 node features for **GATv2**, 22 per-step node features for **ST-GAT**, and first 22 columns for **LightGBM**. Target-related features (`dist_to_target`, `rank_dist_to_target`) have been intentionally dropped to guarantee target-agnostic / adversarial operational capability.

| Idx | Feature Name | Source / Derivation | Avail. | Exact Computation |
|---:|---|---|---|---|
| 0 | `speed` | Dataset col / Velocity norm | A | `||v_i||_2 = sqrt(vx^2 + vy^2 + vz^2)` |
| 1 | `accel` | Dataset col / Finite diff | A | `Δspeed / Δt` |
| 2 | `heading_rate` | Dataset col / Finite diff | A | `wrap(Δheading) / Δt` |
| 3 | `flight_path_angle` | Dataset col / Velocity ratio | A | `atan2(vz, sqrt(vx^2 + vy^2))` |
| 4 | `heading_sin` | Derived (`heading`) | B | `np.sin(heading)` |
| 5 | `heading_cos` | Derived (`heading`) | B | `np.cos(heading)` |
| 6 | `dist_to_centroid` | Dataset / `rel_*` | A/B | `||[rel_x, rel_y, rel_z]||_2` (exact match to col) |
| 7 | `rank_dist_to_centroid` | Derived (`dist_to_centroid`) | B | `(rank_average(dist_to_centroid) - 1) / max(N_t - 1, 1)` |
| 8 | `rel_z` | Dataset col | A | Raw `rel_z = z - c_z` |
| 9 | `num_neighbors` | Dataset col | A | `sum_{j != i} I[D_ij <= 55.0]` |
| 10 | `centroid_vel_alignment` | Derived (`vx, vy, vz`, centroid vel) | B | `(v_i · v_c) / (||v_i|| * ||v_c|| + 1e-6)` (replaces redundant `neighbor_frac`) |
| 11 | `is_isolated` | Derived (`num_neighbors`) | B | `float(num_neighbors == 0)` (`0.0` or `1.0`) |
| 12 | `mean_neighbor_dist` | Dataset col | A | Raw `mean_neighbor_dist` (`55.0` when isolated) |
| 13 | `min_neighbor_dist` | Dataset col | A | Raw `min_neighbor_dist` (`55.0` when isolated) |
| 14 | `weighted_density` | Dataset col | A | `sum_{j != i: D_ij <= 55} exp(-D_ij^2 / 324)` |
| 15 | `angular_coverage` | Dataset col | A | Fraction of 48 spherical sectors (12 azimuth x 4 elevation) occupied |
| 16 | `n_within_r15` | Derived (`rel_*`) | B | `sum_{j != i} I[D_ij <= 15.0]` |
| 17 | `n_within_r30` | Derived (`rel_*`) | B | `sum_{j != i} I[D_ij <= 30.0]` |
| 18 | `n_within_r45` | Derived (`rel_*`) | B | `sum_{j != i} I[D_ij <= 45.0]` |
| 19 | `nn_bearing_entropy` | Derived (`rel_*`, `heading`) | B | Normalized 12-bin azimuthal Shannon entropy of radius-55 neighbors in `i`'s heading frame (see below) |
| 20 | `local_density_contrast` | Derived (`weighted_density`, graph) | B | `weighted_density_i - mean_{j in N_55(i)}(weighted_density_j)` (`0.0` if `|N_55(i)| == 0`) |
| 21 | `degree_centrality_c` | Derived (`rel_*`) | B | `sum_{j != i} I[D_ij <= 55.0] / max(N_t - 1, 1)` |

**Exact formula for `nn_bearing_entropy` (index 21):**  
For agent $i$, let $N_{55}(i) = \{j \neq i : D_{ij} \le 55.0\}$. If $|N_{55}(i)| == 0$, return `0.0`. Otherwise, for each $j \in N_{55}(i)$, compute relative bearing $\theta_{ij} = \text{wrap}_{[-\pi, \pi)}(\text{atan2}(\text{rel}_{y,j} - \text{rel}_{y,i},\ \text{rel}_{x,j} - \text{rel}_{x,i}) - \text{heading}_i)$, bin into $12$ equal sectors of width $2\pi/12$, let $p_b = c_b / |N_{55}(i)|$, and return $-\sum_{b: c_b > 0} p_b \ln(p_b) / \ln(12) \in [0, 1]$.

### 4.4 Additional Shared Feature Groups

#### `LAG_DIFF_FEATURES_30` (LightGBM only, indices 24–53)
Causal within-`(case_id, agent_id)` differences over lag offsets $k \in \{1, 2, 4, 8, 16\}$ sampled steps ($2, 4, 8, 16, 32$ sim steps):
- Source variables (6): `dist_to_centroid` (`dist`), `rank_dist_to_centroid` (`rank`), `num_neighbors` (`nb`), `weighted_density` (`dens`), `angular_coverage` (`ang`), `speed` (`speed`).
- Ordered strictly as outer loop `var in [dist, rank, nb, dens, ang, speed]`, inner loop `lag in [1, 2, 4, 8, 16]`:
  `d{lag}_{tag}(t) = x(t) - x(t - 2*lag)` if agent index $m \ge \text{lag}$, else `NaN`.

#### `EDGE_FEATURES_7` (GATv2 & ST-GAT, per directed edge `src -> dst`)
| Idx | Name | Source | Exact Computation (for edge `src -> dst`; self-loop `src == dst` in brackets) |
|---:|---|---|---|
| 0 | `dist_norm` | `rel_*` | `||rel[src] - rel[dst]||_2 / 55.0` `[0.0]` |
| 1 | `rel_speed` | `vx, vy, vz` | `||v[src] - v[dst]||_2` `[0.0]` |
| 2 | `closing_rate` | `rel_*`, `v_*` | `-dot(v[src] - v[dst], (rel[src] - rel[dst]) / (dist + 1e-6))` (`>0` = approaching) `[0.0]` |
| 3 | `bearing_cos` | `rel_*`, `heading` | `cos(atan2(dy, dx) - heading[dst])` where `dx, dy = (rel[src] - rel[dst])[:2]` `[1.0]` |
| 4 | `bearing_sin` | `rel_*`, `heading` | `sin(atan2(dy, dx) - heading[dst])` `[0.0]` |
| 5 | `dheading_cos` | `heading` | `cos(heading[src] - heading[dst])` `[1.0]` |
| 6 | `dheading_sin` | `heading` | `sin(heading[src] - heading[dst])` `[0.0]` |

#### `GLOBAL_FEATURES_6` (GATv2 & ST-GAT, per snapshot $t$)
| Idx | Name | Source | Exact Computation (over alive nodes $V_t$, reconstructed graph $G_t$) |
|---:|---|---|---|
| 0 | `alive_count` | `V_t` | `float(N_t)` |
| 1 | `swarm_spread` | `dist_to_centroid` | `sqrt(mean_{i in V_t}(dist_to_centroid_i^2))` |
| 2 | `swarm_centroid_speed` | `vx, vy, vz` | `||mean_{i in V_t}([vx_i, vy_i, vz_i])||_2` |
| 3 | `graph_density` | `G_t` (undirected, no self-loops) | `2 * |E_undir| / (N_t * (N_t - 1))` if `N_t >= 2` else `0.0` |
| 4 | `n_components` | `G_t` | `float(connected_components(G_t))` (via `scipy.sparse.csgraph`) |
| 5 | `largest_component_frac` | `G_t` | `max_component_size / max(N_t, 1)` |

*Duplication check:* Zero overlap between `BASE_FEATURES_24` and `GLOBAL_FEATURES_6`.

---

## 5. Model-Specific Specifications

### 5.1 Model 1 — Ensemble (`LightGBM`)

- **Architecture:** `lightgbm.LGBMClassifier(n_estimators=900, learning_rate=0.05, num_leaves=127, min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1)`.
- **Exact Features (`54`):** `BASE_FEATURES_24` (0–23) + `LAG_DIFF_FEATURES_30` (24–53).
- **Tensor / Array Dimensions:** `X: np.ndarray[float32]` of shape `[N_rows, 54]`, `y: np.ndarray[int8]` of shape `[N_rows]`.
- **Input Construction & Preprocessing:** No feature scaling. Keep `NaN` in `LAG_DIFF_FEATURES_30` when `agent_step_idx < lag`. Exclude `is_anomaly == True` rows from training `fit()`; retain all rows for val/test evaluation.
- **Training Configuration:** Early stopping `50` rounds monitoring validation `average_precision` (`PR-AUC`). Save booster (`model.txt`) + config + feature schema.
- **Inference Procedure:** Accepts `history_df` up to $t$ (at least 1 row at $t$, ideally 17 sampled steps `[t-32..t]` for full lags); computes `BASE_FEATURES_24` on snapshot(s) + causal lags at $t$, outputs `[prob, pred_class]` aligned to `(case_id, step, agent_id)` at $t$.

### 5.2 Model 2 — Recurrent (`2-Layer Unidirectional GRU`)

- **Architecture:**
  - `gru = nn.GRU(input_size=24, hidden_size=96, num_layers=2, batch_first=True, dropout=0.1)`
  - `head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 1))`
  - Forward with left-padded `(x [B, 16, 24], mask [B, 16])`: compute `lengths = mask.sum(dim=1).long().clamp(min=1)`; shift each row's `lengths[b]` valid steps to indices `0..lengths[b]-1` (right-padded), apply `pack_padded_sequence(..., enforce_sorted=False)`, run `gru`, take top-layer final state `h_n[-1]` (`[B, 96]`), pass through `head` $\rightarrow$ logits `[B]`.
- **Exact Features (`24`):** `BASE_FEATURES_24` per timestep.
- **Tensor Dimensions:** `x: float32[B, 16, 24]`, `mask: bool[B, 16]`, `y: float32[B]`.
- **Input Construction & Preprocessing:** Standardize `BASE_FEATURES_24` using `(mu, sd)` fitted strictly on non-anomalous, non-padded training rows (`sd = clamp_min(std, 1e-6)`). Zero out padded positions (`mask == False`) *after* normalization so padded slots are exact `0.0`. Exclude windows whose target row `(t)` is `is_anomaly == True` during training.
- **Training Configuration:** `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=20)`, `BCEWithLogitsLoss(pos_weight=neg_train/pos_train)`, `batch_size=512`, `max_epochs=20`, early stopping `patience=5` on val `PR-AUC`, `clip_grad_norm_(1.0)`, `seed=0`, `device="auto"`.
- **Inference Procedure:**
  - **Window mode:** Accepts causal history `[t-30..t]` (up to 16 steps), left-pads if `< 16`, returns aligned predictions at $t$.
  - **True O(1) streaming mode:** Maintains `h_state: dict[agent_id, Tensor[2, 1, 96]]`. At new snapshot $t$: normalize `x_t [N_t, 24]`, initialize `h = zeros(2, 1, 96)` for new agents, drop disappeared agents from `h_state`, run single-step `out, h_new = gru(x_t.unsqueeze(1), h_cat)` (`[N_t, 1, 24]` + `[2, N_t, 96]`), classify `head(out[:, 0, :])`, store `h_new` in `h_state`. Mathematically identical to `pack_padded_sequence` window output.

### 5.3 Model 3 — Spatial Graph (`3-Layer GATv2`)

- **Architecture:**
  - `node_in = nn.Linear(24, 96)`
  - `edge_mlp = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 32))`
  - 3 blocks `l = 1, 2, 3`:
    - `conv_l = GATv2Conv(96, 96, heads=4, concat=False, edge_dim=32, dropout=0.1, add_self_loops=False)`
    - `norm_l = nn.LayerNorm(96)`
    - `gfuse_l = nn.Linear(96 + 6, 96)`
    - Block computation: `h = h + F.gelu(norm_l(conv_l(h, edge_index, edge_emb)))`; `h = h + gfuse_l(torch.cat([h, global_feat[batch]], dim=-1))`
  - `head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 96), nn.GELU(), nn.Dropout(0.1), nn.Linear(96, 1))`
- **Exact Features:** Node `BASE_FEATURES_24` (`24`), Edge `EDGE_FEATURES_7` (`7`), Global `GLOBAL_FEATURES_6` (`6`).
- **Tensor Dimensions (PyG `Data` / `Batch`):** `x: [N_batch, 24]`, `edge_index: [2, E_batch]`, `edge_attr: [E_batch, 7]`, `global_feat: [B, 6]`, `batch: [N_batch]`, `y: [N_batch]`, `loss_mask: bool[N_batch]` (`~is_anomaly` during training).
- **Input Construction & Preprocessing:** Standardize `x` (`mu_x, sd_x`), `edge_attr` (`mu_e, sd_e`), and `global_feat` (`mu_g, sd_g`) on training split only. Always include explicit self-loops `(i, i)` in `edge_index` with `EDGE_FEATURES_7` self-loop values normalized by `(mu_e, sd_e)`.
- **Training Configuration:** PyG `DataLoader(batch_size=32, shuffle=True)`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=25)`, `BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")` masked by `loss_mask`, `max_epochs=25`, early stopping `patience=5` on val `PR-AUC`, `clip_grad_norm_(1.0)`, `seed=0`, `device="auto"`.
- **Inference Procedure:** Single-snapshot inference (`predict(snapshot_df_t)`). Builds radius-55 graph from `rel_*`, normalizes with saved scalers, runs forward pass, returns predictions aligned to `agent_id` at $t$.

### 5.4 Model 4 — Spatio-Temporal (`Shared GATv2 + Per-Node GRU`)

- **Architecture:**
  - **Shared spatial encoder:** Identical 3-layer GATv2 backbone as Model 3 (omitting Model 3's final `head`), mapping alive nodes at step $\tau$ to `z_{b, \tau, u} ∈ R^{96}`.
  - **Vectorized per-step spatial pass:** Given batch of `B` windows of `T=8` steps padded to `N_batch_max <= 75`: gather all alive nodes where `node_mask[b, tau, u] == True`, offset each `(b, tau)` subgraph's `edge_index` into one flat PyG batch (`[N_alive_total, 24]`), run spatial backbone once, scatter outputs into `Z: [B, 8, N_batch_max, 96]` (zeros where `node_mask == False`).
  - **Per-node temporal encoder:** Select only target nodes alive at the window end (`target_mask = node_mask[:, -1, :]` $\rightarrow$ `N_eval` nodes). Concatenate per-step global vector `global_t` (`[B, 8, 6]` broadcast to `[N_eval, 8, 6]`) with `Z` $\rightarrow$ `seq_in [N_eval, 8, 102]`. Compute valid length `L = node_mask_eval.sum(dim=1)` (`1..8`), pack with `pack_padded_sequence` (shifting valid steps left, identical to Model 2), run `nn.GRU(102, 96, num_layers=1, batch_first=True)`, take `h_n[-1]` (`[N_eval, 96]`), pass through `head` (`LayerNorm(96) -> Linear(96, 96) -> GELU -> Dropout(0.1) -> Linear(96, 1)`) $\rightarrow$ logits `[N_eval]`.
- **Exact Features:** Per-step Node `BASE_FEATURES_24` (`24`), Per-step Edge `EDGE_FEATURES_7` (`7`), Per-step Global `GLOBAL_FEATURES_6` (`6`).
- **Tensor Dimensions:** `x: [B, 8, N_max, 24]`, `node_mask: bool[B, 8, N_max]`, `global_feat: [B, 8, 6]`, per-step sparse `(edge_index_{b,tau}, edge_attr_{b,tau})`, `y_end: [B, N_max]`, `anomaly_end: bool[B, N_max]`.
- **Input Construction & Preprocessing:** Reuse Model 3's train-fitted `(mu_x, sd_x, mu_e, sd_e, mu_g, sd_g)`. Zero out `x` where `node_mask == False` after normalization. Exclude `anomaly_end == True` nodes from training loss via `loss_mask`.
- **Training Configuration:**
  - `batch_size=8` windows (fits comfortably within **4 GB VRAM** on RTX 3050; max `8 × 8 × 75 = 4,800` nodes/batch).
  - **Training timestep stride:** `train_stride=4` (sample every 4th snapshot step per scenario during training $\approx 6,600$ train windows/epoch; **validation and test MUST use `stride=1`** on all snapshots).
  - `AdamW(lr=1e-3, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=20)`, `BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")` masked by `~anomaly_end`, `max_epochs=20`, early stopping `patience=5` on full val `PR-AUC`, `clip_grad_norm_(1.0)`, `seed=0`, `device="auto"`.
- **Inference Procedure:** Requires a causal historical buffer of up to `T=8` consecutive sampled snapshots `[t-14..t]` (left-padded if `t < 14`). Not O(1)-state streaming; supports rolling ring-buffer online inference at cost `O(8 × graph_size)`.

---

## 6. Temporal and Graph Handling Rules

### 6.1 Temporal Rules

1. **Step spacing:** `step` increments by `2` per sampled observation. Index $k = \text{step} / 2$.
2. **Causal window retrieval:** Window of length $W$ ending at $k$ selects indices $[\max(0, k - W + 1), \dots, k]$.
3. **Left-padding & positioning:** When $k < W - 1$, place the $k + 1$ real observations at tensor indices $[W - k - 1, \dots, W - 1]$ and zeros at $[0, \dots, W - k - 2]$. **Current observation $k$ is always at index `-1` (`W - 1`).**
4. **Masks:** `mask[i] = True` for real observations, `False` for left-padded slots. Never apply `GRU` steps to padded zeros (`pack_padded_sequence` on valid slice).
5. **No cross-scenario or future leakage:** Windows strictly stay within one `case_id` and never access `step > t`.

### 6.2 Graph Construction Rules

1. **Nodes:** Alive agents at snapshot `t` sorted by ascending `agent_id`.
2. **Distance metric:** **3D Euclidean** distance using `["rel_x", "rel_y", "rel_z"]` (`D_ij <= 55.0`). Never use 2D distance.
3. **Directionality & self-loops:** Undirected radius-55 proximity represented as bidirectional directed edges `(src -> dst)` and `(dst -> src)` for all $i \neq j$ with $D_{ij} \le 55.0$, **plus explicit self-loops `(i -> i)` for all $i \in V_t$**. Set `add_self_loops=False` in `GATv2Conv`.
4. **Isolated nodes:** Handled automatically via explicit self-loops (`edge_attr = [0, 0, 0, 1, 0, 1, 0]`) and broadcast `GLOBAL_FEATURES_6`.
5. **Dynamic topology:** Recompute `(edge_index_tau, edge_attr_tau)` independently from `rel_*` and `v_*` at every step $\tau$; edges never persist across steps.
6. **Graph batching:** Use PyG `Batch` (Model 3) or flattened `(b, tau)` subgraph indexing (Model 4). Never mix edges across scenarios or timesteps.

### 6.3 Attrition and Variable Swarm Sizes

1. **Agent disappearance:** Once an agent disappears due to attrition, it has no observations at later steps (`step` ends permanently).
2. **ST-GAT attrition policy:** Let $V_{\text{win}}$ be the union of agents alive at any $\tau$ in the $T=8$ window (ordered by `agent_id`, padded to batch `N_max <= 75`). At each $\tau$, `node_mask[b, tau, u] = True` iff agent $u$ is alive at $\tau$. Dead agents are **excluded** from the spatial graph at $\tau$ (no zero-position/zero-velocity ghost nodes, no edges) and are **excluded** from window-end prediction/loss (`node_mask[b, -1, u] == False`).
3. **Never zero-impute missing agents into the spatial graph.**

---

## 7. Leakage Prevention and Data Splitting

### 7.1 Explicit Input Allowlist & Prohibition Enforcement

- **Allowed raw columns for feature derivation:** `step`, `rel_x`, `rel_y`, `rel_z`, `dist_to_centroid`, `dist_to_target`, `vx`, `vy`, `vz`, `speed`, `heading`, `flight_path_angle`, `accel`, `heading_rate`, `num_neighbors`, `mean_neighbor_dist`, `min_neighbor_dist`, `weighted_density`, `angular_coverage`.
- **Forbidden as model inputs (programmatically asserted absent before `fit`/`predict`):** `criticality_score`, `was_promoted`, `is_critical`, `case_id`, `agent_id`, `n_obstacles`, `obstacle_density`, `mean_obstacle_size`, `clutter_index`, `obstacle_clustering`, `num_agents`, `attrition_rate`, `degree_centrality`, `closeness_centrality`, `betweenness_centrality`, `eigenvector_centrality`, `rel_x`, `rel_y`, `vx`, `vy`, `vz`, `heading`.
- `case_id`, `step`, `agent_id` are retained solely in metadata DataFrames for grouping, alignment, and evaluation breakdowns. `was_promoted` is retained solely in evaluation metadata for `pre_promotion` / `post_promotion` slicing.

### 7.2 Deterministic 70 / 15 / 15 Scenario-Grouped Split

All four models MUST load the identical frozen scenario partition from `artifacts/splits/primary_split.json` (`70 train / 15 val / 15 test`):

- **Deterministic construction algorithm (run once in Stage 2):**
  1. Build scenario metadata table (`100` rows): `case_id`, `num_agents` (9 levels), `attrition_rate` (4 levels), `has_promo = (promo_agents > 0)`, `pos_rate`.
  2. Sort scenarios deterministically by `("num_agents", "attrition_rate", "has_promo", "pos_rate", "case_id")`.
  3. Allocate **15 test** and **15 val** scenarios via stratified round-robin / remainder-quota assignment across `num_agents` (giving `{1 or 2}` test and `{1 or 2}` val scenarios per size so **all 9 swarm sizes**, **all 4 attrition levels**, and **promoted scenarios** are represented in train, val, and test) with fixed RNG `seed=0`, and assign the remaining **70** scenarios to `train`.
  4. Verify `len(train) == 70`, `len(val) == 15`, `len(test) == 15`, pairwise disjointness, all 9 sizes and 4 attrition levels present in each split, and balanced positive rate (`~0.38–0.40` in each split).
- **Strict isolation:** Fit all scalers (`mu, sd`) on `train` only; tune operating threshold on `val` only; touch `test` only once per final model checkpoint.

---

## 8. Common Evaluation Framework

One shared evaluator module computes identical metrics for all four models on `100%` of the test split (`stride=1`):

1. **Threshold Selection:** Pick `thr ∈ np.linspace(0.02, 0.98, 97)` maximizing F1 on **validation** predictions. Apply frozen `thr` to **test** predictions.
2. **Required Metrics:**
   - **Primary:** `PR-AUC`, `ROC-AUC`, `F1`, `Precision`, `Recall`, `MCC`, `Balanced Accuracy`.
   - **Operational:** `Recall @ >=90% Precision` (max recall on PR curve where `precision >= 0.90`, else `0.0`), `FNR (1 - Recall)`, `Top-K Precision` & `Top-K Recall` (where `K = sum(is_critical)` per `(case_id, step)` snapshot, averaged over snapshots with `K > 0`).
3. **Required Breakdowns (All Models):**
   - Overall (full test set + anomaly-filtered `~is_anomaly` check)
   - Per swarm size (all 9 sizes present in test)
   - Per attrition condition (`none`, `light`, `medium`, `heavy`)
   - Promotion subsets:
     - `never_promoted`: all rows of agents that never experience a promotion
     - `pre_promotion`: rows of promoted agents strictly before `first_promo_step` (`is_critical == 0`)
     - `post_promotion`: rows of promoted agents at/after `first_promo_step` (`is_critical == 1`)
   - Mission periods: within-scenario progress tertiles (`early: [0, 1/3)`, `mid: [1/3, 2/3)`, `late: [2/3, 1]`) + `t = 0` cross-section
   - Aggregation levels: Row-level AND Agent-level (`mean(prob)` per `(case_id, agent_id)` vs `mean(is_critical) > 0.5`).
4. **Sanity Checks (Minimal, Mandatory):**
   - **SC-1 Leakage check:** Assert forbidden columns absent from input schema + confirm `case_id` disjointness across train/val/test.
   - **SC-2 Causality check:** Perturb future rows (`step > t`) in a test scenario and assert predictions at `<= t` are bit-for-bit identical across all 4 models.
   - **SC-3 Graph check:** Verify 3D radius-55 graph reproduces `num_neighbors` within `±1` on clean snapshots and self-loops exist for isolated nodes.
   - **SC-4 Temporal mask & attrition check:** Verify left-padded early steps (`k < W-1`) match unpadded streaming execution and disappeared agents receive zero edges/messages post-attrition.
   - **SC-5 Preprocessing parity check:** Verify `predict()` on raw snapshot/window produces identical normalized inputs and probabilities to offline test evaluation.

---

## 9. Project Structure

```text
criticalagentclassification-kilo/
├── dataset/                                  # READ-ONLY (untouched)
│   ├── Dataset_Metadata.md
│   └── swarm_criticality_dataset_100_scenarios.csv
├── experiment/                               # Existing preliminary scripts/artifacts (preserve)
├── report/
│   ├── Final_Design_Specification.md
│   └── Final_Implementation_Plan.md          # This blueprint
└── src/                                      # New clean implementation package
    ├── config.py                             # Paths, seeds, feature allowlists, hyperparams
    ├── data/
    │   ├── loader.py                         # CSV loading, integrity checks, anomaly flagging
    │   ├── splitter.py                       # Deterministic 70/15/15 split -> primary_split.json
    │   ├── features.py                       # Shared BASE_FEATURES_24, EDGE_FEATURES_7, GLOBAL_FEATURES_6
    │   ├── adapters.py                       # LightGBM lags, GRU windows, GATv2 Data, ST-GAT windows
    │   └── scaler.py                         # Train-only standardization fit/transform/save/load
    ├── models/
    │   ├── lgbm_model.py                     # LightGBM wrapper + train/predict
    │   ├── gru_model.py                      # 2-layer GRU + packed forward + streaming state
    │   ├── gat_model.py                      # 3-layer GATv2 + edge MLP + global concat
    │   └── stgat_model.py                    # Shared GATv2 + vectorized scatter + per-node GRU
    ├── training/
    │   ├── trainer.py                        # Shared PyTorch train loop, early stop, checkpointing
    │   ├── train_lgbm.py                     # CLI entry point for LightGBM
    │   ├── train_gru.py                      # CLI entry point for GRU
    │   ├── train_gat.py                      # CLI entry point for GATv2
    │   └── train_stgat.py                    # CLI entry point for ST-GAT
    ├── evaluation/
    │   ├── metrics.py                        # Primary, operational, Top-K, threshold selection
    │   └── evaluate.py                       # Unified evaluation harness + JSON/CSV breakdown export
    ├── inference/
    │   └── predictor.py                      # Unified predict() API for all 4 models
    └── tests/
        └── test_sanity.py                    # SC-1 through SC-5 automated verification suite
```

All training artifacts save under `artifacts/runs/<model_name>/`: `best_checkpoint.{pt,txt}`, `model_config.json`, `feature_schema.json`, `preprocessing.json`, `training_history.json`, `val_metrics.json`, `test_metrics.json`, and `test_predictions.parquet`.

---

## 10. Implementation Sequence

| Stage | Module(s) | Implements | Depends On | Output Artifacts | Completion Tests |
|---|---|---|---|---|---|
| **1** | `src/config.py` | Seeds, device auto-select, exact feature lists, forbid list, model configs. | None | `src/config.py` | Import check; assert `len(BASE_FEATURES_24) == 24`, `len(LAG_DIFF_FEATURES_30) == 30`, no overlap with forbid list. |
| **2** | `src/data/loader.py`, `src/data/splitter.py` | Read-only CSV load, integrity assertions, causal `is_anomaly`, deterministic `70/15/15` split. | Stage 1 | `artifacts/splits/primary_split.json` | Assert `70/15/15` counts, 0 overlap, all 9 sizes & 4 attrition levels in each split, `~520` anomaly rows flagged. |
| **3** | `src/data/features.py`, `src/data/scaler.py` | Shared `BASE_FEATURES_24`, `EDGE_FEATURES_7`, `GLOBAL_FEATURES_6`, 3D radius-55 graph, train-only scaler. | Stage 2 | Cached/verified feature functions + scaler | **SC-1** (forbid check), **SC-3** (graph reproduces `num_neighbors` `±1`, isolated self-loops valid), rank error `0.0`. |
| **4** | `src/data/adapters.py`, `src/tests/test_sanity.py` | Model adapters (LightGBM 54-col lags, GRU `[B,16,24]` left-pad, GATv2 PyG `Data`, ST-GAT `T=8` windows). | Stage 3 | `src/data/adapters.py`, `test_sanity.py` | **SC-2** (causality perturbation test), **SC-4** (left-pad & attrition masking). |
| **5** | `src/evaluation/metrics.py`, `src/evaluation/evaluate.py` | Unified threshold picker, primary + operational + Top-K metrics, all mandatory breakdowns. | Stage 2 | Evaluation module | Unit test on synthetic predictions verifying Top-K, Recall@90%P, promotion & tertile slices. |
| **6** | `src/models/lgbm_model.py`, `src/training/train_lgbm.py` | LightGBM training, checkpointing, unified evaluation run. | Stages 4, 5 | `artifacts/runs/lgbm/*` | Trains end-to-end; saves all 7 required run artifacts + test breakdowns. |
| **7** | `src/models/gru_model.py`, `src/training/trainer.py`, `src/training/train_gru.py` | Shared PyTorch trainer + 2-layer GRU (`W=16`) + streaming parity test + evaluation. | Stages 4, 5 | `artifacts/runs/gru/*` | Streaming step-by-step output matches left-padded `W=16` window output within `1e-5`. |
| **8** | `src/models/gat_model.py`, `src/training/train_gat.py` | 3-layer GATv2 (`hidden=96, heads=4`) training + evaluation. | Stages 4, 5, 7 | `artifacts/runs/gatv2/*` | Batching handles variable `N` (`18..75`) cleanly; saves all 7 artifacts + breakdowns. |
| **9** | `src/models/stgat_model.py`, `src/training/train_stgat.py` | Shared GATv2 + per-node GRU (`T=8`, `train_stride=4`, `batch_size=8`) full training + full `stride=1` eval. | Stages 7, 8 | `artifacts/runs/stgat/*` | Runs within 4 GB VRAM (or CPU fallback); evaluates all val/test snapshots at `stride=1`. |
| **10** | `src/inference/predictor.py` | Unified `predict()` wrapper loading independent checkpoints & preprocessing + **SC-5** parity test. | Stages 6–9 | `src/inference/predictor.py`, `artifacts/comparison_summary.csv` | **SC-5** passes for all 4 models; all `test_sanity.py` checks pass. |

---

## 11. Acceptance Criteria

Implementation is complete ONLY when all checks pass:

1. All four models train independently via their CLI entry points without requiring the other three models.
2. All four models load the identical `70/15/15` scenario split from `artifacts/splits/primary_split.json`.
3. Zero scenario leakage, zero target leakage, zero future-information leakage (**SC-1** and **SC-2** pass).
4. Shared feature calculations are single-sourced in `src/data/features.py` with exact schemas `[54]`, `[B, 16, 24]`, `([N, 24], [E, 7], [1, 6])`, and `([B, 8, N_max, 24], [E_t, 7], [B, 8, 6])`.
5. Variable swarm sizes (`18`–`75`), isolated nodes, and permanent agent attrition are handled without ghost zero-nodes or edge leaks (**SC-3** and **SC-4** pass).
6. Each model saves all 7 required run artifacts (`best_checkpoint`, `model_config`, `feature_schema`, `preprocessing`, `training_history`, `val_metrics`, `test_metrics`).
7. Training and `predict()` inference produce identical normalized tensors and aligned `[case_id, step, agent_id, prob, pred_class, threshold]` outputs (**SC-5** passes).
8. Unified evaluation reports all primary metrics, operational metrics (`Recall@90%P`, `FNR`, `Top-K`), and all required breakdowns on `100%` of the test split (`stride=1`).
9. `dataset/swarm_criticality_dataset_100_scenarios.csv` checksum/mtime remains unchanged.

---

## 12. Remaining Risks and Assumptions

1. **Unvalidated derived feature gain (`n_within_r{15,30,45}`, `nn_bearing_entropy`, `local_density_contrast`):** Defined in `report/Final_Design_Specification.md` §4.2/§5.2/§6.3/§7.4 and locked here with exact deterministic formulas, but not previously coded in `experiment/*.py`.
2. **PyTorch CUDA wheel status:** `experiment/.venv` currently has `torch 2.15.0+cpu` while the host has an RTX 3050 (4 GB VRAM). Installing a CUDA-enabled PyTorch build in `.venv` will substantially accelerate GRU, GATv2, and ST-GAT training; CPU fallback remains functional via `train_stride=4` for ST-GAT.
3. **Small-swarm & post-promotion hard tail:** Preliminary results showed `n=18` LOSO PR-AUC (`0.905`) and post-promotion recall (`~0.40`) are substantially harder than overall metrics. Reported transparently via mandatory breakdowns; threshold calibration per swarm size is deferred unless required post-evaluation.
4. **ST-GAT full-training compute & variance:** ST-GAT has not yet completed full training; initial implementation uses `seed=0`. Optional multi-seed runs (`seeds 1, 2`) may follow after all four single-seed pipelines pass acceptance criteria.

