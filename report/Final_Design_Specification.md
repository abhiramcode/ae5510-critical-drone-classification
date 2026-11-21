# Critical Node Identification in Aerial Swarms
## Final Investigation, Dataset Audit, and Four-Model Design Specification

**Dataset:** `dataset/swarm_criticality_dataset_100_scenarios.csv` (925,737,609 bytes; read-only)
**Metadata:** `dataset/Dataset_Metadata.md`
**Status:** investigation + preliminary experiments complete. Full model training is out of scope for this phase.
**Reproduction:** `experiment/` (scripts + `artifacts/*.json`), Python 3.12 via `uv`.

---

## 1. Executive Summary

### 1.1 What the dataset actually is

- **1,639,166 rows, 37 columns, 100 scenarios**, ordered scenario → `step` → `agent_id`; `step` is sampled every 2 simulation steps.
- Swarm sizes: **{18, 24, 30, 37, 45, 52, 60, 68, 75}** (12/13/15/8/9/9/10/9/15 scenarios respectively). Attrition: **none 24, light 24, medium 27, heavy 25**.
- Rows per scenario: **3,546–45,000** (median 14,431.5). Sampled steps per scenario: **197–600** (i.e., 394–1,200 sim steps).
- **Zero duplicate keys**, **zero missing values**, no NaN. Relative positions sum to ≈0 per snapshot (max |sum| ≈ 1.1e−11). `dist_to_centroid = ||rel||` exactly; `speed = ||v||` exactly.
- **Positive rate 38.9% overall**, 27.8%–68.3% per scenario.
- Two undocumented columns exist: **`attrition_rate`** (the mission attrition condition) and **`was_promoted`** (a label-transition flag).

### 1.2 The target is dynamic (and partly exogenous)

- `is_critical` is constant per agent **except for 85 discrete 0→1 promotion events in 41 scenarios** (no 1→0 transitions; non-promoted agents never change).
- Promotions occur **only in attrition scenarios**: 49 agents in heavy, 30 in medium, 6 in light, 0 in none. Promotion step range **146–1006**, median 478.
- `was_promoted` flips with the label (0 before, 1 at/after) → it **directly encodes the transition** and must not be a feature.
- Promoted agents are usually **peripheral** just before promotion (e.g., c003 agent 15: dist_to_centroid 79.5→92.7, neighbors 2→0), and their detection is the hard tail: post-promotion recall ≈ **0.40** vs ≥0.93 overall.

### 1.3 Leakage findings

- **`criticality_score` is a near-oracle simulator heuristic**: ROC-AUC **0.9984**, PR-AUC **0.9979**; at `step=0`, decoys ≤0.0747 and critical agents ≥0.2911 (full-dataset cross-section). It also rises *before* promotions (c003 agent 15: 0.336 @602 → 0.356 @640 → 0.617 @642).
- It is **not a simple label copy**: a LightGBM regressor predicts it from leak-free geometry with **R² = 0.662** (Spearman 0.798), i.e. mostly geometric but partly privileged.
- **Decision:** excluded from all four primary models; used only in a documented leakage-ablation run. Adding it lifts PR-AUC from **0.9897 → 0.9997**, confirming its role as a shortcut.
- `case_id` and `agent_id` are identifiers. `agent_id` shows **no meaningful association** with the label (point-biserial 0.019; positive rate 0.36–0.42 across relative-ID deciles). Both are excluded as inputs; `case_id` is the group key.

### 1.4 Is the task learnable without leakage? Yes.

- Best leak-free model (LightGBM, static + lag features through W≈16): **PR-AUC 0.9897, ROC-AUC 0.9923, F1 0.9463** on a scenario-grouped test split.
- **Agent-level aggregation** (mean score over an agent's trajectory): **PR-AUC 0.9992, F1 0.9934**.
- **Single-snapshot t=0** is much harder: PR-AUC 0.8815, F1 0.797. Trajectory information matters, especially for early identification.
- Critical agents are visibly **protected/escorted**: median `min_neighbor_dist` 6.35 vs 14.12, `weighted_density` 8.81 vs 2.18, `angular_coverage` 0.438 vs 0.292, `dist_to_centroid` 19.45 vs 33.25, `rank_dist_to_centroid` 0.271 vs 0.649 (critical vs decoy).
- Dynamics add value: static PR-AUC 0.9782 → +1 lag 0.9830 → lags through 8 0.9883 → lags through 16 **0.9897**.
- **Local-only (distributed) features nearly match the centralized model**: 0.9806 vs 0.9897. Global centroid/ranks/centralities are not essential.

### 1.5 Recommended four models

| Family | Selected architecture | Evidence-backed expectation |
|---|---|---|
| Ensemble | **LightGBM** on kinematics + local escort geometry + multi-lag deltas | PR-AUC ≈0.99; simplest, strongest baseline |
| Temporal | **2-layer unidirectional GRU**, W=16 | ≈0.99 with W=16; smoke GRU 0.9876 vs flattened GBDT 0.9901 at W=8 → use GRU for streaming, not accuracy |
| Spatio-Graph | **3-layer GATv2**, radius-55 reconstructed graph, edge features | Smoke GAT 0.9879 vs same-snapshot tabular 0.9769 (**+0.011 PR-AUC** from message passing) |
| Spatio-Temporal | **GATv2 encoder + per-node GRU**, T=8 | Expected ≈0.99; components validated; full training not run (compute) |

### 1.6 Most important inference-time considerations

1. The **radius-55 sensing graph is reconstructible** from `rel_*`; no privileged adjacency is needed. It matches the simulator's graph within ±1 neighbor in **299/300** sampled snapshots.
2. `criticality_score` and `was_promoted` are **non-deployable** (simulator/transition leakage).
3. Promotions are **not reliably predictable before they happen** (median score jump +0.004; only 78.6% positive, mean +0.115). Report promotion subsets separately.
4. **Early mission time is harder** (first progress quintile PR-AUC 0.9735 vs 0.9953 in the last), and **small swarms are harder** (n=18: 0.9046 under leave-one-size-out).
5. Environment/scenario features add **no measurable value** (no-env model even slightly better: 0.9834 vs 0.9825), so the deployable feature set can drop them.

---

## 2. Dataset Audit

### 2.1 Structure and integrity (verified on the full file)

| Property | Result |
|---|---|
| Rows / columns | 1,639,166 / 37 |
| Scenarios | 100 (`c001`…`c100`) |
| Duplicate `(case_id, step, agent_id)` | **0** |
| Missing / NaN cells | **0** |
| Key ordering | scenario → step → agent_id |
| Step cadence | 0, 2, 4, … (all even) |
| Agent IDs | 0…n−1, persistent, contiguous; permanent disappearance at attrition |
| Static-column consistency | all scenario-level columns constant per scenario |
| `rel` consistency | `sum(rel)≈0` per snapshot (max 1.1e−11); `dist_to_centroid=||rel||` exact |
| Kinematics consistency | `speed=||v||` exact (max err 3.6e−15) |
| Label transitions | 85 rows, **all 0→1**; 85 agents; 41 scenarios; no 1→0 |

### 2.2 Swarm composition

| Size | 18 | 24 | 30 | 37 | 45 | 52 | 60 | 68 | 75 |
|---|---|---|---|---|---|---|---|---|---|
| Scenarios | 12 | 13 | 15 | 8 | 9 | 9 | 10 | 9 | 15 |

Attrition: none 24, light 24, medium 27, heavy 25.
Positive rate by attrition: heavy 0.423, medium 0.396, light 0.365, none 0.378.
Overall positive rate: **0.389**; per-scenario range **0.278–0.683**.

### 2.3 Feature inventory (37 columns, grouped)

**Scenario/environment (static):** `case_id` (id), `num_agents`, `n_obstacles`, `obstacle_density`, `mean_obstacle_size`, `clutter_index`, `obstacle_clustering`, `attrition_rate`.
**Time/identity:** `step`, `agent_id`.
**Position/geometry:** `rel_x`, `rel_y`, `rel_z` (relative to swarm centroid), `dist_to_centroid`, `rank_dist_to_centroid`, `dist_to_target`, `rank_dist_to_target`.
**Kinematics:** `vx`, `vy`, `vz`, `speed`, `heading`, `flight_path_angle`, `accel`, `heading_rate`.
**Local swarm geometry:** `num_neighbors`, `mean_neighbor_dist`, `min_neighbor_dist`, `weighted_density`, `angular_coverage` (occupied of 12×4 = 48 angular bins).
**Target-related/labels:** `criticality_score` (heuristic), `is_critical` (label), `was_promoted` (transition flag).
**Graph centrality:** `degree_centrality`, `closeness_centrality`, `betweenness_centrality`, `eigenvector_centrality` (Radius-55 neighbor graph).

Semantics and redundancy established by experiment:

- `rel_x, rel_y` are redundant with `dist_to_centroid`; `rel_z` adds altitude.
- `vx,vy,vz` are redundant with `speed`, `heading`, `flight_path_angle` (`vz` ↔ flight-path angle Spearman 0.987).
- `degree_centrality` ↔ `closeness_centrality` Spearman **0.993**.
- `weighted_density` ↔ `mean_neighbor_dist` −0.893, ↔ `min_neighbor_dist` −0.888.
- `n_obstacles` ↔ `clutter_index` 0.906; `obstacle_density` ↔ `clutter_index` 0.869.
- `dist_to_centroid` ↔ `degree_centrality` −0.859; ↔ `rank_dist_to_centroid` 0.812.
- `num_neighbors` ↔ `angular_coverage` 0.783.

### 2.4 Data-quality flags

1. **`obstacle_clustering` is non-zero when `n_obstacles = 0`** (c004 0.612, c007 0.525). It behaves like a sampled configuration parameter, not a realized-layout statistic. Treated as conditional/no-value.
2. **Row-level anomalies (~0.07%)**: a small set of rows have positions inconsistent with the agent's velocity (`|Δposition| > 1.5 × 2·speed`) and neighbor stats inconsistent with the reported position. They concentrate in **c002 (2.37% of its rows; agents 24/29/6/28)** and **c099 (0.23%)**. Exclude/flag these rows for training; they are a data artifact, not a graph-definition issue.
3. **Variable scenario duration** (197–600 sampled steps): fixed absolute windows are unsafe; pad/mask and use causal windows.
4. **Temporal sampling** (every 2 sim steps): `accel`/`heading_rate` refer to the previous sim step; treat all lags as causal.

### 2.5 Neighbor/graph definition (important)

The simulator's graph is a **3D radius-55 proximity graph over alive agents**:
- 270/400 random snapshots reproduce `num_neighbors` **exactly**; 299/300 within ±1.
- Using 2D distance is wrong (42/400 exact; errors up to 44).
- Residuals are the row anomalies of §2.4, not occlusion: deficits do **not** scale with obstacle density/clutter.
- `num_neighbors` and `degree_centrality·(alive−1)` agree within 0.5 for 92.8% of rows; the remainder are ±1 boundary cases or anomalies.

**Implication:** rebuild the graph from `rel_*`; never use simulator adjacency or raw centrality columns as privileged inputs.

### 2.6 Target behaviour

| Metric | Value |
|---|---|
| Overall positive rate | 0.389 |
| Critical agents (median) | `min_neighbor_dist` 6.35, `weighted_density` 8.81, `angular_coverage` 0.438, `dist_to_centroid` 19.45, rank 0.271, `num_neighbors` 43, degree 0.833 |
| Decoys (median) | 14.12, 2.18, 0.292, 33.25, 0.649, 32, 0.729 |
| Promotions | 85 agents, 41 scenarios, steps 146–1006 (median 478) |
| Promotion by attrition | heavy 49, medium 30, light 6, none 0 |
| `criticality_score` alone | ROC-AUC 0.9984, PR-AUC 0.9979 |
| `criticality_score` from geometry | R² 0.662, Spearman 0.798 |
| `agent_id` ↔ label | point-biserial 0.019 |

**Interpretation:** criticality is strongly associated with being **centrally placed and densely escorted**; decoys form a protective shell. Kinematics differ only mildly. This is a genuine, observable signature — not merely a simulator artifact — but `criticality_score` shortcuts it.

---

## 3. Inference-Time Information Model

### 3.1 Primary operational setting (locked)

**Centralized fused snapshot.** A swarm operations/fusion node receives periodic broadcast telemetry from every alive agent: id, position, velocity, heading, flight-path angle, accel/heading-rate commands, timestamp; plus mission inputs (target point, planned obstacle map, swarm size, attrition condition). It retains a short causal history buffer. Inference for time *t* uses only samples ≤ *t*.

The dataset's relative coordinates are sufficient:
- `rel_i − rel_j` gives the exact i→j vector (translation-invariant);
- therefore the radius-55 graph, bearings, closing rates, shells, local density, and graph centralities are all derivable;
- absolute coordinates are not needed except as summarized by `dist_to_target`.

### 3.2 Information classes

**A. Directly available:** agent kinematics, `rel_*`, `dist_to_centroid`, `dist_to_target`, `num_neighbors`, `mean/min_neighbor_dist`, `weighted_density`, `angular_coverage`, scenario size, attrition condition, mission time, environmental summaries (map).

**B. Implicitly derivable:** centroid/ranks (centralized), pairwise distances, radius graph, shell occupancy, azimuth/elevation coverage, neighbor-bearing entropy, degree/closeness/betweenness/eigenvector centrality, components, swarm spread, centroid speed, causal deltas, radial/target-approach rates.

**C. Conditionally available:** absolute geodetic coordinates; full obstacle layout (needed only if a line-of-sight communication model is added); communication quality/latency (not in the dataset); true target coordinates (only distances are given).

**D. Unavailable:** `is_critical`, `was_promoted`, `criticality_score` (simulator-internal), future steps, post-attrition observations, scenario outcome, scenario end time, `case_id` outcome identity.

**E. Uncertain:** the exact `criticality_score` formula; the promotion generative mechanism; whether `obstacle_clustering` reflects realized obstacles; whether every agent always broadcasts.

### 3.3 Distributed variant (secondary)

Each agent uses its own telemetry + 1-hop neighbor states: kinematics, neighbor distances/counts/bearings, local density/coverage, ego-graph. Illegal: centroid, global ranks, global centralities, swarm aggregates. **Experimentally supported:** a local-only LightGBM reaches PR-AUC **0.9806** vs **0.9897** centralized — a small loss, making onboard deployment credible.

### 3.4 Leakage register

| Source | Risk | Decision |
|---|---|---|
| `criticality_score` | Label proxy; perfect separation at t=0; anticipates promotions; R²=0.66 from geometry | Exclude; ablation only |
| `was_promoted` | Direct transition flag (0→1 exactly with the label) | Exclude |
| `is_critical` | Label | Exclude |
| `case_id` | Encodes size/attrition/env and scenario positive rate | Group key only |
| `agent_id` | Potential role-assignment artifact (tested: negligible; 0.019) | Key only |
| Future/whole-trajectory aggregates | Temporal leakage | Forbidden |
| Aggregates containing the target | Target leakage | Forbidden |
| Raw simulator adjacency/centralities | Not provided; would be privileged | Recompute from geometry |

---

## 4. Feature Engineering and Selection

### 4.1 Shared base snapshot features (per agent, per t)

`speed, accel, heading_rate, flight_path_angle, heading_sin, heading_cos, dist_to_centroid, rank_dist_to_centroid, rel_z, dist_to_target, rank_dist_to_target, num_neighbors, centroid_vel_alignment, mean_neighbor_dist, min_neighbor_dist, weighted_density, angular_coverage, is_isolated`

*(Note: In the final production pipeline, `neighbor_frac` was replaced by `centroid_vel_alignment` to eliminate 0.9999 collinearity with `degree_centrality_c`).*

### 4.2 Derived escort/geometry features (all observable)

| Name | Definition | Source | Availability |
|---|---|---|---|
| `alive_count` | #alive agents at t | snapshot | B |
| `neighbor_frac` | `num_neighbors/(alive_count−1)` | num_neighbors | B |
| `n_within_r{15,30,45}` | neighbor counts in shells | pairwise distance | B |
| `az_coverage`, `el_coverage` | occupied azimuth/elevation bins | pairwise bearing | B |
| `nn_bearing_entropy` | entropy of nearest-neighbor bearings | bearing | B |
| `neighbor_rank_mean/std` | mean/std rank of neighbors | ranks + graph | B |
| `local_density_contrast` | own density − mean neighbor density | graph | B |
| `escort_ratio` | inner-half vs outer-half neighbor distance | graph | B |
| `is_isolated` | `1{num_neighbors=0}` | num_neighbors | B |
| `radial_velocity` | `d(dist_to_centroid)/dt` | history | B |
| `target_approach_rate` | `d(dist_to_target)/dt` | history | B |
| `degree_centrality_c` | recomputed degree/(alive−1) | graph | B |
| `betweenness_centrality_c`, `eigenvector_centrality_c`, `closeness_centrality_c` | NetworkX on reconstructed graph | graph | B (centralized) |
| `swarm_spread`, `swarm_centroid_speed`, `graph_density`, `n_components`, `largest_component_frac`, `mean_degree` | swarm/graph stats | graph | B |

### 4.3 Temporal features (causal, W = 16 samples = 32 sim steps)

Lags `d{1,2,4,8,16}` of `dist_to_centroid, rank_dist_to_centroid, num_neighbors, weighted_density, angular_coverage, speed`, plus rolling means/stds if desired.
**Window evidence:** static 0.9782 → lag1 0.9830 → lags≤8 0.9883 → lags≤16 **0.9897** PR-AUC. Use **W = 16**; W = 8 is an acceptable latency/compute trade-off.

### 4.4 Experimental findings that changed the design

| Finding | Evidence | Design consequence |
|---|---|---|
| `criticality_score` is a near-oracle | ROC-AUC 0.9984; t=0 separation; +0.01 PR-AUC over leak-free | Exclude; report ablation |
| `was_promoted` is a transition flag | 0→1 exactly at promotion; 0 rows with F label | Exclude |
| Labels are static except promotions | 85/85 transitions are 0→1; non-promoted agents never change | Current-state target + promotion subset metrics |
| Escort geometry carries the signal | medians by label (§2.6); kinematics-only PR-AUC 0.9302 | Prioritize local escort + kinematics |
| Temporal history helps | 0.9782 → 0.9897 through W=16 | W=16 for temporal/graph-temporal |
| Global centralities are redundant | removing all: 0.9816 vs 0.9825 | Keep only degree in ensemble; GNN learns structure |
| Environment features add nothing | no-env 0.9834 vs 0.9825 | Drop from deployable features |
| Scenario prior is weak | env+alive-only PR-AUC 0.4487, ROC 0.54 | Performance comes from per-agent geometry |
| Local-only is nearly as good | 0.9806 vs 0.9897 | Distributed variant is viable |
| Small swarms harder | LOSO n=18 0.9046; n=68 0.9946 | Report per-size; consider size-aware training |
| Promotions are the hard tail | post-promo recall 0.397; promoted-agent PR-AUC 0.838 | Dedicated peripheral/detachment handling |
| Graph message passing helps | GAT 0.9879 vs same-snapshot tabular 0.9769 (+0.011) | Justifies GAT/ST models |
| GRU does not beat flattened GBDT at W=8 | 0.9876 vs 0.9901 | GRU justified by streaming/deployment, not accuracy |
| Early mission is harder | progress quintile 0 PR-AUC 0.9735 vs 0.9953 last | Report inference-time stratification |
| Agent-level aggregation is easier | 0.9992 PR-AUC | Report row- and agent-level metrics |

---

## 5. Model 1 — Ensemble (LightGBM)

### 5.1 Recommendation

**LightGBM** binary classifier, `class_weight="balanced"`, 500–900 trees, `num_leaves=127`, early stopping on validation PR-AUC.
**Why:** 1.6M rows, ~40 heterogeneous tabular features, non-linear escort interactions; histogram GBDT is fast and accurate. Random Forest/ExtraTrees are robust but weaker on subtle interactions at this scale; XGBoost is comparable but slower; CatBoost adds cost without a categorical-heavy advantage.

### 5.2 Final feature specification

| Feature | Source | Feature Type | Inference Availability | Derivation Required | Model Role | Include/Exclude |
|---|---|---|---|---|---|---|
| `speed`, `accel`, `heading_rate`, `flight_path_angle` | dataset | num, dynamic | A | none | motion | Include |
| `heading_sin`, `heading_cos` | derived | num | B | sin/cos(heading) | orientation | Include |
| `dist_to_centroid`, `rank_dist_to_centroid` | dataset | num | A/B | rank recomputed over alive | radial role | Include |
| `rel_z` | dataset | num | A | none | altitude | Include |
| `dist_to_target`, `rank_dist_to_target` | dataset | num | A/B | rank recomputed | mission proximity | Include |
| `num_neighbors`, `centroid_vel_alignment`, `is_isolated` | dataset/derived | num/binary | A/B | velocity alignment / ratio | escort density | Include (`centroid_vel_alignment` replaces redundant `neighbor_frac`) |
| `mean_neighbor_dist`, `min_neighbor_dist`, `weighted_density`, `angular_coverage` | dataset | num | A | none | escort signature | Include |
| `n_within_r15`, `n_within_r30`, `n_within_r45` | derived | num | B | pairwise distances | shell occupancy | Include |
| `nn_bearing_entropy`, `local_density_contrast` | derived | num | B | geometry/graph | escort uniformity | Include |
| `degree_centrality_c` | derived | num | B | NetworkX on reconstructed graph | structural role | Include |
| `d{1,2,4,8,16}_{dist,rank,nb,dens,ang,speed}` | derived | num | B | causal within-agent lags | temporal summary | Include |
| `elapsed_step` | dataset | num | A | none | mission clock (ablate) | Conditional |
| `closeness/betweenness/eigenvector_centrality_c` | derived | num | B (centralized) | NetworkX | redundancy | Exclude (≈0 gain) |
| `n_obstacles`, `obstacle_density`, `mean_obstacle_size`, `clutter_index`, `obstacle_clustering` | dataset | num | A/C | none | environment | Exclude (no gain; caveat) |
| `num_agents`, `attrition_rate` | dataset | cat | A | ordinal | scenario prior | Exclude from inputs (stratification only) |
| `criticality_score` | dataset | num | D | — | — | **Exclude (leak)** |
| `was_promoted` | dataset | bool | D | — | — | **Exclude (leak)** |
| `is_critical` | dataset | bool | D | — | label | **Exclude** |
| `case_id`, `agent_id` | dataset | id | — | — | keys | **Exclude** |
| `rel_x`, `rel_y`, `vx`, `vy`, `vz` | dataset | num | A | redundant | — | Exclude |

### 5.3 Input representation and preprocessing

- One row per `(scenario, agent, t)` with `t` ≥ 15 windows; rows drop the first 15 samples per agent for full lag availability (or use `min_periods=1` with a `history_available` flag).
- `X ∈ R^{N×≈45}` float32; no scaling; no categorical encoding needed (env dropped).
- Flag/exclude the 0.07% inconsistent rows (§2.4).
- **Expected:** PR-AUC ≈0.99, ROC ≈0.992, F1 ≈0.946 at a validation-tuned threshold; agent-level ≈0.999.

---

## 6. Model 2 — Temporal Sequence (LSTM-GRU)

### 6.1 Recommendation

**Unidirectional 2-layer GRU, hidden 96**, static-vector head.
**Why GRU, not LSTM or a hybrid:** comparable capacity with ~25% fewer parameters, which matters with ~1.4M sequences and a weak per-step signal; a hybrid adds parameters without evidence of benefit. No bidirectional state (future leakage).
**Smoke-test reality check:** 1-layer GRU W=8 reached PR-AUC 0.9876 / F1 0.9446; a LightGBM on the *flattened same windows* reached 0.9901 / 0.9522. Therefore the GRU's value is **streaming inference, constant memory, and deployment**, not beating GBDT on this feature-rich dataset.

### 6.2 Sequence construction

- Sample: `(scenario, agent, t_k)` with `W=16` causal samples ending at `t_k`.
- Tensor `[B, 16, F]` + `mask[B,16]` + optional static head vector.
- Left-pad at the start (masked); train on `k ≥ W−1`. Never include future steps. Series end at attrition; no post-loss samples.
- If a lower-latency deployment requires W=8, accept ≈0.988 PR-AUC.

### 6.3 Final feature specification

| Feature | Source | Feature Type | Inference Availability | Derivation Required | Model Role | Include/Exclude |
|---|---|---|---|---|---|---|
| `speed`, `accel`, `heading_rate`, `flight_path_angle` | dataset | num, dynamic | A | none | motion | Include |
| `heading_sin`, `heading_cos` | derived | num | B | sin/cos | orientation | Include |
| `dist_to_centroid`, `rank_dist_to_centroid`, `rel_z` | dataset | num | A/B | rank recomputed | radial role | Include |
| `dist_to_target`, `rank_dist_to_target` | dataset | num | A/B | rank recomputed | mission proximity | Include |
| `num_neighbors`, `centroid_vel_alignment`, `is_isolated` | dataset/derived | num | A/B | alignment / ratio | escort density | Include (`centroid_vel_alignment` replaces redundant `neighbor_frac`) |
| `mean/min_neighbor_dist`, `weighted_density`, `angular_coverage` | dataset | num | A | none | local escort | Include |
| `local_density_contrast`, `nn_bearing_entropy`, `n_within_r30`, `n_within_r45` | derived | num | B | geometry/graph | escort structure | Include |
| `degree_centrality_c` | derived | num | B | graph | connectivity | Include |
| `elapsed_step` | dataset | num | A | none | time context (head) | Conditional |
| `swarm_spread`, `swarm_centroid_speed`, `graph_density` | derived | num | B (centralized) | snapshot | context (optional) | Conditional |
| `criticality_score`, `was_promoted`, `is_critical` | dataset | — | D | — | — | **Exclude (leak)** |
| `case_id`, `agent_id` | dataset | id | — | — | keys | **Exclude** |
| env/scenario features | dataset | num/cat | A | — | — | Exclude (no gain) |
| bidirectional state / future steps | — | — | D | — | — | **Exclude (leak)** |

### 6.4 Input representation

`[B, 16, 24]` float32 + mask + optional `[B, 3]` static vector.
Head: `GRU_out[last valid] ⊕ static → LayerNorm → Linear(128) → GELU → Dropout(0.1) → Linear(1)`.
Optimizer AdamW, lr 1e-3, cosine schedule, `BCEWithLogits(pos_weight)`.
**Expected:** ≈0.99 PR-AUC at W=16; per-size and per-promotion breakdowns mandatory.

---

## 7. Model 3 — Spatio-Graph (GATv2)

### 7.1 Graph construction (reconstructed, not privileged)

- **Nodes:** alive agents.
- **Edges:** undirected 3D radius-55 proximity graph from `rel_*`; both directions; self-loops; no simulator adjacency.
- **Edge features:** `distance/55`, relative speed `||v_i−v_j||`, closing rate `(v_j−v_i)·û_{i→j}`, `bearing_cos/sin` (i→j rotated into i's heading frame), `Δheading_cos/sin`.
- **Node features:** kinematics + radial + local escort geometry (no global centralities; the GNN learns structural importance).
- **Global vector (broadcast):** `alive_count, swarm_spread, swarm_centroid_speed, graph_density, n_components, largest_component_frac` (optional; env dropped).
- **Disconnected/isolated nodes:** self-loops + broadcast global; no component merging.
- **Varying n:** PyG batching with node masks; no intra-sample padding.

### 7.2 Architecture

3 × `GATv2Conv(hidden=96, heads=4, edge_dim=7, concat=False)`, residual + LayerNorm + GELU, dropout 0.1; edge MLP; global vector re-concatenated after each block; 2-layer MLP node head; `BCEWithLogits(pos_weight)`.
**Why GATv2:** heterogeneous neighbor importance, native edge features, inductive to any n/topology. GCN ignores edge attributes and needs fixed normalized adjacency; GraphSAGE is less expressive; GIN targets graph-level tasks.

### 7.3 Smoke-test evidence

Scenario-grouped, 8,000 train / 2,500 val / 2,500 test snapshots, 21 node features, 3D radius-55 edges:

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| **GATv2** (2 layers, 48 hidden, 4 epochs) | **0.9879** | 0.9909 | **0.9435** |
| LightGBM on identical snapshots (same node features, no graph) | 0.9769 | 0.9825 | 0.9202 |

Message passing adds **+0.011 PR-AUC** over the same features without structure. This validates the graph family, while the absence of a larger gap reflects the strength of local escort geometry.

### 7.4 Final feature specification

| Feature | Source | Feature Type | Inference Availability | Derivation Required | Model Role | Include/Exclude |
|---|---|---|---|---|---|---|
| node kinematics (6) | dataset/derived | num | A | sin/cos | node state | Include |
| node radial (`dist/rank` centroid, `rel_z`, `dist/rank` target) | dataset | num | A/B | rank recomputed | global-relative position | Include |
| node local escort (12: count, frac, mean/min dist, density, coverage, shells, entropy, neighbor-rank mean, contrast, isolation) | dataset/derived | num | A/B | pairwise | local structure | Include |
| node `degree_centrality_c` | derived | num | B | graph | connectivity | Include (optional) |
| edge distance, relative speed, closing rate | derived | num | B | pairwise + velocity | interaction strength | Include |
| edge bearing cos/sin, Δheading cos/sin | derived | num | B | geometry + heading | formation direction | Include |
| broadcast global vector (6) | derived | num | B | snapshot/graph | context | Include |
| raw `degree/closeness/betweenness/eigenvector` columns | dataset | num | B | — | — | Exclude (recompute) |
| `criticality_score`, `was_promoted`, `is_critical` | dataset | — | D | — | — | **Exclude (leak)** |
| `case_id`, `agent_id` | dataset | id | — | — | keys | **Exclude** |
| simulator adjacency (not provided) | — | — | D | — | — | **Exclude (privileged)** |

**Final input:** `Data(x=[N,~24], edge_index=[2,E], edge_attr=[E,7], global=[1,6])` per snapshot; PyG batch.
**Expected:** ≈0.988–0.99 PR-AUC; strong on large swarms, weaker on n=18/24.

---

## 8. Model 4 — Spatio-Temporal (GATv2 + per-node GRU)

### 8.1 Recommendation

A **shared GATv2 spatial encoder applied at each of T steps**, followed by a **per-node GRU** over the T embeddings and a classifier.
- **Not ST-GCN:** assumes a fixed graph; topology changes every step.
- **Not a full Temporal Graph Network:** no edge-event memory is needed; edges are re-derived from positions and nodes only disappear (attrition).
- **Not pure spatio-temporal attention:** more parameters, weaker inductive bias for short T.

### 8.2 Input representation

`[B, T=8, N_max, F≈24]` node tensor + per-step `edge_index_t [2,E_t]`, `edge_attr_t [E_t,7]`, `node_mask [B,T,N_max]`, `global_t [B,T,6]`.
**Topology:** recomputed each step; edges do not persist.
**Attrition:** a lost agent is masked out from its disappearance onward; samples exist only for agents alive at the window end.
**Varying n:** pad N to batch max, mask; no cross-scenario mixing.
**Temporal window:** T=8 primary (16 sim steps; lag evidence: W=8 gives 0.9883, W=16 gives 0.9897); T=16 ablation.
**Head:** per-node GRU(96) over T; last valid hidden state → LayerNorm → MLP → logit; global vector concatenated at each GRU step.
**Loss:** `BCEWithLogits(pos_weight)`; AdamW; early stop on val PR-AUC.

### 8.3 Why this over the alternatives

- Preserves the validated GAT spatial encoder and the validated temporal lag benefit.
- Handles time-varying topology and attrition natively.
- Avoids TGN memory/edge-event machinery that the data cannot justify.
- Compute-aware: T=8 with stride-subsampled training timesteps.

### 8.4 Final feature specification

| Feature | Source | Feature Type | Inference Availability | Derivation Required | Model Role | Include/Exclude |
|---|---|---|---|---|---|---|
| per-step node features (Model 3 set) | dataset/derived | num, dynamic | A/B | as Model 3 | spatial state | Include |
| per-step edge features (7) | derived | num, dynamic | B | pairwise + heading | interaction | Include |
| per-step `edge_index_t` | derived | graph | B | radius 55 | dynamic topology | Include |
| per-step global vector (6) | derived | num, dynamic | B | snapshot/graph | context | Include |
| `node_mask` / padding | derived | mask | — | alive + batch | variable n/attrition | Include |
| explicit rolling deltas | derived | num | B | causal | — | Not needed (GRU) |
| persistent/static edges | — | — | D | — | — | Exclude (recompute) |
| future steps | — | — | D | — | — | **Exclude (leak)** |
| `criticality_score`, `was_promoted`, `is_critical` | dataset | — | D | — | — | **Exclude (leak)** |
| `case_id`, `agent_id` | dataset | id | — | — | keys | **Exclude** |

**Final input:** `[B,8,N_max,24]` + per-step edge tensors + masks + `[B,8,6]` global.
**Status:** full ST training not run (CPU budget); components validated independently (GAT 0.9879, GRU 0.9876, temporal lags 0.9897). Expected ≈0.99.

---

## 9. Four-Model Feature Comparison

| Feature group | Ensemble | GRU | GAT | ST-GAT |
|---|---|---|---|---|
| Kinematics (speed, accel, rates, FPA, sin/cos heading) | ✔ | ✔ per step | ✔ node | ✔ node + time |
| Radial geometry (dist/rank centroid, `rel_z`; target dropped) | ✔ | ✔ per step | ✔ node | ✔ node + time |
| Local escort (counts, distances, density, coverage, shells, entropy) | ✔ | ✔ per step | ✔ node | ✔ node + time |
| Recomputed degree centrality | ✔ | ✔ per step | ✔ (optional) | ✔ (optional) |
| Betweenness/eigenvector/closeness | ✘ (redundant) | ✘ | learned by MPNN | learned |
| Causal lag features | ✔ (1/2/4/8/16) | learned by GRU | ✘ | learned |
| Edge features (distance, closing, bearing, Δheading) | ✘ | ✘ | ✔ | ✔ per step |
| Time-varying topology | ✘ | ✘ | ✘ (snapshot) | ✔ |
| Global swarm context | ✘ | optional head | ✔ broadcast | ✔ per step |
| Environment/scenario features | ✘ (no gain) | ✘ | ✘ | ✘ |
| `criticality_score`, `was_promoted` | ✘ leak | ✘ leak | ✘ leak | ✘ leak |
| `case_id`, `agent_id` | keys | keys | keys | keys |
| Future information | ✘ | ✘ | ✘ | ✘ |

**Shared boundary:** telemetry + history ≤ t + mission/map inputs. **Model-specific:** tabular summaries vs raw sequences vs graphs vs graph sequences.

---

## 10. Decision Log

| Decision | Alternatives | Selected | Reasoning | Evidence |
|---|---|---|---|---|
| Inference boundary | distributed; hybrid | centralized fused snapshot (primary) | matches global feature semantics; supports all four models | local-only 0.9806 vs 0.9897 centralized |
| Target | static t=0; end-of-scenario | `is_critical(t)`, causal | promotions make labels dynamic | 85 discrete 0→1 transitions |
| `criticality_score` | include; conditional | exclude from all models | near-oracle; anticipates promotions | ROC 0.9984; +0.01 PR-AUC; R² 0.66 from geometry |
| `was_promoted` | include as flag | exclude | directly encodes the transition | flips exactly with label |
| `case_id`/`agent_id` | include | keys only | identifiers; possible role artifacts | agent-ID association 0.019 |
| Raw centralities | use columns | recompute on reconstructed graph | guarantees identical graph semantics | radius-55 reproduces counts ±1 |
| Ensemble | RF/XGB/CatBoost | LightGBM | fastest/accurate at this scale | PR-AUC 0.9897 |
| Temporal model | LSTM/hybrid/BiLSTM | 2-layer GRU, W=16 | parameter efficiency; no future leakage; streaming | smoke 0.9876; lag evidence 0.9897 |
| Window | W=2/8/32 | W=16 (W=8 trade-off) | gains continue through 16 | 0.9782→0.9830→0.9883→0.9897 |
| Graph arch | GCN/SAGE/GIN/GAT | GATv2, edge features | edge attributes matter; inductive | GAT 0.9879 vs tabular 0.9769 |
| Edges | kNN/complete/simulator | radius-55 3D from `rel_*` | matches sensing model | 299/300 ±1 |
| ST arch | ST-GCN/TGN/attention | GATv2 + per-node GRU | time-varying topology; no edge memory | component evidence |
| ST window | T=4/16 | T=8 primary | balances latency/compute; W=16 ablation | 0.9883 vs 0.9897 |
| Attrition | pad/impute | drop post-loss; mask | dead agents are unobservable | permanent ID loss |
| Environment features | include | drop | no measurable gain | no-env 0.9834 vs 0.9825 |
| Global centralities | include in ensemble | only degree | redundant | remove all: 0.9816 vs 0.9825 |
| Splits | random rows | scenario-grouped + LOSO/LODO | rows/agents correlated | LOSO 0.905–0.995; LODO 0.967–0.989 |
| Metrics | accuracy | PR-AUC, F1, recall, Top-K, per-subset | imbalance + operational FN cost | base rate 0.389 |
| Anomalies | keep | flag/exclude 0.07% inconsistent rows | positions conflict with velocity/stats | c002 2.37%, c099 0.23% |
| Promotion handling | oversample/ignore | separate reporting + detachment features | exogenous timing; operational honesty | post-promo recall 0.40 |

---

## 11. Data Splitting and Evaluation Strategy

### 11.1 Splits

- **Primary:** group split by `case_id`, 70/15/15 scenarios, stratified over `(size × attrition)`, **5 seeds**; report mean ± std. (Preliminary results use seed 0.)
- **Generalization suites:** leave-one-size-out (9 folds), leave-one-attrition-out (4 folds), leave-one-clutter-tertile-out.
- **Forbidden:** row-level, agent-level, or timestep-level random splits; windows crossing scenarios; test statistics in normalization.
- Fit scalers/encoders on train scenarios only.
- For temporal models, windows are built within a scenario; no cross-scenario padding.

### 11.2 Metrics

- **Primary:** PR-AUC, F1 at a validation-tuned threshold, recall at ≥90% precision, false-negative rate.
- **Secondary:** ROC-AUC, precision, balanced accuracy, MCC, Brier/ECE.
- **Ranking:** Top-K precision/recall with K = true critical count per timestep; NDCG@K; per-scenario Spearman.
- **Aggregation:** row-level and agent-level; macro average over scenarios.
- **Mandatory subsets:** never-promoted, pre-promotion, post-promotion; per-size; per-attrition; isolated vs escorted; early/mid/late mission.
- **Baselines:** majority; logistic regression; `criticality_score` reference (non-deployable); the four models.

### 11.3 Preliminary results (scenario-grouped, seed 0, validation-tuned threshold)

| Model | PR-AUC | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| Static LightGBM | 0.9782 | 0.9828 | 0.9212 | 0.909 | 0.934 |
| **LightGBM + lags (W≈16)** | **0.9897** | **0.9923** | **0.9463** | 0.938 | 0.955 |
| Local-only LightGBM | 0.9806 | 0.9846 | 0.9261 | 0.911 | 0.942 |
| No-env LightGBM | 0.9834 | 0.9876 | 0.9328 | 0.930 | 0.936 |
| Minus `min_neighbor_dist` | 0.9758 | 0.9815 | 0.9112 | 0.888 | 0.936 |
| Kinematics only | 0.9302 | 0.9465 | 0.8441 | 0.855 | 0.834 |
| Scenario prior only | 0.4487 | 0.5406 | 0.5725 | 1.000 | 0.401 |
| Plus `elapsed_step` | 0.9863 | 0.9898 | 0.9374 | 0.936 | 0.938 |
| `criticality_score` (leak) | 0.9997 | 0.9997 | 0.9948 | 0.992 | 0.998 |
| **GATv2 (smoke)** | **0.9879** | 0.9909 | **0.9435** | 0.940 | 0.947 |
| Same-snapshot LightGBM | 0.9769 | 0.9825 | 0.9202 | 0.925 | 0.916 |
| **GRU W=8 (smoke)** | 0.9876 | 0.9907 | 0.9446 | 0.944 | 0.945 |
| Flattened-window LightGBM | 0.9901 | 0.9925 | 0.9522 | 0.951 | 0.953 |

**Generalization:** LOSO PR-AUC 0.9046 (n=18) → 0.9946 (n=68); LODO 0.9674 (medium) → 0.9885 (none).
**Mission progress:** PR-AUC 0.9735 (first quintile) → 0.9953 (last).
**t=0 cross-section:** PR-AUC 0.8815, F1 0.797.
**Agent-level:** PR-AUC 0.9992, F1 0.9934.
**Promotions:** post-promotion recall 0.397 (temporal) / 0.365 (static); never-critical FPR 0.041/0.047; 78.6% positive score jumps; median jump +0.004.

**Caveats:** single seed; threshold tuned on a 17-scenario validation split; smoke GNN/GRU use subsets (8k graphs / 150k windows). Final training must repeat with 5 seeds and full data.

---

## 12. Outstanding Questions and Uncertainties

1. **Promotion mechanism**: unknown whether promotions are scripted, role handover, or exposure-driven. Pre-event prediction is weak (median score jump ≈0). Treat as an exogenous event and report separately.
2. **`criticality_score` provenance**: R² 0.66 from geometry suggests a mixed geometric/label-aware computation. It is excluded; re-admit only if the generation code proves it label-independent.
3. **Why critical agents are escorted**: whether the formation is a deliberate protection strategy or a label-independent emergent pattern. The data show a strong escort signature but cannot prove causality.
4. **Small-swarm degradation** (n=18 PR-AUC 0.905): fewer neighbors/coarser ranks; consider size-aware normalization or per-size calibration.
5. **Row anomalies** (~0.07%, c002/c099): position–velocity inconsistency. Exclude/flag; root cause unknown.
6. **`obstacle_clustering` with zero obstacles**: likely a configuration parameter; currently dropped, so this does not affect models.
7. **Distributed deployment**: local-only results are close; real communication latency/loss is not modeled.
8. **ST model**: not fully trained; expected ≈0.99 but unverified. GPU + stride-subsampled training required.
9. **Absolute target coordinates**: only distances are given; bearing-to-target is not reconstructible without them.
10. **Threshold transfer**: validation-tuned thresholds may not transfer across swarm sizes; calibrate per size or use ranking/Top-K operationally.

---

## 13. Final Recommendations

### 13.1 Definitive designs

1. **Ensemble:** LightGBM on kinematics + radial + local escort + recomputed degree + causal lags (1/2/4/8/16). No env, no leak features. PR-AUC ≈0.99, F1 ≈0.946.
2. **Temporal:** 2-layer unidirectional GRU (hidden 96), W=16, masked causal windows; local+kinematic+degree features. Use for streaming/edge deployment; accuracy ≈0.99, not superior to GBDT at W=8.
3. **Spatio-Graph:** 3-layer GATv2 (hidden 96, 4 heads, edge_dim 7) on the reconstructed radius-55 graph; local+radial node features; 7 edge features; minimal global vector. Message passing gives +0.011 PR-AUC over identical features without structure.
4. **Spatio-Temporal:** shared GATv2 encoder + per-node GRU(96), T=8, per-step topology, node masks; expected ≈0.99. Train with stride-subsampled timesteps.

### 13.2 Exact final inputs

- **Ensemble:** `X[N,≈45]` tabular vector (feature table §5.2), `case_id` group key.
- **GRU:** `[B,16,24]` + `mask[B,16]` + optional static `[B,3]`.
- **GAT:** `Data(x=[N,24], edge_index=[2,E], edge_attr=[E,7], global=[1,6])`.
- **ST-GAT:** `[B,8,N_max,24]` + per-step `edge_index/edge_attr` + `node_mask[B,8,N_max]` + `global[B,8,6]`.

### 13.3 What must not be used

`criticality_score`, `was_promoted`, `is_critical`, `case_id`/`agent_id` as inputs, future steps, post-attrition observations, simulator adjacency, aggregates containing the label, and any whole-trajectory statistic.

### 13.4 Reporting requirements

Report row-level and agent-level metrics; per-size and per-attrition breakdowns; LOSO/LODO; early/mid/late mission; never/pre/post-promotion subsets; Top-K; calibration; false-negative rate at the operating threshold. State the single-seed preliminary numbers as preliminary.

---

## Appendix A — Experiment Log (Phase 6 reporting format)

| # | Tested | Why | Method | Result | Implication |
|---|---|---|---|---|---|
| E1 | Integrity | validate metadata | full-file pandas audit | 1,639,166×37; 0 dup/NaN; rel/distance exact | data is clean except 0.07% anomalies |
| E2 | Scenario profile | know composition | groupby audit | 100 scenarios; sizes/attrition; 197–600 steps | stratified grouped splits |
| E3 | Label dynamics | define target | per-agent diffs | 85 0→1 promotions; no 1→0 | current-state target; promotion subsets |
| E4 | `criticality_score` | leakage | AUC/stability/regression | AUC 0.9984; R² 0.66 from geometry | exclude; leak ablation only |
| E5 | `was_promoted` | leakage | transition check | flips exactly with label | exclude |
| E6 | Graph definition | reconstruct structure | radius/2D/lag tests | 299/300 ±1; 3D 55 | rebuild graph from `rel_*` |
| E7 | Redundancy | reduce features | Spearman/MI | degree↔closeness 0.993; env weak | drop redundant/central features |
| E8 | Leak-free baselines | feasibility | grouped LightGBM | PR-AUC 0.9897; F1 0.9463 | task is learnable without leakage |
| E9 | Ablations | isolate signal | feature groups | kinematics 0.9302; local-only 0.9806; prior 0.4487 | local escort + motion dominate |
| E10 | Window | choose history | lags 1–16 | 0.9782→0.9897 | W=16 (W=8 trade-off) |
| E11 | Promotion behaviour | failure mode | static vs temporal subsets | post-promo recall 0.40 | dedicated handling needed |
| E12 | LOSO/LODO | generalization | held-out size/attrition | 0.905–0.995 / 0.967–0.989 | generalizes; small swarms harder |
| E13 | GAT smoke | validate graph model | 8k graphs, GATv2 | 0.9879 vs 0.9769 tabular | graph family justified |
| E14 | GRU smoke | validate temporal model | 150k windows, GRU | 0.9876 vs 0.9901 flattened | GRU for streaming, not accuracy |
| E15 | ST model | feasibility | not run | — | components validated; GPU needed |

## Appendix B — Reproduction

```powershell
uv venv experiment/.venv --python 3.12
uv pip install --python experiment/.venv/Scripts/python.exe pandas numpy scikit-learn lightgbm networkx pyarrow torch torch-geometric
experiment\.venv\Scripts\python.exe experiment\audit_dataset.py
experiment\.venv\Scripts\python.exe experiment\diagnose_graph.py
experiment\.venv\Scripts\python.exe experiment\diagnose_graph2.py
experiment\.venv\Scripts\python.exe experiment\diagnose_graph3.py
experiment\.venv\Scripts\python.exe experiment\diagnose_positions.py
experiment\.venv\Scripts\python.exe experiment\prelim_lgbm.py
experiment\.venv\Scripts\python.exe experiment\experiments2.py
experiment\.venv\Scripts\python.exe experiment\experiments3.py
experiment\.venv\Scripts\python.exe experiment\smoke_gat.py
experiment\.venv\Scripts\python.exe experiment\smoke_gru.py
```

Artifacts: `experiment/artifacts/audit_summary.json`, `scenario_profile.csv`, `promotions.csv`, `prelim_results.json`, `experiments2.json`, `experiments3.json`, `smoke_gat.json`, `smoke_gru.json`, `graph_residuals.csv`.

---

## 14. Phase 7 Final Implementation & Cross-Model Benchmark Results

The end-to-end production pipeline implemented under `src/` trained, tuned, and evaluated all four leak-free architectures on the frozen 70/15/15 scenario split (`artifacts/splits/primary_split.json`).

### 14.1 Cross-Model Performance Summary

Evaluated on the held-out test split (15 scenarios, 305,016 agent observations) at operating thresholds tuned strictly on the validation set:

| Model | PR-AUC | ROC-AUC | F1 | Recall | Precision | MCC | Top-K Recall | Recall@90%P | Agent-Level F1 | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GAT (Shared GAT+GRU)** | **0.9921** | **0.9950** | **0.9619** | 0.9616 | **0.9623** | **0.9400** | **0.9598** | **0.9829** | **1.0000** | 2417.2s |
| **GRU (2-Layer)** | **0.9923** | 0.9946 | 0.9593 | **0.9653** | 0.9535 | 0.9357 | 0.9560 | 0.9802 | 0.9960 | 695.9s |
| **LightGBM (54 Feat)** | 0.9910 | 0.9941 | 0.9503 | 0.9536 | 0.9471 | 0.9215 | 0.9539 | 0.9740 | 0.9960 | **89.0s** |
| **GATv2 (2-Layer)** | 0.9890 | 0.9923 | 0.9495 | 0.9554 | 0.9436 | 0.9200 | 0.9499 | 0.9703 | 0.9980 | 547.0s |

### 14.2 Key Engineering & Scientific Findings

1. **Spatio-Temporal Integration (ST-GAT & GRU Champions):**
   - **ST-GAT** achieves the highest overall F1 (**0.9619**), MCC (**0.9400**), Precision (**0.9623**), and a perfect Agent-Level Trajectory F1 of **1.0000** (zero false agent misclassifications across all held-out scenarios). Combining a 2-layer spatial GATv2 backbone with a per-node temporal GRU over $T=8$ sliding steps allows it to fully capture coordinated swarm trajectories and protective escort shielding formations.
   - **GRU** delivers matching peak PR-AUC (**0.9923**) and the highest raw recall (**0.9653**), while providing an $O(1)$ streaming inference interface (`predict_step_streaming`) that updates hidden states in $0.06$ms without buffering historical snapshots.
2. **Tabular GBDT Efficiency (LightGBM):**
   - LightGBM achieves **0.9910 PR-AUC** and **0.9503 F1** while training in only **89.0s** (27x faster than ST-GAT, 8x faster than GRU). With causal multi-lag delta features over offsets $\{1, 2, 4, 8, 16\}$, it captures temporal dynamics with minimal compute footprint.
3. **Spatial Graph Neural Network Remediation (GATv2):**
   - Resolving edge feature distortion (preserving physical units: $dist/55 \in [0, 1]$, relative velocities, trigonometric bearings) and removing global context clutter unlocked GATv2 performance to **0.9890 PR-AUC**, **0.9923 ROC-AUC**, and **0.9495 F1** with **0.9980 Agent F1**.
   - Spatial message-passing over dynamic radius-55 graphs provides strong instantaneous relational modeling without requiring temporal lag histories.
4. **Feature Independence & Redundancy Removal:**
   - Replacing redundant `neighbor_frac` ($0.9999$ correlation with `degree_centrality_c`) with `centroid_vel_alignment` ($\frac{\mathbf{v}_i \cdot \mathbf{v}_c}{\|\mathbf{v}_i\|\|\mathbf{v}_c\| + \epsilon}$) eliminated multicollinearity, providing distinct kinematic coordination information that directly separates coherent swarm core agents from peripheral decoys.

### 14.3 Automated Sanity Suite Verification (SC-1 to SC-5)

All 5 core integrity checks were codified in `src/tests/test_sanity.py` and passed with zero violations:
- **SC-1 (Leakage & Split Isolation):** Confirmed strict absence of `criticality_score`, `was_promoted`, `is_critical`, identifiers, and environment features from input schemas. Verified primary split scenario sets are 100% mutually disjoint.
- **SC-2 (Causal Invariance):** Perturbation of future observations ($t > t_{\text{cut}}$) resulted in bit-for-bit identical feature matrices for all historical steps ($t \le t_{\text{cut}}$).
- **SC-3 (Graph Fidelity & Self-Loops):** Reconstructed 3D radius-55 sensing graphs matched dataset neighbor counts within $\pm 1$ neighbor and guaranteed self-loops for isolated agents.
- **SC-4 (Temporal Masking & Attrition):** Verified left-padded sequence alignment, causal zero-padding at $t < W$, and proper exclusion of attrited agents.
- **SC-5 (Unified Inference API Parity):** Verified end-to-end telemetry loading, snapshot prediction, and streaming inference for all 4 models via `SwarmCriticalityPredictor`.

### 14.4 Production Execution Commands

```powershell
# Run automated verification suite
experiment\.venv\Scripts\python.exe -u src/tests/test_sanity.py

# Train individual models
experiment\.venv\Scripts\python.exe -u src/training/train_lgbm.py
experiment\.venv\Scripts\python.exe -u src/training/train_gru.py
experiment\.venv\Scripts\python.exe -u src/training/train_gat.py
experiment\.venv\Scripts\python.exe -u src/training/train_stgat.py

# Generate cross-model comparison summary
experiment\.venv\Scripts\python.exe src/evaluation/summarize.py
```

The dataset remains untouched.
