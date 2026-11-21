# Ground-Station Inference Walkthrough: From Raw Telemetry to Critical-Node Predictions

This guide walks through how real-time inference works at a ground station for all four models (**LightGBM**, **GRU**, **GATv2**, and **ST-GAT**), checks whether your proposed telemetry packet is sufficient, and explains every transformation from raw packets to per-agent predictions in plain terms.

---

## 1. Short Answer: Is Your Ground-Station Telemetry Enough?

Suppose at timestep $t$, your ground station receives one synchronized telemetry packet for every currently alive drone $i \in \{1, \dots, N_t\}$ containing:

- `agent_id`
- Global position: `(x, y, z)`
- Global velocity: `(vx, vy, vz)`
- Along-track acceleration: `accel`
- Orientation: `heading`, `flight_path_angle`

**Yes — with two minor caveats that are easy to handle:**

| Info Type | Needed? | Verdict & Measured Evidence |
|---|---|---|
| **Obstacle / clutter info** (`n_obstacles`, density, clutter, etc.) | **NO** | **Not needed at all.** Dropped from all 4 models because experiments proved environment features add zero gain (`0.9834` PR-AUC without env vs `0.9825` with env). |
| **Target info** (`dist_to_target`, `rank_dist_to_target`) | **NO** | **Formally dropped.** The models operate completely target-agnostic on **22 base features**, ensuring zero dependency on mission waypoint access. LightGBM PR-AUC remained exceptional (`0.9915`). |
| **`heading_rate` & `accel`** | **DERIVABLE VIA CAUSAL DIFFERENCES** | The ground station computes both causally from downlinked position and velocity: $\Delta v / \Delta t$ and $\Delta \psi / \Delta t$ ($r = 0.9953$ for heading rate, $r = 0.9561$ for accel). |
| **Precomputed escort stats** (`num_neighbors`, `weighted_density`, `angular_coverage`, etc.) | **DERIVABLE FROM `(x, y, z)`** | Drones do **not** need to compute or transmit these. The ground station computes all neighborhood, shell, density, angular-coverage, and graph features directly from the agents' `(x, y, z)` positions. |

> **Finalized Architecture Contract:**  
> The models train on the **22 canonical base features** (`BASE_FEATURES_22`), completely excluding target information (`dist_to_target`, `rank_dist_to_target`). Real-time ground station inference derives all kinematics, radial geometry, and escort features dynamically from downlinked 3D position `(x, y, z)` and velocity `(vx, vy, vz)`.

---

## 2. Minimum Telemetry Needed at the Ground Station

At every sampling tick $t$ (every 2 simulation steps in the dataset, $\Delta t = 0.10\text{s}$), the ground station needs **one table of $N_t$ rows** (one row per alive agent):

| Field | Required? | Role |
|---|---|---|
| `step` / `timestamp` | **Yes** | Orders snapshots in time; never fed into models. |
| `agent_id` | **Yes** | Tracks each drone across time and aligns output predictions; never fed into models. |
| `px, py, pz` (or `x, y, z`) | **Yes** | 3D Position. Used to compute centroid, relative positions, 3D radius-55 graph, neighbor distances, density, angular coverage, and edge bearings. |
| `vx, vy, vz` | **Yes** | 3D Velocity. Gives scalar `speed`, `heading_sin/cos`, `flight_path_angle`, `centroid_vel_alignment`, and causal `accel` / `heading_rate`. |

Notice how **few** raw signals this is: just position, velocity, acceleration, and orientation per drone. Everything else is engineered at the ground station.

---

## 3. Step-by-Step: What the Ground Station Computes at Time $t$

Imagine at $t = 30$, 24 drones report in. Here is the exact shared pipeline that turns their raw telemetry into model inputs:

### Step 1 — Strip Identifiers Aside
Keep `(step, agent_id)` in a separate metadata list so we can attach each predicted probability back to the right drone at the end. Identifiers never enter any model.

### Step 2 — Center Positions on the Swarm Centroid (Translation Invariance)
Compute the average position of all alive drones right now:
$$\mathbf{c}_t = \frac{1}{N_t}\sum_{i=1}^{N_t} (x_i, y_i, z_i)$$
Subtract $\mathbf{c}_t$ from every drone's position:
$$\text{rel}_i = (x_i - c_{x,t},\ y_i - c_{y,t},\ z_i - c_{z,t})$$
Why? Now the swarm can fly anywhere in the world (over `(0, 0)` or `(5000, -2000)`), and the features remain identical. From $\text{rel}_i$ we immediately get:
- `rel_z` (altitude relative to the swarm center)
- `dist_to_centroid` $= \|\text{rel}_i\|_2$ (how far out from the center the drone is)
- `rank_dist_to_centroid` $= \frac{\text{rank}_i - 1}{N_t - 1} \in [0, 1]$ (`0` = dead center, `1` = outermost edge)

*(If target `(tx, ty, tz)` is known, similarly compute `dist_to_target` and `rank_dist_to_target`.)*

### Step 3 — Derive Clean Kinematics
From `(vx, vy, vz)` and `heading`:
- `speed` $= \sqrt{v_x^2 + v_y^2 + v_z^2}$
- `heading_sin` $= \sin(\text{heading})$, `heading_cos` $= \cos(\text{heading})$ (avoids the $-\pi / +\pi$ wrap-around jump)
- `accel`, `flight_path_angle`, and `heading_rate` (direct or from $\Delta\text{heading}/\Delta t$)

### Step 4 — Build the 3D Radius-55 Neighbor Graph
Compute the 3D Euclidean distance $D_{ij} = \|\text{rel}_i - \text{rel}_j\|_2$ between every pair of alive drones $(i, j)$.  
Connect two drones $i \neq j$ with an edge whenever $D_{ij} \le 55.0$ units (the sensing/escort radius).

> **Important:** Dead/lost drones from earlier timesteps are **absent** from the packet at $t$, so they are NOT in the graph and NEVER appear as zero-position ghost nodes.

### Step 5 — Compute Per-Drone Escort & Geometry Features
Using the 3D distances $D_{ij}$ and the radius-55 graph, compute for each drone $i$:
- **Neighbor counts & isolation:** `num_neighbors` (within radius 55), `centroid_vel_alignment = (v_i · v_c) / (||v_i|| * ||v_c|| + 1e-6)` (replaces redundant `neighbor_frac`), `is_isolated` (`1` if `num_neighbors == 0` else `0`).
- **Distances:** `mean_neighbor_dist`, `min_neighbor_dist` (`0.0` if isolated).
- **Local density & surrounding:**
  - `weighted_density`: Gaussian distance-weighted neighbor count $\sum_{j \in N_{55}(i)} \exp(-D_{ij}^2 / 324)$ (verified `0.99999` correlation with the dataset column!).
  - `angular_coverage`: fraction of 48 3D angular sectors (12 azimuth $\times$ 4 elevation) around drone $i$ occupied by at least one neighbor (measures "how surrounded/escorted am I?").
  - **Concentric shells:** `n_within_r15`, `n_within_r30`, `n_within_r45` (tight escort vs loose escort).
  - `nn_bearing_entropy`: how evenly spread neighbors are around drone $i$'s heading (`0` = all on one side, `1` = evenly surrounded on all sides).
  - `local_density_contrast`: `own weighted_density - mean(neighbors' weighted_density)` (are you at the peak of an escort cluster?).
  - `degree_centrality_c`: `num_neighbors / (N_t - 1)` on the reconstructed graph.

At the end of Steps 2–5, every alive drone $i$ has **one clean flat feature vector** `x_t[i]` (24 numbers in the full plan, or 21–22 if target/yaw-rate are dropped).

### Step 6 — Compute Edge & Global Features (Only Needed for GATv2 & ST-GAT)
- **For each directed edge `neighbor src -> receiver dst` (plus a self-loop `i -> i`):** compute 7 numbers describing how `src` moves relative to `dst`:
  `[dist/55, rel_speed, closing_rate, bearing_cos, bearing_sin, dheading_cos, dheading_sin]`.
- **For the whole swarm snapshot at $t$:** compute 6 numbers summarizing swarm state:
  `[alive_count, swarm_spread, swarm_centroid_speed, graph_density, n_components, largest_component_frac]`.  
  *(Note: `swarm_centroid_speed` is just $\|\frac{1}{N_t}\sum_i (v_{x,i}, v_{y,i}, v_{z,i})\|_2$, computed directly from your global velocities!)*

### Step 7 — Update the Ground Station's Short Causal Buffer
The ground station keeps a small rolling ring buffer in RAM storing the last **16 snapshots** per active `agent_id` (plus the last **8 snapshot graphs** if running ST-GAT). When a drone disappears (attrition), its buffer entry is removed.

---

## 4. How Each of the Four Models Runs Inference

All four models start from the **exact same Step 1–7 outputs**, then package them differently:

```text
Raw Telemetry Packet at time t (N_t alive drones)
        │
        ▼  [Shared Feature Builder: Steps 1–6]
   ┌────┴─────────────────────────────────────────────────────────┐
   │  Per-agent vector x_t [N_t, 24]                              │
   │  + Graph (edge_index [2, E_t], edge_attr [E_t, 7], glob [6]) │
   └────┬──────────────┬───────────────┬──────────────┬───────────┘
        │              │               │              │
        ▼              ▼               ▼              ▼
   [LightGBM]        [GRU]          [GATv2]       [ST-GAT]
  Needs buffer     O(1) state     0 history!     8-step graph
  up to 16 prev    h_{t-1} per     Runs on       ring buffer
   snapshots       agent (or       snapshot t     [t-14 .. t]
                   16-step buf)      alone
```

### Model 1: LightGBM (Tabular + Causal Lags)
- **What it needs from the buffer:** Each alive drone's current `x_t` plus 5 historical snapshots at offsets `1, 2, 4, 8, 16` ticks back (`[N_t, 54]` total = 24 current features + 30 `current - past` differences on `dist_to_centroid`, `rank_dist_to_centroid`, `num_neighbors`, `weighted_density`, `angular_coverage`, `speed`).
- **What happens at the very start of a mission ($t = 0$ to $t = 30$)?** Unavailable lags are simply `NaN` — LightGBM handles `NaN` natively, so it still predicts immediately from $t = 0$ onward (improving steadily as history fills up to 16 ticks).
- **Scaling:** None required (tree splits operate on raw units).
- **Output:** `prob = model.predict_proba(X_54)[:, 1]`.

### Model 2: 2-Layer GRU (Streaming Recurrent)
- **Two equivalent ways to run it:**
  1. **True Streaming Mode ($O(1)$ memory per drone — ideal for real-time):**  
     The ground station stores one small hidden-state tensor `h[agent_id]` of shape `[2, 96]` (~768 bytes per drone!). When snapshot $t$ arrives, it standardizes `x_t` using the training `(mu, sd)` scaler, feeds **only the single step `x_t`** + `h[agent_id]` into the GRU, gets the updated `h_new[agent_id]`, and passes it through the MLP head to get `sigmoid(logit)`. New drones start with `h = 0`; lost drones are deleted from the dictionary.
  2. **Window Mode (stateless):**  
     Pull the last `W=16` steps of `x` from the buffer, left-pad with zeros if $t < 30$ (`mask` marks real steps), and run the packed GRU. Because of `pack_padded_sequence`, both modes produce **identical probabilities**.
- **Does it use neighbor graph edges?** No — each drone is scored independently using its own 24-feature trajectory.

### Model 3: 3-Layer GATv2 (Single-Snapshot Graph)
- **What it needs from the buffer:** **Nothing — zero history required!** It works on a single isolated snapshot $t$.
- **How it works:**
  1. Standardizes node features `[N_t, 24]`, edge features `[E_t, 7]`, and global vector `[1, 6]` using training scalers.
  2. Runs 3 rounds of graph attention message passing (`GATv2Conv`) over the alive drones and their radius-55 neighbors (each round also fuses the global swarm context vector).
  3. Each drone effectively "asks its 1-hop, 2-hop, and 3-hop neighbors" whether they are forming an escort shell around it.
  4. Passes each node's final 96-dim embedding through the MLP head $\rightarrow$ `sigmoid(logit)`.
- **Variable swarm size:** Works natively for any $N_t$ (whether 18, 50, or 75 drones) with no padding needed.

### Model 4: ST-GAT (8-Step Dynamic Spatio-Temporal Graph)
- **What it needs from the buffer:** A rolling ring buffer of the last **8 consecutive snapshots** (`T = 8`, spanning 16 simulation steps; left-padded with empty steps if $t < 14$).
- **Why it cannot do $O(1)$ hidden-state streaming like GRU:** Because drones move, the neighbor graph `edge_index_tau` changes at every step $\tau$. So at each step $\tau$, the spatial GATv2 encoder must run on the snapshot graph at $\tau$.
- **How it works (step by step):**
  1. For each of the 8 buffered snapshots $\tau \in \{t-14, \dots, t\}$, run the **shared 3-layer GATv2 encoder** over the drones that were alive *at that step $\tau$* (so if a drone died 3 steps ago, it participated in message passing before it died, and is completely masked out after it died).
  2. This gives each currently alive drone $i$ a sequence of up to 8 spatial embeddings `z_{i, tau}` (`96` dims each), concatenated with the 6 global features (`102` dims/step).
  3. Run the **per-node 1-layer GRU** over those 8 steps (skipping left-padded empty steps if $t < 14$) and apply the classification head $\rightarrow$ `sigmoid(logit)` for each drone alive at $t$.

---

## 5. Final Output at the Ground Station

Every model returns the exact same aligned structure for timestep $t$:

```text
case_id | step | agent_id | prob   | pred_class | threshold
--------+------+----------+--------+------------+----------
live_01 |  30  |    0     | 0.041  | False (0)  | 0.45
live_01 |  30  |    1     | 0.932  | True  (1)  | 0.45
...
```

Here `threshold` is the fixed operating threshold tuned on the validation set (never guessed at runtime).

---

## 6. Quick Comparison Across All 4 Models at Inference Time

| Property | LightGBM | GRU | GATv2 | ST-GAT |
|---|---|---|---|---|
| **History needed** | Up to 16 past ticks (works at $t=0$ with `NaN` lags) | $O(1)$ hidden state per drone (or 16-tick window) | **0 (single snapshot)** | Rolling 8-snapshot graph buffer |
| **Sees neighbor interactions directly?** | Indirectly (via summary stats) | Indirectly (via summary stats) | **Yes (3-hop message passing + edges)** | **Yes (3-hop + 8-step evolution)** |
| **Handles variable swarm size & attrition?** | Yes | Yes (drop lost agent's state) | Yes (native graph size $N_t$) | Yes (mask dead nodes per step) |
| **Needs feature standardization?** | No | Yes (`mu, sd`) | Yes (`mu_x, mu_e, mu_g`) | Yes (`mu_x, mu_e, mu_g`) |
| **Compute per tick (75 drones)** | < 1 ms (CPU) | < 1 ms (CPU/GPU) | ~1–3 ms | ~5–15 ms (8 graphs + GRU) |

---

## 7. Two Decisions to Confirm Before Starting Implementation

Before Gemini Flash writes `src/config.py` and `src/data/features.py`, confirm these two deployment assumptions:

1. **Target Coordinates & Yaw Rate Assumption:**
   - **Option A (Friendly C2 Ground Station — current `Final_Implementation_Plan.md`):** Ground station knows the mission target `(tx, ty, tz)` and receives/derives `heading_rate`. Keeps all **24** base features (`PR-AUC ≈ 0.9893` in preliminary check).
   - **Option B (Target-Agnostic / Passive Radar Ground Station):** Drop `dist_to_target`, `rank_dist_to_target`, and (if unavailable) `heading_rate` so inference requires **only** `(id, x, y, z, vx, vy, vz, accel, heading, flight_path_angle)` — reducing preliminary LightGBM PR-AUC by only `0.0015–0.0022` (`0.9871–0.9878`).
2. **Recompute Neighborhood Stats During Training Too:**
   - Because drones do not transmit `num_neighbors`, `mean_neighbor_dist`, `min_neighbor_dist`, `weighted_density`, or `angular_coverage` in your telemetry packet, updating `src/data/features.py` to compute all five directly from `(x, y, z)` (`weighted_density = sum exp(-d^2 / 324)`, `angular_coverage` = 12×4 bins) during **both training and inference** eliminates any subtle simulator-vs-ground-station formula mismatch.

