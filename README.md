# Critical Agent Identification in Autonomous Aerial Swarms

An end-to-end, target-agnostic machine learning framework for identifying high-value command and communication relay nodes within coordinated unmanned aerial vehicle (UAV) swarms.

Operating under realistic telemetry constraints where the ground station receives only downlinked 3D Cartesian position and velocity vectors $(P_x, P_y, P_z, V_x, V_y, V_z)$, our framework causally derives a **22-dimensional invariant feature representation** and benchmarks four model families across 15 unseen test swarm scenarios (305,016 held-out frames).

---

## 🚀 Key Results on Held-Out Test Swarms (305,016 Observations)

| Model Architecture | Operating $\theta^*$ | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | MCC | Top-$K$ Rec | Binary NDCG | Graded NDCG | Agent-Level F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (52 Lags)** | $0.4800$ | $0.9915$ | $0.9944$ | $0.9541$ | $0.9504$ | $0.9577$ | $0.9277$ | $0.9550$ | $0.9662$ | $0.9188$ | $0.9980$ |
| **GRU (2-Layer)** | $0.5900$ | **$0.9929$** | **$0.9949$** | $0.9579$ | **$0.9668$** | $0.9493$ | $0.9334$ | $0.9596$ | $0.9698$ | **$0.9230$** | **$1.0000$** |
| **GATv2 (Spatial GNN)** | $0.6800$ | $0.9906$ | $0.9937$ | $0.9535$ | $0.9539$ | $0.9531$ | $0.9267$ | $0.9547$ | $0.9665$ | $0.9148$ | $0.9980$ |
| **ST-GAT (Spatio-Temporal)** | $0.4800$ | $0.9922$ | $0.9948$ | **$0.9591$** | $0.9596$ | **$0.9585$** | **$0.9354$** | **$0.9615$** | **$0.9712$** | $0.9134$ | **$1.0000$** |

- **Flawless Mission Identification:** Both GRU and ST-GAT achieve a perfect **$1.0000$ Agent-Level Trajectory F1**, eliminating whole-mission identity errors across hundreds of drone trajectories.
- **Order-Aware Salvo Precision:** Under a realistic 5-missile interceptor salvo budget ($K=5$), ST-GAT achieves **$98.72\%$ precision** ($<1.3$ wasted interceptors per 100 shots).
- **Dynamic In-Flight Promotion Tracking:** When an active commander is eliminated and a decoy assumes command ($0 \to 1$), GRU rapidly re-identifies the promoted node with **$82.06\%$ recall**.

---

## 📐 Canonical 22 Target-Agnostic Feature Schema

All features are derived causally from $(P, V)$ telemetry with sampling period $\Delta t = 0.05\text{ s}$ ($\text{sample\_every} = 2$ simulation ticks):

1. **Kinematics (6):** `speed`, `accel`, `heading_rate`, `flight_path_angle`, `heading_sin`, `heading_cos`
2. **Centroid Geometry (3):** `dist_to_centroid`, `rank_dist_to_centroid`, `rel_z`
3. **Escort Density & Multi-Scale Shells (8):** `num_neighbors`, `mean_neighbor_dist`, `min_neighbor_dist`, `weighted_density`, `angular_coverage`, `n_within_r15`, `n_within_r30`, `n_within_r45`
4. **Symmetry & Alignment (4):** `nn_bearing_entropy`, `local_density_contrast`, `centroid_vel_alignment`, `is_isolated`
5. **Local Dynamic Graph (1):** `degree_centrality_c`

---

## 🛠️ Project Structure

```text
├── .python-version               # Pinned Python version (3.12)
├── pyproject.toml                # uv / PEP 621 project configuration
├── uv.lock                       # uv lockfile
├── dataset/
│   └── Dataset_Metadata.md       # Dataset scenario descriptions and schemas
├── artifacts/                    # Model checkpoints, scalers, and evaluation metrics
├── report/
│   ├── Model_Evaluation_Metrics_Guide.md # Comprehensive metrics computation breakdown
│   ├── final-report/             # IEEE Conference LaTeX paper and bibliography
│   └── ppt/                      # Presentation slides and visual assets
├── src/
│   ├── config.py                 # Central hyperparameters and feature schemas
│   ├── data/                     # Causal derivation and temporal graph windowing
│   ├── models/                   # LightGBM, 2-Layer GRU, GATv2, and ST-GAT
│   ├── evaluation/               # Multi-tier decision, ranking, and salvo metrics
│   ├── inference/                # Unified causal streaming inference engine
│   └── tests/                    # Sanity check test suite (SC-1 to SC-5)
└── README.md
```

---

## ⚡ Quickstart with `uv`

### 1. Environment Setup
Install dependencies and sync the virtual environment using [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync
```

### 2. Run Sanity Checks
Verify data leakage isolation, causal graph invariance, and unified inference:

```bash
uv run python src/tests/test_sanity.py
```

### 3. Unified Streaming Inference Example
Inference consumes raw telemetry $(P_x, P_y, P_z, V_x, V_y, V_z)$ and predicts criticality probabilities online:

```bash
uv run python src/inference/predict.py
```
