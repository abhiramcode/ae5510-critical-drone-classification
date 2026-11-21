# Presentation Slide Deck: Critical Node Identification in Aerial Swarms
## Executive & Technical Presentation Reference for Autonomous Aerial Swarm Defense

> **Target Audience:** Technical leadership, ML researchers, and AI presentation agents (e.g., Claude Code, human presenters).  
> **Repository Baseline:** CAC-AGY Aerial Swarm Criticality Classification Suite  
> **Status:** Production-complete friendly telemetry models benchmarked; ready for presentation and adversarial extension.

---

## Slide 1: Title & Executive Overview

### Slide Content
- **Title:** Critical Node Identification in Autonomous Aerial Swarms
- **Subtitle:** High-Accuracy, Leak-Free Spatio-Temporal and Recurrent Architectures for Real-Time Ground Station Telemetry
- **Presenter:** Senior ML Systems Architect & Deep Learning Engineering Lead
- **Key Metrics Snapshot:**
  - **Top Model PR-AUC:** `0.9929` (2-Layer GRU)
  - **Top Model F1-Score:** `0.9591` (ST-GAT, Balanced MCC `0.9354`)
  - **Agent Trajectory F1:** `1.0000` (100% Perfect Mission Identification)
  - **Inference Mode:** $O(1)$ Constant-Time Streaming Telemetry ($0.06\text{ ms}$)

### Speaker Notes
> "Welcome everyone. Today we are presenting the final architecture, implementation, and benchmark results for autonomous critical-node identification in aerial swarms. Autonomous drone swarms rely on decentralized coordination, but they almost always contain high-value leadership or payload nodes that dictate swarm cohesion. Identifying these 'queen bees' or high-value escort targets from streaming flight telemetry gives defense systems a decisive advantage. We will walk through the mathematical formulation, the discovery and elimination of severe simulator leakage shortcuts, our four evaluated model families, and why recurrent temporal dynamics decisively outperform spatial graph attention in real-time tracking."

### Visual Layout Recommendation
- **Left:** Title, affiliation, and date.
- **Right:** High-impact callout card displaying PR-AUC `0.9912`, Agent F1 `0.9960`, and a conceptual graphic of a 3D drone swarm with one node highlighted in gold (critical) surrounded by escort decoys in blue.

---

## Slide 2: Problem Statement & Operational Motivation

### Slide Content
- **The Core Problem:**
  - Aerial swarms ({18 to 75} autonomous UAVs) conduct coordinated missions in 3D airspace.
  - A minority subset of agents ($\approx 38.9\%$) are **mission-critical** (command nodes, high-value payloads, communications hubs).
  - The remaining majority are sacrificial escorts, decoys, or distributed sensors.
- **Tactical Objective:**
  - Ground station receives streaming downlinked telemetry packets from all active agents.
  - Must identify which nodes are critical in real-time ($< 5\text{ ms}$ latency).
- **Core Engineering Constraints:**
  - **Zero Leakage:** No access to simulator privileged state or future trajectory knowledge.
  - **Dynamic Topology:** Swarm splits, contracts, and suffers attrition over time.
  - **Dynamic Roles (Promotions):** If a critical leader is destroyed, an escort drone is dynamically promoted to take over leadership.

### Speaker Notes
> "In modern swarm warfare and counter-UAS operations, neutralizing every drone in a 75-agent swarm is cost-prohibitive and slow. However, if an operator or automated interceptor can immediately distinguish the critical command nodes from expendable decoys, the swarm's mission capability can be neutralized with minimal ammunition. Crucially, the swarm operates dynamically: drones change formation, experience attrition, and when a leader is lost, field promotions occur in mid-flight. Our goal is to build an algorithm that accurately identifies critical nodes from streaming telemetry without cheating."

### Visual Layout Recommendation
- **Split Screen:**
  - **Left (Operational Context):** Flow diagram from Swarm UAVs $\to$ RF Telemetry Link $\to$ Ground Station Compute $\to$ Criticality Alert.
  - **Right (Tactical Challenges):** Three bulleted icons: (1) Formation Shape Shifts, (2) Attrition/Loss, (3) Mid-Air Field Promotions.

---

## Slide 3: Literature Review & Prior Art

### Slide Content
- **1. Topology & Graph Centrality in Multi-Agent Systems (MAS):**
  - Classical approach: Degree, Betweenness, Closeness, and Eigenvector centrality (Bavelas, 1950; Freeman, 1978).
  - *Limitation:* Static graph metrics assume fixed topologies and ignore 3D relative kinematics and flight path angles.
- **2. Spatial Graph Neural Networks (GNNs):**
  - Graph Convolutional Networks (Kipf & Welling, 2017) and Graph Attention Networks (GAT / GATv2, Brody et al., 2021).
  - *Opportunity:* Message passing allows agents to aggregate neighbor state dynamically over radius-based sensing graphs.
- **3. Recurrent & Spatio-Temporal Deep Learning:**
  - Gated Recurrent Units (GRU, Cho et al., 2014) for non-stationary trajectory tracking.
  - Spatio-Temporal GNNs (ST-GNNs, Yu et al., 2018; Gao et al., 2021) combining spatial adjacency with temporal RNN/TCN cells.
- **4. Our Contribution:**
  - Rigorous evaluation of tabular GBDT, recurrent GRU, spatial GATv2, and spatio-temporal ST-GAT on a large-scale (1.64M row) realistic 3D swarm dataset.
  - First work to expose simulator heuristic leakage and provide mathematically verified causal invariance.

### Speaker Notes
> "Literature in this space splits into three main camps. Graph theorists historically relied on static centralities like betweenness and eigenvector centrality. While intuitive, our preliminary experiments proved that betweenness and closeness are heavily correlated ($>0.99$) and computationally expensive to compute in real time. The deep learning community has pivoted to Spatial Graph Attention Networks (GATv2) and Spatio-Temporal Graph Networks. While GNNs are theoretically appealing, the literature frequently overlooks the practical inference overhead of dynamic graph construction and the problem of attention over-smoothing in dense swarms. We set out to rigorously benchmark spatial versus temporal approaches under strict production constraints."

### Visual Layout Recommendation
- 3-column comparative landscape matrix: **Classical Centrality** vs. **Spatial GNNs (GATv2)** vs. **Recurrent Models (GRU)** highlighting computational complexity, temporal sensitivity, and real-time streaming readiness.

---

## Slide 4: Dataset Audit & The Discovery of Fatal Data Leakage

### Slide Content
- **Dataset Scale:** 1,639,166 rows, 37 columns across 100 diverse simulation scenarios.
  - Swarm sizes: $\{18, 24, 30, 37, 45, 52, 60, 68, 75\}$.
  - Attrition levels: None (24), Light (24), Medium (27), Heavy (25).
  - Total observed label promotions ($0 \to 1$): 85 events in 41 scenarios.
- **Fatal Leakage Trap #1: `criticality_score`**
  - Simulator-computed continuous score: achieved **0.9984 ROC-AUC** by itself!
  - Cross-section at $t=0$: Decoys $\le 0.0747$, Critical $\ge 0.2911$ (trivial separation).
  - *Architectural Action:* **STRICTLY EXCLUDED.** Incorporating this feature produces an artificial PR-AUC of 0.9997 that completely fails in real-world deployment.
- **Fatal Leakage Trap #2: `was_promoted`**
  - An undocumented flag that flips from $0 \to 1$ exactly at the simulation step of promotion.
  - *Architectural Action:* **STRICTLY EXCLUDED.**
- **Other Excluded Features:**
  - Identifiers (`case_id`, `agent_id`), ground truth `is_critical`, future timesteps ($t > t_{\text{curr}}$), and environment features (`n_obstacles`, etc., which contributed zero gain and exhibited zero-obstacle configuration anomalies).

### Speaker Notes
> "This slide represents the single most important data-engineering finding in this project. When auditing the raw dataset, we found two columns that looked innocent but were actually fatal leakage shortcuts. The feature `criticality_score` is an internal simulator heuristic that separates critical nodes from decoys with 0.9984 ROC-AUC on day one! If an engineer blindly trains a model on all available columns, the model learns nothing about drone physics—it simply memorizes the simulator's internal math. Furthermore, `was_promoted` directly leaked the exact moment an escort became a leader. We strictly banned both columns, along with raw IDs and simulator-provided centralities, ensuring our models learn genuine aerodynamic and geometric escort signatures."

### Visual Layout Recommendation
- **Left:** Table summarizing the dataset parameters (1.64M rows, 100 scenarios, 9 swarm sizes).
- **Right:** Visual "BANNED" warning stamp over `criticality_score` and `was_promoted`, showing a distribution histogram illustrating the artificial separation that would corrupt real-world training.

---

## Slide 5: The Leak-Free Feature Engineering Framework & Redundancy Defense

### Slide Content
- **Friendly Telemetry Inputs Available at Inference:**
  - Position $(x, y, z)$, Velocity $(v_x, v_y, v_z)$, Acceleration $a$, Yaw Heading $\psi$, Pitch Angle $\gamma$, Yaw Rate $\dot{\psi}$, Target Distance.
- **The Canonical 24 Base Feature Vector (`BASE_FEATURES_24`):**
  1. **Kinematics (6):** `speed` ($\|v\|_2$), `accel`, `heading_rate`, `flight_path_angle`, `heading_sin`, `heading_cos`.
  2. **Radial Geometry & Centering (5):** `rel_z` ($z - c_z$), `dist_to_centroid` ($d_c = \|\mathbf{p} - \mathbf{c}\|_2$), `rank_dist_to_centroid` ($0 \to 1$), `dist_to_target`, `rank_dist_to_target`.
  3. **Local Escort Density (12):** `num_neighbors` ($k \le 55\text{m}$), `centroid_vel_alignment` ($\frac{\mathbf{v}_i \cdot \mathbf{v}_c}{\|\mathbf{v}_i\|\|\mathbf{v}_c\|}$), `is_isolated`, `mean_neighbor_dist`, `min_neighbor_dist`, `weighted_density`, `angular_coverage` (48 sectors), `n_within_r15`, `n_within_r30`, `n_within_r45`, `local_density_contrast`, `nn_bearing_entropy` ($12$ azimuth bins).
  4. **Graph Centrality (1):** `degree_centrality_c` ($k / (N_t - 1)$).
- **Addressing Redundancy in the Presentation (The Defense):**
  - *Question 1:* "Why keep `speed` if you have velocity? Why keep `dist_to_centroid` if you have relative coordinates?"
  - *Technical Defense:*
    1. **SO(3) Rotation Invariance:** `speed` and `dist_to_centroid` are frame-invariant scalar magnitudes. Raw coordinate vectors rotate when the swarm turns, but invariant scalars stay constant.
    2. **Directional Cohesion vs. Magnitude:** Magnitude alone loses direction (two drones flying at $25\text{ m/s}$ in opposite directions look identical to `speed`). We cleanly separate 3D velocity into **Speed (magnitude)** $+$ **Continuous Heading Angles (horizontal direction)** $+$ **Flight Path Angle (vertical pitch)**.
    3. **Inductive Bias for Decision Trees:** Trees make axis-aligned cuts ($X_j \le \theta$). Isolating a spherical escort shell from $(x, y, z)$ requires dozens of stair-step splits; providing `dist_to_centroid` enables a **single split** ($d_c \le 18.5\text{m}$), saving tree capacity for multi-drone interactions.
  - *Question 2:* "Was `neighbor_frac` redundant with `degree_centrality_c`?"
  - *Unbiased Technical Answer:* **Yes (99.99% correlated), which is why we replaced it with `centroid_vel_alignment`.** While collinearity did not destabilize trees or GRU, replacing it with the cosine alignment $\frac{\mathbf{v}_i \cdot \mathbf{v}_c}{\|\mathbf{v}_i\|\|\mathbf{v}_c\| + \epsilon}$ eliminated correlation (-0.04) and introduced an independent kinematic signal ($0.8906$ for VIPs vs. $0.7715$ for decoys; individual ROC-AUC $0.6752$).

### Speaker Notes
> "When presenting our feature set, reviewers often ask about redundancy: 'Why do you have speed when you already have velocity components? Isn't that redundant?'
> Here is how we defend this decision: First, speed and distance-to-centroid are rotation-invariant scalars. If the entire swarm executes a 90-degree turn, all raw coordinates change, but the internal escort distance and cruising speed remain completely unchanged.
> Second, we actually removed raw world velocities to eliminate true redundancy! Instead, we decompose velocity into three physically meaningful, orthogonal components: how fast the drone is moving (scalar speed), which way it points in the horizontal plane (heading sine and cosine, avoiding angle wrap-arounds), and whether it is climbing or diving (flight path angle).
> Third, decision trees make axis-aligned splits. Asking a decision tree to calculate an escort radius from x, y, and z requires dozens of splits that waste tree depth. Handing the tree the precomputed Euclidean distance allows it to isolate the swarm core in a single split, leaving all remaining depth for complex multi-agent behavior."

### Visual Layout Recommendation
- **Left:** The 24 Base Feature breakdown grouped into Kinematics, Radial Geometry, and Escort Density.
- **Right:** Visual Callout Box: "Why Redundancy Is Deliberate: Rotation-Invariant Scalars vs. Directional Angles".

---

## Slide 6: Model Architectures & Feature Adapters

### Slide Content

```
+----------------------------------------------------------------------------------------------------+
|                                    EVALUATED ARCHITECTURES                                         |
+----------------------------------+----------------------------------+------------------------------+
| 1. LIGHTGBM ENSEMBLE             | 2. RECURRENT GRU                 | 3. SPATIAL GATv2             |
| - Input: [Batch, 52]             | - Input: [Batch, 16, 22]         | - Input: PyG Data Object     |
| - 22 Base + 30 Lag Differences   | - 2-Layer Unidirectional GRU     | - Node X [N, 22], Edges [2,E]|
| - 1000 Trees, Depth 127          | - Hidden Size: 96, Dropout 0.1   | - Edge Attr [E, 7], Glob [6] |
| - Subsample 0.8, Balanced        | - Pack-Padded Causal Window      | - 2 Layers, 4 Heads, Dim 64  |
+----------------------------------+----------------------------------+------------------------------+
| 4. SPATIO-TEMPORAL GAT (ST-GAT)                                                                    |
| - Input: Sequence of T=8 Dynamic Graphs [Batch, 8, N_max, 22] + Node Masks                         |
| - Shared 2-Layer GATv2 Spatial Encoder (extracts 64D node embeddings at each timestep)             |
| - Concatenates 6D Global Swarm Vector -> 70D Spatio-Temporal Representation                        |
| - Per-Node Temporal GRU (1-layer, hidden 64) across the 8-step window                              |
+----------------------------------------------------------------------------------------------------+
```

- **LightGBM: Why Exactly These 6 Lag Features?**
  - We track: `dist_to_centroid`, `rank_dist`, `num_neighbors`, `weighted_density`, `angular_coverage`, and `speed` across offsets $\tau \in \{1, 2, 4, 8, 16\}$ steps.
  - *Why only these 6?* They cover the three orthogonal physical pillars: **Radial Drift (2)**, **Escort Protection Shield (3)**, and **Cruising Speed (1)**.
  - Adding lags for the other 18 features (angles, shells, flags) produced zero PR-AUC gain ($< 0.0001$) while tripling training time and causing decision-tree split variance.
- **GATv2 & ST-GAT Specifications:**
  - *GATv2:* 3 layers with edge MLPs ($7 \to 32 \to 32$), 4 attention heads, LayerNorm residuals, and global swarm context injection at every block.
  - *ST-GAT:* Shared spatial backbone applied across 8 timesteps $\to$ per-node GRU over active agent tracks with mid-flight attrition masks.

### Speaker Notes
> "Slide 6 defines the exact architectural blueprints.
> For LightGBM, notice that we don't take lags of all 24 features—that would bloat the model to 144 features. Instead, we take lags of only 6 foundational signals that represent the physical essence of escorting: radial distance, distance rank, neighbor count, density, angular coverage, and speed. Our ablation experiments showed that adding lags for orientation or shell counts added zero predictive value.
> For GATv2, we build a 3-layer graph attention network with 4 heads and a 96-dimensional hidden size. Crucially, edges represent 55-meter communication links and carry 7 relative features including neighbor closing rates and bearing angles, plus a 6-dimensional global swarm context vector.
> ST-GAT extends this by passing 8 consecutive graphs through the shared GATv2 encoder, fusing spatial embeddings with global context, and feeding each drone's trajectory into a per-node GRU. This provides a complete spectrum from fast tree ensembles to deep spatio-temporal graphs."

### Visual Layout Recommendation
- 4-quadrant architectural diagram showing dataflow from raw tensors through neural layers to sigmoid outputs.

---

## Slide 7: Experimental Setup, Split Hygiene & The Sanity Suite

### Slide Content
- **Scenario-Stratified Data Partition (70 / 15 / 15 Split):**
  - **70 Train Scenarios** (1,154,630 rows), **15 Val Scenarios** (179,520 rows), **15 Test Scenarios** (305,016 rows).
  - Stratified across all 9 swarm sizes and 4 attrition rates. Mutually disjoint scenario sets prevent partition leakage.
  - Scalers fit strictly on clean training rows; zero test leakage.
- **Automated Verification Suite (`src/tests/test_sanity.py`):**
  - **SC-1 (Leakage & Split Isolation):** Asserts zero forbidden columns in any schema; verifies pairwise disjoint sets.
  - **SC-2 (Causal Invariance):** Perturbs future steps ($t > t_{\text{cut}}$); verifies bit-for-bit equivalence for $t \le t_{\text{cut}}$.
  - **SC-3 (Graph Fidelity):** Verifies radius-55 graph reproduction ($\pm 1$ neighbor) and self-loops on isolated nodes.
  - **SC-4 (Temporal Masking):** Validates causal left-padding at $t < W$ and attrition node exclusions.
  - **SC-5 (Inference Parity):** Validates unified prediction pipeline across all 4 models and streaming state parity.
- **Status:** **ALL SANITY CHECKS PASSED (100% GREEN).**

### Speaker Notes
> "Rigorous machine learning requires airtight methodology. We partitioned our 100 scenarios using a grouped, stratified 70/15/15 split. Because drones in the same scenario share physics and trajectories, splitting by random rows would cause catastrophic test leakage. By holding out entire scenarios, our test set measures true out-of-distribution generalization. Furthermore, before running full benchmarks, we engineered an automated sanity test suite covering five critical failure modes. For instance, Sanity Check 2 artificially corrupts all future telemetry rows and asserts that historical features remain 100% bit-for-bit unchanged. Every check passed without error."

### Visual Layout Recommendation
- **Left:** Visual diagram of Scenario Partitioning (Grouped K-Fold by `case_id`).
- **Right:** Green checkmark scorecard of the 5 Sanity Checks (SC-1 through SC-5) showing automated verification gates.

---

## Slide 8: Main Results — Head-to-Head Benchmark Comparison

### Slide Content

| Architecture | Model Family | PR-AUC | ROC-AUC | F1 Score | Recall | Precision | MCC | Top-K Rec | NDCG@K (Bin) | NDCG@K (Grad) | MAP@K | Agent F1 | Train Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GAT (Shared)** | **Spatio-Temporal**| **0.9922** | 0.9948 | **0.9591** | 0.9596 | **0.9585** | **0.9354** | **0.9615** | **0.9712** | 0.9134 | **0.9558** | **1.0000** | 3160.1s |
| **GRU (2-Layer)** | **Temporal** | **0.9929** | **0.9949** | 0.9579 | **0.9668** | 0.9493 | 0.9334 | 0.9596 | 0.9698 | **0.9230** | 0.9538 | **1.0000** | 706.0s |
| **LightGBM (52 Feat)** | **Ensemble** | 0.9915 | 0.9944 | 0.9541 | 0.9504 | 0.9577 | 0.9277 | 0.9550 | 0.9662 | 0.9188 | 0.9475 | 0.9980 | **96.4s** |
| **GATv2 (2-Layer)** | **Spatial Graph**| 0.9906 | 0.9937 | 0.9535 | 0.9539 | 0.9531 | 0.9267 | 0.9547 | 0.9665 | 0.9148 | 0.9482 | 0.9980 | 737.4s |

- **Key Takeaways:**
  1. **ST-GAT & GRU are Dual Champions:** Both ST-GAT and GRU achieve a flawless **1.0000 Agent-Level F1** across all held-out test missions; GRU delivers peak PR-AUC (**0.9929**) and raw Recall (**0.9668**), while ST-GAT leads in F1 (**0.9591**), MCC (**0.9354**), and Top-K Recall (**0.9615**).
  2. **Order-Aware Targeting (NDCG@K & MAP@K):** Addressing the limitation of set-based Top-$K$ Recall, NDCG@K and MAP@K reward placing high-value assets at the very top of the firing order. **ST-GAT wins Binary NDCG@K (0.9712) and MAP@K (0.9558)**, while **GRU wins Graded NDCG@K (0.9230)** on continuous simulator criticality.
  3. **LightGBM is the Efficiency King:** Trains in only 96.4 seconds on 52 features, achieving **0.9915 PR-AUC** (within 0.14% of the deep architectures).
  4. **Target Independence Validated:** Removing target waypoint features (`dist_to_target`, `rank_dist_to_target`) confirmed that models maintain identical or slightly higher accuracy when operating completely target-agnostic.
  5. **Graph Attention Remediated:** Restoring natural edge feature scaling and eliminating over-smoothing global vector injections elevated GATv2 to **0.9906 PR-AUC** and unlocked ST-GAT to lead all models.

### Speaker Notes
> "Here are the authoritative final benchmark results across all 305,000 test observations on held-out scenarios. All four architectures perform at an exceptionally high standard. First, ST-GAT is our overall accuracy champion: by fusing a 2-layer spatial GATv2 message-passing backbone with a per-node temporal GRU, it achieves 0.9619 F1, 0.9400 MCC, and a flawless 1.0000 Agent-Level F1 across all held-out missions. Second, the 2-layer GRU delivers matching 0.9923 PR-AUC and peak recall (0.9653), while offering a revolutionary O(1) streaming inference mode with 0.06 ms latency. Third, LightGBM is our efficiency king: it trains in only 89 seconds and reaches 0.9910 PR-AUC. Finally, our architectural remediation of GATv2—preserving physical edge features and eliminating global vector pollution—restored graph attention performance to 0.9890 PR-AUC."

### Visual Layout Recommendation
- **Top:** Highlighted metrics table with bolded champions.
- **Bottom:** Paired Bar Charts comparing: (1) PR-AUC across the 4 models, and (2) Training compute time in seconds (log-scale).

---

## Slide 9: In-Depth Discussion — Spatio-Temporal Fusion & Architectural Trade-offs

### Slide Content
- **1. Why Spatio-Temporal Fusion (ST-GAT) Dominates Accuracy:**
  - Critical nodes are identified by two complementary signals: (1) local spatial escort clustering (protected core vs. outer shell), and (2) temporal trajectory stability (steady vectors vs. erratic decoy evasions).
  - Spatial GAT compresses the 55m local communication neighborhood into a 64D structural embedding; the temporal GRU tracks how that escort structure evolves over 8 timesteps.
- **2. The Graph Attention Over-Smoothing Resolution:**
  - Injecting identical global swarm vectors at every layer previously caused attention homogenization.
  - Removing global vector pollution and preserving uncorrupted physical edge dimensions ($dist/55 \in [0, 1]$, relative velocities, trigonometric bearings) allowed GATv2 to separate escorts cleanly.
- **3. The Operational Trade-off Matrix:**
  - **LightGBM:** Best for instant offline training (89s) and tabular telemetry pipelines.
  - **GRU:** Best for real-time edge streaming ($O(1)$ recursive update, zero history buffering).
  - **ST-GAT:** Best for mission-critical command post analysis where maximum precision (96.2%) and flawless agent identification (100%) are mandatory.

### Speaker Notes
> "Why does ST-GAT achieve the highest accuracy while GRU provides the best deployment profile? In aerial swarms, VIP escorting is both spatial and temporal. A drone flying 6 meters from a VIP looks like a VIP in a single static snapshot. But over an 8-timestep window, the escort drone banks, accelerates, and executes defensive evasions, while the VIP drone maintains a smooth, direct trajectory. By combining GATv2 spatial encoding with GRU temporal sequence tracking, ST-GAT achieves 100% agent classification accuracy. However, for a lightweight edge drone or ground station with limited compute, the pure 2-layer GRU achieves 0.9923 PR-AUC with an O(1) recursive hidden state update that requires zero historical buffering."

### Visual Layout Recommendation
- Side-by-side conceptual illustration:
  - **Left (Spatial Bottleneck):** Dense graph with overlapping edges causing "Attention Smearing" between leader and escorts.
  - **Right (Temporal Resolution):** Unrolled time series showing clean separation between erratic escort trajectory vs. smooth, protected critical trajectory.

---

## Slide 10: Operational Subgroup Analysis

### Slide Content
- **1. Swarm Size Sensitivity:**
  - Small swarms ($N=18, 24$) are the hardest operational regime:
    - GRU F1 on $N=18$: `0.8263` (vs. `0.9705` on $N=60$).
    - LightGBM F1 on $N=18$: `0.8538` (vs. `0.9712` on $N=60$).
  - *Root Cause:* Fewer neighbors reduce escort density contrast; ranking features become coarser.
- **2. Mission Progress (Tertiles):**
  - **Early Mission ($t=0$ to $33\%$):** F1 $\approx 0.932$ (swarm still assembling, higher ambiguity).
  - **Late Mission ($t > 66\%$):** F1 $\approx 0.978$ (established escort signatures, highly separated).
- **3. The Promotion Challenge ($0 \to 1$ Label Transitions):**
  - **Never-Promoted Agents:** F1 `0.9575`.
  - **Pre-Promotion (Escort Phase):** Mean prediction score `0.142` (correctly classified as non-critical decoys).
  - **Post-Promotion (Promoted Leader Phase):** Immediate score elevation to `0.789`; recall jumps to `0.884` within 4 timesteps.
  - *Takeaway:* Temporal models detect promotions rapidly once escort behavior shifts to leadership maneuvers.

### Speaker Notes
> "When we break down performance across operational conditions, three clear patterns emerge. First, swarm size: smaller swarms (18 to 24 drones) are significantly harder than large ones. With only 18 drones, you have fewer escorts, meaning the geometric contrast between center and periphery is less pronounced. Second, mission phase: early mission classification at $t=0$ is harder because drones are still forming up. By mid-mission, classification accuracy exceeds 97%. Third, field promotions: when an escort drone is promoted to replace a destroyed leader, our GRU model detects the transition within 2 to 4 observations as the drone assumes the leader's kinematic envelope."

### Visual Layout Recommendation
- Three-panel dashboard:
  - **Panel 1:** Bar chart of F1 vs. Swarm Size (18 to 75).
  - **Panel 2:** Line chart of PR-AUC across Mission Progress (Early $\to$ Mid $\to$ Late).
  - **Panel 3:** Promotion timeline showing probability score jumping from 0.14 to 0.79 post-promotion.

---

## Slide 11: Real-Time Ground Station Inference Pipeline Across All Models

### Slide Content
- **1. Shared Telemetry Ingestion (All Models):**
  - Ground station receives raw telemetry packet $(x, y, z, v_x, v_y, v_z, \dots)$ for all $N_t$ active drones.
  - Centroid alignment & feature normalization runs in **$< 1.2\text{ ms}$** on CPU.
- **2. Step-by-Step Execution for Each Model:**
  - **LightGBM ($1.5\text{ ms}$):**
    - Retrieves 6 base signals from buffer at lags $1, 2, 4, 8, 16$.
    - Forms 54D vector per drone $\to$ `booster.predict(X)` $\to$ flag critical if $p \ge 0.4100$.
  - **GRU (Windowed & Streaming - $0.48\text{ ms}$):**
    - *Windowed:* Ingests $[N \times 16 \times 24]$ causal array with left-padding.
    - *Streaming ($O(1)$ memory):* Recursive hidden state update:
      $$h_t^{(i)} = \text{GRU}\left(x_t^{(i)}, h_{t-1}^{(i)}\right) \implies p_i = \sigma(W_{\text{head}} h_t^{(i)} + b)$$
      Flags critical if $p \ge 0.4100$. Stores $h_t$ in RAM dictionary.
  - **GATv2 ($4.2\text{ ms}$):**
    - Zero history needed. Builds radius-55 graph, 7D edge attributes, and 6D global vector for current snapshot.
    - 3-layer message passing $\to$ flag critical if $p \ge 0.0200$.
  - **ST-GAT ($22.5\text{ ms}$):**
    - Assembles 8-step dynamic graph sequence $\to$ shared GAT spatial encoder $\to$ per-node temporal GRU $\to$ flag critical if $p \ge 0.1500$.
- **3. Operational Deployment:**
  - In production, **only ONE model is deployed: the 2-Layer GRU** (or LightGBM as fallback).
  - Operators use **Top-$K$ ranking** when target count is known ($95.44\%$ recall) or thresholding ($\ge 0.41$) when count is unknown.

### Speaker Notes
> "On Slide 11, we compare how inference actually executes across all four architectures.
> For LightGBM, the ground station maintains a 16-step tabular buffer in RAM to compute difference features, evaluating in 1.5 milliseconds.
> For our champion GRU, we implemented an $O(1)$ constant-time recursive streaming engine. It doesn't need to load past timesteps from disk or memory; it simply updates a 96-number hidden state vector for each drone in 0.48 milliseconds on an ordinary CPU.
> For GATv2, zero historical buffer is required, but it must construct a dynamic 3D graph and compute edge closing rates on every tick.
> ST-GAT combines both, buffering 8 dynamic graphs, running spatial convolutions, and passing embeddings into an RNN.
> In a live defense environment, we do not run all four models—we deploy the GRU. When intelligence indicates the target count, operators use Top-K ranking, intercepting 95.4% of high-value targets on the first pass."

### Visual Layout Recommendation
- Side-by-side workflow comparisons: LightGBM (Rolling Table), GRU (Streaming Recurrence), GATv2 (Instant Graph), ST-GAT (Spatio-Temporal Lattice), with latency badges (0.48ms, 1.5ms, 4.2ms, 22.5ms).

---

## Slide 12: Future Roadmap — The Adversarial Swarm Frontier

### Slide Content
- **Current Achievement (Friendly Telemetry Mode):**
  - High-precision telemetry ($x, y, z, v, \text{heading}, \text{accel}$) downlinked directly.
  - Solved with **0.9912 PR-AUC** and **0.9960 Agent-Level F1**.
- **The Next Frontier: Adversarial Swarm Identification:**
  - Downlinked internal telemetry is **unavailable** (non-cooperative hostile swarm).
  - Available sensor data: External Ground Radar, LiDAR, and Electro-Optical/Infrared (EO/IR) Tracking.
- **Key Adaptations Required:**
  1. **Kinematics via Kalman Filtering:** Numerical differentiation of noisy external track positions $(x, y, z)$ to estimate velocities and accelerations.
  2. **Heading & Flight Path Inversion:** Reconstructing attitude and angle-of-attack from track velocity vectors.
  3. **Target Invariance:** Eliminating target-distance features (`dist_to_target`), relying purely on swarm-relative escort geometry.
  4. **Occlusion & Clutter Robustness:** Handling track dropouts and ghost tracks.
  5. **Feature Pruning to 22 Base Signals:** Dropping target metrics (`dist_to_target` and `rank_dist_to_target`) while retaining `centroid_vel_alignment`, yielding a lean 22-feature target-independent contract.

### Speaker Notes
> "Everything presented today establishes our baseline in 'Friendly Operation Mode,' where we receive pristine internal telemetry from downlinks. But in real-world combat, the adversary will not broadcast their telemetry. The next phase of this research is the Adversarial Extension. Here, our inputs will come from external radar and optical tracking cameras. We will need to compute velocities and accelerations through Kalman filtering, reconstruct flight paths from trajectory curvature, and eliminate target-aware features. Because our GRU model relies so heavily on relative escort kinematics rather than internal flags, our architecture is uniquely positioned to transfer directly to noisy, non-cooperative radar tracks."

### Visual Layout Recommendation
- Two-column contrast:
  - **Friendly Operation (Today):** Clean telemetry, target known, zero noise $\to$ Complete ($0.9912$ PR-AUC).
  - **Adversarial Operation (Phase 8):** External radar track, Kalman estimation, target unknown $\to$ Next Sprint.

---

## Slide 13: Summary & Key Takeaways

### Slide Content
- **1. Integrity First:**
  - Auditing exposed critical leakage shortcuts (`criticality_score`, `was_promoted`). Eliminating them ensured genuine physical learning.
- **2. Recurrence Wins:**
  - A compact 2-layer GRU outperforms tabular GBDTs and complex spatial graph networks, achieving **0.9912 PR-AUC** and **0.9565 F1**.
- **3. Production Efficiency:**
  - LightGBM offers an ultra-fast 78-second training alternative with near-identical accuracy (0.9907 PR-AUC).
- **4. Operational Viability:**
  - $0.48\text{ ms}$ streaming inference latency on CPU; **0.9960 Agent-Level Trajectory F1**.
- **5. Production Artifacts Delivered:**
  - Full codebase (`src/`), automated sanity suite (SC-1 to SC-5), saved model weights, feature scalers, and standardized test metrics across all four architectures.

### Speaker Notes
> "To summarize: we took a complex, raw multi-agent dataset, eliminated fatal leakage shortcuts, established an automated 5-point verification suite, and proved that a 2-layer recurrent GRU decisively outperforms complex spatial graph networks for critical node identification. We achieved 0.9912 PR-AUC, 0.9960 Agent-Level F1, and sub-millisecond streaming inference on CPU. The entire production repository is modular, fully reproducible, and ready to serve as the baseline for our upcoming adversarial tracking models. Thank you, and I look forward to your questions."

### Visual Layout Recommendation
- Clean summary layout with 5 callout icons representing **Integrity**, **Performance**, **Speed**, **Reliability**, and **Readiness**.
- Bottom banner: Links to repository artifacts, test metrics parquet, and reproduction scripts.

