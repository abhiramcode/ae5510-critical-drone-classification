# The Complete Guide to Swarm Critical-Node Evaluation Metrics
## Mathematical Definitions, Step-by-Step Calculations, and Cross-Model Performance Analysis

**Target Document:** `./report/Model_Evaluation_Metrics_Guide.md`  
**Dataset Baseline:** 100 Aerial Swarm Scenarios (1,639,166 rows, 305,016 held-out test rows)  
**Evaluated Models:** LightGBM (52 Tabular Lags), 2-Layer GRU, Spatial GATv2, Spatio-Temporal ST-GAT  
**Artifact References:** `artifacts/comparison_summary.csv`, `artifacts/runs/*/test_metrics.json`  

---

## 1. Executive Overview & The Metric Landscape

In critical asset identification for autonomous aerial swarms, evaluation is not a simple matter of looking at standard "Accuracy." 

### 1.1 Why Standard Accuracy is Deceptive
In our dataset of 100 simulation scenarios, the overall base rate of critical nodes is approximately **$38.9\%$** (ranging from $27.8\%$ to $68.3\%$ across different swarm sizes and attrition scenarios). In many operational military contexts, critical nodes represent a small minority (e.g., 2 to 5 command drones in a 60-drone swarm, or $\approx 5\% - 10\%$).

If a naive model simply classified **every single drone as a decoy ($0$)**, it would achieve:
$$\text{Accuracy} = \frac{\text{Correct Decoys}}{\text{Total Drones}} = \frac{61.1\%}{100\%} = 61.1\%$$
Yet such a model is **100% useless operationally** because every command drone slips past undetected!

To capture the true operational effectiveness of our four models without bias, our evaluation framework measures **12 complementary metrics** across three fundamental dimensions:
1. **Threshold-Dependent Decision Metrics:** Precision, Recall, False Negative Rate, F1-Score, Matthews Correlation Coefficient (MCC), and Balanced Accuracy.
2. **Threshold-Independent Ranking Metrics:** Precision-Recall AUC (PR-AUC), Receiver Operating Characteristic AUC (ROC-AUC), Recall at $\ge 90\%$ Precision, and Top-$K$ Recall.
3. **Operational Mission Metrics:** Agent-Level Trajectory F1, Mission Timeline Progression ($t=0$ vs. Late Mission), and Dynamic Promotion Tracking.

---

### 1.2 Master Benchmark Comparison Table
The authoritative metrics below were evaluated on the **305,016 held-out test observations** across 15 complete, unseen simulation scenarios (`c001`, `c010`, `c012`, `c022`, `c032`, `c041`, `c055`, `c058`, `c062`, `c068`, `c072`, `c076`, `c081`, `c085`, `c093`):

| Metric | LightGBM (GBDT) | GRU (2-Layer) | GATv2 (Spatial GNN) | ST-GAT (Spatio-Temporal) | Metric Champion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Operating Threshold ($\theta^*$)** | $0.4800$ | $0.5900$ | $0.6800$ | $0.4800$ | — |
| **PR-AUC (Primary Metric)** | $0.9915$ | **$0.9929$** | $0.9906$ | $0.9922$ | **GRU ($0.9929$)** |
| **ROC-AUC** | $0.9944$ | **$0.9949$** | $0.9937$ | $0.9948$ | **GRU ($0.9949$)** |
| **F1-Score** | $0.9541$ | $0.9579$ | $0.9535$ | **$0.9591$** | **ST-GAT ($0.9591$)** |
| **Recall (Sensitivity)** | $0.9504$ | **$0.9668$** | $0.9539$ | $0.9596$ | **GRU ($0.9668$)** |
| **Precision** | $0.9577$ | $0.9493$ | $0.9531$ | **$0.9585$** | **ST-GAT ($0.9585$)** |
| **False Negative Rate (Miss Rate)** | $4.96\%$ | **$3.32\%$** | $4.61\%$ | $4.04\%$ | **GRU ($3.32\%$)** |
| **Matthews Corr. Coeff. (MCC)** | $0.9277$ | $0.9334$ | $0.9267$ | **$0.9354$** | **ST-GAT ($0.9354$)** |
| **Balanced Accuracy** | $0.9631$ | **$0.9685$** | $0.9634$ | $0.9678$ | **GRU ($0.9685$)** |
| **Recall @ 90% Precision** | $0.9770$ | **$0.9823$** | $0.9758$ | $0.9802$ | **GRU ($0.9823$)** |
| **Top-$K$ Recall** | $0.9550$ | $0.9596$ | $0.9547$ | **$0.9615$** | **ST-GAT ($0.9615$)** |
| **Binary NDCG@K (Order-Aware)** | $0.9662$ | $0.9698$ | $0.9665$ | **$0.9712$** | **ST-GAT ($0.9712$)** |
| **Graded NDCG@K (Criticality-Aware)** | $0.9188$ | **$0.9230$** | $0.9148$ | $0.9134$ | **GRU ($0.9230$)** |
| **MAP@K (Mean Average Precision)** | $0.9475$ | $0.9538$ | $0.9482$ | **$0.9558$** | **ST-GAT ($0.9558$)** |
| **Mean Reciprocal Rank (MRR)** | $0.9950$ | **$0.9957$** | $0.9956$ | $0.9947$ | **GRU ($0.9957$)** |
| **Spearman Rank Corr. ($\rho$)** | $0.7600$ | $0.7558$ | **$0.7766$** | $0.7673$ | **GATv2 ($0.7766$)** |
| **Agent-Level Trajectory F1** | $0.9980$ | **$1.0000$** | $0.9980$ | **$1.0000$** | **GRU & ST-GAT (1.0000)** |
| **Post-Promotion Recall** | $69.43\%$ | **$82.06\%$** | $62.38\%$ | $72.37\%$ | **GRU ($82.06\%$)** |
| **$t=0$ Snapshot PR-AUC** | **$0.8966$** | $0.5836$ | $0.6767$ | $0.7179$ | **LightGBM ($0.8966$)** |
| **Training Time (CPU)** | **$96.4\text{ s}$** | $706.0\text{ s}$ | $737.4\text{ s}$ | $3160.1\text{ s}$ | **LightGBM ($96.4\text{ s}$)** |

---

## 2. Foundations: From Probabilities to Decisions

Every model outputs a continuous raw probability score for each alive drone $i$ at timestamp $t$:
$$p_i(t) = P(\text{agent } i \text{ is critical} \mid \text{telemetry up to } t) \in [0.0, 1.0]$$

To convert continuous probabilities into discrete military or tactical decisions (**Critical: $1$** vs. **Decoy: $0$**), the ground station applies an operating threshold $\theta^*$:
$$\hat{y}_i = \begin{cases} 1 & \text{if } p_i \ge \theta^* \\ 0 & \text{if } p_i < \theta^* \end{cases}$$

### 2.1 How the Operating Threshold $\theta^*$ is Chosen
A naive system hardcodes $\theta = 0.50$. However, in imbalanced datasets or calibrated neural networks, $0.50$ is rarely optimal.

Our framework uses `pick_optimal_threshold()` in `src/evaluation/metrics.py`. It runs a fine-grained grid search of 97 steps from $\theta = 0.02$ to $\theta = 0.98$ **strictly on validation set predictions**, selecting the threshold that maximizes the validation F1-score:
$$\theta^* = \arg\max_{\theta \in [0.02, 0.98]} \text{F1}(y_{\text{val}}, p_{\text{val}} \ge \theta)$$

Once tuned on validation scenarios, $\theta^*$ is frozen and applied to the held-out test scenarios without modification. Notice the optimal thresholds discovered for our models:
- **LightGBM:** $\theta^* = 0.4800$ (close to natural $0.50$, indicating well-balanced tree log-odds).
- **GRU:** $\theta^* = 0.5900$ (higher threshold because GRU probabilities are sharply polarized toward $1.0$).
- **GATv2:** $\theta^* = 0.6800$ (higher threshold to suppress false positive edge noise in dense graphs).
- **ST-GAT:** $\theta^* = 0.4800$ (balanced calibration combining spatial and temporal features).

---

### 2.2 The Four Quadrants of the Confusion Matrix
Once predictions are thresholded, comparing predicted labels $\hat{y}$ against true ground-truth labels $y$ partitions all observations into four fundamental quadrants:

```
                          TRUE SITUATION (Ground Truth)
                       Critical (1)          Decoy (0)
                   +--------------------+--------------------+
PREDICTED   Crit 1 | True Positive (TP) | False Positive(FP) |
DECISION           | "Leader Shot Down" | "Decoy Intercepted"|
                   +--------------------+--------------------+
            Decoy 0| False Negative(FN) | True Negative (TN) |
                   | "LEADER MISSED!"   | "Decoy Ignored"    |
                   +--------------------+--------------------+
```

- **True Positive (TP):** Drone is critical ($1$) and model predicted critical ($1$). *Tactical Outcome: High-value asset successfully targeted.*
- **False Positive (FP):** Drone is a decoy ($0$) but model predicted critical ($1$). *Tactical Outcome: Interceptor ammunition wasted on a sacrificial decoy.*
- **False Negative (FN):** Drone is critical ($1$) but model predicted decoy ($0$). *Tactical Outcome: Disastrous. Enemy command drone escapes interception.*
- **True Negative (TN):** Drone is a decoy ($0$) and model predicted decoy ($0$). *Tactical Outcome: Correctly conserving ammunition.*

---

### 2.3 Toy Example: 10-Drone Swarm Snapshot
To make the calculations crystal clear, let us follow a simplified 10-drone swarm throughout this guide:

| Drone ID | Ground Truth ($y$) | Model Predicted Prob ($p$) | Decision ($\theta^* = 0.50$) | Quadrant |
| :---: | :---: | :---: | :---: | :---: |
| **Drone 0** | **$1$ (Critical)** | `0.95` | **$1$** | **TP** |
| **Drone 1** | **$1$ (Critical)** | `0.88` | **$1$** | **TP** |
| **Drone 2** | **$1$ (Critical)** | `0.82` | **$1$** | **TP** |
| **Drone 3** | **$1$ (Critical)** | `0.42` | $0$ | **FN** |
| **Drone 4** | $0$ (Decoy) | `0.75` | **$1$** | **FP** |
| **Drone 5** | $0$ (Decoy) | `0.20` | $0$ | **TN** |
| **Drone 6** | $0$ (Decoy) | `0.15` | $0$ | **TN** |
| **Drone 7** | $0$ (Decoy) | `0.08` | $0$ | **TN** |
| **Drone 8** | $0$ (Decoy) | `0.05` | $0$ | **TN** |
| **Drone 9** | $0$ (Decoy) | `0.02` | $0$ | **TN** |

**Summary Counts for Toy Swarm:**
- Total drones: $N = 10$
- True Criticals: $P = 4$
- True Decoys: $N_{\text{neg}} = 6$
- $\text{TP} = 3$ (Drones 0, 1, 2)
- $\text{FP} = 1$ (Drone 4)
- $\text{FN} = 1$ (Drone 3)
- $\text{TN} = 5$ (Drones 5, 6, 7, 8, 9)

---

## 3. Threshold-Dependent Classification Metrics

---

### 3.1 Precision (Positive Predictive Value)

#### 1) Definition
Precision answers the operational question:  
> *"Of all the drones my model sounded an alarm on, what percentage actually were critical assets?"*

It measures the **purity and reliability** of the alerts.

#### 2) Mathematical Formula & Calculation
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

In our 10-drone toy example:
$$\text{Precision} = \frac{3}{3 + 1} = \frac{3}{4} = 0.7500 \quad (75.0\%)$$

If an interceptor battery fired at the 4 flagged drones, 3 hits would kill command drones, while 1 missile would be wasted on a decoy.

#### 3) Actual Model Results & Comparison
| Model | Precision | Rank |
| :--- | :---: | :---: |
| **ST-GAT** | **$0.9585$** | **#1** |
| **LightGBM** | $0.9577$ | #2 |
| **GATv2** | $0.9531$ | #3 |
| **GRU** | $0.9493$ | #4 |

#### Architectural Explanation
- **Why ST-GAT and LightGBM win on Precision:**  
  ST-GAT combines spatial graph structure with temporal history. It demands that a drone not only look like a leader spatially (surrounded by tight escorts) but also behave like a leader over time (maintaining steady non-evasive heading). This double requirement drastically cuts down False Positives (only $4.15\%$ false alarm rate). LightGBM achieves high precision through strict axis-aligned cuts on radial rank (`rank_dist_to_centroid <= 0.22`), effectively ignoring fringe drones.
- **Why GRU has slightly lower Precision ($0.9493$):**  
  GRU is tuned for maximum sensitivity. Because it focuses heavily on trajectory smoothness, occasionally an escort drone cruising smoothly on a straight path is flagged as critical, generating a few extra False Positives.

---

### 3.2 Recall (Sensitivity / True Positive Rate)

#### 1) Definition
Recall answers the life-or-death tactical question:  
> *"Of all the genuine critical command drones present in the sky, what percentage did my model successfully detect?"*

It measures the **completeness** of threat detection.

#### 2) Mathematical Formula & Calculation
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

In our 10-drone toy example:
$$\text{Recall} = \frac{3}{3 + 1} = \frac{3}{4} = 0.7500 \quad (75.0\%)$$

Here, 1 critical drone (Drone 3) slipped through unflagged (False Negative).

#### 3) Actual Model Results & Comparison
| Model | Recall | Rank |
| :--- | :---: | :---: |
| **GRU (2-Layer)** | **$0.9668$** | **#1** |
| **ST-GAT** | $0.9596$ | #2 |
| **GATv2** | $0.9539$ | #3 |
| **LightGBM** | $0.9504$ | #4 |

#### Architectural Explanation
- **Why GRU is the Undisputed Recall Champion ($96.68\%$):**  
  GRU maintains continuous hidden state memory ($\mathbf{h}_t$) over a 16-step window ($1.6$ seconds of flight). Escort decoys frequently execute jerky defensive turns to block incoming fire, causing spikes in `heading_rate` and `accel`. Even if a critical node momentarily moves slightly outward during a collision avoidance maneuver, GRU's recurrent memory "remembers" its prior 15 steps of stable, centered flight and refuses to drop the positive label. This catches edge-case leaders that snapshot models miss.

---

### 3.3 False Negative Rate (FNR / Miss Rate)

#### 1) Definition
The False Negative Rate is the exact complement of Recall:  
> *"What percentage of high-value threats did the defense system fail to catch?"*

In air defense, False Negatives carry the highest operational penalty: a single missed command drone can complete its strike mission.

#### 2) Mathematical Formula & Calculation
$$\text{FNR} = 1.0 - \text{Recall} = \frac{\text{FN}}{\text{TP} + \text{FN}}$$

In our 10-drone toy example:
$$\text{FNR} = 1.0 - 0.7500 = 0.2500 \quad (25.0\%)$$

#### 3) Actual Model Results & Comparison
| Model | False Negative Rate (Miss Rate) | Rank |
| :--- | :---: | :---: |
| **GRU (2-Layer)** | **$3.32\%$** | **#1 (Lowest Risk)** |
| **ST-GAT** | $4.04\%$ | #2 |
| **GATv2** | $4.61\%$ | #3 |
| **LightGBM** | $4.96\%$ | #4 |

GRU misses only **33 out of every 1,000 critical drone encounters**, whereas LightGBM misses 50 out of 1,000. In life-critical automated defense, GRU reduces missed threats by **$33\%$** compared to the tabular baseline.

---

### 3.4 F1-Score (Harmonic Mean of Precision and Recall)

#### 1) Definition
The F1-score is the standard unified metric balancing Precision and Recall into a single number:  
> *"How well does the model maximize both threat detection (Recall) and false alarm suppression (Precision) simultaneously?"*

#### 2) Why Harmonic Mean instead of Simple Average?
The simple arithmetic mean of Precision ($P$) and Recall ($R$) is $\frac{P + R}{2}$. But consider a useless model that flags every drone as critical:
- $\text{Recall} = 1.0$ (catches all leaders).
- $\text{Precision} = 0.389$ (38.9% base rate).
- Arithmetic Mean $= \frac{1.0 + 0.389}{2} = 0.6945$ (looks deceptively good!).

The **harmonic mean** strongly penalizes extreme imbalances:
$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

For the useless model above, $\text{F1} = 2 \cdot \frac{1.0 \cdot 0.389}{1.0 + 0.389} = 0.560$, rightly punishing the terrible precision.

In our 10-drone toy example:
$$\text{F1} = 2 \cdot \frac{0.75 \cdot 0.75}{0.75 + 0.75} = 0.7500$$
Using raw counts:
$$\text{F1} = \frac{2 \times 3}{2 \times 3 + 1 + 1} = \frac{6}{8} = 0.7500$$

#### 3) Actual Model Results & Comparison
| Model | F1-Score | Rank |
| :--- | :---: | :---: |
| **ST-GAT** | **$0.9591$** | **#1** |
| **GRU (2-Layer)** | $0.9579$ | #2 |
| **LightGBM** | $0.9541$ | #3 |
| **GATv2** | $0.9535$ | #4 |

ST-GAT wins the overall F1 crown because it balances near-peak precision ($0.9585$) with high recall ($0.9596$).

---

### 3.5 Matthews Correlation Coefficient (MCC)

#### 1) Definition
The Matthews Correlation Coefficient is regarded by statisticians as the **single most rigorous and informative score** for binary classification on imbalanced datasets. It evaluates the Pearson correlation coefficient between the actual binary ground truth and the model's predictions across **all four quadrants** of the confusion matrix.

#### 2) Mathematical Formula & Calculation
$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$

- Range: $[-1.0, +1.0]$.
  - $+1.0$: Perfect prediction.
  - $0.0$: No better than random coin tossing.
  - $-1.0$: Total disagreement (predicting the exact opposite of reality).

In our 10-drone toy example:
- Numerator: $(3 \times 5) - (1 \times 1) = 15 - 1 = 14$
- Denominator: $\sqrt{(3+1)(3+1)(5+1)(5+1)} = \sqrt{4 \times 4 \times 6 \times 6} = \sqrt{576} = 24$
$$\text{MCC} = \frac{14}{24} \approx 0.5833$$

#### 3) Actual Model Results & Comparison
| Model | MCC Score | Rank |
| :--- | :---: | :---: |
| **ST-GAT** | **$0.9354$** | **#1** |
| **GRU (2-Layer)** | $0.9334$ | #2 |
| **LightGBM** | $0.9277$ | #3 |
| **GATv2** | $0.9267$ | #4 |

An MCC exceeding **$0.93$** across 305,000 samples indicates near-perfect operational correlation with true swarm leadership structure.

---

### 3.6 Balanced Accuracy

#### 1) Definition
Balanced Accuracy avoids sample-size bias by computing the simple unweighted average of the model's accuracy on the positive class (**Recall / Sensitivity**) and its accuracy on the negative class (**Specificity**):
> *"If the swarm had an exactly equal 50/50 split of critical nodes and decoys, what would our accuracy be?"*

#### 2) Mathematical Formula & Calculation
$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
$$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{1}{2} \left( \frac{\text{TP}}{\text{TP} + \text{FN}} + \frac{\text{TN}}{\text{TN} + \text{FP}} \right)$$

In our 10-drone toy example:
- $\text{Sensitivity} = \frac{3}{3 + 1} = 0.7500$
- $\text{Specificity} = \frac{5}{5 + 1} = \frac{5}{6} \approx 0.8333$
$$\text{Balanced Accuracy} = \frac{0.7500 + 0.8333}{2} = 0.7917 \quad (79.17\%)$$

#### 3) Actual Model Results & Comparison
| Model | Balanced Accuracy | Rank |
| :--- | :---: | :---: |
| **GRU (2-Layer)** | **$0.9685$** | **#1** |
| **ST-GAT** | $0.9678$ | #2 |
| **GATv2** | $0.9634$ | #3 |
| **LightGBM** | $0.9631$ | #4 |

GRU achieves the highest balanced accuracy because its specificity ($97.02\%$) and sensitivity ($96.68\%$) are symmetrically outstanding.

---

## 4. Threshold-Free & Ranking Metrics

While threshold metrics evaluate the model at one fixed cutoff ($\theta^*$), military commanders often operate under dynamic constraints:
- Sometimes an interceptor battery only has enough ammunition to shoot **5 drones** $\implies$ Rank-based selection.
- Sometimes rules of engagement require **$90\%$ minimum certainty** before authorizing fire $\implies$ Variable precision thresholding.

---

### 4.1 Precision-Recall Curve & PR-AUC (Average Precision)

#### 1) Definition
**PR-AUC (Area Under the Precision-Recall Curve)** is the **Primary Metric** of this entire project.  
It measures the area under the curve formed by plotting Precision on the $y$-axis and Recall on the $x$-axis as the decision threshold $\theta$ sweeps continuously from $1.0$ down to $0.0$.

#### 2) Why is PR-AUC Superior to ROC-AUC for Imbalanced Swarms?
In imbalanced classification, ROC-AUC plots True Positive Rate vs. False Positive Rate ($\text{FP} / \text{Total Negatives}$). If there are 100,000 decoys, an algorithm could make 1,000 false alarms and its False Positive Rate would still look tiny ($1\%$), giving a deceptively high ROC-AUC ($> 0.99$).

PR-AUC replaces False Positive Rate with **Precision** ($\text{TP} / [\text{TP} + \text{FP}]$). If 1,000 false alarms occur, Precision collapses, directly punishing the PR-AUC score. A high PR-AUC proves the model is truly separating the positive class without generating hidden false alarms.

#### 3) Step-by-Step Calculation Walkthrough
PR-AUC is computed as the weighted Riemann sum of Precision at each threshold step where Recall increases:
$$\text{PR-AUC} = \sum_{k=1}^{M} \big(R(k) - R(k-1)\big) \cdot P(k)$$

Let us calculate PR-AUC by hand for our 10-drone toy example by sorting probabilities in descending order:

| Rank | Drone | True $y$ | Prob $p$ | Cumulative TP | Cumulative FP | Recall ($TP/4$) | Precision ($TP/[TP+FP]$) | $\Delta \text{Recall}$ | Area Contribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| — | — | — | — | 0 | 0 | 0.00 | 1.00 | — | — |
| 1 | D0 | **1** | `0.95` | 1 | 0 | 0.25 | $1/1 = 1.00$ | 0.25 | $0.25 \times 1.00 = 0.250$ |
| 2 | D1 | **1** | `0.88` | 2 | 0 | 0.50 | $2/2 = 1.00$ | 0.25 | $0.25 \times 1.00 = 0.250$ |
| 3 | D2 | **1** | `0.82` | 3 | 0 | 0.75 | $3/3 = 1.00$ | 0.25 | $0.25 \times 1.00 = 0.250$ |
| 4 | D4 | 0 | `0.75` | 3 | 1 | 0.75 | $3/4 = 0.75$ | 0.00 | $0.00$ |
| 5 | D3 | **1** | `0.42` | 4 | 1 | 1.00 | $4/5 = 0.80$ | 0.25 | $0.25 \times 0.80 = 0.200$ |
| 6 | D5 | 0 | `0.20` | 4 | 2 | 1.00 | $4/6 = 0.67$ | 0.00 | $0.00$ |
| 7 | D6 | 0 | `0.15` | 4 | 3 | 1.00 | $4/7 = 0.57$ | 0.00 | $0.00$ |
| 8 | D7 | 0 | `0.08` | 4 | 4 | 1.00 | $4/8 = 0.50$ | 0.00 | $0.00$ |
| 9 | D8 | 0 | `0.05` | 4 | 5 | 1.00 | $4/9 = 0.44$ | 0.00 | $0.00$ |
| 10 | D9 | 0 | `0.02` | 4 | 6 | 1.00 | $4/10 = 0.40$| 0.00 | $0.00$ |

Total PR-AUC:
$$\text{PR-AUC} = 0.250 + 0.250 + 0.250 + 0.200 = \mathbf{0.9500}$$

#### 4) Actual Model Results & Comparison
| Model | PR-AUC | Rank |
| :--- | :---: | :---: |
| **GRU (2-Layer)** | **$0.9929$** | **#1** |
| **ST-GAT** | $0.9922$ | #2 |
| **LightGBM** | $0.9915$ | #3 |
| **GATv2** | $0.9906$ | #4 |

GRU wins PR-AUC because its predicted probabilities exhibit the cleanest rank separation: nearly all critical nodes are assigned scores $>0.90$, while decoys receive scores $<0.05$.

---

### 4.2 ROC Curve & ROC-AUC (Receiver Operating Characteristic)

#### 1) Definition
ROC-AUC measures the area under the curve plotting **True Positive Rate (Recall)** against **False Positive Rate ($\text{FPR} = \text{FP} / \text{Negatives}$)** as the threshold varies.

#### 2) Probabilistic Meaning
ROC-AUC has an intuitive physical interpretation:  
> *"If you randomly pick one critical drone and one decoy from the sky, what is the probability that the AI model gave the critical drone a higher score than the decoy?"*

A score of $0.50$ is random chance; $1.00$ is perfect sorting.

#### 3) Actual Model Results & Comparison
| Model | ROC-AUC | Rank |
| :--- | :---: | :---: |
| **GRU (2-Layer)** | **$0.9949$** | **#1** |
| **ST-GAT** | $0.9948$ | #2 |
| **LightGBM** | $0.9944$ | #3 |
| **GATv2** | $0.9937$ | #4 |

With a ROC-AUC of **$0.9949$**, if a ground station operator randomly samples a critical drone and a decoy, GRU correctly ranks the critical drone higher **$99.49\%$ of the time**.

---

### 4.3 Recall at $\ge 90\%$ Precision (`Recall@90%P`)

#### 1) Definition
In combat rules of engagement (ROE), commanders often mandate a minimum certainty before firing:  
> *"If the weapon system must maintain at least a $90\%$ hit rate on genuine command drones (no more than 1 wasted shot in 10), what percentage of the total enemy command drones can we successfully eliminate?"*

#### 2) Calculation
We trace the Precision-Recall curve, find all operating points where $\text{Precision} \ge 0.90$, and identify the maximum Recall achieved:
$$\text{Recall@90\%P} = \max \big\{ \text{Recall}(\theta) \mid \text{Precision}(\theta) \ge 0.90 \big\}$$

In our 10-drone toy example:
- At Rank 3 (threshold `0.82`), Precision is $1.00$ ($\ge 0.90$) and Recall is $0.75$.
- At Rank 5 (threshold `0.42`), Precision drops to $0.80$ (which is $< 0.90$).
- Therefore, the highest Recall achievable with $\ge 90\%$ Precision is **$0.7500$ ($75.0\%$)**.

#### 3) Actual Model Results & Comparison
| Model | Recall @ 90% Precision | Rank |
| :--- | :---: | :---: |
| **GRU (2-Layer)** | **$0.9823$ ($98.23\%$)** | **#1** |
| **ST-GAT** | $0.9802$ ($98.02\%$) | #2 |
| **LightGBM** | $0.9770$ ($97.70\%$) | #3 |
| **GATv2** | $0.9758$ ($97.58\%$) | #4 |

Under strict 90% precision rules of engagement, GRU neutralizes **$98.23\%$** of all critical nodes in the sky!

---

### 4.4 Top-$K$ Precision & Recall (Rank-Based Targeting)

#### 1) Definition
In many military operations, prior signals intelligence (SIGINT) reveals the exact swarm composition:  
> *"The enemy has launched a 30-drone swarm consisting of exactly 6 command/relay drones and 24 escort decoys."*

In this scenario, operators do not care about arbitrary probability thresholds like $0.50$. They set $K = 6$ and instruct the targeting computer:  
> *"Select the top 6 drones with the highest predicted criticality scores and fire."*

#### 2) Mathematical Formula & Calculation
For each individual snapshot at step $t$:
1. Let $K_t = \sum_{i=1}^{N_t} y_i$ be the true count of critical nodes present.
2. Sort all $N_t$ drones by predicted probability in descending order.
3. Select the top $K_t$ ranked drones.
4. Count how many of these top $K_t$ drones are truly critical ($\text{Hits}_t$).
5. Since the number of predictions equals the number of true targets ($K_t$), Top-$K$ Precision and Top-$K$ Recall are mathematically identical:
   $$\text{Top-}K\text{ Recall} = \text{Top-}K\text{ Precision} = \frac{\text{Hits}_t}{K_t}$$
6. Average this ratio across all snapshots in the scenario:
   $$\text{Top-}K = \frac{1}{S} \sum_{s=1}^{S} \frac{\text{Hits}_s}{K_s}$$

In our 10-drone toy example:
- True criticals: $K = 4$.
- Top 4 scored drones: Drone 0 (`0.95`), Drone 1 (`0.88`), Drone 2 (`0.82`), Drone 4 (`0.75`).
- True criticals among top 4: D0 (Yes), D1 (Yes), D2 (Yes), D4 (No).
- $\text{Hits} = 3$.
$$\text{Top-}K = \frac{3}{4} = 0.7500 \quad (75.0\%)$$

#### 3) Actual Model Results & Comparison
| Model | Top-$K$ Recall (Dynamic $K = \sum y_i$) | Rank |
| :--- | :---: | :---: |
| **ST-GAT** | **$0.9615$ ($96.15\%$)** | **#1** |
| **GRU (2-Layer)** | $0.9596$ ($95.96\%$) | #2 |
| **LightGBM** | $0.9550$ ($95.50\%$) | #3 |
| **GATv2** | $0.9547$ ($95.47\%$) | #4 |

#### 4) Fixed Operational Salvo Budget ($K = 5$)
In real combat deployments, defense batteries do not have infinite ammunition or exact knowledge of $K$; rather, they have a **fixed salvo budget** (e.g. 5 interceptor missiles ready in the launcher). The command system fires at the top 5 highest-confidence threats.

| Model | Precision@5 (Salvo Hit Rate) | Recall@5 (Swarm Damage) | Binary NDCG@5 (Order Quality) | Graded NDCG@5 (Master Node Priority) |
| :--- | :---: | :---: | :---: | :---: |
| **ST-GAT** | **$0.9872$ ($98.72\%$)** | **$0.3810$** | **$0.9884$** | $0.8319$ |
| **GRU (2-Layer)** | $0.9843$ ($98.43\%$) | $0.3785$ | $0.9864$ | **$0.8708$** |
| **GATv2** | $0.9839$ ($98.39\%$) | $0.3786$ | $0.9862$ | $0.8435$ |
| **LightGBM** | $0.9831$ ($98.31\%$) | $0.3774$ | $0.9852$ | $0.8616$ |

> **Operational Insight:** Across all 305,016 test observations, an automated 5-missile salvo guided by ST-GAT scores a **$98.72\%$ hit rate** (fewer than 1.3 wasted missiles in every 100 shots fired), eliminating over $38\%$ of the swarm's core leadership in the very first volley! GRU delivers the highest Graded NDCG@5 ($0.8708$), ensuring that master flight commanders are prioritized before secondary relays.

#### Architectural Explanation
**Why ST-GAT wins Top-$K$ Targeting ($96.15\%$):**  
In dense swarms, multiple escort drones may fly at very similar speeds. However, spatial graph attention calculates dynamic edge weights based on pairwise closing rates and bearing angles. ST-GAT places the true command drones at the very pinnacle of the relative ranking within each snapshot, giving it superior targeting discrimination.

---

### 4.5 Normalized Discounted Cumulative Gain (`NDCG@K`) — Rewarding Correct Ordering

#### 1) The Core Limitation of Top-$K$ Recall
While Top-$K$ Recall evaluates whether the model captured the correct set of critical nodes, **it is completely blind to internal ranking order**.
Consider two hypothetical models predicting on a snapshot with $K = 4$ critical nodes:
- **Model A Ranked Predictions:** `[Critical, Critical, Critical, Decoy]` $\implies$ 3 hits out of 4 ($\text{Top-}K = 75\%$).
- **Model B Ranked Predictions:** `[Decoy, Critical, Critical, Critical]` $\implies$ 3 hits out of 4 ($\text{Top-}K = 75\%$).

Under Top-$K$ Recall, both models receive identical scores ($0.7500$).  
However, in combat air defense, interceptors fire sequentially at Rank 1, then Rank 2, then Rank 3...
- **Model A** immediately eliminates genuine commanders with its first 3 missiles.
- **Model B** wastes its very first (highest-probability) missile on a sacrificial decoy!

To evaluate and reward models that place the highest-value threats at the very top of the priority list, we employ **NDCG@K (Normalized Discounted Cumulative Gain)**.

#### 2) Mathematical Formulas & Position Discounting
Each item placed at rank $i \in \{1, \dots, K\}$ contributes a relevance gain $r_{\pi(i)}$, discounted logarithmically by its position:
$$\text{Discount}(i) = \frac{1}{\log_2(i + 1)}$$

Rank discounts scale as follows:
- Rank 1: $\frac{1}{\log_2(2)} = 1.000$ (Full credit)
- Rank 2: $\frac{1}{\log_2(3)} \approx 0.6309$
- Rank 3: $\frac{1}{\log_2(4)} = 0.5000$
- Rank 4: $\frac{1}{\log_2(5)} \approx 0.4307$
- Rank 5: $\frac{1}{\log_2(6)} \approx 0.3869$

1. **Discounted Cumulative Gain (DCG@K):**
   $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{r_{\pi(i)}} - 1}{\log_2(i + 1)}$$
2. **Ideal Discounted Cumulative Gain (IDCG@K):**
   The maximum possible DCG achievable if the ground-truth items were perfectly sorted in descending order:
   $$\text{IDCG}@K = \sum_{i=1}^K \frac{2^{r^*_i} - 1}{\log_2(i + 1)}$$
3. **Normalized DCG (NDCG@K):**
   $$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K} \in [0.0, 1.0]$$

---

#### 3) Two Complementary Ground-Truth Formulations: Binary vs. Graded

##### Approach A: Binary NDCG@K ($r_i = y_i \in \{0, 1\}$)
- Uses the verified binary label `is_critical`.
- A critical drone yields gain $2^1 - 1 = 1.0$; a decoy yields gain $2^0 - 1 = 0.0$.
- Rewards placing critical drones at ranks $1, 2, \dots$ ahead of decoys. Requires no continuous assumptions.

##### Approach B: Graded NDCG@K ($r_i = \text{criticality\_score}_i \in [0.0, 1.0]$)
- Uses the simulator's continuous `criticality_score` as ground-truth relevance.
- **Why this is strictly valid:** While `criticality_score` is prohibited from model input features during training and inference (to prevent simulator leakage), it represents the ground-truth strategic importance of each agent (master controllers $\approx 0.95$, secondary relays $\approx 0.70$, decoys $\approx 0.02$).
- Graded NDCG@K directly answers: *"Did the AI model prioritize the highest-value command drone over secondary command drones?"*

---

#### 4) Hand Calculation on Toy 10-Drone Swarm
Recall our toy 10-drone snapshot ($K = 4$ true criticals):
- **Model Ranking:** D0 ($y=1, c=0.95$), D1 ($y=1, c=0.88$), D2 ($y=1, c=0.82$), D4 ($y=0, c=0.03$).

##### Step 1: Binary NDCG@4:
- Predicted Top 4 relevance: $[1, 1, 1, 0]$
$$\text{DCG}_{\text{bin}} = \frac{1.0}{\log_2 2} + \frac{1.0}{\log_2 3} + \frac{1.0}{\log_2 4} + \frac{0.0}{\log_2 5} = 1.000 + 0.6309 + 0.5000 + 0.0 = 2.1309$$
- Ideal Top 4 relevance: $[1, 1, 1, 1]$
$$\text{IDCG}_{\text{bin}} = 1.000 + 0.6309 + 0.5000 + 0.4307 = 2.5616$$
$$\text{Binary NDCG}@4 = \frac{2.1309}{2.5616} = \mathbf{0.8319} \quad (83.19\%)$$

*(Contrast this with Model B `[0, 1, 1, 1]`, which yields $\text{DCG} = 0 + 0.6309 + 0.5000 + 0.4307 = 1.5616 \implies \text{NDCG} = \mathbf{0.6096}$. NDCG clearly penalizes the early mistake by over 22 percentage points!)*

---

#### 5) Actual Model Results & Comparison on Held-Out Test Set (305,016 rows)

| Model | Binary NDCG@K | Graded NDCG@K | Top-$K$ Recall (Baseline) | Champion |
| :--- | :---: | :---: | :---: | :---: |
| **ST-GAT** | **$0.9712$** | $0.9134$ | **$0.9615$** | **ST-GAT (Binary Order Champion)** |
| **GRU (2-Layer)** | $0.9698$ | **$0.9230$** | $0.9596$ | **GRU (Graded Hierarchy Champion)** |
| **GATv2** | $0.9665$ | $0.9148$ | $0.9547$ | #3 |
| **LightGBM** | $0.9662$ | $0.9188$ | $0.9550$ | #4 |

#### Architectural Explanation
1. **Why ST-GAT wins Binary NDCG@K ($0.9712$):**  
   Spatial attention identifies the tightest core geometry, while temporal 1D convolutions filter out momentary jitter. This dual constraint pushes genuine leaders almost exclusively to ranks 1, 2, and 3, almost never placing a decoy at the head of the queue.
2. **Why GRU wins Graded NDCG@K ($0.9230$):**  
   The simulator's master command drone maintains steady, smooth flight over the longest trajectory. GRU's 16-step recurrent hidden state accumulates this smooth motion over time, naturally assigning its highest probabilities ($>0.95$) to the primary leader and intermediate probabilities ($0.80 - 0.85$) to secondary relays.

---

### 4.6 Mean Average Precision at $K$ (`MAP@K`) & Mean Reciprocal Rank (`MRR`)

#### 1) Definitions & Formulas
- **Mean Average Precision (MAP@K):** Evaluates the precision at every rank $i \le K$ where a true critical node appears:
  $$\text{AP}@K = \frac{1}{K} \sum_{i=1}^K P@i \cdot y_{\pi(i)} \quad \text{where } P@i = \frac{1}{i} \sum_{j=1}^i y_{\pi(j)}$$
  $\text{MAP}@K$ is the average of $\text{AP}@K$ across all snapshots.
- **Mean Reciprocal Rank (MRR):** Measures how quickly the model locates the *very first* true critical node:
  $$\text{MRR} = \frac{1}{\text{rank of first critical node}}$$

#### 2) Actual Model Results & Comparison
| Model | MAP@K | MRR | Tactical Interpretation |
| :--- | :---: | :---: | :--- |
| **ST-GAT** | **$0.9558$** | $0.9947$ | **#1 Precision at Every Rank:** Minimum false alarms at ranks 1 through $K$. |
| **GRU (2-Layer)** | $0.9538$ | **$0.9957$** | **#1 First-Target Acquisition:** First critical node found at rank 1 in $99.57\%$ of snapshots. |
| **GATv2** | $0.9482$ | $0.9956$ | Strong spatial ranking. |
| **LightGBM** | $0.9475$ | $0.9950$ | High baseline ranking accuracy. |

---

### 4.7 Spearman Rank Correlation ($\rho$) with Swarm Criticality

#### 1) Definition
Spearman's rank correlation coefficient measures how well the monotonic ordering of model predicted probabilities matches the continuous `criticality_score` across the entire swarm snapshot:
$$\rho = 1.0 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$
where $d_i = \text{rank}(p_i) - \text{rank}(\text{criticality\_score}_i)$.

#### 2) Actual Model Results & Comparison
| Model | Spearman Rank Correlation ($\rho$) | Rank |
| :--- | :---: | :---: |
| **GATv2 (Spatial GNN)** | **$0.7766$** | **#1** |
| **ST-GAT** | $0.7673$ | #2 |
| **LightGBM** | $0.7600$ | #3 |
| **GRU (2-Layer)** | $0.7558$ | #4 |

#### Architectural Explanation
**Why GATv2 wins whole-swarm Spearman Correlation ($0.7766$):**  
In the simulation engine, `criticality_score` was computed using full-swarm graph centrality metrics (degree, closeness, betweenness). GATv2's message passing directly operates over the exact radius-55 communication graph, allowing its continuous node representations to correlate most faithfully with the simulator's graph-wide topological scoring.

---

## 5. Specialized Mission & Temporal Metrics

---

### 5.1 Agent-Level Trajectory F1 (`Agent F1`)

#### 1) Definition
In real-world tracking, the ground station does not observe a drone for only a single snapshot; it tracks the drone over seconds or minutes.  
Agent-Level Trajectory F1 answers:  
> *"If we average all predicted probabilities for a drone over its entire flight path, how accurately do we classify its identity over the mission?"*

#### 2) Calculation
For each unique drone $i$ in scenario $s$, compute its trajectory mean score:
$$\bar{p}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} p_i(t)$$
Label the drone critical if $\bar{p}_i \ge \theta_{\text{agent}}^*$. Then compute the standard F1-score across all unique drones in the held-out scenarios.

#### 3) Actual Model Results & Comparison
| Model | Agent-Level F1 | Agent Precision | Agent Recall | Rank |
| :--- | :---: | :---: | :---: | :---: |
| **ST-GAT** | **$1.0000$ ($100.0\%$)** | **$1.0000$** | **$1.0000$** | **#1 (Flawless)** |
| **GRU (2-Layer)** | **$1.0000$ ($100.0\%$)** | **$1.0000$** | **$1.0000$** | **#1 (Flawless)** |
| **LightGBM** | $0.9980$ ($99.80\%$) | $0.9960$ | $1.0000$ | #3 |
| **GATv2** | $0.9980$ ($99.80\%$) | $0.9960$ | $1.0000$ | #3 |

#### Operational Takeaway
Both **ST-GAT and GRU achieved a perfect 1.0000 Agent F1** across all 15 held-out test scenarios!  
While a single snapshot might occasionally experience a momentary misclassification during an evasive maneuver, averaging scores over an agent's trajectory completely washes out transient noise, ensuring **zero mission-level identification errors**.

---

### 5.2 Mission Progress Breakdown ($t=0$ Cold-Start vs. Late Mission)

Swarm behavior changes dramatically over time:
- At $t=0$ (Mission Launch): Drones are just departing staging areas; velocities are uniform; no historical trajectory exists.
- At Late Mission ($t > 0.67$): Formation escort bubbles have settled; attrition has occurred; trajectories are well-established.

#### Actual Model Performance across Mission Timelines
| Model | $t=0$ Snapshot PR-AUC | Early Mission ($0 - 33\%$) | Mid Mission ($33 - 67\%$) | Late Mission ($67 - 100\%$) |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | **$0.8966$** | $0.9850$ | $0.9942$ | **$0.9976$** |
| **ST-GAT** | $0.7179$ | $0.9812$ | $0.9938$ | $0.9960$ |
| **GATv2** | $0.6767$ | $0.9780$ | $0.9915$ | $0.9949$ |
| **GRU (2-Layer)** | $0.5836$ | $0.9821$ | $0.9950$ | $0.9964$ |

#### Key Insights:
1. **The $t=0$ Cold-Start Dilemma:**  
   At the very first frame ($t=0$), GRU has no prior memory (15 of its 16 window slots are zero-padded), yielding a modest $0.5836$ PR-AUC. In contrast, **LightGBM leads at $t=0$ ($0.8966$)** because decision trees immediately split on static radial geometry (`dist_to_centroid`, `min_neighbor_dist`).
2. **Rapid Convergence:**  
   By step 8 (less than 1 second of flight), GRU and ST-GAT catch up completely ($>0.98$). By late mission, all models achieve near-perfection ($>0.995$).

---

### 5.3 Dynamic Field Promotion Tracking

One of the hardest challenges in the dataset is **in-flight role transitions**:
- In attrition scenarios, when a critical leader drone is shot down, the swarm's decentralized algorithm dynamically promotes an escort decoy to become the new leader.
- There are **85 discrete $0 \to 1$ promotion events** across 41 scenarios.
- Prior to promotion, the drone was an escort decoy ($y=0$); after promotion, it is a critical command node ($y=1$).

We measure **Post-Promotion Recall**:
> *"Once an escort is promoted to command, what percentage of its subsequent flight frames does the model correctly recognize as critical?"*

| Model | Post-Promotion Recall | Tactical Assessment |
| :--- | :---: | :--- |
| **GRU (2-Layer)** | **$82.06\%$** | **Outstanding Dynamic Tracking:** Internal hidden state adapts quickly as promoted drone migrates to core. |
| **ST-GAT** | $72.37\%$ | **Strong:** Spatial graph attention detects newly formed escort ring within 8 steps. |
| **LightGBM** | $69.43\%$ | **Moderate:** Lag features lag behind the transition for several timesteps. |
| **GATv2** | $62.38\%$ | **Struggling:** Purely spatial model without temporal momentum struggles with newly promoted peripheral leaders. |

---

## 6. Model Selection Guide: Which Model for Which Mission?

Based on our complete metric analysis, there is no single "best" model for all scenarios—each architecture fills a distinct tactical niche:

```
+----------------------------------------------------------------------------------------------------+
|                                  OPERATIONAL DEPLOYMENT MATRIX                                     |
+-------------------+--------------------+------------------------+----------------------------------+
| MODEL             | STRONGEST METRIC   | INFERENCE PROFILE      | RECOMMENDED DEPLOYMENT           |
+-------------------+--------------------+------------------------+----------------------------------+
| **GRU (2-Layer)** | PR-AUC: 0.9929     | O(1) Recursive Stream  | **Forward Airborne Interceptor** |
|                   | Recall: 0.9668     | Latency: 0.06 ms/agent | Low compute, battery-powered,    |
|                   | Promo Rec: 82.1%   | Zero history buffer    | requires zero packet history.    |
+-------------------+--------------------+------------------------+----------------------------------+
| **ST-GAT**        | F1-Score: 0.9591   | T=8 Sliding Graph Win  | **Centralized Command Post**     |
|                   | Top-K Rec: 0.9615  | Latency: 18.5 ms       | Heavy ground station workstation |
|                   | Precision: 0.9585  | Requires PyG graph     | prioritizing maximum precision.  |
+-------------------+--------------------+------------------------+----------------------------------+
| **LightGBM**      | Train: 96.4 s      | Tabular Vector (52D)   | **Rapid Baseline & Cold-Start**  |
|                   | t=0 PR-AUC: 0.8966 | Latency: 1.5 ms        | Non-neural environments, offline |
|                   | PR-AUC: 0.9915     | Requires lag buffer    | analysis, immediate t=0 launch.  |
+-------------------+--------------------+------------------------+----------------------------------+
| **GATv2**         | Agent F1: 0.9980   | Single-Snapshot Graph  | **Memoryless Spatial Tracking**  |
|                   | PR-AUC: 0.9906     | Latency: 3.8 ms        | When historical frame storage is |
|                   | Prec: 0.9531       | Zero history buffer    | strictly prohibited by policy.   |
+-------------------+--------------------+------------------------+----------------------------------+
```

---

## 7. Summary Reference Cheat Sheet

- **Precision** $= \frac{\text{TP}}{\text{TP} + \text{FP}} \implies$ How clean are our alarms? *(ST-GAT wins: $95.85\%$)*
- **Recall** $= \frac{\text{TP}}{\text{TP} + \text{FN}} \implies$ Did we catch all leaders? *(GRU wins: $96.68\%$)*
- **Miss Rate (FNR)** $= 1 - \text{Recall} \implies$ What got through? *(GRU wins: lowest risk at $3.32\%$)*
- **F1-Score** $= 2 \frac{P \cdot R}{P + R} \implies$ Best overall balance. *(ST-GAT wins: $0.9591$)*
- **MCC** $\implies$ True four-quadrant correlation. *(ST-GAT wins: $0.9354$)*
- **PR-AUC** $\implies$ Area under Precision-Recall curve across all thresholds. *(GRU wins: $0.9929$)*
- **Recall@90%P** $\implies$ How much can we destroy with $90\%$ certainty? *(GRU wins: $98.23\%$)*
- **Top-$K$ Recall** $\implies$ Did we pick the right leaders when prior count $K$ is known? *(ST-GAT wins: $96.15\%$)*
- **Binary NDCG@K** $\implies$ Did we put critical leaders at the very top of the list? *(ST-GAT wins: $0.9712$)*
- **Graded NDCG@K** $\implies$ Did we prioritize master commanders over secondary relays? *(GRU wins: $0.9230$)*
- **MAP@K** $\implies$ Mean precision at every rank where a leader appears. *(ST-GAT wins: $0.9558$)*
- **Spearman Rank Corr. ($\rho$)** $\implies$ Whole-swarm alignment with continuous criticality. *(GATv2 wins: $0.7766$)*
- **Agent F1** $\implies$ Trajectory-level classification. *(ST-GAT and GRU tie for perfection: $1.0000$)*
- **Post-Promotion Recall** $\implies$ How fast do we catch replacement leaders? *(GRU wins: $82.06\%$)*

