"""Demonstration: End-to-end Swarm Critical-Node Inference from Pure Telemetry.

Demonstrates real-time inference at the Ground Station using strictly 3D position
and velocity telemetry downlinks:
    - 3D Coordinates: (Px, Py, Pz)
    - 3D Velocities:  (Vx, Vy, Vz)
without onboard IMU commands, target coordinates, or environment data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.inference.predictor import SwarmCriticalityPredictor, predict_all_models
from src.data.loader import load_raw_dataset


def generate_synthetic_telemetry_stream(n_agents: int = 24, n_steps: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a realistic continuous 3D telemetry stream for an aerial swarm."""
    np.random.seed(42)
    dt = 0.10  # 10 Hz observation rate

    # Cluster half the swarm in protected center (VIP + tight escorts) and half on perimeter (decoys)
    agent_types = {}
    base_pos = {}
    base_vel = {}

    for i in range(n_agents):
        if i < 4:
            # VIP Core nodes: tightly centered, forward cruise
            agent_types[i] = "Core VIP"
            r = np.random.uniform(2.0, 12.0)
            theta = np.random.uniform(0, 2 * np.pi)
            base_pos[i] = np.array([r * np.cos(theta), r * np.sin(theta), np.random.uniform(-3.0, 3.0)])
            base_vel[i] = np.array([22.0 + np.random.normal(0, 0.2), np.random.normal(0, 0.2), np.random.normal(0, 0.1)])
        elif i < 14:
            # Close Bodyguards: within 15-25m
            agent_types[i] = "Inner Escort"
            r = np.random.uniform(12.0, 22.0)
            theta = np.random.uniform(0, 2 * np.pi)
            base_pos[i] = np.array([r * np.cos(theta), r * np.sin(theta), np.random.uniform(-6.0, 6.0)])
            base_vel[i] = np.array([22.0 + np.random.normal(0, 0.5), np.random.normal(0, 0.5), np.random.normal(0, 0.3)])
        else:
            # Outer Decoys: on 35-65m perimeter, weaving/evading
            agent_types[i] = "Outer Decoy"
            r = np.random.uniform(32.0, 55.0)
            theta = np.random.uniform(0, 2 * np.pi)
            base_pos[i] = np.array([r * np.cos(theta), r * np.sin(theta), np.random.uniform(-15.0, 15.0)])
            base_vel[i] = np.array([21.0 + np.random.normal(0, 1.5), np.random.normal(0, 2.0), np.random.normal(0, 1.0)])

    rows = []
    # Swarm centroid travels forward at 22 m/s along X
    centroid_pos = np.array([0.0, 0.0, 500.0])

    for s in range(n_steps):
        step_id = s * 2
        t = s * dt
        c_t = centroid_pos + np.array([22.0 * t, 0.0, 0.0])

        for aid in range(n_agents):
            # Pos = centroid + relative offset + slight wander
            p = c_t + base_pos[aid] + np.random.normal(0, 0.1, 3) * s
            v = base_vel[aid] + np.random.normal(0, 0.05, 3)

            rows.append({
                "agent_id": aid,
                "step": step_id,
                "role_type": agent_types[aid],
                "px": float(p[0]),
                "py": float(p[1]),
                "pz": float(p[2]),
                "vx": float(v[0]),
                "vy": float(v[1]),
                "vz": float(v[2]),
            })

    full_stream = pd.DataFrame(rows)
    curr_step = (n_steps - 1) * 2
    snap_df = full_stream[full_stream["step"] == curr_step].copy()
    hist_df = full_stream[full_stream["step"] < curr_step].copy()
    return snap_df, hist_df


def run_sample_inference(use_dataset_sample: bool = True) -> pd.DataFrame:
    """Run real-time inference on a swarm telemetry packet across all 4 models."""
    print("=" * 80)
    print("GROUND STATION REAL-TIME TELEMETRY INFERENCE DEMONSTRATION")
    print("Contract: Minimal Downlink Strictly (Px, Py, Pz, Vx, Vy, Vz)")
    print("=" * 80)

    if use_dataset_sample:
        print("\n[1/3] Extracting held-out scenario telemetry slice (c001_n60_none @ step 40)...")
        df_raw = load_raw_dataset()
        sub = df_raw[df_raw["case_id"] == "c001_n60_none"].sort_values(["agent_id", "step"]).copy()
        steps = sorted(sub["step"].unique())
        target_step = steps[20]

        # Convert to pure telemetry contract
        telemetry_cols = ["agent_id", "step", "rel_x", "rel_y", "rel_z", "vx", "vy", "vz", "is_critical"]
        sub_telemetry = sub[telemetry_cols].rename(columns={
            "rel_x": "px",
            "rel_y": "py",
            "rel_z": "pz",
        })

        hist_df = sub_telemetry[sub_telemetry["step"] < target_step].copy()
        snap_df = sub_telemetry[sub_telemetry["step"] == target_step].copy()
        ground_truth = dict(zip(snap_df["agent_id"], snap_df["is_critical"]))
    else:
        print("\n[1/3] Synthesizing continuous 3D telemetry stream (24 agents, 16 steps)...")
        snap_df, hist_df = generate_synthetic_telemetry_stream(n_agents=24, n_steps=16)
        ground_truth = None

    print(f"      Alive agents in packet: {len(snap_df)}")
    print(f"      Telemetry sample (Agent #0):")
    a0 = snap_df.iloc[0]
    print(f"        Pos = [{a0['px']:.2f}, {a0['py']:.2f}, {a0['pz']:.2f}] m")
    print(f"        Vel = [{a0['vx']:.2f}, {a0['vy']:.2f}, {a0['vz']:.2f}] m/s")

    print("\n[2/3] Executing unified inference across all four models...")
    print("      - Deriving 22 kinematics, radial distances, and escort graph metrics from (p, v)")
    print("      - Evaluating LightGBM, 2-Layer GRU, GATv2, and ST-GAT...")

    results = predict_all_models(snap_df, hist_df)

    if ground_truth is not None:
        results["Ground_Truth"] = results["agent_id"].map(ground_truth)

    print("\n[3/3] Inference Results Summary Table (First 15 Agents):")
    cols_to_show = [
        "agent_id",
        "prob_lgbm", "pred_lgbm",
        "prob_gru", "pred_gru",
        "prob_gatv2", "pred_gatv2",
        "prob_stgat", "pred_stgat",
        "prob_consensus", "consensus_critical",
    ]
    if "Ground_Truth" in results.columns:
        cols_to_show.append("Ground_Truth")

    display_df = results[cols_to_show].head(15).copy()

    # Format probabilities as percentages
    for c in ["prob_lgbm", "prob_gru", "prob_gatv2", "prob_stgat", "prob_consensus"]:
        display_df[c] = display_df[c].apply(lambda p: f"{p * 100:.1f}%")

    print(display_df.to_string(index=False))

    # Overall agreement summary
    print("\nModel Consensus Metrics:")
    n_total = len(results)
    lgbm_crit = results["pred_lgbm"].sum()
    gru_crit = results["pred_gru"].sum()
    gat_crit = results["pred_gatv2"].sum()
    stgat_crit = results["pred_stgat"].sum()
    consensus_crit = results["consensus_critical"].sum()

    print(f"  Total Agents Evaluated: {n_total}")
    print(f"  Critical Agents Identified by LightGBM: {lgbm_crit} / {n_total}")
    print(f"  Critical Agents Identified by GRU:      {gru_crit} / {n_total}")
    print(f"  Critical Agents Identified by GATv2:    {gat_crit} / {n_total}")
    print(f"  Critical Agents Identified by ST-GAT:   {stgat_crit} / {n_total}")
    print(f"  Ensemble Consensus Critical Identified: {consensus_crit} / {n_total}")

    if ground_truth is not None:
        true_crit = sum(ground_truth.values())
        print(f"  Actual Ground Truth Critical Agents:    {true_crit} / {n_total}")
        consensus_acc = (results["consensus_critical"] == results["Ground_Truth"]).mean()
        print(f"  Ensemble Consensus Accuracy:            {consensus_acc * 100:.2f}%")

    print("\nDemonstration completed successfully.")
    return results


if __name__ == "__main__":
    run_sample_inference(use_dataset_sample=True)

