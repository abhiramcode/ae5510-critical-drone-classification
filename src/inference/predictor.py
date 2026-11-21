"""Unified inference engine for aerial swarm critical-node identification.

Provides a standardized, leak-free inference API for all four trained models:
- "lgbm": Tabular LightGBM GBDT (52 features: 22 base + 30 causal lags)
- "gru": 2-layer unidirectional GRU (22 base features, windowed or O(1) streaming)
- "gatv2": 2-layer GATv2 spatial GNN (22 node, 7 edge, 6 global features)
- "stgat": Spatio-temporal GATv2+GRU (dynamic T=8 sliding windows)

Operates strictly from minimal downlinked telemetry:
- 3D Position: (Px, Py, Pz) or (x, y, z)
- 3D Velocity: (Vx, Vy, Vz) or (vx, vy, vz)
Deriving all kinematics, relative centroid geometry, and local escort graphs dynamically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import (
    RUNS_DIR,
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    LGBM_FEATURES_52,
    LGBM_FEATURES_54,
    DEVICE,
)
from src.data.features import (
    compute_derived_snapshot_features,
    compute_causal_lags,
    extract_snapshot_graph,
    derive_features_from_telemetry,
    validate_feature_matrix,
)
from src.data.scaler import SwarmFeatureScaler
from src.models.lgbm_model import SwarmLightGBM
from src.models.gru_model import SwarmGRU
from src.models.gat_model import SwarmGATv2
from src.models.stgat_model import SwarmSTGAT


def prepare_telemetry_snapshot(
    snapshot_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    dt_obs: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Ensure telemetry frames have all 22 base features derived from position and velocity."""
    # Check if snapshot is raw telemetry or already contains dataset base columns
    needs_derivation = not all(col in snapshot_df.columns for col in BASE_FEATURES_22)

    if not needs_derivation:
        # Already has full dataset feature columns, just ensure derived features like centroid_vel_alignment exist
        feat_snap = compute_derived_snapshot_features(snapshot_df)
        feat_hist = compute_derived_snapshot_features(history_df) if history_df is not None and len(history_df) > 0 else None
        return feat_snap, feat_hist

    # Raw telemetry: derive from position & velocity
    if history_df is not None and len(history_df) > 0:
        # History is available: compute causal derivatives across consecutive timesteps
        all_raw = pd.concat([history_df, snapshot_df], ignore_index=True)
        steps = sorted(all_raw["step"].unique())
        derived_steps: list[pd.DataFrame] = []
        prev_sub = None
        for s in steps:
            curr_sub = all_raw[all_raw["step"] == s].copy()
            d_curr = derive_features_from_telemetry(curr_sub, prev_snapshot_df=prev_sub, dt_obs=dt_obs)
            derived_steps.append(d_curr)
            prev_sub = curr_sub

        full_derived = pd.concat(derived_steps, ignore_index=True)
        curr_step = snapshot_df["step"].iloc[0] if "step" in snapshot_df.columns else steps[-1]
        feat_snap = full_derived[full_derived["step"] == curr_step].copy()
        feat_hist = full_derived[full_derived["step"] < curr_step].copy()
        return feat_snap, feat_hist
    else:
        # First snapshot with no prior history
        feat_snap = derive_features_from_telemetry(snapshot_df, prev_snapshot_df=None, dt_obs=dt_obs)
        return feat_snap, None


class SwarmCriticalityPredictor:
    """Unified inference predictor for aerial swarm critical-node identification."""

    VALID_MODELS = {"lgbm", "gru", "gatv2", "stgat"}

    def __init__(
        self,
        model_name: str,
        run_dir: Path | str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name.lower().strip()
        if self.model_name not in self.VALID_MODELS:
            raise ValueError(f"Invalid model_name '{model_name}'. Must be one of {self.VALID_MODELS}")

        self.run_dir = Path(run_dir) if run_dir is not None else RUNS_DIR / self.model_name
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        self.device = device or DEVICE

        # 1. Load scaler
        scaler_path = self.run_dir / "preprocessing.json"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler preprocessing file missing: {scaler_path}")
        self.scaler = SwarmFeatureScaler.load(scaler_path)

        # 2. Load model & threshold
        self.model: Any = None
        self.threshold: float = 0.5
        self._load_model()

    def _load_model(self) -> None:
        """Instantiate and load model weights from checkpoint."""
        if self.model_name == "lgbm":
            self.model = SwarmLightGBM.load(self.run_dir)
            self.threshold = self.model.threshold
        elif self.model_name == "gru":
            self.model, self.threshold = SwarmGRU.load(self.run_dir, device=self.device)
        elif self.model_name == "gatv2":
            self.model, self.threshold = SwarmGATv2.load(self.run_dir, device=self.device)
        elif self.model_name == "stgat":
            self.model, self.threshold = SwarmSTGAT.load(self.run_dir, device=self.device)

    def predict_snapshot(
        self,
        snapshot_df: pd.DataFrame,
        history_df: pd.DataFrame | None = None,
        dt_obs: float = 0.10,
    ) -> pd.DataFrame:
        """Ground station real-time inference on a single snapshot of alive agents.

        Parameters
        ----------
        snapshot_df : pd.DataFrame
            Telemetry rows for alive agents at timestamp t.
            Accepts raw [agent_id, px, py, pz, vx, vy, vz] or precomputed features.
        history_df : pd.DataFrame | None
            Historical telemetry rows for alive agents at timestamps < t.
            Required for temporal models (LightGBM, GRU, ST-GAT).
        dt_obs : float
            Observation interval (default: 0.10s).

        Returns
        -------
        pd.DataFrame with columns: ["agent_id", "prob", "pred_class"]
        """
        feat_snap, feat_hist = prepare_telemetry_snapshot(snapshot_df, history_df, dt_obs=dt_obs)

        if self.model_name == "gatv2":
            # GATv2 requires only current snapshot telemetry
            feat_snap = feat_snap.sort_values("agent_id").reset_index(drop=True)

            raw_nodes = feat_snap[BASE_FEATURES_22].to_numpy(dtype=np.float32)
            norm_nodes = self.scaler.transform_nodes(raw_nodes)

            _, ei, ea, g = extract_snapshot_graph(feat_snap)
            ea_norm = self.scaler.transform_edges(ea)
            g_norm = self.scaler.transform_globals(g)

            x_t = torch.from_numpy(norm_nodes).to(self.device)
            ei_t = torch.from_numpy(ei).to(self.device)
            ea_t = torch.from_numpy(ea_norm).to(self.device)
            g_t = torch.from_numpy(g_norm).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_t, ei_t, ea_t, g_t)
                probs = torch.sigmoid(logits).cpu().numpy()

            res = pd.DataFrame({
                "agent_id": feat_snap["agent_id"].values,
                "prob": probs,
                "pred_class": (probs >= self.threshold).astype(int),
            })
            return res

        elif self.model_name == "lgbm":
            # Combine history and current snapshot to compute causal lag diffs
            if feat_hist is None or len(feat_hist) == 0:
                full_df = feat_snap.copy()
            else:
                full_df = pd.concat([feat_hist, feat_snap], ignore_index=True)

            feat_df = compute_causal_lags(full_df)

            curr_step = feat_snap["step"].iloc[0]
            curr_slice = feat_df[feat_df["step"] == curr_step].sort_values("agent_id").reset_index(drop=True)

            X = curr_slice[LGBM_FEATURES_52].to_numpy(dtype=np.float32)
            probs = self.model.predict_proba(X)

            res = pd.DataFrame({
                "agent_id": curr_slice["agent_id"].values,
                "prob": probs,
                "pred_class": (probs >= self.threshold).astype(int),
            })
            return res

        elif self.model_name == "gru":
            if feat_hist is None or len(feat_hist) == 0:
                full_df = feat_snap.copy()
            else:
                full_df = pd.concat([feat_hist, feat_snap], ignore_index=True)

            curr_step = feat_snap["step"].iloc[0]
            alive_agents = feat_snap["agent_id"].unique()

            # Build W=16 causal sequences for each agent
            W = 16
            D = len(BASE_FEATURES_22)
            windows: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            agent_order: list[int] = []

            for aid in sorted(alive_agents):
                agent_hist = full_df[
                    (full_df["agent_id"] == aid) & (full_df["step"] <= curr_step)
                ].sort_values("step")

                raw_arr = agent_hist[BASE_FEATURES_22].to_numpy(dtype=np.float32)
                norm_arr = self.scaler.transform_nodes(raw_arr)

                valid_len = min(len(norm_arr), W)
                w_arr = np.zeros((W, D), dtype=np.float32)
                m_arr = np.zeros(W, dtype=bool)

                w_arr[W - valid_len:] = norm_arr[-valid_len:]
                m_arr[W - valid_len:] = True

                windows.append(w_arr)
                masks.append(m_arr)
                agent_order.append(aid)

            x_batch = torch.from_numpy(np.stack(windows)).to(self.device)
            m_batch = torch.from_numpy(np.stack(masks)).to(self.device)

            self.model.eval()
            with torch.no_grad():
                logits = self.model(x_batch, m_batch)
                probs = torch.sigmoid(logits).cpu().numpy()

            res = pd.DataFrame({
                "agent_id": agent_order,
                "prob": probs,
                "pred_class": (probs >= self.threshold).astype(int),
            })
            return res

        elif self.model_name == "stgat":
            # Assemble T=8 historical snapshots
            if feat_hist is None or len(feat_hist) == 0:
                full_df = feat_snap.copy()
            else:
                full_df = pd.concat([feat_hist, feat_snap], ignore_index=True)

            steps = sorted(full_df["step"].unique())
            window_steps = steps[-8:]
            T = len(window_steps)

            curr_step = feat_snap["step"].iloc[0]
            curr_agents = sorted(feat_snap["agent_id"].unique())

            # Map agents to local indices
            all_window_df = full_df[full_df["step"].isin(window_steps)]
            unique_agents = sorted(all_window_df["agent_id"].unique())
            agent_to_idx = {aid: i for i, aid in enumerate(unique_agents)}
            N_max = len(unique_agents)

            x_seq = np.zeros((T, N_max, len(BASE_FEATURES_22)), dtype=np.float32)
            node_mask = np.zeros((T, N_max), dtype=bool)
            edge_indices: list[torch.Tensor] = []
            edge_attrs: list[torch.Tensor] = []
            globals_list: list[np.ndarray] = []

            for tau, s in enumerate(window_steps):
                s_df = full_df[full_df["step"] == s].sort_values("agent_id").reset_index(drop=True)
                s_aids = s_df["agent_id"].values
                local_indices = [agent_to_idx[aid] for aid in s_aids]

                raw_n = s_df[BASE_FEATURES_22].to_numpy(dtype=np.float32)
                norm_n = self.scaler.transform_nodes(raw_n)

                x_seq[tau, local_indices] = norm_n
                node_mask[tau, local_indices] = True

                _, ei, ea, g = extract_snapshot_graph(s_df)
                ea_norm = self.scaler.transform_edges(ea)
                g_norm = self.scaler.transform_globals(g)

                edge_indices.append(torch.from_numpy(ei).long())
                edge_attrs.append(torch.from_numpy(ea_norm).float())
                globals_list.append(g_norm)

            target_mask = np.zeros(N_max, dtype=bool)
            eval_indices = [agent_to_idx[aid] for aid in curr_agents]
            target_mask[eval_indices] = True

            from src.data.adapters import STGATWindowItem
            item = STGATWindowItem(
                x_seq=torch.from_numpy(x_seq).float(),
                node_mask=torch.from_numpy(node_mask),
                edge_indices=edge_indices,
                edge_attrs=edge_attrs,
                global_seq=torch.from_numpy(np.stack(globals_list)).float(),
                target_mask=torch.from_numpy(target_mask),
                y_end=torch.zeros(N_max, dtype=torch.float32),
                anomaly_mask=torch.ones(N_max, dtype=torch.bool),
                agent_ids=unique_agents,
                case_id="live",
                step_end=curr_step,
            )

            self.model.eval()
            with torch.no_grad():
                logits, _, _ = self.model.forward_single_window(item, device=self.device)
                probs = torch.sigmoid(logits).cpu().numpy()

            res = pd.DataFrame({
                "agent_id": curr_agents,
                "prob": probs,
                "pred_class": (probs >= self.threshold).astype(int),
            })
            return res

        else:
            raise NotImplementedError(f"Snapshot prediction not implemented for {self.model_name}")

    def predict_step_streaming(
        self,
        snapshot_df: pd.DataFrame,
        hidden_states: dict[int, torch.Tensor] | None = None,
        dt_obs: float = 0.10,
    ) -> tuple[pd.DataFrame, dict[int, torch.Tensor]]:
        """Streaming O(1) step inference for GRU."""
        if self.model_name != "gru":
            raise ValueError(f"predict_step_streaming is only supported for GRU (got {self.model_name})")

        hidden_states = hidden_states or {}
        feat_snap, _ = prepare_telemetry_snapshot(snapshot_df, None, dt_obs=dt_obs)
        feat_snap = feat_snap.sort_values("agent_id").reset_index(drop=True)

        raw_nodes = feat_snap[BASE_FEATURES_22].to_numpy(dtype=np.float32)
        norm_nodes = self.scaler.transform_nodes(raw_nodes)

        probs: list[float] = []
        new_h: dict[int, torch.Tensor] = {}

        self.model.eval()
        with torch.no_grad():
            for i, aid in enumerate(feat_snap["agent_id"].values):
                x_i = torch.from_numpy(norm_nodes[i:i + 1]).to(self.device)  # [1, 22]
                prev_h = hidden_states.get(int(aid), None)
                if prev_h is not None:
                    prev_h = prev_h.to(self.device)

                logit_i, next_h = self.model.step_streaming(x_i, prev_h)
                prob_i = float(torch.sigmoid(logit_i).cpu().item())

                probs.append(prob_i)
                new_h[int(aid)] = next_h.cpu()

        res = pd.DataFrame({
            "agent_id": feat_snap["agent_id"].values,
            "prob": np.array(probs, dtype=np.float32),
            "pred_class": (np.array(probs) >= self.threshold).astype(int),
        })
        return res, new_h


def predict_all_models(
    snapshot_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    dt_obs: float = 0.10,
) -> pd.DataFrame:
    """Run all 4 models on a given telemetry snapshot and produce a consolidated decision table."""
    models = ["lgbm", "gru", "gatv2", "stgat"]
    dfs = []
    for m in models:
        pred = SwarmCriticalityPredictor(m)
        res = pred.predict_snapshot(snapshot_df, history_df, dt_obs=dt_obs)
        res = res.rename(columns={
            "prob": f"prob_{m}",
            "pred_class": f"pred_{m}",
        })
        dfs.append(res.set_index("agent_id"))

    combined = pd.concat(dfs, axis=1).reset_index()
    # Compute ensemble consensus probability (average across the 4 models)
    combined["prob_consensus"] = combined[[f"prob_{m}" for m in models]].mean(axis=1)
    combined["consensus_critical"] = (combined["prob_consensus"] >= 0.50).astype(int)
    return combined
