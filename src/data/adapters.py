"""Data adapters converting shared features into model-specific tensor representations.

Provides:
- LightGBM tabular adapter (with 54 features)
- GRU sequence adapter (with causal left-padding, mask, and [B, 16, 24] tensors)
- GATv2 graph adapter (PyG Data per snapshot)
- ST-GAT spatio-temporal window adapter (dynamic T=8 graph sequences with attrition masks)
"""
from __future__ import annotations

from typing import Generator
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.config import (
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    LGBM_FEATURES_52,
    LGBM_FEATURES_54,
    TARGET_COLUMN,
)
from src.data.features import (
    validate_feature_matrix,
    extract_snapshot_graph,
)
from src.data.scaler import SwarmFeatureScaler


# -----------------------------------------------------------------------------
# Model 1 Adapter: LightGBM
# -----------------------------------------------------------------------------
def prepare_lgbm_data(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract LightGBM feature matrix [N, 52], binary target, and alignment metadata."""
    validate_feature_matrix(LGBM_FEATURES_52)

    X = df[LGBM_FEATURES_52].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int8)

    meta_cols = ["case_id", "step", "agent_id", "num_agents", "attrition_rate", "is_anomaly", "was_promoted"]
    if "criticality_score" in df.columns:
        meta_cols.append("criticality_score")
    meta = df[meta_cols].copy()
    return X, y, meta


# -----------------------------------------------------------------------------
# Model 2 Adapter: GRU Dataset
# -----------------------------------------------------------------------------
class GRUDataset(Dataset):
    """PyTorch Dataset yielding left-padded causal feature sequences [16, 22]."""

    def __init__(
        self,
        windows: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        anomalies: np.ndarray,
        meta_df: pd.DataFrame,
    ) -> None:
        self.windows = windows  # [N_rows, 16, 22] float32
        self.masks = masks      # [N_rows, 16] bool
        self.targets = targets  # [N_rows] float32
        self.anomalies = anomalies  # [N_rows] bool
        self.meta_df = meta_df.reset_index(drop=True)
        self.n_samples = len(windows)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.tensor(self.targets[idx], dtype=torch.float32),
            torch.tensor(self.anomalies[idx], dtype=torch.bool),
        )


def prepare_gru_dataset(
    df: pd.DataFrame,
    scaler: SwarmFeatureScaler,
    window_length: int = 16,
) -> GRUDataset:
    """Build GRUDataset from DataFrame with preassembled left-padded causal windows."""
    validate_feature_matrix(BASE_FEATURES_22)

    # Ensure deterministic causal ordering
    df_sorted = df.sort_values(["case_id", "agent_id", "step"]).reset_index(drop=True)
    g = df_sorted.groupby(["case_id", "agent_id"], observed=True)
    cumcounts = g.cumcount().to_numpy(dtype=np.int32)
    agent_lens = g.size().to_numpy(dtype=np.int32)

    raw_features = df_sorted[BASE_FEATURES_22].to_numpy(dtype=np.float32)
    norm_features = scaler.transform_nodes(raw_features)
    targets = df_sorted[TARGET_COLUMN].to_numpy(dtype=np.float32)
    anomalies = df_sorted["is_anomaly"].to_numpy(dtype=bool)

    N_samples = len(df_sorted)
    W = window_length
    D = len(BASE_FEATURES_22)

    X_windows = np.zeros((N_samples, W, D), dtype=np.float32)
    masks = np.zeros((N_samples, W), dtype=bool)

    curr = 0
    for L in agent_lens:
        sub = norm_features[curr:curr + L]
        if L >= W:
            windows = np.lib.stride_tricks.sliding_window_view(sub, window_shape=(W, D))[:, 0, :, :]
            X_windows[curr + W - 1: curr + L] = windows
            masks[curr + W - 1: curr + L] = True
            for k in range(W - 1):
                valid_len = k + 1
                X_windows[curr + k, W - valid_len:] = sub[:valid_len]
                masks[curr + k, W - valid_len:] = True
        else:
            for k in range(L):
                valid_len = k + 1
                X_windows[curr + k, W - valid_len:] = sub[:valid_len]
                masks[curr + k, W - valid_len:] = True
        curr += L

    meta_cols = ["case_id", "step", "agent_id", "num_agents", "attrition_rate", "is_anomaly", "was_promoted"]
    if "criticality_score" in df_sorted.columns:
        meta_cols.append("criticality_score")
    meta = df_sorted[meta_cols].copy()

    return GRUDataset(
        windows=X_windows,
        masks=masks,
        targets=targets,
        anomalies=anomalies,
        meta_df=meta,
    )


# -----------------------------------------------------------------------------
# Model 3 Adapter: GATv2 PyG Graph List
# -----------------------------------------------------------------------------
def prepare_gat_graphs(
    df: pd.DataFrame,
    scaler: SwarmFeatureScaler,
) -> list[Data]:
    """Convert snapshots into a list of PyG Data objects with normalized features."""
    validate_feature_matrix(BASE_FEATURES_24)

    df_sorted = df.sort_values(["case_id", "step", "agent_id"]).reset_index(drop=True)
    groups = df_sorted.groupby(["case_id", "step"], observed=True)

    graphs: list[Data] = []
    for (case_id, step), grp in groups:
        x, edge_index, edge_attr, global_feat = extract_snapshot_graph(grp)

        # Standardize using train-fitted scaler
        x_norm = scaler.transform_nodes(x)
        ea_norm = scaler.transform_edges(edge_attr)
        gf_norm = scaler.transform_globals(global_feat.reshape(1, -1)).reshape(1, 6)

        y = grp[TARGET_COLUMN].to_numpy(dtype=np.float32)
        is_bad = grp["is_anomaly"].to_numpy(dtype=bool)
        agent_ids = grp["agent_id"].to_numpy(dtype=np.int32)
        was_promoted = grp["was_promoted"].to_numpy(dtype=bool)
        crit_score = (
            torch.from_numpy(grp["criticality_score"].to_numpy(dtype=np.float32))
            if "criticality_score" in grp.columns
            else None
        )

        data = Data(
            x=torch.from_numpy(x_norm),
            edge_index=torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(ea_norm),
            global_feat=torch.from_numpy(gf_norm),
            y=torch.from_numpy(y),
            loss_mask=torch.from_numpy(~is_bad),
            agent_id=torch.from_numpy(agent_ids),
            was_promoted=torch.from_numpy(was_promoted),
            criticality_score=crit_score,
            num_agents=int(grp["num_agents"].iloc[0]),
            attrition_rate=str(grp["attrition_rate"].iloc[0]),
            case_id=str(case_id),
            step=int(step),
        )
        graphs.append(data)

    return graphs


# -----------------------------------------------------------------------------
# Model 4 Adapter: ST-GAT Windowed Dataset
# -----------------------------------------------------------------------------
class STGATWindowItem:
    """A single spatio-temporal window containing dynamic graphs and node masks."""

    def __init__(
        self,
        x_seq: torch.Tensor,            # [T, N_max, 24]
        node_mask: torch.Tensor,        # [T, N_max] bool
        global_seq: torch.Tensor,      # [T, 6]
        edge_indices: list[torch.Tensor],  # List of [2, E_tau] for each tau
        edge_attrs: list[torch.Tensor],    # List of [E_tau, 7] for each tau
        y_end: torch.Tensor,            # [N_max] float32
        target_mask: torch.Tensor,      # [N_max] bool (nodes alive at window end)
        anomaly_mask: torch.Tensor,     # [N_max] bool (loss mask: alive & ~is_anomaly)
        agent_ids: np.ndarray | list[int] | None = None, # [N_max]
        case_id: str = "",
        step_end: int = 0,
        was_promoted: np.ndarray | None = None, # [N_max]
        criticality_scores: np.ndarray | None = None, # [N_max]
        num_agents: int = 0,
        attrition_rate: str = "",
        **kwargs: Any,
    ) -> None:
        self.x_seq = x_seq
        self.node_mask = node_mask
        self.global_seq = global_seq
        self.edge_indices = edge_indices
        self.edge_attrs = edge_attrs
        self.y_end = y_end
        self.target_mask = target_mask
        self.anomaly_mask = anomaly_mask
        self.agent_ids = np.array(agent_ids) if agent_ids is not None else np.empty(0, dtype=np.int32)
        self.case_id = case_id
        self.step_end = step_end
        self.was_promoted = was_promoted if was_promoted is not None else np.zeros(len(self.agent_ids), dtype=bool)
        self.criticality_scores = criticality_scores
        self.num_agents = num_agents
        self.attrition_rate = attrition_rate


class STGATDataset(Dataset):
    """PyTorch Dataset yielding window items for ST-GAT."""

    def __init__(self, windows: list[STGATWindowItem]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> STGATWindowItem:
        return self.windows[idx]


def build_stgat_windows(
    df: pd.DataFrame,
    scaler: SwarmFeatureScaler,
    window_steps: int = 8,
    stride: int = 1,
) -> list[STGATWindowItem]:
    """Extract causal sliding windows of T consecutive graphs per scenario."""
    validate_feature_matrix(BASE_FEATURES_22)

    df_sorted = df.sort_values(["case_id", "step", "agent_id"]).reset_index(drop=True)
    scenarios = df_sorted.groupby("case_id", observed=True)

    windows: list[STGATWindowItem] = []
    T = window_steps

    for case_id, scen_df in scenarios:
        unique_steps = np.sort(scen_df["step"].unique())
        n_snaps = len(unique_steps)
        if n_snaps == 0:
            continue

        step_groups = {s: grp for s, grp in scen_df.groupby("step", observed=True)}

        # Iterate over end steps with specified stride
        for end_idx in range(0, n_snaps, stride):
            step_end = unique_steps[end_idx]
            start_idx = max(0, end_idx - T + 1)
            active_steps = unique_steps[start_idx:end_idx + 1]
            valid_len = len(active_steps)

            # Union of alive agents across active steps in window
            window_dfs = [step_groups[s] for s in active_steps]
            union_agents = sorted(set().union(*[sub["agent_id"].tolist() for sub in window_dfs]))
            n_agents_win = len(union_agents)
            agent_to_idx = {a: i for i, a in enumerate(union_agents)}

            # Allocate tensors
            x_seq = np.zeros((T, n_agents_win, len(BASE_FEATURES_22)), dtype=np.float32)
            node_mask = np.zeros((T, n_agents_win), dtype=bool)
            global_seq = np.zeros((T, 6), dtype=np.float32)
            edge_indices: list[torch.Tensor] = [torch.empty((2, 0), dtype=torch.int64) for _ in range(T)]
            edge_attrs: list[torch.Tensor] = [torch.empty((0, 7), dtype=torch.float32) for _ in range(T)]

            # Left-pad: place valid steps at [T - valid_len .. T - 1]
            for tau_offset, s in enumerate(active_steps):
                tau = T - valid_len + tau_offset
                sub = step_groups[s]
                x_sub, ei, ea, g_sub = extract_snapshot_graph(sub)

                x_sub = scaler.transform_nodes(x_sub)
                ea_sub = scaler.transform_edges(ea)
                g_sub = scaler.transform_globals(g_sub.reshape(1, -1)).reshape(6)

                sub_agents = sub["agent_id"].to_numpy()
                mapped_indices = np.array([agent_to_idx[a] for a in sub_agents], dtype=np.int64)

                x_seq[tau, mapped_indices] = x_sub
                node_mask[tau, mapped_indices] = True
                global_seq[tau] = g_sub

                # Local snapshot edge_index matches x_alive order in sub
                edge_indices[tau] = torch.from_numpy(ei).long()
                edge_attrs[tau] = torch.from_numpy(ea_sub).float()

            # Target at window end
            end_sub = step_groups[step_end]
            end_agent_ids = end_sub["agent_id"].to_numpy()
            end_targets = end_sub[TARGET_COLUMN].to_numpy(dtype=np.float32)
            end_anomalies = end_sub["is_anomaly"].to_numpy(dtype=bool)
            end_promos = end_sub["was_promoted"].to_numpy(dtype=bool)
            has_crit = "criticality_score" in end_sub.columns
            end_crits = end_sub["criticality_score"].to_numpy(dtype=np.float32) if has_crit else None

            y_end = np.zeros(n_agents_win, dtype=np.float32)
            target_mask = np.zeros(n_agents_win, dtype=bool)
            anomaly_mask = np.zeros(n_agents_win, dtype=bool)
            promos_end = np.zeros(n_agents_win, dtype=bool)
            crits_end = np.zeros(n_agents_win, dtype=np.float32) if has_crit else None

            for i_end, (a_id, y_val, is_bad, is_promo) in enumerate(zip(end_agent_ids, end_targets, end_anomalies, end_promos)):
                u = agent_to_idx[a_id]
                y_end[u] = y_val
                target_mask[u] = True
                anomaly_mask[u] = not is_bad
                promos_end[u] = is_promo
                if has_crit and end_crits is not None and crits_end is not None:
                    crits_end[u] = end_crits[i_end]

            item = STGATWindowItem(
                x_seq=torch.from_numpy(x_seq),
                node_mask=torch.from_numpy(node_mask),
                global_seq=torch.from_numpy(global_seq),
                edge_indices=edge_indices,
                edge_attrs=edge_attrs,
                y_end=torch.from_numpy(y_end),
                target_mask=torch.from_numpy(target_mask),
                anomaly_mask=torch.from_numpy(anomaly_mask),
                agent_ids=np.array(union_agents, dtype=np.int32),
                case_id=str(case_id),
                step_end=int(step_end),
                was_promoted=promos_end,
                criticality_scores=crits_end,
                num_agents=int(scen_df["num_agents"].iloc[0]),
                attrition_rate=str(scen_df["attrition_rate"].iloc[0]),
            )
            windows.append(item)

    return windows
