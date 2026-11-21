"""Spatio-Temporal Graph Neural Network (ST-GAT).

Combines a shared 3-layer GATv2 spatial encoder (applied at each timestep over dynamic
topologies) with a per-node 1-layer GRU over the spatial-temporal embedding sequences.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from src.config import (
    STGAT_CONFIG,
    BASE_FEATURES_22,
    BASE_FEATURES_24,
    EDGE_FEATURES_7,
    GLOBAL_FEATURES_6,
)
from src.models.gat_model import SwarmGATv2
from src.data.adapters import STGATWindowItem


class SwarmSTGAT(nn.Module):
    """Shared GATv2 spatial encoder + per-node GRU for dynamic aerial swarm classification."""

    def __init__(
        self,
        in_channels: int = 22,
        hidden_size: int = 64,
        edge_dim: int = 7,
        num_heads: int = 4,
        num_layers: int = 2,
        global_dim: int = 6,
        temporal_hidden: int = 64,
        spatial_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.global_dim = global_dim
        self.temporal_hidden = temporal_hidden

        # Shared spatial GATv2 encoder
        self.spatial_encoder = SwarmGATv2(
            in_channels=in_channels,
            hidden_size=hidden_size,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            global_dim=global_dim,
            dropout=spatial_dropout,
        )

        # Per-node temporal GRU (takes spatial embedding [64])
        self.temporal_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Final classification head
        self.head = nn.Sequential(
            nn.LayerNorm(temporal_hidden),
            nn.Linear(temporal_hidden, temporal_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(temporal_hidden, 1),
        )

    def forward_window(
        self,
        item: STGATWindowItem,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single spatio-temporal window item.

        Returns:
            logits_eval: [N_eval] predictions for nodes alive at window end
            y_eval: [N_eval] ground-truth labels for nodes alive at window end
            loss_mask_eval: [N_eval] loss mask (alive & ~is_anomaly)
        """
        T = item.x_seq.shape[0]  # 8
        N_max = item.x_seq.shape[1]

        # 1. Spatial pass per timestep
        spatial_embeddings = torch.zeros((T, N_max, self.hidden_size), device=device)

        for tau in range(T):
            alive_mask = item.node_mask[tau].to(device)
            if not alive_mask.any():
                continue

            alive_indices = torch.where(alive_mask)[0]

            # Node features
            x_alive = item.x_seq[tau, alive_indices].to(device)

            # Direct local edge index and attributes
            ei_local = item.edge_indices[tau].to(device)
            ea_local = item.edge_attrs[tau].to(device)

            # Run shared spatial backbone
            z_alive = self.spatial_encoder.extract_spatial_embeddings(
                x=x_alive,
                edge_index=ei_local,
                edge_attr=ea_local,
            )
            spatial_embeddings[tau, alive_indices] = z_alive

        # 2. Temporal pass for nodes alive at window end (target_mask == True)
        eval_indices = torch.where(item.target_mask.to(device))[0]
        N_eval = len(eval_indices)
        if N_eval == 0:
            return torch.empty(0, device=device), torch.empty(0, device=device), torch.empty(0, device=device)

        # Assemble [N_eval, T, hidden_size] shifted for pack_padded_sequence
        seq_eval = spatial_embeddings[:, eval_indices].permute(1, 0, 2)  # [N_eval, T, hidden_size]
        masks_eval = item.node_mask[:, eval_indices].permute(1, 0).to(device)  # [N_eval, T]
        lengths = masks_eval.sum(dim=1).long().clamp(min=1)

        # Shift valid steps left
        seq_shifted = torch.zeros_like(seq_eval)
        for i in range(N_eval):
            vlen = lengths[i].item()
            seq_shifted[i, :vlen] = seq_eval[i, T - vlen:]

        packed_in = pack_padded_sequence(
            seq_shifted,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.temporal_gru(packed_in)
        h_top = h_n[-1]  # [N_eval, temporal_hidden]

        logits = self.head(h_top).squeeze(-1)
        y_eval = item.y_end[eval_indices].to(device)
        loss_mask = item.anomaly_mask[eval_indices].to(device)

        return logits, y_eval, loss_mask

    forward_single_window = forward_window

    def save(self, run_dir: Path, threshold: float = 0.5) -> None:
        """Save PyTorch checkpoint and model configuration."""
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), run_dir / "best_checkpoint.pt")

        with open(run_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "in_channels": self.in_channels,
                "hidden_size": self.hidden_size,
                "edge_dim": self.edge_dim,
                "global_dim": self.global_dim,
                "temporal_hidden": self.temporal_hidden,
                "threshold": float(threshold),
            }, f, indent=2)

        with open(run_dir / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump({
                "node_features": BASE_FEATURES_24,
                "edge_features": EDGE_FEATURES_7,
                "global_features": GLOBAL_FEATURES_6,
            }, f, indent=2)

    @classmethod
    def load(cls, run_dir: Path, device: str = "cpu") -> tuple[SwarmSTGAT, float]:
        """Load trained ST-GAT model and threshold from disk."""
        with open(run_dir / "model_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(
            in_channels=config["in_channels"],
            hidden_size=config["hidden_size"],
            edge_dim=config["edge_dim"],
            global_dim=config["global_dim"],
            temporal_hidden=config["temporal_hidden"],
        )
        state_dict = torch.load(run_dir / "best_checkpoint.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        threshold = float(config.get("threshold", 0.5))
        return model, threshold

