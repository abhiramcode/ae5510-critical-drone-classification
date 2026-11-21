"""3-layer GATv2 spatial graph neural network with edge features and global fusion.

Uses GATv2 message passing with edge embeddings, LayerNorm residuals, and
broadcast global swarm context re-concatenated at each block.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from src.config import GAT_CONFIG, BASE_FEATURES_22, BASE_FEATURES_24, EDGE_FEATURES_7, GLOBAL_FEATURES_6


class SwarmGATv2(nn.Module):
    """2-layer GATv2 spatial graph neural network with edge features."""

    def __init__(
        self,
        in_channels: int = 22,
        hidden_size: int = 64,
        edge_dim: int = 7,
        num_heads: int = 4,
        num_layers: int = 2,
        global_dim: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.global_dim = global_dim

        self.node_in = nn.Linear(in_channels, hidden_size)
        self.c1 = GATv2Conv(
            hidden_size,
            hidden_size,
            heads=num_heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.c2 = GATv2Conv(
            hidden_size,
            hidden_size,
            heads=num_heads,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.n1 = nn.LayerNorm(hidden_size)
        self.n2 = nn.LayerNorm(hidden_size)

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def extract_spatial_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        global_feat: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run node embeddings and GATv2 layers to produce spatial representations [N, hidden]."""
        h = self.node_in(x)
        h = h + F.gelu(self.n1(self.c1(h, edge_index, edge_attr)))
        h = h + F.gelu(self.n2(self.c2(h, edge_index, edge_attr)))
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        global_feat: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass predicting per-node logits [N]."""
        h = self.extract_spatial_embeddings(x, edge_index, edge_attr, global_feat, batch)
        logits = self.head(h).squeeze(-1)
        return logits

    def save(self, run_dir: Path, threshold: float = 0.5) -> None:
        """Save PyTorch checkpoint and model configuration."""
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), run_dir / "best_checkpoint.pt")

        with open(run_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "in_channels": self.in_channels,
                "hidden_size": self.hidden_size,
                "edge_dim": self.edge_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "global_dim": self.global_dim,
                "threshold": float(threshold),
            }, f, indent=2)

        with open(run_dir / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump({
                "node_features": BASE_FEATURES_24,
                "edge_features": EDGE_FEATURES_7,
                "global_features": GLOBAL_FEATURES_6,
            }, f, indent=2)

    @classmethod
    def load(cls, run_dir: Path, device: str = "cpu") -> tuple[SwarmGATv2, float]:
        """Load trained GATv2 model and threshold from disk."""
        with open(run_dir / "model_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(
            in_channels=config["in_channels"],
            hidden_size=config["hidden_size"],
            edge_dim=config["edge_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            global_dim=config["global_dim"],
        )
        state_dict = torch.load(run_dir / "best_checkpoint.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        threshold = float(config.get("threshold", 0.5))
        return model, threshold

