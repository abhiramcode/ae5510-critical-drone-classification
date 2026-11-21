"""Train-only feature standardizer for neural models.

Fits mean and standard deviation strictly on the training partition and persists
the statistics for exact reproducibility across training and inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import torch


class SwarmFeatureScaler:
    """Standardizes node, edge, and global feature tensors with train-only parameters."""

    def __init__(self) -> None:
        self.node_mean: np.ndarray | None = None
        self.node_std: np.ndarray | None = None
        self.edge_mean: np.ndarray | None = None
        self.edge_std: np.ndarray | None = None
        self.global_mean: np.ndarray | None = None
        self.global_std: np.ndarray | None = None

    def fit_nodes(self, x: np.ndarray) -> SwarmFeatureScaler:
        """Fit standardization parameters on 2D node features array [N, 24]."""
        self.node_mean = np.mean(x, axis=0).astype(np.float32)
        std = np.std(x, axis=0).astype(np.float32)
        self.node_std = np.maximum(std, 1e-6)
        return self

    def fit_edges(self, edge_attrs: np.ndarray) -> SwarmFeatureScaler:
        """Edge attributes (normalized distance, unit bearings, relative velocities) are preserved in physical scale."""
        self.edge_mean = np.zeros(7, dtype=np.float32)
        self.edge_std = np.ones(7, dtype=np.float32)
        return self

    def fit_globals(self, global_feats: np.ndarray) -> SwarmFeatureScaler:
        """Fit standardization parameters on 2D global features array [B, 6]."""
        self.global_mean = np.mean(global_feats, axis=0).astype(np.float32)
        std = np.std(global_feats, axis=0).astype(np.float32)
        self.global_std = np.maximum(std, 1e-6)
        return self

    def transform_nodes(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Transform node features using fitted parameters."""
        if self.node_mean is None or self.node_std is None:
            raise RuntimeError("Scaler must be fitted before transforming node features")
        if isinstance(x, torch.Tensor):
            mu = torch.from_numpy(self.node_mean).to(x.device, x.dtype)
            sd = torch.from_numpy(self.node_std).to(x.device, x.dtype)
            return (x - mu) / sd
        return (x - self.node_mean) / self.node_std

    def transform_edges(self, ea: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Edge attributes are naturally bounded in physical units ([0, 1] and [-1, 1]); preserved without z-scoring."""
        return ea
        if isinstance(ea, torch.Tensor):
            mu = torch.from_numpy(self.edge_mean).to(ea.device, ea.dtype)
            sd = torch.from_numpy(self.edge_std).to(ea.device, ea.dtype)
            return (ea - mu) / sd
        return (ea - self.edge_mean) / self.edge_std

    def transform_globals(self, g: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Transform global features using fitted parameters."""
        if self.global_mean is None or self.global_std is None:
            raise RuntimeError("Scaler must be fitted before transforming global features")
        if isinstance(g, torch.Tensor):
            mu = torch.from_numpy(self.global_mean).to(g.device, g.dtype)
            sd = torch.from_numpy(self.global_std).to(g.device, g.dtype)
            return (g - mu) / sd
        return (g - self.global_mean) / self.global_std

    def to_dict(self) -> dict[str, list[float]]:
        """Serialize scaler statistics to dictionary."""
        return {
            "node_mean": self.node_mean.tolist() if self.node_mean is not None else [],
            "node_std": self.node_std.tolist() if self.node_std is not None else [],
            "edge_mean": self.edge_mean.tolist() if self.edge_mean is not None else [],
            "edge_std": self.edge_std.tolist() if self.edge_std is not None else [],
            "global_mean": self.global_mean.tolist() if self.global_mean is not None else [],
            "global_std": self.global_std.tolist() if self.global_std is not None else [],
        }

    def save(self, path: Path) -> None:
        """Save scaler parameters to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SwarmFeatureScaler:
        """Load scaler parameters from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls()
        if data.get("scaling") == "none" or "node_mean" not in data:
            return scaler
        scaler.node_mean = np.array(data["node_mean"], dtype=np.float32) if data["node_mean"] else None
        scaler.node_std = np.array(data["node_std"], dtype=np.float32) if data["node_std"] else None
        scaler.edge_mean = np.array(data["edge_mean"], dtype=np.float32) if data["edge_mean"] else None
        scaler.edge_std = np.array(data["edge_std"], dtype=np.float32) if data["edge_std"] else None
        scaler.global_mean = np.array(data["global_mean"], dtype=np.float32) if data["global_mean"] else None
        scaler.global_std = np.array(data["global_std"], dtype=np.float32) if data["global_std"] else None
        return scaler

