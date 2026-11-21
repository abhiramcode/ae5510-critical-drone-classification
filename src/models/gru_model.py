"""2-layer unidirectional GRU sequence model.

Supports both left-padded batch window execution (via pack_padded_sequence)
and O(1) step-by-step streaming inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.config import GRU_CONFIG, BASE_FEATURES_22, BASE_FEATURES_24


class SwarmGRU(nn.Module):
    """2-layer GRU sequence classifier for aerial swarm critical node identification."""

    def __init__(
        self,
        input_size: int = 22,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Process left-padded batch [B, W, 24].

        Returns:
            logits: [B]
        """
        out, _ = self.gru(x)
        h_top = out[:, -1]
        logits = self.head(h_top).squeeze(-1)
        return logits

    def step_streaming(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform O(1) single-step streaming inference for alive agents at step t.

        Args:
            x_t: [N_alive, 24]
            h_prev: [2, N_alive, 96] or None

        Returns:
            logits: [N_alive]
            h_next: [2, N_alive, 96]
        """
        if x_t.dim() == 2:
            x_in = x_t.unsqueeze(1)  # [N, 1, 24]
        else:
            x_in = x_t

        if h_prev is None:
            out, h_next = self.gru(x_in)
        else:
            out, h_next = self.gru(x_in, h_prev)

        logits = self.head(out.squeeze(1)).squeeze(-1)
        return logits, h_next

    def save(self, run_dir: Path, threshold: float = 0.5) -> None:
        """Save PyTorch checkpoint and model configuration."""
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), run_dir / "best_checkpoint.pt")

        with open(run_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "threshold": float(threshold),
            }, f, indent=2)

        with open(run_dir / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump({"features": BASE_FEATURES_24}, f, indent=2)

    @classmethod
    def load(cls, run_dir: Path, device: str = "cpu") -> tuple[SwarmGRU, float]:
        """Load trained GRU model and threshold from disk."""
        with open(run_dir / "model_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
        )
        state_dict = torch.load(run_dir / "best_checkpoint.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        threshold = float(config.get("threshold", 0.5))
        return model, threshold
