"""Shared PyTorch training loop with early stopping on validation PR-AUC.

Used across GRU, GATv2, and ST-GAT models for standardized training,
learning rate scheduling, loss masking, and checkpoint preservation.
"""
from __future__ import annotations

import time
from typing import Callable, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


class PyTorchTrainer:
    """Manages training epochs, early stopping, and evaluation for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Any = None,
        patience: int = 5,
        clip_grad_norm: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.patience = patience
        self.clip_grad_norm = clip_grad_norm
        self.device = device

        self.best_score: float = -1.0
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_epoch: int = 0
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        train_fn: Callable[[], float],
        eval_fn: Callable[[], tuple[np.ndarray, np.ndarray]],
        max_epochs: int = 20,
        desc: str = "Model",
    ) -> dict[str, Any]:
        """Run training loop over max_epochs with early stopping on validation PR-AUC."""
        epochs_no_improve = 0
        t0 = time.time()

        for epoch in range(1, max_epochs + 1):
            ep_start = time.time()
            train_loss = train_fn()

            if self.scheduler is not None:
                self.scheduler.step()

            # Evaluate on validation set
            y_val, p_val = eval_fn()
            val_pr_auc = float(average_precision_score(y_val, p_val))
            val_roc_auc = float(roc_auc_score(y_val, p_val))

            ep_time = time.time() - ep_start
            print(
                f"[{desc} Epoch {epoch:2d}/{max_epochs:2d}] "
                f"Loss: {train_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f} | "
                f"ROC-AUC: {val_roc_auc:.4f} ({ep_time:.1f}s)",
                flush=True,
            )

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "epoch_time_sec": ep_time,
            }
            self.history.append(record)

            if val_pr_auc > self.best_score:
                self.best_score = val_pr_auc
                self.best_epoch = epoch
                self.best_state_dict = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"[{desc}] Early stopping triggered at epoch {epoch} "
                        f"(Best PR-AUC: {self.best_score:.4f} at epoch {self.best_epoch})",
                        flush=True,
                    )
                    break

        total_time = time.time() - t0
        print(f"[{desc}] Training completed in {total_time:.1f}s. Best Epoch: {self.best_epoch}", flush=True)

        # Restore best model state
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return {
            "best_epoch": self.best_epoch,
            "best_val_pr_auc": self.best_score,
            "total_time_sec": total_time,
            "epochs_trained": len(self.history),
            "history": self.history,
        }
