"""LightGBM tabular ensemble wrapper.

Handles training, early stopping on validation PR-AUC, checkpointing, and prediction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import lightgbm as lgb
import numpy as np

from src.config import LGBM_CONFIG, LGBM_FEATURES_52, LGBM_FEATURES_54


class SwarmLightGBM:
    """Wrapper around LightGBMClassifier for 52-feature swarm criticality identification."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or LGBM_CONFIG.copy()
        self.feature_names = LGBM_FEATURES_52.copy()
        self.clf: lgb.LGBMClassifier | None = None
        self.booster: lgb.Booster | None = None
        self.best_iteration: int = 0
        self.threshold: float = 0.5

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> SwarmLightGBM:
        """Train LightGBM with validation early stopping on PR-AUC (average_precision)."""
        early_stopping_rounds = self.config.get("early_stopping_rounds", 50)
        params = {k: v for k, v in self.config.items() if k != "early_stopping_rounds"}

        self.clf = lgb.LGBMClassifier(**params, n_jobs=-1, verbose=-1)

        self.clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        self.best_iteration = int(self.clf.best_iteration_ or self.clf.n_estimators)
        self.booster = self.clf.booster_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return positive class probability array [N]."""
        if self.booster is not None:
            return self.booster.predict(X).astype(np.float32)
        elif self.clf is not None:
            return self.clf.predict_proba(X)[:, 1].astype(np.float32)
        raise RuntimeError("Model has not been trained or loaded")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary prediction using operating threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(np.int8)

    def save(self, run_dir: Path) -> None:
        """Save model checkpoint, schema, and config to directory."""
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.clf is None and self.booster is None:
            raise RuntimeError("Cannot save untrained model")

        # 1. Booster checkpoint
        booster = self.booster or self.clf.booster_
        booster.save_model(str(run_dir / "best_checkpoint.txt"))

        # 2. Config & metadata
        with open(run_dir / "model_config.json", "w", encoding="utf-8") as f:
            json.dump({
                "config": self.config,
                "best_iteration": self.best_iteration,
                "threshold": self.threshold,
            }, f, indent=2)

        # 3. Feature schema
        with open(run_dir / "feature_schema.json", "w", encoding="utf-8") as f:
            json.dump({"features": self.feature_names}, f, indent=2)

        # 4. Preprocessing schema
        with open(run_dir / "preprocessing.json", "w", encoding="utf-8") as f:
            json.dump({"scaling": "none", "n_features": len(self.feature_names)}, f, indent=2)

    @classmethod
    def load(cls, run_dir: Path) -> SwarmLightGBM:
        """Load trained booster and config from disk."""
        model = cls()
        with open(run_dir / "model_config.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        model.config = meta["config"]
        model.best_iteration = meta["best_iteration"]
        model.threshold = meta.get("threshold", 0.5)

        with open(run_dir / "feature_schema.json", "r", encoding="utf-8") as f:
            model.feature_names = json.load(f)["features"]

        model.booster = lgb.Booster(model_file=str(run_dir / "best_checkpoint.txt"))
        return model

