"""Scenario-grouped deterministic data splitting.

Produces a deterministic 70% train / 15% validation / 15% test scenario partition,
ensuring all 9 swarm sizes and all 4 attrition conditions are represented in each split
without scenario leakage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict
import numpy as np
import pandas as pd

from src.config import SPLITS_DIR, RANDOM_SEED


class SplitSchema(TypedDict):
    train: list[str]
    val: list[str]
    test: list[str]


def compute_deterministic_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> SplitSchema:
    """Compute an exact 70/15/15 scenario partition stratified across size and attrition."""
    meta = (
        df.groupby("case_id", observed=True)
        .agg(
            swarm_size=("num_agents", "first"),
            attr=("attrition_rate", "first"),
            pos_rate=("is_critical", "mean"),
            has_promo=("was_promoted", "any"),
        )
        .reset_index()
    )

    quota_te: dict[int, int] = {18: 2, 24: 2, 30: 2, 37: 1, 45: 1, 52: 1, 60: 2, 68: 2, 75: 2}
    quota_va: dict[int, int] = {18: 2, 24: 2, 30: 2, 37: 1, 45: 2, 52: 2, 60: 1, 68: 1, 75: 2}

    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    base_attrs: list[str] = ["none", "light", "medium", "heavy"]
    for idx, sz in enumerate(sorted(meta["swarm_size"].unique())):
        grp = meta[meta["swarm_size"] == sz].copy()
        rotated_attrs = base_attrs[idx % 4:] + base_attrs[:idx % 4]
        sub_ids: list[list[str]] = []
        for attr in rotated_attrs:
            a_grp = grp[grp["attr"] == attr]
            ids = a_grp["case_id"].astype(str).tolist()
            rng.shuffle(ids)
            sub_ids.append(ids)

        ordered: list[str] = []
        while any(len(x) > 0 for x in sub_ids):
            for x in sub_ids:
                if x:
                    ordered.append(x.pop(0))

        nt = quota_te[sz]
        nv = quota_va[sz]
        test.extend(ordered[:nt])
        val.extend(ordered[nt:nt + nv])
        train.extend(ordered[nt + nv:])

    # Validation assertions
    assert len(train) == 70, f"Expected 70 train scenarios, got {len(train)}"
    assert len(val) == 15, f"Expected 15 val scenarios, got {len(val)}"
    assert len(test) == 15, f"Expected 15 test scenarios, got {len(test)}"
    assert not (set(train) & set(val)), "Leakage detected between train and val"
    assert not (set(train) & set(test)), "Leakage detected between train and test"
    assert not (set(val) & set(test)), "Leakage detected between val and test"

    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def get_or_create_primary_split(
    df: pd.DataFrame,
    out_path: Path = SPLITS_DIR / "primary_split.json",
    seed: int = RANDOM_SEED,
) -> SplitSchema:
    """Retrieve frozen primary split from disk or create and persist it if absent."""
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"train": data["train"], "val": data["val"], "test": data["test"]}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    split = compute_deterministic_split(df, seed=seed)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    return split

