"""Master training runner for GATv2 and ST-GAT, cross-model comparison, and live inference demo."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.train_gat import train_gat_pipeline
from src.training.train_stgat import train_stgat_pipeline
from src.evaluation.summarize import generate_comparison_summary
from src.tests.test_sanity import test_sc5_telemetry_derivation_and_inference_parity
from src.inference.demo_sample_inference import run_sample_inference


def main():
    t_start = time.time()
    print("=" * 70)
    print("CRITICAL NODE IDENTIFICATION: GATv2 & ST-GAT RETRAINING & TOURNAMENT")
    print("=" * 70)

    # 1. Train Model 3: GATv2 Spatial Graph Neural Network
    print("\n>>> [1/4] Running Model 3: GATv2 Spatial GNN Training...")
    gat_res = train_gat_pipeline()

    # 2. Train Model 4: ST-GAT Spatio-Temporal Graph Neural Network
    print("\n>>> [2/4] Running Model 4: ST-GAT Spatio-Temporal GNN Training...")
    stgat_res = train_stgat_pipeline()

    # 3. Generate Cross-Model Comparison Table
    print("\n>>> [3/4] Generating Cross-Model Comparison Tournament Summary...")
    summary_df = generate_comparison_summary()

    # 4. Verify Sanity Check SC-5 (Inference Parity & Telemetry Derivation)
    print("\n>>> [4/4] Executing Sanity Check SC-5 (Pure Telemetry Inference Parity)...")
    test_sc5_telemetry_derivation_and_inference_parity()
    print("Sanity Check SC-5 PASSED!")

    # 5. Run Live Inference Demonstration from Raw Telemetry
    print("\n>>> Executing Live Sample Inference Demonstration (Px, Py, Pz, Vx, Vy, Vz)...")
    run_sample_inference(use_dataset_sample=True)

    print(f"\nAll operations completed successfully in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

