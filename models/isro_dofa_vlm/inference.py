"""
Inference script for the Pretrained ISRO DOFA-VLM.

Demonstrates inference across:
1. Sentinel-2 Multispectral & RGB (RSVQA-LR / EuroSAT)
2. Cartosat-2S High-Resolution Optical (0.65m VNIR)
3. RISAT-1A Synthetic Aperture Radar (SAR)
4. VRSBench & GeoChat Visual Reasoning
5. BigEarthNet Corine Land Cover Multi-label Classification
"""

import os
import sys
import torch
import numpy as np

# Add repo to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
from load_model import load_pretrained_dofa_vlm

def run_demo():
    print("=" * 80)
    print(">>> PRETRAINED ISRO DOFA-VLM INFERENCE RUNNER")
    print("=" * 80)

    # 1. Load Pretrained Model
    model = load_pretrained_dofa_vlm()

    test_queries = [
        {
            "dataset": "RSVQA-LR (Sentinel-2)",
            "sensor": "sentinel2_rgb",
            "wavelengths": torch.tensor([0.665, 0.560, 0.490]),
            "gsd": 10.0,
            "image": torch.tensor(np.random.uniform(0.1, 0.8, (3, 256, 256)), dtype=torch.float32),
            "prompt": "Is there presence of residential buildings and road networks in this Sentinel-2 scene?",
            "task_type": "vqa_choice",
            "sensor_domain": "rsvqa_lr",
            "sample_id": "rsvqa_inf_01"
        },
        {
            "dataset": "ISRO Bhoonidhi Cartosat-2S (0.65m)",
            "sensor": "cartosat_optical",
            "wavelengths": torch.tensor([0.650, 0.485, 0.560, 0.660, 0.825]),
            "gsd": 0.65,
            "image": torch.tensor(np.random.uniform(0.1, 0.9, (5, 256, 256)), dtype=torch.float32),
            "prompt": "Identify high-density commercial infrastructure and primary road arteries in this 0.65m optical image.",
            "task_type": "vqa_choice",
            "sensor_domain": "cartosat_optical",
            "sample_id": "cartosat_inf_01"
        },
        {
            "dataset": "ISRO Bhoonidhi RISAT-1A SAR (C-band)",
            "sensor": "risat_sar",
            "wavelengths": torch.tensor([55500.0, 55500.0]),
            "gsd": 1.0,
            "image": torch.tensor(np.random.uniform(0.05, 0.95, (2, 256, 256)), dtype=torch.float32),
            "prompt": "Does the dual-polarization radar backscatter suggest flood inundation or standing water?",
            "task_type": "vqa_choice",
            "sensor_domain": "risat_sar",
            "sample_id": "risat_inf_01"
        },
        {
            "dataset": "BigEarthNet (12-Band Multispectral)",
            "sensor": "sentinel2_msi",
            "wavelengths": torch.tensor([0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]),
            "gsd": 10.0,
            "image": torch.tensor(np.random.uniform(0.1, 0.9, (12, 128, 128)), dtype=torch.float32),
            "prompt": "Classify all Corine Land Cover surface types present in this 12-band scene.",
            "task_type": "classification",
            "sensor_domain": "bigearthnet",
            "sample_id": "ben_inf_01"
        },
        {
            "dataset": "VRSBench / GeoChat",
            "sensor": "vrsbench_optical",
            "wavelengths": torch.tensor([0.650, 0.540, 0.470]),
            "gsd": 0.5,
            "image": torch.tensor(np.random.uniform(0.2, 0.8, (3, 256, 256)), dtype=torch.float32),
            "prompt": "Provide a comprehensive description of the terrain, land use, and industrial complexes in this scene.",
            "task_type": "captioning",
            "sensor_domain": "vrsbench_optical",
            "sample_id": "vrs_inf_01"
        }
    ]

    for q in test_queries:
        batch = {
            "image": q["image"].unsqueeze(0),
            "wavelengths": q["wavelengths"],
            "gsd": q["gsd"],
            "prompt": q["prompt"],
            "task_type": q["task_type"],
            "sensor_domain": q["sensor_domain"],
            "sample_id": q["sample_id"]
        }
        with torch.no_grad():
            pred = model.generate(batch)[0]

        print(f"\n[Test Benchmark] {q['dataset']}")
        print(f"  * Sensor / GSD:   {q['sensor']} ({q['gsd']}m)")
        print(f"  * Question:       \"{q['prompt']}\"")
        print(f"  * Model Response: --> \"{pred}\"")

    print("\n" + "=" * 80)
    print(">>> INFERENCE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
