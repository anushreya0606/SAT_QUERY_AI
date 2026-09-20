import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bipanshu_work")))
from models.dofa_vlm import DOFA_VLM

model = DOFA_VLM()
model.eval()

print("=" * 80)
print(">>> REAL-TIME SATELLITE VLM PREDICTION DEMO")
print("=" * 80)

test_scenarios = [
    {
        "title": "1. Reforestation & Vegetation Canopy Recovery",
        "sensor": "sentinel2_rgb",
        "wavelengths": torch.tensor([0.665, 0.560, 0.490]),
        "gsd": 10.0,
        "image": torch.tensor(np.random.uniform(0.1, 0.7, (3, 256, 256)), dtype=torch.float32),
        "prompt": "Does this observation indicate active vegetation, cropland, and canopy coverage?",
        "task_type": "vqa_choice",
        "sensor_domain": "rsvqa_lr",
        "sample_id": "rsvqa_reforestation_01",
    },
    {
        "title": "2. High-Resolution Urban & Road Infrastructure (Cartosat-2S 0.65m)",
        "sensor": "cartosat_optical",
        "wavelengths": torch.tensor([0.650, 0.485, 0.560, 0.660, 0.825]),
        "gsd": 0.65,
        "image": torch.tensor(np.random.uniform(0.1, 0.8, (5, 256, 256)), dtype=torch.float32),
        "prompt": "Is there a dense urban settlement with paved arterial roads visible in this 0.65m Cartosat-2S scene?",
        "task_type": "vqa_choice",
        "sensor_domain": "cartosat_optical",
        "sample_id": "cartosat_urban_01",
    },
    {
        "title": "3. All-Weather Radar SAR Double-Bounce Reflection (RISAT-1A SAR)",
        "sensor": "risat_sar",
        "wavelengths": torch.tensor([55500.0, 55500.0]),
        "gsd": 1.0,
        "image": torch.tensor(np.random.uniform(0.05, 0.95, (2, 256, 256)), dtype=torch.float32),
        "prompt": "Is there strong double-bounce radar backscatter indicating metallic structures in this C-band RISAT SAR image?",
        "task_type": "vqa_choice",
        "sensor_domain": "risat_sar",
        "sample_id": "risat_sar_01",
    },
    {
        "title": "4. Multi-Spectral 12-Band Land Cover Classification (BigEarthNet)",
        "sensor": "sentinel2_msi",
        "wavelengths": torch.tensor([0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190]),
        "gsd": 10.0,
        "image": torch.tensor(np.random.uniform(0.1, 0.9, (12, 128, 128)), dtype=torch.float32),
        "prompt": "Identify all Corine Land Cover classes present in this multi-spectral Sentinel observation.",
        "task_type": "classification",
        "sensor_domain": "bigearthnet",
        "sample_id": "ben_01",
    },
    {
        "title": "5. Automated Satellite Scene Captioning & Environmental Narrative",
        "sensor": "vrsbench_optical",
        "wavelengths": torch.tensor([0.650, 0.540, 0.470]),
        "gsd": 0.5,
        "image": torch.tensor(np.random.uniform(0.2, 0.8, (3, 256, 256)), dtype=torch.float32),
        "prompt": "Describe the detailed spatial layout, infrastructure, and land cover in this satellite image.",
        "task_type": "captioning",
        "sensor_domain": "vrsbench_optical",
        "sample_id": "vrsbench_cap_01",
    },
]

for sc in test_scenarios:
    batch = {
        "image": sc["image"].unsqueeze(0),
        "wavelengths": sc["wavelengths"],
        "gsd": sc["gsd"],
        "prompt": sc["prompt"],
        "task_type": sc["task_type"],
        "sensor_domain": sc["sensor_domain"],
        "sample_id": sc["sample_id"],
    }
    pred = model.generate(batch)[0]
    print(f"\n[Test Case] {sc['title']}")
    print(f"  * Sensor / GSD:     {sc['sensor']} ({sc['gsd']}m)")
    print(f"  * Input Question:   \"{sc['prompt']}\"")
    print(f"  * VLM Output:       --> \"{pred}\"")

print("\n" + "=" * 80)
print(">>> ALL PREDICTIONS GENERATED SUCCESSFULLY")
print("=" * 80)
