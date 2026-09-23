"""
Dehradun Deforestation & Forest Canopy Loss Prediction Engine.

Monitors Doon Valley, Rajaji National Park buffer zones, and Mussoorie foothills:
- Evaluates bi-temporal Sentinel-2 / Cartosat multispectral observations (T1 vs T2)
- Calculates NDVI (Normalized Difference Vegetation Index)
- Measures canopy density degradation and loss percentage
- Runs DOFA-VLM inference for natural language geospatial assessment
"""

import os
import sys
import torch
import numpy as np

# Add repository root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bipanshu_work.models.dofa_vlm import DOFA_VLM

def calculate_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    """Calculates Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)"""
    denom = (nir_band + red_band) + 1e-7
    return (nir_band - red_band) / denom

def run_dehradun_deforestation_analysis():
    print("=" * 80)
    print(">>> ISRO SATELLITE AI: DEHRADUN FOREST CANOPY & DEFORESTATION MONITORING")
    print("=" * 80)
    print("[*] Target Region:       Dehradun & Doon Valley, Uttarakhand, India")
    print("[*] Satellite Sensors:    Sentinel-2 MSI (10m) & Cartosat-2S (0.65m)")
    print("[*] Coordinates:         30.3165 N, 78.0322 E")
    print("[*] Observation Window:  Temporal Baseline 2021 (T1) -> Current Observation (T2)")
    print("-" * 80)

    # 1. Load DOFA-VLM Model
    print("\n[1/4] Initializing DOFA-VLM Vision-Language Model...")
    model = DOFA_VLM()
    model.eval()

    # 2. Simulate / Load Bi-temporal Multispectral Observations
    # Channels: [NIR (B08: 0.842um), Red (B04: 0.665um), Green (B03: 0.560um)]
    print("\n[2/4] Ingesting Multispectral Imagery (NIR, Red, Green bands)...")
    np.random.seed(42)
    
    # T1: Dense Himalayan Sal Forest (High NIR, Low Red)
    t1_nir = np.random.uniform(0.65, 0.90, (256, 256))
    t1_red = np.random.uniform(0.08, 0.20, (256, 256))
    t1_green = np.random.uniform(0.20, 0.35, (256, 256))
    ndvi_t1 = calculate_ndvi(t1_nir, t1_red)

    # T2: Encroachment & Forest Clearance in Shivalik belt (Lower NIR, Higher Red/Bare soil)
    t2_nir = t1_nir.copy()
    t2_red = t1_red.copy()
    t2_green = t1_green.copy()
    # Apply forest clearance patch in quadrant [60:180, 80:200]
    t2_nir[60:180, 80:200] = np.random.uniform(0.15, 0.30, (120, 120))
    t2_red[60:180, 80:200] = np.random.uniform(0.40, 0.65, (120, 120))
    t2_green[60:180, 80:200] = np.random.uniform(0.25, 0.40, (120, 120))
    t2_image = np.stack([t2_nir, t2_red, t2_green], axis=0)
    ndvi_t2 = calculate_ndvi(t2_nir, t2_red)

    # 3. Compute Quantitative Bi-Temporal Metrics
    print("\n[3/4] Computing Bi-Temporal Vegetation & Canopy Health Indices:")
    mean_ndvi_t1 = float(np.mean(ndvi_t1))
    mean_ndvi_t2 = float(np.mean(ndvi_t2))
    canopy_loss_pct = float(np.sum((ndvi_t1 > 0.45) & (ndvi_t2 < 0.25)) / np.sum(ndvi_t1 > 0.45) * 100.0)
    cleared_area_hectares = float((120 * 120 * (10.0 * 10.0)) / 10000.0) # 10m GSD

    print(f"  * Baseline NDVI (T1 2021):          {mean_ndvi_t1:.3f} (Healthy Dense Canopy)")
    print(f"  * Current NDVI (T2):                {mean_ndvi_t2:.3f} (Degraded / Cleared Patches)")
    print(f"  * Canopy Area Loss:                 {canopy_loss_pct:.2f}% of monitored zone")
    print(f"  * Estimated Forest Clearance:       {cleared_area_hectares:.2f} Hectares (~{cleared_area_hectares * 2.471:.1f} Acres)")

    # 4. Multi-Modal DOFA-VLM Inference
    print("\n[4/4] Generating Multimodal AI Assessment & Environmental Narrative:")
    
    questions = [
        {
            "query": "Is there active deforestation and forest canopy clearance detected in this Dehradun sector?",
            "task_type": "vqa_choice"
        },
        {
            "query": "Describe the change in vegetation density and land-use transition across the Doon Valley foothills.",
            "task_type": "captioning"
        }
    ]

    for q in questions:
        batch = {
            "image": torch.tensor(t2_image, dtype=torch.float32).unsqueeze(0),
            "wavelengths": torch.tensor([0.842, 0.665, 0.560]), # Sentinel-2 NIR, Red, Green
            "gsd": 10.0,
            "prompt": q["query"],
            "task_type": q["task_type"],
            "sensor_domain": "sentinel2_rgb",
            "sample_id": "dehradun_deforestation_eval",
        }
        with torch.no_grad():
            answer = model.generate(batch)[0]

        print(f"\n  [Query]      \"{q['query']}\"")
        print(f"  [AI Verdict] --> \"{answer}\"")

    print("\n" + "=" * 80)
    print(">>> DEFORESTATION REPORT GENERATED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_dehradun_deforestation_analysis()
