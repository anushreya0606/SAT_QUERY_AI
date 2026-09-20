# Pretrained ISRO DOFA-VLM Model Package

Local pretrained checkpoint and standalone toolkit for Sensor-Agnostic Vision-Language understanding across satellite datasets.

---

## 🛰️ Architecture & Capabilities

The **ISRO DOFA-VLM** model combines:
1. **DOFA Dynamic Wavelength & GSD ViT Encoder**: Ingests variable spectral bands (optical, multispectral, and SAR) and spatial resolutions dynamically via physical wavelength embeddings (µm) and GSD embeddings (meters).
2. **Multimodal Projector**: Projects 768-dim spatial-spectral tokens into a 1024-dim multimodal embedding space.
3. **Multi-Task Decoders**: Generates answers for VQA, multi-label classifications for Corine Land Cover / EuroSAT, and scene narratives for Captioning.

---

## 📦 Contents of this Folder

| File | Description |
| :--- | :--- |
| [`model_weights.pt`](file:///d:/BENCHMARKS/pretrained_models/isro_dofa_vlm/model_weights.pt) | Serialized PyTorch state dictionary checkpoint (556 MB). |
| [`config.json`](file:///d:/BENCHMARKS/pretrained_models/isro_dofa_vlm/config.json) | Architecture dimensions, sensor wavelength specs, and hyperparameters. |
| [`vocab.json`](file:///d:/BENCHMARKS/pretrained_models/isro_dofa_vlm/vocab.json) | Vocabulary tokens, EuroSAT classes, and BigEarthNet 19 Corine Land Cover categories. |
| [`dataset_manifest.json`](file:///d:/BENCHMARKS/pretrained_models/isro_dofa_vlm/dataset_manifest.json) | Metadata of present datasets (RSVQA, VRSBench, EuroSAT, GeoChat, Cartosat, RISAT). |
| [`load_model.py`](file:///d:/BENCHMARKS/pretrained_models/isro_dofa_vlm/load_model.py) | 1-line Python loader helper function (`load_pretrained_dofa_vlm`). |
| [`inference.py`](file:///d:/BENCHMARKS/pretrained_models/isro_dofa_vlm/inference.py) | Standalone inference script running test cases across all satellite sensors. |
| [`train_present_datasets.py`](file:///d:/BENCHMARKS/pretrained_models/isro_dofa_vlm/train_present_datasets.py) | Complete pretraining / fine-tuning pipeline to re-train weights on local data. |

---

## 🚀 Quickstart: Loading and Inference

### 1. In Python:
```python
from pretrained_models.isro_dofa_vlm.load_model import load_pretrained_dofa_vlm
import torch

# Load the pretrained model
model = load_pretrained_dofa_vlm()

# Prepare sample batch
batch = {
    "image": torch.randn(1, 3, 256, 256),            # [B, C, H, W]
    "wavelengths": torch.tensor([0.665, 0.560, 0.490]), # Sentinel-2 RGB wavelengths (µm)
    "gsd": 10.0,                                      # 10m Ground Sampling Distance
    "prompt": "Is there presence of residential buildings in this scene?",
    "task_type": "vqa_choice",
    "sensor_domain": "sentinel2_rgb",
}

# Run generation
prediction = model.generate(batch)[0]
print(f"Model prediction: {prediction}")
```

### 2. Run CLI Demo:
```bash
python pretrained_models/isro_dofa_vlm/inference.py
```

---

## 🔒 Git Protection
This folder is added to `.gitignore` so large weight files and local cache checkpoints stay local and are never pushed to git.
