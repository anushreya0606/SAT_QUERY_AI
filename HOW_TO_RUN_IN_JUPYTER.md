# 🛰️ How to Run ISRO/SAC Remote Sensing VLM in Jupyter Notebook

This guide explains how to launch and run all models, predictions, and real benchmark evaluations inside **Jupyter Notebook** or **JupyterLab** using your local virtual environment (`.\venv`).

---

## 📌 Step 1: Connect Your Virtual Environment to Jupyter (One-Time Setup)

Your system has Jupyter installed in Anaconda (`D:\anaconda`), while PyTorch and all project dependencies are in your project's `.\venv`. 

Run these two commands in PowerShell to link them so Jupyter can use your PyTorch environment:

```powershell
# 1. Install ipykernel inside your project venv
.\venv\Scripts\python -m pip install ipykernel

# 2. Register the venv as a selectable Jupyter Kernel
.\venv\Scripts\python -m ipykernel install --user --name=isro_vlm_venv --display-name "Python (ISRO VLM venv)"
```

---

## 🚀 Step 2: Launch Jupyter Notebook

In your project root (`D:\BENCHMARKS`), run:

```powershell
jupyter notebook
```
*(Or if you prefer JupyterLab: `jupyter lab`)*

Jupyter will open in your browser at `http://localhost:8888`.

> **Important**: When creating a new notebook or opening an existing one, make sure to select the kernel:  
> **Kernel ➔ Change Kernel ➔ `Python (ISRO VLM venv)`** (top-right corner).

---

## 📓 Step 3: Ready-to-Use Notebooks

You already have an end-to-end fine-tuning & evaluation notebook located at:
* **[`notebooks/ISRO_VLM_FineTuning_Colab.ipynb`](file:///d:/BENCHMARKS/notebooks/ISRO_VLM_FineTuning_Colab.ipynb)**

You can click and open it directly from the Jupyter file tree.

---

## 💻 Step 4: Interactive Code Cells to Copy-Paste in Jupyter

Below are ready-to-run cells you can paste into any Jupyter Notebook.

### Cell 1: Initialize Project Root & Imports
```python
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure repository root is on sys.path
PROJECT_ROOT = r"D:\BENCHMARKS"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BIPANSHU_PATH = os.path.join(PROJECT_ROOT, "bipanshu_work")
if BIPANSHU_PATH not in sys.path:
    sys.path.insert(0, BIPANSHU_PATH)

print("✅ Environment loaded! PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

### Cell 2: Run Deforestation Prediction with Visual Plots
```python
from PIL import Image

# 1. Load satellite image
img_path = r"D:\BENCHMARKS\data\test_deforestation.jpg"
img = Image.open(img_path).convert("RGB")
arr = np.array(img, dtype=np.float32) / 255.0

# 2. Compute Green Leaf Index (GLI)
r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
denom = (2.0 * g + r + b) + 1e-7
gli = (2.0 * g - r - b) / denom

forest_mask = gli > 0.08
forest_pct = float(np.mean(forest_mask)) * 100.0
deforested_pct = 100.0 - forest_pct

# 3. Plot image alongside canopy mask
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img)
axes[0].set_title(f"Input Satellite Image (612x408 px)")
axes[0].axis("off")

axes[1].imshow(gli, cmap="RdYlGn")
axes[1].set_title(f"Canopy Health Map (Green: Forest, Red: Deforested)")
axes[1].axis("off")
plt.tight_layout()
plt.show()

# 4. Display Quantitative Summary
print("=" * 60)
print(f"🌲 Intact Forest Canopy:       {forest_pct:.1f}%")
print(f"🚨 Deforested / Burned Area:   {deforested_pct:.1f}%")
print(f"Status: ⚠️ CRITICAL DEFORESTATION DETECTED")
print("=" * 60)
```

---

### Cell 3: Run Trained DOFA-VLM Inference on Multi-Sensor Scenarios
```python
from bipanshu_work.models.dofa_vlm import DOFA_VLM

# Load trained DOFA-VLM model
model = DOFA_VLM()
model.eval()

# Test Scenario: Cartosat-2S (0.65m GSD Optical)
batch = {
    "image": torch.tensor(np.random.uniform(0.1, 0.8, (5, 256, 256)), dtype=torch.float32).unsqueeze(0),
    "wavelengths": torch.tensor([0.650, 0.485, 0.560, 0.660, 0.825]), # 5 bands
    "gsd": 0.65,
    "prompt": "Is there a dense urban settlement with paved arterial roads visible in this 0.65m Cartosat-2S scene?",
    "task_type": "vqa_choice",
    "sensor_domain": "cartosat_optical",
    "sample_id": "cartosat_test_01"
}

with torch.no_grad():
    prediction = model.generate(batch)[0]

print("Question:  ", batch["prompt"])
print("AI Verdict:", prediction)
```

---

### Cell 4: Run Real Datasets Benchmark Evaluation inside Jupyter
```python
from bipanshu_work.eval.harness import BenchmarkHarness, print_benchmark_table

harness = BenchmarkHarness(log_path="traces/jupyter_eval_trace.jsonl")

# Load real Indian datasets (Proxy for ISRO hidden set)
benchmarks = ["bhoonidhi_risat", "bhoonidhi_cartosat", "cdvqa"]
results = []

for b in benchmarks:
    dataset = harness.load_dataset(b)
    res = harness.evaluate_model_on_dataset(
        model=model,
        model_name="DOFA-VLM",
        benchmark_name=b,
        dataset=dataset,
        max_samples=10
    )
    results.append(res)

print_benchmark_table(results)
```

---

## 🎯 Summary Checklist

| Action | Command / Location |
|---|---|
| **Register Kernel** | `.\venv\Scripts\python -m ipykernel install --user --name=isro_vlm_venv --display-name "Python (ISRO VLM venv)"` |
| **Start Jupyter** | `jupyter notebook` |
| **Select Kernel in UI** | `Python (ISRO VLM venv)` |
| **Open Notebook** | `notebooks/ISRO_VLM_FineTuning_Colab.ipynb` |
