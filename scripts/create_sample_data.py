"""
Synthetic and Sample Data Generator for ISRO/SAC Remote Sensing VLM.

Generates realistic mock datasets so users and evaluators can immediately run
and test the full pipeline (evaluation, training, CLI, preprocessing, web UI)
out of the box without downloading multi-gigabyte datasets.

Generates:
  1. datasets_raw/vrsbench  (captioning, VQA, visual grounding)
  2. datasets_raw/cartosat  (Cartosat-2S PAN & MS high-resolution data)
  3. datasets_raw/risat     (RISAT SAR dual-pol HH/HV data)
  4. datasets_raw/cdvqa     (Change Detection VQA pre/post pairs)
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw


def generate_synthetic_scene(scene_type: str = "urban", size: int = 512) -> Image.Image:
    """Generate a synthetic remote sensing scene with realistic patterns."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)

    if scene_type == "urban":
        # Grayscale background (roads/pavements)
        arr[:] = [140, 140, 140]
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        # Roads (dark gray)
        draw.rectangle([0, size // 2 - 20, size, size // 2 + 20], fill=(60, 60, 60))
        draw.rectangle([size // 2 - 20, 0, size // 2 + 20, size], fill=(60, 60, 60))
        # Buildings (various colors)
        draw.rectangle([40, 40, 180, 180], fill=(190, 80, 70))      # Red brick roof
        draw.rectangle([300, 50, 460, 190], fill=(80, 110, 180))    # Blue steel roof
        draw.rectangle([50, 300, 200, 450], fill=(220, 210, 190))   # Concrete rooftop
        draw.rectangle([320, 320, 470, 460], fill=(90, 150, 90))    # Garden/Park
        return img

    elif scene_type == "agricultural":
        # Green-brown patchwork fields
        arr[:] = [80, 130, 60]
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, size // 2, size // 2], fill=(160, 140, 80))        # Fallow field
        draw.rectangle([size // 2, 0, size, size // 2], fill=(60, 150, 60))       # Lush crop
        draw.rectangle([0, size // 2, size // 2, size], fill=(110, 90, 50))       # Bare soil
        draw.rectangle([size // 2, size // 2, size, size], fill=(45, 120, 50))     # Forest plot
        # Canal / Irrigation channel
        draw.line([(0, 100), (size, 400)], fill=(40, 90, 160), width=8)
        return img

    elif scene_type == "coastal":
        # Water and coastline
        arr[:] = [30, 70, 140]  # Deep blue water
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        # Land boundary
        draw.polygon([(0, 0), (size // 3, 0), (size // 2, size), (0, size)], fill=(180, 160, 120))  # Sand/coast
        draw.polygon([(0, 0), (size // 4, 0), (size // 3, size), (0, size)], fill=(70, 130, 60))    # Vegetation
        # Ships/vessels in water
        draw.rectangle([350, 200, 410, 230], fill=(240, 240, 240))  # Vessel
        return img

    else:
        # Default textured noise
        noise = np.random.randint(50, 200, (size, size, 3), dtype=np.uint8)
        return Image.fromarray(noise)


def create_sample_datasets(root_dir: str = "datasets_raw"):
    """Creates mock datasets in root_dir."""
    print(f"Creating sample datasets in '{root_dir}'...")

    # 1. VRSBench
    vrs_dir = os.path.join(root_dir, "vrsbench")
    vrs_img_dir = os.path.join(vrs_dir, "images")
    os.makedirs(vrs_img_dir, exist_ok=True)

    img1 = generate_synthetic_scene("urban")
    img2 = generate_synthetic_scene("agricultural")
    img3 = generate_synthetic_scene("coastal")

    img1.save(os.path.join(vrs_img_dir, "000001.jpg"))
    img2.save(os.path.join(vrs_img_dir, "000002.jpg"))
    img3.save(os.path.join(vrs_img_dir, "000003.jpg"))

    # VRSBench annotations
    caption_ann = [
        {"image_id": "000001", "caption": "An aerial remote sensing image of an urban area with crossing highways and commercial buildings."},
        {"image_id": "000002", "caption": "An agricultural zone divided into multiple rectangular crop fields with an irrigation canal."},
        {"image_id": "000003", "caption": "A coastal shoreline with sandy beach, coastal greenery, and a vessel anchored offshore."}
    ]
    with open(os.path.join(vrs_dir, "test_caption.json"), "w") as f:
        json.dump(caption_ann, f, indent=2)

    vqa_ann = [
        {"image_id": "000001", "question": "Are there road intersections present?", "answer": "Yes"},
        {"image_id": "000002", "question": "What is the primary land use?", "answer": "Agriculture"},
        {"image_id": "000003", "question": "Is water visible in this scene?", "answer": "Yes"}
    ]
    with open(os.path.join(vrs_dir, "test_vqa.json"), "w") as f:
        json.dump(vqa_ann, f, indent=2)

    grounding_ann = [
        {"image_id": "000001", "expression": "the red roof building", "bbox": [40.0, 40.0, 180.0, 180.0]},
        {"image_id": "000002", "expression": "the bare soil plot", "bbox": [0.0, 256.0, 256.0, 512.0]},
        {"image_id": "000003", "expression": "the ship offshore", "bbox": [350.0, 200.0, 410.0, 230.0]}
    ]
    with open(os.path.join(vrs_dir, "test_grounding.json"), "w") as f:
        json.dump(grounding_ann, f, indent=2)

    print("  [OK] VRSBench sample data created.")

    # 2. Cartosat-2S (Panchromatic high-resolution)
    cartosat_dir = os.path.join(root_dir, "cartosat")
    cartosat_pan_dir = os.path.join(cartosat_dir, "pan")
    os.makedirs(cartosat_pan_dir, exist_ok=True)

    pan1 = img1.convert("L")
    pan2 = img2.convert("L")
    pan1.save(os.path.join(cartosat_pan_dir, "IMG_CARTOSAT_001.tif"))
    pan2.save(os.path.join(cartosat_pan_dir, "IMG_CARTOSAT_002.tif"))

    cartosat_ann = [
        {"image_id": "IMG_CARTOSAT_001", "task_type": "captioning", "prompt": "Describe this Cartosat-2S high-resolution image.", "target": "Cartosat-2S 0.65m panchromatic imagery capturing urban residential structures with distinct roof geometries and roads."},
        {"image_id": "IMG_CARTOSAT_001", "task_type": "vqa", "question": "Can individual buildings be identified?", "answer": "Yes"},
        {"image_id": "IMG_CARTOSAT_002", "task_type": "vqa", "question": "Is this an agricultural or industrial area?", "answer": "Agricultural"}
    ]
    with open(os.path.join(cartosat_dir, "test_annotations.json"), "w") as f:
        json.dump(cartosat_ann, f, indent=2)

    print("  [OK] Cartosat-2S sample data created.")

    # 3. RISAT SAR (C-band dual-pol HH / HV)
    risat_dir = os.path.join(root_dir, "risat")
    hh_dir = os.path.join(risat_dir, "hh")
    hv_dir = os.path.join(risat_dir, "hv")
    os.makedirs(hh_dir, exist_ok=True)
    os.makedirs(hv_dir, exist_ok=True)

    # Simulate SAR speckle pattern
    np.random.seed(42)
    sar_base = np.array(img1.convert("L"), dtype=np.float32) / 255.0
    speckle_hh = np.random.exponential(scale=1.0, size=sar_base.shape)
    speckle_hv = np.random.exponential(scale=0.8, size=sar_base.shape)

    hh_arr = np.clip((sar_base * speckle_hh) * 255.0, 0, 255).astype(np.uint8)
    hv_arr = np.clip((sar_base * speckle_hv * 0.5) * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(hh_arr).save(os.path.join(hh_dir, "RISAT_001_HH.tif"))
    Image.fromarray(hv_arr).save(os.path.join(hv_dir, "RISAT_001_HV.tif"))

    risat_ann = [
        {"image_id": "RISAT_001", "task_type": "captioning", "prompt": "Describe this RISAT C-band SAR scene.", "target": "RISAT C-band SAR dual-polarization image displaying high backscatter from built-up structures and low backscatter from smooth paved surfaces."},
        {"image_id": "RISAT_001", "task_type": "vqa", "question": "Does this radar image exhibit speckle noise?", "answer": "Yes"},
        {"image_id": "RISAT_001", "task_type": "vqa", "question": "Are high backscatter corner reflectors visible?", "answer": "Yes"}
    ]
    with open(os.path.join(risat_dir, "test_annotations.json"), "w") as f:
        json.dump(risat_ann, f, indent=2)

    print("  [OK] RISAT SAR sample data created.")

    # 4. CDVQA (Change Detection)
    cdvqa_dir = os.path.join(root_dir, "cdvqa")
    pre_dir = os.path.join(cdvqa_dir, "images", "pre")
    post_dir = os.path.join(cdvqa_dir, "images", "post")
    ann_dir = os.path.join(cdvqa_dir, "annotations")
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Pre-change: green field; Post-change: new building added
    pre_img = generate_synthetic_scene("agricultural")
    post_img = pre_img.copy()
    post_draw = ImageDraw.Draw(post_img)
    post_draw.rectangle([200, 200, 320, 320], fill=(220, 60, 60))  # New constructed structure

    pre_img.save(os.path.join(pre_dir, "000001.png"))
    post_img.save(os.path.join(post_dir, "000001.png"))

    cdvqa_ann = [
        {
            "image_pre": "pre/000001.png",
            "image_post": "post/000001.png",
            "question": "Has new construction occurred in the area?",
            "answer": "Yes",
            "type": "existence"
        },
        {
            "image_pre": "pre/000001.png",
            "image_post": "post/000001.png",
            "question": "What type of change occurred?",
            "answer": "Building construction",
            "type": "type"
        }
    ]
    with open(os.path.join(ann_dir, "test.json"), "w") as f:
        json.dump(cdvqa_ann, f, indent=2)

    print("  [OK] CDVQA sample data created.")
    print(f"\nAll sample datasets successfully generated under '{root_dir}'!")


if __name__ == "__main__":
    create_sample_datasets()
