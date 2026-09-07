"""
RISAT (Radar Imaging Satellite) dataset loader for the ISRO/SAC Remote Sensing VLM.

RISAT-1 and RISAT-2 are C-band and X-band SAR satellites operated by ISRO.
They provide HH, HV, VV, and VH polarisation imagery in various imaging modes
(Fine Resolution Stripmap, Coarse Resolution ScanSAR, etc.).

This module provides:
  - RISATDataset: PyTorch Dataset for SAR imagery with optional VQA / captioning annotations
  - Dual-polarisation support (HH + HV → 3-channel pseudo-colour composite)
  - dB conversion, speckle filtering, and normalisation
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class RISATDataset(Dataset):
    """
    PyTorch Dataset for RISAT SAR imagery.

    Supports three polarisation modes:
      - single:  One polarisation (HH or HV) — replicated to 3 channels
      - dual:    HH + HV → 3-channel composite (HH, HV, HH-HV ratio)
      - quad:    Full quad-pol (not yet supported — falls back to dual)

    Expected directory structure:
        risat/
        ├── hh/                          ← HH polarisation GeoTIFFs
        │   ├── SAR_20231015_001_HH.tif
        │   └── ...
        ├── hv/                          ← HV polarisation GeoTIFFs (optional)
        │   ├── SAR_20231015_001_HV.tif
        │   └── ...
        └── annotations.json             ← Optional VQA / captioning annotations

    Annotation format (per entry):
        {
            "image_id": "SAR_20231015_001",
            "task_type": "vqa",
            "question": "Is flooding visible in this SAR image?",
            "answer": "Yes"
        }
    """

    # Typical RISAT SAR dB clipping range
    DB_MIN = -25.0
    DB_MAX =   5.0

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        polarisation: str = "dual",
        task_type: str = "vqa",
        annotation_file: Optional[str] = None,
        image_size: int = 336,
        transform: Optional[transforms.Compose] = None,
        apply_speckle_filter: bool = True,
        db_range: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            data_path:            Root directory for the RISAT dataset.
            split:                'train', 'val', or 'test'.
            polarisation:         'single' | 'dual'.
            task_type:            'captioning' | 'vqa' | 'grounding'.
            annotation_file:      Explicit path to annotation JSON (auto-detected if None).
            image_size:           Target spatial resolution.
            transform:            Custom torchvision transforms.
            apply_speckle_filter: Apply 3×3 box-car speckle reduction.
            db_range:             (min_dB, max_dB) clipping range. Defaults to (-25, 5).
        """
        self.data_path = Path(data_path)
        self.split = split
        self.polarisation = polarisation.lower()
        self.task_type = task_type
        self.image_size = image_size
        self.apply_speckle_filter = apply_speckle_filter
        self.db_range = db_range or (self.DB_MIN, self.DB_MAX)

        # Image directories
        self.hh_dir = self.data_path / "hh"
        self.hv_dir = self.data_path / "hv"
        if not self.hh_dir.exists():
            self.hh_dir = self.data_path  # flat structure

        # Build transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        self.items: List[Dict[str, Any]] = []
        self._load_annotations(annotation_file)

    # ---------------------------------------------------------------------- #
    #  Annotation Loading                                                      #
    # ---------------------------------------------------------------------- #

    def _find_annotation_file(self) -> Optional[Path]:
        candidates = [
            self.data_path / f"{self.split}_annotations.json",
            self.data_path / f"annotations_{self.split}.json",
            self.data_path / "annotations.json",
            self.data_path / f"{self.split}.json",
            self.data_path / f"risat_{self.split}.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_annotations(self, annotation_file: Optional[str]):
        ann_path = Path(annotation_file) if annotation_file else self._find_annotation_file()

        if ann_path and ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            annotations = raw if isinstance(raw, list) else raw.get("annotations", raw.get("data", []))
            for ann in annotations:
                ann_task = ann.get("task_type")
                if ann_task and ann_task != self.task_type:
                    continue
                item = self._parse_annotation(ann)
                if item:
                    self.items.append(item)
            logger.info(
                f"RISAT: loaded {len(self.items)} annotated items "
                f"({self.split}/{self.polarisation}/{self.task_type})"
            )
        else:
            self._index_images()
            logger.info(
                f"RISAT: no annotation file found; indexed {len(self.items)} images "
                f"from '{self.hh_dir}'."
            )

    def _parse_annotation(self, ann: Dict) -> Optional[Dict[str, Any]]:
        image_id = str(ann.get("image_id", ann.get("img_id", "")))
        hh_path = self._resolve_hh_path(image_id)
        hv_path = self._resolve_hv_path(image_id) if self.polarisation == "dual" else None

        if self.task_type == "captioning":
            return {
                "hh_path": hh_path,
                "hv_path": hv_path,
                "prompt": ann.get("prompt", "Describe this RISAT SAR remote sensing image."),
                "target": ann.get("target", ann.get("caption", "")),
                "task_type": "captioning",
                "image_id": image_id,
            }
        elif self.task_type == "vqa":
            return {
                "hh_path": hh_path,
                "hv_path": hv_path,
                "prompt": ann.get("question", ann.get("prompt", "")),
                "target": ann.get("answer", ann.get("target", "")),
                "task_type": "vqa",
                "image_id": image_id,
            }
        elif self.task_type == "grounding":
            return {
                "hh_path": hh_path,
                "hv_path": hv_path,
                "prompt": f"Locate: {ann.get('expression', ann.get('prompt', ''))}",
                "target": ann.get("bbox", ann.get("target", [])),
                "task_type": "grounding",
                "image_id": image_id,
            }
        return None

    def _index_images(self):
        """Build image-only dataset from directory scan."""
        extensions = {".tif", ".tiff"}
        for p in sorted(self.hh_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions:
                image_id = p.stem
                hv_path = self._resolve_hv_path(image_id) if self.polarisation == "dual" else None
                self.items.append({
                    "hh_path": str(p),
                    "hv_path": hv_path,
                    "prompt": "Describe this RISAT SAR remote sensing image.",
                    "target": "",
                    "task_type": self.task_type,
                    "image_id": image_id,
                })

    def _resolve_hh_path(self, image_id: str) -> str:
        for ext in [".tif", ".tiff"]:
            p = self.hh_dir / f"{image_id}_HH{ext}"
            if p.exists():
                return str(p)
            p = self.hh_dir / f"{image_id}{ext}"
            if p.exists():
                return str(p)
        return str(self.hh_dir / f"{image_id}_HH.tif")

    def _resolve_hv_path(self, image_id: str) -> Optional[str]:
        if not self.hv_dir.exists():
            return None
        for ext in [".tif", ".tiff"]:
            p = self.hv_dir / f"{image_id}_HV{ext}"
            if p.exists():
                return str(p)
        return None

    # ---------------------------------------------------------------------- #
    #  SAR Image Processing                                                    #
    # ---------------------------------------------------------------------- #

    def _read_band(self, path: str) -> np.ndarray:
        """Read a single SAR band as float32 ndarray (H, W)."""
        try:
            import rasterio
            with rasterio.open(path) as src:
                return src.read(1).astype(np.float32)
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"rasterio failed for '{path}': {e}")

        # PIL fallback
        try:
            img = Image.open(path).convert("L")
            return np.array(img, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Could not read '{path}': {e}. Returning zeros.")
            return np.zeros((512, 512), dtype=np.float32)

    def _to_db(self, band: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.clip(band, 1e-10, None))

    def _speckle_filter(self, band: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import uniform_filter
            return uniform_filter(band, size=3).astype(np.float32)
        except ImportError:
            return band

    def _normalise(self, band: np.ndarray) -> np.ndarray:
        db_min, db_max = self.db_range
        band = np.clip(band, db_min, db_max)
        return (band - db_min) / (db_max - db_min)

    def _process_band(self, band: np.ndarray) -> np.ndarray:
        if band.max() > 1.0:
            band = self._to_db(band)
        if self.apply_speckle_filter:
            band = self._speckle_filter(band)
        return self._normalise(band)

    def _build_composite(self, hh_path: str, hv_path: Optional[str]) -> Image.Image:
        """Build a 3-channel pseudo-colour SAR composite."""
        hh_raw = self._read_band(hh_path)
        hh = self._process_band(hh_raw)

        if hv_path and os.path.exists(hv_path):
            hv_raw = self._read_band(hv_path)
            hv = self._process_band(hv_raw)
            ratio = np.clip(hh - hv, 0.0, 1.0)
            composite = np.stack([hh, hv, ratio], axis=-1)  # (H, W, 3)
        else:
            composite = np.stack([hh, hh, hh], axis=-1)    # grayscale replicated

        arr_uint8 = (composite * 255).astype(np.uint8)
        return Image.fromarray(arr_uint8).convert("RGB")

    # ---------------------------------------------------------------------- #
    #  Dataset Interface                                                       #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dict with:
          - image:      Transformed tensor (3, H, W)
          - prompt:     Text prompt / question
          - target:     Ground truth string or bbox list
          - task_type:  'captioning' | 'vqa' | 'grounding'
          - image_id:   Unique identifier
        """
        item = self.items[idx].copy()
        pil_image = self._build_composite(item["hh_path"], item.get("hv_path"))
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = pil_image
        item["image"] = image
        # Remove internal path keys for cleaner batch dicts
        item.pop("hh_path", None)
        item.pop("hv_path", None)
        return item

    def __repr__(self) -> str:
        return (
            f"RISATDataset(pol={self.polarisation}, split={self.split}, "
            f"task={self.task_type}, n={len(self)})"
        )
