"""
Cartosat-2S dataset loader for the ISRO/SAC Remote Sensing VLM.

Cartosat-2S provides:
  - Panchromatic (PAN) imagery at 0.6 m spatial resolution
  - Multispectral (MS) imagery at 2.0 m spatial resolution

This module provides a PyTorch Dataset that reads Cartosat-2S imagery and pairs
it with optional annotation files (captions, QA pairs) for evaluation and fine-tuning.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class CartosatDataset(Dataset):
    """
    PyTorch Dataset for Cartosat-2S panchromatic and/or multispectral imagery.

    Supports:
      - Standalone usage (image-only, for unsupervised evaluation)
      - Paired with a JSON annotation file for captioning, VQA, or grounding

    Expected directory structure (flexible):
        cartosat/
        ├── pan/                     ← Panchromatic GeoTIFFs
        │   ├── IMG_20231015_001.tif
        │   └── ...
        ├── ms/                      ← Multispectral GeoTIFFs (optional)
        │   ├── IMG_20231015_001_MS.tif
        │   └── ...
        └── annotations.json         ← Optional annotations

    Annotation format (per entry):
        {
            "image_id": "IMG_20231015_001",
            "task_type": "captioning",          // or "vqa", "grounding"
            "prompt": "Describe this image.",
            "target": "Urban residential area with road network.",  // caption or answer
            // For vqa:
            // "question": "What land cover is visible?",
            // "answer": "Agricultural",
            // For grounding:
            // "expression": "the large water body",
            // "bbox": [120, 85, 340, 210]
        }
    """

    # Channel statistics for Cartosat-2S (estimated from ISRO spectral sheets)
    PAN_MEAN  = [0.42, 0.42, 0.42]   # grayscale replicated to 3 channels
    PAN_STD   = [0.20, 0.20, 0.20]
    MS_MEAN   = [0.37, 0.40, 0.30]   # approximate BGR → RGB
    MS_STD    = [0.18, 0.18, 0.16]

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        mode: str = "pan",
        task_type: str = "captioning",
        annotation_file: Optional[str] = None,
        image_size: int = 336,
        transform: Optional[transforms.Compose] = None,
        ms_bands: Optional[List[int]] = None,
    ):
        """
        Args:
            data_path:       Root directory for the Cartosat dataset.
            split:           'train', 'val', or 'test' (used to find annotation file).
            mode:            'pan' (panchromatic) or 'ms' (multispectral).
            task_type:       'captioning', 'vqa', or 'grounding'.
            annotation_file: Path to the annotation JSON. If None, auto-detected.
            image_size:      Target spatial resolution.
            transform:       Custom torchvision transforms. If None, uses defaults.
            ms_bands:        1-indexed MS bands to select for RGB composite.
                             Defaults to [3, 2, 1] (Red, Green, Blue for typical MS).
        """
        self.data_path = Path(data_path)
        self.split = split
        self.mode = mode.lower()
        self.task_type = task_type
        self.image_size = image_size
        self.ms_bands = ms_bands or [3, 2, 1]

        # Image subdirectory
        self.image_dir = self.data_path / self.mode
        if not self.image_dir.exists():
            self.image_dir = self.data_path  # flat structure fallback

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            if self.mode == "pan":
                mean, std = self.PAN_MEAN, self.PAN_STD
            else:
                mean, std = self.MS_MEAN, self.MS_STD
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        self.items: List[Dict[str, Any]] = []
        self._load_annotations(annotation_file)

    # ---------------------------------------------------------------------- #
    #  Annotation Loading                                                      #
    # ---------------------------------------------------------------------- #

    def _find_annotation_file(self) -> Optional[Path]:
        """Search for a suitable annotation file."""
        candidates = [
            self.data_path / f"{self.split}_annotations.json",
            self.data_path / f"annotations_{self.split}.json",
            self.data_path / "annotations.json",
            self.data_path / f"{self.split}.json",
            self.data_path / f"cartosat_{self.split}.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _load_annotations(self, annotation_file: Optional[str]):
        """Load annotation JSON if available, otherwise build image-only index."""
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
                f"Cartosat-2S: loaded {len(self.items)} annotated items "
                f"({self.split}/{self.mode}/{self.task_type})"
            )
        else:
            # Image-only mode — index all TIF files
            self._index_images()
            logger.info(
                f"Cartosat-2S: no annotation file found; indexed {len(self.items)} images "
                f"from '{self.image_dir}'."
            )

    def _parse_annotation(self, ann: Dict) -> Optional[Dict[str, Any]]:
        """Parse a single annotation entry."""
        image_id = ann.get("image_id", ann.get("img_id", ""))
        image_path = self._resolve_image_path(str(image_id))

        if self.task_type == "captioning":
            return {
                "image_path": image_path,
                "prompt": ann.get("prompt", "Describe this Cartosat-2S remote sensing image."),
                "target": ann.get("target", ann.get("caption", "")),
                "task_type": "captioning",
                "image_id": str(image_id),
                "mode": self.mode,
            }
        elif self.task_type == "vqa":
            return {
                "image_path": image_path,
                "prompt": ann.get("question", ann.get("prompt", "")),
                "target": ann.get("answer", ann.get("target", "")),
                "task_type": "vqa",
                "image_id": str(image_id),
                "mode": self.mode,
            }
        elif self.task_type == "grounding":
            return {
                "image_path": image_path,
                "prompt": f"Locate: {ann.get('expression', ann.get('prompt', ''))}",
                "target": ann.get("bbox", ann.get("target", [])),
                "task_type": "grounding",
                "image_id": str(image_id),
                "mode": self.mode,
            }
        return None

    def _index_images(self):
        """Build an image-only dataset with a generic captioning prompt."""
        extensions = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
        for p in sorted(self.image_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in extensions:
                self.items.append({
                    "image_path": str(p),
                    "prompt": "Describe this Cartosat-2S remote sensing image.",
                    "target": "",
                    "task_type": self.task_type,
                    "image_id": p.stem,
                    "mode": self.mode,
                })

    def _resolve_image_path(self, image_id: str) -> str:
        """Resolve image_id to a full file path."""
        # Try common extensions
        for ext in [".tif", ".tiff", ".jpg", ".jpeg", ".png"]:
            candidate = self.image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return str(candidate)
        # Try with _PAN / _MS suffix
        suffix = "_PAN" if self.mode == "pan" else "_MS"
        for ext in [".tif", ".tiff"]:
            candidate = self.image_dir / f"{image_id}{suffix}{ext}"
            if candidate.exists():
                return str(candidate)
        # Return best guess
        return str(self.image_dir / f"{image_id}.tif")

    # ---------------------------------------------------------------------- #
    #  Image Loading                                                           #
    # ---------------------------------------------------------------------- #

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load a Cartosat-2S image as a 3-channel PIL Image.

        PAN: single-band → replicated to RGB.
        MS:  multi-band → select RGB composite bands.
        """
        try:
            import rasterio
            with rasterio.open(image_path) as src:
                if self.mode == "pan":
                    band = src.read(1).astype(np.float32)
                    # Percentile stretch
                    lo, hi = np.percentile(band, 2), np.percentile(band, 98)
                    if hi > lo:
                        band = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
                    arr_uint8 = (band * 255).astype(np.uint8)
                    return Image.fromarray(arr_uint8).convert("RGB")
                else:
                    # Read selected bands
                    bands = [
                        src.read(b).astype(np.float32)
                        for b in self.ms_bands
                        if b <= src.count
                    ]
                    if not bands:
                        bands = [src.read(1).astype(np.float32)]
                    while len(bands) < 3:
                        bands.append(bands[-1])
                    arr = np.stack(bands[:3], axis=-1)  # (H, W, 3)
                    # Percentile stretch
                    for i in range(3):
                        lo = np.percentile(arr[:, :, i], 2)
                        hi = np.percentile(arr[:, :, i], 98)
                        if hi > lo:
                            arr[:, :, i] = np.clip((arr[:, :, i] - lo) / (hi - lo), 0.0, 1.0)
                    arr_uint8 = (arr * 255).astype(np.uint8)
                    return Image.fromarray(arr_uint8).convert("RGB")

        except ImportError:
            logger.debug("rasterio not available; using Pillow.")
        except Exception as e:
            logger.warning(f"Could not read '{image_path}' with rasterio: {e}.")

        # Pillow fallback
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not load '{image_path}': {e}. Returning blank image.")
            return Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

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
          - image_id:   Unique image identifier
          - mode:       'pan' or 'ms'
        """
        item = self.items[idx].copy()
        image = self._load_image(item["image_path"])
        if self.transform:
            image = self.transform(image)
        item["image"] = image
        return item

    def __repr__(self) -> str:
        return (
            f"CartosatDataset(mode={self.mode}, split={self.split}, "
            f"task={self.task_type}, n={len(self)})"
        )
