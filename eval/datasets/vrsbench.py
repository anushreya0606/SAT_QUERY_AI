import os
import json
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from PIL import Image
from torchvision import transforms


class VRSBenchDataset(Dataset):
    """
    VRSBench: Visual Remote Sensing Benchmark.
    Supports captioning, VQA, and visual grounding tasks.

    Expected directory structure:
        vrsbench/
        ├── images/
        │   ├── 000001.jpg
        │   └── ...
        ├── test_caption.json       # [{image_id, caption}, ...]
        ├── test_vqa.json           # [{image_id, question, answer}, ...]
        └── test_grounding.json     # [{image_id, expression, bbox}, ...]
    """

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        task_type: str = "captioning",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 336,
    ):
        self.data_path = data_path
        self.split = split
        self.task_type = task_type
        self.image_dir = os.path.join(data_path, "images")

        # Default transforms
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
        self._load_annotations()

    def _load_annotations(self):
        """Load annotations from the JSON file for the given split and task."""
        ann_filename = f"{self.split}_{self.task_type}.json"
        ann_path = os.path.join(self.data_path, ann_filename)

        if not os.path.exists(ann_path):
            # Try alternative naming conventions
            short_task = self.task_type[:-3] if self.task_type.endswith("ing") else self.task_type
            alt_names = [
                f"{self.split}_{short_task}.json",
                f"{self.split}.json",
                f"{self.task_type}_{self.split}.json",
                f"{short_task}_{self.split}.json",
                f"annotations_{self.split}_{self.task_type}.json",
                f"annotations_{self.split}_{short_task}.json",
            ]
            for alt in alt_names:
                alt_path = os.path.join(self.data_path, alt)
                if os.path.exists(alt_path):
                    ann_path = alt_path
                    break
            else:
                print(
                    f"Warning: Annotation file not found at {ann_path}. "
                    f"Dataset will be empty. Tried: {ann_filename}, {', '.join(alt_names)}"
                )
                return

        with open(ann_path, "r") as f:
            raw_data = json.load(f)

        # Handle both list-of-dicts and dict-of-lists formats
        if isinstance(raw_data, list):
            annotations = raw_data
        elif isinstance(raw_data, dict) and "annotations" in raw_data:
            annotations = raw_data["annotations"]
        elif isinstance(raw_data, dict) and "data" in raw_data:
            annotations = raw_data["data"]
        else:
            annotations = raw_data if isinstance(raw_data, list) else [raw_data]

        for ann in annotations:
            item = self._parse_annotation(ann)
            if item is not None:
                self.items.append(item)

        print(f"Loaded {len(self.items)} items from VRSBench ({self.split}/{self.task_type})")

    def _parse_annotation(self, ann: Dict) -> Optional[Dict[str, Any]]:
        """Parse a single annotation entry into a standardized format."""
        # Determine image filename
        image_id = ann.get("image_id", ann.get("img_id", ann.get("image", "")))
        if isinstance(image_id, int):
            image_filename = f"{image_id:06d}.jpg"
        else:
            image_filename = str(image_id)
            if not image_filename.endswith((".jpg", ".png", ".tif", ".tiff")):
                image_filename += ".jpg"

        image_path = os.path.join(self.image_dir, image_filename)

        if self.task_type == "captioning":
            return {
                "image_path": image_path,
                "prompt": "Describe this remote sensing image in detail.",
                "target": ann.get("caption", ann.get("text", "")),
                "task_type": "captioning",
                "image_id": str(image_id),
            }
        elif self.task_type == "vqa":
            return {
                "image_path": image_path,
                "prompt": ann.get("question", ""),
                "target": ann.get("answer", ann.get("answers", "")),
                "task_type": "vqa",
                "image_id": str(image_id),
            }
        elif self.task_type == "grounding":
            bbox = ann.get("bbox", ann.get("bounding_box", []))
            expression = ann.get("expression", ann.get("text", ann.get("caption", "")))
            return {
                "image_path": image_path,
                "prompt": f"Locate: {expression}",
                "target": bbox,
                "task_type": "grounding",
                "image_id": str(image_id),
            }
        return None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
        - image: PIL Image or transformed tensor
        - prompt: The text prompt for the model
        - target: Ground truth (caption string, answer, or bounding box list)
        - task_type: 'captioning', 'vqa', or 'grounding'
        - image_id: Unique image identifier
        """
        item = self.items[idx].copy()

        # Load image
        image_path = item["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            item["image"] = image
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            # Return a black placeholder image
            item["image"] = Image.new("RGB", (336, 336), (0, 0, 0))
            if self.transform:
                item["image"] = self.transform(item["image"])

        return item
