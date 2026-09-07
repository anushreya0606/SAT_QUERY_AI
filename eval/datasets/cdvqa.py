import os
import json
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from torchvision import transforms


class CDVQADataset(Dataset):
    """
    CDVQA: Change Detection Visual Question Answering.
    Takes image pairs (pre-change and post-change) with questions about changes.

    Supports question types: existence, type, number, position of changes.

    Expected directory structure:
        cdvqa/
        ├── images/
        │   ├── pre/
        │   │   ├── 000001.png
        │   │   └── ...
        │   └── post/
        │       ├── 000001.png
        │       └── ...
        ├── annotations/
        │   ├── test.json
        │   └── ...
        └── (or flat structure with annotation file)

    Annotation format (per entry):
        {
            "image_pre": "pre/000001.png",
            "image_post": "post/000001.png",
            "question": "Has anything changed?",
            "answer": "Yes",
            "type": "existence"
        }
    """

    QUESTION_TYPES = [
        "existence",      # Has anything changed?
        "type",           # What type of change occurred?
        "number",         # How many changes?
        "position",       # Where did the change occur?
    ]

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 336,
        filter_types: Optional[List[str]] = None,
    ):
        """
        Args:
            data_path: Path to the CDVQA dataset root.
            split: One of 'train', 'val', 'test'.
            transform: Optional torchvision transforms (applied to both images).
            image_size: Target image size.
            filter_types: If set, only include questions of these types.
        """
        self.data_path = data_path
        self.split = split
        self.filter_types = filter_types

        # Image directories — support both flat and nested structures
        self.image_dir = os.path.join(data_path, "images")
        if not os.path.exists(self.image_dir):
            self.image_dir = data_path

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
        """Load annotations from JSON file."""
        # Try various file naming conventions
        ann_patterns = [
            os.path.join(self.data_path, "annotations", f"{self.split}.json"),
            os.path.join(self.data_path, f"{self.split}.json"),
            os.path.join(self.data_path, f"cdvqa_{self.split}.json"),
            os.path.join(self.data_path, f"annotations_{self.split}.json"),
            os.path.join(self.data_path, "annotations", f"cdvqa_{self.split}.json"),
        ]

        ann_path = None
        for pattern in ann_patterns:
            if os.path.exists(pattern):
                ann_path = pattern
                break

        if ann_path is None:
            print(
                f"Warning: Could not find CDVQA annotation file at {self.data_path}. "
                f"Dataset will be empty."
            )
            return

        with open(ann_path, "r") as f:
            raw_data = json.load(f)

        # Handle different JSON structures
        if isinstance(raw_data, list):
            annotations = raw_data
        elif isinstance(raw_data, dict):
            annotations = raw_data.get("annotations", raw_data.get("data", []))
        else:
            annotations = []

        for ann in annotations:
            q_type = ann.get("type", ann.get("question_type", "unknown")).lower()

            if self.filter_types and q_type not in self.filter_types:
                continue

            # Resolve image paths
            img_pre = ann.get("image_pre", ann.get("img_pre", ann.get("image1", "")))
            img_post = ann.get("image_post", ann.get("img_post", ann.get("image2", "")))

            # Build full paths
            pre_path = self._resolve_image_path(img_pre)
            post_path = self._resolve_image_path(img_post)

            self.items.append({
                "image_pre_path": pre_path,
                "image_post_path": post_path,
                "prompt": ann.get("question", ""),
                "target": ann.get("answer", ""),
                "task_type": "cdvqa",
                "question_type": q_type,
            })

        print(f"Loaded {len(self.items)} items from CDVQA ({self.split} split)")

    def _resolve_image_path(self, img_ref: str) -> str:
        """Resolve an image reference to a full path."""
        if os.path.isabs(img_ref):
            return img_ref

        # Try relative to image_dir first
        path = os.path.join(self.image_dir, img_ref)
        if os.path.exists(path):
            return path

        # Try relative to data_path
        path = os.path.join(self.data_path, img_ref)
        if os.path.exists(path):
            return path

        # Return best guess
        return os.path.join(self.image_dir, img_ref)

    def _load_image(self, path: str) -> Image.Image:
        """Load an image, returning a black placeholder on failure."""
        try:
            return Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not load image {path}: {e}")
            return Image.new("RGB", (336, 336), (0, 0, 0))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
        - image_pre: Pre-change image (PIL Image or tensor)
        - image_post: Post-change image (PIL Image or tensor)
        - prompt: The question text
        - target: The expected answer text
        - task_type: 'cdvqa'
        - question_type: 'existence', 'type', 'number', or 'position'
        """
        item = self.items[idx].copy()

        # Load both images
        image_pre = self._load_image(item["image_pre_path"])
        image_post = self._load_image(item["image_post_path"])

        if self.transform:
            image_pre = self.transform(image_pre)
            image_post = self.transform(image_post)

        item["image_pre"] = image_pre
        item["image_post"] = image_post
        # Also provide a combined 'image' key for compatibility with the runner
        # Concatenate along channel dimension for models that expect a single input
        item["image"] = (image_pre, image_post)

        return item

    def get_question_type_distribution(self) -> Dict[str, int]:
        """Return a count of questions per question type."""
        dist: Dict[str, int] = {}
        for item in self.items:
            qtype = item.get("question_type", "unknown")
            dist[qtype] = dist.get(qtype, 0) + 1
        return dist
