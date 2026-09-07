import os
import json
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from PIL import Image
from torchvision import transforms


class RSVQADataset(Dataset):
    """
    RSVQA: Remote Sensing Visual Question Answering.
    Supports both Low-Resolution (LR, based on Sentinel-2) and
    High-Resolution (HR) variants.

    Expected directory structure:
        rsvqa_lr/ (or rsvqa_hr/)
        ├── images/
        │   ├── 000001.tif (or .png/.jpg)
        │   └── ...
        ├── questions/
        │   ├── test_questions.json
        │   └── test_answers.json
        │   (or flat structure)
        ├── questions.json   # [{id, img_id, question, type}, ...]
        └── answers.json     # [{id, question_id, answer, active}, ...]

    Alternatively supports HuggingFace-style:
        rsvqa_lr/
        ├── images/
        └── LR_split_test_questions.json
        └── LR_split_test_answers.json
    """

    QUESTION_TYPES = [
        "presence",
        "comparison",
        "rural_urban",
        "count",
    ]

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        variant: str = "lr",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 336,
        filter_types: Optional[List[str]] = None,
    ):
        """
        Args:
            data_path: Path to the RSVQA dataset root.
            split: One of 'train', 'val', 'test'.
            variant: 'lr' for Low-Resolution, 'hr' for High-Resolution.
            transform: Optional torchvision transforms.
            image_size: Target image size.
            filter_types: If set, only include questions of these types.
        """
        self.data_path = data_path
        self.split = split
        self.variant = variant.lower()
        self.filter_types = filter_types
        self.image_dir = os.path.join(data_path, "images")

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

    def _find_file(self, patterns: List[str]) -> Optional[str]:
        """Search for a file matching any of the given patterns."""
        for pattern in patterns:
            path = os.path.join(self.data_path, pattern)
            if os.path.exists(path):
                return path

            # Try in a 'questions' subdirectory
            path_sub = os.path.join(self.data_path, "questions", pattern)
            if os.path.exists(path_sub):
                return path_sub

        return None

    def _load_annotations(self):
        """Load and merge question + answer annotations."""
        prefix = self.variant.upper()

        # Find questions file
        q_patterns = [
            f"{prefix}_split_{self.split}_questions.json",
            f"{self.split}_questions.json",
            "questions.json",
            f"{prefix}_{self.split}_questions.json",
        ]
        q_path = self._find_file(q_patterns)

        # Find answers file
        a_patterns = [
            f"{prefix}_split_{self.split}_answers.json",
            f"{self.split}_answers.json",
            "answers.json",
            f"{prefix}_{self.split}_answers.json",
        ]
        a_path = self._find_file(a_patterns)

        if q_path is None or a_path is None:
            print(
                f"Warning: Could not find RSVQA annotation files at {self.data_path}. "
                f"Searched for patterns: {q_patterns}"
            )
            return

        # Load questions
        with open(q_path, "r") as f:
            raw_questions = json.load(f)

        # Load answers
        with open(a_path, "r") as f:
            raw_answers = json.load(f)

        # Normalize: both could be list or dict with 'questions'/'answers' key
        questions = raw_questions if isinstance(raw_questions, list) else raw_questions.get("questions", [])
        answers = raw_answers if isinstance(raw_answers, list) else raw_answers.get("answers", [])

        # Build answer lookup: question_id -> answer
        answer_map: Dict[Any, str] = {}
        for ans in answers:
            # Only use active answers
            if ans.get("active", True):
                qid = ans.get("question_id", ans.get("id"))
                answer_map[qid] = ans.get("answer", "")

        # Build items by joining questions with answers
        for q in questions:
            q_id = q.get("id", q.get("question_id"))
            q_type = q.get("type", q.get("question_type", "unknown")).lower()
            img_id = q.get("img_id", q.get("image_id"))

            # Optional filtering by question type
            if self.filter_types and q_type not in self.filter_types:
                continue

            answer = answer_map.get(q_id)
            if answer is None:
                continue

            # Determine image path
            if isinstance(img_id, int):
                # Try common naming patterns
                image_filename = f"{img_id}.tif"
            else:
                image_filename = str(img_id)

            image_path = os.path.join(self.image_dir, image_filename)

            # If .tif doesn't exist, try .png and .jpg
            if not os.path.exists(image_path):
                for ext in [".png", ".jpg", ".jpeg"]:
                    alt_path = os.path.splitext(image_path)[0] + ext
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break

            self.items.append({
                "image_path": image_path,
                "prompt": q.get("question", ""),
                "target": answer,
                "task_type": "vqa",
                "question_type": q_type,
                "image_id": str(img_id),
                "question_id": str(q_id),
            })

        print(
            f"Loaded {len(self.items)} items from RSVQA-{self.variant.upper()} "
            f"({self.split} split)"
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a dictionary containing:
        - image: PIL Image or transformed tensor
        - prompt: The question text
        - target: The expected answer text
        - task_type: 'vqa'
        - question_type: e.g., 'presence', 'comparison', 'count', 'rural_urban'
        """
        item = self.items[idx].copy()

        image_path = item["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            item["image"] = image
        except (FileNotFoundError, OSError) as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            item["image"] = Image.new("RGB", (336, 336), (0, 0, 0))
            if self.transform:
                item["image"] = self.transform(item["image"])

        return item

    def get_question_type_distribution(self) -> Dict[str, int]:
        """Return a count of questions per question type."""
        dist: Dict[str, int] = {}
        for item in self.items:
            qtype = item.get("question_type", "unknown")
            dist[qtype] = dist.get(qtype, 0) + 1
        return dist
