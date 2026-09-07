"""
Centralized configuration for the ISRO/SAC Remote Sensing VLM project.
All configs use Pydantic BaseModel for validation and sensible defaults.
"""

import os
import json
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

from enum import Enum
from typing import Optional, List, Literal
from pydantic import BaseModel, Field



class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class SensorType(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"


class ModelConfig(BaseModel):
    """Configuration for the VLM model."""

    model_name: str = Field(
        default="llava-hf/llava-1.5-7b-hf",
        description="HuggingFace model ID or local path",
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Device to run inference on. 'auto' selects CUDA if available.",
    )
    dtype: str = Field(
        default="float16",
        description="Model dtype: 'float16', 'bfloat16', or 'float32'",
    )
    max_new_tokens: int = Field(
        default=256,
        description="Maximum number of new tokens to generate",
    )
    use_dummy: bool = Field(
        default=False,
        description="If True, use dummy predictions instead of real model inference",
    )
    vision_encoder: str = Field(
        default="openai/clip-vit-large-patch14-336",
        description="Vision encoder model ID",
    )
    image_size: int = Field(default=336, description="Input image size for the vision encoder")
    adapter_path: Optional[str] = Field(
        default=None,
        description="Path to trained LoRA adapter directory (e.g. checkpoints/final_lora_adapter)",
    )

    def get_device(self) -> str:
        if self.device == DeviceType.AUTO:
            if _TORCH_AVAILABLE:
                return "cuda" if torch.cuda.is_available() else "cpu"
            return "cpu"
        return self.device.value

    def get_dtype(self):
        if not _TORCH_AVAILABLE:
            return None
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float16)



class DataConfig(BaseModel):
    """Configuration for datasets and data paths."""

    data_root: str = Field(
        default="./datasets_raw",
        description="Root directory for raw downloaded datasets",
    )
    processed_root: str = Field(
        default="./datasets_processed",
        description="Root directory for preprocessed datasets",
    )
    image_size: int = Field(default=336, description="Target image size after preprocessing")

    # Dataset-specific paths (relative to data_root)
    vrsbench_dir: str = Field(default="vrsbench", description="VRSBench dataset subdirectory")
    rsvqa_lr_dir: str = Field(default="rsvqa_lr", description="RSVQA-LR dataset subdirectory")
    rsvqa_hr_dir: str = Field(default="rsvqa_hr", description="RSVQA-HR dataset subdirectory")
    cdvqa_dir: str = Field(default="cdvqa", description="CDVQA dataset subdirectory")

    # Splits
    train_split: str = Field(default="train")
    val_split: str = Field(default="val")
    test_split: str = Field(default="test")

    def get_dataset_path(self, dataset_name: str) -> str:
        """Return the full path for a given dataset name."""
        dir_map = {
            "vrsbench": self.vrsbench_dir,
            "rsvqa_lr": self.rsvqa_lr_dir,
            "rsvqa_hr": self.rsvqa_hr_dir,
            "cdvqa": self.cdvqa_dir,
        }
        subdir = dir_map.get(dataset_name, dataset_name)
        return os.path.join(self.data_root, subdir)


class EvalConfig(BaseModel):
    """Configuration for the evaluation harness."""

    output_dir: str = Field(default="outputs", description="Directory to save evaluation results")
    run_name: str = Field(default="eval_run", description="Name tag for this evaluation run")
    batch_size: int = Field(default=1, description="Batch size for evaluation")
    num_workers: int = Field(default=4, description="Number of DataLoader workers")
    datasets: List[str] = Field(
        default=["vrsbench"],
        description="List of dataset names to evaluate on",
    )
    task_types: List[str] = Field(
        default=["captioning", "vqa", "grounding"],
        description="Task types to evaluate",
    )
    save_predictions: bool = Field(
        default=True,
        description="Whether to save per-sample predictions alongside the trace",
    )
    resume_from: Optional[str] = Field(
        default=None,
        description="Path to a partial trace JSON to resume evaluation from",
    )


class TrainingConfig(BaseModel):
    """Configuration for LoRA fine-tuning."""

    # LoRA parameters
    lora_rank: int = Field(default=16, description="LoRA rank (r)")
    lora_alpha: int = Field(default=32, description="LoRA alpha scaling factor")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout rate")
    lora_target_modules: List[str] = Field(
        default=["q_proj", "v_proj"],
        description="Which linear layers to apply LoRA to",
    )

    # Training hyperparameters
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=4, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation steps")
    warmup_ratio: float = Field(default=0.03, description="Warmup ratio")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")

    # I/O
    output_dir: str = Field(default="checkpoints", description="Directory to save model checkpoints")
    logging_steps: int = Field(default=10, description="Log every N steps")
    save_steps: int = Field(default=500, description="Save checkpoint every N steps")
    eval_steps: int = Field(default=250, description="Evaluate every N steps")

    # Misc
    seed: int = Field(default=42, description="Random seed")
    fp16: bool = Field(default=True, description="Use mixed precision (fp16)")
    use_wandb: bool = Field(default=False, description="Log to Weights & Biases")
    wandb_project: str = Field(default="isro-vlm", description="W&B project name")


class ProjectConfig(BaseModel):
    """Top-level config combining all sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, path: str) -> "ProjectConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str):
        """Save config to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def default(cls) -> "ProjectConfig":
        """Return default configuration."""
        return cls()
