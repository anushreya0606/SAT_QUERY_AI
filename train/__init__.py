"""
train package — ISRO/SAC Remote Sensing VLM.

Provides:
  - LoRA fine-tuning trainer via PEFT
  - Custom multi-modal batch data collator
"""

from train.data_collator import MultiModalDataCollator
from train.trainer import LoRATrainer

__all__ = [
    "MultiModalDataCollator",
    "LoRATrainer",
]
