"""
LoRA fine-tuning trainer for the ISRO/SAC Remote Sensing VLM.

Wraps HuggingFace's Trainer with:
  - PEFT LoRA applied to the language decoder (q_proj, v_proj)
  - Mixed-precision (fp16 / bf16) training
  - Gradient checkpointing for memory efficiency
  - Checkpoint saving and resumption
  - Optional Weights & Biases logging
  - Domain-aware dataset composition (Sentinel-2 + Cartosat-2S + RISAT)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA fine-tuner for LLaVA-1.5 (or compatible VLMs) on remote sensing data.

    Usage:
        from config import ProjectConfig, TrainingConfig
        from train.trainer import LoRATrainer

        cfg = ProjectConfig.default()
        trainer = LoRATrainer(cfg.training, cfg.model)
        trainer.train(train_dataset, val_dataset)

    The trainer applies LoRA adapters only to the LLM backbone's attention
    projections, keeping the vision encoder and all other weights frozen.
    This reduces trainable parameters by ~99% while achieving strong domain
    adaptation with only a few hundred labelled remote sensing examples.
    """

    def __init__(self, training_config, model_config=None):
        """
        Args:
            training_config: TrainingConfig instance (from config.py).
            model_config:    ModelConfig instance (from config.py). If None,
                             uses TrainingConfig's output_dir and fp16 settings.
        """
        self.training_cfg = training_config
        self.model_cfg = model_config

        self.model = None
        self.processor = None
        self._lora_applied = False

    # ---------------------------------------------------------------------- #
    #  Model Preparation                                                       #
    # ---------------------------------------------------------------------- #

    def _load_base_model(self, model_name: str):
        """Load the base LLaVA model and processor for fine-tuning."""
        try:
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            import torch

            device_map = "auto"
            dtype = torch.float16 if self.training_cfg.fp16 else torch.float32

            logger.info(f"Loading base model '{model_name}' for LoRA fine-tuning...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            logger.info("Base model loaded.")
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Run: pip install transformers accelerate"
            ) from e

    def _apply_lora(self):
        """Wrap the language model backbone with PEFT LoRA adapters."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.training_cfg.lora_rank,
                lora_alpha=self.training_cfg.lora_alpha,
                lora_dropout=self.training_cfg.lora_dropout,
                target_modules=self.training_cfg.lora_target_modules,
                bias="none",
            )

            # Apply LoRA only to the language model component
            if hasattr(self.model, "language_model"):
                self.model.language_model = get_peft_model(
                    self.model.language_model, lora_config
                )
                self.model.language_model.print_trainable_parameters()
            else:
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()

            self._lora_applied = True
            logger.info(
                f"LoRA applied: r={self.training_cfg.lora_rank}, "
                f"alpha={self.training_cfg.lora_alpha}, "
                f"modules={self.training_cfg.lora_target_modules}"
            )
        except ImportError as e:
            raise ImportError(
                f"PEFT not installed: {e}. "
                "Run: pip install peft>=0.6.0"
            ) from e

    def _freeze_vision_encoder(self):
        """Freeze the vision tower (CLIP encoder) — we only fine-tune the LLM."""
        if hasattr(self.model, "vision_tower"):
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen.")
        if hasattr(self.model, "multi_modal_projector"):
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = False
            logger.info("Multi-modal projector frozen.")

    # ---------------------------------------------------------------------- #
    #  Training                                                                #
    # ---------------------------------------------------------------------- #

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        model_name: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Run LoRA fine-tuning.

        Args:
            train_dataset:          PyTorch Dataset for training.
            val_dataset:            Optional Dataset for periodic validation.
            model_name:             HuggingFace model ID. Falls back to ModelConfig.
            resume_from_checkpoint: Path to a prior checkpoint to resume from.

        Returns:
            TrainOutput with loss history and final checkpoint path.
        """
        try:
            from transformers import TrainingArguments, Trainer
            import torch
        except ImportError as e:
            raise ImportError(f"transformers not installed: {e}") from e

        from train.data_collator import MultiModalDataCollator

        # Resolve model name
        if model_name is None:
            model_name = (
                self.model_cfg.model_name
                if self.model_cfg
                else "llava-hf/llava-1.5-7b-hf"
            )

        # 1. Load base model
        self._load_base_model(model_name)

        # 2. Freeze vision encoder
        self._freeze_vision_encoder()

        # 3. Apply LoRA adapters
        self._apply_lora()

        # 4. Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        # 5. Set up W&B if requested
        if self.training_cfg.use_wandb:
            try:
                import wandb
                wandb.init(project=self.training_cfg.wandb_project)
                logger.info(f"W&B logging enabled: project='{self.training_cfg.wandb_project}'")
            except ImportError:
                logger.warning("wandb not installed. Continuing without W&B logging.")

        # 6. Build HuggingFace TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.training_cfg.output_dir,
            num_train_epochs=self.training_cfg.num_epochs,
            per_device_train_batch_size=self.training_cfg.batch_size,
            gradient_accumulation_steps=self.training_cfg.gradient_accumulation_steps,
            learning_rate=self.training_cfg.learning_rate,
            weight_decay=self.training_cfg.weight_decay,
            warmup_ratio=self.training_cfg.warmup_ratio,
            max_grad_norm=self.training_cfg.max_grad_norm,
            fp16=self.training_cfg.fp16 and torch.cuda.is_available(),
            logging_steps=self.training_cfg.logging_steps,
            save_steps=self.training_cfg.save_steps,
            eval_steps=self.training_cfg.eval_steps if val_dataset else None,
            evaluation_strategy="steps" if val_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=bool(val_dataset),
            seed=self.training_cfg.seed,
            dataloader_pin_memory=torch.cuda.is_available(),
            remove_unused_columns=False,   # Required for multi-modal models
            report_to="wandb" if self.training_cfg.use_wandb else "none",
        )

        # 7. Build collator
        collator = MultiModalDataCollator(
            processor=self.processor,
            max_length=2048,
        )

        # 8. Instantiate HuggingFace Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )

        # 9. Train!
        logger.info(
            f"Starting LoRA fine-tuning: epochs={self.training_cfg.num_epochs}, "
            f"lr={self.training_cfg.learning_rate}, "
            f"batch={self.training_cfg.batch_size}×{self.training_cfg.gradient_accumulation_steps} steps"
        )
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.info("Training complete.")
        logger.info(f"Train loss: {result.training_loss:.4f}")

        # 10. Save final LoRA adapter weights
        final_ckpt = os.path.join(self.training_cfg.output_dir, "final_lora_adapter")
        if hasattr(self.model, "language_model") and self._lora_applied:
            self.model.language_model.save_pretrained(final_ckpt)
        elif self._lora_applied:
            self.model.save_pretrained(final_ckpt)
        self.processor.save_pretrained(final_ckpt)
        logger.info(f"LoRA adapter saved to '{final_ckpt}'.")

        return result

    # ---------------------------------------------------------------------- #
    #  Inference from Checkpoint                                               #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def load_from_checkpoint(
        base_model_name: str,
        lora_checkpoint_dir: str,
        device: Optional[str] = None,
    ):
        """
        Load a LoRA-adapted model from a saved checkpoint for inference.

        Args:
            base_model_name:     HuggingFace base model ID.
            lora_checkpoint_dir: Path to the saved LoRA adapter directory.
            device:              Target device. Auto-detects CUDA if None.

        Returns:
            (model, processor) tuple ready for inference.
        """
        try:
            from peft import PeftModel
            from transformers import LlavaForConditionalGeneration, AutoProcessor

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(f"Loading base model '{base_model_name}'...")
            processor = AutoProcessor.from_pretrained(base_model_name)
            base_model = LlavaForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            logger.info(f"Merging LoRA adapter from '{lora_checkpoint_dir}'...")
            if hasattr(base_model, "language_model"):
                base_model.language_model = PeftModel.from_pretrained(
                    base_model.language_model, lora_checkpoint_dir
                )
                base_model.language_model = base_model.language_model.merge_and_unload()
            else:
                base_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)
                base_model = base_model.merge_and_unload()

            base_model.eval()
            logger.info("LoRA checkpoint loaded and merged.")
            return base_model, processor

        except ImportError as e:
            raise ImportError(
                f"peft or transformers not installed: {e}. "
                "Run: pip install peft transformers"
            ) from e
