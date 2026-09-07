"""
Multi-modal batch data collator for the ISRO/SAC Remote Sensing VLM.

Handles:
  - Variable-length text token sequences (left-padding for LLaVA)
  - Single-image samples (captioning, VQA, grounding)
  - Dual-image samples (CDVQA change detection — pre/post image pairs)
  - Flexible label masking for causal LM fine-tuning
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

# Sentinel value used to mask non-predicted tokens in the loss
LABEL_IGNORE_INDEX = -100


@dataclass
class MultiModalDataCollator:
    """
    Custom data collator for multi-modal remote sensing batches.

    Designed for use with HuggingFace Trainer and LLaVA-style models.

    The collator performs:
      1. Tokenises prompt + target text pairs (if not pre-tokenised).
      2. Pads token sequences to the batch maximum length.
      3. Creates attention masks.
      4. Masks prompt tokens in ``labels`` (only target tokens contribute to loss).
      5. Stacks image tensors (single or dual for CDVQA).

    Args:
        processor:      HuggingFace processor (AutoProcessor for LLaVA).
        max_length:     Maximum token sequence length. Longer sequences are truncated.
        padding_side:   'left' (required by LLaVA) or 'right'.
        label_pad_id:   Token ID used for masked (non-supervised) label positions.
        return_tensors: Output tensor type ('pt' for PyTorch).
    """

    processor: Any = None
    max_length: int = 2048
    padding_side: str = "left"
    label_pad_id: int = LABEL_IGNORE_INDEX
    return_tensors: str = "pt"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of dataset items into a training batch.

        Expected item keys (from dataset __getitem__):
          - 'image':      torch.Tensor (3, H, W)  OR  tuple of two tensors for CDVQA
          - 'prompt':     str — the input question / instruction
          - 'target':     str — the expected answer / caption
          - 'task_type':  str

        Returns a dict suitable for model.forward(**batch):
          - input_ids:       (B, L)
          - attention_mask:  (B, L)
          - pixel_values:    (B, 3, H, W) or (B, 6, H, W) for CDVQA
          - labels:          (B, L) with prompt positions masked as LABEL_IGNORE_INDEX
        """
        if self.processor is None:
            return self._collate_without_processor(batch)

        return self._collate_with_processor(batch)

    # ---------------------------------------------------------------------- #
    #  Collation WITH a HuggingFace processor (full training mode)            #
    # ---------------------------------------------------------------------- #

    def _collate_with_processor(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Tokenise, pad, and stack using the HuggingFace processor."""
        from transformers import AutoProcessor

        images = []
        input_ids_list = []
        label_ids_list = []

        # LLaVA chat template
        _PROMPT_TMPL = "USER: <image>\n{prompt}\nASSISTANT:"

        for item in batch:
            image = self._resolve_image(item)
            images.append(image)

            prompt_text = _PROMPT_TMPL.format(prompt=item.get("prompt", ""))
            target_text = str(item.get("target", ""))
            full_text = prompt_text + " " + target_text + self.processor.tokenizer.eos_token

            # Tokenise full sequence
            enc_full = self.processor.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            full_ids = enc_full["input_ids"].squeeze(0)

            # Tokenise prompt only to compute mask boundary
            enc_prompt = self.processor.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt_len = enc_prompt["input_ids"].shape[-1]

            # Labels: mask prompt tokens with LABEL_IGNORE_INDEX
            labels = full_ids.clone()
            labels[:prompt_len] = self.label_pad_id

            input_ids_list.append(full_ids)
            label_ids_list.append(labels)

        # Pad sequences
        pad_id = self.processor.tokenizer.pad_token_id or 0
        input_ids = self._pad_sequences(input_ids_list, pad_value=pad_id)
        labels    = self._pad_sequences(label_ids_list, pad_value=self.label_pad_id)
        attention_mask = (input_ids != pad_id).long()

        # Process images through the processor
        try:
            pixel_values = self.processor(
                images=images, return_tensors="pt"
            )["pixel_values"]
        except Exception as e:
            logger.warning(f"processor image encoding failed: {e}. Using raw tensors.")
            pixel_values = torch.stack([img if isinstance(img, torch.Tensor) else
                                        torch.zeros(3, 336, 336) for img in images])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

    # ---------------------------------------------------------------------- #
    #  Collation WITHOUT a processor (evaluation / stub mode)                 #
    # ---------------------------------------------------------------------- #

    def _collate_without_processor(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Minimal collation when no processor is provided.
        Stacks image tensors and groups other fields into lists.
        Useful for the EvaluationRunner which calls model.predict() per item.
        """
        images = []
        prompts = []
        targets = []
        task_types = []

        for item in batch:
            images.append(self._resolve_image(item))
            prompts.append(item.get("prompt", ""))
            targets.append(item.get("target", ""))
            task_types.append(item.get("task_type", "unknown"))

        # Stack image tensors if possible
        try:
            image_tensor = torch.stack([
                img if isinstance(img, torch.Tensor) else torch.zeros(3, 336, 336)
                for img in images
            ])
        except Exception:
            image_tensor = images

        return {
            "image": image_tensor,
            "prompt": prompts,
            "target": targets,
            "task_type": task_types,
        }

    # ---------------------------------------------------------------------- #
    #  Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _resolve_image(self, item: Dict[str, Any]) -> Any:
        """
        Resolve the image field from a dataset item.

        For CDVQA, 'image' is a tuple (pre, post); we concatenate along the
        channel dimension to produce a (6, H, W) tensor.
        """
        image = item.get("image")
        if isinstance(image, tuple) and len(image) == 2:
            pre, post = image
            if isinstance(pre, torch.Tensor) and isinstance(post, torch.Tensor):
                return torch.cat([pre, post], dim=0)  # (6, H, W)
            # PIL fallback — pick first image
            return pre
        return image

    def _pad_sequences(
        self,
        sequences: List[torch.Tensor],
        pad_value: int = 0,
    ) -> torch.Tensor:
        """
        Pad a list of 1-D tensors to the same length.

        Uses left-padding (required by LLaVA's causal attention).
        """
        max_len = max(s.shape[0] for s in sequences)
        padded = []
        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                if self.padding_side == "left":
                    padded_seq = torch.cat([
                        torch.full((pad_len,), pad_value, dtype=seq.dtype),
                        seq,
                    ])
                else:
                    padded_seq = torch.cat([
                        seq,
                        torch.full((pad_len,), pad_value, dtype=seq.dtype),
                    ])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        return torch.stack(padded)
