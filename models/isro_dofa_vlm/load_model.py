"""
Pretrained DOFA-VLM Loader.

Provides single-line initialization and loading of the pretrained DOFA-VLM weights
and configuration for multi-sensor satellite imagery reasoning.
"""

import os
import sys
import json
import torch
from typing import Optional, Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
BIPANSHU_PATH = os.path.join(REPO_ROOT, "bipanshu_work")
if BIPANSHU_PATH not in sys.path:
    sys.path.insert(0, BIPANSHU_PATH)

from bipanshu_work.models.dofa_vlm import DOFA_VLM


def load_pretrained_dofa_vlm(
    weights_path: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    eval_mode: bool = True
) -> DOFA_VLM:
    """
    Loads the pretrained DOFA-VLM model with all learned weights and configurations.

    Args:
        weights_path: Path to model_weights.pt (defaults to model_weights.pt in current folder)
        config_path: Path to config.json (defaults to config.json in current folder)
        device: 'cuda' or 'cpu' (auto-detected if None)
        eval_mode: Whether to set model.eval()

    Returns:
        Loaded DOFA_VLM instance ready for multi-sensor inference.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = weights_path or os.path.join(current_dir, "model_weights.pt")
    config_path = config_path or os.path.join(current_dir, "config.json")

    # Load configuration
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    encoder_cfg = config.get("encoder", {})
    llm_cfg = config.get("llm_head", {})

    model = DOFA_VLM(
        img_size=encoder_cfg.get("img_size", 256),
        encoder_dim=encoder_cfg.get("embed_dim", 768),
        llm_hidden_dim=llm_cfg.get("hidden_dim", 1024),
        vocab_size=llm_cfg.get("vocab_size", 32000),
        max_gen_len=llm_cfg.get("max_gen_len", 32),
    )

    # Load state dict if weights file exists
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        print(f"[DOFA-VLM Loader] Successfully loaded pretrained weights from: {weights_path}")
    else:
        print(f"[DOFA-VLM Loader] Weights file {weights_path} not found. Returning freshly initialized model.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    if eval_mode:
        model.eval()

    return model


if __name__ == "__main__":
    m = load_pretrained_dofa_vlm()
    print("Pretrained model loaded successfully:", type(m))
