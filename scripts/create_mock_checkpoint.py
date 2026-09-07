"""
Utility to generate a valid mock LoRA checkpoint directory.

Allows immediate local testing of:
  - isro-vlm export
  - checkpoint loading in app.py and BaselineVLM
  - verification that the pipeline handles adapter directories properly
"""

import os
import json
import torch


def create_mock_lora_checkpoint(output_dir: str = "checkpoints/final_lora_adapter"):
    """Creates a mock PEFT LoRA adapter directory with valid config and weights."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. adapter_config.json
    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": "llava-hf/llava-1.5-7b-hf",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "revision": None,
        "target_modules": [
            "q_proj",
            "v_proj"
        ],
        "task_type": "CAUSAL_LM"
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # 2. README.md
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("# ISRO Remote Sensing LoRA Adapter\nTrained on VRSBench and Cartosat-2S remote sensing benchmarks.\n")

    # 3. Dummy adapter weights (safetensors or pt)
    dummy_weights = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros((16, 4096), dtype=torch.float16),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros((4096, 16), dtype=torch.float16),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.zeros((16, 4096), dtype=torch.float16),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.zeros((4096, 16), dtype=torch.float16),
    }

    try:
        from safetensors.torch import save_file
        save_file(dummy_weights, os.path.join(output_dir, "adapter_model.safetensors"))
    except ImportError:
        torch.save(dummy_weights, os.path.join(output_dir, "adapter_model.bin"))

    print(f"Mock LoRA adapter checkpoint created at '{output_dir}'.")


if __name__ == "__main__":
    create_mock_lora_checkpoint()
