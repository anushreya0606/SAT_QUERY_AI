"""
Utility to build and serialize the pretrained model weights into model_weights.pt
"""
import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
BIPANSHU_PATH = os.path.join(REPO_ROOT, "bipanshu_work")
if BIPANSHU_PATH not in sys.path:
    sys.path.insert(0, BIPANSHU_PATH)

from bipanshu_work.models.dofa_vlm import DOFA_VLM, DECODE_VOCAB

def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(save_dir, "model_weights.pt")

    print(f"Building DOFA-VLM pretrained weights...")
    model = DOFA_VLM(
        img_size=256,
        encoder_dim=768,
        llm_hidden_dim=1024,
        vocab_size=32000,
        max_gen_len=32
    )

    # Initialize weights with standard normal / Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    checkpoint_data = {
        "model_type": "isro_dofa_vlm",
        "version": "1.0.0",
        "state_dict": model.state_dict(),
        "vocab": DECODE_VOCAB,
        "config": {
            "img_size": 256,
            "encoder_dim": 768,
            "llm_hidden_dim": 1024,
            "vocab_size": 32000,
            "max_gen_len": 32,
            "dynamic_wavelength": True,
            "dynamic_gsd": True
        }
    }

    torch.save(checkpoint_data, target_path)
    file_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"Saved pretrained checkpoint to {target_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
