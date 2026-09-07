"""
Benchmark Dataset Downloader for ISRO/SAC Remote Sensing VLM.

Automates downloading public remote sensing benchmark datasets:
  - VRSBench (Visual Remote Sensing Benchmark)
  - RSVQA (Remote Sensing Visual Question Answering - LR / HR)
  - CDVQA (Change Detection VQA)

Usage:
  python scripts/download_benchmarks.py --dataset vrsbench --output-dir datasets_raw/vrsbench
  python scripts/download_benchmarks.py --dataset rsvqa_lr --output-dir datasets_raw/rsvqa_lr
  python scripts/download_benchmarks.py --dataset all --output-dir datasets_raw
"""

import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("downloader")

DATASET_URLS = {
    "vrsbench": {
        "repo_id": "VRSBench/VRSBench",
        "description": "VRSBench: Visual Remote Sensing Benchmark (captioning, VQA, grounding)",
        "hf_type": "dataset"
    },
    "rsvqa_lr": {
        "repo_id": "RSVQA/RSVQA-LR",
        "description": "RSVQA Low-Resolution (Sentinel-2 base)",
        "hf_type": "dataset"
    },
    "rsvqa_hr": {
        "repo_id": "RSVQA/RSVQA-HR",
        "description": "RSVQA High-Resolution (Aerial optical base)",
        "hf_type": "dataset"
    },
    "cdvqa": {
        "repo_id": "CDVQA/CDVQA",
        "description": "CDVQA: Change Detection VQA",
        "hf_type": "dataset"
    }
}


def download_hf_dataset(repo_id: str, local_dir: str):
    """Download dataset repository snapshot from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading '{repo_id}' to '{local_dir}' from Hugging Face Hub...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"Successfully downloaded '{repo_id}' to '{local_dir}'.")
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. Run: pip install huggingface_hub"
        )
    except Exception as e:
        logger.error(f"Failed to download {repo_id}: {e}")
        logger.info("If authentication is required, run `huggingface-cli login` first.")


def main():
    parser = argparse.ArgumentParser(description="Download Remote Sensing VLM Datasets")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_URLS.keys()) + ["all"],
        help="Dataset identifier to download"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets_raw",
        help="Root directory where datasets will be stored"
    )

    args = parser.parse_args()

    if args.dataset == "all":
        targets = list(DATASET_URLS.keys())
    else:
        targets = [args.dataset]

    for d in targets:
        info = DATASET_URLS[d]
        logger.info(f"=== {info['description']} ===")
        target_path = os.path.join(args.output_dir, d)
        os.makedirs(target_path, exist_ok=True)
        download_hf_dataset(info["repo_id"], target_path)


if __name__ == "__main__":
    main()
