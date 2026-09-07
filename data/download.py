"""
Dataset download script for the ISRO/SAC VLM project.
Downloads public benchmark datasets from HuggingFace and other sources.
"""

import os
import sys
import argparse
import shutil
from typing import Optional


DATASETS = {
    "rsvqa_lr": {
        "description": "RSVQA Low-Resolution (Sentinel-2 based)",
        "source": "HuggingFace: Mehdi/rsvqa-lr (or manual)",
        "hf_id": None,  # Set if a reliable HuggingFace mirror exists
    },
    "rsvqa_hr": {
        "description": "RSVQA High-Resolution",
        "source": "Official: https://rsvqa.sylvainlobry.com/",
        "hf_id": None,
    },
    "vrsbench": {
        "description": "Visual Remote Sensing Benchmark",
        "source": "GitHub / HuggingFace",
        "hf_id": None,
    },
    "cdvqa": {
        "description": "Change Detection Visual Question Answering",
        "source": "GitHub: YZHJessica/CDVQA",
        "hf_id": None,
    },
}


def download_with_huggingface(hf_id: str, output_dir: str, split: Optional[str] = None):
    """Download a dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset

        print(f"  Downloading from HuggingFace: {hf_id}")
        kwargs = {"cache_dir": output_dir}
        if split:
            kwargs["split"] = split

        dataset = load_dataset(hf_id, **kwargs)
        print(f"  ✓ Downloaded successfully to {output_dir}")
        return dataset
    except Exception as e:
        print(f"  ✗ Failed to download from HuggingFace: {e}")
        return None


def download_rsvqa_lr(output_dir: str):
    """Download RSVQA Low-Resolution dataset."""
    dataset_dir = os.path.join(output_dir, "rsvqa_lr")
    os.makedirs(dataset_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("RSVQA Low-Resolution (Sentinel-2)")
    print("=" * 60)

    # Try HuggingFace first
    try:
        from datasets import load_dataset

        print("  Attempting HuggingFace download...")
        dataset = load_dataset("Mehdi/rsvqa-lr", cache_dir=dataset_dir)
        print("  ✓ RSVQA-LR downloaded from HuggingFace")

        # Save to expected directory structure
        images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"  Dataset cached at: {dataset_dir}")
        return
    except Exception:
        pass

    # Fallback: manual download instructions
    print("  Could not auto-download. Please download manually:")
    print("  1. Visit: https://rsvqa.sylvainlobry.com/")
    print("  2. Download the 'Low Resolution' dataset")
    print(f"  3. Extract to: {dataset_dir}")
    print("  Expected structure:")
    print(f"    {dataset_dir}/images/")
    print(f"    {dataset_dir}/questions.json")
    print(f"    {dataset_dir}/answers.json")


def download_rsvqa_hr(output_dir: str):
    """Download RSVQA High-Resolution dataset."""
    dataset_dir = os.path.join(output_dir, "rsvqa_hr")
    os.makedirs(dataset_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("RSVQA High-Resolution")
    print("=" * 60)
    print("  This dataset requires manual download:")
    print("  1. Visit: https://rsvqa.sylvainlobry.com/")
    print("  2. Download the 'High Resolution' dataset (requires registration)")
    print(f"  3. Extract to: {dataset_dir}")
    print("  Expected structure:")
    print(f"    {dataset_dir}/images/")
    print(f"    {dataset_dir}/questions.json")
    print(f"    {dataset_dir}/answers.json")


def download_vrsbench(output_dir: str):
    """Download VRSBench dataset."""
    dataset_dir = os.path.join(output_dir, "vrsbench")
    os.makedirs(dataset_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("VRSBench (Visual Remote Sensing Benchmark)")
    print("=" * 60)

    # Try HuggingFace
    try:
        from datasets import load_dataset

        print("  Attempting HuggingFace download...")
        dataset = load_dataset("lhrs/VRSBench", cache_dir=dataset_dir)
        print("  ✓ VRSBench downloaded from HuggingFace")

        images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        print(f"  Dataset cached at: {dataset_dir}")
        return
    except Exception:
        pass

    # Fallback instructions
    print("  Could not auto-download. Please download manually:")
    print("  1. Check HuggingFace for 'VRSBench' or visit the official repo")
    print("  2. Download images and annotation JSON files")
    print(f"  3. Extract to: {dataset_dir}")
    print("  Expected structure:")
    print(f"    {dataset_dir}/images/")
    print(f"    {dataset_dir}/test_caption.json")
    print(f"    {dataset_dir}/test_vqa.json")
    print(f"    {dataset_dir}/test_grounding.json")


def download_cdvqa(output_dir: str):
    """Download CDVQA dataset."""
    dataset_dir = os.path.join(output_dir, "cdvqa")
    os.makedirs(dataset_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CDVQA (Change Detection Visual Question Answering)")
    print("=" * 60)

    # Try HuggingFace
    try:
        from datasets import load_dataset

        print("  Attempting HuggingFace download...")
        dataset = load_dataset("YZHJessica/CDVQA", cache_dir=dataset_dir)
        print("  ✓ CDVQA downloaded from HuggingFace")
        print(f"  Dataset cached at: {dataset_dir}")
        return
    except Exception:
        pass

    # Fallback instructions
    print("  Could not auto-download. Please download manually:")
    print("  1. Visit: https://github.com/YZHJessica/CDVQA")
    print("  2. Follow the download instructions in the README")
    print(f"  3. Extract to: {dataset_dir}")
    print("  Expected structure:")
    print(f"    {dataset_dir}/images/pre/")
    print(f"    {dataset_dir}/images/post/")
    print(f"    {dataset_dir}/annotations/test.json")


def print_bhoonidhi_instructions():
    """Print instructions for obtaining ISRO satellite data."""
    print("\n" + "=" * 60)
    print("ISRO Satellite Data (Cartosat-2S / RISAT)")
    print("=" * 60)
    print("  These datasets are NOT publicly downloadable.")
    print("  You must obtain them through ISRO's Bhoonidhi portal:")
    print()
    print("  1. Register at: https://bhoonidhi.nrsc.gov.in/")
    print("  2. Search for Cartosat-2S (optical, ~0.6m resolution)")
    print("     and/or RISAT-2B (SAR, C-band)")
    print("  3. Submit a data request for your area of interest")
    print("  4. Download and place data in your datasets directory")
    print()
    print("  NOTE: Cartosat-2S provides high-res optical imagery.")
    print("        RISAT provides Synthetic Aperture Radar (SAR) data.")
    print("        Both require preprocessing (see data/preprocessing.py)")


def list_datasets():
    """List all available datasets and their status."""
    print("\nAvailable datasets:")
    print("-" * 60)
    for name, info in DATASETS.items():
        print(f"  {name:15s} — {info['description']}")
        print(f"  {'':15s}   Source: {info['source']}")
    print()
    print("  cartosat/risat — ISRO proprietary (see --bhoonidhi)")


def download_datasets(output_dir: str, dataset: Optional[str] = None):
    """
    Downloads datasets for the ISRO/SAC VLM project.

    Args:
        output_dir: Directory to save downloaded data.
        dataset: Specific dataset to download, or None for all.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Download directory: {os.path.abspath(output_dir)}")

    download_fns = {
        "rsvqa_lr": download_rsvqa_lr,
        "rsvqa_hr": download_rsvqa_hr,
        "vrsbench": download_vrsbench,
        "cdvqa": download_cdvqa,
    }

    if dataset:
        if dataset in download_fns:
            download_fns[dataset](output_dir)
        else:
            print(f"Unknown dataset: {dataset}")
            list_datasets()
            sys.exit(1)
    else:
        for name, fn in download_fns.items():
            fn(output_dir)

    print_bhoonidhi_instructions()
    print("\n" + "=" * 60)
    print("Download script finished.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets for the ISRO/SAC VLM project."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets_raw",
        help="Directory to save downloaded data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(DATASETS.keys()),
        help="Download a specific dataset (default: all).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets.",
    )
    parser.add_argument(
        "--bhoonidhi",
        action="store_true",
        help="Print ISRO Bhoonidhi download instructions.",
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.bhoonidhi:
        print_bhoonidhi_instructions()
    else:
        download_datasets(args.output_dir, args.dataset)
