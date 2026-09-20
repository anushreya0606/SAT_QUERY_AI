"""
Automated Downloader for GeoChat-Instruct and EuroSAT Datasets.
Downloads GeoChat_Instruct.json and EuroSAT into data/geochat and data/eurosat.
"""

import os
import sys
import time
import urllib.request

GEOCHAT_URL = "https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/resolve/main/GeoChat_Instruct.json"
EUROSAT_BASE = "https://huggingface.co/datasets/timm/eurosat-rgb/resolve/main"

EUROSAT_FILES = [
    ("data/train-00000-of-00001.parquet", "train.parquet"),
    ("data/test-00000-of-00001.parquet", "test.parquet"),
    ("data/validation-00000-of-00001.parquet", "val.parquet"),
]

def download_file(url: str, dest_path: str):
    """Download a file with progress reporting."""
    print(f"[*] Downloading {os.path.basename(dest_path)}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB
            
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                f.write(buffer)
                
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    speed = mb_down / max(time.time() - start_time, 0.01)
                    sys.stdout.write(f"\r    -> {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%) @ {speed:.2f} MB/s")
                    sys.stdout.flush()
            print()
        print(f"[OK] Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed downloading {url}: {e}")
        return False


def download_geochat(base_dir: str):
    target_dir = os.path.join(base_dir, "data", "geochat")
    os.makedirs(target_dir, exist_ok=True)
    dest_file = os.path.join(target_dir, "GeoChat_Instruct.json")
    
    print("\n" + "=" * 70)
    print(">>> 1. DOWNLOADING GEOCHAT-INSTRUCT (318K RS INSTRUCTION PAIRS)")
    print("=" * 70)
    
    if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1000000:
        print(f"[-] GeoChat_Instruct.json already exists ({os.path.getsize(dest_file):,} bytes). Skipping.")
    else:
        download_file(GEOCHAT_URL, dest_file)


def download_eurosat(base_dir: str):
    target_dir = os.path.join(base_dir, "data", "eurosat")
    os.makedirs(target_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print(">>> 2. DOWNLOADING EUROSAT (27,000 SENTINEL-2 SATELLITE PATCHES)")
    print("=" * 70)
    
    for remote_rel, local_fname in EUROSAT_FILES:
        dest_file = os.path.join(target_dir, local_fname)
        remote_url = f"{EUROSAT_BASE}/{remote_rel}"
        
        if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1000:
            print(f"[-] {local_fname} already exists ({os.path.getsize(dest_file):,} bytes). Skipping.")
            continue
            
        download_file(remote_url, dest_file)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Download GeoChat
    download_geochat(base_dir)
    
    # Download EuroSAT
    download_eurosat(base_dir)
    
    print("\n" + "=" * 70)
    print(">>> ALL DOWNLOADS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
