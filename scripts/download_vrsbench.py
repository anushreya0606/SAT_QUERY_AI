"""
Automatic Downloader for VRSBench Dataset from HuggingFace (xiang709/VRSBench).
Downloads evaluation JSON annotations and validation images directly into data/vrsbench/.
"""

import os
import sys
import time
import zipfile
import urllib.request

HF_BASE_URL = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main"

FILES_TO_DOWNLOAD = [
    ("VRSBench_EVAL_vqa.json", "test_vqa.json"),
    ("VRSBench_EVAL_Cap.json", "test_caption.json"),
    ("VRSBench_EVAL_referring.json", "test_grounding.json"),
    ("VRSBench_train.json", "train.json"),
    ("Images_val.zip", "Images_val.zip"),
]

def download_file(url: str, dest_path: str):
    """Download a file with progress reporting."""
    print(f"[*] Downloading {os.path.basename(dest_path)}...")
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


def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "vrsbench"))
    os.makedirs(target_dir, exist_ok=True)
    images_dir = os.path.join(target_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 80)
    print(">>> AUTOMATIC VRSBENCH DATASET DOWNLOADER")
    print(f"[*] Destination: {target_dir}")
    print("=" * 80)

    for remote_fname, local_fname in FILES_TO_DOWNLOAD:
        dest_file = os.path.join(target_dir, local_fname)
        remote_url = f"{HF_BASE_URL}/{remote_fname}"

        # If already exists and has reasonable size, skip
        if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1000:
            print(f"[-] {local_fname} already exists ({os.path.getsize(dest_file):,} bytes). Skipping.")
            continue

        download_file(remote_url, dest_file)

        # If it's a zip file, extract it
        if local_fname.endswith(".zip") and os.path.exists(dest_file):
            print(f"[*] Extracting {local_fname} into {target_dir}...")
            try:
                with zipfile.ZipFile(dest_file, "r") as zip_ref:
                    zip_ref.extractall(target_dir)
                print(f"[OK] Extracted {local_fname}")
            except Exception as e:
                print(f"[!] Warning extracting {local_fname}: {e}")

    print("\n" + "=" * 80)
    print(">>> VRSBENCH DOWNLOAD COMPLETE!")
    print(f"Files in {target_dir}:")
    for f in os.listdir(target_dir):
        fpath = os.path.join(target_dir, f)
        if os.path.isfile(fpath):
            print(f"  - {f} ({os.path.getsize(fpath):,} bytes)")
        else:
            print(f"  - {f}/ (directory)")
    print("=" * 80)


if __name__ == "__main__":
    main()
