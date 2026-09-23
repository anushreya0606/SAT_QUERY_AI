"""
Real Dataset Benchmark Evaluation & Prediction Runner
=====================================================
Evaluates the trained DOFA-VLM model against real satellite Earth observation datasets:
1. ISRO Bhoonidhi RISAT-1 C-Band SAR
2. ISRO Bhoonidhi Cartosat-2S (0.65m PAN / VNIR)
3. RSVQA Real Sentinel-2 Low-Resolution QA
4. CDVQA Bi-Temporal Change Detection

Usage:
    python scripts/run_real_benchmarks_prediction.py
"""

import os
import sys
import json
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
BIPANSHU_PATH = os.path.join(REPO_ROOT, "bipanshu_work")
if BIPANSHU_PATH not in sys.path:
    sys.path.insert(0, BIPANSHU_PATH)

from bipanshu_work.eval.harness import BenchmarkHarness, print_benchmark_table
from bipanshu_work.models.dofa_vlm import DOFA_VLM


def main():
    print("=" * 80)
    print(">>> RUNNING DOFA-VLM BENCHMARK EVALUATION ON REAL DATASETS")
    print("=" * 80)

    # 1. Initialize Harness & Model
    harness = BenchmarkHarness(log_path="traces/real_eval_trace.jsonl")
    model = DOFA_VLM()
    model.eval()

    benchmarks_to_evaluate = [
        {"name": "bhoonidhi_risat", "samples": 20, "desc": "ISRO RISAT-1 C-Band SAR Real Data"},
        {"name": "bhoonidhi_cartosat", "samples": 20, "desc": "ISRO Cartosat-2S 0.65m Optical Real Data"},
        {"name": "rsvqa_lr", "samples": 15, "desc": "RSVQA Sentinel-2 Real QA Pairs"},
        {"name": "cdvqa", "samples": 10, "desc": "CDVQA Real Bi-Temporal Change Detection"},
    ]

    all_results = []

    for b in benchmarks_to_evaluate:
        b_name = b["name"]
        print(f"\n[+] Loading Real Dataset: {b['desc']} ({b_name})...")
        try:
            ds = harness.load_dataset(b_name)
            res = harness.evaluate_model_on_dataset(
                model=model,
                model_name="DOFA-VLM (Trained)",
                benchmark_name=b_name,
                dataset=ds,
                max_samples=b["samples"],
            )
            all_results.append(res)
        except Exception as e:
            print(f"[-] Could not evaluate on {b_name}: {e}")

    # Print summary table
    if all_results:
        print_benchmark_table(all_results)

    # Save summary results
    out_json = "real_benchmark_run_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Detailed results saved to: {out_json}")


if __name__ == "__main__":
    main()
