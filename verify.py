"""
Pipeline smoke test for the ISRO/SAC Remote Sensing VLM evaluation harness.

Validates:
  1. Config system (Pydantic models)
  2. Schema (ExecutionTrace / ExecutionTraceStep)
  3. Evaluation runner with dummy model and dataset
  4. Metrics calculation (classification, text generation)
  5. Trace JSON serialisation and schema validity
  6. CLI import and help text
"""

import os
import sys
import json
import traceback

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"

failures = []

def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as e:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        failures.append((name, str(e)))


# --------------------------------------------------------------------------- #
print("\n=== ISRO/SAC VLM Pipeline Smoke Test ===\n")
# --------------------------------------------------------------------------- #

# 1. Config system
def test_config():
    from config import ProjectConfig, ModelConfig
    cfg = ProjectConfig.default()
    assert cfg.model.model_name == "llava-hf/llava-1.5-7b-hf"
    assert cfg.eval.batch_size == 1
    device = cfg.model.get_device()
    assert device in ("cuda", "cpu")
check("Config system - ProjectConfig defaults", test_config)

# 2. Schema
def test_schema():
    from schema.trace import ExecutionTrace, ExecutionTraceStep
    import uuid
    trace = ExecutionTrace(run_id=str(uuid.uuid4()))
    step = ExecutionTraceStep(
        step_id=str(uuid.uuid4()),
        module="test",
        action="verify",
        inputs={"x": 1},
        outputs={"y": 2},
    )
    trace.add_step(step)
    json_str = trace.to_json()
    parsed = json.loads(json_str)
    assert "run_id" in parsed
    assert len(parsed["steps"]) == 1
check("Schema - ExecutionTrace serialises to valid JSON", test_schema)

# 3. Dummy dataset + evaluation runner
def test_runner():
    from eval.runner import EvaluationRunner

    class _DummyModel:
        def predict(self, batch):
            t = batch.get("task_type", "vqa")
            if t == "captioning":
                return "A remote sensing image showing urban areas."
            elif t in ("vqa", "cdvqa"):
                return "Yes"
            elif t == "grounding":
                return [10.0, 10.0, 50.0, 50.0]
            return "dummy"

    class _DummyDataset:
        def __init__(self):
            self.items = [
                {
                    "task_type": "captioning",
                    "image": "placeholder",
                    "prompt": "Describe the image.",
                    "target": "A remote sensing image showing urban areas.",
                },
                {
                    "task_type": "vqa",
                    "image": "placeholder",
                    "prompt": "Is this urban?",
                    "target": "Yes",
                },
            ]
        def __iter__(self):
            return iter(self.items)

    runner = EvaluationRunner(_DummyModel(), _DummyDataset(), run_name="smoke_test")
    metrics = runner.run(output_dir="outputs")
    assert isinstance(metrics, dict), "metrics should be a dict"
check("EvaluationRunner - end-to-end dummy pipeline", test_runner)

# 4. Trace file written and valid JSON
def test_trace_file():
    trace_path = os.path.join("outputs", "smoke_test_trace.json")
    assert os.path.exists(trace_path), f"Trace file not found: {trace_path}"
    with open(trace_path) as f:
        data = json.load(f)
    assert "run_id" in data
    assert "steps" in data
    assert len(data["steps"]) == 2
    assert "final_metrics" in data
check("Trace file - written and schema-valid", test_trace_file)

# 5. Metrics module (classification)
def test_metrics_classification():
    from eval.metrics import Evaluator
    ev = Evaluator()
    result = ev.calculate_classification_metrics(
        predictions=["Yes", "No", "Yes"],
        references=["Yes", "Yes", "Yes"],
    )
    assert "accuracy" in result
    assert "f1_macro" in result
    assert 0.0 <= result["accuracy"] <= 1.0
check("Metrics - classification accuracy + F1", test_metrics_classification)

# 6. Metrics module (IoU / grounding)
def test_metrics_grounding():
    from eval.metrics import Evaluator
    ev = Evaluator()
    pred_boxes = [[0, 0, 50, 50], [10, 10, 60, 60]]
    ref_boxes  = [[0, 0, 50, 50], [20, 20, 70, 70]]
    result = ev.calculate_grounding_metrics(pred_boxes, ref_boxes)
    assert "mean_iou" in result
    assert result["mean_iou"] >= 0.0
check("Metrics - grounding IoU + Acc@0.5", test_metrics_grounding)

# 7. Data config path resolution
def test_data_config():
    from config import DataConfig
    dc = DataConfig()
    path = dc.get_dataset_path("vrsbench")
    assert "vrsbench" in path
check("DataConfig - dataset path resolution", test_data_config)

# 8. CLI imports cleanly
def test_cli_import():
    import importlib
    cli = importlib.import_module("cli")
    parser = cli.build_parser()
    assert parser is not None
check("CLI - imports cleanly and parser builds", test_cli_import)

# --------------------------------------------------------------------------- #
print()
if failures:
    print(f"{'='*40}")
    print(f"  FAILED: {len(failures)} test(s)")
    for name, err in failures:
        print(f"  [FAIL] {name}: {err}")
    print(f"{'='*40}\n")
    sys.exit(1)
else:
    print(f"{'='*40}")
    print(f"  All tests passed! Pipeline is healthy.")
    print(f"{'='*40}\n")
    print("  Next steps:")
    print("  * Run: python cli.py eval --dataset vrsbench --data-path <path> --dummy")
    print("  * Run: python cli.py train --dataset vrsbench --train-data <path>")
    print("  * Run: python cli.py preprocess --input-dir <raw> --output-dir <out> --sensor optical")

