import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from schema.trace import ExecutionTrace, ExecutionTraceStep
from eval.metrics import Evaluator


class EvaluationRunner:
    """
    Orchestrates end-to-end evaluation of a VLM on one or more datasets.

    Produces a strict JSON execution trace (see schema/trace.py) and aggregated
    metric scores written to the output directory.
    """

    def __init__(self, model: Any, dataset: Any, run_name: str = "eval_run"):
        self.model = model
        self.dataset = dataset
        self.run_name = run_name
        self.evaluator = Evaluator()
        self.trace = ExecutionTrace(run_id=str(uuid.uuid4()))

    def run(self, output_dir: str = "outputs") -> Dict[str, float]:
        """
        Run the full evaluation loop.

        Args:
            output_dir: Directory to write the JSON trace file.

        Returns:
            Dictionary of metric_name -> score.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Starting evaluation run: {self.run_name} (Run ID: {self.trace.run_id})")

        # Separate prediction/reference buckets per task type for correct metric dispatch
        caption_preds: List[str] = []
        caption_refs: List[List[str]] = []

        vqa_preds: List[Any] = []
        vqa_refs: List[Any] = []

        grounding_preds: List[List[float]] = []
        grounding_refs: List[List[float]] = []

        for idx, batch in enumerate(tqdm(self.dataset, desc="Evaluating")):
            task_type = batch.get("task_type", "unknown")

            # Strip non-serialisable values (raw images / tensors) before logging
            inputs_for_trace: Dict[str, str] = {
                k: str(v) for k, v in batch.items() if k not in ("image", "image_pre", "image_post")
            }

            prediction = self.model.predict(batch)

            step = ExecutionTraceStep(
                step_id=str(uuid.uuid4()),
                module="vlm_model",
                action="inference",
                inputs=inputs_for_trace,
                outputs={"prediction": str(prediction)},
                metadata={"batch_idx": idx, "task_type": task_type},
            )
            self.trace.add_step(step)

            # Route to the correct bucket
            if task_type == "captioning":
                caption_preds.append(str(prediction))
                caption_refs.append([batch.get("target", "")])
            elif task_type in ("vqa", "cdvqa"):
                vqa_preds.append(prediction)
                vqa_refs.append(batch.get("target", batch.get("answer", "")))
            elif task_type == "grounding":
                grounding_preds.append(prediction)
                grounding_refs.append(batch.get("target", []))

        # Aggregate metrics per task type
        print("Calculating metrics...")
        final_metrics: Dict[str, float] = {}

        if caption_preds:
            final_metrics.update(
                self.evaluator.calculate_text_metrics(caption_preds, caption_refs)
            )
        if vqa_preds:
            final_metrics.update(
                self.evaluator.calculate_classification_metrics(vqa_preds, vqa_refs)
            )
        if grounding_preds:
            final_metrics.update(
                self.evaluator.calculate_grounding_metrics(grounding_preds, grounding_refs)
            )

        self.trace.final_metrics = final_metrics
        self.trace.end_time = datetime.utcnow()

        # Persist trace
        trace_path = os.path.join(output_dir, f"{self.run_name}_trace.json")
        self.trace.save(trace_path)
        print(f"Evaluation complete. Trace saved to {trace_path}")
        print("Final Metrics:", final_metrics)

        return final_metrics
