import numpy as np
from typing import List, Dict, Any, Union, Optional
try:
    import evaluate
except ImportError:
    evaluate = None
from sklearn.metrics import accuracy_score, f1_score


class Evaluator:
    """
    Comprehensive evaluation metrics for the ISRO/SAC Remote Sensing VLM.
    Supports text generation (captioning), classification (VQA),
    visual grounding, and change detection tasks.
    """

    def __init__(self):
        # Lazy load external evaluate metrics on demand to avoid network hangs
        self.bleu = None
        self.meteor = None
        self.rouge = None
        self.cider = None

    @staticmethod
    def _safe_load_metric(name: str):
        """Safely load an evaluate metric, returning None on failure."""
        if evaluate is None:
            return None
        try:
            return evaluate.load(name)
        except Exception as e:
            print(f"Warning: Could not load metric '{name}': {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Text Generation Metrics (Captioning / Generative VQA)
    # ------------------------------------------------------------------ #

    def calculate_text_metrics(
        self,
        predictions: List[str],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """
        Calculate NLP metrics for captioning / generative VQA.

        Args:
            predictions: List of predicted strings.
            references: List of lists of reference strings
                        (multiple references per prediction).

        Returns:
            Dict of metric_name -> score.
        """
        metrics: Dict[str, float] = {}

        # BLEU-4
        if self.bleu:
            try:
                bleu_results = self.bleu.compute(
                    predictions=predictions, references=references
                )
                metrics["bleu4"] = bleu_results.get("bleu", 0.0)
            except Exception as e:
                print(f"Error computing BLEU with evaluate: {e}")

        # NLTK BLEU fallback if evaluate is unavailable or failed
        if "bleu4" not in metrics:
            try:
                from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
                sf = SmoothingFunction().method1
                tokenized_preds = [p.lower().split() for p in predictions]
                tokenized_refs = [[r.lower().split() for r in ref_list] for ref_list in references]
                metrics["bleu4"] = float(corpus_bleu(tokenized_refs, tokenized_preds, smoothing_function=sf))
            except Exception as e:
                print(f"Error computing NLTK BLEU fallback: {e}")
                metrics["bleu4"] = 0.0

        # METEOR
        if self.meteor:
            try:
                meteor_results = self.meteor.compute(
                    predictions=predictions, references=references
                )
                metrics["meteor"] = meteor_results.get("meteor", 0.0)
            except Exception as e:
                print(f"Error computing METEOR: {e}")

        # ROUGE-L
        if self.rouge:
            try:
                rouge_results = self.rouge.compute(
                    predictions=predictions,
                    references=[refs[0] for refs in references],  # single ref for rouge
                )
                metrics["rouge_l"] = rouge_results.get("rougeL", 0.0)
            except Exception as e:
                print(f"Error computing ROUGE-L: {e}")

        # CIDEr (optional, requires pycocoevalcap)
        if self.cider is not None:
            try:
                # CIDEr expects {id: [sentence]} dicts
                gts = {i: refs for i, refs in enumerate(references)}
                res = {i: [pred] for i, pred in enumerate(predictions)}
                cider_score, _ = self.cider.compute_score(gts, res)
                metrics["cider"] = float(cider_score)
            except Exception as e:
                print(f"Error computing CIDEr: {e}")

        return metrics

    # ------------------------------------------------------------------ #
    #  Classification Metrics (Categorical VQA / Scene Classification)
    # ------------------------------------------------------------------ #

    def calculate_classification_metrics(
        self,
        predictions: List[Union[str, int]],
        references: List[Union[str, int]],
    ) -> Dict[str, float]:
        """
        Calculate classification metrics for categorical VQA or scene classification.

        Args:
            predictions: List of predicted labels (str or int).
            references: List of ground truth labels (str or int).

        Returns:
            Dict with 'accuracy' and 'f1_macro'.
        """
        acc = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average="macro", zero_division=0)
        return {"accuracy": acc, "f1_macro": f1}

    def calculate_per_type_accuracy(
        self,
        predictions: List[Union[str, int]],
        references: List[Union[str, int]],
        question_types: List[str],
    ) -> Dict[str, float]:
        """
        Calculate per-question-type accuracy for VQA tasks.

        Args:
            predictions: List of predicted labels.
            references: List of ground truth labels.
            question_types: List of question type strings (same length).

        Returns:
            Dict with per-type accuracy, e.g. {'accuracy_presence': 0.85, ...}
        """
        type_preds: Dict[str, list] = {}
        type_refs: Dict[str, list] = {}
        for pred, ref, qtype in zip(predictions, references, question_types):
            type_preds.setdefault(qtype, []).append(pred)
            type_refs.setdefault(qtype, []).append(ref)

        metrics: Dict[str, float] = {}
        for qtype in sorted(type_preds.keys()):
            acc = accuracy_score(type_refs[qtype], type_preds[qtype])
            metrics[f"accuracy_{qtype}"] = acc

        # Overall
        metrics["accuracy_overall"] = accuracy_score(references, predictions)
        return metrics

    # ------------------------------------------------------------------ #
    #  Visual Grounding Metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        Format: [x_min, y_min, x_max, y_max]
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(box1_area + box2_area - intersection_area)

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def calculate_grounding_metrics(
        self,
        pred_boxes: List[List[float]],
        ref_boxes: List[List[float]],
        iou_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Calculate Acc@0.5 and mean IoU for visual grounding tasks.

        Args:
            pred_boxes: List of predicted bounding boxes [x1, y1, x2, y2].
            ref_boxes: List of ground truth bounding boxes.
            iou_threshold: IoU threshold for accuracy calculation.

        Returns:
            Dict with 'acc_at_05' and 'mean_iou'.
        """
        if not pred_boxes or not ref_boxes:
            return {"acc_at_05": 0.0, "mean_iou": 0.0}

        ious = [self.calculate_iou(p, r) for p, r in zip(pred_boxes, ref_boxes)]
        mean_iou = float(np.mean(ious))
        acc_at_05 = sum(1 for iou in ious if iou >= iou_threshold) / len(ious)

        thresh_str = str(iou_threshold).replace(".", "")
        return {
            f"acc_at_{thresh_str}": acc_at_05,
            "mean_iou": mean_iou,
        }

    # ------------------------------------------------------------------ #
    #  Change Detection Metrics (CDVQA)
    # ------------------------------------------------------------------ #

    def calculate_change_detection_metrics(
        self,
        predictions: List[Union[str, int]],
        references: List[Union[str, int]],
        question_types: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Calculate metrics specific to change detection VQA.

        Args:
            predictions: List of predicted answers.
            references: List of ground truth answers.
            question_types: Optional list of question types for per-type breakdown.

        Returns:
            Dict of metric_name -> score.
        """
        metrics = self.calculate_classification_metrics(predictions, references)
        # Prefix with 'cd_' to distinguish from standard VQA
        metrics = {f"cd_{k}": v for k, v in metrics.items()}

        if question_types:
            per_type = self.calculate_per_type_accuracy(
                predictions, references, question_types
            )
            per_type = {f"cd_{k}": v for k, v in per_type.items()}
            metrics.update(per_type)

        return metrics
