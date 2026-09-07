"""
isro-vlm CLI — Command-line interface for the ISRO/SAC Remote Sensing VLM.

Entry point registered in pyproject.toml:
    [project.scripts]
    isro-vlm = "cli:main"

Subcommands:
    eval    — Run evaluation on a dataset, producing a JSON trace + metrics
    train   — Fine-tune the VLM with LoRA on remote sensing data
    preprocess — Batch-preprocess a directory of raw imagery
    export  — Export / merge a LoRA checkpoint back to full model weights

Example usage:
    isro-vlm eval --dataset vrsbench --data-path ./datasets_raw/vrsbench --task captioning
    isro-vlm eval --dataset risat --data-path ./datasets_raw/risat --task vqa --dummy
    isro-vlm train --config configs/lora_finetune.json
    isro-vlm preprocess --input-dir ./raw --output-dir ./processed --sensor optical
    isro-vlm export --base-model llava-hf/llava-1.5-7b-hf --lora-ckpt ./checkpoints/final_lora_adapter
"""

import argparse
import json
import logging
import os
import sys

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("isro-vlm")


# =========================================================================== #
#  Subcommand: eval                                                            #
# =========================================================================== #

def cmd_eval(args: argparse.Namespace):
    """Run evaluation on a dataset and write a JSON execution trace."""
    from config import ProjectConfig, ModelConfig, EvalConfig, DataConfig
    from eval.runner import EvaluationRunner
    from models.baseline import BaselineVLM

    # Load or build config
    if args.config:
        cfg = ProjectConfig.from_json(args.config)
    else:
        cfg = ProjectConfig.default()

    # CLI overrides
    if args.dummy:
        cfg.model.use_dummy = True
    if args.model:
        cfg.model.model_name = args.model
    if getattr(args, "adapter_path", None):
        cfg.model.adapter_path = args.adapter_path
    if args.output_dir:
        cfg.eval.output_dir = args.output_dir
    if args.run_name:
        cfg.eval.run_name = args.run_name

    # Build dataset
    dataset = _build_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        task_type=args.task,
        split=args.split,
        image_size=cfg.model.image_size,
    )

    # Build model
    model = BaselineVLM.from_config(cfg.model)
    logger.info(f"Model: {model}")

    # Run evaluation
    runner = EvaluationRunner(model, dataset, run_name=cfg.eval.run_name)
    metrics = runner.run(output_dir=cfg.eval.output_dir)

    # Pretty-print metrics table
    print("\n" + "=" * 50)
    print(f"  EVALUATION RESULTS — {cfg.eval.run_name}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<30} {v:.4f}")
    print("=" * 50)
    return metrics


def _build_dataset(
    dataset_name: str,
    data_path: str,
    task_type: str = "captioning",
    split: str = "test",
    image_size: int = 336,
):
    """Factory that returns the correct Dataset class for the given dataset name."""
    name = dataset_name.lower()

    if name == "vrsbench":
        from eval.datasets.vrsbench import VRSBenchDataset
        return VRSBenchDataset(data_path, split=split, task_type=task_type, image_size=image_size)

    elif name in ("rsvqa_lr", "rsvqa-lr", "rsvqa"):
        from eval.datasets.rsvqa import RSVQADataset
        return RSVQADataset(data_path, split=split, variant="lr", image_size=image_size)

    elif name in ("rsvqa_hr", "rsvqa-hr"):
        from eval.datasets.rsvqa import RSVQADataset
        return RSVQADataset(data_path, split=split, variant="hr", image_size=image_size)

    elif name == "cdvqa":
        from eval.datasets.cdvqa import CDVQADataset
        return CDVQADataset(data_path, split=split, image_size=image_size)

    elif name == "cartosat":
        from data.cartosat import CartosatDataset
        return CartosatDataset(data_path, split=split, task_type=task_type, image_size=image_size)

    elif name == "risat":
        from data.risat import RISATDataset
        return RISATDataset(data_path, split=split, task_type=task_type, image_size=image_size)

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: vrsbench, rsvqa_lr, rsvqa_hr, cdvqa, cartosat, risat"
        )


# =========================================================================== #
#  Subcommand: train                                                           #
# =========================================================================== #

def cmd_train(args: argparse.Namespace):
    """Fine-tune the VLM with LoRA on remote sensing datasets."""
    from config import ProjectConfig
    from train.trainer import LoRATrainer

    if args.config:
        cfg = ProjectConfig.from_json(args.config)
    else:
        cfg = ProjectConfig.default()

    # CLI overrides
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.model:
        cfg.model.model_name = args.model

    # Build datasets
    if not args.train_data or not args.dataset:
        logger.error("--train-data and --dataset are required for training.")
        sys.exit(1)

    train_dataset = _build_dataset(
        dataset_name=args.dataset,
        data_path=args.train_data,
        task_type=args.task,
        split="train",
        image_size=cfg.model.image_size,
    )
    val_dataset = None
    if args.val_data:
        val_dataset = _build_dataset(
            dataset_name=args.dataset,
            data_path=args.val_data,
            task_type=args.task,
            split="val",
            image_size=cfg.model.image_size,
        )

    # Train
    trainer = LoRATrainer(cfg.training, cfg.model)
    result = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_name=cfg.model.model_name,
        resume_from_checkpoint=args.resume,
    )
    logger.info(f"Training complete. Final loss: {result.training_loss:.4f}")


# =========================================================================== #
#  Subcommand: preprocess                                                      #
# =========================================================================== #

def cmd_preprocess(args: argparse.Namespace):
    """Batch-preprocess a directory of raw imagery."""
    from data.preprocess import batch_preprocess

    logger.info(
        f"Preprocessing '{args.input_dir}' → '{args.output_dir}' "
        f"[sensor={args.sensor}, size={args.size}]"
    )
    output_paths = batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sensor_type=args.sensor,
        size=args.size,
    )
    logger.info(f"Done. {len(output_paths)} files written to '{args.output_dir}'.")


# =========================================================================== #
#  Subcommand: export                                                          #
# =========================================================================== #

def cmd_export(args: argparse.Namespace):
    """Merge a LoRA adapter into the base model and export full weights."""
    from train.trainer import LoRATrainer

    logger.info(
        f"Merging LoRA adapter '{args.lora_ckpt}' into base model '{args.base_model}'..."
    )
    model, processor = LoRATrainer.load_from_checkpoint(
        base_model_name=args.base_model,
        lora_checkpoint_dir=args.lora_ckpt,
        device=args.device,
    )
    output_dir = args.output or os.path.join(args.lora_ckpt, "merged_model")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Merged model saved to '{output_dir}'.")


# =========================================================================== #
#  Argument Parser                                                             #
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isro-vlm",
        description="ISRO/SAC Remote Sensing Vision-Language Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with dummy model (no GPU needed):
  isro-vlm eval --dataset vrsbench --data-path ./data/vrsbench --task captioning --dummy

  # Evaluate with real LLaVA model:
  isro-vlm eval --dataset cartosat --data-path ./data/cartosat --task vqa

  # Fine-tune with LoRA:
  isro-vlm train --dataset vrsbench --train-data ./data/vrsbench --task captioning

  # Preprocess Cartosat-2S imagery:
  isro-vlm preprocess --input-dir ./raw/cartosat --output-dir ./processed/cartosat --sensor optical

  # Export merged LoRA checkpoint:
  isro-vlm export --base-model llava-hf/llava-1.5-7b-hf --lora-ckpt ./checkpoints/final_lora_adapter
        """,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------ eval
    eval_p = subparsers.add_parser("eval", help="Run evaluation on a dataset")
    eval_p.add_argument("--dataset", required=True,
                        choices=["vrsbench", "rsvqa_lr", "rsvqa_hr", "cdvqa", "cartosat", "risat"],
                        help="Dataset to evaluate on")
    eval_p.add_argument("--data-path", required=True,
                        help="Path to the dataset root directory")
    eval_p.add_argument("--task", default="captioning",
                        choices=["captioning", "vqa", "grounding"],
                        help="Task type (default: captioning)")
    eval_p.add_argument("--split", default="test", help="Dataset split (default: test)")
    eval_p.add_argument("--model", default=None,
                        help="HuggingFace model ID (overrides config)")
    eval_p.add_argument("--adapter-path", default=None,
                        help="Path to trained LoRA adapter checkpoint (e.g. checkpoints/final_lora_adapter)")
    eval_p.add_argument("--output-dir", default="outputs",
                        help="Directory to write trace JSON (default: outputs/)")
    eval_p.add_argument("--run-name", default="eval_run",
                        help="Name tag for this run (default: eval_run)")
    eval_p.add_argument("--config", default=None,
                        help="Path to a ProjectConfig JSON file")
    eval_p.add_argument("--dummy", action="store_true",
                        help="Use dummy model predictions (for pipeline testing)")
    eval_p.set_defaults(func=cmd_eval)

    # ----------------------------------------------------------------- train
    train_p = subparsers.add_parser("train", help="Fine-tune VLM with LoRA")
    train_p.add_argument("--dataset", required=True,
                         choices=["vrsbench", "rsvqa_lr", "rsvqa_hr", "cdvqa", "cartosat", "risat"],
                         help="Dataset to train on")
    train_p.add_argument("--train-data", required=True,
                         help="Path to training data root")
    train_p.add_argument("--val-data", default=None,
                         help="Path to validation data root (optional)")
    train_p.add_argument("--task", default="captioning",
                         choices=["captioning", "vqa", "grounding"],
                         help="Task type (default: captioning)")
    train_p.add_argument("--model", default=None,
                         help="HuggingFace model ID (overrides config)")
    train_p.add_argument("--output-dir", default=None,
                         help="Checkpoint output directory (overrides config)")
    train_p.add_argument("--epochs", type=int, default=None,
                         help="Number of training epochs (overrides config)")
    train_p.add_argument("--lr", type=float, default=None,
                         help="Learning rate (overrides config)")
    train_p.add_argument("--config", default=None,
                         help="Path to a ProjectConfig JSON file")
    train_p.add_argument("--resume", default=None,
                         help="Resume from this checkpoint path")
    train_p.set_defaults(func=cmd_train)

    # ------------------------------------------------------------- preprocess
    pre_p = subparsers.add_parser("preprocess", help="Batch-preprocess raw imagery")
    pre_p.add_argument("--input-dir", required=True,
                       help="Root directory of raw images")
    pre_p.add_argument("--output-dir", required=True,
                       help="Root directory for processed PNGs")
    pre_p.add_argument("--sensor", default="optical",
                       choices=["optical", "sar", "multispectral"],
                       help="Sensor/preprocessing type (default: optical)")
    pre_p.add_argument("--size", type=int, default=336,
                       help="Output spatial resolution in pixels (default: 336)")
    pre_p.set_defaults(func=cmd_preprocess)

    # -------------------------------------------------------------- export
    exp_p = subparsers.add_parser("export", help="Merge LoRA adapter and export full model")
    exp_p.add_argument("--base-model", required=True,
                       help="HuggingFace base model ID (e.g. llava-hf/llava-1.5-7b-hf)")
    exp_p.add_argument("--lora-ckpt", required=True,
                       help="Path to saved LoRA adapter directory")
    exp_p.add_argument("--output", default=None,
                       help="Output directory for merged model (default: <lora-ckpt>/merged_model)")
    exp_p.add_argument("--device", default=None,
                       help="Device: 'cuda' or 'cpu' (auto-detected if not set)")
    exp_p.set_defaults(func=cmd_export)

    return parser


# =========================================================================== #
#  Entry Point                                                                 #
# =========================================================================== #

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
