"""
Unit tests for the CLI parser and interface.
"""

import pytest
from cli import build_parser


def test_cli_parser_commands():
    parser = build_parser()
    assert parser.prog == "isro-vlm"

    # Test eval subparser
    eval_args = parser.parse_args([
        "eval",
        "--dataset", "vrsbench",
        "--data-path", "datasets_raw/vrsbench",
        "--task", "vqa",
        "--dummy",
    ])
    assert eval_args.command == "eval"
    assert eval_args.dataset == "vrsbench"
    assert eval_args.dummy is True

    # Test train subparser
    train_args = parser.parse_args([
        "train",
        "--dataset", "cartosat",
        "--train-data", "datasets_raw/cartosat",
        "--epochs", "5",
        "--lr", "1e-4",
    ])
    assert train_args.command == "train"
    assert train_args.epochs == 5
    assert train_args.lr == 1e-4

    # Test preprocess subparser
    pre_args = parser.parse_args([
        "preprocess",
        "--input-dir", "raw/test",
        "--output-dir", "processed/test",
        "--sensor", "sar",
    ])
    assert pre_args.command == "preprocess"
    assert pre_args.sensor == "sar"

    # Test export subparser
    exp_args = parser.parse_args([
        "export",
        "--base-model", "llava-hf/llava-1.5-7b-hf",
        "--lora-ckpt", "checkpoints/test",
    ])
    assert exp_args.command == "export"
