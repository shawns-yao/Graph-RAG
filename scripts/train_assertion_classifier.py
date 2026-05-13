#!/usr/bin/env python3
"""Fine-tune a small Chinese BERT model for assertion-status classification."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_graph_rag.indexing.assertion_classifier import mark_entity  # noqa: E402
from agentic_graph_rag.indexing.assertion_rules import ASSERTION_LABELS  # noqa: E402


def _load_optional_deps():
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with:\n"
            "  pip install transformers torch scikit-learn\n"
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    return rows[:limit]


class AssertionDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = {label: index for index, label in enumerate(ASSERTION_LABELS)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        marked = mark_entity(str(row["text"]), str(row["entity"]))
        encoded = self.tokenizer(
            marked,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        encoded["labels"] = self.label_to_id[str(row["label"])]
        return encoded


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    predictions, labels = eval_pred
    pred_ids = predictions.argmax(axis=-1)
    accuracy = float((pred_ids == labels).mean())
    return {"accuracy": accuracy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/assertion/weak_train.jsonl")
    parser.add_argument("--dev", default="data/assertion/gold_reviewed_assertion.jsonl")
    parser.add_argument("--output-dir", default="models/assertion_tinybert")
    parser.add_argument("--model-name", default="uer/chinese_roberta_L-2_H-128")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-dev-examples", type=int, default=0)
    args = parser.parse_args()

    (
        _torch,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = _load_optional_deps()

    train_rows = load_jsonl(Path(args.train))
    dev_path = Path(args.dev)
    dev_rows = load_jsonl(dev_path) if dev_path.exists() else []
    if not dev_rows:
        dev_rows = train_rows[: max(1, len(train_rows) // 10)]
    train_rows = limit_rows(train_rows, args.max_train_examples)
    dev_rows = limit_rows(dev_rows, args.max_dev_examples)
    if not train_rows:
        raise SystemExit(f"No training rows found in {args.train}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ASSERTION_LABELS),
        id2label={index: label for index, label in enumerate(ASSERTION_LABELS)},
        label2id={label: index for index, label in enumerate(ASSERTION_LABELS)},
    )

    train_dataset = AssertionDataset(train_rows, tokenizer, args.max_length)
    dev_dataset = AssertionDataset(dev_rows, tokenizer, args.max_length)
    output_dir = Path(args.output_dir)

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "logging_steps": 20,
        "report_to": [],
    }
    strategy_arg = _strategy_arg_name(TrainingArguments)
    training_kwargs[strategy_arg] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_map = {
        "label_to_id": {label: index for index, label in enumerate(ASSERTION_LABELS)},
        "id_to_label": {str(index): label for index, label in enumerate(ASSERTION_LABELS)},
        "metrics": metrics,
    }
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _strategy_arg_name(training_arguments_cls: Any) -> str:
    params = inspect.signature(training_arguments_cls.__init__).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


if __name__ == "__main__":
    main()
