"""Assertion status classifier wrappers with rule fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_core.config import get_settings

from agentic_graph_rag.indexing.assertion_rules import (
    ASSERTION_LABELS,
    AssertionLabel,
    classify_assertion_by_rules,
)


@dataclass(frozen=True)
class AssertionPrediction:
    label: AssertionLabel
    confidence: float
    cue: str = ""
    model: str = "rules"


@dataclass(frozen=True)
class AssertionClassifierConfig:
    model_path: str = ""
    model_name: str = "huawei-noah/TinyBERT_4L_zh"
    threshold: float = 0.75
    max_length: int = 160


class RuleAssertionClassifier:
    def predict(self, text: str, entity: str) -> AssertionPrediction:
        decision = classify_assertion_by_rules(text, entity)
        return AssertionPrediction(
            label=decision.label,
            confidence=decision.confidence,
            cue=decision.cue,
            model="rules",
        )


class TinyBertAssertionClassifier:
    def __init__(self, config: AssertionClassifierConfig) -> None:
        if not config.model_path:
            raise ValueError("assertion model path is required")
        model_dir = Path(config.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"assertion model path does not exist: {model_dir}")

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TinyBERT assertion classifier requires optional dependency "
                "`transformers`. Install with: pip install transformers torch"
            ) from exc

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id_to_label = _load_id_to_label(model_dir)

    def predict(self, text: str, entity: str) -> AssertionPrediction:
        rule = classify_assertion_by_rules(text, entity)
        if rule.label != "affirmed" and rule.confidence >= self.config.threshold:
            return AssertionPrediction(rule.label, rule.confidence, rule.cue, "rules")

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TinyBERT assertion classifier requires `torch`. "
                "Install with: pip install torch"
            ) from exc

        marked = mark_entity(text, entity)
        encoded = self.tokenizer(
            marked,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]
            confidence, label_id = torch.max(probs, dim=-1)

        label = self.id_to_label.get(int(label_id), "affirmed")
        if confidence.item() < self.config.threshold:
            return AssertionPrediction(rule.label, rule.confidence, rule.cue, "rules")
        return AssertionPrediction(
            label=label,  # type: ignore[arg-type]
            confidence=float(confidence.item()),
            cue="",
            model="tinybert",
        )


def mark_entity(text: str, entity: str) -> str:
    start = text.find(entity)
    if start < 0:
        return text
    end = start + len(entity)
    return f"{text[:start]}[E]{text[start:end]}[/E]{text[end:]}"


def load_assertion_classifier(settings: Any | None = None):
    cfg = settings or get_settings()
    indexing = cfg.indexing if hasattr(cfg, "indexing") else cfg
    if not getattr(indexing, "assertion_model_enabled", False):
        return RuleAssertionClassifier()

    model_path = getattr(indexing, "assertion_model_path", "")
    if not model_path:
        return RuleAssertionClassifier()

    config = AssertionClassifierConfig(
        model_path=model_path,
        model_name=getattr(indexing, "assertion_model_name", "huawei-noah/TinyBERT_4L_zh"),
        threshold=float(getattr(indexing, "assertion_model_threshold", 0.75)),
        max_length=int(getattr(indexing, "assertion_model_max_length", 160)),
    )
    return TinyBertAssertionClassifier(config)


def _load_id_to_label(model_dir: Path) -> dict[int, AssertionLabel]:
    path = model_dir / "label_map.json"
    if not path.exists():
        return {index: label for index, label in enumerate(ASSERTION_LABELS)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("id_to_label", payload)
    return {int(index): label for index, label in labels.items()}
