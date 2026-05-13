#!/usr/bin/env python3
"""Import assertion silver-set rows into Neo4j for database smoke testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core.config import get_settings  # noqa: E402
from rag_core.neo4j_utils import open_neo4j_session  # noqa: E402

PHRASE_LABEL = "PhraseNode"
PASSAGE_LABEL = "PassageNode"
REL_TYPE = "MENTIONED_IN"


def load_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit and len(rows) >= limit:
            break
    return rows


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _validate_row(row: dict[str, Any], index: int) -> None:
    required = ("text", "entity", "label", "start", "end")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"Missing fields at row {index}: {', '.join(missing)}")
    text = str(row["text"])
    entity = str(row["entity"])
    start = int(row["start"])
    end = int(row["end"])
    if start < 0 or end > len(text) or start >= end:
        raise ValueError(f"Invalid offsets at row {index}: {start}/{end}")
    if text[start:end] != entity:
        raise ValueError(
            f"Offset validation failed at row {index}: "
            f"expected {entity}, got {text[start:end]}"
        )


def init_assertion_test_indexes(driver: Any) -> None:
    """Create small lookup indexes for isolated silver-set smoke tests."""
    with open_neo4j_session(driver) as session:
        session.run(
            f"CREATE INDEX assertion_phrase_dataset IF NOT EXISTS "
            f"FOR (n:{PHRASE_LABEL}) ON (n.assertion_dataset)"
        )
        session.run(
            f"CREATE INDEX assertion_passage_dataset IF NOT EXISTS "
            f"FOR (n:{PASSAGE_LABEL}) ON (n.assertion_dataset)"
        )


def import_silver_assertions(
    rows: list[dict[str, Any]],
    driver: Any,
    *,
    dataset: str = "assertion_silver_v1",
    batch_size: int = 100,
) -> dict[str, Any]:
    """Upsert silver-set rows as PhraseNode-[:MENTIONED_IN]->PassageNode records."""
    if not rows:
        return {"dataset": dataset, "imported": 0}
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    payload: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        _validate_row(row, index)
        text = str(row["text"])
        entity = str(row["entity"])
        source = str(row.get("source", ""))
        phrase_id = _stable_id("assertion-phrase", dataset, entity)
        passage_id = _stable_id("assertion-passage", dataset, text, entity, row["start"], row["end"])
        payload.append(
            {
                "phrase_id": phrase_id,
                "passage_id": passage_id,
                "entity": entity,
                "text": text,
                "chunk_id": str(row.get("source", passage_id)),
                "dataset": dataset,
                "source": source,
                "assertion_status": str(row["label"]),
                "weak_label": str(row.get("weak_label", "")),
                "model_label": str(row.get("model_label", "")),
                "rule_label": str(row.get("rule_label", "")),
                "model_confidence": float(row.get("model_confidence", 0.0) or 0.0),
                "rule_confidence": float(row.get("rule_confidence", 0.0) or 0.0),
                "needs_review": bool(row.get("needs_review", False)),
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "span_start": int(row["start"]),
                "span_end": int(row["end"]),
                "cue": str(row.get("cue", "")),
                "notes": str(row.get("notes", "")),
            }
        )

    imported = 0
    with open_neo4j_session(driver) as session:
        for offset in range(0, len(payload), batch_size):
            batch = payload[offset : offset + batch_size]
            session.run(
                f"""
                UNWIND $rows AS row
                MERGE (ph:{PHRASE_LABEL} {{id: row.phrase_id}})
                SET ph.name = row.entity,
                    ph.entity_type = 'MedicalEntity',
                    ph.assertion_dataset = row.dataset,
                    ph.source = 'assertion_silver'
                MERGE (pa:{PASSAGE_LABEL} {{id: row.passage_id}})
                SET pa.text = row.text,
                    pa.chunk_id = row.chunk_id,
                    pa.assertion_dataset = row.dataset,
                    pa.source = 'assertion_silver'
                MERGE (ph)-[r:{REL_TYPE}]->(pa)
                SET r.assertion_status = row.assertion_status,
                    r.weak_label = row.weak_label,
                    r.model_label = row.model_label,
                    r.rule_label = row.rule_label,
                    r.model_confidence = row.model_confidence,
                    r.rule_confidence = row.rule_confidence,
                    r.needs_review = row.needs_review,
                    r.confidence = row.confidence,
                    r.span_start = row.span_start,
                    r.span_end = row.span_end,
                    r.cue = row.cue,
                    r.notes = row.notes,
                    r.dataset = row.dataset,
                    r.source = row.source
                """,
                rows=batch,
            )
            imported += len(batch)
    return {"dataset": dataset, "imported": imported}


def verify_silver_assertions(driver: Any, *, dataset: str = "assertion_silver_v1") -> dict[str, Any]:
    """Return database counts for the isolated assertion silver-set dataset."""
    with open_neo4j_session(driver) as session:
        total = session.run(
            f"""
            MATCH (: {PHRASE_LABEL})-[r:{REL_TYPE}]->(:{PASSAGE_LABEL})
            WHERE r.dataset = $dataset
            RETURN count(r) AS total
            """.replace(": ", ":"),
            dataset=dataset,
        ).single()["total"]
        labels = session.run(
            f"""
            MATCH (: {PHRASE_LABEL})-[r:{REL_TYPE}]->(:{PASSAGE_LABEL})
            WHERE r.dataset = $dataset
            RETURN r.assertion_status AS label, count(r) AS count
            ORDER BY label
            """.replace(": ", ":"),
            dataset=dataset,
        )
        label_counts = {record["label"]: int(record["count"]) for record in labels}
        needs_review = session.run(
            f"""
            MATCH (: {PHRASE_LABEL})-[r:{REL_TYPE}]->(:{PASSAGE_LABEL})
            WHERE r.dataset = $dataset AND r.needs_review = true
            RETURN count(r) AS total
            """.replace(": ", ":"),
            dataset=dataset,
        ).single()["total"]

    return {
        "dataset": dataset,
        "total": int(total),
        "labels": label_counts,
        "needs_review": int(needs_review),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/assertion/silver_assertion.jsonl")
    parser.add_argument("--dataset", default="assertion_silver_v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--skip-indexes", action="store_true")
    args = parser.parse_args()

    from neo4j import GraphDatabase

    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        if not args.skip_indexes:
            init_assertion_test_indexes(driver)
        if not args.verify_only:
            summary = import_silver_assertions(
                load_jsonl(Path(args.input), limit=args.limit),
                driver,
                dataset=args.dataset,
                batch_size=args.batch_size,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(verify_silver_assertions(driver, dataset=args.dataset), ensure_ascii=False, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
