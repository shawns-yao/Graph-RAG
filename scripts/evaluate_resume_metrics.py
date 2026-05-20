"""Evaluate resume-grade metrics from the local medical benchmark graph.

The script is intentionally deterministic:
- no LLM calls
- no embedding API calls
- no writes to Neo4j

It measures what can be supported by the current local benchmark:
- skeleton deep-extraction cost proxy
- gold entity coverage
- relation recall and false-positive reduction vs a co-occurrence baseline
- 3-hop graph answerability
- existing benchmark vector/cypher accuracy snapshot

Metrics with tiny denominators or 100% values are marked as not recommended for
resume copy even though the raw values are preserved in the JSON output.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_graph_rag.indexing.skeleton import (
    _rank_chunks_for_skeleton_selection,
    build_knn_graph,
    compute_pagerank,
    resolve_skeleton_beta,
    select_skeletal_chunks,
)
from rag_core.config import get_settings
from rag_core.models import Chunk

DEFAULT_GOLD = ROOT / "test" / "medical_benchmark" / "eval_gold" / "medical_resume_gold.json"
DEFAULT_BENCHMARK = ROOT / "test" / "medical_benchmark" / "results" / "benchmark_results.json"
DEFAULT_QUESTIONS = ROOT / "test" / "medical_benchmark" / "questions_master.json"
DEFAULT_OUTPUT = ROOT / "test" / "medical_benchmark" / "results" / "resume_metrics.json"
DEFAULT_BOOTSTRAP_RUNS = 200
DEFAULT_BOOTSTRAP_SEED = 20260520


def _norm(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "").casefold())


def _surfaces(item: dict[str, Any]) -> list[str]:
    values = [str(item.get("name", "")), *(str(alias) for alias in item.get("aliases", []))]
    return [value for value in values if value]


def _contains_surface(text: str, surfaces: list[str]) -> bool:
    normalized = _norm(text)
    return any(_norm(surface) and _norm(surface) in normalized for surface in surfaces)


def _pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _metric(value: float, denominator: int, *, recommended: bool = True, reason: str = "") -> dict[str, Any]:
    if denominator < 20:
        recommended = False
        reason = reason or "denominator below 20"
    if math.isclose(value, 100.0):
        recommended = False
        reason = reason or "100% metric is likely too brittle for resume copy"
    return {
        "value": round(value, 2),
        "denominator": denominator,
        "resume_recommended": recommended,
        "reason": reason,
    }


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9/+.-]*|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?", text)
    stopwords = {"什么", "应该", "如何", "多少", "多久", "患者", "哪些", "是否", "的是", "什么是"}
    return [token for token in tokens if token not in stopwords]


def _load_gold(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {}


def _load_chunks(session: Any) -> list[Chunk]:
    rows = session.run(
        """
        MATCH (c:RagChunk)
        RETURN c.id AS id,
               coalesce(c.content, "") AS content,
               coalesce(c.context, "") AS context,
               coalesce(c.embedding, []) AS embedding,
               coalesce(c.metadata, {}) AS metadata
        ORDER BY c.id
        """
    )
    chunks: list[Chunk] = []
    for row in rows:
        chunks.append(
            Chunk(
                id=str(row["id"]),
                content=str(row["content"] or ""),
                context=str(row["context"] or ""),
                embedding=list(row["embedding"] or []),
                metadata=_metadata_dict(row["metadata"]),
            )
        )
    return chunks


def _load_graph(session: Any) -> dict[str, Any]:
    phrases = [
        {
            "name": str(row["name"] or ""),
            "entity_type": str(row["entity_type"] or ""),
            "aliases": list(row["aliases"] or []),
        }
        for row in session.run(
            """
            MATCH (p:PhraseNode)
            RETURN p.name AS name, p.entity_type AS entity_type, coalesce(p.aliases, []) AS aliases
            """
        )
    ]
    rels = [
        {
            "source": str(row["source"] or ""),
            "relation_type": str(row["relation_type"] or ""),
            "target": str(row["target"] or ""),
        }
        for row in session.run(
            """
            MATCH (a:PhraseNode)-[r:RELATED_TO]->(b:PhraseNode)
            RETURN a.name AS source, coalesce(r.relation_type, "") AS relation_type, b.name AS target
            """
        )
    ]
    passages = [
        {
            "id": str(row["id"] or ""),
            "text": str(row["text"] or ""),
        }
        for row in session.run(
            """
            MATCH (p:PassageNode)
            RETURN p.id AS id, coalesce(p.text, "") AS text
            """
        )
    ]
    return {"phrases": phrases, "relationships": rels, "passages": passages}


def _load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("questions", payload if isinstance(payload, list) else []))


def _select_skeleton(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    embeddings = [chunk.embedding for chunk in chunks]
    if not chunks or any(not emb for emb in embeddings):
        return [], chunks
    graph = build_knn_graph(chunks, embeddings)
    pagerank = compute_pagerank(graph)
    return select_skeletal_chunks(chunks, pagerank)


def _skeleton_selection_trace(chunks: list[Chunk]) -> dict[str, Any]:
    embeddings = [chunk.embedding for chunk in chunks]
    if not chunks or any(not emb for emb in embeddings):
        return {
            "selection_method": "unavailable: missing embeddings",
            "beta": 0.0,
            "formula": "not applied",
            "selected": [],
        }

    graph = build_knn_graph(chunks, embeddings)
    pagerank = compute_pagerank(graph)
    beta = resolve_skeleton_beta(chunks)
    ranked_scores = _rank_chunks_for_skeleton_selection(chunks, pagerank)
    skeletal, _ = select_skeletal_chunks(chunks, pagerank, beta=beta)
    selected_ids = {chunk.id for chunk in skeletal}
    ranked = sorted(ranked_scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "selection_method": (
            "formula: KNN graph -> PageRank -> blended score "
            "(PageRank + entity density + medical section prior + hard-fact signal) "
            "-> greedy diversity selection"
        ),
        "manual_selection": False,
        "beta": round(beta, 4),
        "formula": "n_skeletal = max(1, int(total_chunks * beta)); small docs may keep at least 2",
        "selected_count": len(skeletal),
        "selected": [
            {
                "rank": rank + 1,
                "index": index,
                "chunk_id": chunks[index].id,
                "score": round(score, 6),
                "selected": chunks[index].id in selected_ids,
            }
            for rank, (index, score) in enumerate(ranked)
        ],
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[int(position)], 2)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def _summarize_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "p05": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": round(mean, 2),
        "stdev": round(math.sqrt(variance), 2),
        "min": round(min(values), 2),
        "p05": _percentile(values, 0.05),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 2),
    }


def _bootstrap_skeleton_stability(
    chunks: list[Chunk],
    gold_entities: list[dict[str, Any]],
    *,
    runs: int = DEFAULT_BOOTSTRAP_RUNS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    valid_chunks = [chunk for chunk in chunks if chunk.embedding]
    if len(valid_chunks) < 4 or runs <= 0:
        return {"enabled": False, "reason": "not enough chunks with embeddings"}

    rng = random.Random(seed)
    sample_size = max(4, int(round(len(valid_chunks) * 0.75)))
    sample_size = min(sample_size, len(valid_chunks))
    rows: list[dict[str, Any]] = []
    cost_reductions: list[float] = []
    entity_coverages: list[float] = []

    for run_index in range(runs):
        sample = rng.sample(valid_chunks, sample_size)
        skeletal, peripheral = _select_skeleton(sample)
        coverage = _baseline_entity_coverage(gold_entities, skeletal)["coverage"]
        cost_reduction = _pct(len(sample) - len(skeletal), len(sample))
        rows.append(
            {
                "run": run_index + 1,
                "sample_chunks": len(sample),
                "skeletal_chunks": len(skeletal),
                "peripheral_chunks": len(peripheral),
                "cost_reduction": cost_reduction,
                "skeleton_entity_coverage": coverage,
            }
        )
        cost_reductions.append(cost_reduction)
        entity_coverages.append(coverage)

    return {
        "enabled": True,
        "method": "bootstrap subsampling without replacement from the current Neo4j chunk set",
        "runs": runs,
        "seed": seed,
        "sample_size": sample_size,
        "source_chunk_count": len(valid_chunks),
        "independent_corpus": False,
        "warning": "Repeated runs measure selection stability on the current corpus, not production-scale generalization.",
        "cost_reduction": _summarize_distribution(cost_reductions),
        "skeleton_entity_coverage": _summarize_distribution(entity_coverages),
        "first_10_runs": rows[:10],
    }


def _baseline_entity_candidates(skeletal_chunks: list[Chunk]) -> set[str]:
    candidates: set[str] = set()
    text = "\n".join(chunk.enriched_content for chunk in skeletal_chunks)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9/+.-]{1,24}|[\u4e00-\u9fffA-Za-z0-9/+.-]{2,24}", text):
        cleaned = token.strip(" ，。；：:()（）[]【】")
        if len(cleaned) >= 2:
            candidates.add(_norm(cleaned))
    return candidates


def _entity_coverage(gold_entities: list[dict[str, Any]], graph_phrases: list[dict[str, Any]]) -> dict[str, Any]:
    hits = []
    misses = []
    phrase_texts = [
        " ".join([phrase["name"], *(str(alias) for alias in phrase.get("aliases", []))])
        for phrase in graph_phrases
    ]
    for entity in gold_entities:
        found = any(_contains_surface(text, _surfaces(entity)) for text in phrase_texts)
        (hits if found else misses).append(entity["name"])
    return {
        "hits": len(hits),
        "total": len(gold_entities),
        "coverage": _pct(len(hits), len(gold_entities)),
        "misses": misses,
    }


def _entity_surface_present(item: dict[str, Any], graph_phrases: list[dict[str, Any]]) -> bool:
    phrase_texts = [
        " ".join([phrase["name"], *(str(alias) for alias in phrase.get("aliases", []))])
        for phrase in graph_phrases
    ]
    return any(_contains_surface(text, _surfaces(item)) for text in phrase_texts)


def _entity_judged_accuracy(
    positive_entities: list[dict[str, Any]],
    negative_entities: list[dict[str, Any]],
    graph_phrases: list[dict[str, Any]],
) -> dict[str, Any]:
    true_positives = []
    false_negatives = []
    true_negatives = []
    false_positives = []

    for entity in positive_entities:
        found = _entity_surface_present(entity, graph_phrases)
        (true_positives if found else false_negatives).append(entity["name"])

    for entity in negative_entities:
        found = _entity_surface_present(entity, graph_phrases)
        (false_positives if found else true_negatives).append(entity["name"])

    total = len(positive_entities) + len(negative_entities)
    correct = len(true_positives) + len(true_negatives)
    return {
        "true_positives": len(true_positives),
        "false_negatives": len(false_negatives),
        "true_negatives": len(true_negatives),
        "false_positives": len(false_positives),
        "total": total,
        "accuracy": _pct(correct, total),
        "precision": _pct(len(true_positives), len(true_positives) + len(false_positives)),
        "recall": _pct(len(true_positives), len(true_positives) + len(false_negatives)),
        "false_positive_items": false_positives,
        "false_negative_items": false_negatives,
    }


def _baseline_entity_coverage(
    gold_entities: list[dict[str, Any]],
    skeletal_chunks: list[Chunk],
) -> dict[str, Any]:
    text = "\n".join(chunk.enriched_content for chunk in skeletal_chunks)
    hits = [entity["name"] for entity in gold_entities if _contains_surface(text, _surfaces(entity))]
    candidates = _baseline_entity_candidates(skeletal_chunks)
    gold_hit_norms = {
        _norm(entity["name"])
        for entity in gold_entities
        if _contains_surface(text, _surfaces(entity))
    }
    precision = _pct(len(gold_hit_norms), len(candidates)) if candidates else 0.0
    recall = _pct(len(hits), len(gold_entities))
    f1 = 0.0 if precision + recall == 0 else round(2 * precision * recall / (precision + recall), 2)
    return {
        "hits": len(hits),
        "total": len(gold_entities),
        "coverage": recall,
        "candidate_count": len(candidates),
        "precision_proxy": precision,
        "f1_proxy": f1,
    }


def _rel_matches(rel: dict[str, str], source: str, target: str, relation: str | None = None) -> bool:
    source_hit = _norm(source) in _norm(rel["source"]) or _norm(rel["source"]) in _norm(source)
    target_hit = _norm(target) in _norm(rel["target"]) or _norm(rel["target"]) in _norm(target)
    if not (source_hit and target_hit):
        return False
    if relation is None:
        return True
    rel_type = _norm(rel["relation_type"])
    gold_type = _norm(relation)
    return bool(rel_type and (rel_type == gold_type or rel_type in gold_type or gold_type in rel_type))


def _relation_recall(
    gold_relations: list[dict[str, Any]],
    graph_rels: list[dict[str, str]],
) -> dict[str, Any]:
    hits = []
    misses = []
    for item in gold_relations:
        matched = any(
            _rel_matches(rel, item["source"], item["target"], item.get("relation"))
            for rel in graph_rels
        )
        (hits if matched else misses).append(
            f"{item['source']} --{item.get('relation', '')}--> {item['target']}"
        )
    return {
        "hits": len(hits),
        "total": len(gold_relations),
        "recall": _pct(len(hits), len(gold_relations)),
        "misses": misses,
    }


def _passage_cooccurs(passages: list[dict[str, str]], source: str, target: str) -> bool:
    for passage in passages:
        text = passage["text"]
        if _norm(source) in _norm(text) and _norm(target) in _norm(text):
            return True
    return False


def _false_positive_rates(
    negative_relations: list[dict[str, Any]],
    graph_rels: list[dict[str, str]],
    passages: list[dict[str, str]],
) -> dict[str, Any]:
    baseline_fp = []
    graph_fp = []
    for item in negative_relations:
        label = f"{item['source']} --X--> {item['target']}"
        if _passage_cooccurs(passages, item["source"], item["target"]):
            baseline_fp.append(label)
        if any(_rel_matches(rel, item["source"], item["target"]) for rel in graph_rels):
            graph_fp.append(label)

    total = len(negative_relations)
    baseline_rate = _pct(len(baseline_fp), total)
    graph_rate = _pct(len(graph_fp), total)
    return {
        "baseline_false_positives": len(baseline_fp),
        "graph_false_positives": len(graph_fp),
        "total": total,
        "baseline_false_positive_rate": baseline_rate,
        "graph_false_positive_rate": graph_rate,
        "absolute_reduction_pp": round(baseline_rate - graph_rate, 2),
        "relative_reduction": round((baseline_rate - graph_rate) / baseline_rate * 100.0, 2)
        if baseline_rate
        else 0.0,
        "baseline_fp_items": baseline_fp,
        "graph_fp_items": graph_fp,
    }


def _has_path(session: Any, source: str, target: str, max_hops: int) -> bool:
    row = session.run(
        f"""
        MATCH (a:PhraseNode), (b:PhraseNode)
        WHERE a.name CONTAINS $source AND b.name CONTAINS $target
        MATCH path = (a)-[:RELATED_TO*1..{max_hops}]-(b)
        RETURN count(path) > 0 AS found
        LIMIT 1
        """,
        source=source,
        target=target,
    ).single()
    return bool(row and row["found"])


def _multi_hop_metrics(session: Any, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_hits = []
    graph_hits = []
    misses = []
    for task in tasks:
        one_hop = _has_path(session, task["source"], task["target"], 1)
        three_hop = _has_path(session, task["source"], task["target"], 3)
        if one_hop:
            baseline_hits.append(task["id"])
        if three_hop:
            graph_hits.append(task["id"])
        else:
            misses.append(task["id"])
    total = len(tasks)
    baseline_rate = _pct(len(baseline_hits), total)
    graph_rate = _pct(len(graph_hits), total)
    return {
        "baseline_1hop_hits": len(baseline_hits),
        "graph_3hop_hits": len(graph_hits),
        "total": total,
        "baseline_1hop_accuracy": baseline_rate,
        "graph_3hop_accuracy": graph_rate,
        "absolute_gain_pp": round(graph_rate - baseline_rate, 2),
        "misses": misses,
    }


def _score_text(query_tokens: list[str], text: str) -> float:
    normalized = _norm(text)
    if not query_tokens or not normalized:
        return 0.0
    hits = sum(1 for token in query_tokens if _norm(token) and _norm(token) in normalized)
    numeric_hits = sum(1 for token in query_tokens if re.fullmatch(r"\d+(?:\.\d+)?", token) and token in text)
    return hits / len(query_tokens) + numeric_hits * 0.2


def _top_texts_by_score(query: str, texts: list[str], top_k: int) -> list[str]:
    tokens = _tokenize(query)
    ranked = sorted(
        ((text, _score_text(tokens, text)) for text in texts),
        key=lambda item: item[1],
        reverse=True,
    )
    return [text for text, score in ranked[:top_k] if score > 0]


def _bm25_like_texts(query: str, texts: list[str], top_k: int) -> list[str]:
    anchors = _tokenize(query)
    ranked = []
    for text in texts:
        normalized = _norm(text)
        exact_hits = sum(1 for anchor in anchors if _norm(anchor) and _norm(anchor) in normalized)
        proximity_bonus = 0.0
        if exact_hits >= 2:
            positions = [normalized.find(_norm(anchor)) for anchor in anchors if _norm(anchor) in normalized]
            if positions:
                proximity_bonus = 1.0 / (1.0 + ((max(positions) - min(positions)) / 250.0))
        ranked.append((text, exact_hits + proximity_bonus))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [text for text, score in ranked[:top_k] if score > 0]


def _graph_like_texts(question: dict[str, Any], graph: dict[str, Any], top_k: int) -> list[str]:
    query = str(question.get("query", ""))
    anchors = _tokenize(query)[:6]
    related_terms: set[str] = set(anchors)
    for rel in graph["relationships"]:
        source = str(rel["source"])
        target = str(rel["target"])
        if any(_norm(anchor) and _norm(anchor) in _norm(source) for anchor in anchors):
            related_terms.add(target)
        if any(_norm(anchor) and _norm(anchor) in _norm(target) for anchor in anchors):
            related_terms.add(source)

    texts = [str(item["text"]) for item in graph["passages"]]
    ranked = []
    for text in texts:
        score = sum(1 for term in related_terms if _norm(term) and _norm(term) in _norm(text))
        ranked.append((text, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [text for text, score in ranked[:top_k] if score > 0]


def _question_answerable(question: dict[str, Any], evidence_texts: list[str]) -> bool:
    keywords = [str(keyword) for keyword in question.get("keywords", []) if str(keyword).strip()]
    if not keywords:
        return False
    evidence = "\n".join(evidence_texts)
    hits = sum(1 for keyword in keywords if _norm(keyword) in _norm(evidence))
    return hits / len(keywords) >= 0.5


def _local_retrieval_answerability(
    questions: list[dict[str, Any]],
    chunks: list[Chunk],
    graph: dict[str, Any],
    *,
    single_channel_top_k: int = 3,
) -> dict[str, Any]:
    chunk_texts = [chunk.enriched_content for chunk in chunks]
    rows = []
    by_mode = {mode: {"hits": 0, "total": 0} for mode in ("vector_like", "bm25_like", "graph_like", "fusion")}
    by_mode_type: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))

    for question in questions:
        qtype = str(question.get("query_type", "unknown"))
        vector_texts = _top_texts_by_score(str(question.get("query", "")), chunk_texts, single_channel_top_k)
        bm25_texts = _bm25_like_texts(str(question.get("query", "")), chunk_texts, single_channel_top_k)
        graph_texts = _graph_like_texts(question, graph, single_channel_top_k)
        fusion_texts = []
        seen = set()
        for text in [*graph_texts, *bm25_texts, *vector_texts]:
            key = _norm(text[:240])
            if key and key not in seen:
                seen.add(key)
                fusion_texts.append(text)
            if len(fusion_texts) >= single_channel_top_k * 3:
                break
        mode_texts = {
            "vector_like": vector_texts,
            "bm25_like": bm25_texts,
            "graph_like": graph_texts,
            "fusion": fusion_texts,
        }
        row = {"id": question.get("id"), "query_type": qtype}
        for mode, texts in mode_texts.items():
            ok = _question_answerable(question, texts)
            row[mode] = ok
            by_mode[mode]["total"] += 1
            by_mode[mode]["hits"] += int(ok)
            by_mode_type[mode][qtype]["total"] += 1
            by_mode_type[mode][qtype]["hits"] += int(ok)
        rows.append(row)

    mode_summary = {
        mode: {
            "hits": data["hits"],
            "total": data["total"],
            "answerability": _pct(data["hits"], data["total"]),
        }
        for mode, data in by_mode.items()
    }
    type_summary = {
        mode: {
            qtype: {
                "hits": data["hits"],
                "total": data["total"],
                "answerability": _pct(data["hits"], data["total"]),
            }
            for qtype, data in per_type.items()
        }
        for mode, per_type in by_mode_type.items()
    }
    best_single = max(
        mode_summary[mode]["answerability"]
        for mode in ("vector_like", "bm25_like", "graph_like")
    )
    fusion_gain = round(mode_summary["fusion"]["answerability"] - best_single, 2)
    multi_hop_fusion = type_summary.get("fusion", {}).get("multi_hop", {"answerability": 0.0, "total": 0})
    multi_hop_vector = type_summary.get("vector_like", {}).get("multi_hop", {"answerability": 0.0})
    return {
        "single_channel_top_k": single_channel_top_k,
        "fusion_top_k_budget": single_channel_top_k * 3,
        "mode_summary": mode_summary,
        "type_summary": type_summary,
        "fusion_gain_vs_best_single_pp": fusion_gain,
        "multi_hop_fusion_answerability": multi_hop_fusion["answerability"],
        "multi_hop_gain_vs_vector_pp": round(
            multi_hop_fusion["answerability"] - multi_hop_vector["answerability"],
            2,
        ),
        "rows": rows,
    }


def _load_benchmark_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("mode", ""))].append(row)
    out = {"available": True, "modes": {}}
    for mode, mode_rows in sorted(by_mode.items()):
        passed = sum(1 for row in mode_rows if row.get("kw_pass") is True)
        out["modes"][mode] = {
            "passed": passed,
            "total": len(mode_rows),
            "accuracy": _pct(passed, len(mode_rows)),
            "errors": sum(1 for row in mode_rows if row.get("error")),
        }
    if "vector" in out["modes"] and "cypher" in out["modes"]:
        out["cypher_vs_vector_gain_pp"] = round(
            out["modes"]["cypher"]["accuracy"] - out["modes"]["vector"]["accuracy"],
            2,
        )
    return out


def evaluate(
    gold_path: Path,
    benchmark_path: Path,
    questions_path: Path,
    *,
    bootstrap_runs: int = DEFAULT_BOOTSTRAP_RUNS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    load_dotenv(ROOT / ".env")
    gold = _load_gold(gold_path)
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))

    with driver.session(database=cfg.neo4j.database) as session:
        chunks = _load_chunks(session)
        graph = _load_graph(session)
        skeletal, peripheral = _select_skeleton(chunks)
        skeleton_trace = _skeleton_selection_trace(chunks)
        bootstrap = _bootstrap_skeleton_stability(
            chunks,
            gold["entities"],
            runs=bootstrap_runs,
            seed=bootstrap_seed,
        )
        questions = _load_questions(questions_path)

        entity_graph = _entity_coverage(gold["entities"], graph["phrases"])
        entity_judged = _entity_judged_accuracy(
            gold["entities"],
            gold.get("negative_entities", []),
            graph["phrases"],
        )
        entity_baseline = _baseline_entity_coverage(gold["entities"], skeletal)
        relation_graph = _relation_recall(gold["positive_relations"], graph["relationships"])
        fp = _false_positive_rates(gold["negative_relations"], graph["relationships"], graph["passages"])
        multi_hop = _multi_hop_metrics(session, gold["multi_hop_tasks"])
        retrieval_answerability = _local_retrieval_answerability(questions, chunks, graph)

    driver.close()

    chunk_total = len(chunks)
    skeletal_count = len(skeletal)
    cost_reduction = _pct(chunk_total - skeletal_count, chunk_total)
    entity_gain = round(entity_graph["coverage"] - entity_baseline["coverage"], 2)

    benchmark = _load_benchmark_snapshot(benchmark_path)

    return {
        "metadata": {
            "gold_path": str(gold_path),
            "benchmark_path": str(benchmark_path),
            "questions_path": str(questions_path),
            "corpus_chunks": chunk_total,
            "graph_phrase_nodes": len(graph["phrases"]),
            "graph_related_to_edges": len(graph["relationships"]),
            "graph_passages": len(graph["passages"]),
            "bootstrap_runs": bootstrap_runs,
            "bootstrap_seed": bootstrap_seed,
        },
        "metrics": {
            "skeleton_deep_extraction_cost_reduction": _metric(
                cost_reduction,
                chunk_total,
                reason="small corpus; report only with corpus size",
            ),
            "graph_entity_coverage": _metric(entity_graph["coverage"], entity_graph["total"]),
            "entity_extraction_judged_accuracy": _metric(
                entity_judged["accuracy"],
                entity_judged["total"],
            ),
            "entity_coverage_gain_vs_skeleton_only_pp": _metric(entity_gain, entity_graph["total"]),
            "relation_recall": _metric(relation_graph["recall"], relation_graph["total"]),
            "relation_false_positive_rate": _metric(
                fp["graph_false_positive_rate"],
                fp["total"],
                recommended=not math.isclose(fp["graph_false_positive_rate"], 100.0),
            ),
            "relation_false_positive_reduction_vs_cooccurrence_pp": _metric(
                fp["absolute_reduction_pp"],
                fp["total"],
            ),
            "graph_3hop_accuracy": _metric(multi_hop["graph_3hop_accuracy"], multi_hop["total"]),
            "graph_3hop_gain_vs_1hop_pp": _metric(
                multi_hop["absolute_gain_pp"],
                multi_hop["total"],
            ),
            "fusion_answerability_gain_vs_best_single_pp": _metric(
                retrieval_answerability["fusion_gain_vs_best_single_pp"],
                len(questions),
            ),
            "multi_hop_fusion_answerability_gain_vs_vector_pp": _metric(
                retrieval_answerability["multi_hop_gain_vs_vector_pp"],
                int(
                    retrieval_answerability["type_summary"]
                    .get("fusion", {})
                    .get("multi_hop", {})
                    .get("total", 0)
                ),
            ),
        },
        "details": {
            "skeleton": {
                "total_chunks": chunk_total,
                "skeletal_chunks": skeletal_count,
                "peripheral_chunks": len(peripheral),
                "deep_extraction_cost_reduction": cost_reduction,
                "selection_trace": skeleton_trace,
                "bootstrap_stability": bootstrap,
            },
            "entities": {
                "baseline_skeleton_only": entity_baseline,
                "graph": entity_graph,
                "judged_accuracy": entity_judged,
                "coverage_gain_pp": entity_gain,
            },
            "relations": {
                "graph_recall": relation_graph,
                "false_positive": fp,
            },
            "multi_hop": multi_hop,
            "local_retrieval_answerability": retrieval_answerability,
            "benchmark_snapshot": benchmark,
        },
        "resume_copy_guardrails": [
            "Do not claim production-scale or 10k-document performance from this corpus.",
            "Do not report 100% metrics in resume copy; use them only as smoke-test evidence.",
            "Use 'evidence answerability' unless generation and judge were actually run.",
            "Bootstrap rows are repeated subsamples from the same corpus; use them for stability, not as independent documents.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--benchmark-results", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-runs", type=int, default=DEFAULT_BOOTSTRAP_RUNS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    payload = evaluate(
        args.gold,
        args.benchmark_results,
        args.questions,
        bootstrap_runs=args.bootstrap_runs,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
