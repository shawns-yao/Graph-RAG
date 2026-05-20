"""Live resume metrics evaluation using Neo4j plus real LLM calls.

This is the canonical resume-metric evaluator. It is slower and calls external services:
- Neo4j reads for chunks, graph, and retrieval
- Neo4j structural checks for relation false positives and multi-hop paths
- LLM entity extraction
- LLM answer generation through PipelineService / retrieval agent
- LLM answer judge against the gold answers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentic_graph_rag.agent.tools import bm25_search, cypher_traverse, hybrid_search, vector_search
from agentic_graph_rag.indexing.skeleton import (
    _rank_chunks_for_skeleton_selection,
    build_knn_graph,
    compute_pagerank,
    resolve_skeleton_beta,
    select_skeletal_chunks,
)
from agentic_graph_rag.service import PipelineService
from rag_core.generator import generate_answer
from rag_core.config import get_settings, make_openai_client
from rag_core.models import Chunk

DEFAULT_GOLD = ROOT / "test" / "medical_benchmark" / "eval_gold" / "medical_resume_gold.json"
DEFAULT_QUESTIONS = ROOT / "test" / "medical_benchmark" / "questions_master.json"
DEFAULT_OUTPUT = ROOT / "test" / "medical_benchmark" / "results" / "resume_live_metrics.json"


def _norm(text: object) -> str:
    return re.sub(r"\s+", "", str(text or "").casefold())


def _pct(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100.0, 2)


def _load_json(path: Path) -> Any:
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
    return [
        Chunk(
            id=str(row["id"]),
            content=str(row["content"] or ""),
            context=str(row["context"] or ""),
            embedding=list(row["embedding"] or []),
            metadata=_metadata_dict(row["metadata"]),
        )
        for row in rows
    ]


def _load_graph_counts(session: Any) -> dict[str, int]:
    labels = {
        "rag_chunks": "RagChunk",
        "phrase_nodes": "PhraseNode",
        "passage_nodes": "PassageNode",
    }
    counts: dict[str, int] = {}
    for key, label in labels.items():
        row = session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()
        counts[key] = int(row["count"] if row else 0)
    for key, rel in {"mentioned_in": "MENTIONED_IN", "related_to": "RELATED_TO"}.items():
        row = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS count").single()
        counts[key] = int(row["count"] if row else 0)
    return counts


def _load_graph_structure(session: Any) -> dict[str, list[dict[str, str]]]:
    phrases = [
        {
            "name": str(row["name"] or ""),
            "entity_type": str(row["entity_type"] or ""),
            "aliases": " ".join(str(alias) for alias in row["aliases"] or []),
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


def _entity_coverage(gold_entities: list[dict[str, Any]], graph_phrases: list[dict[str, str]]) -> dict[str, Any]:
    phrase_texts = [" ".join([phrase["name"], phrase.get("aliases", "")]) for phrase in graph_phrases]
    hits = []
    misses = []
    for entity in gold_entities:
        found = any(_contains_any_surface(text, _entity_surfaces(entity)) for text in phrase_texts)
        (hits if found else misses).append(entity["name"])
    return {
        "hits": len(hits),
        "total": len(gold_entities),
        "coverage": _pct(len(hits), len(gold_entities)),
        "misses": misses,
    }


def _baseline_entity_coverage(gold_entities: list[dict[str, Any]], skeletal_chunks: list[Chunk]) -> dict[str, Any]:
    text = "\n".join(chunk.enriched_content for chunk in skeletal_chunks)
    hits = [entity["name"] for entity in gold_entities if _contains_any_surface(text, _entity_surfaces(entity))]
    return {
        "hits": len(hits),
        "total": len(gold_entities),
        "coverage": _pct(len(hits), len(gold_entities)),
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


def _relation_recall(gold_relations: list[dict[str, Any]], graph_rels: list[dict[str, str]]) -> dict[str, Any]:
    hits = []
    misses = []
    for item in gold_relations:
        label = f"{item['source']} --{item.get('relation', '')}--> {item['target']}"
        matched = any(_rel_matches(rel, item["source"], item["target"], item.get("relation")) for rel in graph_rels)
        (hits if matched else misses).append(label)
    return {"hits": len(hits), "total": len(gold_relations), "recall": _pct(len(hits), len(gold_relations)), "misses": misses}


def _passage_cooccurs(passages: list[dict[str, str]], source: str, target: str) -> bool:
    return any(_norm(source) in _norm(passage["text"]) and _norm(target) in _norm(passage["text"]) for passage in passages)


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
        "relative_reduction_percent": round((baseline_rate - graph_rate) / baseline_rate * 100.0, 2)
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


def _run_graph_structure_metrics(
    session: Any,
    gold: dict[str, Any],
    skeletal_chunks: list[Chunk],
) -> dict[str, Any]:
    graph = _load_graph_structure(session)
    graph_entity = _entity_coverage(gold.get("entities", []), graph["phrases"])
    skeleton_entity = _baseline_entity_coverage(gold.get("entities", []), skeletal_chunks)
    relation_recall = _relation_recall(gold.get("positive_relations", []), graph["relationships"])
    relation_fp = _false_positive_rates(gold.get("negative_relations", []), graph["relationships"], graph["passages"])
    multi_hop = _multi_hop_metrics(session, gold.get("multi_hop_tasks", []))
    return {
        "entities": {
            "graph": graph_entity,
            "skeleton_only_baseline": skeleton_entity,
            "coverage_gain_pp": round(graph_entity["coverage"] - skeleton_entity["coverage"], 2),
        },
        "relation_recall": relation_recall,
        "relation_false_positive": relation_fp,
        "multi_hop": multi_hop,
    }


def _select_skeleton(chunks: list[Chunk]) -> tuple[list[Chunk], list[Chunk], dict[str, Any]]:
    embeddings = [chunk.embedding for chunk in chunks]
    if not chunks or any(not emb for emb in embeddings):
        return [], chunks, {"manual_selection": False, "reason": "missing embeddings"}
    graph = build_knn_graph(chunks, embeddings)
    pagerank = compute_pagerank(graph)
    beta = resolve_skeleton_beta(chunks)
    ranked_scores = _rank_chunks_for_skeleton_selection(chunks, pagerank)
    skeletal, peripheral = select_skeletal_chunks(chunks, pagerank, beta=beta)
    selected_ids = {chunk.id for chunk in skeletal}
    ranked = sorted(ranked_scores.items(), key=lambda item: item[1], reverse=True)
    trace = {
        "manual_selection": False,
        "method": "KNN + PageRank + entity density + medical section prior + hard-fact signal + diversity",
        "beta": round(beta, 4),
        "selected_count": len(skeletal),
        "ranked_chunks": [
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
    return skeletal, peripheral, trace


def _chunk_payload(chunks: list[Chunk]) -> list[dict[str, str]]:
    return [
        {
            "id": chunk.id,
            "text": chunk.enriched_content[:1800],
        }
        for chunk in chunks
    ]


def _extract_json(text: str) -> Any:
    cleaned = (text or "").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = max(parts, key=len).replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(1))


def _llm_extract_entities(client: Any, model: str, chunks: list[Chunk], label: str) -> dict[str, Any]:
    prompt = {
        "task": "Extract canonical medical entities from the provided chunks.",
        "rules": [
            "Return only medically meaningful canonical entities.",
            "Do not return metadata leakage, broken fragments, or low-information words.",
            "Include aliases when explicitly present.",
            "Return JSON only.",
        ],
        "output_schema": {
            "entities": [
                {
                    "name": "canonical name",
                    "aliases": ["alias"],
                    "type": "Disease|Drug|Test|Symptom|Procedure|Threshold|Other",
                    "source_chunk_ids": ["chunk id"],
                }
            ]
        },
        "chunks": _chunk_payload(chunks),
    }
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict medical information extraction evaluator."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    content = response.choices[0].message.content or "{}"
    parsed = _extract_json(content)
    entities = parsed.get("entities", []) if isinstance(parsed, dict) else []
    usage = getattr(response, "usage", None)
    return {
        "label": label,
        "chunks": len(chunks),
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "entities": entities,
    }


def _entity_surfaces(entity: dict[str, Any]) -> list[str]:
    return [str(entity.get("name", "")), *(str(alias) for alias in entity.get("aliases", []))]


def _contains_any_surface(text: str, surfaces: list[str]) -> bool:
    normalized = _norm(text)
    return any(_norm(surface) and _norm(surface) in normalized for surface in surfaces)


def _score_entities(
    extracted: list[dict[str, Any]],
    positive_entities: list[dict[str, Any]],
    negative_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    extracted_text = "\n".join(
        " ".join([str(item.get("name", "")), *(str(alias) for alias in item.get("aliases", []))])
        for item in extracted
    )
    true_positive = [
        item["name"] for item in positive_entities if _contains_any_surface(extracted_text, _entity_surfaces(item))
    ]
    false_negative = [item["name"] for item in positive_entities if item["name"] not in true_positive]
    false_positive = [
        item["name"] for item in negative_entities if _contains_any_surface(extracted_text, _entity_surfaces(item))
    ]
    true_negative = [item["name"] for item in negative_entities if item["name"] not in false_positive]
    precision = _pct(len(true_positive), len(true_positive) + len(false_positive))
    recall = _pct(len(true_positive), len(positive_entities))
    accuracy = _pct(len(true_positive) + len(true_negative), len(positive_entities) + len(negative_entities))
    return {
        "true_positive": len(true_positive),
        "false_negative": len(false_negative),
        "true_negative": len(true_negative),
        "false_positive": len(false_positive),
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "misses": false_negative,
        "false_positive_items": false_positive,
    }


def _run_live_extraction(client: Any, model: str, chunks: list[Chunk], gold: dict[str, Any]) -> dict[str, Any]:
    skeletal, peripheral, selection_trace = _select_skeleton(chunks)
    full = _llm_extract_entities(client, model, chunks, "full_document_llm")
    skeleton = _llm_extract_entities(client, model, skeletal, "skeleton_llm")
    full_score = _score_entities(full["entities"], gold["entities"], gold.get("negative_entities", []))
    skeleton_score = _score_entities(skeleton["entities"], gold["entities"], gold.get("negative_entities", []))
    prompt_cost_reduction = _pct(full["prompt_tokens"] - skeleton["prompt_tokens"], full["prompt_tokens"])
    chunk_cost_reduction = _pct(len(chunks) - len(skeletal), len(chunks))
    retained = _pct(skeleton_score["true_positive"], max(1, full_score["true_positive"]))
    return {
        "selection_trace": selection_trace,
        "full_document_llm": {**full, "score": full_score},
        "skeleton_llm": {**skeleton, "score": skeleton_score},
        "peripheral_chunks": len(peripheral),
        "chunk_cost_reduction": chunk_cost_reduction,
        "prompt_token_cost_reduction": prompt_cost_reduction,
        "coverage_retained_vs_full": min(100.0, retained),
        "raw_coverage_retained_vs_full": retained,
        "entity_accuracy_gain_pp": round(skeleton_score["accuracy"] - full_score["accuracy"], 2),
    }


def _keyword_score(answer: str, keywords: list[str]) -> dict[str, Any]:
    if not keywords:
        return {"hits": 0, "total": 0, "ratio": 0.0, "pass": False}
    answer_norm = _norm(answer)
    hits = sum(1 for keyword in keywords if _norm(keyword) in answer_norm)
    ratio = hits / len(keywords)
    return {"hits": hits, "total": len(keywords), "ratio": round(ratio, 4), "pass": ratio >= 0.5}


def _llm_judge(client: Any, model: str, question: dict[str, Any], actual: str) -> dict[str, Any]:
    prompt = {
        "role": "medical QA evaluator",
        "question": question["query"],
        "gold_answer": question.get("answer", ""),
        "system_answer": actual,
        "criteria": {
            "answer_score": "1-5, 5 means complete and correct",
            "hallucination": "true if answer contains unsupported or clinically wrong facts",
            "confidence_score": "0-100 based on correctness, completeness, and evidence grounding",
        },
        "return_json": {
            "answer_score": 1,
            "hallucination": False,
            "confidence_score": 0,
            "reason": "short reason",
        },
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = _extract_json(response.choices[0].message.content or "{}")
        return {
            "judge_ok": True,
            "answer_score": int(data.get("answer_score", 0)),
            "hallucination": bool(data.get("hallucination", False)),
            "confidence_score": float(data.get("confidence_score", 0)),
            "reason": str(data.get("reason", ""))[:300],
        }
    except Exception as exc:
        return {
            "judge_ok": False,
            "answer_score": 0,
            "hallucination": False,
            "confidence_score": 0.0,
            "reason": f"judge_error: {type(exc).__name__}: {exc}"[:300],
        }


def _trace_claim_counts(qa: Any) -> dict[str, int]:
    if not qa.trace or not qa.trace.verification_step:
        return {"claims_total": 0, "claims_supported": 0, "claims_possible": 0, "claims_incorrect": 0}
    step = qa.trace.verification_step
    return {
        "claims_total": int(step.claims_total),
        "claims_supported": int(step.claims_supported),
        "claims_possible": int(step.claims_possible),
        "claims_incorrect": int(step.claims_incorrect),
    }


def _run_one_qa(
    question: dict[str, Any],
    mode: str,
    driver: Any,
    client: Any,
    judge_model: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if mode in {"vector_search", "bm25_search", "cypher_traverse", "hybrid_search"}:
            tool_map = {
                "vector_search": vector_search,
                "bm25_search": bm25_search,
                "cypher_traverse": cypher_traverse,
                "hybrid_search": hybrid_search,
            }
            results = tool_map[mode](question["query"], driver, client)
            qa = generate_answer(question["query"], results, client, reflection_verdict="answer")
        else:
            service = PipelineService(driver, client)
            qa = service.query(question["query"], mode=mode, session_id=f"live-eval::{mode}::{question['id']}")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        keyword = _keyword_score(qa.answer or "", list(question.get("keywords", [])))
        judge = _llm_judge(client, judge_model, question, qa.answer or "")
        claim_counts = _trace_claim_counts(qa)
        return {
            "id": question["id"],
            "query_type": question.get("query_type", ""),
            "difficulty": question.get("difficulty", ""),
            "mode": mode,
            "status": "ok",
            "elapsed_ms": elapsed_ms,
            "answer": qa.answer,
            "answer_status": qa.answer_status,
            "retrieval_status": qa.retrieval_status,
            "verification_status": qa.verification_status,
            "retries": qa.retries,
            "source_count": len(qa.sources),
            "keyword": keyword,
            "judge": judge,
            "claim_counts": claim_counts,
        }
    except Exception as exc:
        return {
            "id": question["id"],
            "query_type": question.get("query_type", ""),
            "mode": mode,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }


def _summarize_qa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    by_mode_type: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        mode = row["mode"]
        qtype = row.get("query_type", "unknown")
        bucket = by_mode.setdefault(
            mode,
            {
                "total": 0,
                "ok": 0,
                "keyword_pass": 0,
                "judge_pass": 0,
                "hallucinations": 0,
                "judge_errors": 0,
                "confidence_sum": 0.0,
                "retries": 0,
                "claims_total": 0,
                "claims_supported": 0,
                "claims_incorrect": 0,
            },
        )
        type_bucket = by_mode_type.setdefault(mode, {}).setdefault(
            qtype,
            {"total": 0, "judge_pass": 0, "keyword_pass": 0, "hallucinations": 0, "judge_errors": 0},
        )
        bucket["total"] += 1
        type_bucket["total"] += 1
        if row.get("status") != "ok":
            continue
        bucket["ok"] += 1
        keyword_pass = bool(row.get("keyword", {}).get("pass"))
        judge_ok = bool(row.get("judge", {}).get("judge_ok", False))
        judge_pass = judge_ok and int(row.get("judge", {}).get("answer_score", 0)) >= 4
        hallucination = bool(row.get("judge", {}).get("hallucination", True))
        confidence = float(row.get("judge", {}).get("confidence_score", 0.0))
        bucket["keyword_pass"] += int(keyword_pass)
        bucket["judge_pass"] += int(judge_pass)
        bucket["judge_errors"] += int(not judge_ok)
        if judge_ok:
            bucket["hallucinations"] += int(hallucination)
            bucket["confidence_sum"] += confidence
        bucket["retries"] += int(row.get("retries") or 0)
        claims = row.get("claim_counts", {})
        bucket["claims_total"] += int(claims.get("claims_total", 0))
        bucket["claims_supported"] += int(claims.get("claims_supported", 0))
        bucket["claims_incorrect"] += int(claims.get("claims_incorrect", 0))
        type_bucket["judge_pass"] += int(judge_pass)
        type_bucket["keyword_pass"] += int(keyword_pass)
        type_bucket["judge_errors"] += int(not judge_ok)
        if judge_ok:
            type_bucket["hallucinations"] += int(hallucination)

    for bucket in by_mode.values():
        total = bucket["total"]
        ok = bucket["ok"]
        judged = max(0, ok - bucket["judge_errors"])
        bucket["keyword_accuracy"] = _pct(bucket["keyword_pass"], total)
        bucket["judge_accuracy"] = _pct(bucket["judge_pass"], total)
        bucket["hallucination_rate"] = _pct(bucket["hallucinations"], judged)
        bucket["avg_confidence"] = round(bucket["confidence_sum"] / judged, 2) if judged else 0.0
        bucket["avg_retries"] = round(bucket["retries"] / ok, 2) if ok else 0.0
        bucket["claim_support_rate"] = _pct(bucket["claims_supported"], bucket["claims_total"])
        bucket["claim_incorrect_rate"] = _pct(bucket["claims_incorrect"], bucket["claims_total"])

    for mode_buckets in by_mode_type.values():
        for bucket in mode_buckets.values():
            total = bucket["total"]
            bucket["judge_accuracy"] = _pct(bucket["judge_pass"], total)
            bucket["keyword_accuracy"] = _pct(bucket["keyword_pass"], total)
            judged = max(0, total - bucket.get("judge_errors", 0))
            bucket["hallucination_rate"] = _pct(bucket["hallucinations"], judged)
    return {"by_mode": by_mode, "by_mode_type": by_mode_type}


def _run_live_qa(
    driver: Any,
    client: Any,
    questions: list[dict[str, Any]],
    *,
    judge_model: str,
    modes: list[str],
    limit: int,
    output: Path,
    existing_payload: dict[str, Any],
) -> dict[str, Any]:
    selected_questions = questions[:limit] if limit > 0 else questions
    existing_rows = {
        (row.get("mode"), row.get("id")): row
        for row in existing_payload.get("qa", {}).get("rows", [])
        if row.get("mode") and row.get("id")
    }
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for question in selected_questions:
            key = (mode, question["id"])
            if key in existing_rows and existing_rows[key].get("status") == "ok":
                rows.append(existing_rows[key])
                continue
            row = _run_one_qa(question, mode, driver, client, judge_model)
            rows.append(row)
            partial = dict(existing_payload)
            partial["qa"] = {"modes": modes, "question_count": len(selected_questions), "rows": rows}
            partial["qa"]["summary"] = _summarize_qa(rows)
            output.write_text(json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{mode} {question['id']} {row.get('status')} {row.get('elapsed_ms')}ms")
    return {"modes": modes, "question_count": len(selected_questions), "rows": rows, "summary": _summarize_qa(rows)}


def _safe_gain(after: float, before: float) -> float:
    return round(after - before, 2)


def _build_resume_fill(payload: dict[str, Any]) -> dict[str, Any]:
    live = payload.get("live_extraction", {})
    graph_structure = payload.get("graph_structure", {})
    qa_summary = payload.get("qa", {}).get("summary", {})
    by_mode = qa_summary.get("by_mode", {})
    by_type = qa_summary.get("by_mode_type", {})

    vector = by_mode.get("vector_search") or by_mode.get("vector") or {}
    fusion = by_mode.get("agent_pattern") or by_mode.get("hybrid_search") or {}
    multi_vector = (by_type.get("vector_search") or by_type.get("vector") or {}).get("multi_hop", {})
    multi_fusion = (by_type.get("agent_pattern") or by_type.get("hybrid_search") or {}).get("multi_hop", {})

    extraction_cost = live.get("prompt_token_cost_reduction")
    graph_coverage = graph_structure.get("entities", {}).get("graph", {}).get("coverage")
    entity_accuracy = live.get("skeleton_llm", {}).get("score", {}).get("accuracy")
    entity_gain = live.get("entity_accuracy_gain_pp")
    entity_coverage_gain = graph_structure.get("entities", {}).get("coverage_gain_pp")

    relation_fp = graph_structure.get("relation_false_positive", {}).get("graph_false_positive_rate")
    relation_fp_reduction = graph_structure.get("relation_false_positive", {}).get("absolute_reduction_pp")
    graph_3hop = graph_structure.get("multi_hop", {}).get("graph_3hop_accuracy")
    graph_3hop_gain = graph_structure.get("multi_hop", {}).get("absolute_gain_pp")

    fusion_acc = fusion.get("judge_accuracy")
    vector_acc = vector.get("judge_accuracy")
    fusion_gain = _safe_gain(fusion_acc, vector_acc) if fusion_acc is not None and vector_acc is not None else None
    multi_fusion_acc = multi_fusion.get("judge_accuracy")
    multi_vector_acc = multi_vector.get("judge_accuracy")
    multi_gain = (
        _safe_gain(multi_fusion_acc, multi_vector_acc)
        if multi_fusion_acc is not None and multi_vector_acc is not None
        else None
    )
    hallucination_before = vector.get("hallucination_rate")
    hallucination_after = fusion.get("hallucination_rate")
    confidence_before = vector.get("avg_confidence")
    confidence_after = fusion.get("avg_confidence")
    confidence_gain = (
        _safe_gain(confidence_after, confidence_before)
        if confidence_after is not None and confidence_before is not None
        else None
    )

    return {
        "skeleton_extraction": {
            "graph_coverage_percent": graph_coverage,
            "cost_reduction_percent": extraction_cost,
            "entity_accuracy_percent": entity_accuracy,
            "entity_accuracy_gain_pp": entity_gain,
            "entity_coverage_gain_pp": entity_coverage_gain,
        },
        "dual_layer_graph": {
            "relation_false_positive_rate_percent": relation_fp,
            "relation_false_positive_reduction_pp": relation_fp_reduction,
            "graph_3hop_accuracy_percent": graph_3hop,
            "graph_3hop_gain_pp": graph_3hop_gain,
        },
        "fusion_retrieval": {
            "vector_overall_accuracy_percent": vector_acc,
            "fusion_overall_accuracy_percent": fusion_acc,
            "fusion_gain_pp": fusion_gain,
            "vector_multi_hop_accuracy_percent": multi_vector_acc,
            "fusion_multi_hop_accuracy_percent": multi_fusion_acc,
            "multi_hop_gain_pp": multi_gain,
        },
        "self_correction": {
            "hallucination_before_percent": hallucination_before,
            "hallucination_after_percent": hallucination_after,
            "confidence_before": confidence_before,
            "confidence_after": confidence_after,
            "confidence_gain": confidence_gain,
            "supplemental_recall_rate_percent": _pct(fusion.get("retries", 0), fusion.get("ok", 0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sections", default="extraction,qa")
    parser.add_argument(
        "--modes",
        default="vector_search,cypher_traverse,agent_pattern",
        help="Comma-separated: vector_search,bm25_search,cypher_traverse,hybrid_search,agent_pattern,agent_llm",
    )
    parser.add_argument("--limit", type=int, default=0, help="Question limit, 0 means all questions.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = get_settings()
    client = make_openai_client(cfg)
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    gold = _load_json(args.gold)
    questions_payload = _load_json(args.questions)
    questions = questions_payload.get("questions", questions_payload)
    sections = {section.strip() for section in args.sections.split(",") if section.strip()}
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    payload: dict[str, Any] = {}
    if args.output.exists():
        payload = _load_json(args.output)

    started = time.perf_counter()
    with driver.session(database=cfg.neo4j.database) as session:
        chunks = _load_chunks(session)
        skeletal, _, _ = _select_skeleton(chunks)
        payload["graph_counts"] = _load_graph_counts(session)
        payload["graph_structure"] = _run_graph_structure_metrics(session, gold, skeletal)

    if "extraction" in sections and not (args.skip_existing and payload.get("live_extraction")):
        payload["live_extraction"] = _run_live_extraction(client, cfg.openai.llm_model_mini, chunks, gold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if "qa" in sections:
        payload["qa"] = _run_live_qa(
            driver,
            client,
            questions,
            judge_model=cfg.openai.llm_model_mini,
            modes=modes,
            limit=args.limit,
            output=args.output,
            existing_payload=payload,
        )

    driver.close()
    payload["resume_fill"] = _build_resume_fill(payload)
    payload["metadata"] = {
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "llm_model": cfg.openai.llm_model,
        "judge_model": cfg.openai.llm_model_mini,
        "questions": len(questions if args.limit == 0 else questions[: args.limit]),
        "sections": sorted(sections),
        "modes": modes,
        "notes": [
            "This run calls Neo4j and real LLM endpoints.",
            "Extraction cost uses prompt tokens when provider usage is available.",
            "Answer accuracy, hallucination, and confidence are LLM-judge metrics against gold answers.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "resume_fill": payload["resume_fill"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
