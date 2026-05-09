#!/usr/bin/env python3
"""Prepare, ingest, and benchmark the GraphRAG-Benchmark medical subset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.runner import run_benchmark  # noqa: E402
from rag_core.config import get_settings, make_openai_client  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402
from scripts.prepare_graphrag_benchmark_corpus import prepare_corpus  # noqa: E402


BENCH_ROOT = ROOT / "GraphRAG-Benchmark"
MEDICAL_CORPUS_JSON = BENCH_ROOT / "Datasets" / "Corpus" / "medical.json"
MEDICAL_QUESTIONS_JSON = BENCH_ROOT / "Datasets" / "Questions" / "medical_questions.json"
DEFAULT_WORKDIR = ROOT / ".tmp" / "graphrag_benchmark_medical"


def _load_questions(path: Path, limit: int | None = None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("medical question file must contain a JSON array")
    rows = [item for item in payload if isinstance(item, dict) and str(item.get("question") or "").strip()]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _run_ingest(
    prepared_dir: Path,
    use_gpu: bool,
    skip_enrichment: bool,
    skip_skeleton: bool,
) -> None:
    python_exe = shutil.which("py")
    if not python_exe:
        raise RuntimeError("Windows py launcher not found")
    cmd = [
        python_exe,
        str(ROOT / "scripts" / "ingest.py"),
        str(prepared_dir),
    ]
    if skip_enrichment:
        cmd.append("--skip-enrichment")
    if skip_skeleton:
        cmd.append("--skip-skeleton")
    if use_gpu:
        cmd.append("--use-gpu")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG-Benchmark medical set against this project")
    parser.add_argument("--limit-docs", type=int, default=None, help="Optional max medical corpus docs to ingest")
    parser.add_argument("--limit-questions", type=int, default=None, help="Optional max medical questions to evaluate")
    parser.add_argument(
        "--prepared-dir",
        default=str(DEFAULT_WORKDIR / "prepared_corpus"),
        help="Directory for extracted text files before ingest",
    )
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=120000,
        help="Split the single medical corpus into paragraph-preserving files of this size before ingest",
    )
    parser.add_argument(
        "--results-path",
        default=str(DEFAULT_WORKDIR / "results_medical.json"),
        help="Path for benchmark JSON output",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["graph_only", "graph_h2", "vector_only", "bm25_only", "hybrid", "hybrid_rerank"],
        help="Benchmark modes to execute",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel question workers")
    parser.add_argument("--eval-concurrency", type=int, default=2, help="Evaluation concurrency")
    parser.add_argument("--use-gpu", action="store_true", help="Pass --use-gpu to ingest")
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip per-chunk LLM enrichment during ingest to keep large benchmark corpora tractable",
    )
    parser.add_argument(
        "--skip-skeleton",
        action="store_true",
        help="Skip graph skeleton extraction during ingest to run vector/BM25 benchmark baselines only",
    )
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    files = prepare_corpus(
        str(MEDICAL_CORPUS_JSON),
        str(prepared_dir),
        limit=args.limit_docs,
        max_chars_per_file=args.max_chars_per_file,
        overwrite=True,
    )
    print(f"prepared_files={len(files)}")

    _run_ingest(
        prepared_dir,
        use_gpu=args.use_gpu,
        skip_enrichment=args.skip_enrichment,
        skip_skeleton=args.skip_skeleton,
    )

    questions = _load_questions(MEDICAL_QUESTIONS_JSON, limit=args.limit_questions)
    cfg = get_settings()
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    client = make_openai_client(cfg)
    try:
        payload = run_benchmark(
            driver,
            client,
            questions=questions,
            modes=args.modes,
            max_workers=args.max_workers,
            eval_concurrency=args.eval_concurrency,
        )
    finally:
        driver.close()

    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results_path={results_path}")


if __name__ == "__main__":
    main()
