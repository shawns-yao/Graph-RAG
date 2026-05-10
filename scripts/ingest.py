#!/usr/bin/env python3
"""Ingest documents into the Agentic Graph RAG pipeline.

Usage:
    python scripts/ingest.py <file_or_directory> [options]

Options:
    --skip-enrichment   Skip LLM contextual enrichment (faster, no OpenAI calls for enrichment)
    --skip-skeleton     Skip skeleton indexing (no entity extraction, just vector store)
    --use-gpu           Enable GPU acceleration for Docling document parsing

Examples:
    python scripts/ingest.py data/sample_graph_rag.txt
    python scripts/ingest.py data/sample_graph_rag.txt --skip-enrichment
    python scripts/ingest.py ~/documents/ --use-gpu
"""

import argparse
import logging
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pymangle"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("ingest")


class IngestCancelled(RuntimeError):
    """Raised when ingest should stop immediately."""


class _IngestSignalGuard:
    """Track cancellation requests so long-running ingest can stop cleanly."""

    def __init__(self) -> None:
        self._cancelled = False

    def request_stop(self, *_args: object) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self, stage: str) -> None:
        if self._cancelled:
            raise IngestCancelled(f"ingest cancelled during {stage}")


def _build_ingest_openai_client(cfg, signal_guard: _IngestSignalGuard):
    from rag_core.config import make_openai_client
    from rag_core.llm_resilience import LLMCallController, wrap_client_with_resilience

    raw_client = make_openai_client(cfg, profile="ingest")
    controller = LLMCallController(
        max_retries=cfg.ingest.llm_max_retries,
        initial_backoff_seconds=cfg.ingest.llm_initial_backoff_seconds,
        max_backoff_seconds=cfg.ingest.llm_max_backoff_seconds,
        jitter_seconds=cfg.ingest.llm_jitter_seconds,
        max_consecutive_failures=cfg.ingest.llm_max_consecutive_failures,
        total_budget_seconds=cfg.ingest.llm_total_budget_seconds,
        should_abort=signal_guard.is_cancelled,
    )
    return wrap_client_with_resilience(raw_client, controller)


def ingest_file(
    file_path: str,
    *,
    skip_enrichment: bool = False,
    skip_skeleton: bool = False,
    use_gpu: bool = False,
    signal_guard: _IngestSignalGuard | None = None,
) -> None:
    """Ingest a single file through the full pipeline."""
    from neo4j import GraphDatabase
    from rag_core.chunker import chunk_document, chunk_document_for_graph
    from rag_core.config import get_settings
    from rag_core.embedder import embed_chunks
    from rag_core.enricher import enrich_chunks
    from rag_core.llm_resilience import LLMFatalError
    from rag_core.loader import load_document
    from rag_core.models import Chunk
    from rag_core.vector_store import VectorStore

    cfg = get_settings()
    guard = signal_guard or _IngestSignalGuard()
    if not cfg.openai.api_key and not cfg.openai.base_url:
        logger.error("LLM_API_KEY / LLM_BASE_URL is not set. Please configure it in .env")
        sys.exit(1)

    # 1. Load document
    guard.raise_if_cancelled("load")
    logger.info("Loading: %s (GPU=%s)", file_path, use_gpu)
    document = load_document(file_path, use_gpu=use_gpu)
    text = document.markdown
    logger.info("Loaded %d characters", len(text))

    if not text.strip():
        logger.warning("Document is empty, skipping: %s", file_path)
        return

    # 2. Chunk
    guard.raise_if_cancelled("chunk")
    chunks = chunk_document(document)
    logger.info("Created %d chunks", len(chunks))
    graph_chunks = chunk_document_for_graph(document)
    logger.info("Prepared %d graph chunks", len(graph_chunks))

    openai_client = None
    if not skip_enrichment or not skip_skeleton:
        openai_client = _build_ingest_openai_client(cfg, guard)

    # 3. Enrich (optional)
    if not skip_enrichment:
        guard.raise_if_cancelled("enrichment")
        logger.info("Enriching chunks with LLM context...")
        chunks = enrich_chunks(chunks, openai_client=openai_client)
        logger.info("Enrichment complete")
    else:
        logger.info("Skipping enrichment (--skip-enrichment)")

    # 4. Embed
    guard.raise_if_cancelled("embedding")
    logger.info("Embedding %d chunks...", len(chunks))
    chunks = embed_chunks(chunks, openai_client=openai_client)
    logger.info("Embedding complete")

    # 5. Store in vector index
    driver = GraphDatabase.driver(cfg.neo4j.uri, auth=(cfg.neo4j.user, cfg.neo4j.password))
    try:
        store = VectorStore(driver=driver)
        try:
            store.init_index()
        except Exception as exc:
            if "ServiceUnavailable" in type(exc).__name__ or "Connection refused" in str(exc):
                logger.error(
                    "Cannot connect to Neo4j at %s. "
                    "Is it running? Try: docker compose up -d",
                    cfg.neo4j.uri,
                )
                sys.exit(1)
            raise
        stored = store.add_chunks(chunks)
        logger.info("Stored %d chunks in Neo4j vector index", stored)

        # 6. Skeleton indexing (optional)
        if not skip_skeleton:
            from agentic_graph_rag.indexing.dual_node import (
                build_dual_graph,
                embed_phrase_nodes,
                init_passage_index,
                init_phrase_index,
            )
            from agentic_graph_rag.indexing.skeleton import build_skeleton_index

            graph_chunk_map = {chunk.id: chunk for chunk in graph_chunks}
            graph_embed_inputs = [
                Chunk.model_validate(chunk.model_dump())
                for chunk in graph_chunks
            ]
            if graph_embed_inputs:
                logger.info("Embedding %d graph chunks...", len(graph_embed_inputs))
                graph_embed_inputs = embed_chunks(graph_embed_inputs, openai_client=openai_client)
                for graph_chunk in graph_embed_inputs:
                    graph_chunk_map[graph_chunk.id] = graph_chunk

            graph_ready_chunks = list(graph_chunk_map.values())
            embeddings = [c.embedding for c in graph_ready_chunks if c.embedding]

            guard.raise_if_cancelled("skeleton")
            logger.info("Building skeleton index...")
            try:
                entities, relationships, skeletal, peripheral = build_skeleton_index(
                    graph_ready_chunks,
                    embeddings,
                    openai_client=openai_client,
                    driver=driver,
                )
            except LLMFatalError:
                logger.error("Skeleton indexing aborted due to repeated upstream LLM failures")
                raise
            logger.info(
                "Skeleton: %d entities, %d relationships (%d skeletal, %d peripheral)",
                len(entities), len(relationships), len(skeletal), len(peripheral),
            )

            if entities:
                guard.raise_if_cancelled("dual-graph")
                logger.info("Building dual-node graph...")
                phrase_nodes, passage_nodes, link_count = build_dual_graph(
                    entities, chunks, driver, relationships=relationships,
                )
                logger.info(
                    "Dual graph: %d phrase nodes, %d passage nodes, %d links",
                    len(phrase_nodes), len(passage_nodes), link_count,
                )

                guard.raise_if_cancelled("phrase-embedding")
                logger.info("Embedding phrase nodes...")
                updated = embed_phrase_nodes(phrase_nodes, driver, openai_client)
                logger.info("Updated %d phrase node embeddings", updated)
                init_phrase_index(driver)
                init_passage_index(driver)
        else:
            logger.info("Skipping skeleton indexing (--skip-skeleton)")

    finally:
        driver.close()

    logger.info("Done: %s", file_path)


def main() -> None:
    from rag_core.llm_resilience import LLMFatalError

    parser = argparse.ArgumentParser(
        description="Ingest documents into Agentic Graph RAG",
    )
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip LLM enrichment")
    parser.add_argument("--skip-skeleton", action="store_true", help="Skip skeleton indexing")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU for Docling")
    args = parser.parse_args()
    signal_guard = _IngestSignalGuard()
    previous_handlers: dict[int, object] = {}
    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, signal_guard.request_stop)

    try:
        target = os.path.abspath(args.path)
        if not os.path.exists(target):
            logger.error("Path does not exist: %s", target)
            sys.exit(1)

        files: list[str] = []
        if os.path.isfile(target):
            files = [target]
        elif os.path.isdir(target):
            for name in sorted(os.listdir(target)):
                full = os.path.join(target, name)
                if os.path.isfile(full) and not name.startswith("."):
                    files.append(full)

        if not files:
            logger.error("No files found at: %s", target)
            sys.exit(1)

        logger.info("Ingesting %d file(s)...", len(files))
        for f in files:
            signal_guard.raise_if_cancelled("file-loop")
            ingest_file(
                f,
                skip_enrichment=args.skip_enrichment,
                skip_skeleton=args.skip_skeleton,
                use_gpu=args.use_gpu,
                signal_guard=signal_guard,
            )

        logger.info("All done. %d file(s) ingested.", len(files))
    except IngestCancelled as exc:
        logger.warning("%s", exc)
        sys.exit(130)
    except LLMFatalError as exc:
        logger.error("Ingest aborted: %s", exc)
        sys.exit(1)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
