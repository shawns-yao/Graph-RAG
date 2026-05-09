#!/usr/bin/env python3
"""Inspect PhraseNode alias metadata in the configured Neo4j database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core.config import get_settings  # noqa: E402
from rag_core.neo4j_utils import open_neo4j_session  # noqa: E402


def _normalize_name(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _fetch_alias_payload(driver, *, name: str | None, limit: int) -> list[dict]:
    cypher = [
        "MATCH (p:PhraseNode)",
    ]
    params: dict[str, object] = {"limit": limit}

    if name:
        cypher.append(
            "WHERE toLower(trim(p.name)) = $name "
            "OR any(alias IN coalesce(p.aliases, []) WHERE toLower(trim(alias)) = $name)"
        )
        params["name"] = _normalize_name(name)

    cypher.extend(
        [
            "RETURN p.name AS name,",
            "       p.entity_type AS entity_type,",
            "       coalesce(p.aliases, []) AS aliases",
            "ORDER BY name",
            "LIMIT $limit",
        ]
    )

    query = "\n".join(cypher)
    with open_neo4j_session(driver) as session:
        result = session.run(query, **params)
        rows: list[dict] = []
        for record in result:
            rows.append(
                {
                    "name": record["name"],
                    "entity_type": record["entity_type"],
                    "aliases": record["aliases"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Entity name or alias to inspect")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to print")
    args = parser.parse_args()

    cfg = get_settings()
    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
    )
    try:
        payload = {
            "neo4j_uri": cfg.neo4j.uri,
            "database": cfg.neo4j.resolved_database,
            "query_name": args.name,
            "results": _fetch_alias_payload(driver, name=args.name, limit=max(1, args.limit)),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
