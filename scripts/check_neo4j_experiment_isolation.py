#!/usr/bin/env python3
"""Inspect Neo4j experiment isolation support for the current environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core.config import get_settings  # noqa: E402
from rag_core.neo4j_utils import neo4j_session_kwargs  # noqa: E402


def _fetch_databases(driver) -> list[dict]:
    with driver.session(database="system") as session:
        result = session.run(
            "SHOW DATABASES YIELD name, type, access, address, currentStatus "
            "RETURN name, type, access, address, currentStatus "
            "ORDER BY name"
        )
        return [dict(record) for record in result]


def _check_target_database(driver, database: str | None) -> dict:
    if not database:
        return {
            "configured_database": None,
            "reachable": True,
            "reason": "No explicit database configured; driver will use server default.",
        }

    try:
        with driver.session(**neo4j_session_kwargs(database=database)) as session:
            value = session.run("RETURN 1 AS ok").single()
    except Exception as exc:  # pragma: no cover - integration path
        return {
            "configured_database": database,
            "reachable": False,
            "reason": str(exc),
        }
    return {
        "configured_database": database,
        "reachable": bool(value and value["ok"] == 1),
        "reason": "",
    }


def main() -> None:
    cfg = get_settings()
    driver = GraphDatabase.driver(
        cfg.neo4j.uri,
        auth=(cfg.neo4j.user, cfg.neo4j.password),
    )
    try:
        payload = {
            "neo4j_uri": cfg.neo4j.uri,
            "configured_database": cfg.neo4j.resolved_database,
            "databases": _fetch_databases(driver),
            "target_database_check": _check_target_database(driver, cfg.neo4j.resolved_database),
        }
        payload["supports_multi_database_admin"] = any(
            item.get("name") == "system" for item in payload["databases"]
        ) and len(payload["databases"]) > 2
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
