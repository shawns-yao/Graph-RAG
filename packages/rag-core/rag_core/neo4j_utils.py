"""Shared helpers for Neo4j session configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_core.config import get_settings

if TYPE_CHECKING:
    from neo4j import Driver


def neo4j_session_kwargs(*, database: str | None = None) -> dict[str, Any]:
    """Return session kwargs for the selected Neo4j database."""
    selected = database
    if selected is None:
        selected = get_settings().neo4j.resolved_database
    if not selected:
        return {}
    return {"database": selected}


def open_neo4j_session(driver: Driver, *, database: str | None = None):
    """Open a driver session with database selection when configured."""
    return driver.session(**neo4j_session_kwargs(database=database))
