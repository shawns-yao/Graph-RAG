"""Tests for rag_core.llm_resilience."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rag_core.llm_resilience import (
    LLMCallController,
    LLMCancelledError,
    LLMCircuitOpenError,
    LLMTimeBudgetExceeded,
    wrap_client_with_resilience,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_controller_retries_until_success_within_limit() -> None:
    attempts = {"count": 0}
    clock = _Clock()
    controller = LLMCallController(
        max_retries=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        jitter_seconds=0.0,
        max_consecutive_failures=10,
        total_budget_seconds=30.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("502 bad gateway")
        return "ok"

    assert controller.call("entity_extract", flaky_call) == "ok"
    assert attempts["count"] == 3


def test_controller_opens_circuit_after_consecutive_502s() -> None:
    clock = _Clock()
    controller = LLMCallController(
        max_retries=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        jitter_seconds=0.0,
        max_consecutive_failures=3,
        total_budget_seconds=30.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    attempts = {"count": 0}

    def always_502() -> None:
        attempts["count"] += 1
        raise RuntimeError("502 upstream unavailable")

    with pytest.raises(LLMCircuitOpenError):
        controller.call("entity_extract", always_502)
    assert attempts["count"] == 3


def test_controller_stops_when_total_budget_would_be_exceeded() -> None:
    clock = _Clock()
    controller = LLMCallController(
        max_retries=5,
        initial_backoff_seconds=3.0,
        max_backoff_seconds=3.0,
        jitter_seconds=0.0,
        max_consecutive_failures=10,
        total_budget_seconds=2.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    def always_502() -> None:
        raise RuntimeError("502 gateway timeout")

    with pytest.raises(LLMTimeBudgetExceeded):
        controller.call("summary", always_502)


def test_controller_aborts_when_cancel_requested() -> None:
    controller = LLMCallController(
        max_retries=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        jitter_seconds=0.0,
        max_consecutive_failures=10,
        total_budget_seconds=30.0,
        should_abort=lambda: True,
    )

    with pytest.raises(LLMCancelledError):
        controller.call("summary", lambda: "never")


def test_wrap_client_routes_chat_calls_through_controller() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = "done"
    controller = MagicMock()
    controller.call.return_value = "done"

    wrapped = wrap_client_with_resilience(client, controller)
    assert wrapped.chat.completions.create(model="m", messages=[]) == "done"
    controller.call.assert_called_once()
    assert wrapped.embeddings is client.embeddings

