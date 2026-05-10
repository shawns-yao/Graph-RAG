"""Resilience helpers for OpenAI-compatible LLM calls during ingest."""

from __future__ import annotations

import random
import re
import time
from typing import Any, Callable


class LLMFatalError(RuntimeError):
    """Base class for ingest-time fatal LLM failures."""


class LLMCircuitOpenError(LLMFatalError):
    """Raised when transient upstream failures keep happening."""


class LLMTimeBudgetExceeded(LLMFatalError):
    """Raised when the per-file ingest LLM budget is exhausted."""


class LLMCancelledError(LLMFatalError):
    """Raised when ingest cancellation has been requested."""


def is_retryable_llm_error(exc: Exception) -> bool:
    """Return True for transient upstream failures worth retrying."""
    text = str(exc).lower()
    retry_markers = (
        "504",
        "503",
        "502",
        "500",
        "429",
        "gateway time-out",
        "gateway timeout",
        "rate limit",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "connection error",
        "eof occurred",
        "retryable",
    )
    return any(marker in text for marker in retry_markers)


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort parse of retry hints from OpenAI-compatible errors."""
    match = re.search(
        r"retry_after['\"]?\s*[:=]\s*(\d+(?:\.\d+)?)",
        str(exc),
        re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


class LLMCallController:
    """Bound retries, enforce a time budget, and stop on repeated 502s."""

    def __init__(
        self,
        *,
        max_retries: int,
        initial_backoff_seconds: float,
        max_backoff_seconds: float,
        jitter_seconds: float,
        max_consecutive_failures: int,
        total_budget_seconds: float,
        should_abort: Callable[[], bool] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._max_retries = max(0, int(max_retries))
        self._initial_backoff_seconds = max(0.0, float(initial_backoff_seconds))
        self._max_backoff_seconds = max(
            self._initial_backoff_seconds,
            float(max_backoff_seconds),
        )
        self._jitter_seconds = max(0.0, float(jitter_seconds))
        self._max_consecutive_failures = max(1, int(max_consecutive_failures))
        self._total_budget_seconds = max(0.0, float(total_budget_seconds))
        self._should_abort = should_abort or (lambda: False)
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._started_at = self._monotonic()
        self._consecutive_failures = 0

    def call(self, operation_name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a remote call under retry, budget, and cancellation policy."""
        attempt = 0
        while True:
            self._raise_if_cancelled(operation_name)
            self._raise_if_budget_exceeded(operation_name)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                if not is_retryable_llm_error(exc):
                    raise

                self._consecutive_failures += 1
                if self._consecutive_failures >= self._max_consecutive_failures:
                    raise LLMCircuitOpenError(
                        f"{operation_name} aborted after "
                        f"{self._consecutive_failures} consecutive upstream failures: {exc}"
                    ) from exc
                if attempt >= self._max_retries:
                    raise

                delay = self._compute_backoff_seconds(exc, attempt)
                if self._would_exceed_budget(delay):
                    raise LLMTimeBudgetExceeded(
                        f"{operation_name} exceeded ingest LLM budget "
                        f"after {attempt + 1} failed attempts"
                    ) from exc
                attempt += 1
                self._sleep_with_abort(delay, operation_name)
                continue

            self._consecutive_failures = 0
            return result

    def _compute_backoff_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after_seconds = _extract_retry_after_seconds(exc)
        if retry_after_seconds is not None:
            delay = min(retry_after_seconds, self._max_backoff_seconds)
        else:
            delay = min(
                self._initial_backoff_seconds * (2**attempt),
                self._max_backoff_seconds,
            )
        if self._jitter_seconds > 0:
            delay += random.uniform(0.0, self._jitter_seconds)
        return delay

    def _elapsed_seconds(self) -> float:
        return max(0.0, self._monotonic() - self._started_at)

    def _would_exceed_budget(self, delay_seconds: float) -> bool:
        if self._total_budget_seconds <= 0:
            return True
        return self._elapsed_seconds() + max(0.0, delay_seconds) > self._total_budget_seconds

    def _raise_if_budget_exceeded(self, operation_name: str) -> None:
        if self._total_budget_seconds <= 0:
            raise LLMTimeBudgetExceeded(
                f"{operation_name} aborted because ingest LLM budget is non-positive"
            )
        if self._elapsed_seconds() > self._total_budget_seconds:
            raise LLMTimeBudgetExceeded(
                f"{operation_name} exceeded ingest LLM budget "
                f"({self._total_budget_seconds:.0f}s)"
            )

    def _raise_if_cancelled(self, operation_name: str) -> None:
        if self._should_abort():
            raise LLMCancelledError(f"{operation_name} cancelled")

    def _sleep_with_abort(self, delay_seconds: float, operation_name: str) -> None:
        if delay_seconds <= 0:
            self._raise_if_cancelled(operation_name)
            return
        remaining = delay_seconds
        while remaining > 0:
            self._raise_if_cancelled(operation_name)
            self._raise_if_budget_exceeded(operation_name)
            step = min(remaining, 0.25)
            self._sleep(step)
            remaining -= step


class _CreateProxy:
    """Wrap an OpenAI-compatible `.create(...)` endpoint."""

    def __init__(self, target: Any, controller: LLMCallController, operation_name: str) -> None:
        self._target = target
        self._controller = controller
        self._operation_name = operation_name

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._controller.call(
            self._operation_name,
            self._target.create,
            *args,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _CompletionsProxy:
    def __init__(self, target: Any, controller: LLMCallController) -> None:
        self._target = target
        self.create = _CreateProxy(target, controller, "chat.completions.create").create

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ChatProxy:
    def __init__(self, target: Any, controller: LLMCallController) -> None:
        self._target = target
        self.completions = _CompletionsProxy(target.completions, controller)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ResilientClientProxy:
    """Attach resilience policy to chat calls while leaving embeddings intact."""

    def __init__(self, target: Any, controller: LLMCallController) -> None:
        self._target = target
        self.chat = _ChatProxy(target.chat, controller)
        self.embeddings = getattr(target, "embeddings", None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def wrap_client_with_resilience(raw_client: Any, controller: LLMCallController) -> Any:
    """Wrap an OpenAI-compatible client for guarded chat calls."""
    return _ResilientClientProxy(raw_client, controller)
