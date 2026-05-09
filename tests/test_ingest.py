"""Tests for scripts.ingest helpers."""

from __future__ import annotations

import pytest

from scripts.ingest import IngestCancelled, _IngestSignalGuard


def test_signal_guard_marks_cancelled() -> None:
    guard = _IngestSignalGuard()
    assert not guard.is_cancelled()
    guard.request_stop()
    assert guard.is_cancelled()


def test_signal_guard_raises_after_cancel() -> None:
    guard = _IngestSignalGuard()
    guard.request_stop()

    with pytest.raises(IngestCancelled):
        guard.raise_if_cancelled("skeleton")
