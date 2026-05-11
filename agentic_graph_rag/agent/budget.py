"""Workflow-level budget tracking for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field


class LLMBudgetExceeded(RuntimeError):
    """Raised before starting an LLM call when workflow budget is exhausted."""


@dataclass
class BudgetTracker:
    """Track actual LLM calls started by one workflow run."""

    max_llm_calls: int = 4
    calls_started: int = 0
    operations: list[str] = field(default_factory=list)

    @property
    def remaining_llm_calls(self) -> int:
        return max(0, self.max_llm_calls - self.calls_started)

    def can_start_llm_call(self) -> bool:
        return self.remaining_llm_calls > 0

    def start_llm_call(self, operation: str) -> None:
        if not self.can_start_llm_call():
            raise LLMBudgetExceeded(
                f"LLM budget exhausted before {operation}: "
                f"{self.calls_started}/{self.max_llm_calls} calls already started"
            )
        self.calls_started += 1
        self.operations.append(operation)

    def snapshot(self) -> dict[str, int | list[str]]:
        return {
            "max_llm_calls": self.max_llm_calls,
            "calls_started": self.calls_started,
            "remaining_llm_calls": self.remaining_llm_calls,
            "operations": list(self.operations),
        }

