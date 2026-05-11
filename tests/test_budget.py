import pytest

from agentic_graph_rag.agent.budget import BudgetTracker, LLMBudgetExceeded


def test_budget_tracker_counts_actual_llm_calls():
    budget = BudgetTracker(max_llm_calls=2)

    budget.start_llm_call("generate")
    budget.start_llm_call("claim_extraction")

    assert budget.remaining_llm_calls == 0
    assert budget.snapshot() == {
        "max_llm_calls": 2,
        "calls_started": 2,
        "remaining_llm_calls": 0,
        "operations": ["generate", "claim_extraction"],
    }


def test_budget_tracker_rejects_call_after_budget_exhausted():
    budget = BudgetTracker(max_llm_calls=1)
    budget.start_llm_call("generate")

    with pytest.raises(LLMBudgetExceeded):
        budget.start_llm_call("regenerate")

    assert budget.operations == ["generate"]


def test_budget_tracker_does_not_count_deterministic_fallback():
    budget = BudgetTracker(max_llm_calls=1)

    fallback_payload = {"tool": "bm25_search"}

    assert budget.calls_started == 0
    assert fallback_payload["tool"] == "bm25_search"

