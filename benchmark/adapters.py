"""Async adapters for OpenAI-compatible clients used by benchmark metrics."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

from rag_core.config import get_settings


@dataclass
class AsyncInvokeResponse:
    """Minimal async invoke response with LangChain-like `.content`."""

    content: str


class OpenAIAsyncJudge:
    """Expose `ainvoke` over the project's sync OpenAI-compatible client."""

    def __init__(self, client: Any, *, model: str | None = None) -> None:
        cfg = get_settings()
        benchmark = cfg.benchmark
        self._client = client
        self._model = model or cfg.openai.llm_model_mini
        self._temperature = cfg.openai.llm_temperature
        self._max_retries = max(0, int(benchmark.llm_max_retries))
        self._initial_backoff_seconds = max(0.0, float(benchmark.llm_initial_backoff_seconds))
        self._max_backoff_seconds = max(self._initial_backoff_seconds, float(benchmark.llm_max_backoff_seconds))
        self._jitter_seconds = max(0.0, float(benchmark.llm_jitter_seconds))

    async def ainvoke(self, prompt: str, config: dict[str, Any] | None = None) -> AsyncInvokeResponse:
        del config
        attempt = 0
        while True:
            try:
                response = await asyncio.to_thread(
                    self._client.chat.completions.create,
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                )
                return AsyncInvokeResponse(content=response.choices[0].message.content or "")
            except Exception:
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._backoff_seconds(attempt))
                attempt += 1

    def _backoff_seconds(self, attempt: int) -> float:
        base = min(self._max_backoff_seconds, self._initial_backoff_seconds * (2**attempt))
        if self._jitter_seconds <= 0:
            return base
        return base + random.uniform(0.0, self._jitter_seconds)


class OpenAIAsyncEmbeddings:
    """Expose async embedding calls over the project's sync client."""

    def __init__(self, client: Any, *, model: str | None = None) -> None:
        cfg = get_settings()
        benchmark = cfg.benchmark
        self._client = client
        self._model = model or cfg.openai.embedding_model
        self._dimensions = cfg.openai.embedding_dimensions
        self._max_retries = max(0, int(benchmark.llm_max_retries))
        self._initial_backoff_seconds = max(0.0, float(benchmark.llm_initial_backoff_seconds))
        self._max_backoff_seconds = max(self._initial_backoff_seconds, float(benchmark.llm_max_backoff_seconds))
        self._jitter_seconds = max(0.0, float(benchmark.llm_jitter_seconds))

    async def aembed_query(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": text,
        }
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        attempt = 0
        while True:
            try:
                response = await asyncio.to_thread(self._client.embeddings.create, **kwargs)
                return list(response.data[0].embedding)
            except Exception:
                if attempt >= self._max_retries:
                    raise
                await asyncio.sleep(self._backoff_seconds(attempt))
                attempt += 1

    def _backoff_seconds(self, attempt: int) -> float:
        base = min(self._max_backoff_seconds, self._initial_backoff_seconds * (2**attempt))
        if self._jitter_seconds <= 0:
            return base
        return base + random.uniform(0.0, self._jitter_seconds)
