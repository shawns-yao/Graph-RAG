"""Trace storage backends: in-memory (default) and optional Redis.

Usage:
    store = create_trace_store()          # auto-selects based on REDIS_URL
    store.put(trace)
    trace = store.get("tr_abc123")
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from collections import OrderedDict

from rag_core.models import PipelineTrace, WorkflowMemoryEntry

logger = logging.getLogger(__name__)

_DEFAULT_MAX = 100
_REDIS_TTL = 3600  # 1 hour
_SESSION_TRACE_MAX = 20


class TraceStore(ABC):
    """Abstract trace storage interface."""

    @abstractmethod
    def put(self, trace: PipelineTrace) -> None: ...

    @abstractmethod
    def get(self, trace_id: str) -> PipelineTrace | None: ...

    @abstractmethod
    def get_session_traces(self, session_id: str, limit: int = _SESSION_TRACE_MAX) -> list[PipelineTrace]: ...

    @abstractmethod
    def get_session_memory(
        self,
        session_id: str,
        limit: int = _SESSION_TRACE_MAX,
    ) -> list[WorkflowMemoryEntry]: ...


class InMemoryTraceStore(TraceStore):
    """Bounded in-memory LRU trace cache (default)."""

    def __init__(self, max_size: int = _DEFAULT_MAX):
        self._cache: OrderedDict[str, PipelineTrace] = OrderedDict()
        self._max = max_size
        self._session_traces: dict[str, list[PipelineTrace]] = {}

    def put(self, trace: PipelineTrace) -> None:
        self._cache[trace.trace_id] = trace
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)
        if trace.session_id:
            session_traces = self._session_traces.setdefault(trace.session_id, [])
            session_traces.append(trace)
            if len(session_traces) > _SESSION_TRACE_MAX:
                del session_traces[:-_SESSION_TRACE_MAX]

    def get(self, trace_id: str) -> PipelineTrace | None:
        return self._cache.get(trace_id)

    def get_session_traces(self, session_id: str, limit: int = _SESSION_TRACE_MAX) -> list[PipelineTrace]:
        if not session_id:
            return []
        return list(self._session_traces.get(session_id, [])[-limit:])

    def get_session_memory(
        self,
        session_id: str,
        limit: int = _SESSION_TRACE_MAX,
    ) -> list[WorkflowMemoryEntry]:
        memory: list[WorkflowMemoryEntry] = []
        for trace in self.get_session_traces(session_id, limit=limit):
            memory.extend(trace.workflow_memory)
        return memory


class RedisTraceStore(TraceStore):
    """Redis-backed trace storage with TTL."""

    def __init__(self, url: str, ttl: int = _REDIS_TTL):
        import redis

        self._client = redis.from_url(url)
        self._ttl = ttl
        self._prefix = "agr:trace:"
        self._session_prefix = "agr:session:"

    def put(self, trace: PipelineTrace) -> None:
        key = f"{self._prefix}{trace.trace_id}"
        self._client.setex(key, self._ttl, trace.model_dump_json())
        if trace.session_id:
            session_key = f"{self._session_prefix}{trace.session_id}:traces"
            pipe = self._client.pipeline()
            pipe.rpush(session_key, trace.trace_id)
            pipe.ltrim(session_key, -_SESSION_TRACE_MAX, -1)
            pipe.expire(session_key, self._ttl)
            pipe.execute()

    def get(self, trace_id: str) -> PipelineTrace | None:
        key = f"{self._prefix}{trace_id}"
        data = self._client.get(key)
        if data is None:
            return None
        return PipelineTrace.model_validate(json.loads(data))

    def get_session_traces(self, session_id: str, limit: int = _SESSION_TRACE_MAX) -> list[PipelineTrace]:
        if not session_id:
            return []
        session_key = f"{self._session_prefix}{session_id}:traces"
        trace_ids = self._client.lrange(session_key, -limit, -1)
        traces: list[PipelineTrace] = []
        for trace_id in trace_ids:
            decoded = trace_id.decode() if isinstance(trace_id, bytes) else str(trace_id)
            trace = self.get(decoded)
            if trace is not None:
                traces.append(trace)
        return traces

    def get_session_memory(
        self,
        session_id: str,
        limit: int = _SESSION_TRACE_MAX,
    ) -> list[WorkflowMemoryEntry]:
        memory: list[WorkflowMemoryEntry] = []
        for trace in self.get_session_traces(session_id, limit=limit):
            memory.extend(trace.workflow_memory)
        return memory


def create_trace_store() -> TraceStore:
    """Factory: use Redis if REDIS_URL is set, otherwise in-memory."""
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            store = RedisTraceStore(redis_url)
            logger.info("Using Redis trace store at %s", redis_url)
            return store
        except Exception as e:
            logger.warning("Redis unavailable (%s), falling back to in-memory", e)
    return InMemoryTraceStore()
