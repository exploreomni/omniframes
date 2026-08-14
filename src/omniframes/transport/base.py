"""The transport seam.

Everything above this module (session, catalog, dataframe, compiler) talks to Omni exclusively
through :class:`QueryTransport`. Nothing above the transport may assume HTTP — a future
BrokerTransport (in-product notebooks) implements the same protocol.

Payloads at this seam are wire-shaped dicts (see docs/CONTRACT_NOTES.md); typed construction
happens in ``compile.querymodel``, typed interpretation in ``types``/``transport.arrow``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa

__all__ = ["PlanResult", "QueryResult", "QueryTransport"]


@dataclass(frozen=True)
class QueryResult:
    """The outcome of one executed query job.

    ``table`` is the raw decoded Arrow table — NOT yet normalized (reserved columns and totals
    rows still present). Callers run ``transport.normalize.normalize`` on it.
    """

    job_id: str
    table: pa.Table
    summary: dict[str, Any]
    cache_metadata: dict[str, Any] = field(default_factory=dict)
    query: dict[str, Any] = field(default_factory=dict)
    workbook_url: str | None = None


@dataclass(frozen=True)
class PlanResult:
    """The outcome of a ``planOnly: true`` run — schema and SQL, no data."""

    job_id: str
    summary: dict[str, Any]
    query: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class QueryTransport(Protocol):
    """Protocol every Omni transport implements.

    Implementations raise the ``omniframes.errors`` hierarchy (never httpx exceptions or raw
    HTTP details): AuthError, FeatureFlagError, ModelPermissionError, QueryError,
    QueryTimeoutError, TransportError.
    """

    def run(
        self, envelope: dict[str, Any], *, deadline_seconds: float | None = None
    ) -> QueryResult:
        """Execute a query envelope (``{"query": ..., ...}``) to a terminal result.

        Drives the full run → wait polling loop internally. ``deadline_seconds`` bounds the
        client-side wall time across all polls (None = transport default).
        """
        ...

    def plan(self, envelope: dict[str, Any]) -> PlanResult:
        """Run with ``planOnly: true`` and return schema/plan info without executing."""
        ...

    def whoami(self, model_ids: tuple[str, ...] = ()) -> dict[str, Any]:
        """Connect-time preflight (works even when the query-api flag is off)."""
        ...

    def list_models(self, **params: Any) -> dict[str, Any]:
        """One page of ``GET /models`` (cursor pagination handled by the caller)."""
        ...

    def list_topics(self, model_id: str) -> dict[str, Any]:
        """``GET /models/{id}/topic`` (list form)."""
        ...

    def get_topic(self, model_id: str, topic_name: str) -> dict[str, Any]:
        """``GET /models/{id}/topic/{name}`` — the full field-metadata payload."""
        ...

    def document_queries(self, document_identifier: str) -> dict[str, Any]:
        """``GET /documents/{identifier}/queries``."""
        ...

    def generate_query(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /ai/generate-query`` (always called with ``runQuery: false``)."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...
