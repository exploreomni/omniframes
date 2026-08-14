"""Saved-query and AI endpoints of the FakeOmniAPI — CONTRACT_NOTES §4.

Two catalog-adjacent endpoints hand a client a **query object it did not write**:

* ``GET /api/v1/documents/{identifier}/queries`` — the queries stored on a document's dashboard.
  The payload is ``{queries: [{id, name, query, url}]}`` and the ``query`` blob is a stored query
  whose ``modelId`` the client must verify (or inject) before running it.  A document with no
  dashboard answers **404**, which is a normal, expected outcome rather than a bug.
* ``POST /api/v1/ai/generate-query`` — natural language in, ``{query, topic, baseView, error}``
  out.  Omniframes always sends ``runQuery: false`` and runs the returned query through its own
  pipeline; the fake **requires** that, because an AI endpoint that executed queries behind the
  client's back would be the one thing the offline lane could never verify.

Everything served here is a real bench-model query: each canned blob is executable through
``/query/run`` unchanged, so a test can follow the whole "discover a query, then run it" path
without hand-writing the query object in the middle.

Nothing here performs I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from tests.fakes.bench_model import BENCH_BASE_VIEW, BENCH_MODEL_ID, BENCH_TOPIC_NAME

__all__ = [
    "BENCH_DOCUMENT_ID",
    "DEFAULT_DOCUMENTS",
    "DOCUMENT_WITHOUT_DASHBOARD",
    "GENERATED_QUERIES",
    "NO_QUERY_GENERATED",
    "RUN_QUERY_REFUSAL",
    "GeneratedQuery",
    "SavedQuery",
    "bench_query",
    "generate_for_prompt",
]

#: The document whose dashboard carries the canned saved queries.
BENCH_DOCUMENT_ID: Final = "bench_dashboard"

#: A document that exists but has **no dashboard** — the documented 404 of §4.  Configured as an
#: identifier mapped to an empty query list.
DOCUMENT_WITHOUT_DASHBOARD: Final = "bench_workbook"

#: 400 detail when ``runQuery`` is anything other than ``false``.
RUN_QUERY_REFUSAL: Final = "runQuery must be false: FakeOmniAPI does not execute generated queries"

#: 400 detail when the prompt matches none of the canned queries (§4: "400 when no query could
#: be generated").
NO_QUERY_GENERATED: Final = "No query could be generated for that prompt"


def bench_query(
    *,
    fields: Sequence[str],
    sorts: Sequence[Mapping[str, Any]] = (),
    limit: int | None = 1000,
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A full bench-model query object, with every key omniframes always sends explicitly (§3).

    These blobs go straight back onto the wire from two endpoints, so they carry the server's
    own defaults spelled out rather than relying on ``createQuery`` to fill them in.
    """
    return {
        "modelId": BENCH_MODEL_ID,
        "table": BENCH_BASE_VIEW,
        "join_paths_from_topic_name": BENCH_TOPIC_NAME,
        "fields": list(fields),
        "filters": dict(filters or {}),
        "sorts": [dict(sort) for sort in sorts],
        "limit": limit,
        "offset": 0,
        "pivots": [],
        "calculations": [],
        "fill_fields": [],
        "column_totals": {},
        "row_totals": {},
        "userEditedSQL": "",
        "default_group_by": True,
        "version": 9,
    }


def _sort(column_name: str, *, descending: bool = False) -> dict[str, Any]:
    return {
        "column_name": column_name,
        "sort_descending": descending,
        "null_sort": "OMNI_DEFAULT",
    }


# --------------------------------------------------------------------------------------
# GET /api/v1/documents/{identifier}/queries
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SavedQuery:
    """One entry of ``{queries: [...]}`` — a stored query plus how to open it in Omni (§4)."""

    id: str
    name: str
    query: Mapping[str, Any]
    url: str

    def to_wire(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "query": dict(self.query), "url": self.url}


def _saved(identifier: str, name: str, query: Mapping[str, Any]) -> SavedQuery:
    return SavedQuery(
        id=identifier,
        name=name,
        query=query,
        url=f"https://bench.example.omni.co/dashboards/{BENCH_DOCUMENT_ID}/{identifier}",
    )


BENCH_SAVED_QUERIES: Final = (
    _saved(
        "q_revenue_by_state",
        "Revenue by state",
        bench_query(
            fields=["users.state", "order_items.total_sale_price"],
            sorts=[_sort("order_items.total_sale_price", descending=True)],
        ),
    ),
    _saved(
        "q_monthly_revenue",
        "Monthly revenue",
        bench_query(
            fields=[
                "order_items.created_at[month]",
                "order_items.total_sale_price",
                "order_items.count",
            ],
            sorts=[_sort("order_items.created_at[month]")],
        ),
    ),
    _saved(
        "q_average_sale_price_by_category",
        "Average sale price by category",
        bench_query(
            fields=["products.category", "order_items.average_sale_price"],
            sorts=[_sort("products.category")],
        ),
    ),
    # A stored query that carries FILTERS.  Without one, the `(stored — sent verbatim, not
    # recompiled)` line docs/HYBRID.md §7 pins by example is unreachable from the suite: every
    # other blob here is filter-free, so `explain()` never renders it.
    _saved(
        "q_completed_west_coast_revenue",
        "Completed west-coast revenue",
        bench_query(
            fields=["users.state", "order_items.total_sale_price"],
            sorts=[_sort("order_items.total_sale_price", descending=True)],
            filters={
                "order_items.status": {
                    "type": "string",
                    "kind": "EQUALS",
                    "values": ["complete"],
                },
                "users.state": {
                    "type": "string",
                    "kind": "EQUALS",
                    "values": ["California", "Oregon", "Washington"],
                },
            },
        ),
    ),
)

#: The default document map.  An identifier mapped to an **empty** list is a document that
#: exists but has no dashboard — the §4 404 — while an identifier that is absent entirely is an
#: unknown document.  Both answer 404, with different messages.
DEFAULT_DOCUMENTS: Final[Mapping[str, tuple[SavedQuery, ...]]] = {
    BENCH_DOCUMENT_ID: BENCH_SAVED_QUERIES,
    DOCUMENT_WITHOUT_DASHBOARD: (),
}


# --------------------------------------------------------------------------------------
# POST /api/v1/ai/generate-query
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedQuery:
    """One canned prompt → query mapping, and the ``{query, topic, baseView, error}`` it serves."""

    #: Lowercase substring that triggers this query; the first match in :data:`GENERATED_QUERIES`
    #: wins, so the list is ordered most specific first.
    prompt: str
    query: Mapping[str, Any] = field(default_factory=dict)
    topic: str | None = BENCH_TOPIC_NAME
    base_view: str | None = BENCH_BASE_VIEW

    def to_wire(self) -> dict[str, Any]:
        return {
            "query": dict(self.query),
            "topic": self.topic,
            "baseView": self.base_view,
            "error": None,
        }


#: The deterministic prompt map.  Ordered: "monthly revenue by state" hits the monthly entry,
#: because the first substring match wins.
GENERATED_QUERIES: Final = (
    GeneratedQuery(
        prompt="monthly revenue",
        query=bench_query(
            fields=["order_items.created_at[month]", "order_items.total_sale_price"],
            sorts=[_sort("order_items.created_at[month]")],
        ),
    ),
    GeneratedQuery(
        prompt="revenue by state",
        query=bench_query(
            fields=["users.state", "order_items.total_sale_price"],
            sorts=[_sort("order_items.total_sale_price", descending=True)],
        ),
    ),
    GeneratedQuery(
        prompt="top products",
        query=bench_query(
            fields=["products.name", "order_items.total_sale_price"],
            sorts=[_sort("order_items.total_sale_price", descending=True)],
            limit=10,
        ),
    ),
)


def generate_for_prompt(prompt: str) -> GeneratedQuery | None:
    """The canned query for ``prompt``, or ``None`` when nothing matches (a 400 — §4)."""
    text = prompt.casefold()
    for candidate in GENERATED_QUERIES:
        if candidate.prompt in text:
            return candidate
    return None
