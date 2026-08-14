"""Golden lane: representative queries compiled to their exact wire JSON.

Each case builds a query (or a full run envelope) in code, serializes it with ``to_wire()``
and compares against a checked-in snapshot under ``snapshots/``. The snapshots are meant to be
diff-reviewed: a change to any of these files in a PR is a change to what omniframes puts on
the wire.

Regenerate after an intentional contract change::

    OMNIFRAMES_UPDATE_SNAPSHOTS=1 uv run pytest tests/golden -q
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from omniframes.compile.querymodel import (
    GRAND_TOTAL_KEY,
    BooleanFilter,
    CachePolicy,
    Calculation,
    CompositeFilter,
    DateFilter,
    DateFilterKind,
    Filter,
    FilterConjunction,
    NullFilter,
    NullSort,
    NumberFilter,
    NumberFilterKind,
    Query,
    QueryFilter,
    RunRequest,
    Sort,
    StringFilter,
    StringFilterKind,
    UserAttributeFilter,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_ENV_VAR = "OMNIFRAMES_UPDATE_SNAPSHOTS"

MODEL_ID = "123e4567-e89b-12d3-a456-426614174000"
REFERENCE_MODEL_ID = "0f8fad5b-d9cb-469f-a165-70867728950e"
BRANCH_ID = "9c3b1e5e-2f4a-4d1b-9a7e-6b0f2d8c4a11"
USER_ID = "5f0b9c7d-3a2e-4c6f-8b1d-7e9a0c2b4d63"

TOPIC = "order_items"


# ---------------------------------------------------------------------------------------
# Snapshot plumbing
# ---------------------------------------------------------------------------------------


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def assert_matches_snapshot(name: str, payload: dict[str, Any]) -> None:
    """Compare ``payload`` to ``snapshots/<name>.json``, writing it when updating."""
    path = SNAPSHOT_DIR / f"{name}.json"
    serialized = _serialize(payload)

    if os.environ.get(UPDATE_ENV_VAR) == "1":
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
        return

    if not path.exists():
        pytest.fail(
            f"missing snapshot {path}; regenerate with "
            f"`{UPDATE_ENV_VAR}=1 uv run pytest tests/golden -q` and review the diff"
        )

    committed = path.read_text(encoding="utf-8")
    assert json.loads(committed) == payload, (
        f"wire payload drifted from {path.name}; if the change is intended regenerate with "
        f"{UPDATE_ENV_VAR}=1"
    )
    assert committed == serialized, f"{path.name} is not in canonical formatting"


# ---------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------


def _topic_query(**overrides: Any) -> Query:
    base: dict[str, Any] = {
        "model_id": MODEL_ID,
        "join_paths_from_topic_name": TOPIC,
        "fields": ["users.state", "order_items.count"],
        "limit": 100,
    }
    base.update(overrides)
    return Query(**base)


def _filtered_query(field_name: str, flt: Filter, **overrides: Any) -> Query:
    return _topic_query(filters={field_name: flt}, **overrides)


def minimal_query() -> Query:
    """The smallest query omniframes emits: a topic, two fields, everything else default."""
    return Query(
        model_id=MODEL_ID,
        join_paths_from_topic_name=TOPIC,
        fields=["users.state", "order_items.total_sale_price"],
    )


def kitchen_sink_query() -> Query:
    """Every query key omniframes knows how to set, in one payload."""
    return Query(
        model_id=MODEL_ID,
        table="order_items",
        join_paths_from_topic_name=TOPIC,
        fields=[
            "users.state",
            "order_items.created_at[month]",
            "order_items.status",
            "order_items.total_sale_price",
            "order_items.count",
            "revenue_x100",
        ],
        filters={
            "users.state": StringFilter(
                StringFilterKind.EQUALS, ["CA", "NY"], case_insensitive=True
            ),
            "order_items.sale_price": NumberFilter(
                NumberFilterKind.BETWEEN, [10, Decimal("99.99")], is_inclusive=False
            ),
            "order_items.created_at": DateFilter(
                DateFilterKind.TIME_FOR_INTERVAL_DURATION, "30 days ago", "30 days"
            ),
            "order_items.created_at[hour_of_day]": NumberFilter(
                NumberFilterKind.GREATER_THAN, [8], is_inclusive=True
            ),
            "users.is_business": BooleanFilter(is_negative=False, treat_nulls_as_false=True),
            "products.category": NullFilter(is_negative=True, ignore_if_unjoinable=True),
        },
        sorts=[
            Sort("order_items.created_at[month]", False, NullSort.LAST),
            Sort("order_items.total_sale_price", True),
        ],
        limit=25_000,
        offset=100,
        pivots=["order_items.status"],
        calculations=[
            Calculation(
                "revenue_x100",
                {
                    "type": "call",
                    "operator": "Omni.OMNI_FX_MULTIPLY",
                    "operands": [
                        {"type": "field", "field_name": "order_items.total_sale_price"},
                        {"type": "literal", "value": 100},
                    ],
                },
            )
        ],
        fill_fields=["order_items.created_at[month]"],
        column_totals=["order_items.total_sale_price", GRAND_TOTAL_KEY],
        row_totals=["order_items.total_sale_price"],
    )


def filter_string_query() -> Query:
    return _filtered_query(
        "users.state", StringFilter(StringFilterKind.CONTAINS, ["cal"], case_insensitive=True)
    )


def filter_number_query() -> Query:
    return _filtered_query(
        "order_items.sale_price",
        NumberFilter(NumberFilterKind.BETWEEN, [10, 99.5], is_inclusive=False),
    )


def filter_date_query() -> Query:
    return _filtered_query(
        "order_items.created_at",
        DateFilter(DateFilterKind.BETWEEN, "2026-01-01", "2026-07-01"),
    )


def filter_boolean_query() -> Query:
    return _filtered_query(
        "users.is_business", BooleanFilter(is_negative=True, treat_nulls_as_false=True)
    )


def filter_null_query() -> Query:
    return _filtered_query("users.state", NullFilter(is_negative=True))


def filter_composite_query() -> Query:
    """Composites nest inside ONE field's entry, recursively, with no depth cap."""
    return _filtered_query(
        "order_items.status",
        CompositeFilter(
            FilterConjunction.OR,
            [
                StringFilter(StringFilterKind.EQUALS, ["complete", "shipped"]),
                CompositeFilter(
                    FilterConjunction.AND,
                    [
                        StringFilter(StringFilterKind.STARTS_WITH, ["proc"]),
                        NullFilter(is_negative=True),
                    ],
                ),
            ],
            is_negative=False,
        ),
    )


def filter_query_query() -> Query:
    """A ``type: "query"`` filter plus the reference it points at."""
    return _filtered_query(
        "users.id",
        QueryFilter("users.id", "top_users", disregard_limit=True),
        static_query_references={
            "top_users": Query(
                model_id=REFERENCE_MODEL_ID,
                table="users",
                fields=["users.id"],
                sorts=[Sort("users.lifetime_value", True)],
                limit=10,
            )
        },
    )


def filter_user_attribute_query() -> Query:
    return _filtered_query("users.country", UserAttributeFilter("allowed_country"))


def sql_job_envelope() -> RunRequest:
    """A raw-SQL job: ``userEditedSQL`` + ``rewriteSql: false`` + server-side sorts (§3.4)."""
    return RunRequest(
        Query.for_sql(
            MODEL_ID,
            "SELECT status, COUNT(*) AS order_count FROM order_items GROUP BY 1",
            sorts=[Sort("order_count", True, NullSort.LAST)],
            column_totals=["order_count"],
            limit=1_000,
        ),
        cache=CachePolicy.STANDARD,
    )


def static_query_references_envelope() -> RunRequest:
    """A full envelope whose query carries ``staticQueryReferences`` (§3.5)."""
    return RunRequest(
        filter_query_query(),
        branch_id=BRANCH_ID,
        cache=CachePolicy.SKIP_CACHE,
        plan_only=False,
        timezone="America/Los_Angeles",
        user_id=USER_ID,
    )


QUERY_CASES: dict[str, Callable[[], Query]] = {
    "minimal_query": minimal_query,
    "kitchen_sink_query": kitchen_sink_query,
    "filter_string": filter_string_query,
    "filter_number": filter_number_query,
    "filter_date": filter_date_query,
    "filter_boolean": filter_boolean_query,
    "filter_null": filter_null_query,
    "filter_composite": filter_composite_query,
    "filter_query": filter_query_query,
    "filter_user_attribute": filter_user_attribute_query,
}

ENVELOPE_CASES: dict[str, Callable[[], RunRequest]] = {
    "envelope_sql_job": sql_job_envelope,
    "envelope_static_query_references": static_query_references_envelope,
}


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(QUERY_CASES))
def test_query_json_snapshot(name: str) -> None:
    query = QUERY_CASES[name]()
    query.validate()

    assert_matches_snapshot(name, query.to_wire())


@pytest.mark.parametrize("name", sorted(ENVELOPE_CASES))
def test_envelope_json_snapshot(name: str) -> None:
    request = ENVELOPE_CASES[name]()
    request.validate()

    assert_matches_snapshot(name, request.to_wire())


def test_every_filter_arm_has_a_snapshot() -> None:
    """The eight arms of CONTRACT_NOTES §3.1 each get their own reviewed payload."""
    covered = {
        json.loads((SNAPSHOT_DIR / f"{name}.json").read_text(encoding="utf-8"))[
            "filters"
        ].popitem()[1]["type"]
        for name in QUERY_CASES
        if name.startswith("filter_")
    }

    assert covered == {
        "string",
        "number",
        "date",
        "boolean",
        "null",
        "composite",
        "query",
        "user_attribute",
    }


def test_no_orphan_snapshots() -> None:
    expected = {f"{name}.json" for name in (*QUERY_CASES, *ENVELOPE_CASES)}
    # `semantic_*.json` belongs to test_semantic_snapshots.py (compiled DataFrames rather than
    # hand-built queries); that module has its own orphan check over the same directory.
    actual = {
        path.name for path in SNAPSHOT_DIR.glob("*.json") if not path.name.startswith("semantic_")
    }

    assert actual == expected, "snapshots directory is out of sync with the case list"
