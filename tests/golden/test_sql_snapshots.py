"""Golden lane: the tier-2 statement omniframes writes, and the envelope it rides in.

Two checked-in artifacts per case, because they answer different questions and drift apart in
different ways:

* ``snapshots/sql_<case>.sql`` — the rendered statement, ``pretty=True``.  A diff here is a
  change to the SQL a warehouse will execute, which is exactly the kind of change that should
  need a human to look at it.
* ``snapshots/semantic_sqltier_<case>.json`` — the whole run envelope, references included.  This is
  where ``rewriteSql: false`` (without which the server silently ignores the SQL),
  ``sqlSortsEnabled: false``, the explicit ``limit`` and each reference's snake_case ``model_id``
  are pinned.

Between them the cases cover every row of docs/SQLTIER.md §1.  Nothing here talks to anything —
compiling is pure.

Regenerate after an intentional change::

    OMNIFRAMES_UPDATE_SNAPSHOTS=1 uv run pytest tests/golden -q
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from tests.golden.test_semantic_snapshots import (
    SNAPSHOT_DIR,
    UPDATE_ENV_VAR,
    _session,
    assert_matches_snapshot,
    topic,
)

from omniframes import functions as F
from omniframes.compile.querymodel import DEFAULT_FETCH_LIMIT
from omniframes.compile.semantic import RemoteStep
from omniframes.dataframe import DataFrame

PREFIX = "sql_"
ENVELOPE_PREFIX = "semantic_sqltier_"

REVENUE = "order_items.total_sale_price"
PRICE = "order_items.sale_price"
QUANTITY = "order_items.quantity"


# ---------------------------------------------------------------------------------------
# Cases — one per row of docs/SQLTIER.md §1
# ---------------------------------------------------------------------------------------


def ad_hoc_group_by() -> DataFrame:
    """The shape tier 2 exists for: a GROUP BY that runs in the warehouse."""
    return topic().group_by("users.state").agg(F.count_distinct("users.id").alias("buyers"))


def pushed_filter() -> DataFrame:
    """A tier-1-expressible filter is pushed **into** the reference, not written as SQL."""
    return (
        topic()
        .filter(F.col("order_items.status") == "complete")
        .group_by("users.state")
        .agg(F.sum(PRICE).alias("revenue"), F.count("order_items.id").alias("n"))
    )


def cross_field_or_where() -> DataFrame:
    """A cross-field OR has no wire filter, so it becomes the outer WHERE over ``ref_1``."""
    return (
        topic()
        .select("order_items.id", "users.state", "users.age")
        .filter((F.col("users.state") == "California") | (F.col("users.age") > 60))
    )


def having_on_an_ad_hoc_aggregate() -> DataFrame:
    """ANSI HAVING cannot see SELECT aliases, so the full aggregate is substituted back in."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.count_distinct("users.id").alias("buyers"))
        .filter(F.col("buyers") > 25)
    )


def computed_column() -> DataFrame:
    """Arithmetic over fields: a SELECT expression aliased to its final user-facing name."""
    return (
        topic()
        .select("users.state", PRICE, QUANTITY)
        .with_column("unit_price", F.col(PRICE) / F.col(QUANTITY))
        .limit(25)
    )


def sort_limit_offset() -> DataFrame:
    """ORDER BY / LIMIT / OFFSET live in the SQL text — ``sqlSortsEnabled`` gates nothing."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.sum(PRICE).alias("revenue"))
        .sort(F.col("revenue").desc())
        .limit(10)
        .offset(5)
    )


def grained_group_key() -> DataFrame:
    """The reference computes the grain; the SQL quotes the bracketed name it produced."""
    month = F.col("order_items.created_at").grain("month").alias("month")
    return topic().group_by(month).agg(F.count_distinct("users.id").alias("buyers"))


def string_predicates() -> DataFrame:
    """LIKE with an escaped pattern, and LOWER() on both sides instead of the unportable ILIKE."""
    return (
        topic()
        .select("order_items.id", "users.state", "users.name")
        .filter(
            F.col("users.state").contains("cal", case_insensitive=True)
            | F.col("users.name").starts_with("A%")
        )
    )


def typed_literals() -> DataFrame:
    """Dates, numbers and booleans as typed literal nodes — never as formatted text."""
    created = F.col("order_items.created_at")
    return (
        topic()
        .select("order_items.id", "order_items.created_at", "order_items.returned", "users.age")
        .filter(
            created.between(date(2025, 7, 1), date(2026, 7, 1))
            | (F.col("order_items.returned") == True)  # noqa: E712 - a Column, not a bool
            | F.col("users.age").between(18, 21)
        )
    )


CASES: dict[str, Callable[[], DataFrame]] = {
    "ad_hoc_group_by": ad_hoc_group_by,
    "pushed_filter": pushed_filter,
    "cross_field_or_where": cross_field_or_where,
    "having_on_an_ad_hoc_aggregate": having_on_an_ad_hoc_aggregate,
    "computed_column": computed_column,
    "sort_limit_offset": sort_limit_offset,
    "grained_group_key": grained_group_key,
    "string_predicates": string_predicates,
    "typed_literals": typed_literals,
}


def mixed_aggregation() -> DataFrame:
    """The tier-2 half of a mixed aggregate: the governed measure stays a tier-1 step."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers"))
    )


# ---------------------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------------------


def sql_step(frame: DataFrame) -> RemoteStep:
    """The one tier-2 step of a frame — the DAG cases have exactly one."""
    steps = [step for step in frame._compiled().steps if step.compilation.is_sql]
    assert len(steps) == 1, f"expected exactly one tier-2 step, got {len(steps)}"
    return steps[0]


def assert_matches_sql_snapshot(name: str, sql: str) -> None:
    path = SNAPSHOT_DIR / f"{name}.sql"
    serialized = sql if sql.endswith("\n") else sql + "\n"

    if os.environ.get(UPDATE_ENV_VAR) == "1":
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
        return

    if not path.exists():
        pytest.fail(
            f"missing snapshot {path}; regenerate with "
            f"`{UPDATE_ENV_VAR}=1 uv run pytest tests/golden -q` and review the diff"
        )
    assert path.read_text(encoding="utf-8") == serialized, (
        f"the generated SQL drifted from {path.name}; if the change is intended regenerate with "
        f"{UPDATE_ENV_VAR}=1"
    )


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_generated_sql_snapshot(name: str) -> None:
    assert_matches_sql_snapshot(f"{PREFIX}{name}", sql_step(CASES[name]()).query.user_edited_sql)


@pytest.mark.parametrize("name", sorted(CASES))
def test_tier_two_envelope_snapshot(name: str) -> None:
    assert_matches_snapshot(f"{ENVELOPE_PREFIX}{name}", CASES[name]()._compiled().envelope)


def test_the_mixed_aggregations_sql_half_is_pinned_too() -> None:
    """A DAG has no single envelope, but its tier-2 step still has SQL worth diff-reviewing."""
    assert_matches_sql_snapshot(
        f"{PREFIX}mixed_aggregation", sql_step(mixed_aggregation()).query.user_edited_sql
    )
    assert_matches_snapshot(
        f"{ENVELOPE_PREFIX}mixed_aggregation", sql_step(mixed_aggregation()).envelope
    )


def test_no_orphan_sql_snapshots() -> None:
    expected = {f"{PREFIX}{name}.sql" for name in (*CASES, "mixed_aggregation")}
    actual = {path.name for path in SNAPSHOT_DIR.glob(f"{PREFIX}*.sql")}

    assert actual == expected, "the generated-SQL snapshots are out of sync with the cases"


def test_no_orphan_tier_two_envelope_snapshots() -> None:
    expected = {f"{ENVELOPE_PREFIX}{name}.json" for name in (*CASES, "mixed_aggregation")}
    actual = {path.name for path in SNAPSHOT_DIR.glob(f"{ENVELOPE_PREFIX}*.json")}

    assert actual == expected, "the tier-2 envelope snapshots are out of sync with the cases"


# ---------------------------------------------------------------------------------------
# The invariants every tier-2 envelope has to carry (docs/SQLTIER.md §2)
# ---------------------------------------------------------------------------------------


def every_query() -> list[dict[str, Any]]:
    return [sql_step(build()).envelope["query"] for build in (*CASES.values(), mixed_aggregation)]


def test_every_tier_two_query_says_do_not_rewrite_my_sql() -> None:
    """``userEditedSQL`` without ``rewriteSql: false`` is silently ignored (CONTRACT_NOTES §3.4)."""
    for query in every_query():
        assert query["userEditedSQL"].strip()
        assert query["rewriteSql"] is False


def test_every_tier_two_query_disables_sql_sorts() -> None:
    """ORDER BY lives in the SQL text, so the envelope's ``sorts`` gate nothing (§2 DECISION)."""
    for query in every_query():
        assert query["sqlSortsEnabled"] is False
        assert query["sorts"] == []
        assert query["column_totals"] == {}


def test_every_tier_two_query_sends_an_explicit_limit_and_version_9() -> None:
    for query in every_query():
        assert query["limit"] is not None
        assert f"LIMIT {query['limit']}" in query["userEditedSQL"]
        assert query["version"] == 9


def test_every_reference_is_unlimited_and_carries_its_snake_case_model_id() -> None:
    """§3.5's extra ``model_id`` — the fake 400s without it, and so does the server."""
    for query in every_query():
        references = query["staticQueryReferences"]
        assert list(references) == ["ref_1"], "deterministic, first-use order, bare identifier"
        for reference in references.values():
            assert reference["model_id"] == reference["modelId"]
            assert reference["limit"] is None, "a paged reference is a wrong answer, not a page"
            assert reference["fields"], "a reference always projects the fields the SQL names"
            assert reference["userEditedSQL"] == "", "the reference is a governed query"


def test_a_tier_two_step_renames_nothing_after_the_fact() -> None:
    """The SQL aliases each column to its final name, so ``normalize`` has nothing to rename."""
    frame = ad_hoc_group_by()
    step = sql_step(frame)

    assert step.alias_map == {}
    assert step.columns == ("users.state", "buyers")
    assert frame.columns == ("users.state", "buyers")


def test_the_dialect_knob_changes_the_generated_sql() -> None:
    """The escape hatch of §3: same plan, same envelope shape, dialect-specific text."""
    session = _session(sql_dialect="bigquery")
    frame = DataFrame(session, ad_hoc_group_by().logical_plan)
    generated = sql_step(frame).query.user_edited_sql

    assert "`users.state`" in generated, "bigquery quotes identifiers with backticks"
    assert '"users.state"' not in generated
    assert '"users.state"' in sql_step(ad_hoc_group_by()).query.user_edited_sql


def test_the_default_limit_applies_to_the_final_select_exactly_as_in_tier_one() -> None:
    query = sql_step(ad_hoc_group_by()).envelope["query"]

    assert query["limit"] == DEFAULT_FETCH_LIMIT
    assert f"LIMIT {DEFAULT_FETCH_LIMIT}" in query["userEditedSQL"]


def test_an_offset_rides_the_sql_text_and_the_envelope_together() -> None:
    query = sql_step(sort_limit_offset()).envelope["query"]

    assert query["limit"] == 10
    assert query["offset"] == 5
    assert "OFFSET 5" in query["userEditedSQL"]


def test_the_snapshot_directory_paths_are_the_ones_this_module_writes() -> None:
    """Guards the shared snapshot directory against a rename that silently orphans files."""
    assert (SNAPSHOT_DIR / f"{PREFIX}ad_hoc_group_by.sql").exists()
    assert (SNAPSHOT_DIR / f"{ENVELOPE_PREFIX}ad_hoc_group_by.json").exists()
    assert Path(SNAPSHOT_DIR).name == "snapshots"
