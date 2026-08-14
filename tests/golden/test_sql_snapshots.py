"""Golden lane: the tier-2 OmniSQL statement omniframes writes, and the envelope it rides in.

Two checked-in artifacts per case, because they answer different questions and drift apart in
different ways:

* ``snapshots/sql_<case>.sql`` — the rendered statement, ``pretty=True``.  A diff here is a
  change to what the server will parse and plan, which is exactly the kind of change that should
  need a human to look at it.
* ``snapshots/semantic_sqltier_<case>.json`` — the whole run envelope.  This is where the
  **absence** of ``rewriteSql`` (the only thing that selects the parsed-OmniSQL path —
  CONTRACT_NOTES §3.6), the absence of ``staticQueryReferences``/``sqlSortsEnabled``, and the
  mirrored ``limit`` are pinned.

Two families of case, because tier 2's reach and the splitter's choice are different contracts:

* :data:`CASES` compile through the **splitter**, so each one also pins that tier 1 declined and
  tier 2 took the node.
* :data:`DIRECT_CASES` compile through :func:`~omniframes.compile.sqlgen.compile_sql` itself.
  Those are shapes tier 2 *can* write but does not always *get* — a measure-only select is tier
  1's by right (docs/SQLTIER.md §4) — so pinning them through the splitter would pin the tier
  choice instead of the emission.

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
    assert_matches_snapshot,
    topic,
)

from omniframes import functions as F
from omniframes.compile.querymodel import DEFAULT_FETCH_LIMIT, WireDict
from omniframes.compile.semantic import RemoteStep, SemanticCompilation, build_envelope
from omniframes.compile.sqlgen import compile_sql
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
    """v1 pushed a tier-1-expressible filter into the reference core; now it is a WHERE (§3.3)."""
    return (
        topic()
        .filter(F.col("order_items.status") == "complete")
        .group_by("users.state")
        .agg(F.sum(PRICE).alias("revenue"), F.count("order_items.id").alias("n"))
    )


def cross_field_or_where() -> DataFrame:
    """A cross-field OR has no wire filter, which is why the statement is written at all."""
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
    """Arithmetic over fields: an expression select item under its generated ``of_expr_n`` alias."""
    return (
        topic()
        .select("users.state", PRICE, QUANTITY)
        .with_column("unit_price", F.col(PRICE) / F.col(QUANTITY))
        .limit(25)
    )


def sort_limit_offset() -> DataFrame:
    """ORDER BY / LIMIT / OFFSET live in the SQL text — the query object's limit is ignored."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.sum(PRICE).alias("revenue"))
        .sort(F.col("revenue").desc())
        .limit(10)
        .offset(5)
    )


def grained_group_key() -> DataFrame:
    """A grain ref is a bare ref: ``${view.field[grain]}``, unaliased, grouped positionally."""
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
    """A governed measure beside an ad-hoc aggregate — ONE statement, no decomposition (§4)."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers"))
    )


def measure_grand_total() -> DataFrame:
    """A measure select with no GROUP BY is the grand total (CONTRACT_NOTES §3.6).

    Tier 1 wins this shape through the splitter, and rightly so; the statement is pinned here
    because tier 2 has to be able to write it for the mixed fallback path.
    """
    return topic().select(F.measure(REVENUE).alias("revenue"))


#: Cases pinned off tier 2's own compiler rather than off the splitter's tier choice.
DIRECT_CASES: dict[str, Callable[[], DataFrame]] = {
    "mixed_aggregation": mixed_aggregation,
    "measure_grand_total": measure_grand_total,
}

ALL_CASES: dict[str, Callable[[], DataFrame]] = {**CASES, **DIRECT_CASES}


# ---------------------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------------------


def sql_step(frame: DataFrame) -> RemoteStep:
    """The one tier-2 step of a frame — the DAG cases have exactly one."""
    steps = [step for step in frame._compiled().steps if step.compilation.is_sql]
    assert len(steps) == 1, f"expected exactly one tier-2 step, got {len(steps)}"
    return steps[0]


def compiled(name: str) -> SemanticCompilation:
    """The tier-2 compilation of a case, however it is reached."""
    frame = ALL_CASES[name]()
    if name in DIRECT_CASES:
        return compile_sql(frame.logical_plan)
    return sql_step(frame).compilation


def envelope(name: str) -> WireDict:
    return build_envelope(compiled(name), None)


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


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_generated_sql_snapshot(name: str) -> None:
    assert_matches_sql_snapshot(f"{PREFIX}{name}", compiled(name).query.user_edited_sql)


@pytest.mark.parametrize("name", sorted(ALL_CASES))
def test_tier_two_envelope_snapshot(name: str) -> None:
    assert_matches_snapshot(f"{ENVELOPE_PREFIX}{name}", envelope(name))


def test_no_orphan_sql_snapshots() -> None:
    expected = {f"{PREFIX}{name}.sql" for name in ALL_CASES}
    actual = {path.name for path in SNAPSHOT_DIR.glob(f"{PREFIX}*.sql")}

    assert actual == expected, "the generated-SQL snapshots are out of sync with the cases"


def test_no_orphan_tier_two_envelope_snapshots() -> None:
    expected = {f"{ENVELOPE_PREFIX}{name}.json" for name in ALL_CASES}
    actual = {path.name for path in SNAPSHOT_DIR.glob(f"{ENVELOPE_PREFIX}*.json")}

    assert actual == expected, "the tier-2 envelope snapshots are out of sync with the cases"


# ---------------------------------------------------------------------------------------
# The invariants every tier-2 envelope has to carry (docs/SQLTIER.md §2)
# ---------------------------------------------------------------------------------------


def every_query() -> list[dict[str, Any]]:
    return [envelope(name)["query"] for name in ALL_CASES]


def test_every_tier_two_query_omits_rewrite_sql_entirely() -> None:
    """An ABSENT key is what selects the parsed-OmniSQL path; ``false`` runs it verbatim (§3.6)."""
    for query in every_query():
        assert query["userEditedSQL"].strip()
        assert "rewriteSql" not in query, "even `false` would send the ${…} refs to the warehouse"


def test_no_tier_two_query_carries_the_v1_mechanism() -> None:
    """The statement IS the plan: there is no reference core and no SQL-sort wrapper (§2)."""
    for query in every_query():
        assert "staticQueryReferences" not in query
        assert "sqlSortsEnabled" not in query
        assert query["sorts"] == []
        assert query["column_totals"] == {}
        assert query["fields"] == []


def test_every_tier_two_statement_binds_against_the_model() -> None:
    """``FROM ${topic}`` is what makes this a governed job rather than warehouse SQL."""
    for query in every_query():
        assert "FROM ${order_items}" in query["userEditedSQL"]
        assert "__OF_REF_" not in query["userEditedSQL"], "no sentinel survives substitution"


def test_every_tier_two_query_sends_an_explicit_limit_and_version_9() -> None:
    """The server ignores the query object's limit here, so the text carries the real one (§2)."""
    for query in every_query():
        assert query["limit"] is not None
        assert f"LIMIT {query['limit']}" in query["userEditedSQL"]
        assert query["version"] == 9


def test_no_tier_two_statement_ever_says_select_distinct() -> None:
    """``SELECT DISTINCT`` is silently stripped by the parser (§3.6), so dedup stays local."""
    for query in every_query():
        assert "SELECT DISTINCT" not in query["userEditedSQL"]


def test_a_tier_two_step_renames_only_what_the_server_does_not_name() -> None:
    """§3.2's two regimes: the bare ref keeps its wire name, the expression item is renamed."""
    frame = ad_hoc_group_by()
    step = sql_step(frame)

    assert step.alias_map == {"of_expr_1": "buyers"}
    assert step.columns == ("users.state", "buyers")
    assert frame.columns == ("users.state", "buyers")


def test_a_bare_ref_is_emitted_without_an_alias_at_all() -> None:
    """The server ignores aliases on bare refs, so writing one would only mislead a reader."""
    statement = compiled("grained_group_key").query.user_edited_sql

    assert "${order_items.created_at[month]}" in statement
    assert "${order_items.created_at[month]} AS" not in statement


def test_the_default_limit_applies_to_the_final_select_exactly_as_in_tier_one() -> None:
    query = envelope("ad_hoc_group_by")["query"]

    assert query["limit"] == DEFAULT_FETCH_LIMIT
    assert f"LIMIT {DEFAULT_FETCH_LIMIT}" in query["userEditedSQL"]


def test_an_offset_rides_the_sql_text_and_the_envelope_together() -> None:
    query = envelope("sort_limit_offset")["query"]

    assert query["limit"] == 10
    assert query["offset"] == 5
    assert "OFFSET 5" in query["userEditedSQL"]


def test_the_snapshot_directory_paths_are_the_ones_this_module_writes() -> None:
    """Guards the shared snapshot directory against a rename that silently orphans files."""
    assert (SNAPSHOT_DIR / f"{PREFIX}ad_hoc_group_by.sql").exists()
    assert (SNAPSHOT_DIR / f"{ENVELOPE_PREFIX}ad_hoc_group_by.json").exists()
    assert Path(SNAPSHOT_DIR).name == "snapshots"
