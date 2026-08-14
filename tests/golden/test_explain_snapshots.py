"""Golden lane: ``explain()`` output for split plans, as checked-in text (docs/HYBRID.md §7).

``explain()`` is the promise that nothing runs locally in secret, so its exact wording is part
of the contract and belongs in a diff-reviewed snapshot rather than in a scatter of substring
assertions.  These cases never talk to anything — compiling is pure.

Regenerate after an intentional change::

    OMNIFRAMES_UPDATE_SNAPSHOTS=1 uv run pytest tests/golden -q
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from tests.fakes import BENCH_MODEL_ID, BENCH_MODEL_NAME, BENCH_TOPIC_NAME
from tests.golden.test_semantic_snapshots import SESSION

from omniframes import functions as F
from omniframes.dataframe import DataFrame
from omniframes.plan import nodes
from omniframes.types import OmniDataType, OmniField, OmniSchema

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_ENV_VAR = "OMNIFRAMES_UPDATE_SNAPSHOTS"
PREFIX = "explain_"

REVENUE = "order_items.total_sale_price"
TOPIC_SCAN = nodes.Scan(
    nodes.TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)
SQL_SCAN = nodes.Scan(
    nodes.SqlScan(
        model_id=BENCH_MODEL_ID,
        sql=(
            "SELECT u.state AS state, SUM(oi.sale_price) AS revenue\n"
            "FROM order_items oi LEFT JOIN users u ON u.id = oi.user_id\n"
            "GROUP BY 1"
        ),
        model_name=BENCH_MODEL_NAME,
    )
)


def topic() -> DataFrame:
    return DataFrame(SESSION, TOPIC_SCAN)


def assert_matches_snapshot(name: str, text: str) -> None:
    path = SNAPSHOT_DIR / f"{name}.txt"
    serialized = text if text.endswith("\n") else text + "\n"

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
        f"explain() drifted from {path.name}; if the change is intended regenerate with "
        f"{UPDATE_ENV_VAR}=1"
    )


# ---------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------


def fully_pushed_down() -> DataFrame:
    """The control case: one query, and the local section says so."""
    return topic().select("users.state", REVENUE).limit(10)


def mixed_aggregation() -> DataFrame:
    """The canonical decomposition — the worked example of docs/HYBRID.md §7."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers"))
        .sort(F.col("buyers").desc())
    )


def ad_hoc_only_aggregation() -> DataFrame:
    """No governed measure: one raw scan, aggregated locally, no join to make."""
    return topic().group_by("products.category").agg(F.avg("order_items.sale_price"))


def udf_filter() -> DataFrame:
    """A UDF pins the frontier: the query below it still filters and limits remotely."""

    def is_coastal(state: str | None) -> bool:
        return state in {"California", "Oregon", "Washington"}

    return (
        topic()
        .select("users.state", "order_items.status")
        .filter(F.col("order_items.status") == "complete")
        .filter(F.udf(is_coastal)("users.state") == True)  # noqa: E712 - a Column, not a bool
    )


def computed_column() -> DataFrame:
    """Arithmetic is not expressible on the wire, so the derived column is computed here."""
    return (
        topic()
        .select("users.state", "order_items.sale_price", "order_items.quantity")
        .with_column("unit_price", F.col("order_items.sale_price") / F.col("order_items.quantity"))
        .limit(25)
    )


def limit_pins_the_frontier() -> DataFrame:
    """The limit rides remote; the operations written above it run on its result (§2.3)."""
    return (
        topic()
        .select("users.state", "order_items.status")
        .limit(100)
        .filter(F.col("users.state") == "California")
    )


def cross_frame_join() -> DataFrame:
    """Two governed queries, split independently, combined here with SQL join semantics (M4)."""
    revenue = topic().group_by("users.state").agg(F.measure(REVENUE).alias("revenue"))
    buyers = topic().group_by("users.state").agg(F.measure("users.count").alias("buyers"))
    return revenue.join(buyers, "users.state", "left").sort(F.col("revenue").desc())


def sql_scan_with_local_ops() -> DataFrame:
    """A raw-SQL job is opaque: it rides the wire as written, and the rest happens here."""
    scan = DataFrame(SESSION, SQL_SCAN)
    return scan.select("state", "revenue").filter(F.col("state") == "California").limit(5)


def sql_tier_statement() -> DataFrame:
    """A tier-2 step in full: the one OmniSQL statement omniframes wrote, elided.

    Long enough to reach the eight-line budget, so the ``… (+K more lines)`` tail is pinned too.
    """
    return (
        topic()
        .group_by("users.state")
        .agg(F.count_distinct("users.id").alias("buyers"), F.sum("order_items.sale_price"))
        .filter(F.col("buyers") > 25)
        .sort(F.col("buyers").desc())
        .limit(10)
    )


def sql_tier_outer_where() -> DataFrame:
    """A cross-field OR: no wire filter expresses it, so every predicate rides the WHERE (§3.3)."""
    return (
        topic()
        .filter(F.col("order_items.status") == "complete")
        .select("users.state", "users.age")
        .filter((F.col("users.state") == "California") | (F.col("users.age") > 60))
    )


def map_pandas_with_hint() -> DataFrame:
    """``map_pandas`` is local by definition; the hint keeps the frame's schema knowable."""
    hint = OmniSchema(
        (
            OmniField(name="state", data_type=OmniDataType.STRING),
            OmniField(name="rank", data_type=OmniDataType.NUMBER),
        )
    )
    return topic().select("users.state").limit(50).map_pandas(rank_states, hint)


def rank_states(frame: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - never called
    raise AssertionError("the golden lane never runs the function, only names it")


CASES: dict[str, Callable[[], DataFrame]] = {
    "fully_pushed_down": fully_pushed_down,
    "mixed_aggregation": mixed_aggregation,
    "ad_hoc_only_aggregation": ad_hoc_only_aggregation,
    "udf_filter": udf_filter,
    "computed_column": computed_column,
    "limit_pins_the_frontier": limit_pins_the_frontier,
    "map_pandas_with_hint": map_pandas_with_hint,
    "sql_tier_statement": sql_tier_statement,
    "sql_tier_outer_where": sql_tier_outer_where,
    "cross_frame_join": cross_frame_join,
    "sql_scan_with_local_ops": sql_scan_with_local_ops,
}


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_explain_snapshot(name: str) -> None:
    assert_matches_snapshot(f"{PREFIX}{name}", CASES[name]().explain())


def test_no_orphan_explain_snapshots() -> None:
    expected = {f"{PREFIX}{name}.txt" for name in CASES}
    actual = {path.name for path in SNAPSHOT_DIR.glob(f"{PREFIX}*.txt")}

    assert actual == expected, "the explain snapshots directory is out of sync with the cases"


def test_a_single_query_plan_keeps_the_pre_m3_rendering() -> None:
    """A plan that did not change must not read as if it had."""
    text = fully_pushed_down().explain()

    assert text.startswith("== Physical plan ==\nRemote [tier 1 · semantic → POST ")
    assert text.endswith("Local [pandas]\n  (none — fully pushed down)")
    assert "Remote step" not in text


def test_every_split_plan_names_every_step_it_runs() -> None:
    """No silent local fallback: each remote query is numbered and each operator is a line."""
    for name, build in CASES.items():
        execution = build()._compiled()
        if execution.root is None:
            continue
        text = build().explain()
        assert "Local [arrow compute]" in text, name
        for index in range(1, len(execution.steps) + 1):
            assert f"Remote step {index} [" in text, f"{name} did not number step {index}"
