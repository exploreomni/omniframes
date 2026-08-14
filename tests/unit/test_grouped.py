"""Unit tests for ``group_by().agg()``, measure filters and ``with_totals()`` (M2).

The group-by is sugar over a projection (docs/DESIGN.md §2), so most of what is asserted here is
an *equality*: whatever the sugar builds must compile to the query the plain ``select()`` builds.
Build-time validation gets the same treatment as the alias rules — a typo fails where it was
written, not at action time.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

import httpx
import pytest
from tests.fakes import (
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

from omniframes import functions as F
from omniframes.dataframe import ROW_TYPE_COLUMN, DataFrame, GroupedData
from omniframes.errors import CompileError
from omniframes.plan import Aggregate, Scan, TopicScan
from omniframes.session import OmniSession
from omniframes.transport import HttpTransport

BASE_URL = "https://bench.example.omni.co"
REVENUE = "order_items.total_sale_price"
COUNT = "order_items.count"

SCAN = Scan(
    TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


@pytest.fixture
def session(handler: FakeOmniAPI) -> Iterator[OmniSession]:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    with client:
        transport = HttpTransport(
            base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
        )
        yield OmniSession.builder.transport(transport).get_or_create()


@pytest.fixture
def df(session: OmniSession) -> DataFrame:
    return DataFrame(session, SCAN)


def query_of(frame: DataFrame) -> dict[str, Any]:
    envelope: dict[str, Any] = frame._compiled().envelope
    query: dict[str, Any] = envelope["query"]
    return query


# --------------------------------------------------------------------------------------
# group_by().agg() is sugar
# --------------------------------------------------------------------------------------


def test_group_by_agg_builds_an_aggregate_node(df: DataFrame) -> None:
    frame = df.group_by("users.state").agg(F.measure(REVENUE))
    plan = frame.logical_plan

    assert isinstance(plan, Aggregate)
    assert [key.expr for key in plan.keys] == [F.col("users.state").expr]
    assert [agg.expr for agg in plan.aggs] == [F.measure(REVENUE).expr]


def test_group_by_agg_compiles_to_the_same_query_as_the_equivalent_select(df: DataFrame) -> None:
    grouped = df.group_by("users.state").agg(F.measure(REVENUE), F.measure(COUNT))
    selected = df.select("users.state", F.measure(REVENUE), F.measure(COUNT))

    assert query_of(grouped) == query_of(selected)
    assert grouped.columns == selected.columns == ("users.state", REVENUE, COUNT)


def test_group_by_is_the_camel_case_alias(df: DataFrame) -> None:
    assert query_of(df.groupBy("users.state").agg(F.measure(COUNT))) == query_of(
        df.group_by("users.state").agg(F.measure(COUNT))
    )


def test_group_by_accepts_names_columns_grains_and_iterables(df: DataFrame) -> None:
    month = F.col("order_items.created_at").grain("month")
    frame = df.group_by(month, ["users.state"]).agg(F.measure(COUNT))

    assert query_of(frame)["fields"] == [
        "order_items.created_at[month]",
        "users.state",
        COUNT,
    ]


def test_group_by_with_no_keys_is_one_aggregate_row(df: DataFrame) -> None:
    frame = df.group_by().agg(F.measure(REVENUE))

    assert query_of(frame)["fields"] == [REVENUE]
    assert frame.collect().num_rows == 1


def test_aliases_survive_the_group_by(df: DataFrame) -> None:
    frame = (
        df.group_by(F.col("users.state").alias("state"))
        .agg(F.measure(REVENUE).alias("revenue"))
        .sort(F.col("revenue").desc())
    )
    query = query_of(frame)

    assert query["fields"] == ["users.state", REVENUE]
    assert [sort["column_name"] for sort in query["sorts"]] == [REVENUE]
    assert frame.columns == ("state", "revenue")


def test_grouped_data_repr_names_its_keys(df: DataFrame) -> None:
    grouped = df.group_by("users.state", F.col("order_items.created_at").grain("month"))

    assert isinstance(grouped, GroupedData)
    assert repr(grouped) == "GroupedData[users.state, order_items.created_at[month]]"
    assert [key.expr for key in grouped.keys] == [
        F.col("users.state").expr,
        F.col("order_items.created_at").grain("month").expr,
    ]


# --------------------------------------------------------------------------------------
# Build-time validation
# --------------------------------------------------------------------------------------


def test_a_measure_is_not_a_group_key(df: DataFrame) -> None:
    with pytest.raises(CompileError, match=r"group_by\(\) takes dimensions"):
        df.group_by(F.measure(REVENUE))


def test_an_aggregation_is_not_a_group_key(df: DataFrame) -> None:
    with pytest.raises(CompileError, match=r"belong in \.agg"):
        df.group_by(F.count_distinct("users.id"))


def test_a_dimension_is_not_an_aggregate(df: DataFrame) -> None:
    with pytest.raises(CompileError, match=r"belong in group_by"):
        df.group_by("users.state").agg("users.age")


def test_agg_needs_at_least_one_aggregate(df: DataFrame) -> None:
    with pytest.raises(CompileError, match="at least one aggregate"):
        df.group_by("users.state").agg()


def test_alias_collisions_fail_when_the_aggregate_is_built(df: DataFrame) -> None:
    with pytest.raises(CompileError, match="used twice"):
        df.group_by(F.col("users.state").alias("x")).agg(F.measure(REVENUE).alias("x"))


def test_ad_hoc_aggregations_compile_to_one_omnisql_statement(df: DataFrame) -> None:
    """M5: the GROUP BY runs in the warehouse, as one governed OmniSQL statement.

    M1 refused this plan, M3 executed it over an unlimited raw scan, and M5 pushes the whole
    aggregate into SQL — the plan never changed, only which tier can express it.
    """
    frame = df.group_by("users.state").agg(F.count_distinct("users.id"))
    text = frame.explain()

    assert isinstance(frame.logical_plan, Aggregate)
    assert frame.columns == ("users.state", "count_distinct(users.id)")
    assert "tier 2 · sql" in text
    assert "COUNT(DISTINCT ${users.id})" in text
    assert "FROM ${order_items}" in text


# --------------------------------------------------------------------------------------
# Measure filters (HAVING)
# --------------------------------------------------------------------------------------


def test_a_measure_filter_after_the_aggregate_is_a_measure_keyed_entry(df: DataFrame) -> None:
    frame = df.group_by("users.state").agg(F.measure(REVENUE)).filter(F.measure(REVENUE) > 50000)

    assert query_of(frame)["filters"] == {
        REVENUE: {
            "type": "number",
            "kind": "GREATER_THAN",
            "values": ["50000"],
            "is_inclusive": False,
        }
    }


def test_a_measure_filter_reads_the_same_before_and_after_the_group_by(df: DataFrame) -> None:
    predicate = F.measure(REVENUE) > 50000
    after = df.group_by("users.state").agg(F.measure(REVENUE)).filter(predicate)
    before = df.filter(predicate).group_by("users.state").agg(F.measure(REVENUE))

    assert query_of(after) == query_of(before)


def test_conditions_on_one_measure_merge_into_a_single_entry(df: DataFrame) -> None:
    revenue = F.measure(REVENUE)
    frame = df.group_by("users.state").agg(revenue).filter(revenue > 20000).filter(revenue < 80000)
    filters = query_of(frame)["filters"]

    assert list(filters) == [REVENUE]
    assert filters[REVENUE]["conjunction"] == "AND"


def test_a_dimension_filter_and_a_measure_filter_are_separate_entries(df: DataFrame) -> None:
    frame = (
        df.group_by("users.state")
        .agg(F.measure(REVENUE))
        .filter(F.col("order_items.status") == "complete")
        .filter(F.measure(REVENUE) > 50000)
    )

    assert set(query_of(frame)["filters"]) == {"order_items.status", REVENUE}


def test_a_filter_written_against_a_measure_alias_still_reads_as_a_having(df: DataFrame) -> None:
    """The alias resolves to the measure's wire name, so the entry is the same HAVING."""
    frame = (
        df.group_by("users.state")
        .agg(F.measure(REVENUE).alias("revenue"))
        .filter(F.col("revenue") > 50000)
    )

    assert list(query_of(frame)["filters"]) == [REVENUE]
    assert f"having: {REVENUE} > 50000" in frame.explain()


def test_an_ad_hoc_aggregation_filter_still_routes_to_a_later_tier(df: DataFrame) -> None:
    frame = df.select("users.state").filter(F.count_distinct("users.id") > 10)

    with pytest.raises(CompileError, match=r"not yet supported: .*tier 2/3"):
        frame.collect()


# --------------------------------------------------------------------------------------
# with_totals()
# --------------------------------------------------------------------------------------


def test_with_totals_asks_for_the_grand_total(df: DataFrame) -> None:
    frame = df.group_by("users.state").agg(F.measure(REVENUE)).with_totals()

    assert query_of(frame)["column_totals"] == {"::total::": {"type": "aggregation"}}


def test_with_totals_is_opt_in(df: DataFrame) -> None:
    frame = df.group_by("users.state").agg(F.measure(REVENUE))

    assert query_of(frame)["column_totals"] == {}


def test_with_totals_survives_further_transformations(df: DataFrame) -> None:
    frame = df.with_totals().group_by("users.state").agg(F.measure(REVENUE)).limit(5)

    assert query_of(frame)["column_totals"] == {"::total::": {"type": "aggregation"}}
    assert ROW_TYPE_COLUMN in frame.columns


def test_with_totals_without_a_measure_says_why(df: DataFrame) -> None:
    frame = df.select("users.state").with_totals()

    with pytest.raises(CompileError, match="at least one governed measure"):
        frame.collect()


def test_the_frame_advertises_the_row_type_column(df: DataFrame) -> None:
    frame = df.group_by("users.state").agg(F.measure(REVENUE)).with_totals()

    assert frame.columns == ("users.state", REVENUE, ROW_TYPE_COLUMN)
    assert frame.collect().column_names == ["users.state", REVENUE, ROW_TYPE_COLUMN]
    assert frame.schema.names == frame.columns, "the derived column is part of the schema too"


# --------------------------------------------------------------------------------------
# explain()
# --------------------------------------------------------------------------------------


def test_explain_renders_the_group_by_the_having_and_the_totals(df: DataFrame) -> None:
    text = (
        df.group_by("users.state")
        .agg(F.measure(REVENUE))
        .filter(F.col("order_items.status") == "complete")
        .filter(F.measure(REVENUE) > 50000)
        .with_totals()
        .explain()
    )

    assert "group by: [users.state]" in text
    assert f"measures: [{REVENUE}]" in text
    assert "filters: order_items.status = 'complete'" in text
    assert f"having: {REVENUE} > 50000" in text
    assert "totals: column_totals [::total::]" in text


def test_explain_keeps_where_and_having_apart(df: DataFrame) -> None:
    text = (
        df.select("users.state")
        .filter(F.col("users.state") == "Ohio")
        .filter(F.measure(COUNT) > 10)
        .explain()
    )

    assert "filters: users.state = 'Ohio'" in text
    assert f"having: {COUNT} > 10" in text
    assert f"filters: {COUNT}" not in text


def test_explain_says_when_an_aggregate_has_no_group_keys(df: DataFrame) -> None:
    text = df.group_by().agg(F.measure(REVENUE)).explain()

    assert "group by: (none — one aggregate row)" in text


def test_explain_of_a_plain_projection_has_no_group_by_line(df: DataFrame) -> None:
    assert "group by:" not in df.select("users.state").explain()


def test_explain_spells_out_the_two_between_bounds(df: DataFrame) -> None:
    """The renderings differ because the semantics do (see Column.between)."""
    numbers = (
        df.select("order_items.quantity")
        .filter(F.col("order_items.quantity").between(3, 5))
        .explain()
    )
    dates = (
        df.select("order_items.created_at")
        .filter(F.col("order_items.created_at").between(date(2025, 7, 1), date(2026, 7, 1)))
        .explain()
    )

    assert "(order_items.quantity >= 3 AND order_items.quantity <= 5)" in numbers
    assert "'2025-07-01' <= order_items.created_at < '2026-07-01'" in dates
