"""Differential lane: tier 2 against tier 3, on the same logical plan (docs/SQLTIER.md §7).

The headline check of M5.  Every case here builds **one** DataFrame and runs it twice — once
compiled normally, so the computation happens in the warehouse as SQL, and once with
``disable_sql``, so the identical question is answered by the local Arrow engine over raw rows.
The two results must agree through the existing comparator.

That is a stronger statement than "the SQL looks right".  NULL semantics, group-key handling,
``count_distinct``'s treatment of NULL, decimal arithmetic and null ordering are all places where
two engines are entitled to disagree, and every one of them is exercised here over the bench
dataset's deliberately NULL-heavy columns rather than argued about in a docstring.

Both sides still go through FakeOmniAPI, so a third implementation (DuckDB) is in the loop on the
tier-2 side and the governed measures on both.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.compile.splitter import SplitOptions, split
from omniframes.dataframe import DataFrame
from omniframes.errors import TruncationWarning
from omniframes.transport import HttpTransport
from tests.differential.comparator import assert_frames_agree
from tests.fakes import (
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

BASE_URL = "https://bench.example.omni.co"

REVENUE = "order_items.total_sale_price"
ID = "order_items.id"
STATE = "users.state"
AGE = "users.age"
STATUS = "order_items.status"
PRICE = "order_items.sale_price"
QUANTITY = "order_items.quantity"
DISCOUNT = "order_items.discount"
CATEGORY = "products.category"
BUYER = "users.id"


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


@pytest.fixture
def orders(handler: FakeOmniAPI) -> Iterator[DataFrame]:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    with client:
        transport = HttpTransport(
            base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
        )
        session = OmniSession.builder.base_url(BASE_URL).transport(transport).get_or_create()
        yield session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)


def collect(frame: DataFrame, *, disable_sql: bool) -> Any:
    """Run ``frame``'s plan with tier 2 on or off, from the same logical plan either way."""
    twin = DataFrame(frame.session, frame.logical_plan)
    twin._execution = split(
        frame.logical_plan,
        options=SplitOptions(envelope=frame.session.envelope_options(), disable_sql=disable_sql),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TruncationWarning)
        return twin.collect()


def agree(frame: DataFrame, *, sort: bool = True) -> Any:
    """Assert the two tiers answer the same question the same way; return the tier-2 table."""
    pushed = collect(frame, disable_sql=False)
    local = collect(frame, disable_sql=True)
    assert_frames_agree(pushed, local, sort=sort)
    return pushed


def tiers(frame: DataFrame) -> tuple[int, int]:
    """The tier each half actually ran at — a case where both are 3 would prove nothing."""
    options = SplitOptions(envelope=frame.session.envelope_options())
    with_sql = split(frame.logical_plan, options=options)
    without = split(frame.logical_plan, options=SplitOptions(disable_sql=True))
    return with_sql.steps[-1].tier, without.steps[-1].tier


# --------------------------------------------------------------------------------------
# The premise: the two runs really are two different tiers
# --------------------------------------------------------------------------------------


def test_the_two_runs_are_a_sql_job_and_a_raw_scan(orders: DataFrame) -> None:
    """Without this the whole file could be comparing tier 3 against itself."""
    frame = orders.group_by(STATE).agg(F.count_distinct(BUYER).alias("buyers"))

    assert tiers(frame) == (2, 1), "tier 2 writes SQL; with it off the scan is a tier-1 query"
    assert "userEditedSQL" in str(frame.explain()) or "tier 2 · sql" in frame.explain()


# --------------------------------------------------------------------------------------
# Ad-hoc aggregation — every function, over NULL-heavy columns and a NULL group key
# --------------------------------------------------------------------------------------


def test_every_aggregate_function_agrees_across_the_two_tiers(orders: DataFrame) -> None:
    """sum/count/count_distinct/avg/min/max at once, grouped by a column that has NULLs."""
    table = agree(
        orders.group_by(STATE).agg(
            F.sum(PRICE).alias("total"),
            F.count(ID).alias("n"),
            F.count_distinct(BUYER).alias("buyers"),
            F.avg(PRICE).alias("mean"),
            F.min(AGE).alias("youngest"),
            F.max(AGE).alias("oldest"),
        )
    )

    assert None in {row[STATE] for row in table.to_pylist()}, "the NULL group is a real group"
    assert sum(row["buyers"] for row in table.to_pylist()) == 497


def test_an_aggregate_over_a_null_heavy_column_agrees(orders: DataFrame) -> None:
    """``discount`` is NULL on most rows: SUM skips them on both sides, COUNT counts the rest."""
    table = agree(
        orders.group_by(STATUS).agg(
            F.sum(DISCOUNT).alias("total"),
            F.count(DISCOUNT).alias("n"),
            F.avg(DISCOUNT).alias("mean"),
            F.min(DISCOUNT).alias("smallest"),
            F.max(DISCOUNT).alias("largest"),
        )
    )

    assert table.num_rows == 5


def test_count_distinct_excludes_null_on_both_sides(orders: DataFrame) -> None:
    table = agree(orders.group_by(STATUS).agg(F.count_distinct(STATE).alias("states")))

    assert all(row["states"] > 0 for row in table.to_pylist())


def test_a_group_less_aggregate_is_one_row_either_way(orders: DataFrame) -> None:
    table = agree(
        orders.group_by().agg(F.sum(PRICE).alias("total"), F.count_distinct(STATE).alias("states"))
    )

    assert table.num_rows == 1


def test_two_group_keys_including_a_nullable_one(orders: DataFrame) -> None:
    agree(orders.group_by(STATUS, CATEGORY).agg(F.count(ID).alias("n"), F.sum(PRICE).alias("t")))


def test_a_grained_group_key_agrees(orders: DataFrame) -> None:
    """The grain is computed by the *reference*, so both tiers group the same bucketed values."""
    month = F.col("order_items.created_at").grain("month").alias("month")
    agree(orders.group_by(month).agg(F.count_distinct(BUYER).alias("buyers")))


# --------------------------------------------------------------------------------------
# Mixed aggregation: the governed half stays tier 1 in both runs
# --------------------------------------------------------------------------------------


def test_a_mixed_aggregate_agrees_across_the_two_tiers(orders: DataFrame) -> None:
    frame = orders.group_by(STATE).agg(
        F.measure(REVENUE).alias("revenue"), F.count_distinct(BUYER).alias("buyers")
    )
    table = agree(frame)

    assert table.column_names == [STATE, "revenue", "buyers"]
    assert table.num_rows == 21, "the NULL-state group is one row on both sides of the align-join"


def test_a_filter_below_a_mixed_aggregate_narrows_both_halves_the_same_way(
    orders: DataFrame,
) -> None:
    agree(
        orders.filter(F.col(STATUS) == "complete")
        .group_by(STATE)
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct(BUYER).alias("buyers"))
    )


# --------------------------------------------------------------------------------------
# HAVING on an ad-hoc output column
# --------------------------------------------------------------------------------------


def test_an_uncompilable_filter_below_an_aggregate_agrees(orders: DataFrame) -> None:
    """The WHERE runs pre-aggregation over the reference's rows on one side, over the raw scan
    on the other — same rows in, same groups out."""
    predicate = (F.col(STATE) == "California") | (F.col(AGE) > 60)
    table = agree(orders.filter(predicate).group_by(STATUS).agg(F.count(ID).alias("n")))

    assert table.num_rows > 0


def test_a_having_on_an_ad_hoc_aggregate_agrees(orders: DataFrame) -> None:
    frame = (
        orders.group_by(STATE)
        .agg(F.count_distinct(BUYER).alias("buyers"))
        .filter(F.col("buyers") > 25)
    )
    table = agree(frame)

    assert 0 < table.num_rows < 21, "the fixture would prove nothing if it kept every group"
    assert all(row["buyers"] > 25 for row in table.to_pylist())


def test_a_having_over_two_aggregates_agrees(orders: DataFrame) -> None:
    frame = (
        orders.group_by(STATE)
        .agg(F.count(ID).alias("n"), F.count_distinct(BUYER).alias("buyers"))
        .filter((F.col("buyers") > 10) & (F.col("n") < 400))
    )

    assert agree(frame).num_rows > 0


# --------------------------------------------------------------------------------------
# Cross-field OR, arithmetic filters, computed columns
# --------------------------------------------------------------------------------------


def test_a_cross_field_or_agrees(orders: DataFrame) -> None:
    """SQL's three-valued logic and Arrow's Kleene ``or`` must drop exactly the same rows."""
    predicate = (F.col(STATE) == "California") | (F.col(AGE) > 60)
    table = agree(orders.select(ID, STATE, AGE).filter(predicate))

    assert table.num_rows > 0


def test_a_negated_cross_field_or_agrees(orders: DataFrame) -> None:
    """``NOT (unknown OR false)`` is unknown, so the row goes — in both engines."""
    predicate = ~((F.col(STATE) == "California") | (F.col(AGE) > 60))
    table = agree(orders.select(ID, STATE, AGE).filter(predicate))

    assert table.num_rows > 0
    assert None not in {row[STATE] for row in table.to_pylist()}


def test_an_arithmetic_filter_agrees(orders: DataFrame) -> None:
    agree(orders.select(ID, PRICE).filter(F.col(PRICE) * F.col(QUANTITY) > 200))


def test_a_computed_column_agrees(orders: DataFrame) -> None:
    agree(
        orders.select(ID, PRICE, QUANTITY).with_column("line_total", F.col(PRICE) * F.col(QUANTITY))
    )


def test_a_computed_column_propagates_nulls_the_same_way(orders: DataFrame) -> None:
    table = agree(
        orders.select(ID, PRICE, DISCOUNT).with_column("net", F.col(PRICE) - F.col(DISCOUNT))
    )

    assert any(row["net"] is None for row in table.to_pylist()), "NULL - x is NULL, both sides"


def test_integer_division_promotes_the_same_way(orders: DataFrame) -> None:
    """SQL's ``/`` over integers and the local engine's both produce a real quotient."""
    agree(orders.select(ID, AGE).with_column("halved", F.col(AGE) / 2))


# --------------------------------------------------------------------------------------
# Sort and limit above an aggregate
# --------------------------------------------------------------------------------------


def test_sort_and_limit_above_an_aggregate_agree_row_for_row(orders: DataFrame) -> None:
    """Ordering is asserted, not normalized away: nulls last in both engines, by construction."""
    frame = (
        orders.group_by(STATE)
        .agg(F.count(ID).alias("n"))
        .sort(F.col("n").desc(), F.col(STATE))
        .limit(6)
    )
    table = agree(frame, sort=False)

    assert table.num_rows == 6


def test_a_sort_that_puts_nulls_last_agrees(orders: DataFrame) -> None:
    frame = orders.group_by(STATE).agg(F.count(ID).alias("n")).sort(F.col(STATE))
    table = agree(frame, sort=False)

    assert table.to_pylist()[-1][STATE] is None, "the NULL group sorts last on both sides"


# --------------------------------------------------------------------------------------
# String predicates and IN, which tier 2 renders as LIKE / IN rather than as wire filters
# --------------------------------------------------------------------------------------


def test_a_case_insensitive_contains_inside_a_cross_field_or_agrees(orders: DataFrame) -> None:
    predicate = F.col(STATE).contains("cal", case_insensitive=True) | (F.col(AGE) > 90)
    table = agree(orders.select(ID, STATE, AGE).filter(predicate))

    assert table.num_rows > 0


def test_an_isin_inside_a_cross_field_or_agrees(orders: DataFrame) -> None:
    predicate = F.col(STATE).isin("Ohio", "Texas", "Georgia") | (F.col(AGE) > 90)
    table = agree(orders.select(ID, STATE, AGE).filter(predicate))

    assert table.num_rows > 0


def test_a_negated_isin_inside_a_cross_field_or_agrees(orders: DataFrame) -> None:
    """``NOT (x IN (...))`` is NULL for a NULL ``x``, so the NULL rows are dropped on both sides.

    Arrow's ``is_in`` answers set membership and returns FALSE (not NULL) for a NULL operand;
    negating that would keep every NULL-state row the SQL tier drops.  This is the one shape
    that catches it, because a positive ``isin`` agrees either way.
    """
    predicate = ~F.col(STATE).isin("California", "Texas") | (F.col(ID) < 0)
    table = agree(orders.select(ID, STATE).filter(predicate))

    assert table.num_rows > 0
    assert all(row[STATE] is not None for row in table.to_pylist()), (
        "a negated predicate never resurrects NULLs (docs/mental-model.md)"
    )


def test_a_null_check_inside_a_cross_field_or_agrees(orders: DataFrame) -> None:
    predicate = F.col(STATE).is_null() | (F.col(AGE) > 90)
    table = agree(orders.select(ID, STATE, AGE).filter(predicate))

    assert table.num_rows > 0
