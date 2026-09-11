"""Differential lane: every operation, pushdown vs an independent pandas reference (§4).

Each case runs the *same logical question* twice — once through the whole omniframes pipeline
against FakeOmniAPI (which executes the governed half in DuckDB and the rest in the local Arrow
engine), and once through :mod:`tests.differential.reference`, which is plain pandas plus
explicit Python.  Three implementations therefore have to agree on every answer, and the ones
that disagree in practice are always the null cases: that is why every fixture here runs over
NULL-heavy columns (``users.state``, ``users.age``, ``order_items.returned``,
``order_items.discount``) rather than the clean ones.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import httpx
import pyarrow as pa
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.dataframe import DataFrame
from omniframes.errors import TruncationWarning
from omniframes.transport import HttpTransport
from tests.differential.comparator import assert_frames_agree
from tests.differential.reference import (
    Row,
    aggregate,
    and_,
    bench_rows,
    keep,
    known,
    not_,
    or_,
    project,
    sql_join,
    stack,
    table_of,
)
from tests.differential.reference import (
    truth as check,
)
from tests.fakes import (
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

BASE_URL = "https://bench.omniapp.co"
REVENUE = "order_items.total_sale_price"

ID = "order_items.id"
STATE = "users.state"
AGE = "users.age"
STATUS = "order_items.status"
PRICE = "order_items.sale_price"
QUANTITY = "order_items.quantity"
DISCOUNT = "order_items.discount"
RETURNED = "order_items.returned"
CATEGORY = "products.category"


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


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    return bench_rows()


def collect(frame: DataFrame) -> Any:
    """Run a frame, ignoring truncation: these fixtures are sized to fit and the lane is about
    values, not about pagination."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TruncationWarning)
        return frame.collect()


# --------------------------------------------------------------------------------------
# Filters, including the Kleene cases
# --------------------------------------------------------------------------------------


def test_equality_on_a_nullable_column(orders: DataFrame, rows: list[Row]) -> None:
    pushdown = collect(orders.select(ID, STATE).filter(F.col(STATE) == "California"))
    reference = project(
        keep(rows, lambda row: check(row[STATE], lambda v: v == "California")),
        {ID: ID, STATE: STATE},
    )

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


def test_a_negated_equality_keeps_neither_the_match_nor_the_nulls(
    orders: DataFrame, rows: list[Row]
) -> None:
    """The canonical Kleene case: ``NOT (NULL = 'California')`` is UNKNOWN, so the row goes."""
    pushdown = collect(orders.select(ID, STATE).filter(~(F.col(STATE) == "California")))
    reference = project(
        keep(rows, lambda row: not_(check(row[STATE], lambda v: v == "California"))),
        {ID: ID, STATE: STATE},
    )

    kept = {row[STATE] for row in pushdown.to_pylist()}
    assert None not in kept, "the unknowns went with the matches"
    assert "California" not in kept
    assert_frames_agree(pushdown, reference)


def test_a_negated_cross_field_or_runs_locally_and_still_drops_the_unknowns(
    orders: DataFrame, rows: list[Row]
) -> None:
    """Cross-field OR has no wire filter, so this whole predicate is the local engine's."""
    predicate = ~((F.col(STATE) == "California") | (F.col(AGE) > 60))
    pushdown = collect(orders.select(ID, STATE, AGE).filter(predicate))
    reference = project(
        keep(
            rows,
            lambda row: not_(
                or_(
                    check(row[STATE], lambda v: v == "California"),
                    check(row[AGE], lambda v: v > 60),
                )
            ),
        ),
        {ID: ID, STATE: STATE, AGE: AGE},
    )

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


def test_and_over_two_unknowns(orders: DataFrame, rows: list[Row]) -> None:
    predicate = (F.col(AGE) > 30) & (F.col(DISCOUNT) > 0)
    pushdown = collect(orders.select(ID, AGE, DISCOUNT).filter(predicate))
    reference = project(
        keep(
            rows,
            lambda row: and_(
                check(row[AGE], lambda v: v > 30),
                check(row[DISCOUNT], lambda v: v > 0),
            ),
        ),
        {ID: ID, AGE: AGE, DISCOUNT: DISCOUNT},
    )

    assert_frames_agree(pushdown, reference)


def test_is_null_and_is_not_null_partition_the_rows(orders: DataFrame, rows: list[Row]) -> None:
    missing = collect(orders.select(ID, STATE).filter(F.col(STATE).is_null()))
    present = collect(orders.select(ID, STATE).filter(F.col(STATE).is_not_null()))

    assert_frames_agree(
        missing, project(keep(rows, lambda row: row[STATE] is None), {ID: ID, STATE: STATE})
    )
    assert_frames_agree(
        present, project(keep(rows, lambda row: row[STATE] is not None), {ID: ID, STATE: STATE})
    )
    assert missing.num_rows + present.num_rows == len(rows), "IS NULL and IS NOT NULL are total"


def test_isin_never_matches_a_null(orders: DataFrame, rows: list[Row]) -> None:
    states = ("Ohio", "Texas", "Georgia")
    pushdown = collect(orders.select(ID, STATE).filter(F.col(STATE).isin(*states)))
    reference = project(
        keep(rows, lambda row: check(row[STATE], lambda v: v in states)), {ID: ID, STATE: STATE}
    )

    assert_frames_agree(pushdown, reference)


def test_a_boolean_column_with_nulls(orders: DataFrame, rows: list[Row]) -> None:
    """``returned`` is nullable, so ``NOT returned`` must not resurrect the unknown rows."""
    pushdown = collect(orders.select(ID, RETURNED).filter(~F.col(RETURNED)))
    reference = project(
        keep(rows, lambda row: not_(check(row[RETURNED], bool))), {ID: ID, RETURNED: RETURNED}
    )

    assert 0 < pushdown.num_rows < len(rows)
    assert_frames_agree(pushdown, reference)


def test_between_includes_both_ends_over_numbers(orders: DataFrame, rows: list[Row]) -> None:
    pushdown = collect(orders.select(ID, QUANTITY).filter(F.col(QUANTITY).between(2, 4)))
    reference = project(
        keep(rows, lambda row: check(row[QUANTITY], lambda v: 2 <= v <= 4)),
        {ID: ID, QUANTITY: QUANTITY},
    )

    assert_frames_agree(pushdown, reference)


def test_a_string_predicate_is_case_sensitive(orders: DataFrame, rows: list[Row]) -> None:
    pushdown = collect(orders.select(ID, STATE).filter(F.col(STATE).starts_with("New")))
    reference = project(
        keep(rows, lambda row: check(row[STATE], lambda v: v.startswith("New"))),
        {ID: ID, STATE: STATE},
    )

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


def test_a_filter_on_arithmetic_runs_locally_over_a_widened_scan(
    orders: DataFrame, rows: list[Row]
) -> None:
    predicate = F.col(PRICE) * F.col(QUANTITY) > 200
    pushdown = collect(orders.select(ID, PRICE).filter(predicate))
    reference = project(
        keep(rows, lambda row: check(row[PRICE], lambda v: v * row[QUANTITY] > 200)),
        {ID: ID, PRICE: PRICE},
    )

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


def test_a_udf_predicate(orders: DataFrame, rows: list[Row]) -> None:
    coastal = {"California", "Oregon", "Washington"}

    def is_coastal(state: str | None) -> bool:
        return state in coastal

    pushdown = collect(orders.select(ID, STATE).filter(F.udf(is_coastal)(STATE)))
    reference = project(keep(rows, lambda row: is_coastal(row[STATE])), {ID: ID, STATE: STATE})

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


# --------------------------------------------------------------------------------------
# Ad-hoc aggregations
# --------------------------------------------------------------------------------------


def test_every_aggregate_over_a_null_heavy_group_key(orders: DataFrame, rows: list[Row]) -> None:
    """sum/count/count_distinct/avg/min/max at once, grouped by a column that has NULLs."""
    pushdown = collect(
        orders.group_by(STATE).agg(
            F.sum(PRICE).alias("total"),
            F.count(ID).alias("n"),
            F.count_distinct("users.id").alias("buyers"),
            F.avg(PRICE).alias("mean"),
            F.min(AGE).alias("youngest"),
            F.max(AGE).alias("oldest"),
        )
    )
    reference = aggregate(
        rows,
        [STATE],
        {
            "total": ("sum", PRICE),
            "n": ("count", ID),
            "buyers": ("count_distinct", "users.id"),
            "mean": ("avg", PRICE),
            "youngest": ("min", AGE),
            "oldest": ("max", AGE),
        },
    )

    assert None in {row[STATE] for row in pushdown.to_pylist()}, "the NULL group is a group"
    assert_frames_agree(pushdown, reference)


def test_aggregates_over_a_column_that_is_itself_null_heavy(
    orders: DataFrame, rows: list[Row]
) -> None:
    """``discount`` is NULL on most rows: sum skips them, count counts what is left."""
    pushdown = collect(
        orders.group_by(STATUS).agg(
            F.sum(DISCOUNT).alias("total"),
            F.count(DISCOUNT).alias("n"),
            F.avg(DISCOUNT).alias("mean"),
            F.min(DISCOUNT).alias("smallest"),
        )
    )
    reference = aggregate(
        rows,
        [STATUS],
        {
            "total": ("sum", DISCOUNT),
            "n": ("count", DISCOUNT),
            "mean": ("avg", DISCOUNT),
            "smallest": ("min", DISCOUNT),
        },
    )

    assert_frames_agree(pushdown, reference)


def test_a_group_less_aggregate_is_one_row(orders: DataFrame, rows: list[Row]) -> None:
    pushdown = collect(
        orders.group_by().agg(F.sum(PRICE).alias("total"), F.count_distinct(STATE).alias("states"))
    )
    reference = aggregate(rows, [], {"total": ("sum", PRICE), "states": ("count_distinct", STATE)})

    assert pushdown.num_rows == 1
    assert_frames_agree(pushdown, reference)


def test_two_group_keys_including_a_nullable_one(orders: DataFrame, rows: list[Row]) -> None:
    pushdown = collect(
        orders.group_by(STATUS, CATEGORY).agg(F.count(ID).alias("n"), F.sum(PRICE).alias("total"))
    )
    reference = aggregate(rows, [STATUS, CATEGORY], {"n": ("count", ID), "total": ("sum", PRICE)})

    assert_frames_agree(pushdown, reference)


def test_an_average_over_decimals_is_a_float_on_both_sides(
    orders: DataFrame, rows: list[Row]
) -> None:
    pushdown = collect(orders.group_by(CATEGORY).agg(F.avg(PRICE).alias("mean")))
    reference = aggregate(rows, [CATEGORY], {"mean": ("avg", PRICE)})

    assert pushdown.schema.field("mean").type.equals(pushdown.schema.field("mean").type)
    assert all(isinstance(row["mean"], float) for row in pushdown.to_pylist())
    assert_frames_agree(pushdown, reference)


# --------------------------------------------------------------------------------------
# Mixed aggregation — the canonical decomposition
# --------------------------------------------------------------------------------------


def test_the_governed_measure_and_the_ad_hoc_aggregate_agree_after_the_join(
    orders: DataFrame, rows: list[Row]
) -> None:
    """``order_items.total_sale_price`` is ``SUM(sale_price)`` server-side; the reference sums
    the same column itself, so the two halves are checked against one independent answer."""
    pushdown = collect(
        orders.group_by(STATE).agg(
            F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers")
        )
    )
    reference = aggregate(
        rows,
        [STATE],
        {"revenue": ("sum", PRICE), "buyers": ("count_distinct", "users.id")},
    )

    assert_frames_agree(pushdown, reference)


def test_the_mixed_aggregate_also_matches_the_checked_in_answers(orders: DataFrame) -> None:
    """The third opinion: a hand-written SQL answer nobody in this pipeline produced."""
    expected = {row["state"]: row for row in known("revenue_and_buyers_by_state")}
    rows_out = collect(
        orders.group_by(STATE).agg(
            F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers")
        )
    ).to_pylist()

    assert len(rows_out) == len(expected)
    for row in rows_out:
        answer = expected[row[STATE]]
        assert row["revenue"] == Decimal(answer["total_sale_price"])
        assert row["buyers"] == answer["distinct_buyers"]


def test_a_filter_below_a_mixed_aggregate_narrows_both_halves(
    orders: DataFrame, rows: list[Row]
) -> None:
    pushdown = collect(
        orders.filter(F.col(STATUS) == "complete")
        .group_by(STATE)
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers"))
    )
    reference = aggregate(
        keep(rows, lambda row: check(row[STATUS], lambda v: v == "complete")),
        [STATE],
        {"revenue": ("sum", PRICE), "buyers": ("count_distinct", "users.id")},
    )

    assert_frames_agree(pushdown, reference)


# --------------------------------------------------------------------------------------
# Derived columns
# --------------------------------------------------------------------------------------


def test_with_column_arithmetic_over_decimals(orders: DataFrame, rows: list[Row]) -> None:
    pushdown = collect(
        orders.select(ID, PRICE, QUANTITY).with_column("line_total", F.col(PRICE) * F.col(QUANTITY))
    )
    reference = project(
        rows,
        {
            ID: ID,
            PRICE: PRICE,
            QUANTITY: QUANTITY,
            "line_total": lambda row: row[PRICE] * row[QUANTITY],
        },
    )

    assert_frames_agree(pushdown, reference)


def test_with_column_arithmetic_propagates_nulls(orders: DataFrame, rows: list[Row]) -> None:
    pushdown = collect(
        orders.select(ID, PRICE, DISCOUNT).with_column("net", F.col(PRICE) - F.col(DISCOUNT))
    )
    reference = project(
        rows,
        {
            ID: ID,
            PRICE: PRICE,
            DISCOUNT: DISCOUNT,
            "net": lambda row: None if row[DISCOUNT] is None else row[PRICE] - row[DISCOUNT],
        },
    )

    assert any(row["net"] is None for row in pushdown.to_pylist()), "NULL - x is NULL"
    assert_frames_agree(pushdown, reference)


def test_a_udf_derives_a_column(orders: DataFrame, rows: list[Row]) -> None:
    def decade(age: int | None) -> str:
        return "unknown" if age is None else f"{age // 10 * 10}s"

    pushdown = collect(
        orders.select(ID, AGE).sort(ID).with_column("decade", F.udf(decade)(AGE)).limit(500)
    )
    reference = project(
        sorted(rows, key=lambda row: row[ID])[:500],
        {ID: ID, AGE: AGE, "decade": lambda row: decade(row[AGE])},
    )

    assert_frames_agree(pushdown, reference, sort=False)


def test_map_pandas_matches_the_same_transformation_in_the_reference(
    orders: DataFrame, rows: list[Row]
) -> None:
    def widen(frame: Any) -> Any:
        result = frame.copy()
        result["doubled"] = result[QUANTITY] * 2
        return result

    pushdown = collect(orders.select(ID, QUANTITY).sort(ID).limit(200).map_pandas(widen))
    reference = project(
        sorted(rows, key=lambda row: row[ID])[:200],
        {ID: ID, QUANTITY: QUANTITY, "doubled": lambda row: row[QUANTITY] * 2},
    )

    assert_frames_agree(pushdown, reference, sort=False)


# --------------------------------------------------------------------------------------
# Sort / limit / offset stacks, and operations above a limit
# --------------------------------------------------------------------------------------


def test_a_sort_limit_offset_stack_returns_the_same_page(
    orders: DataFrame, rows: list[Row]
) -> None:
    """Ordered by a unique non-null key, so "the same page" is a question with one answer."""
    pushdown = collect(orders.select(ID, STATE).sort(ID).limit(25).offset(10))
    reference = project(sorted(rows, key=lambda row: row[ID])[10:35], {ID: ID, STATE: STATE})

    assert_frames_agree(pushdown, reference, sort=False)


def test_a_local_sort_puts_nulls_last(orders: DataFrame, rows: list[Row]) -> None:
    """The sort runs here because a derived column cannot be sorted on the wire."""
    pushdown = collect(
        orders.select(ID, AGE)
        .with_column("doubled", F.col(AGE) * 2)
        .sort(F.col("doubled").desc(), F.col(ID))
        .limit(40)
    )
    ordered = sorted(
        rows,
        key=lambda row: (row[AGE] is None, -(row[AGE] or 0), row[ID]),
    )[:40]
    reference = project(
        ordered,
        {ID: ID, AGE: AGE, "doubled": lambda row: None if row[AGE] is None else row[AGE] * 2},
    )

    assert_frames_agree(pushdown, reference, sort=False)


def test_operations_above_a_limit_run_on_that_page(orders: DataFrame, rows: list[Row]) -> None:
    """The limit rides remote and the filter sees only its result — M1's reading, now executed."""
    pushdown = collect(
        orders.select(ID, STATE).sort(ID).limit(500).filter(F.col(STATE) == "California")
    )
    page = sorted(rows, key=lambda row: row[ID])[:500]
    reference = project(
        keep(page, lambda row: check(row[STATE], lambda v: v == "California")),
        {ID: ID, STATE: STATE},
    )

    assert 0 < pushdown.num_rows < 500
    assert_frames_agree(pushdown, reference, sort=False)


def test_an_aggregate_above_a_limit_aggregates_only_that_page(
    orders: DataFrame, rows: list[Row]
) -> None:
    pushdown = collect(
        orders.select(ID, STATE, PRICE)
        .sort(ID)
        .limit(300)
        .group_by(STATE)
        .agg(F.sum(PRICE).alias("total"), F.count(ID).alias("n"))
    )
    reference = aggregate(
        sorted(rows, key=lambda row: row[ID])[:300],
        [STATE],
        {"total": ("sum", PRICE), "n": ("count", ID)},
    )

    assert_frames_agree(pushdown, reference)


# --------------------------------------------------------------------------------------
# Joins and unions (M4) — SQL semantics, spelled out on both sides
# --------------------------------------------------------------------------------------

USER_ID = "order_items.user_id"
BUYER_ID = "users.id"


def revenue_by_state(orders: DataFrame) -> DataFrame:
    return orders.group_by(STATE).agg(F.sum(PRICE).alias("revenue"))


def orders_by_state(orders: DataFrame) -> DataFrame:
    return orders.group_by(STATE).agg(F.count(ID).alias("n"))


def revenue_reference(rows: list[Row]) -> Any:
    return aggregate(rows, [STATE], {"revenue": ("sum", PRICE)})


def orders_reference(rows: list[Row]) -> Any:
    return aggregate(rows, [STATE], {"n": ("count", ID)})


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_a_join_after_aggregation_agrees_with_the_reference(
    orders: DataFrame, rows: list[Row], how: str
) -> None:
    """``users.state`` is NULL for 30 users and for every orphan, so the NULL key is real."""
    pushdown = collect(revenue_by_state(orders).join(orders_by_state(orders), STATE, how))
    reference = sql_join(
        revenue_reference(rows).to_pylist(),
        orders_reference(rows).to_pylist(),
        on=[STATE],
        how=how,
        left_columns=[STATE, "revenue"],
        right_columns=[STATE, "n"],
    )

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_a_join_over_raw_rows_and_orphan_keys_agrees_with_the_reference(
    orders: DataFrame, rows: list[Row], how: str
) -> None:
    """The fact table's ~1 % orphan ``user_id``s are what makes the outer arms observable.

    The left side is the raw fact column (never NULL, sometimes orphaned); the right side is one
    row per buyer, whose key is NULL for exactly those orphans.  So both kinds of "no match"
    happen at once: a key with no counterpart, and a key that is not a value at all.
    """
    left = orders.select(F.col(ID).alias("item"), F.col(USER_ID).alias("uid"))
    right = orders.group_by(F.col(BUYER_ID).alias("uid")).agg(F.count(ID).alias("orders"))
    buyers = aggregate(rows, [BUYER_ID], {"orders": ("count", ID)}).to_pylist()

    pushdown = collect(left.join(right, "uid", how))
    reference = sql_join(
        [{"item": row[ID], "uid": row[USER_ID]} for row in rows],
        [{"uid": row[BUYER_ID], "orders": row["orders"]} for row in buyers],
        on=["uid"],
        how=how,
        left_columns=["item", "uid"],
        right_columns=["uid", "orders"],
    )

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


def test_a_union_agrees_with_stacking_the_two_references(
    orders: DataFrame, rows: list[Row]
) -> None:
    left = orders.select(ID, AGE).filter(F.col(STATE) == "California")
    right = orders.select(ID, AGE).filter(F.col(STATE) == "Texas")

    pushdown = collect(left.union(right))
    reference = stack(
        project(
            keep(rows, lambda row: check(row[STATE], lambda v: v == "California")),
            {ID: ID, AGE: AGE},
        ),
        project(
            keep(rows, lambda row: check(row[STATE], lambda v: v == "Texas")), {ID: ID, AGE: AGE}
        ),
    )

    assert pushdown.num_rows > 0
    assert_frames_agree(pushdown, reference)


def test_a_union_that_promotes_an_integer_to_a_float_agrees_by_value(
    orders: DataFrame, rows: list[Row]
) -> None:
    """docs/HYBRID.md §3.3: ``int64`` next to ``float64`` widens, and the values must survive."""
    averages = orders.group_by(CATEGORY).agg(F.avg(PRICE).alias("value"))
    counts = orders.group_by(CATEGORY).agg(F.count(ID).alias("value"))

    pushdown = collect(averages.union(counts))
    reference = stack(
        aggregate(rows, [CATEGORY], {"value": ("avg", PRICE)}),
        table_of(
            [
                {CATEGORY: row[CATEGORY], "value": float(row["value"])}
                for row in aggregate(rows, [CATEGORY], {"value": ("count", ID)}).to_pylist()
            ],
            (CATEGORY, "value"),
        ),
    )

    assert pushdown.schema.field("value").type == pa.float64()
    assert_frames_agree(pushdown, reference)
