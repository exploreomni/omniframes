"""M2 end to end: grains, group_by/agg, measure filters and totals against FakeOmniAPI.

Numeric expectations come from ``tests/data/bench/known_answers.json`` — never from a
recomputation inside the test — so every assertion here also holds against the live org.  The
answers are computed with the same windows the queries use, which is why the trailing-12-month
case can be written with ``.between(date(2025, 7, 1), date(2026, 7, 1))``: date ranges are
half-open, and ``[2025-07-01, 2026-07-01)`` is exactly the window the answer file encodes.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.dataframe import ROW_TYPE_COLUMN, DataFrame
from omniframes.errors import TruncationWarning
from omniframes.transport import HttpTransport
from tests.fakes import (
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

BASE_URL = "https://bench.example.omni.co"
BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "bench"

REVENUE = "order_items.total_sale_price"
ORDER_COUNT = "order_items.count"
BUYERS = "users.count"
AVERAGE_SALE_PRICE = "order_items.average_sale_price"
MONTH = "order_items.created_at[month]"

#: The window ``monthly_revenue_trailing_12m`` is computed over — half-open, like every date
#: range omniframes sends (see :meth:`omniframes.column.Column.between`).
WINDOW_START = date(2025, 7, 1)
WINDOW_END = date(2026, 7, 1)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


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
        yield OmniSession.builder.base_url(BASE_URL).transport(transport).get_or_create()


@pytest.fixture
def orders(session: OmniSession) -> DataFrame:
    return session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)


@pytest.fixture(scope="module")
def known_answers() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((BENCH_DIR / "known_answers.json").read_text("utf-8"))
    answers: dict[str, Any] = payload["answers"]
    return answers


def answer_rows(answers: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    entry = answers[key]
    return [dict(zip(entry["columns"], row, strict=True)) for row in entry["rows"]]


def utc(literal: str) -> datetime:
    """A known-answer timestamp as the tz-aware datetime Arrow decodes to."""
    return datetime.fromisoformat(literal)


def run_queries(handler: FakeOmniAPI) -> list[Mapping[str, Any]]:
    queries: list[Mapping[str, Any]] = []
    for request in handler.requests:
        if request.path.endswith("/query/run") and request.query is not None:
            queries.append(request.query)
    return queries


# --------------------------------------------------------------------------------------
# Grains
# --------------------------------------------------------------------------------------


def test_monthly_revenue_over_a_trailing_window(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """A grained group key, a bare-field date filter, two measures — the canonical M2 query."""
    expected = answer_rows(known_answers, "monthly_revenue_trailing_12m")
    month = F.col("order_items.created_at").grain("month")
    frame = (
        orders.group_by(month)
        .agg(F.measure(REVENUE), F.measure(ORDER_COUNT))
        .filter(F.col("order_items.created_at").between(WINDOW_START, WINDOW_END))
        .sort(month)
        .to_pandas()
    )

    assert len(frame) == len(expected) == 12
    for (_, row), answer in zip(frame.iterrows(), expected, strict=True):
        assert row[MONTH] == utc(answer["month"])
        assert str(row[REVENUE]) == answer["total_sale_price"]
        assert row[ORDER_COUNT] == answer["order_items_count"]


def test_the_grain_filter_rule_sends_a_date_filter_on_the_bare_field(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    """Direction 1: a timestamp grain is projected bracketed and filtered BARE."""
    month = F.col("order_items.created_at").grain("month")
    orders.group_by(month).agg(F.measure(ORDER_COUNT)).filter(
        month.between(WINDOW_START, WINDOW_END)
    ).collect()
    query = run_queries(handler)[-1]

    assert query["fields"] == [MONTH, ORDER_COUNT]
    assert list(query["filters"]) == ["order_items.created_at"], "no bracket on the filter key"
    assert query["filters"]["order_items.created_at"]["type"] == "date"


def test_the_grain_filter_rule_sends_a_number_filter_on_the_bracketed_field(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    """Direction 2: a numeric grain is projected AND filtered on the bracketed name.

    Both directions in one query, so the single surviving row must be the July 2025 row of
    ``monthly_revenue_trailing_12m``.
    """
    july = next(
        row
        for row in answer_rows(known_answers, "monthly_revenue_trailing_12m")
        if row["month"].startswith("2025-07")
    )
    month_num = F.col("order_items.created_at").grain("month_num")
    frame = (
        orders.group_by(month_num)
        .agg(F.measure(REVENUE), F.measure(ORDER_COUNT))
        .filter(F.col("order_items.created_at").between(WINDOW_START, WINDOW_END))
        .filter(month_num == 7)
        .to_pandas()
    )
    query = run_queries(handler)[-1]

    assert set(query["filters"]) == {
        "order_items.created_at",
        "order_items.created_at[month_num]",
    }
    assert query["filters"]["order_items.created_at[month_num]"]["type"] == "number"

    assert len(frame) == 1
    assert frame[REVENUE].iloc[0] == Decimal(july["total_sale_price"])
    assert frame[ORDER_COUNT].iloc[0] == july["order_items_count"]


def test_a_grain_can_be_aliased_and_sorted_by_its_alias(
    orders: DataFrame, known_answers: dict[str, Any], handler: FakeOmniAPI
) -> None:
    expected = sorted(
        answer_rows(known_answers, "monthly_revenue_trailing_12m"),
        key=lambda row: row["month"],
        reverse=True,
    )
    month = F.col("order_items.created_at").grain("month").alias("month")
    frame = (
        orders.group_by(month)
        .agg(F.measure(REVENUE).alias("revenue"))
        .filter(F.col("order_items.created_at").between(WINDOW_START, WINDOW_END))
        .sort(F.col("month").desc())
        .to_pandas()
    )
    query = run_queries(handler)[-1]

    assert list(frame.columns) == ["month", "revenue"], "aliases are applied after the result"
    assert query["fields"] == [MONTH, REVENUE], "an alias never reaches the wire"
    assert [sort["column_name"] for sort in query["sorts"]] == [MONTH]
    assert [str(value) for value in frame["revenue"]] == [
        row["total_sale_price"] for row in expected
    ]


# --------------------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------------------


def test_revenue_and_buyers_by_state_mixes_measures_from_two_views(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """``users.count`` is a governed measure like any other — the join comes from the topic."""
    expected = {
        row["state"]: row for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
    }
    table = orders.group_by("users.state").agg(F.measure(REVENUE), F.measure(BUYERS)).collect()
    rows = table.to_pylist()

    assert len(rows) == len(expected) == 21
    for row in rows:
        answer = expected[row["users.state"]]
        assert str(row[REVENUE]) == answer["total_sale_price"]
        assert row[BUYERS] == answer["distinct_buyers"]
    assert None in {row["users.state"] for row in rows}, "the NULL group key survives"


def test_average_sale_price_by_category_stays_exact(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """An average must not degrade to a float on the way through Arrow."""
    expected = {
        row["category"]: row for row in answer_rows(known_answers, "average_sale_price_by_category")
    }
    table = (
        orders.group_by("products.category")
        .agg(F.measure(AVERAGE_SALE_PRICE), F.measure(ORDER_COUNT))
        .collect()
    )

    assert table.num_rows == len(expected) == 11
    for row in table.to_pylist():
        answer = expected[row["products.category"]]
        value = row[AVERAGE_SALE_PRICE]
        assert isinstance(value, Decimal)
        assert str(value) == answer["average_sale_price"]
        assert row[ORDER_COUNT] == answer["order_items_count"]


def test_group_by_agg_and_select_return_the_same_rows(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    grouped = orders.group_by("users.state").agg(F.measure(REVENUE)).collect()
    selected = orders.select("users.state", F.measure(REVENUE)).collect()
    sent = run_queries(handler)

    assert grouped.to_pylist() == selected.to_pylist()
    assert sent[-1] == sent[-2], "the two spellings put the same query on the wire"


# --------------------------------------------------------------------------------------
# Measure filters → HAVING
# --------------------------------------------------------------------------------------


def surviving_states(answers: Mapping[str, Any], keep: Any) -> dict[str | None, dict[str, Any]]:
    kept = {
        row["state"]: row
        for row in answer_rows(answers, "revenue_by_state")
        if keep(Decimal(row["total_sale_price"]))
    }
    assert 0 < len(kept) < 21, "a HAVING fixture that keeps everything or nothing proves nothing"
    return kept


def test_a_measure_filter_keeps_only_the_groups_that_pass(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    expected = surviving_states(known_answers, lambda revenue: revenue > Decimal(50000))
    table = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE))
        .filter(F.measure(REVENUE) > 50000)
        .collect()
    )

    assert table.num_rows == len(expected)
    for row in table.to_pylist():
        assert str(row[REVENUE]) == expected[row["users.state"]]["total_sale_price"]
    assert None in expected, "the NULL group is filtered on its aggregate like any other"


def test_two_conditions_on_one_measure_narrow_the_band(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    expected = surviving_states(
        known_answers, lambda revenue: Decimal(20000) < revenue < Decimal(80000)
    )
    revenue = F.measure(REVENUE)
    table = (
        orders.group_by("users.state")
        .agg(revenue)
        .filter(revenue > 20000)
        .filter(revenue < 80000)
        .collect()
    )

    assert {row["users.state"] for row in table.to_pylist()} == set(expected)


def test_between_and_negation_work_on_the_having_side(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """``between`` is inclusive over numbers here too, and ``~`` flips the whole entry."""
    revenue = F.measure(REVENUE)
    band = surviving_states(known_answers, lambda value: Decimal(20000) <= value <= Decimal(50000))
    inverted = surviving_states(known_answers, lambda value: not value > Decimal(50000))
    between = (
        orders.group_by("users.state").agg(revenue).filter(revenue.between(20000, 50000)).collect()
    )
    negated = orders.group_by("users.state").agg(revenue).filter(~(revenue > 50000)).collect()

    assert {row["users.state"] for row in between.to_pylist()} == set(band)
    assert {row["users.state"] for row in negated.to_pylist()} == set(inverted)


def test_a_filtered_measure_does_not_have_to_be_selected(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """The server force-adds it to the aggregate for the HAVING, then projects it away."""
    expected = surviving_states(known_answers, lambda revenue: revenue > Decimal(50000))
    table = (
        orders.group_by("users.state")
        .agg(F.measure(ORDER_COUNT))
        .filter(F.measure(REVENUE) > 50000)
        .collect()
    )

    assert table.column_names == ["users.state", ORDER_COUNT]
    assert table.num_rows == len(expected)
    for row in table.to_pylist():
        assert row[ORDER_COUNT] == expected[row["users.state"]]["order_items_count"]


def test_a_measure_filter_alone_forces_the_group_by(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """No measure in ``fields``: the filter is what makes this an aggregate at all."""
    expected = surviving_states(known_answers, lambda revenue: revenue > Decimal(50000))
    table = orders.select("users.state").filter(F.measure(REVENUE) > 50000).collect()

    assert table.column_names == ["users.state"]
    assert table.num_rows == len(expected), "one row per surviving group, not per fact row"
    assert {row["users.state"] for row in table.to_pylist()} == set(expected)


def test_where_narrows_rows_and_having_narrows_groups(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    threshold = Decimal("45000")
    expected = [
        row
        for row in answer_rows(known_answers, "monthly_revenue_trailing_12m")
        if Decimal(row["total_sale_price"]) > threshold
    ]
    assert 0 < len(expected) < 12
    month = F.col("order_items.created_at").grain("month")
    table = (
        orders.group_by(month)
        .agg(F.measure(REVENUE))
        .filter(F.col("order_items.created_at").between(WINDOW_START, WINDOW_END))
        .filter(F.measure(REVENUE) > threshold)
        .sort(month)
        .collect()
    )

    assert table.num_rows == len(expected)
    for row, answer in zip(table.to_pylist(), expected, strict=True):
        assert row[MONTH] == utc(answer["month"])
        assert str(row[REVENUE]) == answer["total_sale_price"]


def test_a_measure_filter_on_a_joined_view_still_joins_it(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    expected = {
        row["state"]: row
        for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
        if row["distinct_buyers"] > 25
    }
    assert 0 < len(expected) < 21
    table = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE))
        .filter(F.measure(BUYERS) > 25)
        .collect()
    )

    assert {row["users.state"] for row in table.to_pylist()} == set(expected)


# --------------------------------------------------------------------------------------
# with_totals()
# --------------------------------------------------------------------------------------


def test_with_totals_appends_the_grand_total_row(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    (grand,) = answer_rows(known_answers, "grand_totals")
    table = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE), F.measure(ORDER_COUNT))
        .with_totals()
        .collect()
    )
    rows = table.to_pylist()

    assert table.column_names == ["users.state", REVENUE, ORDER_COUNT, ROW_TYPE_COLUMN]
    assert [row[ROW_TYPE_COLUMN] for row in rows] == ["data"] * 21 + ["total"]

    total = rows[-1]
    assert total["users.state"] is None, "dimension columns are NULL on a totals row"
    assert str(total[REVENUE]) == grand["total_sale_price"]
    assert total[ORDER_COUNT] == grand["order_items_count"]


def test_the_totals_row_re_aggregates_rather_than_summing_the_page(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """Totals are computed post-filter but PRE-limit, which is why they are worth asking for."""
    (grand,) = answer_rows(known_answers, "grand_totals")
    frame = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE), F.measure(ORDER_COUNT))
        .sort(F.col(REVENUE).desc())
        .limit(3)
        .with_totals()
    )
    with pytest.warns(TruncationWarning):  # three of 21 groups: the limit is doing its job
        rows = frame.collect().to_pylist()
    data = [row for row in rows if row[ROW_TYPE_COLUMN] == "data"]

    assert len(rows) == 4
    assert len(data) == 3
    assert rows[-1][ORDER_COUNT] == grand["order_items_count"]
    assert sum(row[ORDER_COUNT] for row in data) < grand["order_items_count"]


def test_totals_respect_the_filters(orders: DataFrame, known_answers: dict[str, Any]) -> None:
    expected = sum(
        row["order_items_count"]
        for row in answer_rows(known_answers, "monthly_revenue_trailing_12m")
    )
    month = F.col("order_items.created_at").grain("month")
    table = (
        orders.group_by(month)
        .agg(F.measure(ORDER_COUNT))
        .filter(F.col("order_items.created_at").between(WINDOW_START, WINDOW_END))
        .with_totals()
        .collect()
    )
    total = table.to_pylist()[-1]

    assert total[ROW_TYPE_COLUMN] == "total"
    assert total[ORDER_COUNT] == expected


def test_with_totals_renders_in_show(orders: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    orders.group_by("users.state").agg(F.measure(ORDER_COUNT)).with_totals().show(3)
    printed = capsys.readouterr().out

    assert ROW_TYPE_COLUMN in printed
    assert "| data" in printed


def test_aliases_apply_to_the_totals_row_too(orders: DataFrame) -> None:
    table = (
        orders.group_by(F.col("users.state").alias("state"))
        .agg(F.measure(REVENUE).alias("revenue"))
        .with_totals()
        .collect()
    )

    assert table.column_names == ["state", "revenue", ROW_TYPE_COLUMN]
    assert table.to_pylist()[-1][ROW_TYPE_COLUMN] == "total"


# --------------------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------------------


def test_an_aggregate_that_fills_its_limit_warns(orders: DataFrame) -> None:
    with pytest.warns(TruncationWarning, match="applied limit"):
        orders.group_by("users.state").agg(F.measure(REVENUE)).limit(5).collect()


def test_an_aggregate_that_fits_does_not_warn(
    orders: DataFrame, recwarn: pytest.WarningsRecorder
) -> None:
    orders.group_by("users.state").agg(F.measure(REVENUE)).collect()

    assert [w for w in recwarn if issubclass(w.category, TruncationWarning)] == []


def test_the_totals_row_does_not_count_towards_the_limit(orders: DataFrame) -> None:
    """21 groups + a totals row is 22 rows for a limit of 21 — but only 21 of them are data."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        table = (
            orders.group_by("users.state").agg(F.measure(REVENUE)).with_totals().limit(22).collect()
        )

    assert table.num_rows == 22

    with pytest.warns(TruncationWarning):
        orders.group_by("users.state").agg(F.measure(REVENUE)).with_totals().limit(21).collect()
