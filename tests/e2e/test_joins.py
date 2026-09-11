"""M4 end to end: joins and unions against FakeOmniAPI (docs/HYBRID.md, DESIGN §2).

The numbers here come from ``tests/data/bench/known_answers.json`` and from the dataset's own
documented shape (``order_items.user_id`` carries ~1 % orphans, ``users.state`` carries NULLs),
never from a recomputation inside the test — so every assertion also holds against the live org.

What these cases really pin is the **firewall**: a user join has SQL semantics, so a NULL key
matches nothing, while the ``AlignJoin`` a decomposed aggregate uses internally pairs the two
NULL groups.  The bench data guarantees both kinds of NULL exist, which is why the row counts
below are different for every join kind.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.dataframe import DataFrame
from omniframes.errors import CompileError, TruncationWarning
from omniframes.transport import HttpTransport
from tests.fakes import BENCH_MODEL_NAME, BENCH_TOPIC_NAME, DEFAULT_TOKEN, FakeOmniAPI

BASE_URL = "https://bench.omniapp.co"
BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "bench"

REVENUE = "order_items.total_sale_price"
ORDER_COUNT = "order_items.count"

#: docs/BENCH_DATASET.md: the fact table, and the ~1 % of it whose ``user_id`` has no user row.
FACT_ROWS = 10_000
ORPHAN_ROWS = 113
MATCHED_ROWS = FACT_ROWS - ORPHAN_ROWS


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


def collect(frame: DataFrame) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TruncationWarning)
        return frame.collect()


def revenue_by_state(orders: DataFrame) -> DataFrame:
    return orders.group_by("users.state").agg(F.measure(REVENUE).alias("revenue"))


def buyers_by_state(orders: DataFrame) -> DataFrame:
    return orders.group_by("users.state").agg(F.measure("users.count").alias("buyers"))


# --------------------------------------------------------------------------------------
# Joining two aggregates on a key that carries NULLs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("how", "rows"),
    [
        # 20 real states on both sides, plus one NULL-state group on each that never matches.
        ("inner", 20),
        ("left", 21),
        ("right", 21),
        ("outer", 22),
    ],
)
def test_null_keys_never_match_in_a_user_join(orders: DataFrame, how: str, rows: int) -> None:
    joined = collect(revenue_by_state(orders).join(buyers_by_state(orders), "users.state", how))

    assert joined.column_names == ["users.state", "revenue", "buyers"]
    assert joined.num_rows == rows
    null_rows = [row for row in joined.to_pylist() if row["users.state"] is None]
    assert len(null_rows) == (0 if how == "inner" else 2 if how == "outer" else 1)


def test_an_outer_join_keeps_each_side_s_null_group_separately(orders: DataFrame) -> None:
    """An align-join would have merged these two into one row; a SQL join must not."""
    joined = collect(revenue_by_state(orders).join(buyers_by_state(orders), "users.state", "outer"))
    null_rows = [row for row in joined.to_pylist() if row["users.state"] is None]

    assert sorted((row["revenue"] is None, row["buyers"] is None) for row in null_rows) == [
        (False, True),
        (True, False),
    ]


def test_the_joined_values_still_match_the_known_answers(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    expected = {
        row["state"]: row for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
    }
    joined = collect(revenue_by_state(orders).join(buyers_by_state(orders), "users.state", "left"))

    for row in joined.to_pylist():
        answer = expected[row["users.state"]]
        assert str(row["revenue"]) == answer["total_sale_price"]
        if row["users.state"] is not None:
            assert row["buyers"] == answer["distinct_buyers"]


def test_each_side_of_a_join_is_its_own_query(orders: DataFrame, handler: FakeOmniAPI) -> None:
    collect(revenue_by_state(orders).join(buyers_by_state(orders), "users.state"))
    queries = [
        request.query
        for request in handler.requests
        if request.path.endswith("/query/run") and request.query is not None
    ]

    assert len(queries) == 2
    assert [query["fields"] for query in queries] == [
        ["users.state", REVENUE],
        ["users.state", "users.count"],
    ]
    assert all(query["filters"] == {} for query in queries), "neither side constrains the other"


# --------------------------------------------------------------------------------------
# The orphan spot check: raw fact rows joined to the users behind them
# --------------------------------------------------------------------------------------


def items(orders: DataFrame) -> DataFrame:
    """One row per order item, keyed by the raw fact column (never NULL, ~1 % orphaned)."""
    return orders.select(
        F.col("order_items.id").alias("item"),
        F.col("order_items.user_id").alias("user_id"),
    )


def users(orders: DataFrame) -> DataFrame:
    """One row per buyer, keyed by ``users.id`` — NULL for the orphans' group."""
    return orders.group_by(F.col("users.id").alias("user_id")).agg(
        F.measure(ORDER_COUNT).alias("orders")
    )


@pytest.mark.parametrize(
    ("how", "rows"),
    [
        ("inner", MATCHED_ROWS),
        ("left", FACT_ROWS),
        # every matched fact row, plus the one unmatched NULL-keyed group of orphans
        ("right", MATCHED_ROWS + 1),
        ("outer", FACT_ROWS + 1),
    ],
)
def test_orphan_fact_rows_drop_out_of_an_inner_join(orders: DataFrame, how: str, rows: int) -> None:
    joined = collect(items(orders).join(users(orders), "user_id", how))

    assert joined.num_rows == rows


def test_a_join_key_may_be_named_by_either_side_s_alias(orders: DataFrame) -> None:
    """``user_id`` is ``order_items.user_id`` on the left and ``users.id`` on the right."""
    joined = collect(items(orders).join(users(orders), "user_id", "inner"))

    assert joined.column_names == ["user_id", "item", "orders"]
    assert all(row["user_id"] is not None for row in joined.to_pylist())


def test_the_left_join_marks_the_orphans_rather_than_dropping_them(orders: DataFrame) -> None:
    joined = collect(items(orders).join(users(orders), "user_id", "left"))
    unmatched = [row for row in joined.to_pylist() if row["orders"] is None]

    assert len(unmatched) == ORPHAN_ROWS
    assert all(row["user_id"] is not None for row in unmatched), "orphan ids exist, users do not"


# --------------------------------------------------------------------------------------
# Unions
# --------------------------------------------------------------------------------------


def test_a_union_stacks_two_aggregates_and_promotes_the_shared_column(
    orders: DataFrame,
) -> None:
    """``avg`` is float, ``count`` is int; the union widens to the type that holds both (§3.3)."""
    averages = orders.group_by("products.category").agg(
        F.avg("order_items.sale_price").alias("value")
    )
    counts = orders.group_by("products.category").agg(F.count("order_items.id").alias("value"))
    stacked = collect(averages.union(counts))

    assert stacked.column_names == ["products.category", "value"]
    assert stacked.schema.field("value").type == pa.float64()
    assert stacked.num_rows == collect(averages).num_rows + collect(counts).num_rows


def test_union_all_is_the_same_operation(orders: DataFrame) -> None:
    left = orders.select("users.state").limit(3)
    right = orders.select("users.state").limit(2)

    assert collect(left.union(right)).num_rows == 5
    assert collect(left.unionAll(right)).num_rows == 5


def test_a_union_whose_column_names_disagree_is_refused_before_anything_runs(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    mismatched = orders.select("users.state", "users.age").union(
        orders.select("users.state", "users.id")
    )

    with pytest.raises(CompileError, match="same columns in the same order"):
        mismatched.collect()
    assert not [r for r in handler.requests if r.path.endswith("/query/run")]


def test_an_alias_makes_a_mismatched_union_line_up(orders: DataFrame) -> None:
    left = orders.select(F.col("users.age").alias("n"))
    right = orders.select(F.col("order_items.quantity").alias("n"))

    assert collect(left.limit(4).union(right.limit(6))).num_rows == 10


# --------------------------------------------------------------------------------------
# explain()
# --------------------------------------------------------------------------------------


def test_explain_shows_both_sub_plans_and_the_join_between_them(orders: DataFrame) -> None:
    text = revenue_by_state(orders).join(buyers_by_state(orders), "users.state", "left").explain()

    assert "Remote step 1 [tier 1" in text
    assert "Remote step 2 [tier 1" in text
    assert "join [left]: step 1 ⨝ step 2 on [users.state]" in text


def test_columns_are_known_without_running_the_join(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    joined = revenue_by_state(orders).join(buyers_by_state(orders), "users.state")

    assert joined.columns == ("users.state", "revenue", "buyers")
    assert not [r for r in handler.requests if r.path.endswith("/query/run")]
