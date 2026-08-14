"""M3 end to end: the hybrid engine against FakeOmniAPI (docs/HYBRID.md).

Numeric expectations come from ``tests/data/bench/known_answers.json`` — never from a
recomputation inside the test — so every assertion here also holds against the live org.  That
matters more here than anywhere else: a decomposed aggregate is computed in two places at once,
and the only way to know the halves agree is to check both against an answer neither of them
produced.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.compile.splitter import SplitOptions, split
from omniframes.dataframe import DataFrame
from omniframes.errors import CompileError, TruncationWarning
from omniframes.transport import HttpTransport
from omniframes.types import OmniDataType, OmniField, OmniSchema
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


def run_queries(handler: FakeOmniAPI) -> list[Mapping[str, Any]]:
    return [
        request.query
        for request in handler.requests
        if request.path.endswith("/query/run") and request.query is not None
    ]


# --------------------------------------------------------------------------------------
# The canonical mixed aggregation (docs/HYBRID.md §2.1)
# --------------------------------------------------------------------------------------


def test_revenue_and_buyers_by_state_decomposes_and_still_matches_the_answer(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    """A governed measure and an ad-hoc aggregation, computed in two places, joined on the key.

    The NULL-state group is the interesting one: it is a real group on both halves, and the
    align-join has to pair the two NULL rows with each other.
    """
    expected = {
        row["state"]: row for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
    }
    table = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers"))
        .collect()
    )
    rows = table.to_pylist()

    assert table.column_names == ["users.state", "revenue", "buyers"]
    assert len(rows) == len(expected) == 21
    for row in rows:
        answer = expected[row["users.state"]]
        assert str(row["revenue"]) == answer["total_sale_price"]
        assert row["buyers"] == answer["distinct_buyers"]

    null_group = next(row for row in rows if row["users.state"] is None)
    assert null_group["buyers"] == expected[None]["distinct_buyers"] > 0
    assert sum(row["buyers"] for row in rows) == 497, "every buyer lands in exactly one group"


def test_the_two_halves_are_a_governed_query_and_a_sql_job_over_an_unlimited_reference(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    """M5: the ad-hoc half is a warehouse GROUP BY, and its reference core is the unlimited one.

    The reference is what M3's raw scan used to be — same fields, same ``limit: null``, for the
    same reason (a silently paged input to an aggregate is a wrong answer).  The difference is
    that the rows never leave the warehouse.
    """
    orders.group_by("users.state").agg(F.measure(REVENUE), F.count_distinct("users.id")).collect()
    measures, sql = run_queries(handler)

    assert measures["fields"] == ["users.state", REVENUE]
    assert measures["limit"] == 50_000
    assert measures.get("userEditedSQL", "") == "", "the governed half stays a tier-1 query"

    assert sql["rewriteSql"] is False, "without it the server silently ignores the SQL"
    assert sql["sqlSortsEnabled"] is False
    assert sql["limit"] == 50_000, "the wire limit is always explicit, SQL job or not"
    reference = sql["staticQueryReferences"]["ref_1"]
    assert reference["fields"] == ["users.state", "users.id"]
    assert reference["limit"] is None, "a capped reference would silently produce a wrong count"
    assert reference["model_id"] == reference["modelId"], "§3.5's extra snake_case key"
    assert 'COUNT(DISTINCT "users.id")' in sql["userEditedSQL"]


def test_sorting_and_limiting_a_decomposed_aggregate_happens_locally(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    expected = sorted(
        answer_rows(known_answers, "revenue_and_buyers_by_state"),
        key=lambda row: row["distinct_buyers"],
        reverse=True,
    )[:3]
    rows = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers"))
        .sort(F.col("buyers").desc())
        .limit(3)
        .collect()
        .to_pylist()
    )

    assert [row["buyers"] for row in rows] == [row["distinct_buyers"] for row in expected]
    assert all(query["sorts"] == [] for query in run_queries(handler)), "the sort ran here"


def test_a_local_having_on_the_ad_hoc_aggregate(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """Tier 2 will push this into SQL (M5); M3 filters the computed column locally."""
    expected = {
        row["state"]
        for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
        if row["distinct_buyers"] > 25
    }
    assert 0 < len(expected) < 21
    rows = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE), F.count_distinct("users.id").alias("buyers"))
        .filter(F.col("buyers") > 25)
        .collect()
        .to_pylist()
    )

    assert {row["users.state"] for row in rows} == expected


def test_an_ad_hoc_only_aggregate_needs_one_query_and_no_join(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    expected = {
        row["status"]: row["order_items_count"]
        for row in answer_rows(known_answers, "count_by_status")
    }
    rows = (
        orders.group_by("order_items.status")
        .agg(F.count("order_items.id").alias("n"))
        .collect()
        .to_pylist()
    )

    assert {row["order_items.status"]: row["n"] for row in rows} == expected
    assert len(run_queries(handler)) == 1


def test_select_and_group_by_agg_agree_on_the_split_too(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    """The M1-era asymmetry is gone: both spellings decompose identically."""
    grouped = orders.group_by("users.state").agg(F.count_distinct("users.id")).collect()
    selected = orders.select("users.state", F.count_distinct("users.id")).collect()
    sent = run_queries(handler)

    assert grouped.to_pylist() == selected.to_pylist()
    assert sent[-1] == sent[-2], "the two spellings put the same query on the wire"


def test_with_totals_over_a_decomposed_aggregate_is_refused(orders: DataFrame) -> None:
    frame = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE), F.count_distinct("users.id"))
        .with_totals()
    )

    with pytest.raises(CompileError, match="with_totals"):
        frame.collect()


# --------------------------------------------------------------------------------------
# The decomposition row cap (§2.1 / §2.4)
# --------------------------------------------------------------------------------------


def capped_session(handler: FakeOmniAPI, rows: int) -> Iterator[OmniSession]:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    with client:
        transport = HttpTransport(
            base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
        )
        yield (
            OmniSession.builder.base_url(BASE_URL)
            .transport(transport)
            .decomposition_row_cap(rows)
            .get_or_create()
        )


def decomposed(frame: DataFrame, rows: int) -> DataFrame:
    """Compile ``frame`` the way M3 did: tier 2 off, so the ad-hoc half is a capped raw scan.

    Since M5 an ad-hoc aggregate over a governed topic rides tier 2, which pulls no raw rows at
    all and therefore has nothing for ``decomposition_row_cap`` to cap.  The cap still governs
    every plan tier 2 cannot express, and ``disable_sql`` is how the test lane reaches that path
    without a public knob nobody should want (docs/SQLTIER.md §5).
    """
    frame._execution = split(
        frame.logical_plan,
        options=SplitOptions(decomposition_row_cap=rows, disable_sql=True),
    )
    return frame


def test_the_row_cap_caps_the_scan_and_warns_loudly_when_it_bites(handler: FakeOmniAPI) -> None:
    """An opt-in cap must never let a partial answer pass for a complete one."""
    for session in capped_session(handler, 100):
        frame = decomposed(
            session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)
            .group_by("users.state")
            .agg(F.count_distinct("users.id")),
            100,
        )
        with pytest.warns(TruncationWarning, match="raw scan.*decomposition_row_cap"):
            frame.collect()

    assert run_queries(handler)[-1]["limit"] == 100


def test_a_cap_that_does_not_bite_is_silent(handler: FakeOmniAPI) -> None:
    for session in capped_session(handler, 50_000):
        frame = decomposed(
            session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)
            .group_by("users.state")
            .agg(F.count_distinct("users.id")),
            50_000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", TruncationWarning)
            assert frame.collect().num_rows == 21


def test_show_still_warns_about_an_intermediate_scan(handler: FakeOmniAPI) -> None:
    """``show()`` suppresses the warning for its own limit, never for a scan nobody asked for.

    Spelled as the two calls ``show()`` makes, because the frame it previews is built inside
    ``show()`` and the cap has to be compiled into that one.
    """
    for session in capped_session(handler, 100):
        frame = decomposed(
            session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)
            .group_by("users.state")
            .agg(F.count_distinct("users.id"))
            .limit(4),
            100,
        )
        with pytest.warns(TruncationWarning, match="raw scan"):
            frame._collect(warn=False)


def test_the_builder_knob_reaches_the_splitter_without_help(handler: FakeOmniAPI) -> None:
    """No ``_execution`` preset here: the cap has to travel builder -> session -> split().

    Every other cap test presets ``frame._execution``, which short-circuits the one place
    ``session.decomposition_row_cap`` is read — so all of them passed with the knob disconnected.
    """
    for session in capped_session(handler, 100):
        frame = (
            session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)
            .select("users.state", "users.id", "order_items.status")
            .filter(F.udf(lambda status: status == "complete")("order_items.status"))
            .group_by("users.state")
            .agg(F.count("users.id").alias("n"))
        )

        assert "limit: 100 (decomposition cap)" in frame.explain()
        with pytest.warns(TruncationWarning, match="decomposition cap"):
            rows = frame.collect().to_pylist()
        assert sum(row["n"] for row in rows) == 100, "the cap really did trim the input"


def test_a_local_decomposition_cap_that_does_not_bite_is_silent(handler: FakeOmniAPI) -> None:
    for session in capped_session(handler, 50_000):
        frame = (
            session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)
            .select("users.state", "users.id", "order_items.status")
            .filter(F.udf(lambda status: status == "complete")("order_items.status"))
            .group_by("users.state")
            .agg(F.count("users.id").alias("n"))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", TruncationWarning)
            assert frame.collect().num_rows > 0


def test_the_builder_rejects_a_nonsense_cap() -> None:
    with pytest.raises(CompileError, match="positive integer or None"):
        OmniSession.builder.decomposition_row_cap(0)


# --------------------------------------------------------------------------------------
# UDFs and map_pandas (§5)
# --------------------------------------------------------------------------------------


def test_a_udf_filter_falls_back_locally_over_a_pushed_down_query(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    coastal = {"California", "Oregon", "Washington"}
    frame = (
        orders.select("users.state", "order_items.status")
        .filter(F.col("order_items.status") == "complete")
        .filter(F.udf(lambda state: state in coastal)("users.state"))
    )
    rows = frame.collect().to_pylist()
    query = run_queries(handler)[-1]

    assert rows, "the fixture would prove nothing if it kept no rows"
    assert {row["users.state"] for row in rows} == coastal & {row["users.state"] for row in rows}
    assert set(query["filters"]) == {"order_items.status"}, "only the compilable half rode remote"
    assert all(row["order_items.status"] == "complete" for row in rows)


def test_a_udf_can_derive_a_column(orders: DataFrame) -> None:
    frame = (
        orders.select("users.state", "users.age")
        .limit(20)
        .with_column(
            "bucket", F.udf(lambda age: "unknown" if age is None else age // 10 * 10)("users.age")
        )
    )
    rows = frame.collect().to_pylist()

    assert frame.columns == ("users.state", "users.age", "bucket")
    for row in rows:
        expected = "unknown" if row["users.age"] is None else row["users.age"] // 10 * 10
        assert row["bucket"] == expected


def test_map_pandas_round_trips_the_frame(orders: DataFrame) -> None:
    hint = OmniSchema(
        (
            OmniField(name="state", data_type=OmniDataType.STRING),
            OmniField(name="loud", data_type=OmniDataType.STRING),
        )
    )

    def shout(frame: pd.DataFrame) -> pd.DataFrame:
        renamed = frame.rename(columns={"users.state": "state"})
        renamed["loud"] = renamed["state"].astype("string").str.upper()
        return renamed

    frame = orders.select("users.state").limit(10).map_pandas(shout, hint)
    rows = frame.collect().to_pylist()

    assert frame.columns == ("state", "loud")
    assert frame.schema.names == ("state", "loud")
    assert all(row["loud"] == (row["state"] or "").upper() for row in rows)
    assert len(rows) == 10


def test_without_a_hint_the_schema_says_why_it_cannot_be_planned(orders: DataFrame) -> None:
    frame = orders.select("users.state").limit(5).map_pandas(lambda frame: frame)

    with pytest.raises(CompileError, match="cannot be planned"):
        _ = frame.schema
    with pytest.raises(CompileError, match="map_pandas"):
        _ = frame.columns
    assert frame.collect().num_rows == 5, "running it still works — only planning it does not"


def test_the_camel_case_aliases_exist(orders: DataFrame) -> None:
    doubled = orders.select("users.age").limit(3).withColumn("d", F.col("users.age") * 2)
    mapped = orders.select("users.age").limit(3).mapInPandas(lambda frame: frame)

    assert doubled.collect().column_names == ["users.age", "d"]
    assert mapped.collect().num_rows == 3


# --------------------------------------------------------------------------------------
# Limit pins the frontier (§2.3)
# --------------------------------------------------------------------------------------


def test_operations_above_a_limit_run_on_its_result(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    """Until M3 this was a ``CompileError``; the reason it gave is now the semantics it has."""
    page = orders.select("users.state", "order_items.status").limit(20)
    filtered = page.filter(F.col("users.state") == "California").collect()
    query = run_queries(handler)[-1]

    assert query["limit"] == 20, "the limit is part of the query, not applied after it"
    assert query["filters"] == {}, "the filter could not ride the same query"
    assert filtered.num_rows <= 20
    assert {row["users.state"] for row in filtered.to_pylist()} <= {"California"}


def test_a_limit_above_local_work_is_applied_here(orders: DataFrame, handler: FakeOmniAPI) -> None:
    """A UDF keeps the derived column local, so the limit above it cannot ride the wire."""
    frame = (
        orders.select("users.state", "users.age")
        .with_column("shouted", F.udf(lambda state: (state or "?").upper())("users.state"))
        .limit(4)
    )
    rows = frame.collect().to_pylist()

    assert len(rows) == 4
    assert run_queries(handler)[-1]["limit"] == 50_000, "the local limit is not a wire limit"


def test_a_split_plan_has_no_single_query_to_read_off(orders: DataFrame) -> None:
    frame = orders.select("users.state").with_column(
        "loud", F.udf(lambda state: (state or "?").upper())("users.state")
    )

    with pytest.raises(CompileError, match="finishes locally"):
        _ = frame.schema


def test_count_and_first_work_over_a_split_plan(orders: DataFrame) -> None:
    frame = orders.group_by("users.state").agg(F.count_distinct("users.id").alias("buyers"))

    assert frame.count() == 21
    row = frame.sort(F.col("buyers").desc()).first()
    assert row is not None
    assert set(row) == {"users.state", "buyers"}


def test_show_renders_a_split_plan(orders: DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
    orders.group_by("users.state").agg(F.count_distinct("users.id").alias("buyers")).show(3)
    printed = capsys.readouterr().out

    assert "buyers" in printed
    assert "only showing top 3 rows" in printed


def test_explain_analyze_asks_the_server_about_every_step(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    text = (
        orders.group_by("users.state")
        .agg(F.measure(REVENUE), F.count_distinct("users.id"))
        .explain(analyze=True)
    )
    planned = [
        request for request in handler.requests if (request.body or {}).get("planOnly") is True
    ]

    assert len(planned) == 2, "one planOnly round trip per remote step"
    assert text.count("Analyzed [planOnly round trip]") == 2


def test_a_filter_on_an_ad_hoc_aggregation_still_routes_to_a_later_tier(orders: DataFrame) -> None:
    """No aggregate produced it, so there is no column to filter — that is tier 2's job (M5)."""
    frame = orders.select("users.state").filter(F.count_distinct("users.id") > 10)

    with pytest.raises(CompileError, match=r"not yet supported: .*tier 2/3"):
        frame.collect()


def test_measures_from_two_views_still_decompose_around_one_ad_hoc_aggregate(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """The governed half keeps the topic's joins; the raw half only fetches what it aggregates."""
    expected = {
        row["state"]: row for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
    }
    rows = (
        orders.group_by("users.state")
        .agg(
            F.measure(REVENUE).alias("revenue"),
            F.measure(ORDER_COUNT).alias("orders"),
            F.count_distinct("users.id").alias("buyers"),
        )
        .collect()
        .to_pylist()
    )

    assert len(rows) == 21
    for row in rows:
        answer = expected[row["users.state"]]
        assert row["revenue"] == Decimal(answer["total_sale_price"])
        assert row["buyers"] == answer["distinct_buyers"]
