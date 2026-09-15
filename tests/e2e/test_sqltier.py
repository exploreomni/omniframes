"""M5 end to end: tier 2 against FakeOmniAPI (docs/SQLTIER.md).

Numeric expectations come from ``tests/data/bench/known_answers.json`` — never from a
recomputation inside the test — so every assertion here also holds against the live org.  That is
the point of running tier 2 through the fake at all: the OmniSQL omniframes writes is resolved
against the bench model and executed for real (DuckDB, over the checked-in parquet), and the
answer has to match one nobody in this pipeline produced.

The envelope assertions read ``handler.requests``, because the wire invariants of §2 — an
**absent** ``rewriteSql`` key, no ``staticQueryReferences`` and no ``sqlSortsEnabled``, and a
``LIMIT`` in the statement text rather than only in the query object — are the ones the server
punishes silently rather than loudly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.dataframe import DataFrame
from omniframes.errors import CompileError, OmniframesError, QueryError
from omniframes.plan import nodes
from omniframes.transport import HttpTransport
from tests.fakes import (
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)
from tests.fakes.bench_model import BENCH_TOPIC

BASE_URL = "https://bench.omniapp.co"
BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "bench"

REVENUE = "order_items.total_sale_price"
STATE = "users.state"
AGE = "users.age"
PRICE = "order_items.sale_price"
BUYER = "users.id"
ID = "order_items.id"


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


def sql_queries(handler: FakeOmniAPI) -> list[Mapping[str, Any]]:
    return [query for query in run_queries(handler) if query.get("userEditedSQL")]


# --------------------------------------------------------------------------------------
# The answers (tests/data/bench/known_answers.json)
# --------------------------------------------------------------------------------------


def test_distinct_topic_name_resolves_to_a_view_for_sql(known_answers: dict[str, Any]) -> None:
    topic = replace(BENCH_TOPIC, name="sales_topic")
    fake = FakeOmniAPI(topic=topic)
    try:
        with httpx.Client(transport=httpx.MockTransport(fake), base_url=BASE_URL) as client:
            transport = HttpTransport(base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client)
            with OmniSession.builder.transport(transport).get_or_create() as session:
                orders = session.read.topic(BENCH_MODEL_NAME, topic.name)
                test_revenue_by_state_as_an_ad_hoc_aggregate_matches_the_known_answers(
                    orders, fake, known_answers
                )
                assert "FROM ${order_items}" in sql_queries(fake)[0]["userEditedSQL"]
                # The fake must reject the original bug, not silently translate a topic to a view.
                with pytest.raises(QueryError, match='No such view "sales_topic"'):
                    transport.run(
                        {
                            "query": {
                                "modelId": BENCH_MODEL_ID,
                                "fields": [],
                                "limit": 1,
                                "userEditedSQL": "SELECT ${order_items.id} FROM ${sales_topic} LIMIT 1",
                            }
                        }
                    )
    finally:
        fake.close()


def test_revenue_by_state_as_an_ad_hoc_aggregate_matches_the_known_answers(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    """``SUM(sale_price)`` written by omniframes must equal the governed measure's own answer."""
    expected = {row["state"]: row for row in answer_rows(known_answers, "revenue_by_state")}
    rows = (
        orders.group_by(STATE)
        .agg(F.sum(PRICE).alias("revenue"), F.count(ID).alias("n"))
        .collect()
        .to_pylist()
    )

    assert len(rows) == len(expected) == 21
    for row in rows:
        answer = expected[row[STATE]]
        assert row["revenue"] == Decimal(answer["total_sale_price"])
        assert row["n"] == answer["order_items_count"]
    assert len(sql_queries(handler)) == 1, "one SQL job, not a scan plus a local aggregate"


def test_distinct_buyers_by_month_rides_one_sql_job(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    """A grained group key: a bare ``${field[month]}`` select item, grouped positionally.

    The month arrives as the ``__raw`` half of the formatted pair the model produces (§5), so
    the values compare as timestamps rather than as ``YYYY-MM`` strings.
    """
    expected = answer_rows(known_answers, "distinct_buyers_by_month")
    month = F.col("order_items.created_at").grain("month").alias("month")
    rows = (
        orders.group_by(month)
        .agg(F.count_distinct(BUYER).alias("buyers"))
        .sort("month")
        .collect()
        .to_pylist()
    )
    query = sql_queries(handler)[-1]

    assert len(rows) == len(expected) == 24
    for row, answer in zip(rows, expected, strict=True):
        assert row["month"] == datetime.fromisoformat(answer["month"])
        assert row["buyers"] == answer["distinct_buyers"]
    assert "rewriteSql" not in query, "an ABSENT key is what selects the parsed path"
    assert "staticQueryReferences" not in query
    assert "${order_items.created_at[month]}" in query["userEditedSQL"]
    assert "GROUP BY\n  1" in query["userEditedSQL"], "positional — the only verified form"


def test_average_sale_price_by_category_matches_the_known_answers(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """AVG over decimals: the warehouse returns a float, exactly as the local engine does."""
    expected = {
        row["category"]: row for row in answer_rows(known_answers, "average_sale_price_by_category")
    }
    rows = (
        orders.group_by("products.category")
        .agg(F.avg(PRICE).alias("mean"), F.count(ID).alias("n"))
        .collect()
        .to_pylist()
    )

    assert len(rows) == len(expected)
    for row in rows:
        answer = expected[row["products.category"]]
        assert row["n"] == answer["order_items_count"]
        assert row["mean"] == pytest.approx(float(answer["average_sale_price"]), rel=1e-9)


def test_a_having_on_an_ad_hoc_aggregate_runs_in_the_warehouse(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    """M3 filtered the computed column here; M5 writes it as a genuine HAVING."""
    expected = {
        row["state"]
        for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
        if row["distinct_buyers"] > 25
    }
    rows = (
        orders.group_by(STATE)
        .agg(F.count_distinct(BUYER).alias("buyers"))
        .filter(F.col("buyers") > 25)
        .collect()
        .to_pylist()
    )

    assert 0 < len(expected) < 21
    assert {row[STATE] for row in rows} == expected
    assert "HAVING" in sql_queries(handler)[-1]["userEditedSQL"]


def test_a_cross_field_or_runs_as_a_sql_where(orders: DataFrame, handler: FakeOmniAPI) -> None:
    """Permanently out of reach for tier 1 (it needs `controls`); tier 2 just writes it."""
    coastal = (F.col(STATE) == "California") | (F.col(AGE) > 60)
    rows = orders.select(ID, STATE, AGE).filter(coastal).collect().to_pylist()
    query = sql_queries(handler)[-1]

    assert rows
    assert all(row[STATE] == "California" or (row[AGE] or 0) > 60 for row in rows)
    assert "staticQueryReferences" not in query, "the statement is the whole plan"
    assert query["filters"] == {}, "nothing rides the query object any more"
    assert "WHERE" in query["userEditedSQL"]


def test_a_computed_column_is_a_select_expression(orders: DataFrame, handler: FakeOmniAPI) -> None:
    rows = (
        orders.select(ID, PRICE, "order_items.quantity")
        .with_column("line_total", F.col(PRICE) * F.col("order_items.quantity"))
        .limit(50)
        .collect()
        .to_pylist()
    )

    assert len(rows) == 50
    for row in rows:
        assert row["line_total"] == row[PRICE] * row["order_items.quantity"]
    # §3.2 regime (b): an expression item is emitted under a generated alias and renamed
    # client-side by suffix match, because the server prefixes it with a scope it picks itself.
    assert "AS of_expr_1" in sql_queries(handler)[-1]["userEditedSQL"]


# --------------------------------------------------------------------------------------
# The mixed aggregation: one OmniSQL statement (docs/SQLTIER.md §4)
# --------------------------------------------------------------------------------------


def test_the_mixed_aggregation_is_exactly_one_request(
    orders: DataFrame, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    """A governed measure is a legal select item, so the whole node rides one statement.

    v1 sent two requests here (tier-1 measures + a tier-2 ad-hoc half) and joined them locally.
    ``${order_items.total_sale_price}`` expands to its governed SQL inside the same statement
    that computes ``COUNT(DISTINCT …)``, so there is no second query and no align-join.
    """
    expected = {
        row["state"]: row for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
    }
    table = (
        orders.group_by(STATE)
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct(BUYER).alias("buyers"))
        .collect()
    )
    rows = table.to_pylist()
    sent = run_queries(handler)

    assert table.column_names == [STATE, "revenue", "buyers"]
    assert len(rows) == len(expected) == 21
    for row in rows:
        answer = expected[row[STATE]]
        assert str(row["revenue"]) == answer["total_sale_price"]
        assert row["buyers"] == answer["distinct_buyers"]
    assert sum(row["buyers"] for row in rows) == 497, "every buyer lands in exactly one group"

    assert len(sent) == 1, "the measure and the ad-hoc aggregate share one statement"
    (generated,) = sent
    statement = generated["userEditedSQL"]
    assert f"${{{REVENUE}}}" in statement
    assert f"COUNT(DISTINCT ${{{BUYER}}})" in statement
    assert "rewriteSql" not in generated


def test_the_null_group_survives_the_governed_expansion(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """One GROUP BY over one statement: the NULL-state group is one row, as it always was."""
    expected = {
        row["state"]: row for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
    }
    rows = (
        orders.group_by(STATE)
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct(BUYER).alias("buyers"))
        .collect()
        .to_pylist()
    )
    null_group = [row for row in rows if row[STATE] is None]

    assert len(null_group) == 1
    assert null_group[0]["buyers"] == expected[None]["distinct_buyers"] > 0


# --------------------------------------------------------------------------------------
# Envelope invariants (docs/SQLTIER.md §2)
# --------------------------------------------------------------------------------------


def test_every_sql_job_omniframes_sends_carries_the_invariants(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    orders.group_by(STATE).agg(F.count_distinct(BUYER).alias("buyers")).collect()
    orders.select(ID, STATE, AGE).filter((F.col(STATE) == "Ohio") | (F.col(AGE) > 60)).collect()
    queries = sql_queries(handler)

    assert len(queries) == 2
    for query in queries:
        assert "rewriteSql" not in query, "even `false` sends the ${…} refs to the warehouse"
        assert "sqlSortsEnabled" not in query, "the statement carries its own sorts"
        assert "staticQueryReferences" not in query, "the statement is the whole plan"
        assert query["sorts"] == []
        assert query["modelId"] == BENCH_MODEL_ID
        assert isinstance(query["limit"], int), "the envelope mirrors the text's LIMIT"
        assert f"LIMIT {query['limit']}" in query["userEditedSQL"], (
            "the query-object limit is IGNORED on this path, so the text must carry it"
        )
        assert query["version"] == 9


def test_explain_shows_tier_two_and_prints_the_statement(orders: DataFrame) -> None:
    text = (
        orders.group_by(STATE)
        .agg(F.count_distinct(BUYER).alias("buyers"))
        .filter(F.col("buyers") > 25)
        .explain()
    )

    assert "Remote [tier 2 · sql → POST /api/v1/query/run]" in text
    assert "topic: order_items   model: bench_ecommerce" in text
    assert "  sql:" in text
    assert f"    SELECT\n      ${{{STATE}}}," in text
    assert "references:" not in text, "the reference core went away with v1"


def test_explain_of_the_mixed_aggregation_shows_one_tier_two_step(orders: DataFrame) -> None:
    text = (
        orders.group_by(STATE)
        .agg(F.measure(REVENUE).alias("revenue"), F.count_distinct(BUYER).alias("buyers"))
        .explain()
    )

    assert "Remote [tier 2 · sql" in text
    assert "Remote step" not in text, "one step, so nothing is numbered"
    assert "align-join" not in text
    assert "raw scan" not in text
    assert f"      ${{{REVENUE}}}," in text, "the measure rides the statement"


def test_a_long_statement_is_truncated_in_explain(orders: DataFrame) -> None:
    text = (
        orders.select(ID, STATE, AGE)
        .filter((F.col(STATE) == "Ohio") | (F.col(AGE) > 60))
        .sort(ID)
        .explain()
    )

    assert "… (+" in text
    assert "more lines)" in text


def test_explain_analyze_still_asks_the_server_about_a_sql_step(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    text = orders.group_by(STATE).agg(F.count_distinct(BUYER)).explain(analyze=True)
    planned = [
        request for request in handler.requests if (request.body or {}).get("planOnly") is True
    ]

    assert len(planned) == 1
    assert text.count("Analyzed [planOnly round trip]") == 1


# --------------------------------------------------------------------------------------
# Shapes that stay tier 3 (docs/SQLTIER.md §1)
# --------------------------------------------------------------------------------------


def test_a_udf_still_pins_the_frontier_and_the_query_below_stays_governed(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    coastal = {"California", "Oregon", "Washington"}
    rows = (
        orders.select(STATE, "order_items.status")
        .filter(F.col("order_items.status") == "complete")
        .filter(F.udf(lambda state: state in coastal)(STATE))
        .collect()
        .to_pylist()
    )

    assert rows
    assert sql_queries(handler) == [], "a UDF has no SQL rendering, so nothing was written"
    assert set(run_queries(handler)[-1]["filters"]) == {"order_items.status"}


def test_an_operation_over_a_raw_sql_scan_is_never_re_derived(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    """An opaque scan is a wall in both directions — tier 2 included."""
    scan = session.read.sql(
        BENCH_MODEL_NAME,
        "SELECT u.state AS state, COUNT(*) AS n FROM order_items oi "
        "LEFT JOIN users u ON u.id = oi.user_id GROUP BY 1",
    )
    rows = scan.select("state", "n").filter(F.col("state") == "Texas").collect().to_pylist()
    sent = run_queries(handler)

    assert len(rows) == 1
    assert len(sent) == 1
    assert "staticQueryReferences" not in sent[0], "the user's SQL went as written"
    assert sent[0]["rewriteSql"] is False, "verbatim, not parsed as OmniSQL"


def test_an_aggregate_above_a_user_limit_still_decomposes(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    """The limit pins the frontier: a page-then-aggregate is the question that was asked."""
    rows = (
        orders.select(STATE, PRICE)
        .sort(ID)
        .limit(300)
        .group_by(STATE)
        .agg(F.sum(PRICE).alias("total"))
        .collect()
        .to_pylist()
    )

    assert rows
    assert sql_queries(handler) == []
    assert run_queries(handler)[-1]["limit"] == 300


# --------------------------------------------------------------------------------------
# A statement the model does not bind (docs/SQLTIER.md §8)
# --------------------------------------------------------------------------------------


def test_a_field_the_model_cannot_bind_is_reported_as_omniframes_own_sql(
    orders: DataFrame,
) -> None:
    """``Could not substitute Omni SQL`` on a tier-2 step is never the user's SQL to fix.

    Nothing in the dataframe code that produced it mentions SQL, so the server's bare text reads
    as a mystery.  The mapping says whose statement it is, what the model would not bind, and
    where to read the statement — and keeps the server's own words at the end.
    """
    frame = orders.group_by("users.retired_column").agg(F.count_distinct(BUYER))

    with pytest.raises(QueryError, match="tier-2 statement omniframes generated") as caught:
        frame.collect()
    error = caught.value
    assert "no field 'users.retired_column'" in str(error)
    assert "explain()" in str(error)
    assert "Could not substitute Omni SQL" in str(error), "the server's own words survive"
    assert error.statement is not None
    assert "${users.retired_column}" in error.statement


def test_a_topic_base_view_the_model_cannot_bind_names_the_view(session: OmniSession) -> None:
    """The FROM arm of the same error.  Built off a scan node, because ``read.topic`` checks."""
    frame = DataFrame(
        session,
        nodes.Scan(
            nodes.TopicScan(
                model_name=BENCH_MODEL_NAME,
                model_id=BENCH_MODEL_ID,
                topic="sales_topic",
                base_view="retired_view",
            )
        ),
    )

    with pytest.raises(QueryError, match="no view 'retired_view'"):
        frame.group_by(STATE).agg(F.count_distinct(BUYER)).collect()


def test_a_raw_sql_scan_keeps_the_servers_own_error(session: OmniSession) -> None:
    """§8: the SQL is the user's own, so nothing is re-attributed to omniframes."""
    frame = session.read.sql(BENCH_MODEL_NAME, "SELECT * FROM no_such_table")

    with pytest.raises(QueryError) as caught:
        frame.collect()
    assert "omniframes generated" not in str(caught.value)
    assert caught.value.statement is None


# --------------------------------------------------------------------------------------
# omni_url() (docs/SQLTIER.md §6)
# --------------------------------------------------------------------------------------


def test_omni_url_returns_the_header_the_server_minted(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    """The client plumbing, not the URL format.

    docs/SQLTIER.md §6 and CONTRACT_NOTES §6 #10 both say the shape is LIVE-VALIDATE and must
    not be pinned offline — ``/w/fake/<job-id>`` is the fake's invention, so asserting a prefix
    of it would fail against any other equally plausible format while proving nothing extra.
    Equality against the header the fake actually minted is the claim this test's name makes.
    """
    url = orders.select(STATE, F.measure(REVENUE)).limit(10).omni_url()
    envelope = handler.requests[-1].body

    assert envelope["workbookUrl"] is True
    assert handler.workbook_urls, "the fake minted one for this run"
    assert url == handler.workbook_urls[-1]


def test_omni_url_refuses_a_tier_two_frame_before_sending_anything(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    """A workbook explores model fields, and a tier-2 job has a statement instead."""
    frame = orders.group_by(STATE).agg(F.count_distinct(BUYER))
    before = len(handler.requests)

    with pytest.raises(CompileError, match="governed query"):
        frame.omni_url()
    assert len(handler.requests) == before, "refused client-side, before any request"


def test_omni_url_refuses_the_mixed_aggregation_too(orders: DataFrame) -> None:
    """It is one query again since §4's collapse — but a tier-2 one, so still no workbook."""
    frame = orders.group_by(STATE).agg(F.measure(REVENUE), F.count_distinct(BUYER))

    with pytest.raises(CompileError, match="governed query"):
        frame.omni_url()


def test_omni_url_refuses_a_split_plan(orders: DataFrame) -> None:
    frame = orders.select(STATE).with_column(
        "loud", F.udf(lambda state: (state or "?").upper())(STATE)
    )

    with pytest.raises(CompileError, match="compiles to several"):
        frame.omni_url()


def test_omni_url_refuses_a_raw_sql_job(session: OmniSession) -> None:
    frame = session.read.sql(BENCH_MODEL_NAME, "SELECT 1 AS x")

    with pytest.raises(CompileError, match="governed query"):
        frame.omni_url()


def test_omni_url_says_so_when_the_server_mints_no_url(
    orders: DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An org without workbook links answers the query and sends no header; say so plainly."""
    original = HttpTransport.run

    def without_url(
        self: HttpTransport, envelope: dict[str, Any], *, deadline_seconds: float | None = None
    ) -> Any:
        return replace(
            original(self, envelope, deadline_seconds=deadline_seconds), workbook_url=None
        )

    monkeypatch.setattr(HttpTransport, "run", without_url)
    with pytest.raises(OmniframesError, match="no workbook URL"):
        orders.select(STATE).limit(1).omni_url()
