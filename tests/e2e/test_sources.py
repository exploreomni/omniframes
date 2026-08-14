"""M4 end to end: the three sources omniframes does not compile itself (CONTRACT_NOTES §3.4/§4).

``read.sql`` sends a statement the user wrote; ``read.saved_query`` and ``session.ask`` send a
query *Omni* wrote.  All three are opaque to the compiler, which is the point: the bytes that go
out are the bytes that came in, and everything written on top of them runs in the local engine.

The one thing that must never happen is the silent failure of CONTRACT_NOTES §3.4 —
``userEditedSQL`` without ``rewriteSql: false``, which the server answers by ignoring the SQL and
running a well-formed query nobody asked for.  The fake reproduces that literally, so the
assertions on ``handler.requests`` here are the proof that omniframes cannot trip over it.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.dataframe import DataFrame
from omniframes.errors import CompileError, OmniframesError, TruncationWarning
from omniframes.transport import HttpTransport
from omniframes.types import OmniDataType
from tests.fakes import (
    BENCH_DOCUMENT_ID,
    BENCH_MODEL_NAME,
    DEFAULT_TOKEN,
    DOCUMENT_WITHOUT_DASHBOARD,
    FakeOmniAPI,
)

BASE_URL = "https://bench.example.omni.co"
BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "bench"

REVENUE_BY_STATE_SQL = """
SELECT u.state AS state,
       SUM(oi.sale_price) AS revenue,
       COUNT(*) AS items
FROM order_items oi
LEFT JOIN users u ON u.id = oi.user_id
GROUP BY 1
ORDER BY 1
""".strip()


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


def make_session(fake: FakeOmniAPI) -> OmniSession:
    client = httpx.Client(transport=httpx.MockTransport(fake), base_url=BASE_URL)
    transport = HttpTransport(
        base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
    )
    return OmniSession.builder.base_url(BASE_URL).transport(transport).get_or_create()


@pytest.fixture
def session(handler: FakeOmniAPI) -> OmniSession:
    return make_session(handler)


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


def run_queries(fake: FakeOmniAPI) -> list[Mapping[str, Any]]:
    return [
        request.query
        for request in fake.requests
        if request.path.endswith("/query/run") and request.query is not None
    ]


# --------------------------------------------------------------------------------------
# read.sql — the raw-SQL job (CONTRACT_NOTES §3.4)
# --------------------------------------------------------------------------------------


def test_a_sql_scan_answers_the_same_question_the_semantic_query_does(
    session: OmniSession, known_answers: dict[str, Any]
) -> None:
    """Same numbers, different tier: the oracle does not care which one produced them."""
    expected = {row["state"]: row for row in answer_rows(known_answers, "revenue_by_state")}
    rows = collect(session.read.sql(BENCH_MODEL_NAME, REVENUE_BY_STATE_SQL)).to_pylist()

    assert len(rows) == len(expected) == 21
    for row in rows:
        answer = expected[row["state"]]
        assert str(row["revenue"]) == answer["total_sale_price"]
        assert row["items"] == answer["order_items_count"]
    assert any(row["state"] is None for row in rows), "the LEFT JOIN's NULL group survives"


def test_every_sql_envelope_carries_the_do_not_rewrite_marker(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    """Without it the server ignores the SQL and answers a question nobody asked (§3.4)."""
    frame = session.read.sql(BENCH_MODEL_NAME, REVENUE_BY_STATE_SQL)
    collect(frame)
    _ = frame.schema

    sql_queries = [query for query in run_queries(handler) if query.get("userEditedSQL")]
    assert len(sql_queries) == 2, "one run, one planOnly"
    for query in sql_queries:
        assert query["rewriteSql"] is False
        assert query["userEditedSQL"] == REVENUE_BY_STATE_SQL


def test_the_schema_of_a_sql_scan_comes_from_a_plan_only_round_trip(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    """``summary.fields`` on a SQL job is all-dimensions (§6.2); normalize handles it."""
    schema = session.read.sql(BENCH_MODEL_NAME, REVENUE_BY_STATE_SQL).schema

    assert schema.names == ("state", "revenue", "items")
    assert schema["state"].data_type is OmniDataType.STRING
    assert schema["revenue"].data_type is OmniDataType.NUMBER
    assert all(field.is_dimension for field in schema.fields), "a SQL job has no measures"
    assert [r.body["planOnly"] for r in handler.requests if r.path.endswith("/query/run")] == [True]


def test_operations_on_top_of_a_sql_scan_run_locally_over_its_result(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    frame = (
        session.read.sql(BENCH_MODEL_NAME, REVENUE_BY_STATE_SQL)
        .select("state", "revenue")
        .filter(F.col("state").is_not_null())
        .sort(F.col("state"))
        .limit(3)
    )
    rows = collect(frame).to_pylist()

    assert [row["state"] for row in rows] == ["Arizona", "California", "Colorado"]
    assert isinstance(rows[0]["revenue"], Decimal)
    assert len(run_queries(handler)) == 1, "one query; the rest happened here"


def test_explain_names_the_sql_and_every_local_operator(session: OmniSession) -> None:
    text = (
        session.read.sql(BENCH_MODEL_NAME, REVENUE_BY_STATE_SQL)
        .select("state", "revenue")
        .filter(F.col("state") == "California")
        .explain()
    )

    assert "[tier 2 · raw SQL job → POST /api/v1/query/run]" in text
    assert "rewriteSql: false" in text
    assert "SELECT u.state AS state," in text
    assert "Local [arrow compute]" in text
    assert "filter: state = 'California'" in text


def test_the_columns_of_a_bare_sql_scan_are_not_guessed(session: OmniSession) -> None:
    frame = session.read.sql(BENCH_MODEL_NAME, REVENUE_BY_STATE_SQL)

    with pytest.raises(CompileError, match="raw-SQL job decides its own result columns"):
        _ = frame.columns
    assert frame.select("state").columns == ("state",), "a select() answers the question instead"


def test_a_sql_scan_joins_a_governed_query_like_any_other_frame(session: OmniSession) -> None:
    """Each side splits independently, so the tiers can differ — the join does not care."""
    sql = session.read.sql(
        BENCH_MODEL_NAME,
        'SELECT DISTINCT u.state AS "users.state", 1 AS flag FROM users u WHERE u.state IS NOT NULL',
    )
    governed = (
        session.read.topic(BENCH_MODEL_NAME, "order_items")
        .group_by("users.state")
        .agg(F.measure("order_items.total_sale_price").alias("revenue"))
    )
    joined = collect(governed.join(sql, "users.state", "inner")).to_pylist()

    assert len(joined) == 20, "the NULL-state group has no counterpart, and NULL matches nothing"
    assert all(row["flag"] == 1 for row in joined)


def test_read_sql_needs_a_statement(session: OmniSession) -> None:
    with pytest.raises(CompileError, match="needs a SQL statement"):
        session.read.sql(BENCH_MODEL_NAME, "   ")


# --------------------------------------------------------------------------------------
# read.saved_query — a stored query, run verbatim (CONTRACT_NOTES §4)
# --------------------------------------------------------------------------------------


def test_the_saved_queries_of_a_document_can_be_listed(session: OmniSession) -> None:
    assert session.read.saved_queries(BENCH_DOCUMENT_ID) == (
        ("Revenue by state", 0),
        ("Monthly revenue", 1),
        ("Average sale price by category", 2),
        ("Completed west-coast revenue", 3),
    )


def test_a_saved_query_round_trips_through_the_pipeline(
    session: OmniSession, handler: FakeOmniAPI, known_answers: dict[str, Any]
) -> None:
    expected = {row["state"]: row for row in answer_rows(known_answers, "revenue_by_state")}
    frame = session.read.saved_query(BENCH_DOCUMENT_ID, "Revenue by state")
    rows = collect(frame).to_pylist()

    assert frame.columns == ("users.state", "order_items.total_sale_price")
    assert len(rows) == len(expected)
    for row in rows:
        assert (
            str(row["order_items.total_sale_price"])
            == expected[row["users.state"]]["total_sale_price"]
        )


def test_the_stored_blob_goes_back_on_the_wire_unchanged(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    """Omniframes did not write this query; normalizing it would change somebody's answer."""
    stored = handler.documents[BENCH_DOCUMENT_ID][1].query
    collect(session.read.saved_query(BENCH_DOCUMENT_ID, "Monthly revenue"))

    assert run_queries(handler)[0] == dict(stored)


def test_a_saved_query_may_be_picked_by_index(session: OmniSession) -> None:
    frame = session.read.saved_query(BENCH_DOCUMENT_ID, 2)

    assert frame.columns == ("products.category", "order_items.average_sale_price")
    assert collect(frame).num_rows > 0


def test_the_first_saved_query_is_the_default(session: OmniSession) -> None:
    assert session.read.saved_query(BENCH_DOCUMENT_ID).columns == (
        session.read.saved_query(BENCH_DOCUMENT_ID, 0).columns
    )


def test_operations_above_a_saved_query_run_locally(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    frame = session.read.saved_query(BENCH_DOCUMENT_ID, "Revenue by state").select(
        F.col("users.state").alias("state")
    )
    rows = collect(frame).to_pylist()

    assert frame.columns == ("state",)
    assert len(rows) == 21
    assert run_queries(handler)[0]["fields"] == ["users.state", "order_items.total_sale_price"]


def test_a_document_without_a_dashboard_is_its_own_answer(session: OmniSession) -> None:
    """One of §4's two 404s: the document exists, it just has nothing saved on it."""
    with pytest.raises(OmniframesError, match="exists but has no dashboard"):
        session.read.saved_query(DOCUMENT_WITHOUT_DASHBOARD)


def test_an_unknown_document_is_the_other_404(session: OmniSession) -> None:
    with pytest.raises(OmniframesError, match="no document 'nope' is visible"):
        session.read.saved_query("nope")


def test_a_missing_saved_query_lists_what_the_document_has(session: OmniSession) -> None:
    with pytest.raises(CompileError, match="'Revenue by state' \\[0\\]"):
        session.read.saved_query(BENCH_DOCUMENT_ID, "Revenue by county")
    with pytest.raises(CompileError, match="index 9 is out of range"):
        session.read.saved_query(BENCH_DOCUMENT_ID, 9)


def test_a_stored_query_without_a_model_id_says_how_to_supply_one(handler: FakeOmniAPI) -> None:
    from tests.fakes import SavedQuery, bench_query

    blob = {k: v for k, v in bench_query(fields=["users.state"]).items() if k != "modelId"}
    handler.documents["headless"] = (SavedQuery("q", "Headless", blob, "https://example"),)
    session = make_session(handler)

    with pytest.raises(CompileError, match="carries no modelId"):
        session.read.saved_query("headless")
    injected = session.read.saved_query("headless", model=BENCH_MODEL_NAME)
    assert injected.columns == ("users.state",)


def test_a_saved_query_explains_itself_by_name(session: OmniSession) -> None:
    text = session.read.saved_query(BENCH_DOCUMENT_ID, "Revenue by state").explain()

    assert "saved query: Revenue by state (bench_dashboard)" in text
    assert "fields: [users.state, order_items.total_sale_price]" in text
    assert "sort: order_items.total_sale_price DESC" in text
    assert "limit: 1000   version: 9" in text, "the stored limit, not the library default"


def test_a_saved_querys_filters_are_named_but_not_recompiled(session: OmniSession) -> None:
    """docs/HYBRID.md §7 pins this line by example: the keys, and *why* only the keys.

    omniframes did not write the blob, so it does not re-read the filters into its own typed
    model — naming them is the honest amount of detail, and saying "sent verbatim" is what tells
    a reader the compiler had no hand in them.
    """
    text = session.read.saved_query(BENCH_DOCUMENT_ID, "Completed west-coast revenue").explain()

    assert (
        "  filters: [order_items.status, users.state] (stored — sent verbatim, not recompiled)"
    ) in text


def test_a_saved_query_with_no_limit_key_warns_about_the_servers_default(
    handler: FakeOmniAPI,
) -> None:
    """An absent ``limit`` is the server's 1000-row default, and it truncates like any other.

    The blob still rides the wire verbatim (omniframes did not write it), but ``explain()`` says
    which limit will apply and the action warns when the result comes back full — otherwise a
    query that silently dropped 9000 rows reads as ``unlimited (null)``.
    """
    from tests.fakes import SavedQuery, bench_query

    blob = bench_query(fields=["order_items.id", "order_items.sale_price"])
    del blob["limit"]
    handler.documents["nolimit"] = (SavedQuery("q", "No limit", blob, "https://example"),)
    session = make_session(handler)
    frame = session.read.saved_query("nolimit")

    assert "limit: 1000 (server default — the stored query carries no limit)" in frame.explain()
    with pytest.warns(TruncationWarning, match="1000 rows"):
        table = frame.collect()
    assert table.num_rows == 1000
    sent = handler.requests[-1].query
    assert sent is not None
    assert "limit" not in sent, "the blob is still sent exactly as it arrived"


# --------------------------------------------------------------------------------------
# session.ask — generate-query with runQuery: false (CONTRACT_NOTES §4)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "columns"),
    [
        ("monthly revenue", ("order_items.created_at[month]", "order_items.total_sale_price")),
        ("revenue by state", ("users.state", "order_items.total_sale_price")),
        ("top products please", ("products.name", "order_items.total_sale_price")),
    ],
)
def test_ask_runs_the_generated_query_through_the_pipeline(
    session: OmniSession, prompt: str, columns: tuple[str, ...]
) -> None:
    frame = session.ask(prompt, model=BENCH_MODEL_NAME)
    table = collect(frame)

    assert frame.columns == columns
    assert table.column_names == list(columns)
    assert table.num_rows > 0


def test_ask_always_sends_run_query_false(session: OmniSession, handler: FakeOmniAPI) -> None:
    """The server's default is ``true``, which would run the query outside this pipeline (§4)."""
    session.ask("revenue by state", model=BENCH_MODEL_NAME, topic="order_items")
    generated = [r for r in handler.requests if r.path.endswith("/ai/generate-query")]

    assert len(generated) == 1
    assert generated[0].body["runQuery"] is False
    assert generated[0].body["currentTopicName"] == "order_items"


def test_ask_answers_the_known_numbers(session: OmniSession, known_answers: dict[str, Any]) -> None:
    expected = {row["state"]: row for row in answer_rows(known_answers, "revenue_by_state")}
    rows = collect(session.ask("revenue by state", model=BENCH_MODEL_NAME)).to_pylist()

    for row in rows:
        assert (
            str(row["order_items.total_sale_price"])
            == expected[row["users.state"]]["total_sale_price"]
        )


def test_ask_explains_itself_as_the_prompt(session: OmniSession) -> None:
    text = session.ask("revenue by state", model=BENCH_MODEL_NAME).explain()

    assert 'ask("revenue by state")' in text
    assert "generated query" in text


def test_a_prompt_omni_cannot_answer_is_reported_with_the_prompt(session: OmniSession) -> None:
    with pytest.raises(OmniframesError, match="could not generate a query for 'how tall is Ben'"):
        session.ask("how tall is Ben", model=BENCH_MODEL_NAME)


def test_exhausted_ai_credits_get_their_own_message() -> None:
    fake = FakeOmniAPI(ai_credits_exhausted=True)
    try:
        with pytest.raises(OmniframesError, match="AI credits are exhausted"):
            make_session(fake).ask("revenue by state", model=BENCH_MODEL_NAME)
    finally:
        fake.close()


def test_ask_needs_a_prompt(session: OmniSession) -> None:
    with pytest.raises(CompileError, match="needs a prompt"):
        session.ask("  ", model=BENCH_MODEL_NAME)


def test_an_asked_frame_can_be_joined_like_any_other(session: OmniSession) -> None:
    asked = session.ask("revenue by state", model=BENCH_MODEL_NAME)
    buyers = (
        session.read.topic(BENCH_MODEL_NAME, "order_items")
        .group_by("users.state")
        .agg(F.measure("users.count").alias("buyers"))
    )
    joined = collect(asked.join(buyers, "users.state", "inner"))

    assert joined.column_names == [
        "users.state",
        "order_items.total_sale_price",
        "buyers",
    ]
    assert joined.num_rows == 20, "NULL states on both sides, matching neither"


def test_a_sql_scan_can_be_selected_from_above_a_join(session: OmniSession) -> None:
    """The SQL's columns are unknown until it runs, so the check happens where the data is."""
    sql = session.read.sql(
        BENCH_MODEL_NAME,
        'SELECT u.state AS "users.state", MIN(u.country) AS country '
        "FROM users u WHERE u.state IS NOT NULL GROUP BY 1",
    )
    governed = (
        session.read.topic(BENCH_MODEL_NAME, "order_items")
        .group_by("users.state")
        .agg(F.measure("order_items.total_sale_price").alias("revenue"))
    )
    rows = collect(governed.join(sql, "users.state", "inner").select("users.state", "country"))

    assert rows.column_names == ["users.state", "country"]
    assert rows.num_rows == 20


def test_a_sql_scan_unions_a_governed_query_when_the_names_line_up(session: OmniSession) -> None:
    sql = session.read.sql(BENCH_MODEL_NAME, "SELECT 'Atlantis' AS \"users.state\"")
    governed = session.read.topic(BENCH_MODEL_NAME, "order_items").select("users.state").limit(3)
    stacked = collect(governed.union(sql))

    assert stacked.column_names == ["users.state"]
    assert stacked.num_rows == 4
    assert "Atlantis" in [row["users.state"] for row in stacked.to_pylist()]


def test_a_union_with_a_sql_scan_whose_names_differ_fails_where_the_data_is(
    session: OmniSession,
) -> None:
    sql = session.read.sql(BENCH_MODEL_NAME, "SELECT 'Atlantis' AS province")
    governed = session.read.topic(BENCH_MODEL_NAME, "order_items").select("users.state").limit(3)

    with pytest.raises(CompileError, match="same columns in the same order"):
        collect(governed.union(sql))
