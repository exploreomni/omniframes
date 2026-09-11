"""The M1 read path, end to end against FakeOmniAPI.

Numeric expectations come from ``tests/data/bench/known_answers.json`` — never from a
recomputation inside the test — so the same assertions hold against the live org.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.dataframe import DEFAULT_FETCH_LIMIT, DataFrame
from omniframes.errors import FeatureFlagError, ModelPermissionError, TruncationWarning
from omniframes.transport import HttpTransport
from omniframes.types import OmniDataType
from tests.fakes import (
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
    RecordedRequest,
)

BASE_URL = "https://bench.omniapp.co"
BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "bench"
BRANCH_ID = "9c3b1e5e-2f4a-4d1b-9a7e-6b0f2d8c4a11"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def build_session(handler: FakeOmniAPI, **options: Any) -> OmniSession:
    """A session wired to the fake through the real HttpTransport."""
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    transport = HttpTransport(
        base_url=BASE_URL,
        api_key=DEFAULT_TOKEN,
        client=client,
        sleep=lambda _: None,
        branch_id=options.get("branch"),
    )
    builder = OmniSession.builder.base_url(BASE_URL).transport(transport)
    for name, value in options.items():
        getattr(builder, name)(value)
    return builder.get_or_create()


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


@pytest.fixture
def session(handler: FakeOmniAPI) -> OmniSession:
    return build_session(handler)


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


def rows_by_state(answers: Mapping[str, Any]) -> dict[str | None, int]:
    return {
        row["state"]: row["order_items_count"] for row in answer_rows(answers, "revenue_by_state")
    }


def run_requests(handler: FakeOmniAPI) -> list[RecordedRequest]:
    return [request for request in handler.requests if request.path.endswith("/query/run")]


# --------------------------------------------------------------------------------------
# Building and verifying
# --------------------------------------------------------------------------------------


def test_building_a_session_touches_nothing(handler: FakeOmniAPI) -> None:
    build_session(handler)

    assert handler.requests == []


def test_verify_reports_the_key_and_its_permissions(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    payload = session.verify()

    assert handler.paths == ["GET /api/v1/whoami"]
    assert payload["keyScope"] == "organization"
    assert "QUERY_TOPICS" in payload["rolesByModel"][BENCH_MODEL_ID]["permissions"]


def test_a_catalog_walk_finds_the_model_topic_views_and_fields(session: OmniSession) -> None:
    model = session.catalog.model(BENCH_MODEL_NAME)
    topics = session.catalog.topics(model.name)
    topic = session.catalog.topic(model.name, topics[0].name)

    assert model.id == BENCH_MODEL_ID
    assert [t.name for t in topics] == [BENCH_TOPIC_NAME]
    assert [view.name for view in topic.views] == ["order_items", "users", "products"]
    assert topic.field("users.state").data_type is OmniDataType.STRING
    assert {view.name for view in session.catalog.views(model.name)} == {
        "order_items",
        "users",
        "products",
    }


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def test_filtering_on_a_joined_dimension_matches_the_known_answer(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    frame = (
        orders.select("order_items.id", "users.state")
        .filter(F.col("users.state") == "California")
        .to_pandas()
    )

    assert len(frame) == rows_by_state(known_answers)["California"] == 1260
    assert set(frame["users.state"]) == {"California"}


def test_the_null_state_group_includes_the_orphan_user_ids(
    orders: DataFrame, known_answers: dict[str, Any]
) -> None:
    """The topic's LEFT joins keep fact rows whose ``user_id`` has no user."""
    frame = (
        orders.select("order_items.id", "users.id", "users.state")
        .filter(F.col("users.state").is_null())
        .to_pandas()
    )

    assert len(frame) == rows_by_state(known_answers)[None] == 590
    assert frame["users.id"].isna().any(), "no orphan rows survived the join"


def test_select_filter_sort_limit_in_one_query(orders: DataFrame) -> None:
    frame = (
        orders.select("order_items.id", "order_items.status")
        .filter(F.col("order_items.status").isin("cancelled", "returned"))
        .sort(F.col("order_items.id").desc())
        .limit(4)
        .to_pandas()
    )

    assert list(frame["order_items.id"]) == sorted(frame["order_items.id"], reverse=True)
    assert set(frame["order_items.status"]) <= {"cancelled", "returned"}


def test_aliases_are_renamed_client_side_and_are_sortable(orders: DataFrame) -> None:
    frame = (
        orders.select(
            F.col("users.state").alias("state"), F.col("order_items.id").alias("order_item")
        )
        .filter(F.col("state") == "Texas")
        .sort(F.col("order_item"))
        .limit(3)
        .to_pandas()
    )

    assert list(frame.columns) == ["state", "order_item"]
    assert set(frame["state"]) == {"Texas"}


def test_offset_skips_rows_of_the_sorted_result(orders: DataFrame) -> None:
    ordered = orders.select("order_items.id").sort("order_items.id")
    first = ordered.limit(5).to_pandas()["order_items.id"].tolist()
    shifted = ordered.limit(5).offset(2).to_pandas()["order_items.id"].tolist()

    assert first == [1, 2, 3, 4, 5]
    assert shifted == [3, 4, 5, 6, 7]


def test_count_and_first(orders: DataFrame, known_answers: dict[str, Any]) -> None:
    expected = {
        row["status"]: row["order_items_count"]
        for row in answer_rows(known_answers, "count_by_status")
    }
    frame = orders.select("order_items.id").sort("order_items.id")

    assert frame.first() == {"order_items.id": 1}
    assert frame.filter(F.col("order_items.status") == "returned").count() == expected["returned"]


def test_a_number_filter_is_inclusive_where_asked(orders: DataFrame) -> None:
    quantities = orders.select("order_items.quantity")
    strictly = quantities.filter(F.col("order_items.quantity") > 7).to_pandas()
    inclusive = quantities.filter(F.col("order_items.quantity") >= 7).to_pandas()

    assert set(strictly["order_items.quantity"]) == {8}
    assert set(inclusive["order_items.quantity"]) == {7, 8}


def test_between_over_numbers_includes_both_ends_end_to_end(orders: DataFrame) -> None:
    """PySpark-consistent, and NOT the wire's BETWEEN kind (whose upper bound is exclusive)."""
    frame = (
        orders.select("order_items.quantity")
        .filter(F.col("order_items.quantity").between(3, 5))
        .to_pandas()
    )

    assert set(frame["order_items.quantity"]) == {3, 4, 5}


def test_between_over_dates_is_half_open_end_to_end(orders: DataFrame) -> None:
    """The date arm has no inclusive upper bound; the window is [low, high)."""
    created = F.col("order_items.created_at")
    frame = (
        orders.select("order_items.created_at")
        .filter(created.between(date(2026, 6, 1), date(2026, 7, 1)))
        .to_pandas()
    )
    stamps = [value.isoformat() for value in frame["order_items.created_at"]]

    assert stamps
    assert all(
        "2026-06-01T00:00:00+00:00" <= stamp < "2026-07-01T00:00:00+00:00" for stamp in stamps
    )


def test_a_composite_or_within_one_field(orders: DataFrame) -> None:
    status = F.col("order_items.status")
    frame = (
        orders.select("order_items.status")
        .filter((status == "cancelled") | (status == "returned"))
        .to_pandas()
    )

    assert set(frame["order_items.status"]) == {"cancelled", "returned"}


def test_a_negated_filter(orders: DataFrame) -> None:
    frame = (
        orders.select("order_items.status")
        .filter(~(F.col("order_items.status") == "complete"))
        .to_pandas()
    )

    assert "complete" not in set(frame["order_items.status"])


def test_reading_a_bare_view_uses_table_instead_of_the_topic(
    session: OmniSession, handler: FakeOmniAPI
) -> None:
    frame = session.read.view(BENCH_MODEL_NAME, "order_items").select("order_items.id").limit(2)
    frame.collect()
    query = run_requests(handler)[-1].query

    assert query is not None
    assert query["table"] == "order_items"
    assert "join_paths_from_topic_name" not in query


# --------------------------------------------------------------------------------------
# Schema and explain
# --------------------------------------------------------------------------------------


def test_schema_comes_from_a_plan_only_round_trip(orders: DataFrame, handler: FakeOmniAPI) -> None:
    schema = orders.select(
        "users.state", "users.is_business", "order_items.created_at", "order_items.sale_price"
    ).schema

    assert schema.names == (
        "users.state",
        "users.is_business",
        "order_items.created_at",
        "order_items.sale_price",
    )
    assert [f.data_type for f in schema.fields] == [
        OmniDataType.STRING,
        OmniDataType.BOOLEAN,
        OmniDataType.TIMESTAMP,
        OmniDataType.NUMBER,
    ]
    submitted = run_requests(handler)
    assert len(submitted) == 1
    assert submitted[0].body["planOnly"] is True, "df.schema plans; it never executes"


def test_explain_describes_the_pushed_down_query(orders: DataFrame) -> None:
    text = (
        orders.select("users.state", "order_items.status")
        .filter(F.col("users.state") == "California")
        .limit(5)
        .explain()
    )

    assert "Remote [tier 1 · semantic → POST /api/v1/query/run]" in text
    assert f"topic: {BENCH_TOPIC_NAME}   model: {BENCH_MODEL_NAME}" in text
    assert "fields: [users.state, order_items.status]" in text
    assert "filters: users.state = 'California'" in text
    assert "limit: 5" in text
    assert "(none — fully pushed down)" in text


def test_explain_analyze_appends_the_servers_sql(orders: DataFrame, handler: FakeOmniAPI) -> None:
    text = orders.select("users.state").limit(2).explain(analyze=True)

    assert "Analyzed [planOnly round trip]" in text
    assert "SELECT" in text
    assert run_requests(handler)[-1].body["planOnly"] is True


def test_explain_analyze_says_so_when_the_sql_is_redacted() -> None:
    """Without VIEW_SQL the server blanks display_sql; explain must not pretend there is none."""
    fake = FakeOmniAPI(redact_sql=True)
    session = build_session(fake)
    frame = session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME).select("users.state")

    text = frame.limit(2).explain(analyze=True)

    assert "VIEW_SQL" in text
    assert "SELECT" not in text
    fake.close()


# --------------------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------------------


def test_a_full_page_warns_and_a_partial_one_does_not(
    orders: DataFrame, recwarn: pytest.WarningsRecorder
) -> None:
    with pytest.warns(TruncationWarning):
        orders.select("order_items.id").limit(3).collect()

    recwarn.clear()
    orders.select("users.state").filter(F.col("users.state") == "California").collect()

    assert [w for w in recwarn if issubclass(w.category, TruncationWarning)] == []


# --------------------------------------------------------------------------------------
# The wait loop
# --------------------------------------------------------------------------------------


def test_a_slow_job_is_polled_through_the_whole_stack() -> None:
    fake = FakeOmniAPI(slow_job_polls=3)
    session = build_session(fake)
    frame = session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME).select("order_items.id")

    table = frame.limit(4).collect()

    assert table.num_rows == 4
    assert fake.paths.count("GET /api/v1/query/wait") == 3, "the run window expired three times"
    fake.close()


# --------------------------------------------------------------------------------------
# Permission failures
# --------------------------------------------------------------------------------------


def test_a_disabled_feature_flag_reaches_the_user_as_an_admin_task() -> None:
    fake = FakeOmniAPI(feature_flag_off=True)
    session = build_session(fake)
    frame = session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME).select("users.state")

    with pytest.raises(FeatureFlagError) as caught:
        frame.limit(1).collect()

    message = str(caught.value)
    assert "admin" in message
    assert "query-api" in message
    fake.close()


def test_missing_query_topics_is_reported_as_a_model_permission() -> None:
    fake = FakeOmniAPI(permissions=("QUERY_FULL_MODEL", "VIEW_SQL"))
    session = build_session(fake)
    frame = session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME).select("users.state")

    with pytest.raises(ModelPermissionError) as caught:
        frame.limit(1).collect()

    assert caught.value.permission == "QUERY_TOPICS"
    assert "admin" in str(caught.value)
    fake.close()


def test_a_bare_view_query_names_query_full_model_in_its_403() -> None:
    """``read.view`` needs QUERY_FULL_MODEL, and its 403 says so rather than naming topics."""
    fake = FakeOmniAPI(permissions=("VIEW_SQL",))
    session = build_session(fake)
    frame = session.read.view(BENCH_MODEL_NAME, "order_items").select("order_items.id")

    with pytest.raises(ModelPermissionError) as caught:
        frame.limit(1).collect()

    assert caught.value.permission == "QUERY_FULL_MODEL"
    assert "QUERY_FULL_MODEL" in str(caught.value)
    fake.close()


# --------------------------------------------------------------------------------------
# Envelope invariants
# --------------------------------------------------------------------------------------


def test_every_envelope_carries_version_9_and_an_explicit_limit(
    orders: DataFrame, handler: FakeOmniAPI
) -> None:
    orders.select("users.state").filter(F.col("users.state") == "Ohio").collect()
    orders.select("users.state").limit(2).collect()
    orders.select("users.state").limit(None).collect()
    assert orders.select("users.state").schema.names == ("users.state",)

    limits = []
    for request in run_requests(handler):
        query = request.query
        assert query is not None
        assert query["version"] == 9
        assert "limit" in query, "omniframes always sends an explicit limit"
        limits.append(query["limit"])

    assert limits == [DEFAULT_FETCH_LIMIT, 2, None, DEFAULT_FETCH_LIMIT]


def test_branch_id_is_sent_only_when_the_session_has_a_branch(handler: FakeOmniAPI) -> None:
    plain = build_session(handler)
    plain.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME).select("users.state").limit(1).collect()
    assert "branchId" not in run_requests(handler)[-1].body

    branched = build_session(handler, branch=BRANCH_ID)
    branched.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME).select("users.state").limit(1).collect()
    last = run_requests(handler)[-1]

    assert last.body["branchId"] == BRANCH_ID
    assert last.query is not None
    assert "branchId" not in last.query, "branchId inside query is a hard 400"


def test_the_api_key_never_reaches_the_request_log_or_a_repr(
    orders: DataFrame, handler: FakeOmniAPI, session: OmniSession
) -> None:
    orders.select("users.state").limit(1).collect()
    serialized = json.dumps([vars(request) for request in handler.requests], default=str)

    assert DEFAULT_TOKEN not in serialized
    assert "Authorization" not in serialized
    assert DEFAULT_TOKEN not in repr(session)
