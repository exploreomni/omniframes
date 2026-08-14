"""Tests for the FakeOmniAPI itself.

The fake is the substrate every other lane stands on, so it gets tested the way a real server
would be: with a raw ``httpx.Client`` over ``httpx.MockTransport``, parsing the bytes back with
the production parsers (:mod:`omniframes.transport.ndjson`, :mod:`omniframes.transport.arrow`).
Nothing here imports a transport implementation — that keeps this file honest about what the
*wire* looks like rather than about what a client happens to accept.

Numeric expectations come from ``tests/data/bench/known_answers.json`` (never from a
recomputation inside the test), which is also what the live lane will assert against.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from omniframes.errors import QueryError
from omniframes.transport.arrow import check_missing_fields, decode_result, schema_from_summary
from omniframes.transport.ndjson import (
    REDACTED_ERROR_MESSAGE,
    Footer,
    Header,
    JobStatus,
    StreamAccumulator,
    parse_response,
)
from omniframes.transport.normalize import normalize
from omniframes.types import OmniDataType
from tests.fakes import (
    BENCH_DOCUMENT_ID,
    BENCH_MODEL_ID,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    DOCUMENT_WITHOUT_DASHBOARD,
    NDJSON_CONTENT_TYPE,
    NO_QUERY_GENERATED,
    RUN_QUERY_REFUSAL,
    TOTAL_INDICATOR_COLUMN,
    WORKBOOK_URL_HEADER,
    FakeOmniAPI,
    SavedQuery,
    bench_query,
)

BASE_URL = "https://bench.example.omni.co"
BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "bench"
OTHER_MODEL_ID = "11111111-2222-4333-8444-555555555555"

#: The six governed measures of the bench model (docs/bench_omni_model.md §5).
TOTAL_SALE_PRICE = "order_items.total_sale_price"
ORDER_ITEMS_COUNT = "order_items.count"
TOTAL_QUANTITY = "order_items.total_quantity"
AVERAGE_SALE_PRICE = "order_items.average_sale_price"
USERS_COUNT = "users.count"
PRODUCTS_COUNT = "products.count"

TRAILING_12M = {
    "type": "date",
    "kind": "BETWEEN",
    "left_side": "2025-07-01",
    "right_side": "2026-07-01",
}


# --------------------------------------------------------------------------------------
# Fixtures & helpers
# --------------------------------------------------------------------------------------


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


@pytest.fixture
def client(handler: FakeOmniAPI) -> Iterator[httpx.Client]:
    with make_client(handler) as http_client:
        yield http_client


@pytest.fixture(scope="module")
def known_answers() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((BENCH_DIR / "known_answers.json").read_text("utf-8"))
    answers: dict[str, Any] = payload["answers"]
    return answers


def make_client(handler: FakeOmniAPI, *, token: str | None = DEFAULT_TOKEN) -> httpx.Client:
    headers = {"Authorization": f"Bearer {token}"} if token is not None else {}
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=BASE_URL,
        headers=headers,
    )


def query_body(**overrides: Any) -> dict[str, Any]:
    """A wire query envelope in the fake's scope, with every key omniframes always sends."""
    query: dict[str, Any] = {
        "modelId": BENCH_MODEL_ID,
        "table": "order_items",
        "join_paths_from_topic_name": BENCH_TOPIC_NAME,
        "fields": ["order_items.id"],
        "filters": {},
        "sorts": [],
        "limit": None,
        "offset": 0,
        "pivots": [],
        "calculations": [],
        "fill_fields": [],
        "column_totals": {},
        "row_totals": {},
        "userEditedSQL": "",
        "default_group_by": True,
        "version": 9,
    }
    envelope: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in {"planOnly", "cache", "resultType", "formatResults", "workbookUrl", "userId"}:
            envelope[key] = value
        else:
            query[key] = value
    return {"query": query, **envelope}


def run(client: httpx.Client, **overrides: Any) -> httpx.Response:
    return client.post("/api/v1/query/run", json=query_body(**overrides))


def single_job(response: httpx.Response) -> Any:
    parsed = parse_response(response.content)
    assert len(parsed.jobs) == 1, parsed.lines
    return parsed.jobs[0]


def rows(response: httpx.Response) -> list[dict[str, Any]]:
    job = single_job(response)
    assert job.status is JobStatus.COMPLETE, job.failure_reason
    assert job.result is not None
    return decode_result(job.result).to_pylist()


def answer_rows(answers: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    entry = answers[key]
    return [dict(zip(entry["columns"], row, strict=True)) for row in entry["rows"]]


def one_answer(answers: Mapping[str, Any], key: str) -> dict[str, Any]:
    """The single row of a one-row known answer (``grand_totals``)."""
    only = answer_rows(answers, key)
    assert len(only) == 1
    return only[0]


def utc(literal: str) -> datetime:
    """A known-answer timestamp string as the tz-aware datetime Arrow decodes to."""
    return datetime.fromisoformat(literal)


def aggregation(*keys: str) -> dict[str, Any]:
    """A ``query.column_totals`` payload (CONTRACT_NOTES §3)."""
    return {key: {"type": "aggregation"} for key in keys}


def number_filter(kind: str, *values: str, **extra: Any) -> dict[str, Any]:
    """A ``type: "number"`` filter arm — the values are STRINGS on the wire (§3.1)."""
    return {"type": "number", "kind": kind, "values": list(values), **extra}


def composite(conjunction: str, *arms: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """A ``type: "composite"`` entry: several conditions inside ONE field's filter entry (§3.1)."""
    return {"type": "composite", "conjunction": conjunction, "filters": list(arms), **extra}


#: The three markers that tell the server "run my SQL, do not rewrite it" (§3.4).  Any ONE of
#: them turns ``userEditedSQL`` into a raw-SQL job; with none of them the SQL is ignored.
NO_REWRITE_MARKERS: dict[str, dict[str, Any]] = {
    "rewriteSql": {"rewriteSql": False},
    "parsed": {"parsed": False},
    "dbtMode": {"dbtMode": True},
}

#: A raw-SQL query written against the fake's **warehouse schema**: the three bench tables are
#: registered in DuckDB under their bare names, with no schema qualifier
#: (docs/bench_omni_model.md §1 / :mod:`tests.fakes.sqljobs`).
REVENUE_SQL = (
    "SELECT u.state AS state,\n"
    "       SUM(oi.sale_price) AS total_sale_price,\n"
    "       COUNT(*) AS order_items_count\n"
    "FROM order_items oi\n"
    "LEFT JOIN users u ON u.id = oi.user_id\n"
    "GROUP BY 1"
)

SIDECAR_REVENUE = "total_sale_price__omni_summ"
SIDECAR_COUNT = "order_items_count__omni_summ"


def sql_run(
    client: httpx.Client, sql: str, *, marker: str = "rewriteSql", **overrides: Any
) -> httpx.Response:
    """Run ``sql`` as a raw-SQL job: ``userEditedSQL`` plus one no-rewrite marker (§3.4)."""
    return run(client, userEditedSQL=sql, **NO_REWRITE_MARKERS[marker], **overrides)


def reference(**overrides: Any) -> dict[str, Any]:
    """A ``staticQueryReferences`` entry: a full query object plus snake_case ``model_id`` (§3.5)."""
    payload: dict[str, Any] = query_body(**overrides)["query"]
    payload["model_id"] = BENCH_MODEL_ID
    return payload


# --------------------------------------------------------------------------------------
# NDJSON framing (CONTRACT_NOTES §2.2)
# --------------------------------------------------------------------------------------


def test_run_response_is_text_ndjson(client: httpx.Client) -> None:
    response = run(client)
    assert response.status_code == 200
    assert response.headers["content-type"] == NDJSON_CONTENT_TYPE
    assert "application/x-ndjson" not in response.headers["content-type"]


def test_every_framed_line_is_terminated_by_a_newline(client: httpx.Client) -> None:
    body = run(client).content.decode("utf-8")
    assert body.endswith("\n"), "the contract puts a separator after EVERY framed line"
    lines = body.split("\n")
    assert lines[-1] == "", "a trailing empty chunk is the proof of the final separator"
    for line in lines[:-1]:
        assert line, "no blank lines inside the stream"
        assert isinstance(json.loads(line), dict), "one JSON object per line"


def test_line_order_is_header_then_jobs_then_footer(client: httpx.Client) -> None:
    parsed = parse_response(run(client).content)
    assert isinstance(parsed.lines[0], Header)
    assert isinstance(parsed.lines[-1], Footer)
    assert parsed.header is not None
    assert len(parsed.header.job_ids) == 1
    assert parsed.jobs[0].job_id == parsed.header.job_ids[0]


def test_footer_timed_out_is_the_string_false(client: httpx.Client) -> None:
    raw_footer = json.loads(run(client).content.decode("utf-8").strip().split("\n")[-1])
    assert raw_footer["timed_out"] == "false"
    assert raw_footer["timed_out"] is not False, "timed_out is a STRING on the NDJSON path"
    assert raw_footer["remaining_job_ids"] == []


def test_footer_timed_out_is_true_iff_jobs_remain() -> None:
    fake = FakeOmniAPI(slow_job_polls=1)
    with make_client(fake) as client:
        raw_footer = json.loads(run(client).content.decode("utf-8").strip().split("\n")[-1])
    fake.close()
    assert raw_footer["timed_out"] == "true"
    assert len(raw_footer["remaining_job_ids"]) == 1


def test_wait_response_has_no_header_line() -> None:
    fake = FakeOmniAPI(slow_job_polls=1)
    with make_client(fake) as client:
        parsed = parse_response(run(client).content)
        job_id = parsed.remaining_job_ids[0]
        wait = parse_response(client.get("/api/v1/query/wait", params={"jobIds": job_id}).content)
    fake.close()
    assert wait.header is None, "/query/wait repeats the framing minus the header line"
    assert [job.job_id for job in wait.jobs] == [job_id]


# --------------------------------------------------------------------------------------
# Authentication (CONTRACT_NOTES §1)
# --------------------------------------------------------------------------------------


def test_missing_authorization_header_is_a_400_envelope(handler: FakeOmniAPI) -> None:
    with make_client(handler, token=None) as client:
        response = client.get("/api/v1/whoami")
    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": 400,
            "message": "Bad authorization header, must be formatted as Bearer <token>",
        }
    }


@pytest.mark.parametrize("header", ["", "Bearer", "Bearer ", "Token abc", DEFAULT_TOKEN])
def test_malformed_authorization_header_is_a_400(handler: FakeOmniAPI, header: str) -> None:
    with httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL) as client:
        response = client.get("/api/v1/whoami", headers={"Authorization": header})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == 400


def test_invalid_token_is_403_not_401(handler: FakeOmniAPI) -> None:
    with make_client(handler, token="omni_osk_" + "x" * 56) as client:
        response = client.get("/api/v1/whoami")
    assert response.status_code == 403, "the OpenAPI spec says 401; the server says 403"
    assert response.json() == {"error": {"code": 403, "message": "Invalid bearer token"}}


def test_configured_tokens_are_accepted() -> None:
    fake = FakeOmniAPI(tokens={"omni_osk_" + "a" * 56, "omni_osk_" + "b" * 56})
    with make_client(fake, token="omni_osk_" + "b" * 56) as client:
        assert client.get("/api/v1/whoami").status_code == 200
    with make_client(fake) as client:
        assert client.get("/api/v1/whoami").status_code == 403
    fake.close()


def test_bearer_scheme_is_case_insensitive(handler: FakeOmniAPI) -> None:
    with httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL) as client:
        response = client.get(
            "/api/v1/whoami", headers={"Authorization": f"bearer {DEFAULT_TOKEN}"}
        )
    assert response.status_code == 200


# --------------------------------------------------------------------------------------
# /whoami (CONTRACT_NOTES §1)
# --------------------------------------------------------------------------------------


def test_whoami_shape(client: httpx.Client) -> None:
    payload = client.get("/api/v1/whoami").json()
    assert set(payload["user"]) == {"id", "membershipId"}
    assert payload["keyScope"] == "organization"
    assert payload["rolesByModelTruncated"] is False
    role = payload["rolesByModel"][BENCH_MODEL_ID]
    assert set(role) == {"roleName", "baseRole", "connectionId", "permissions"}
    assert {"QUERY_TOPICS", "QUERY_FULL_MODEL", "VIEW_SQL"} <= set(role["permissions"])


def test_whoami_model_filter_accepts_csv_and_repeats(client: httpx.Client) -> None:
    everything = client.get("/api/v1/whoami").json()["rolesByModel"]
    assert len(everything) > 1

    one = client.get("/api/v1/whoami", params={"modelId": BENCH_MODEL_ID}).json()
    assert list(one["rolesByModel"]) == [BENCH_MODEL_ID]

    other = next(model_id for model_id in everything if model_id != BENCH_MODEL_ID)
    csv = client.get("/api/v1/whoami", params={"modelId": f"{BENCH_MODEL_ID},{other}"}).json()
    assert set(csv["rolesByModel"]) == {BENCH_MODEL_ID, other}


def test_whoami_works_when_the_feature_flag_is_off() -> None:
    """whoami skips the ``query-api`` check, which is what makes it the right preflight."""
    fake = FakeOmniAPI(feature_flag_off=True)
    with make_client(fake) as client:
        assert client.get("/api/v1/whoami").status_code == 200
        assert run(client).status_code == 403
    fake.close()


def test_whoami_reports_configured_scope_and_permissions() -> None:
    fake = FakeOmniAPI(key_scope="user", permissions=("QUERY_TOPICS",))
    with make_client(fake) as client:
        payload = client.get("/api/v1/whoami").json()
    fake.close()
    assert payload["keyScope"] == "user"
    assert payload["rolesByModel"][BENCH_MODEL_ID]["permissions"] == ["QUERY_TOPICS"]


# --------------------------------------------------------------------------------------
# /models (CONTRACT_NOTES §4)
# --------------------------------------------------------------------------------------


def test_models_first_page(client: httpx.Client) -> None:
    payload = client.get("/api/v1/models", params={"pageSize": 2}).json()
    page_info = payload["pageInfo"]
    assert set(page_info) == {"hasNextPage", "nextCursor", "pageSize", "totalRecords"}
    assert page_info["pageSize"] == 2
    assert page_info["hasNextPage"] is True
    assert len(payload["records"]) == 2
    record = payload["records"][0]
    assert record["id"] == BENCH_MODEL_ID
    assert set(record) == {
        "id",
        "name",
        "modelKind",
        "connectionId",
        "baseModelId",
        "createdAt",
        "updatedAt",
        "deletedAt",
    }


def test_models_cursor_walks_every_record_exactly_once(client: httpx.Client) -> None:
    seen: list[str] = []
    params: dict[str, Any] = {"pageSize": 2}
    while True:
        payload = client.get("/api/v1/models", params=params).json()
        seen.extend(record["id"] for record in payload["records"])
        if not payload["pageInfo"]["hasNextPage"]:
            assert payload["pageInfo"]["nextCursor"] is None
            break
        params = {"pageSize": 2, "cursor": payload["pageInfo"]["nextCursor"]}
    assert len(seen) == len(set(seen))
    assert len(seen) == payload["pageInfo"]["totalRecords"]


def test_models_bad_cursor_uses_its_own_error_envelope(client: httpx.Client) -> None:
    response = client.get("/api/v1/models", params={"cursor": "not-a-cursor"})
    assert response.status_code == 400
    assert response.json() == {"error": "Invalid cursor", "success": False}


@pytest.mark.parametrize("page_size", [0, 101, "many"])
def test_models_rejects_out_of_range_page_size(client: httpx.Client, page_size: Any) -> None:
    response = client.get("/api/v1/models", params={"pageSize": page_size})
    assert response.status_code == 400
    assert response.json()["status"] == 400


def test_models_name_filter(client: httpx.Client) -> None:
    payload = client.get("/api/v1/models", params={"name": "ecommerce"}).json()
    assert [record["name"] for record in payload["records"]] == [
        "bench_ecommerce",
        "bench_ecommerce_branch",
    ]


# --------------------------------------------------------------------------------------
# Topic catalog (CONTRACT_NOTES §4)
# --------------------------------------------------------------------------------------


def test_topic_list(client: httpx.Client) -> None:
    payload = client.get(f"/api/v1/models/{BENCH_MODEL_ID}/topic").json()
    assert payload["success"] is True
    assert len(payload["topics"]) == 1
    topic = payload["topics"][0]
    assert topic["name"] == BENCH_TOPIC_NAME
    assert topic["base_view_name"] == "order_items"
    assert topic["hidden"] is False


def test_topic_detail_views_and_relationships(client: httpx.Client) -> None:
    topic = client.get(f"/api/v1/models/{BENCH_MODEL_ID}/topic/{BENCH_TOPIC_NAME}").json()["topic"]
    assert [view["name"] for view in topic["views"]] == ["order_items", "users", "products"]

    order_items = topic["views"][0]
    assert {m["field_name"] for m in order_items["measures"]} == {
        "total_sale_price",
        "count",
        "total_quantity",
        "average_sale_price",
    }
    assert {d["field_name"] for d in order_items["dimensions"]} >= {"status", "sale_price"}

    assert [(r["left_view_name"], r["right_view_name"]) for r in topic["relationships"]] == [
        ("order_items", "users"),
        ("order_items", "products"),
    ]
    assert all(r["join_type"] == "always_left" for r in topic["relationships"])
    assert topic["relationships"][0]["sql"] == "${order_items.user_id} = ${users.id}"


@pytest.mark.parametrize(
    ("field_name", "data_type"),
    [
        ("users.state", "STRING"),
        ("users.age", "NUMBER"),
        ("users.is_business", "BOOLEAN"),
        ("users.created_at", "TIMESTAMP"),
        ("users.lifetime_value", "NUMBER"),
        ("products.introduced_on", "TIMESTAMP"),
        ("order_items.returned", "BOOLEAN"),
        ("order_items.sale_price", "NUMBER"),
        ("order_items.notes", "STRING"),
        ("order_items.total_sale_price", "NUMBER"),
    ],
)
def test_topic_detail_field_data_types(
    client: httpx.Client, field_name: str, data_type: str
) -> None:
    topic = client.get(f"/api/v1/models/{BENCH_MODEL_ID}/topic/{BENCH_TOPIC_NAME}").json()["topic"]
    catalog = {
        field["fully_qualified_name"]: field
        for view in topic["views"]
        for field in view["dimensions"] + view["measures"]
    }
    field = catalog[field_name]
    assert field["data_type"] == data_type
    assert field["view_name"] == field_name.split(".")[0]
    assert field["is_dimension"] is (field["aggregate_type"] is None)


def test_unknown_topic_is_404(client: httpx.Client) -> None:
    response = client.get(f"/api/v1/models/{BENCH_MODEL_ID}/topic/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "Topic nope not found", "status": 404}


def test_unknown_model_is_404(client: httpx.Client) -> None:
    response = client.get(f"/api/v1/models/{OTHER_MODEL_ID}/topic")
    assert response.status_code == 404
    assert response.json()["detail"].startswith("Model ")


# --------------------------------------------------------------------------------------
# Happy path: known answers, decoded from real Arrow
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["complete", "shipped", "processing", "cancelled", "returned"])
def test_row_count_by_status_matches_known_answers(
    client: httpx.Client, known_answers: dict[str, Any], status: str
) -> None:
    """Raw-row scan: the known-answer *counts* are the oracle for how many rows come back."""
    expected = {
        row["status"]: row["order_items_count"]
        for row in answer_rows(known_answers, "count_by_status")
    }[status]
    response = run(
        client,
        filters={"order_items.status": {"type": "string", "kind": "EQUALS", "values": [status]}},
    )
    assert len(rows(response)) == expected


def test_row_count_for_a_joined_dimension_matches_known_answers(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    by_state = {
        row["state"]: row["order_items_count"]
        for row in answer_rows(known_answers, "revenue_by_state")
    }
    response = run(
        client,
        fields=["order_items.id", "users.state"],
        filters={"users.state": {"type": "string", "kind": "EQUALS", "values": ["California"]}},
    )
    returned = rows(response)
    assert len(returned) == by_state["California"]
    assert {row["users.state"] for row in returned} == {"California"}


def test_left_join_keeps_orphan_user_ids(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """The NULL ``users.state`` group is NULL states *plus* the fact table's orphan user_ids."""
    expected = {
        row["state"]: row["order_items_count"]
        for row in answer_rows(known_answers, "revenue_by_state")
    }[None]
    response = run(
        client,
        fields=["order_items.id", "order_items.user_id", "users.id", "users.state"],
        filters={"users.state": {"type": "null"}},
    )
    returned = rows(response)
    assert len(returned) == expected
    assert any(row["users.id"] is None for row in returned), "no orphan rows survived the join"
    assert all(row["users.state"] is None for row in returned)


def test_date_between_matches_the_trailing_twelve_months(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = sum(
        row["order_items_count"]
        for row in answer_rows(known_answers, "monthly_revenue_trailing_12m")
    )
    response = run(
        client,
        filters={
            "order_items.created_at": {
                "type": "date",
                "kind": "BETWEEN",
                "left_side": "2025-07-01",
                "right_side": "2026-07-01",
            }
        },
    )
    assert len(rows(response)) == expected


def test_result_columns_are_keyed_by_requested_field_name(client: httpx.Client) -> None:
    fields = ["users.state", "order_items.sale_price", "order_items.created_at"]
    job = single_job(run(client, fields=fields, limit=5))
    assert job.result is not None
    table = decode_result(job.result)
    assert table.schema.names == fields, "column order follows the requested field order"
    assert table.num_rows == 5


def test_summary_fields_are_the_schema_authority(client: httpx.Client) -> None:
    fields = [
        "users.state",
        "users.age",
        "users.is_business",
        "users.created_at",
        "users.lifetime_value",
        "products.introduced_on",
    ]
    job = single_job(run(client, fields=fields, limit=3))
    assert job.summary is not None
    schema = schema_from_summary(job.summary["fields"])
    assert schema.names == tuple(fields)
    assert [f.data_type for f in schema.fields] == [
        OmniDataType.STRING,
        OmniDataType.NUMBER,
        OmniDataType.BOOLEAN,
        OmniDataType.TIMESTAMP,
        OmniDataType.NUMBER,
        OmniDataType.TIMESTAMP,
    ]
    assert schema["users.created_at"].raw["date_type"] == "timestamp"
    assert schema["products.introduced_on"].raw["date_type"] == "date"


def test_decimal_and_timestamp_columns_survive_the_arrow_round_trip(client: httpx.Client) -> None:
    returned = rows(
        run(client, fields=["order_items.sale_price", "order_items.created_at"], limit=4)
    )
    assert all(isinstance(row["order_items.sale_price"], Decimal) for row in returned)
    assert all(row["order_items.created_at"].tzinfo is not None for row in returned)


def test_summary_carries_display_sql_and_cache_type(client: httpx.Client) -> None:
    job = single_job(run(client, fields=["users.state"], limit=1))
    assert job.summary is not None
    assert job.summary["cache_type"] == "MISS"
    display_sql = job.summary["display_sql"]
    assert display_sql.startswith("SELECT ")
    assert 'LEFT JOIN "users" ON "users"."id" = "order_items"."user_id"' in display_sql
    assert job.cache_metadata == {"cache_type": "MISS", "job_id": job.job_id}
    assert job.raw["query"]["modelId"] == BENCH_MODEL_ID


def test_limit_and_offset(client: httpx.Client) -> None:
    sorts = [{"column_name": "order_items.id", "sort_descending": False, "null_sort": "LAST"}]
    first = rows(run(client, fields=["order_items.id"], sorts=sorts, limit=5))
    shifted = rows(run(client, fields=["order_items.id"], sorts=sorts, limit=5, offset=2))
    assert [row["order_items.id"] for row in first] == [1, 2, 3, 4, 5]
    assert [row["order_items.id"] for row in shifted] == [3, 4, 5, 6, 7]


def test_sort_descending(client: httpx.Client) -> None:
    returned = rows(
        run(
            client,
            fields=["order_items.id"],
            sorts=[{"column_name": "order_items.id", "sort_descending": True}],
            limit=3,
        )
    )
    assert [row["order_items.id"] for row in returned] == [10_000, 9_999, 9_998]


def test_absent_limit_falls_back_to_the_server_default(client: httpx.Client) -> None:
    body = query_body()
    del body["query"]["limit"]
    response = client.post("/api/v1/query/run", json=body)
    assert len(rows(response)) == 1000, "the server substitutes limit: 1000 when the key is absent"


# --------------------------------------------------------------------------------------
# Filters (CONTRACT_NOTES §3.1)
# --------------------------------------------------------------------------------------


def test_number_filter_values_arrive_as_strings(client: httpx.Client) -> None:
    returned = rows(
        run(
            client,
            fields=["order_items.quantity"],
            filters={
                "order_items.quantity": {
                    "type": "number",
                    "kind": "BETWEEN",
                    "values": ["3", "5"],
                }
            },
        )
    )
    assert returned
    assert {row["order_items.quantity"] for row in returned} == {3, 4}, "BETWEEN upper is exclusive"


def test_number_filter_is_inclusive(client: httpx.Client) -> None:
    def quantities(**flt: Any) -> set[int]:
        returned = rows(
            run(
                client,
                fields=["order_items.quantity"],
                filters={
                    "order_items.quantity": {
                        "type": "number",
                        "kind": "GREATER_THAN",
                        "values": ["7"],
                        **flt,
                    }
                },
            )
        )
        return {row["order_items.quantity"] for row in returned}

    assert quantities() == {8}
    assert quantities(is_inclusive=True) == {7, 8}


def test_boolean_filter_is_negative_selects_false(client: httpx.Client) -> None:
    def returned_values(**flt: Any) -> set[bool | None]:
        returned = rows(
            run(
                client,
                fields=["order_items.returned"],
                filters={"order_items.returned": {"type": "boolean", **flt}},
                limit=200,
            )
        )
        return {row["order_items.returned"] for row in returned}

    assert returned_values(is_negative=False) == {True}
    assert returned_values(is_negative=True) == {False}
    assert returned_values() == {True, False, None}, "is_negative absent is a no-op placeholder"


def test_null_filter_both_directions(client: httpx.Client) -> None:
    present = rows(
        run(
            client,
            fields=["order_items.discount"],
            filters={"order_items.discount": {"type": "null", "is_negative": True}},
            limit=50,
        )
    )
    absent = rows(
        run(
            client,
            fields=["order_items.discount"],
            filters={"order_items.discount": {"type": "null"}},
            limit=50,
        )
    )
    assert all(row["order_items.discount"] is not None for row in present)
    assert all(row["order_items.discount"] is None for row in absent)


def test_string_filter_kinds(client: httpx.Client) -> None:
    def states(**flt: Any) -> set[str]:
        returned = rows(
            run(client, fields=["users.state"], filters={"users.state": {"type": "string", **flt}})
        )
        return {row["users.state"] for row in returned}

    assert states(kind="STARTS_WITH", values=["New"]) == {"New Jersey", "New York"}
    assert states(kind="ENDS_WITH", values=["ia"]) >= {"California", "Georgia", "Pennsylvania"}
    assert states(kind="CONTAINS", values=["exa"], case_insensitive=True) == {"Texas"}
    assert states(kind="EQUALS", values=["Ohio", "Texas"]) == {"Ohio", "Texas"}


def test_string_filter_is_negative(client: httpx.Client) -> None:
    returned = rows(
        run(
            client,
            fields=["order_items.status"],
            filters={
                "order_items.status": {
                    "type": "string",
                    "kind": "EQUALS",
                    "values": ["complete"],
                    "is_negative": True,
                }
            },
        )
    )
    assert "complete" not in {row["order_items.status"] for row in returned}


def test_string_is_empty_matches_empty_string_and_null(client: httpx.Client) -> None:
    """IS_EMPTY covers ``''`` *and* NULL — here the NULLs come from the orphan-user LEFT join."""
    returned = rows(
        run(
            client,
            fields=["users.signup_source"],
            filters={"users.signup_source": {"type": "string", "kind": "IS_EMPTY", "values": []}},
        )
    )
    assert returned
    assert {row["users.signup_source"] for row in returned} == {"", None}
    assert "" in {row["users.signup_source"] for row in returned}


@pytest.mark.parametrize(
    ("kind", "side", "literal"),
    [
        ("ON_OR_AFTER", "left_side", "2026-06-01"),
        ("BEFORE", "right_side", "2024-08-01 00:00:00"),
    ],
)
def test_date_filter_open_ended_kinds(
    client: httpx.Client, kind: str, side: str, literal: str
) -> None:
    returned = rows(
        run(
            client,
            fields=["order_items.created_at"],
            filters={"order_items.created_at": {"type": "date", "kind": kind, side: literal}},
        )
    )
    assert returned
    boundary = f"{literal[:10]}T00:00:00+00:00"
    stamps = [row["order_items.created_at"].isoformat() for row in returned]
    if kind == "ON_OR_AFTER":
        assert all(stamp >= boundary for stamp in stamps)
    else:
        assert all(stamp < boundary for stamp in stamps), "BEFORE is exclusive"


def test_date_filter_on_a_date_typed_column(client: httpx.Client) -> None:
    returned = rows(
        run(
            client,
            fields=["products.introduced_on"],
            filters={
                "products.introduced_on": {
                    "type": "date",
                    "kind": "ON_OR_AFTER",
                    "left_side": "2026-01-01",
                }
            },
        )
    )
    assert returned
    assert all(row["products.introduced_on"].isoformat() >= "2026-01-01" for row in returned)


def test_composite_filter_or(client: httpx.Client) -> None:
    returned = rows(
        run(
            client,
            fields=["order_items.status"],
            filters={
                "order_items.status": {
                    "type": "composite",
                    "conjunction": "OR",
                    "filters": [
                        {"type": "string", "kind": "EQUALS", "values": ["cancelled"]},
                        {"type": "string", "kind": "EQUALS", "values": ["returned"]},
                    ],
                }
            },
        )
    )
    assert {row["order_items.status"] for row in returned} == {"cancelled", "returned"}


def test_filters_across_fields_are_anded(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    both = rows(
        run(
            client,
            fields=["order_items.status", "users.state"],
            filters={
                "order_items.status": {"type": "string", "kind": "EQUALS", "values": ["complete"]},
                "users.state": {"type": "string", "kind": "EQUALS", "values": ["California"]},
            },
        )
    )
    by_state = {
        row["state"]: row["order_items_count"]
        for row in answer_rows(known_answers, "revenue_by_state")
    }
    assert 0 < len(both) < by_state["California"]
    assert all(
        row["order_items.status"] == "complete" and row["users.state"] == "California"
        for row in both
    )


def test_relative_date_literals_are_refused_rather_than_guessed(client: httpx.Client) -> None:
    job = single_job(
        run(
            client,
            filters={
                "order_items.created_at": {
                    "type": "date",
                    "kind": "ON_OR_AFTER",
                    "left_side": "30 days ago",
                }
            },
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "absolute date literal" in (job.error_message or "")


# --------------------------------------------------------------------------------------
# Governed measures (docs/bench_omni_model.md §5; grouping semantics per DESIGN.md §2)
# --------------------------------------------------------------------------------------


def test_measures_alone_are_one_aggregate_row(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = one_answer(known_answers, "grand_totals")
    returned = rows(
        run(
            client,
            fields=[
                TOTAL_SALE_PRICE,
                ORDER_ITEMS_COUNT,
                USERS_COUNT,
                TOTAL_QUANTITY,
                PRODUCTS_COUNT,
            ],
        )
    )
    assert len(returned) == 1, "measures with no dimension aggregate the whole table"
    row = returned[0]
    assert str(row[TOTAL_SALE_PRICE]) == expected["total_sale_price"]
    assert row[ORDER_ITEMS_COUNT] == expected["order_items_count"]
    assert row[USERS_COUNT] == expected["distinct_buyers"], "users.count is COUNT(DISTINCT id)"
    assert row[USERS_COUNT] != expected["distinct_user_ids"], "orphan user_ids are not buyers"
    assert row[TOTAL_QUANTITY] == expected["total_quantity"]
    assert row[PRODUCTS_COUNT] == expected["distinct_products"]


def test_a_dimension_next_to_a_measure_is_the_group_by(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """Selecting dimensions and measures together IS the group-by — no explicit GROUP BY key."""
    expected = {row["state"]: row for row in answer_rows(known_answers, "revenue_by_state")}
    returned = rows(run(client, fields=["users.state", TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT]))
    assert len(returned) == len(expected)
    for row in returned:
        answer = expected[row["users.state"]]
        assert str(row[TOTAL_SALE_PRICE]) == answer["total_sale_price"]
        assert row[ORDER_ITEMS_COUNT] == answer["order_items_count"]
    assert None in {row["users.state"] for row in returned}, "the NULL group key survives"


def test_measures_from_two_views_in_one_query(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = {
        row["state"]: row for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
    }
    returned = rows(run(client, fields=["users.state", TOTAL_SALE_PRICE, USERS_COUNT]))
    assert len(returned) == len(expected)
    for row in returned:
        answer = expected[row["users.state"]]
        assert str(row[TOTAL_SALE_PRICE]) == answer["total_sale_price"]
        assert row[USERS_COUNT] == answer["distinct_buyers"]


def test_average_measure_matches_known_answers(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = {
        row["category"]: row for row in answer_rows(known_answers, "average_sale_price_by_category")
    }
    returned = rows(
        run(client, fields=["products.category", AVERAGE_SALE_PRICE, ORDER_ITEMS_COUNT])
    )
    assert len(returned) == len(expected)
    for row in returned:
        answer = expected[row["products.category"]]
        value = row[AVERAGE_SALE_PRICE]
        assert isinstance(value, Decimal), "an average must not degrade to a float on the wire"
        assert str(value) == answer["average_sale_price"]
        assert row[ORDER_ITEMS_COUNT] == answer["order_items_count"]


def test_measures_aggregate_the_filtered_rows(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = {row["status"]: row for row in answer_rows(known_answers, "count_by_status")}
    returned = rows(
        run(
            client,
            fields=[ORDER_ITEMS_COUNT, TOTAL_SALE_PRICE],
            filters={
                "order_items.status": {
                    "type": "string",
                    "kind": "EQUALS",
                    "values": ["complete", "shipped"],
                }
            },
        )
    )
    assert len(returned) == 1
    assert returned[0][ORDER_ITEMS_COUNT] == sum(
        expected[status]["order_items_count"] for status in ("complete", "shipped")
    )
    assert returned[0][TOTAL_SALE_PRICE] == sum(
        (Decimal(expected[status]["total_sale_price"]) for status in ("complete", "shipped")),
        Decimal(0),
    )


def test_measure_summary_metadata(client: httpx.Client) -> None:
    job = single_job(run(client, fields=["users.state", TOTAL_SALE_PRICE, USERS_COUNT]))
    assert job.summary is not None
    fields = job.summary["fields"]
    assert list(fields) == ["users.state", TOTAL_SALE_PRICE, USERS_COUNT]

    dimension = fields["users.state"]
    assert dimension["is_dimension"] is True
    assert dimension["aggregate_type"] is None

    measure = fields[TOTAL_SALE_PRICE]
    assert measure["is_dimension"] is False
    assert measure["aggregate_type"] == "sum"
    assert measure["data_type"] == "NUMBER"
    assert measure["view_name"] == "order_items"
    assert measure["fully_qualified_name"] == TOTAL_SALE_PRICE
    assert fields[USERS_COUNT]["aggregate_type"] == "count_distinct"
    assert schema_from_summary(fields)[TOTAL_SALE_PRICE].data_type is OmniDataType.NUMBER


def test_measures_are_sortable(client: httpx.Client, known_answers: dict[str, Any]) -> None:
    ranked = sorted(
        answer_rows(known_answers, "revenue_by_state"),
        key=lambda row: Decimal(row["total_sale_price"]),
        reverse=True,
    )
    returned = rows(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            sorts=[{"column_name": TOTAL_SALE_PRICE, "sort_descending": True}],
            limit=3,
        )
    )
    assert [row["users.state"] for row in returned] == [row["state"] for row in ranked[:3]]
    assert [str(row[TOTAL_SALE_PRICE]) for row in returned] == [
        row["total_sale_price"] for row in ranked[:3]
    ]


def test_sorting_by_a_field_outside_an_aggregate_query_is_refused(client: httpx.Client) -> None:
    """In an aggregate query an unselected sort key has no value — refuse instead of guessing."""
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            sorts=[{"column_name": "order_items.id", "sort_descending": False}],
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "query.fields" in (job.error_message or "")


def test_plan_only_reports_the_measure_schema(client: httpx.Client) -> None:
    job = single_job(
        run(client, fields=["users.state", TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT], planOnly=True)
    )
    assert job.status is JobStatus.PLANNED
    assert job.result is None
    assert job.summary is not None
    schema = schema_from_summary(job.summary["fields"])
    assert schema.names == ("users.state", TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT)
    assert [f.data_type for f in schema.fields] == [
        OmniDataType.STRING,
        OmniDataType.NUMBER,
        OmniDataType.NUMBER,
    ]


# --------------------------------------------------------------------------------------
# Time grains (CONTRACT_NOTES §3.2) and the grain-filter rule (§3.1)
# --------------------------------------------------------------------------------------

MONTH = "order_items.created_at[month]"
MONTH_NUM = "order_items.created_at[month_num]"


def test_monthly_revenue_via_a_grain_and_a_bare_field_filter(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """The grain-filter rule, direction 1: a *date* filter keys on the BARE field name."""
    expected = answer_rows(known_answers, "monthly_revenue_trailing_12m")
    returned = rows(
        run(
            client,
            fields=[MONTH, TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT],
            filters={"order_items.created_at": TRAILING_12M},
            sorts=[{"column_name": MONTH, "sort_descending": False}],
        )
    )
    assert len(returned) == len(expected) == 12
    for row, answer in zip(returned, expected, strict=True):
        month = row[MONTH]
        assert month == utc(answer["month"])
        assert month.tzinfo is not None, "timestamps come back tz-aware UTC"
        assert str(row[TOTAL_SALE_PRICE]) == answer["total_sale_price"]
        assert row[ORDER_ITEMS_COUNT] == answer["order_items_count"]


def test_revenue_by_category_and_month(client: httpx.Client, known_answers: dict[str, Any]) -> None:
    """Two group keys — a joined dimension and a grain — plus two measures."""
    expected = {
        (row["category"], utc(row["month"])): row
        for row in answer_rows(known_answers, "revenue_by_category_month")
    }
    job = single_job(
        run(
            client,
            fields=["products.category", MONTH, TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT],
            filters={"order_items.created_at": TRAILING_12M},
        )
    )
    assert job.summary is not None
    assert "GROUP BY" in job.summary["display_sql"]
    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert len(returned) == len(expected) == 132
    for row in returned:
        answer = expected[(row["products.category"], row[MONTH])]
        assert str(row[TOTAL_SALE_PRICE]) == answer["total_sale_price"]
        assert row[ORDER_ITEMS_COUNT] == answer["order_items_count"]


def test_distinct_buyers_by_month(client: httpx.Client, known_answers: dict[str, Any]) -> None:
    expected = answer_rows(known_answers, "distinct_buyers_by_month")
    returned = rows(
        run(
            client,
            fields=[MONTH, USERS_COUNT],
            sorts=[{"column_name": MONTH, "sort_descending": False}],
        )
    )
    assert len(returned) == len(expected) == 24
    for row, answer in zip(returned, expected, strict=True):
        assert row[MONTH] == utc(answer["month"])
        assert row[USERS_COUNT] == answer["distinct_buyers"]


def test_numeric_grain_filters_on_the_bracketed_name(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """The grain-filter rule, direction 2: a *number* filter keys on the BRACKETED grain name.

    Both directions in one query: the trailing-12-month window arrives on the bare timestamp
    while the month-of-year restriction arrives on ``created_at[month_num]``, so the single row
    that survives must be the July 2025 row of ``monthly_revenue_trailing_12m``.
    """
    july = next(
        row
        for row in answer_rows(known_answers, "monthly_revenue_trailing_12m")
        if row["month"].startswith("2025-07")
    )
    returned = rows(
        run(
            client,
            fields=[MONTH_NUM, TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT],
            filters={
                "order_items.created_at": TRAILING_12M,
                MONTH_NUM: {"type": "number", "kind": "EQUALS", "values": ["7"]},
            },
        )
    )
    assert len(returned) == 1
    assert returned[0][MONTH_NUM] == 7
    assert str(returned[0][TOTAL_SALE_PRICE]) == july["total_sale_price"]
    assert returned[0][ORDER_ITEMS_COUNT] == july["order_items_count"]


@pytest.mark.parametrize(
    ("grain", "data_type", "check"),
    [
        ("year", "TIMESTAMP", lambda v: v == utc("2026-01-01T00:00:00+00:00")),
        ("quarter", "TIMESTAMP", lambda v: v == utc("2026-04-01T00:00:00+00:00")),
        ("month", "TIMESTAMP", lambda v: v == utc("2026-06-01T00:00:00+00:00")),
        ("week", "TIMESTAMP", lambda v: v.weekday() == 0),
        ("date", "TIMESTAMP", lambda v: v == utc("2026-06-30T00:00:00+00:00")),
        ("day_of_week_num", "NUMBER", lambda v: 0 <= v <= 6),
        ("day_of_month", "NUMBER", lambda v: v == 30),
        ("month_num", "NUMBER", lambda v: v == 6),
        ("hour_of_day", "NUMBER", lambda v: 0 <= v <= 23),
        ("week_of_year", "NUMBER", lambda v: 1 <= v <= 53),
        ("quarter_of_year", "NUMBER", lambda v: v == 2),
        ("day_of_year", "NUMBER", lambda v: v == 181),
        ("month_name", "STRING", lambda v: v == "June"),
        ("day_of_week_name", "STRING", lambda v: isinstance(v, str)),
    ],
)
def test_every_grain_projects_its_own_type(
    client: httpx.Client, grain: str, data_type: str, check: Any
) -> None:
    """One row of the anchor day (2026-06-30) seen through each grain."""
    name = f"order_items.created_at[{grain}]"
    job = single_job(
        run(
            client,
            fields=[name],
            filters={
                "order_items.created_at": {
                    "type": "date",
                    "kind": "ON_OR_AFTER",
                    "left_side": "2026-06-30",
                }
            },
            sorts=[{"column_name": name, "sort_descending": True}],
            limit=1,
        )
    )
    assert job.summary is not None
    assert job.summary["missing_fields"] == []
    assert list(job.summary["fields"]) == [name], "the summary key is the exact requested name"
    assert job.summary["fields"][name]["data_type"] == data_type
    assert job.summary["fields"][name]["field_name"] == f"created_at[{grain}]"
    assert job.result is not None
    table = decode_result(job.result)
    assert table.schema.names == [name]
    assert check(table.to_pylist()[0][name])


def test_grain_names_are_case_insensitive_but_the_key_is_as_requested(
    client: httpx.Client,
) -> None:
    requested = "order_items.created_at[MONTH]"
    job = single_job(run(client, fields=[requested, ORDER_ITEMS_COUNT]))
    assert job.summary is not None
    assert job.summary["missing_fields"] == []
    assert list(job.summary["fields"]) == [requested, ORDER_ITEMS_COUNT]
    assert job.result is not None
    assert decode_result(job.result).schema.names == [requested, ORDER_ITEMS_COUNT]


def test_a_grain_keeps_a_date_column_date_shaped(client: httpx.Client) -> None:
    """``products.introduced_on`` is a DATE; its grains must not silently become timestamps."""
    name = "products.introduced_on[year]"
    job = single_job(run(client, fields=[name, PRODUCTS_COUNT]))
    assert job.summary is not None
    assert job.summary["fields"][name]["data_type"] == "TIMESTAMP", "Omni has no DATE data_type"
    assert job.summary["fields"][name]["date_type"] == "date"
    assert job.result is not None
    values = [row[name] for row in decode_result(job.result).to_pylist()]
    assert values
    assert all(not isinstance(value, datetime) for value in values), "a date stays a date"
    assert all(value.month == 1 and value.day == 1 for value in values)


def test_a_grained_dimension_groups_and_sorts_like_any_other(client: httpx.Client) -> None:
    name = "order_items.created_at[quarter]"
    returned = rows(
        run(
            client,
            fields=[name, ORDER_ITEMS_COUNT],
            sorts=[{"column_name": name, "sort_descending": True}],
        )
    )
    quarters = [row[name] for row in returned]
    assert quarters == sorted(quarters, reverse=True)
    assert sum(row[ORDER_ITEMS_COUNT] for row in returned) == 10_000
    assert len(set(quarters)) == len(quarters), "a grain is a group key, so it is distinct"


@pytest.mark.parametrize(
    "name",
    [
        "order_items.created_at[fortnight]",
        "order_items.created_at[]",
        "products.introduced_on[hour_of_day]",
        "users.state[month]",
        "order_items.count[month]",
        "users.nope[month]",
    ],
)
def test_an_unusable_grain_lands_in_missing_fields(client: httpx.Client, name: str) -> None:
    """Invalid grain → ``summary.missing_fields``, never a hard error (§3.2)."""
    job = single_job(run(client, fields=["users.state", name], limit=2))
    assert job.status is JobStatus.COMPLETE
    assert job.summary is not None
    assert job.summary["missing_fields"] == [name]
    assert job.result is not None
    assert decode_result(job.result).schema.names == ["users.state"]


def test_a_grained_dimension_survives_planonly(client: httpx.Client) -> None:
    planned = single_job(run(client, fields=[MONTH, ORDER_ITEMS_COUNT], planOnly=True))
    executed = single_job(run(client, fields=[MONTH, ORDER_ITEMS_COUNT], limit=2))
    assert planned.summary is not None
    assert executed.summary is not None
    assert schema_from_summary(planned.summary["fields"]) == schema_from_summary(
        executed.summary["fields"]
    )


# --------------------------------------------------------------------------------------
# Measure filters compile to HAVING (CONTRACT_NOTES §3.1, "Measure filters → HAVING")
# --------------------------------------------------------------------------------------

#: The revenue threshold every HAVING test below is written around: it keeps 9 of the 21
#: ``revenue_by_state`` groups, the NULL-state group among them.
REVENUE_THRESHOLD = "50000"


def surviving_states(answers: Mapping[str, Any], keep: Any) -> dict[str | None, dict[str, Any]]:
    """The ``revenue_by_state`` rows a HAVING over ``total_sale_price`` must leave standing.

    The expectation is *computed from the known answers*, never from a recomputation over the
    parquet: the oracle stays the checked-in file even when the predicate lives in the test.
    """
    kept = {
        row["state"]: row
        for row in answer_rows(answers, "revenue_by_state")
        if keep(Decimal(row["total_sale_price"]))
    }
    assert 0 < len(kept) < 21, "a HAVING fixture that keeps everything or nothing proves nothing"
    return kept


def test_a_measure_filter_becomes_a_having_over_the_aggregate(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """The headline behavior: the filter is translated against the measure's aggregate SQL."""
    expected = surviving_states(known_answers, lambda revenue: revenue > Decimal(REVENUE_THRESHOLD))
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            filters={TOTAL_SALE_PRICE: number_filter("GREATER_THAN", REVENUE_THRESHOLD)},
        )
    )
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.summary is not None
    display_sql = job.summary["display_sql"]
    assert 'HAVING SUM("order_items"."sale_price") > 50000' in display_sql
    assert display_sql.index("GROUP BY") < display_sql.index("HAVING"), "HAVING is post-GROUP BY"

    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert len(returned) == len(expected)
    for row in returned:
        answer = expected[row["users.state"]]
        assert str(row[TOTAL_SALE_PRICE]) == answer["total_sale_price"]
    assert None in expected, "the NULL group is filtered on its aggregate like any other group"


@pytest.mark.parametrize(
    ("arm", "keep"),
    [
        (
            number_filter("GREATER_THAN", REVENUE_THRESHOLD),
            lambda revenue: revenue > Decimal(REVENUE_THRESHOLD),
        ),
        (
            number_filter("GREATER_THAN", "50150.60", is_inclusive=True),
            lambda revenue: revenue >= Decimal("50150.60"),
        ),
        (number_filter("LESS_THAN", "20000"), lambda revenue: revenue < Decimal("20000")),
        (
            number_filter("LESS_THAN", "18887.96", is_inclusive=True),
            lambda revenue: revenue <= Decimal("18887.96"),
        ),
        (
            number_filter("BETWEEN", "20000", REVENUE_THRESHOLD),
            lambda revenue: Decimal("20000") <= revenue < Decimal(REVENUE_THRESHOLD),
        ),
        (
            number_filter("EQUALS", "128597.44"),
            lambda revenue: revenue == Decimal("128597.44"),
        ),
    ],
)
def test_every_number_kind_works_on_the_having_side(
    client: httpx.Client, known_answers: dict[str, Any], arm: dict[str, Any], keep: Any
) -> None:
    """The number kinds mean the same thing over an aggregate as they do over a column (§3.1)."""
    expected = surviving_states(known_answers, keep)
    returned = rows(
        run(client, fields=["users.state", TOTAL_SALE_PRICE], filters={TOTAL_SALE_PRICE: arm})
    )
    assert {row["users.state"] for row in returned} == set(expected)
    for row in returned:
        assert str(row[TOTAL_SALE_PRICE]) == expected[row["users.state"]]["total_sale_price"]


def test_a_filtered_measure_that_is_not_selected_is_projected_away(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """Force-added to the aggregate for the HAVING, then projected away — never a result column."""
    expected = surviving_states(known_answers, lambda revenue: revenue > Decimal(REVENUE_THRESHOLD))
    job = single_job(
        run(
            client,
            fields=["users.state", ORDER_ITEMS_COUNT],
            filters={TOTAL_SALE_PRICE: number_filter("GREATER_THAN", REVENUE_THRESHOLD)},
        )
    )
    assert job.summary is not None
    assert 'SUM("order_items"."sale_price")' in job.summary["display_sql"], (
        "the unselected measure is still aggregated, for the HAVING"
    )
    assert list(job.summary["fields"]) == ["users.state", ORDER_ITEMS_COUNT]
    assert TOTAL_SALE_PRICE not in job.summary["fields"]

    assert job.result is not None
    table = decode_result(job.result)
    assert table.schema.names == ["users.state", ORDER_ITEMS_COUNT], (
        "a filtered-but-unselected measure must not appear as a result column"
    )
    returned = table.to_pylist()
    assert len(returned) == len(expected)
    for row in returned:
        assert row[ORDER_ITEMS_COUNT] == expected[row["users.state"]]["order_items_count"]


@pytest.mark.parametrize(
    ("entry", "keep"),
    [
        (
            composite(
                "AND",
                number_filter("GREATER_THAN", REVENUE_THRESHOLD),
                number_filter("LESS_THAN", "80000"),
            ),
            lambda revenue: Decimal(REVENUE_THRESHOLD) < revenue < Decimal("80000"),
        ),
        (
            composite(
                "OR",
                number_filter("LESS_THAN", "20000"),
                number_filter("GREATER_THAN", "100000"),
            ),
            lambda revenue: revenue < Decimal("20000") or revenue > Decimal("100000"),
        ),
        (
            composite(
                "AND",
                number_filter("GREATER_THAN", REVENUE_THRESHOLD),
                number_filter("LESS_THAN", "80000"),
                is_negative=True,
            ),
            lambda revenue: not Decimal(REVENUE_THRESHOLD) < revenue < Decimal("80000"),
        ),
    ],
)
def test_composite_measure_filters_stay_inside_one_entry(
    client: httpx.Client, known_answers: dict[str, Any], entry: dict[str, Any], keep: Any
) -> None:
    """Several conditions on one measure arrive as a composite: the server keys filters by field."""
    expected = surviving_states(known_answers, keep)
    returned = rows(
        run(client, fields=["users.state", TOTAL_SALE_PRICE], filters={TOTAL_SALE_PRICE: entry})
    )
    assert {row["users.state"] for row in returned} == set(expected)
    for row in returned:
        assert str(row[TOTAL_SALE_PRICE]) == expected[row["users.state"]]["total_sale_price"]


def test_measure_filter_is_negative_inverts_the_having(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = surviving_states(
        known_answers, lambda revenue: not revenue > Decimal(REVENUE_THRESHOLD)
    )
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            filters={
                TOTAL_SALE_PRICE: number_filter("GREATER_THAN", REVENUE_THRESHOLD, is_negative=True)
            },
        )
    )
    assert job.summary is not None
    assert 'HAVING NOT (SUM("order_items"."sale_price") > 50000)' in job.summary["display_sql"]
    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert {row["users.state"] for row in returned} == set(expected)
    for row in returned:
        assert str(row[TOTAL_SALE_PRICE]) == expected[row["users.state"]]["total_sale_price"]


def test_a_measure_filter_forces_the_group_by_with_no_measure_selected(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """No aggregation context in ``fields`` — the measure filter alone makes it an aggregate."""
    expected = surviving_states(known_answers, lambda revenue: revenue > Decimal(REVENUE_THRESHOLD))
    job = single_job(
        run(
            client,
            fields=["users.state"],
            filters={TOTAL_SALE_PRICE: number_filter("GREATER_THAN", REVENUE_THRESHOLD)},
        )
    )
    assert job.summary is not None
    display_sql = job.summary["display_sql"]
    assert "GROUP BY" in display_sql, "the server force-groups a query with a measure filter"
    assert display_sql.index("GROUP BY") < display_sql.index("HAVING")

    assert job.result is not None
    table = decode_result(job.result)
    assert table.schema.names == ["users.state"]
    states = [row["users.state"] for row in table.to_pylist()]
    assert set(states) == set(expected)
    assert len(states) == len(expected), "one row per surviving group, not one row per fact row"


def test_a_dimension_filter_and_a_measure_filter_land_on_opposite_sides(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """WHERE narrows the rows, HAVING narrows the groups the narrowed rows produced."""
    threshold = Decimal("45000")
    expected = [
        row
        for row in answer_rows(known_answers, "monthly_revenue_trailing_12m")
        if Decimal(row["total_sale_price"]) > threshold
    ]
    assert 0 < len(expected) < 12
    job = single_job(
        run(
            client,
            fields=[MONTH, TOTAL_SALE_PRICE],
            filters={
                "order_items.created_at": TRAILING_12M,
                TOTAL_SALE_PRICE: number_filter("GREATER_THAN", str(threshold)),
            },
            sorts=[{"column_name": MONTH, "sort_descending": False}],
        )
    )
    assert job.summary is not None
    display_sql = job.summary["display_sql"]
    assert display_sql.index("WHERE") < display_sql.index("GROUP BY") < display_sql.index("HAVING")
    assert "created_at" in display_sql[display_sql.index("WHERE") : display_sql.index("GROUP BY")]

    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert len(returned) == len(expected)
    for row, answer in zip(returned, expected, strict=True):
        assert row[MONTH] == utc(answer["month"])
        assert str(row[TOTAL_SALE_PRICE]) == answer["total_sale_price"]


def test_a_measure_filter_joins_the_view_it_names(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """``users.count`` is only reachable through the join, filtered or selected."""
    expected = {
        row["state"]: row
        for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
        if row["distinct_buyers"] > 25
    }
    assert 0 < len(expected) < 21
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            filters={USERS_COUNT: number_filter("GREATER_THAN", "25")},
        )
    )
    assert job.summary is not None
    assert 'LEFT JOIN "users"' in job.summary["display_sql"]
    assert 'HAVING COUNT(DISTINCT "users"."id") > 25' in job.summary["display_sql"]
    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert {row["users.state"] for row in returned} == set(expected)
    for row in returned:
        assert str(row[TOTAL_SALE_PRICE]) == expected[row["users.state"]]["total_sale_price"]


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"type": "string", "kind": "CONTAINS", "values": ["1"]}, "'string'"),
        ({"type": "null"}, "'null'"),
        ({"type": "boolean", "is_negative": True}, "'boolean'"),
        (composite("OR", {"type": "null"}), "'null'"),
        (number_filter("SQL_LIKE", "1"), "'SQL_LIKE'"),
    ],
)
def test_an_unimplemented_measure_filter_kind_is_refused_by_name(
    client: httpx.Client, entry: dict[str, Any], expected: str
) -> None:
    """What the fake does not compile it refuses loudly, naming the kind it was handed."""
    job = single_job(
        run(client, fields=["users.state", TOTAL_SALE_PRICE], filters={TOTAL_SALE_PRICE: entry})
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    message = job.error_message or ""
    assert expected in message
    assert TOTAL_SALE_PRICE in message


def test_column_totals_next_to_a_measure_filter_are_refused(client: httpx.Client) -> None:
    """What a total over a HAVING-restricted group set aggregates is not pinned — so: refuse."""
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            filters={TOTAL_SALE_PRICE: number_filter("GREATER_THAN", REVENUE_THRESHOLD)},
            column_totals=aggregation("::total::"),
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "column_totals" in (job.error_message or "")


# --------------------------------------------------------------------------------------
# Column totals (CONTRACT_NOTES §2.7)
# --------------------------------------------------------------------------------------


def test_column_totals_append_a_grand_total_row(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = one_answer(known_answers, "grand_totals")
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT, USERS_COUNT],
            column_totals=aggregation("::total::"),
        )
    )
    assert job.status is JobStatus.COMPLETE
    assert job.result is not None
    table = decode_result(job.result)
    assert TOTAL_INDICATOR_COLUMN in table.schema.names

    returned = table.to_pylist()
    data = [row for row in returned if row[TOTAL_INDICATOR_COLUMN] is None]
    totals = [row for row in returned if row[TOTAL_INDICATOR_COLUMN] is not None]
    assert len(data) == 21, "the data rows are untouched"
    assert len(totals) == 1

    total = totals[0]
    assert total[TOTAL_INDICATOR_COLUMN] == "::total::", "the grand-total indicator value (§2.7)"
    assert total["users.state"] is None, "dimension columns are NULL on a totals row"
    assert str(total[TOTAL_SALE_PRICE]) == expected["total_sale_price"]
    assert total[ORDER_ITEMS_COUNT] == expected["order_items_count"]
    assert total[USERS_COUNT] == expected["distinct_buyers"], (
        "a total re-aggregates the measure; it is not the sum of the group values"
    )


def test_column_totals_are_pre_limit(client: httpx.Client, known_answers: dict[str, Any]) -> None:
    """The totals row aggregates the post-filter, PRE-limit rows (docs/bench_omni_model.md)."""
    expected = one_answer(known_answers, "grand_totals")
    returned = rows(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT],
            sorts=[{"column_name": TOTAL_SALE_PRICE, "sort_descending": True}],
            limit=3,
            column_totals=aggregation("::total::"),
        )
    )
    assert len(returned) == 4, "three data rows plus the totals row"
    total = returned[-1]
    assert total[TOTAL_INDICATOR_COLUMN] == "::total::"
    assert str(total[TOTAL_SALE_PRICE]) == expected["total_sale_price"]
    assert total[ORDER_ITEMS_COUNT] == expected["order_items_count"]
    assert sum(row[ORDER_ITEMS_COUNT] for row in returned[:3]) < total[ORDER_ITEMS_COUNT]


def test_column_totals_respect_filters(client: httpx.Client, known_answers: dict[str, Any]) -> None:
    expected = sum(
        row["order_items_count"]
        for row in answer_rows(known_answers, "monthly_revenue_trailing_12m")
    )
    returned = rows(
        run(
            client,
            fields=[MONTH, ORDER_ITEMS_COUNT],
            filters={"order_items.created_at": TRAILING_12M},
            column_totals=aggregation("::total::"),
        )
    )
    total = returned[-1]
    assert total[TOTAL_INDICATOR_COLUMN] == "::total::"
    assert total[MONTH] is None
    assert total[ORDER_ITEMS_COUNT] == expected


def test_column_totals_keyed_on_one_measure_leave_the_others_null(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = one_answer(known_answers, "grand_totals")
    returned = rows(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT],
            column_totals=aggregation(ORDER_ITEMS_COUNT),
        )
    )
    total = returned[-1]
    assert total[TOTAL_INDICATOR_COLUMN] == "column_total"
    assert total[ORDER_ITEMS_COUNT] == expected["order_items_count"]
    assert total[TOTAL_SALE_PRICE] is None, "only the requested column is totaled"


def test_the_totals_row_is_the_one_the_normalizer_expects(client: httpx.Client) -> None:
    """End-to-end §2.7: the client strips the indicator and exposes the row via with_totals()."""
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            column_totals=aggregation("::total::"),
        )
    )
    assert job.result is not None
    assert job.summary is not None
    result = normalize(decode_result(job.result), job.summary["fields"], keep_totals=True)
    assert result.dropped_columns == (TOTAL_INDICATOR_COLUMN,)
    assert result.data.num_rows == 21
    assert result.has_totals
    assert result.totals is not None
    assert result.totals.num_rows == 1
    assert result.total_row_types == ("::total::",)
    assert result.data.schema.names == ["users.state", TOTAL_SALE_PRICE]


def test_semantic_results_carry_no_omni_summ_sidecars(client: httpx.Client) -> None:
    """``__omni_summ`` columns are a raw-SQL artifact — a semantic job must not invent them."""
    job = single_job(
        run(
            client,
            fields=["users.state", TOTAL_SALE_PRICE],
            column_totals=aggregation("::total::"),
        )
    )
    assert job.result is not None
    names = decode_result(job.result).schema.names
    assert not any("__omni_summ" in name for name in names)
    assert names == ["users.state", TOTAL_SALE_PRICE, TOTAL_INDICATOR_COLUMN]
    assert job.summary is not None
    assert TOTAL_INDICATOR_COLUMN not in job.summary["fields"], (
        "summary.fields describes requested fields, not Omni's reserved columns"
    )


def test_a_plain_query_has_no_indicator_column(client: httpx.Client) -> None:
    job = single_job(run(client, fields=["users.state", TOTAL_SALE_PRICE]))
    assert job.result is not None
    assert TOTAL_INDICATOR_COLUMN not in decode_result(job.result).schema.names


@pytest.mark.parametrize(
    ("totals", "fields", "expected"),
    [
        (aggregation("::total::"), ["users.state"], "at least one measure"),
        (aggregation("users.state"), ["users.state", TOTAL_SALE_PRICE], "is not one"),
        (aggregation(USERS_COUNT), ["users.state", TOTAL_SALE_PRICE], "is not one"),
        ({"::total::": {"type": "nonsense"}}, [TOTAL_SALE_PRICE], "aggregation"),
        ({"::total::": "yes"}, [TOTAL_SALE_PRICE], "aggregation"),
    ],
)
def test_unusable_column_totals_are_a_plan_error(
    client: httpx.Client, totals: dict[str, Any], fields: list[str], expected: str
) -> None:
    job = single_job(run(client, fields=fields, column_totals=totals))
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert expected in (job.error_message or "")


# --------------------------------------------------------------------------------------
# Raw-SQL jobs (CONTRACT_NOTES §3.4)
# --------------------------------------------------------------------------------------


def test_a_raw_sql_job_answers_the_known_answers(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """Tier 2 end to end: the SQL really runs, against the same rows the semantic path sees."""
    expected = {row["state"]: row for row in answer_rows(known_answers, "revenue_by_state")}
    job = single_job(sql_run(client, REVENUE_SQL))
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert len(returned) == len(expected) == 21
    for row in returned:
        answer = expected[row["state"]]
        assert str(row["total_sale_price"]) == answer["total_sale_price"]
        assert row["order_items_count"] == answer["order_items_count"]
    assert None in {row["state"] for row in returned}, "the LEFT join keeps the orphan rows"


@pytest.mark.parametrize("marker", list(NO_REWRITE_MARKERS))
def test_every_no_rewrite_marker_selects_the_sql_path(client: httpx.Client, marker: str) -> None:
    job = single_job(sql_run(client, "SELECT 42 AS answer", marker=marker))
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None
    assert decode_result(job.result).to_pylist() == [{"answer": 42}]


def test_user_edited_sql_without_a_marker_is_silently_ignored(client: httpx.Client) -> None:
    """The §3.4 trap, reproduced literally: no marker ⇒ the SQL is dropped, the model job runs.

    This is the behavior a client has to be protected against — the answer that comes back is
    perfectly well-formed and has nothing to do with the SQL that was sent.
    """
    job = single_job(run(client, fields=["users.state"], userEditedSQL=REVENUE_SQL, limit=3))
    assert job.status is JobStatus.COMPLETE
    assert job.raw["query"]["userEditedSQL"] == REVENUE_SQL, "the SQL was sent…"
    assert job.summary is not None
    assert list(job.summary["fields"]) == ["users.state"], "…and ignored"
    assert job.result is not None
    table = decode_result(job.result)
    assert table.schema.names == ["users.state"]
    assert table.num_rows == 3


def test_raw_sql_summary_fields_are_synthetic_dimensions(client: httpx.Client) -> None:
    """§2.3 for a SQL job: fields synthesized from the result schema, every one a dimension."""
    job = single_job(sql_run(client, REVENUE_SQL))
    assert job.summary is not None
    fields = job.summary["fields"]
    assert list(fields) == ["state", "total_sale_price", "order_items_count"]
    assert all(field["is_dimension"] is True for field in fields.values()), (
        "the view Omni wraps around userEditedSQL has an empty measures map"
    )
    assert all(field["aggregate_type"] is None for field in fields.values())
    assert [field["data_type"] for field in fields.values()] == ["STRING", "NUMBER", "NUMBER"]
    assert job.summary["display_sql"] == REVENUE_SQL
    assert job.summary["missing_fields"] == []

    assert job.result is not None
    schema = schema_from_summary(fields)
    assert schema.names == tuple(decode_result(job.result).schema.names)
    assert [f.data_type for f in schema.fields] == [
        OmniDataType.STRING,
        OmniDataType.NUMBER,
        OmniDataType.NUMBER,
    ]


def test_a_raw_sql_job_is_framed_exactly_like_a_model_job(client: httpx.Client) -> None:
    job = single_job(sql_run(client, REVENUE_SQL))
    assert job.cache_metadata == {"cache_type": "MISS", "job_id": job.job_id}
    assert job.raw["stream_stats"]["server_stream"] > 0
    assert job.summary is not None
    assert job.summary["cache_type"] == "MISS"


def test_sql_sorts_enabled_applies_the_envelope_sorts(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    ranked = sorted(
        answer_rows(known_answers, "revenue_by_state"),
        key=lambda row: Decimal(row["total_sale_price"]),
        reverse=True,
    )
    job = single_job(
        sql_run(
            client,
            REVENUE_SQL,
            sqlSortsEnabled=True,
            sorts=[{"column_name": "total_sale_price", "sort_descending": True}],
        )
    )
    assert job.summary is not None
    assert "ORDER BY" in job.summary["display_sql"]
    assert job.summary["display_sql"].startswith("SELECT * FROM ("), "the sort wraps the SQL"
    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert [row["state"] for row in returned[:3]] == [row["state"] for row in ranked[:3]]


def test_sorts_are_stripped_when_sql_sorts_are_disabled(client: httpx.Client) -> None:
    """Falsy ``sqlSortsEnabled`` forces ``sorts`` to ``[]`` — silently, not as an error (§3.4)."""
    sorts = [{"column_name": "total_sale_price", "sort_descending": True}]
    job = single_job(sql_run(client, REVENUE_SQL, sorts=sorts))
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.summary is not None
    assert job.summary["display_sql"] == REVENUE_SQL
    assert "ORDER BY" not in job.summary["display_sql"]
    assert job.result is not None
    assert decode_result(job.result).num_rows == 21, "the rows are all there, just unsorted"


def test_a_sort_on_a_column_the_sql_does_not_produce_is_refused(client: httpx.Client) -> None:
    """A SQL job sorts by *result column*; a model field name has no meaning out here."""
    job = single_job(
        sql_run(
            client,
            REVENUE_SQL,
            sqlSortsEnabled=True,
            sorts=[{"column_name": "users.state", "sort_descending": False}],
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "not a column of the userEditedSQL result" in (job.error_message or "")


def test_measure_keyed_filters_are_silently_skipped_on_a_sql_job(client: httpx.Client) -> None:
    """Source-pinned §3.1: the SQL wrapper view has no measures, so the entry is a pure no-op."""
    plain = rows(sql_run(client, REVENUE_SQL))
    filtered = rows(
        sql_run(
            client,
            REVENUE_SQL,
            filters={TOTAL_SALE_PRICE: number_filter("GREATER_THAN", "1000000")},
        )
    )
    assert filtered == plain, "a measure filter must not narrow a SQL job — not even to nothing"
    assert len(plain) == 21


def test_a_dimension_keyed_filter_on_a_sql_job_is_refused(client: httpx.Client) -> None:
    """Dimension filters reach raw SQL through Omni's templating, which the fake will not fake."""
    job = single_job(
        sql_run(
            client,
            REVENUE_SQL,
            filters={"users.state": {"type": "string", "kind": "EQUALS", "values": ["Ohio"]}},
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "templating" in (job.error_message or "")


def test_sql_the_warehouse_rejects_is_an_in_band_query_error(client: httpx.Client) -> None:
    job = single_job(sql_run(client, "SELECT * FROM no_such_table"))
    assert job.status is JobStatus.ERROR
    assert job.error_type == "QUERY", "a warehouse failure is not a planner failure"
    assert "no_such_table" in (job.error_message or "")


def test_plan_only_on_a_sql_job_reports_the_schema_without_data(client: httpx.Client) -> None:
    planned = single_job(sql_run(client, REVENUE_SQL, planOnly=True))
    executed = single_job(sql_run(client, REVENUE_SQL))
    assert planned.status is JobStatus.PLANNED
    assert planned.result is None
    assert planned.summary is not None
    assert executed.summary is not None
    assert schema_from_summary(planned.summary["fields"]) == schema_from_summary(
        executed.summary["fields"]
    )


def test_the_envelope_limit_is_inert_on_a_sql_job(client: httpx.Client) -> None:
    """The SQL owns its own row count; the contract never says the server re-limits it."""
    unlimited = rows(sql_run(client, REVENUE_SQL))
    limited = rows(sql_run(client, REVENUE_SQL, limit=3, offset=2))
    assert limited == unlimited
    assert len(limited) == 21


def test_redaction_blanks_the_sql_of_a_raw_sql_job() -> None:
    fake = FakeOmniAPI(redact_sql=True)
    with make_client(fake) as client:
        job = single_job(sql_run(client, REVENUE_SQL))
    fake.close()
    assert job.summary is not None
    assert job.summary["display_sql"] == ""
    assert job.result is not None, "redaction hides SQL, not data"


# --------------------------------------------------------------------------------------
# __omni_summ sidecars on a raw-SQL column_totals job (CONTRACT_NOTES §2.7)
# --------------------------------------------------------------------------------------


def sql_totals_job(client: httpx.Client, **overrides: Any) -> Any:
    """``REVENUE_SQL`` with both numeric columns totaled — the sidecar-producing shape."""
    return single_job(
        sql_run(
            client,
            REVENUE_SQL,
            sqlSortsEnabled=True,
            column_totals=aggregation("total_sale_price", "order_items_count"),
            **overrides,
        )
    )


def test_sql_column_totals_emit_omni_summ_sidecars(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = one_answer(known_answers, "grand_totals")
    job = sql_totals_job(client)
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None
    table = decode_result(job.result)
    assert table.schema.names == [
        "state",
        "total_sale_price",
        SIDECAR_REVENUE,
        "order_items_count",
        SIDECAR_COUNT,
        TOTAL_INDICATOR_COLUMN,
    ], "each sidecar sits next to the column it totals"

    returned = table.to_pylist()
    assert len(returned) == 22, "21 data rows plus exactly ONE appended totals row"
    data, total = returned[:-1], returned[-1]
    assert all(row[SIDECAR_REVENUE] is None and row[SIDECAR_COUNT] is None for row in data)
    assert all(row[TOTAL_INDICATOR_COLUMN] is None for row in data)

    assert total[TOTAL_INDICATOR_COLUMN] == "column_total", "totals requested per column (§2.7)"
    assert total["state"] is None
    assert total["total_sale_price"] is None, "the base column is NULL; the sidecar carries it"
    assert str(total[SIDECAR_REVENUE]) == expected["total_sale_price"]
    assert total[SIDECAR_COUNT] == expected["order_items_count"]


def test_the_sidecar_name_is_the_lowercased_column_name(client: httpx.Client) -> None:
    """§2.7 spells the sidecar prefix in lowercase even when the SQL column is not."""
    job = single_job(
        sql_run(
            client,
            'SELECT COUNT(*) AS "ORDER_ITEMS_COUNT" FROM order_items',
            sqlSortsEnabled=True,
            column_totals=aggregation("ORDER_ITEMS_COUNT"),
        )
    )
    assert job.result is not None
    assert decode_result(job.result).schema.names == [
        "ORDER_ITEMS_COUNT",
        "order_items_count__omni_summ",
        TOTAL_INDICATOR_COLUMN,
    ]


def test_the_sql_totals_row_round_trips_through_the_normalizer(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """End-to-end §2.7: the client strips the sidecars and reads the totals values out of them."""
    expected = one_answer(known_answers, "grand_totals")
    job = sql_totals_job(client)
    assert job.result is not None
    assert job.summary is not None
    result = normalize(decode_result(job.result), job.summary["fields"], keep_totals=True)

    assert result.dropped_columns == (SIDECAR_REVENUE, SIDECAR_COUNT, TOTAL_INDICATOR_COLUMN)
    assert result.data.schema.names == ["state", "total_sale_price", "order_items_count"]
    assert result.data.num_rows == 21
    assert result.total_row_types == ("column_total",)

    assert result.totals is not None
    total = result.totals.to_pylist()[0]
    assert str(total["total_sale_price"]) == expected["total_sale_price"], (
        "the sidecar supplied the value the base column left NULL"
    )
    assert total["order_items_count"] == expected["order_items_count"]


def test_sql_column_totals_need_sql_sorts_enabled(client: httpx.Client) -> None:
    """``sqlSortsEnabled`` gates ``column_totals`` too — falsy strips them silently (§3.4)."""
    job = single_job(sql_run(client, REVENUE_SQL, column_totals=aggregation("total_sale_price")))
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None
    names = decode_result(job.result).schema.names
    assert names == ["state", "total_sale_price", "order_items_count"]
    assert TOTAL_INDICATOR_COLUMN not in names
    assert not any("__omni_summ" in name for name in names)


@pytest.mark.parametrize(
    ("totals", "expected"),
    [
        (aggregation("state"), "only numeric columns"),
        (aggregation("users.state"), "is not a column"),
        (aggregation("::total::"), "empty measures map"),
        ({"total_sale_price": {"type": "nonsense"}}, "aggregation"),
    ],
)
def test_unusable_sql_column_totals_are_a_plan_error(
    client: httpx.Client, totals: dict[str, Any], expected: str
) -> None:
    job = single_job(sql_run(client, REVENUE_SQL, sqlSortsEnabled=True, column_totals=totals))
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert expected in (job.error_message or "")


# --------------------------------------------------------------------------------------
# staticQueryReferences (CONTRACT_NOTES §3.5; LIVE-VALIDATE #1)
# --------------------------------------------------------------------------------------

#: A reference's columns are the referenced query's field names verbatim, dots and all, so the
#: outer SQL quotes them.  The refKey itself is a bare identifier — the LIVE-VALIDATE #1
#: assumption this whole section is written on.
JOIN_REFERENCES_SQL = (
    'SELECT r."users.state" AS state,\n'
    '       r."order_items.total_sale_price" AS revenue,\n'
    '       b."users.count" AS buyers\n'
    "FROM state_revenue r\n"
    'JOIN state_buyers b ON b."users.state" = r."users.state"\n'
    "ORDER BY 1"
)


def test_a_reference_is_queryable_from_the_outer_sql_as_a_bare_identifier(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    """The semantic engine materializes the reference; the SQL names it as a table."""
    expected = {
        row["state"]: row
        for row in answer_rows(known_answers, "revenue_by_state")
        if row["state"] is not None
    }
    job = single_job(
        sql_run(
            client,
            'SELECT r."users.state" AS state, r."order_items.total_sale_price" AS revenue\n'
            "FROM state_revenue r\n"
            'WHERE r."users.state" IS NOT NULL\n'
            "ORDER BY 1",
            staticQueryReferences={
                "state_revenue": reference(fields=["users.state", TOTAL_SALE_PRICE], limit=None)
            },
        )
    )
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert len(returned) == len(expected) == 20
    for row in returned:
        assert str(row["revenue"]) == expected[row["state"]]["total_sale_price"]


def test_two_references_join_inside_one_sql_query(
    client: httpx.Client, known_answers: dict[str, Any]
) -> None:
    expected = {
        row["state"]: row
        for row in answer_rows(known_answers, "revenue_and_buyers_by_state")
        if row["state"] is not None
    }
    job = single_job(
        sql_run(
            client,
            JOIN_REFERENCES_SQL,
            staticQueryReferences={
                "state_revenue": reference(fields=["users.state", TOTAL_SALE_PRICE], limit=None),
                "state_buyers": reference(fields=["users.state", USERS_COUNT], limit=None),
            },
        )
    )
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None
    returned = decode_result(job.result).to_pylist()
    assert len(returned) == len(expected) == 20
    for row in returned:
        answer = expected[row["state"]]
        assert str(row["revenue"]) == answer["total_sale_price"]
        assert row["buyers"] == answer["distinct_buyers"]


def test_a_reference_query_compiles_through_the_semantic_engine(client: httpx.Client) -> None:
    """A reference is a real semantic query: grains and filters compile the same way."""
    job = single_job(
        sql_run(
            client,
            'SELECT COUNT(*) AS months FROM monthly_rev WHERE "order_items.count" > 0',
            staticQueryReferences={
                "monthly_rev": reference(
                    fields=[MONTH, TOTAL_SALE_PRICE, ORDER_ITEMS_COUNT],
                    filters={"order_items.created_at": TRAILING_12M},
                    limit=None,
                )
            },
        )
    )
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None
    assert decode_result(job.result).to_pylist() == [{"months": 12}]


def test_an_invalid_reference_query_is_a_plan_error_naming_the_reference(
    client: httpx.Client,
) -> None:
    job = single_job(
        sql_run(
            client,
            "SELECT * FROM bad_ref",
            staticQueryReferences={"bad_ref": reference(fields=["users.nope"])},
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    message = job.error_message or ""
    assert "bad_ref" in message
    assert "no resolvable fields" in message


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"fields": ["users.state"]}, "model_id"),
        ({"fields": ["users.state"], "model_id": OTHER_MODEL_ID}, "not found"),
    ],
)
def test_a_reference_carries_its_own_snake_case_model_id(
    client: httpx.Client, payload: dict[str, Any], expected: str
) -> None:
    entry = query_body(**{k: v for k, v in payload.items() if k != "model_id"})["query"]
    if "model_id" in payload:
        entry["model_id"] = payload["model_id"]
    job = single_job(sql_run(client, "SELECT * FROM ref", staticQueryReferences={"ref": entry}))
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert expected in (job.error_message or "")


@pytest.mark.parametrize("key", ["monthly rev", "1st_ref", "order-items", "a.b"])
def test_a_reference_key_must_be_a_bare_sql_identifier(client: httpx.Client, key: str) -> None:
    """LIVE-VALIDATE #1: the fake assumes the refKey is used as a table identifier verbatim."""
    job = single_job(
        sql_run(
            client,
            "SELECT 1 AS x",
            staticQueryReferences={key: reference(fields=["users.state"])},
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "bare SQL identifier" in (job.error_message or "")


def test_references_without_a_raw_sql_outer_query_are_refused(client: httpx.Client) -> None:
    """The other two consumers (``type: "query"`` filters, XLOOKUP calcs) are still unmodeled."""
    job = single_job(
        run(
            client,
            fields=["users.state"],
            staticQueryReferences={"state_revenue": reference(fields=["users.state"])},
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "raw-SQL outer query" in (job.error_message or "")


def test_a_query_typed_filter_arm_is_still_refused(client: httpx.Client) -> None:
    job = single_job(
        run(
            client,
            fields=["users.state"],
            filters={
                "users.state": {
                    "type": "query",
                    "query_id": "state_revenue",
                    "field_name": "users.state",
                }
            },
        )
    )
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert "'query'" in (job.error_message or "")


def test_workbook_url_with_references_is_a_400(client: httpx.Client) -> None:
    """The §2.1 refinement — and the reason it is an envelope 400, not a job error."""
    response = sql_run(
        client,
        "SELECT * FROM state_revenue",
        staticQueryReferences={"state_revenue": reference(fields=["users.state"])},
        workbookUrl=True,
    )
    assert response.status_code == 400
    assert response.json() == {
        "detail": "workbookUrl cannot be combined with a non-empty query.staticQueryReferences",
        "status": 400,
    }
    assert sql_run(client, "SELECT 1 AS x", workbookUrl=True).status_code == 200


def test_workbook_url_is_echoed_as_a_response_header(client: httpx.Client) -> None:
    """§2.1: ``workbookUrl: true`` puts the URL on a **header**, next to the ordinary stream.

    The URL *shape* is LIVE-VALIDATE #10 — the fake mints a recognizable placeholder so a client's
    plumbing can be tested; only the header's presence and its name are pinned by the contract.
    """
    response = run(client, fields=["users.state"], workbookUrl=True)
    job = single_job(response)

    assert response.status_code == 200
    assert response.headers["content-type"] == NDJSON_CONTENT_TYPE
    assert job.status is JobStatus.COMPLETE, job.error_message
    assert job.result is not None, "the rows still come back; the header rides alongside them"
    assert response.headers[WORKBOOK_URL_HEADER] == (
        f"https://bench.example.omni.co/w/fake/{job.job_id}"
    )


def test_no_workbook_url_header_without_the_flag(client: httpx.Client) -> None:
    assert WORKBOOK_URL_HEADER not in run(client, fields=["users.state"]).headers
    assert WORKBOOK_URL_HEADER not in run(client, fields=["users.state"], workbookUrl=False).headers


def test_a_reference_view_does_not_outlive_its_job(client: httpx.Client) -> None:
    """References are per-job temp views: the next SQL job must not see the previous one's."""
    first = single_job(
        sql_run(
            client,
            "SELECT COUNT(*) AS n FROM state_revenue",
            staticQueryReferences={"state_revenue": reference(fields=["users.state"])},
        )
    )
    assert first.status is JobStatus.COMPLETE, first.error_message
    second = single_job(sql_run(client, "SELECT COUNT(*) AS n FROM state_revenue"))
    assert second.status is JobStatus.ERROR
    assert second.error_type == "QUERY"


# --------------------------------------------------------------------------------------
# GET /api/v1/documents/{identifier}/queries (CONTRACT_NOTES §4)
# --------------------------------------------------------------------------------------


def test_document_queries_serve_runnable_bench_queries(client: httpx.Client) -> None:
    payload = client.get(f"/api/v1/documents/{BENCH_DOCUMENT_ID}/queries").json()
    queries = payload["queries"]
    assert len(queries) == 4
    for saved in queries:
        assert set(saved) == {"id", "name", "query", "url"}
        assert saved["query"]["modelId"] == BENCH_MODEL_ID, "verify the modelId before running"

    by_name = {saved["name"]: saved for saved in queries}
    returned = rows(
        client.post("/api/v1/query/run", json={"query": by_name["Revenue by state"]["query"]})
    )
    assert len(returned) == 21
    assert set(returned[0]) == {"users.state", TOTAL_SALE_PRICE}


def test_a_document_without_a_dashboard_is_a_404(client: httpx.Client) -> None:
    """§4's documented 404 — an ordinary answer a client must not treat as an outage."""
    response = client.get(f"/api/v1/documents/{DOCUMENT_WITHOUT_DASHBOARD}/queries")
    assert response.status_code == 404
    assert response.json() == {
        "detail": f"Document {DOCUMENT_WITHOUT_DASHBOARD} does not have a dashboard",
        "status": 404,
    }


def test_an_unknown_document_is_a_404(client: httpx.Client) -> None:
    response = client.get("/api/v1/documents/nope/queries")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document nope not found", "status": 404}


def test_the_document_map_is_configurable() -> None:
    saved = SavedQuery(
        id="q_states",
        name="Just states",
        query=bench_query(fields=["users.state"]),
        url="https://bench.example.omni.co/dashboards/deck/q_states",
    )
    fake = FakeOmniAPI(documents={"deck": (saved,)})
    with make_client(fake) as client:
        configured = client.get("/api/v1/documents/deck/queries")
        default_gone = client.get(f"/api/v1/documents/{BENCH_DOCUMENT_ID}/queries")
    fake.close()
    assert configured.json() == {"queries": [saved.to_wire()]}
    assert default_gone.status_code == 404


# --------------------------------------------------------------------------------------
# POST /api/v1/ai/generate-query (CONTRACT_NOTES §4)
# --------------------------------------------------------------------------------------


def generate(client: httpx.Client, prompt: str, **overrides: Any) -> httpx.Response:
    body: dict[str, Any] = {"modelId": BENCH_MODEL_ID, "prompt": prompt, "runQuery": False}
    body.update(overrides)
    return client.post("/api/v1/ai/generate-query", json=body)


@pytest.mark.parametrize(
    ("prompt", "fields"),
    [
        ("show me monthly revenue", [MONTH, TOTAL_SALE_PRICE]),
        ("Revenue by state, please", ["users.state", TOTAL_SALE_PRICE]),
        ("what are the top products?", ["products.name", TOTAL_SALE_PRICE]),
    ],
)
def test_a_generated_query_runs_through_query_run(
    client: httpx.Client, prompt: str, fields: list[str]
) -> None:
    payload = generate(client, prompt).json()
    assert set(payload) == {"query", "topic", "baseView", "error"}
    assert payload["topic"] == BENCH_TOPIC_NAME
    assert payload["baseView"] == "order_items"
    assert payload["error"] is None
    assert payload["query"]["fields"] == fields

    returned = rows(client.post("/api/v1/query/run", json={"query": payload["query"]}))
    assert returned
    assert set(returned[0]) == set(fields)


def test_the_prompt_map_is_deterministic(client: httpx.Client) -> None:
    first = generate(client, "monthly revenue").json()
    second = generate(client, "MONTHLY REVENUE for the year").json()
    assert first == second, "matching is case-insensitive and substring-based"


@pytest.mark.parametrize(
    "body",
    [
        {"modelId": BENCH_MODEL_ID, "prompt": "monthly revenue"},
        {"modelId": BENCH_MODEL_ID, "prompt": "monthly revenue", "runQuery": True},
        {"modelId": BENCH_MODEL_ID, "prompt": "monthly revenue", "runQuery": "false"},
        {"modelId": BENCH_MODEL_ID, "prompt": "monthly revenue", "runQuery": 0},
    ],
)
def test_generate_query_requires_run_query_false(
    client: httpx.Client, body: dict[str, Any]
) -> None:
    """The server default is ``true``; an absent key would mean "run it behind my back"."""
    response = client.post("/api/v1/ai/generate-query", json=body)
    assert response.status_code == 400
    assert response.json() == {"detail": RUN_QUERY_REFUSAL, "status": 400}


def test_an_ungeneratable_prompt_is_a_400(client: httpx.Client) -> None:
    response = generate(client, "how tall is the Eiffel tower")
    assert response.status_code == 400, "§4: 400 when no query could be generated"
    assert response.json() == {"detail": NO_QUERY_GENERATED, "status": 400}


def test_generate_query_validates_its_model_and_prompt(client: httpx.Client) -> None:
    assert generate(client, "monthly revenue", modelId=OTHER_MODEL_ID).status_code == 404
    assert generate(client, "monthly revenue", modelId="not-a-uuid").status_code == 400
    assert generate(client, "   ").status_code == 400


def test_generate_query_does_not_need_the_query_api_flag() -> None:
    """With ``runQuery: false`` the endpoint skips the feature-flag check (§4)."""
    fake = FakeOmniAPI(feature_flag_off=True)
    with make_client(fake) as client:
        generated = generate(client, "monthly revenue")
        assert run(client).status_code == 403
    fake.close()
    assert generated.status_code == 200


def test_the_ai_credit_shutoff_is_a_402() -> None:
    fake = FakeOmniAPI(ai_credits_exhausted=True)
    with make_client(fake) as client:
        response = generate(client, "monthly revenue")
    fake.close()
    assert response.status_code == 402
    assert response.json()["status"] == 402


# --------------------------------------------------------------------------------------
# missing_fields (CONTRACT_NOTES §2.3)
# --------------------------------------------------------------------------------------


def test_unknown_fields_are_dropped_not_rejected(client: httpx.Client) -> None:
    job = single_job(
        run(
            client,
            fields=["users.state", "users.stat", "order_items.created_at[fortnight]"],
            limit=3,
        )
    )
    assert job.status is JobStatus.COMPLETE, "the server does NOT hard-error on missing fields"
    assert job.summary is not None
    assert job.summary["missing_fields"] == ["users.stat", "order_items.created_at[fortnight]"]
    assert job.result is not None
    assert decode_result(job.result).schema.names == ["users.state"]


def test_missing_fields_are_surfaced_by_the_client(client: httpx.Client) -> None:
    job = single_job(run(client, fields=["users.state", "users.stat"], limit=1))
    with pytest.raises(QueryError, match=r"users\.stat"):
        check_missing_fields(job.summary, job_id=job.job_id)


def test_a_query_with_no_resolvable_fields_is_a_plan_error(client: httpx.Client) -> None:
    job = single_job(run(client, fields=["users.nope"]))
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"


# --------------------------------------------------------------------------------------
# planOnly (CONTRACT_NOTES §2.4)
# --------------------------------------------------------------------------------------


def test_plan_only_returns_a_schema_without_data(client: httpx.Client) -> None:
    job = single_job(run(client, fields=["users.state", "order_items.quantity"], planOnly=True))
    assert job.status is JobStatus.PLANNED
    assert job.is_terminal is False, "PLANNED is not a terminal status"
    assert job.result is None
    assert job.cache_metadata is None
    assert job.summary is not None
    assert list(job.summary["fields"]) == ["users.state", "order_items.quantity"]
    assert job.raw["query"]["fields"] == ["users.state", "order_items.quantity"]


def test_plan_only_schema_matches_the_executed_schema(client: httpx.Client) -> None:
    fields = ["users.state", "users.lifetime_value", "order_items.created_at", "users.is_business"]
    planned = single_job(run(client, fields=fields, planOnly=True))
    executed = single_job(run(client, fields=fields, limit=2))
    assert planned.summary is not None
    assert executed.summary is not None
    assert schema_from_summary(planned.summary["fields"]) == schema_from_summary(
        executed.summary["fields"]
    )


def test_plan_only_with_result_type_is_a_400(client: httpx.Client) -> None:
    response = run(client, planOnly=True, resultType="csv")
    assert response.status_code == 400
    assert response.json() == {
        "detail": "planOnly and resultType cannot both be provided",
        "status": 400,
    }


# --------------------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------------------


def test_query_against_another_model_is_an_in_band_plan_error(client: httpx.Client) -> None:
    response = run(client, modelId=OTHER_MODEL_ID)
    assert response.status_code == 200, "a planner failure is in-band, not an HTTP error"
    job = single_job(response)
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert job.is_failure
    assert job.client_result_id == "null"
    assert job.client_result_id_or_none is None
    assert OTHER_MODEL_ID in (job.error_message or "")


def test_wrong_topic_name_is_a_plan_error(client: httpx.Client) -> None:
    job = single_job(run(client, join_paths_from_topic_name="users_topic"))
    assert job.status is JobStatus.ERROR
    assert job.error_type == "PLAN"
    assert job.error_message == "Topic users_topic not found"


@pytest.mark.parametrize(
    ("body", "expected_detail"),
    [
        ({}, "query is required"),
        ({"query": []}, "query is required"),
        ({"query": {"modelId": "not-a-uuid"}}, "query.modelId must be a UUID"),
        (
            {"query": {"modelId": BENCH_MODEL_ID, "fields": [1, 2]}},
            "query.fields must be an array of strings",
        ),
        (
            {"query": {"modelId": BENCH_MODEL_ID, "fields": [], "limit": 0}},
            "query.limit must be a positive integer or null",
        ),
        (
            {"query": {"modelId": BENCH_MODEL_ID, "branchId": BENCH_MODEL_ID}},
            "branchId is a top-level field, not a query field",
        ),
    ],
)
def test_malformed_bodies_use_the_detail_envelope(
    client: httpx.Client, body: dict[str, Any], expected_detail: str
) -> None:
    response = client.post("/api/v1/query/run", json=body)
    assert response.status_code == 400
    assert response.json() == {"detail": expected_detail, "status": 400}


def test_body_that_is_not_json_is_a_400(client: httpx.Client) -> None:
    response = client.post(
        "/api/v1/query/run",
        content=b'{"query": ',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["status"] == 400


def test_openapi_cache_values_are_rejected(client: httpx.Client) -> None:
    """The published enum (``disabled``/``normal``/…) 400s; only the four real values work."""
    assert run(client, cache="normal").status_code == 400
    assert run(client, cache="SkipRequery").status_code == 200


def test_feature_flag_off_blocks_query_endpoints_only() -> None:
    fake = FakeOmniAPI(feature_flag_off=True)
    with make_client(fake) as client:
        response = run(client)
        assert response.status_code == 403
        assert response.json() == {"detail": "Feature not enabled", "status": 403}
        assert client.get("/api/v1/query/wait", params={"jobIds": "x"}).status_code == 403
        assert client.get(f"/api/v1/models/{BENCH_MODEL_ID}/topic").status_code == 200
    fake.close()


def test_missing_query_topics_permission_is_permission_denied() -> None:
    fake = FakeOmniAPI(permissions=("QUERY_FULL_MODEL", "VIEW_SQL"))
    with make_client(fake) as client:
        response = run(client)
        assert response.status_code == 403
        assert response.json() == {"detail": "Permission denied", "status": 403}
        assert client.get("/api/v1/whoami").status_code == 200
    fake.close()


def test_redaction_without_view_sql() -> None:
    fake = FakeOmniAPI(redact_sql=True)
    with make_client(fake) as client:
        failed = single_job(run(client, modelId=OTHER_MODEL_ID))
        succeeded = single_job(run(client, fields=["users.state"], limit=1))
        topic = client.get(f"/api/v1/models/{BENCH_MODEL_ID}/topic/{BENCH_TOPIC_NAME}").json()
    fake.close()

    assert failed.error_message == REDACTED_ERROR_MESSAGE
    assert failed.is_redacted
    assert OTHER_MODEL_ID not in (failed.error_message or "")

    assert succeeded.summary is not None
    assert succeeded.summary["display_sql"] == ""
    assert succeeded.summary["omni_sql"] == ""
    assert succeeded.summary["fields"]["users.state"]["sql"] == ""
    assert succeeded.result is not None, "redaction hides SQL, not data"

    assert topic["topic"]["views"][0]["dimensions"][0]["sql"] == ""


def test_rate_limit_after_n_requests() -> None:
    fake = FakeOmniAPI(rate_limited_after=2)
    with make_client(fake) as client:
        assert client.get("/api/v1/whoami").status_code == 200
        assert client.get("/api/v1/whoami").status_code == 200
        blocked = client.get("/api/v1/whoami")
        assert blocked.status_code == 429
        assert blocked.headers["X-Omni-Waf-Action"] == "block"
        assert run(client).status_code == 429, "the WAF sits in front of every endpoint"
    fake.close()


# --------------------------------------------------------------------------------------
# The wait loop (CONTRACT_NOTES §2.2)
# --------------------------------------------------------------------------------------


def test_wait_loop_completes_after_the_configured_number_of_polls() -> None:
    fake = FakeOmniAPI(slow_job_polls=3)
    accumulator = StreamAccumulator()
    with make_client(fake) as client:
        accumulator.add_response(run(client).content)
        jobs_after_run = accumulator.jobs
        assert jobs_after_run == (), "the run response timed out before any job finished"
        run_footer = accumulator.footer
        assert run_footer is not None
        assert run_footer.timed_out_raw == "true"
        assert len(accumulator.remaining_job_ids) == 1
        submitted = accumulator.submitted_job_ids

        polls = 0
        while accumulator.pending_job_ids:
            polls += 1
            assert polls <= 5, "wait loop did not converge"
            response = client.get(
                "/api/v1/query/wait",
                params={"jobIds": ",".join(accumulator.pending_job_ids)},
            )
            assert response.headers["content-type"] == "text/ndjson"
            accumulator.add_response(response.content)
    fake.close()

    assert polls == 3
    assert accumulator.is_done
    final_footer = accumulator.footer
    assert final_footer is not None
    assert final_footer.timed_out_raw == "false"
    assert final_footer.remaining_job_ids == ()
    assert accumulator.duplicate_job_ids == (), "job lines arrive exactly once across the cycle"
    job = accumulator.jobs[0]
    assert job.job_id == submitted[0], "the job id came from the header line"
    assert job.status is JobStatus.COMPLETE
    assert job.result is not None
    assert decode_result(job.result).num_rows == 10_000


def test_timed_out_wait_slice_returns_only_a_footer() -> None:
    fake = FakeOmniAPI(slow_job_polls=2)
    with make_client(fake) as client:
        job_id = parse_response(run(client).content).remaining_job_ids[0]
        first = parse_response(client.get("/api/v1/query/wait", params={"jobIds": job_id}).content)
    fake.close()
    assert first.jobs == ()
    assert first.footer is not None
    assert first.footer.remaining_job_ids == (job_id,)
    assert first.footer.timed_out_raw == "true"


def test_wait_accepts_the_legacy_job_ids_parameter() -> None:
    fake = FakeOmniAPI(slow_job_polls=1)
    with make_client(fake) as client:
        job_id = parse_response(run(client).content).remaining_job_ids[0]
        parsed = parse_response(
            client.get("/api/v1/query/wait", params={"job_ids": json.dumps([job_id])}).content
        )
    fake.close()
    assert [job.job_id for job in parsed.jobs] == [job_id]


def test_wait_on_an_unknown_job_id(client: httpx.Client) -> None:
    parsed = parse_response(
        client.get("/api/v1/query/wait", params={"jobIds": "no-such-job"}).content
    )
    assert len(parsed.jobs) == 1
    job = parsed.jobs[0]
    assert job.status is JobStatus.FAILED
    assert job.error_message == "Unable to find job"
    assert parsed.footer is not None
    assert parsed.footer.remaining_job_ids == ()


def test_wait_without_job_ids_is_a_400(client: httpx.Client) -> None:
    response = client.get("/api/v1/query/wait")
    assert response.status_code == 400
    assert response.json() == {"detail": "jobIds is required", "status": 400}


def test_job_ids_are_unique_per_submission(client: httpx.Client) -> None:
    first = single_job(run(client, limit=1)).job_id
    second = single_job(run(client, limit=1)).job_id
    assert first != second


# --------------------------------------------------------------------------------------
# Request log & secret hygiene
# --------------------------------------------------------------------------------------


def test_request_log_records_method_path_params_and_body(
    client: httpx.Client, handler: FakeOmniAPI
) -> None:
    client.get("/api/v1/whoami", params={"modelId": BENCH_MODEL_ID})
    run(client, fields=["users.state"], limit=7)

    assert handler.paths == [
        "GET /api/v1/whoami",
        "POST /api/v1/query/run",
    ]
    whoami = handler.requests[0]
    assert whoami.param("modelId") == BENCH_MODEL_ID
    assert whoami.body is None

    submitted = handler.last_request
    assert submitted.query is not None
    assert submitted.query["fields"] == ["users.state"]
    assert submitted.query["limit"] == 7
    assert submitted.query["version"] == 9


def test_request_log_and_repr_never_contain_the_token(
    client: httpx.Client, handler: FakeOmniAPI
) -> None:
    run(client, limit=1)
    client.get("/api/v1/whoami")
    serialized = json.dumps([vars(request) for request in handler.requests], default=str)
    assert DEFAULT_TOKEN not in serialized
    assert "Authorization" not in serialized
    assert DEFAULT_TOKEN not in repr(handler)
    assert "3f2b1a0c" in repr(handler), "the repr still identifies the model it serves"


def test_unknown_route_is_a_404(client: httpx.Client) -> None:
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not found", "status": 404}


@pytest.mark.parametrize(
    "feature",
    [
        {"pivots": ["users.state"]},
        {"calculations": [{"calc_name": "c", "sql_expression": {}}]},
        {"fill_fields": ["order_items.created_at[month]"]},
        {"row_totals": {"::total::": {"type": "aggregation"}}},
    ],
)
def test_unimplemented_query_features_are_refused_loudly(
    client: httpx.Client, feature: dict[str, Any]
) -> None:
    """A fake that silently ignores a feature would hand back a plausible wrong answer."""
    response = run(client, **feature)
    assert response.status_code == 400
    assert "not implemented by FakeOmniAPI" in response.json()["detail"]
