"""HttpTransport against fixture bytes and hand-built responses (CONTRACT_NOTES §1/§2).

Every "happy" body here is one of the checked-in fixtures under ``fixtures/`` replayed through
``httpx.MockTransport`` — the transport must survive the exact framing
``tests/wire/build_fixtures.py`` produces, including the three-body wait cycle.  Bodies that
cannot be fixtures (HTTP error envelopes, 429s, connect failures) are built inline with the same
helpers the fixture builder uses, so the framing stays identical.

Time is injected: :class:`Clock` stands in for ``time.sleep``/``time.monotonic`` so poll pacing,
retry backoff and the client-side deadline are asserted exactly instead of waited out.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest

from omniframes.errors import (
    AuthError,
    FeatureFlagError,
    ModelPermissionError,
    QueryError,
    QueryTimeoutError,
    TransportError,
)
from omniframes.transport import QueryTransport
from omniframes.transport.http import (
    DEFAULT_RATE_LIMIT_WAIT_SECONDS,
    REDACTED_API_KEY,
    USER_AGENT,
    HttpTransport,
    normalize_base_url,
)
from omniframes.transport.ndjson import REDACTED_ERROR_MESSAGE
from tests.wire import read_fixture
from tests.wire.build_fixtures import (
    HAPPY_FIELDS,
    JOB_ERROR,
    JOB_EXECUTING,
    JOB_FAST,
    JOB_HAPPY,
    JOB_PLANNED,
    JOB_SLOW,
    JOB_UNKNOWN_STATUS,
    MODEL_ID,
    arrow_b64,
    footer_line,
    happy_table,
    header_line,
    ndjson,
    query,
    summary,
)

# A syntactically plausible key (``omni_osk_`` + 50 base62 + 6-char CRC) that is not a real one.
API_KEY = "omni_osk_" + "K3n0mn1" * 7 + "x" + "AB12CD"
BASE_URL = "https://acme.omni.co"

RUN_PATH = "/api/v1/query/run"
WAIT_PATH = "/api/v1/query/wait"

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------------


class Clock:
    """Injected ``time`` — sleeping advances the clock and is recorded."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


class Server:
    """A recording ``httpx.MockTransport`` handler with an optional per-request time cost."""

    def __init__(self, handler: Handler, *, seconds_per_request: float = 0.0) -> None:
        self._handler = handler
        self._seconds_per_request = seconds_per_request
        self.clock = Clock()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.clock.now += self._seconds_per_request
        return self._handler(request)

    @property
    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def body(self, index: int = 0) -> dict[str, Any]:
        payload = json.loads(self.requests[index].content)
        assert isinstance(payload, dict)
        return payload


def transport(server: Server, **kwargs: Any) -> HttpTransport:
    """An HttpTransport wired to ``server`` with deterministic time and no jitter."""
    client = httpx.Client(transport=httpx.MockTransport(server))
    jitter = kwargs.pop("jitter", lambda: 0.0)
    return HttpTransport(
        BASE_URL,
        API_KEY,
        client=client,
        sleep=server.clock.sleep,
        monotonic=server.clock.monotonic,
        jitter=jitter,
        **kwargs,
    )


def ndjson_response(body: bytes, **kwargs: Any) -> httpx.Response:
    headers = {"Content-Type": "text/ndjson", **kwargs.pop("headers", {})}
    return httpx.Response(200, content=body, headers=headers, **kwargs)


def wire_server(
    run: str,
    waits: Sequence[str] = (),
    *,
    seconds_per_request: float = 0.0,
    run_headers: dict[str, str] | None = None,
) -> Server:
    """Replay fixture files: ``run`` answers ``/query/run``, ``waits`` answer each poll.

    Once ``waits`` is exhausted the last body repeats, which is how the never-finishing case
    (deadline expiry) is driven.
    """
    remaining = list(waits)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == RUN_PATH:
            return ndjson_response(read_fixture(run), headers=run_headers or {})
        if request.url.path == WAIT_PATH:
            name = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return ndjson_response(read_fixture(name))
        raise AssertionError(f"unexpected path {request.url.path}")

    return Server(handler, seconds_per_request=seconds_per_request)


def single_body_server(body: bytes, **kwargs: Any) -> Server:
    return Server(lambda _request: ndjson_response(body), **kwargs)


def error_server(status: int, payload: Any, headers: dict[str, str] | None = None) -> Server:
    def handler(_request: httpx.Request) -> httpx.Response:
        if isinstance(payload, str):
            return httpx.Response(status, text=payload, headers=headers)
        return httpx.Response(status, json=payload, headers=headers)

    return Server(handler)


TOPIC_ENVELOPE: dict[str, Any] = {"query": query()}
VIEW_ENVELOPE: dict[str, Any] = {
    "query": {**query(), "join_paths_from_topic_name": None, "table": "order_items"}
}


# --------------------------------------------------------------------------------------------
# Base URL normalization + identity
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("acme.omni.co", "https://acme.omni.co"),
        ("  acme.omni.co  ", "https://acme.omni.co"),
        ("https://acme.omni.co", "https://acme.omni.co"),
        ("https://acme.omni.co/", "https://acme.omni.co"),
        ("acme.omni.co/api", "https://acme.omni.co"),
        ("https://acme.omni.co/api/", "https://acme.omni.co"),
        ("https://acme.omni.co/api/v1", "https://acme.omni.co"),
        ("https://acme.omni.co/api/v1/", "https://acme.omni.co"),
        ("https://acme.omni.co/API/V1", "https://acme.omni.co"),
        ("https://acme.omni.co/api/v1?x=1#frag", "https://acme.omni.co"),
        ("http://localhost:3000/api/v1", "http://localhost:3000"),
        ("https://proxy.internal/omni/api/v1", "https://proxy.internal/omni"),
    ],
)
def test_normalize_base_url(raw, expected):
    assert normalize_base_url(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "ftp://acme.omni.co", "https://"])
def test_normalize_base_url_rejects_garbage(raw):
    with pytest.raises(TransportError):
        normalize_base_url(raw)


def test_transport_satisfies_the_protocol():
    server = wire_server("happy_single_job.ndjson")

    assert isinstance(transport(server), QueryTransport)


def test_constructor_rejects_an_empty_key():
    with pytest.raises(TransportError, match="api_key"):
        HttpTransport(BASE_URL, "   ")


# --------------------------------------------------------------------------------------------
# SECURITY: the key never escapes
# --------------------------------------------------------------------------------------------


def test_repr_redacts_the_api_key():
    server = wire_server("happy_single_job.ndjson")

    rendered = repr(transport(server))

    assert rendered == f"HttpTransport(base_url='{BASE_URL}', api_key={REDACTED_API_KEY})"
    assert API_KEY not in rendered


def test_http_error_text_redacts_an_echoed_api_key():
    server = error_server(403, {"detail": f"Invalid bearer token {API_KEY}", "status": 403})

    with pytest.raises(AuthError) as excinfo:
        transport(server).whoami()

    assert API_KEY not in str(excinfo.value)
    assert REDACTED_API_KEY in str(excinfo.value)


def test_job_error_text_redacts_an_echoed_api_key():
    body = ndjson(
        [
            header_line({JOB_ERROR: "cri-error"}),
            {
                "job_id": JOB_ERROR,
                "status": "ERROR",
                "error_type": "QUERY",
                "error_message": f"connection string leaked the token {API_KEY}",
            },
            footer_line(),
        ]
    )
    server = single_body_server(body)

    with pytest.raises(QueryError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    assert API_KEY not in str(excinfo.value)
    assert REDACTED_API_KEY in str(excinfo.value)


def test_network_error_text_redacts_an_echoed_api_key():
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"TLS handshake failed for Bearer {API_KEY}")

    server = Server(handler)

    with pytest.raises(TransportError) as excinfo:
        transport(server, max_retries=0).whoami()

    assert API_KEY not in str(excinfo.value)


# --------------------------------------------------------------------------------------------
# run(): the happy path and request shaping
# --------------------------------------------------------------------------------------------


def test_run_returns_the_decoded_result():
    server = wire_server("happy_single_job.ndjson")

    result = transport(server).run(TOPIC_ENVELOPE)

    assert result.job_id == JOB_HAPPY
    assert result.table.num_rows == 3
    assert result.table.column_names == [
        "users.state",
        "order_items.order_count",
        "order_items.total_sale_price",
    ]
    assert result.summary["fields"].keys() == HAPPY_FIELDS.keys()
    assert result.cache_metadata["cache_type"] == "MISS"
    assert result.query["modelId"] == MODEL_ID
    assert result.workbook_url is None
    assert server.paths == [RUN_PATH]
    assert server.clock.sleeps == []


def test_run_sends_the_envelope_and_the_auth_headers():
    server = wire_server("happy_single_job.ndjson")

    transport(server).run(TOPIC_ENVELOPE)

    request = server.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}{RUN_PATH}"
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    assert request.headers["User-Agent"] == USER_AGENT
    assert USER_AGENT.startswith("omniframes/")
    assert server.body() == TOPIC_ENVELOPE


def test_run_exposes_the_workbook_url_header():
    server = wire_server(
        "happy_single_job.ndjson",
        run_headers={"X-Omni-Workbook-Url": "https://acme.omni.co/w/abc123"},
    )

    result = transport(server).run({**TOPIC_ENVELOPE, "workbookUrl": True})

    assert result.workbook_url == "https://acme.omni.co/w/abc123"


def test_run_decodes_exotic_arrow_types_untouched():
    server = wire_server("exotic_types.ndjson")

    result = transport(server).run(TOPIC_ENVELOPE)

    assert result.table.schema.field("users.lifetime_value").type.precision == 12
    assert result.table.schema.field("users.created_at").type.tz == "UTC"


def test_run_leaves_reserved_columns_and_totals_rows_in_place():
    # Normalization is the caller's job (QueryResult documents ``table`` as raw).
    server = wire_server("totals_with_sidecars.ndjson")

    result = transport(server).run(TOPIC_ENVELOPE)

    assert result.table.num_rows == 6
    assert "$omni_column_total_indicator" in result.table.column_names


def test_branch_id_is_injected_only_when_the_envelope_lacks_one():
    server = wire_server("happy_single_job.ndjson")
    branch = "11111111-2222-4333-8444-555555555555"

    transport(server, branch_id=branch).run(TOPIC_ENVELOPE)

    assert server.body()["branchId"] == branch


def test_branch_id_in_the_envelope_wins():
    server = wire_server("happy_single_job.ndjson")

    transport(server, branch_id="from-transport").run({**TOPIC_ENVELOPE, "branchId": "explicit"})

    assert server.body()["branchId"] == "explicit"


def test_user_id_is_sent_as_the_query_parameter():
    server = wire_server("happy_single_job.ndjson")

    transport(server, user_id="membership-42").run(TOPIC_ENVELOPE)

    assert server.requests[0].url.params["userId"] == "membership-42"
    assert "userId" not in server.body()


def test_user_id_in_the_body_suppresses_the_query_parameter():
    # Supplying both is a hard 400 (CONTRACT_NOTES §1).
    server = wire_server("happy_single_job.ndjson")

    transport(server, user_id="membership-42").run({**TOPIC_ENVELOPE, "userId": "legacy"})

    assert "userId" not in server.requests[0].url.params
    assert server.body()["userId"] == "legacy"


# --------------------------------------------------------------------------------------------
# run(): the wait loop
# --------------------------------------------------------------------------------------------


def test_run_polls_wait_until_every_job_is_terminal():
    server = wire_server(
        "wait_cycle_run.ndjson",
        ["wait_cycle_wait_timeout.ndjson", "wait_cycle_wait.ndjson"],
    )

    result = transport(server).run(TOPIC_ENVELOPE)

    assert server.paths == [RUN_PATH, WAIT_PATH, WAIT_PATH]
    assert server.requests[1].method == "GET"
    assert server.requests[1].url.params["jobIds"] == JOB_SLOW
    assert server.requests[2].url.params["jobIds"] == JOB_SLOW
    assert {request.headers["User-Agent"] for request in server.requests} == {USER_AGENT}
    # One sleep: the first poll follows the run immediately, the second waits because the
    # pending set did not shrink.
    assert server.clock.sleeps == [1.0]
    assert result.job_id == JOB_FAST


def test_run_poll_interval_is_configurable():
    server = wire_server(
        "wait_cycle_run.ndjson",
        ["wait_cycle_wait_timeout.ndjson", "wait_cycle_wait.ndjson"],
    )

    transport(server, poll_interval_seconds=0.25).run(TOPIC_ENVELOPE)

    assert server.clock.sleeps == [0.25]


def test_a_job_that_reported_executing_first_still_delivers_its_result():
    """The run response may carry a non-terminal line; the answer arrives in a later slice.

    §2.2 documents ``status`` as an OPEN enum in which EXECUTING exists and is non-terminal, and
    ``unknown_status.ndjson`` is exactly such a run response.  Keeping the first line per job id
    would discard the COMPLETE line that follows and report "carried no result payload" for a
    query that succeeded.
    """
    run_body = ndjson(
        [
            header_line({JOB_SLOW: "cri-slow"}),
            {"job_id": JOB_SLOW, "status": "EXECUTING"},
            footer_line([JOB_SLOW]),
        ]
    )
    wait_body = ndjson(
        [
            {
                "job_id": JOB_SLOW,
                "status": "COMPLETE",
                "summary": summary(HAPPY_FIELDS),
                "query": query(),
                "result": arrow_b64(happy_table()),
            },
            footer_line(),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return ndjson_response(run_body if request.url.path == RUN_PATH else wait_body)

    server = Server(handler)
    result = transport(server).run(TOPIC_ENVELOPE)

    assert result.job_id == JOB_SLOW
    assert result.table.num_rows == 3
    assert server.paths == [RUN_PATH, WAIT_PATH]


def test_run_raises_query_timeout_when_the_deadline_elapses():
    server = wire_server(
        "unknown_status.ndjson", ["unknown_status.ndjson"], seconds_per_request=1.0
    )

    with pytest.raises(QueryTimeoutError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE, deadline_seconds=2.5)

    assert excinfo.value.remaining_job_ids == (JOB_EXECUTING, JOB_UNKNOWN_STATUS)
    assert "2.5s" in str(excinfo.value)
    assert server.paths == [RUN_PATH, WAIT_PATH]


def test_max_deadline_seconds_caps_the_per_call_deadline():
    server = wire_server(
        "unknown_status.ndjson", ["unknown_status.ndjson"], seconds_per_request=1.0
    )

    with pytest.raises(QueryTimeoutError, match="1s"):
        transport(server, max_deadline_seconds=1.0).run(TOPIC_ENVELOPE, deadline_seconds=999.0)


# --------------------------------------------------------------------------------------------
# run(): in-band failures
# --------------------------------------------------------------------------------------------


def test_error_line_becomes_a_query_error():
    server = wire_server("error_line.ndjson")

    with pytest.raises(QueryError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    assert excinfo.value.error_type == "QUERY"
    assert excinfo.value.job_id == JOB_ERROR
    assert "Binder Error" in str(excinfo.value)


def test_redacted_error_line_keeps_the_servers_wording():
    server = wire_server("error_line_redacted.ndjson")

    with pytest.raises(QueryError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    assert str(excinfo.value) == REDACTED_ERROR_MESSAGE


def test_complete_but_failed_to_plan_is_a_failure():
    server = wire_server("complete_failed_to_plan.ndjson")

    with pytest.raises(QueryError, match="Failed to plan query"):
        transport(server).run(TOPIC_ENVELOPE)


def test_missing_fields_is_surfaced_as_an_error():
    server = wire_server("missing_fields.ndjson")

    with pytest.raises(QueryError, match=r"users\.stat") as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    assert excinfo.value.error_type == "MISSING_FIELDS"


def test_requery_without_a_result_is_refused():
    server = wire_server("requery_without_result.ndjson")

    with pytest.raises(QueryError, match="materialization"):
        transport(server).run(TOPIC_ENVELOPE)


def test_trailing_upstream_failure_becomes_a_transport_error():
    server = wire_server("trailing_error.ndjson")

    with pytest.raises(TransportError, match="ECONNRESET"):
        transport(server).run(TOPIC_ENVELOPE)


def test_truncated_trailing_failure_becomes_a_transport_error():
    server = wire_server("trailing_error_truncated.ndjson")

    with pytest.raises(TransportError, match="upstream failure"):
        transport(server).run(TOPIC_ENVELOPE)


def test_a_response_without_job_lines_is_a_protocol_error():
    body = ndjson([header_line({JOB_HAPPY: None}), footer_line()])
    server = single_body_server(body)

    with pytest.raises(TransportError, match="no job lines"):
        transport(server).run(TOPIC_ENVELOPE)


def test_a_complete_job_without_a_result_is_a_protocol_error():
    body = ndjson(
        [
            header_line({JOB_HAPPY: None}),
            {"job_id": JOB_HAPPY, "status": "COMPLETE", "summary": summary(HAPPY_FIELDS)},
            footer_line(),
        ]
    )
    server = single_body_server(body)

    with pytest.raises(TransportError, match="no result payload"):
        transport(server).run(TOPIC_ENVELOPE)


# --------------------------------------------------------------------------------------------
# plan()
# --------------------------------------------------------------------------------------------


def test_plan_forces_plan_only_and_returns_the_schema():
    server = wire_server("plan_only.ndjson")

    result = transport(server).plan(TOPIC_ENVELOPE)

    assert server.body()["planOnly"] is True
    assert result.job_id == JOB_PLANNED
    assert list(result.summary["fields"]) == list(HAPPY_FIELDS)
    assert result.query["modelId"] == MODEL_ID


def test_plan_drops_the_fields_the_server_rejects_alongside_plan_only():
    server = wire_server("plan_only.ndjson")

    transport(server).plan(
        {**TOPIC_ENVELOPE, "resultType": "csv", "formatResults": True, "workbookUrl": True}
    )

    body = server.body()
    assert body["planOnly"] is True
    assert "resultType" not in body
    assert "formatResults" not in body
    assert "workbookUrl" not in body


def test_plan_polls_wait_when_the_footer_still_lists_the_job():
    run_body = ndjson([header_line({JOB_PLANNED: "cri-plan"}), footer_line([JOB_PLANNED])])
    wait_body = ndjson(
        [
            {
                "job_id": JOB_PLANNED,
                "status": "PLANNED",
                "summary": summary(HAPPY_FIELDS),
                "query": query(),
            },
            footer_line(),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return ndjson_response(run_body if request.url.path == RUN_PATH else wait_body)

    server = Server(handler)

    result = transport(server).plan(TOPIC_ENVELOPE)

    assert server.paths == [RUN_PATH, WAIT_PATH]
    assert result.job_id == JOB_PLANNED


def test_plan_surfaces_missing_fields():
    body = ndjson(
        [
            header_line({JOB_PLANNED: None}),
            {
                "job_id": JOB_PLANNED,
                "status": "PLANNED",
                "summary": summary(HAPPY_FIELDS, missing_fields=["users.stat"]),
            },
            footer_line(),
        ]
    )
    server = single_body_server(body)

    with pytest.raises(QueryError, match=r"users\.stat"):
        transport(server).plan(TOPIC_ENVELOPE)


# --------------------------------------------------------------------------------------------
# HTTP error mapping (CONTRACT_NOTES §1 — all three envelope shapes)
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        # Express auth middleware envelope.
        (403, {"error": {"code": 403, "message": "Invalid bearer token"}}, AuthError),
        (
            400,
            {"detail": "Bad authorization header, must be formatted as Bearer <token>"},
            AuthError,
        ),
        (
            403,
            {
                "error": {
                    "code": 403,
                    "message": "User no longer has the required permissions to use this API key",
                }
            },
            AuthError,
        ),
        # Remix route envelope.
        (403, {"detail": "Feature not enabled", "status": 403}, FeatureFlagError),
        (403, {"detail": "Permission denied", "status": 403}, ModelPermissionError),
        (404, {"detail": "User with id abc does not exist", "status": 404}, TransportError),
        # /models cursor-parse envelope.
        (400, {"error": "Invalid cursor", "success": False}, TransportError),
        # Non-JSON bodies still map somewhere sane.
        (500, "<html>gateway blew up</html>", TransportError),
        (503, "", TransportError),
    ],
)
def test_http_error_envelopes_map_onto_the_error_hierarchy(status, payload, expected):
    server = error_server(status, payload)

    with pytest.raises(expected):
        transport(server).run(TOPIC_ENVELOPE)


def test_feature_flag_error_tells_the_user_who_can_fix_it():
    server = error_server(403, {"detail": "Feature not enabled", "status": 403})

    with pytest.raises(FeatureFlagError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    message = str(excinfo.value)
    assert "admin" in message
    assert "Query API" in message


def test_permission_denied_infers_query_topics_for_a_topic_query():
    server = error_server(403, {"detail": "Permission denied", "status": 403})

    with pytest.raises(ModelPermissionError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    assert excinfo.value.permission == "QUERY_TOPICS"
    assert "QUERY_TOPICS" in str(excinfo.value)


def test_permission_denied_infers_query_full_model_for_a_bare_view_query():
    server = error_server(403, {"detail": "Permission denied", "status": 403})

    with pytest.raises(ModelPermissionError) as excinfo:
        transport(server).run(VIEW_ENVELOPE)

    assert excinfo.value.permission == "QUERY_FULL_MODEL"


def test_permission_denied_without_context_names_both_permissions():
    server = error_server(403, {"detail": "Permission denied", "status": 403})

    with pytest.raises(ModelPermissionError) as excinfo:
        transport(server).whoami()

    assert excinfo.value.permission is None
    assert "QUERY_TOPICS" in str(excinfo.value)
    assert "QUERY_FULL_MODEL" in str(excinfo.value)


def test_result_type_mode_408_becomes_a_query_timeout():
    server = error_server(
        408,
        {"detail": "Query timed out", "remaining_job_ids": [JOB_SLOW], "timed_out": True},
    )

    with pytest.raises(QueryTimeoutError) as excinfo:
        transport(server).run({**TOPIC_ENVELOPE, "resultType": "csv"})

    assert excinfo.value.remaining_job_ids == (JOB_SLOW,)


def test_transport_error_carries_the_status_and_the_server_text():
    server = error_server(404, {"detail": "Topic nope not found", "status": 404})

    with pytest.raises(TransportError) as excinfo:
        transport(server).get_topic(MODEL_ID, "nope")

    assert "404" in str(excinfo.value)
    assert "Topic nope not found" in str(excinfo.value)


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (404, {"detail": "Document nope not found", "status": 404}),
        (402, {"detail": "AI credits exhausted", "status": 402}),
        (403, {"detail": "Feature not enabled", "status": 403}),
        (403, {"detail": "Permission denied", "status": 403}),
        (403, {"error": {"code": 403, "message": "Invalid bearer token"}}),
    ],
)
def test_every_http_derived_error_records_its_status_as_an_attribute(status, payload):
    """The one HTTP detail that crosses the seam, and it crosses as data rather than as text.

    ``session`` branches on it to tell two documented 404s apart (CONTRACT_NOTES §4); reading it
    back out of the message would make the wording load-bearing.
    """
    server = error_server(status, payload)

    with pytest.raises(TransportError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    assert excinfo.value.status == status


def test_a_rate_limit_that_never_clears_records_429():
    server = error_server(429, {"error": {"code": 429, "message": "Request blocked"}})

    with pytest.raises(TransportError) as excinfo:
        transport(server, max_retries=0).run(TOPIC_ENVELOPE)

    assert excinfo.value.status == 429


def test_a_request_that_never_got_an_answer_has_no_status():
    def refuse(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TransportError) as excinfo:
        transport(Server(refuse), max_retries=0).run(TOPIC_ENVELOPE)

    assert excinfo.value.status is None


def test_a_model_permission_error_keeps_both_its_permission_and_its_status():
    server = error_server(403, {"detail": "Permission denied", "status": 403})

    with pytest.raises(ModelPermissionError) as excinfo:
        transport(server).run(TOPIC_ENVELOPE)

    assert (excinfo.value.permission, excinfo.value.status) == ("QUERY_TOPICS", 403)


# --------------------------------------------------------------------------------------------
# 429 backoff and connect-error retries
# --------------------------------------------------------------------------------------------


def counting_server(
    statuses: Sequence[int],
    *,
    success: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> Server:
    """Answers with ``statuses`` in order; every later request gets the last status."""
    seen = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        index = min(seen["count"], len(statuses) - 1)
        seen["count"] += 1
        status = statuses[index]
        if status == 200:
            return ndjson_response(success or read_fixture("happy_single_job.ndjson"))
        return httpx.Response(status, json={"detail": "Too many requests"}, headers=headers)

    return Server(handler)


def models_payload() -> dict[str, Any]:
    return {"records": [], "pageInfo": {"hasNextPage": False}}


def test_get_429_honours_retry_after_exactly_within_its_budget() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "60"}),
            httpx.Response(200, json=models_payload()),
        ]
    )
    jitter_calls = 0

    def jitter() -> float:
        nonlocal jitter_calls
        jitter_calls += 1
        return 0.0

    server = Server(lambda _request: next(responses))

    transport(server, rate_limit_max_wait_seconds=65, jitter=jitter).list_models()

    assert server.clock.sleeps == [60.0]
    assert len(server.requests) == 2
    assert jitter_calls == 0


def test_get_headerless_429_clamps_its_final_delay_to_the_budget() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        if server.clock.now < 60:
            return httpx.Response(429)
        return httpx.Response(200, json=models_payload())

    server = Server(handler)

    transport(server, rate_limit_max_wait_seconds=65, jitter=lambda: 1.0).list_models()

    assert sum(server.clock.sleeps) == 65.0
    assert server.clock.now == 65.0
    assert server.clock.sleeps[-1] == 5.75


def test_persistent_headerless_get_429_exhausts_its_budget() -> None:
    server = Server(lambda _request: httpx.Response(429))

    with pytest.raises(TransportError) as excinfo:
        transport(server, rate_limit_max_wait_seconds=10).whoami()

    assert excinfo.value.status == 429
    assert sum(server.clock.sleeps) <= 10
    assert f"{len(server.requests)} 429 response(s)" in str(excinfo.value)


def test_get_retry_after_that_exceeds_its_remaining_budget_fails_without_sleeping() -> None:
    server = Server(lambda _request: httpx.Response(429, headers={"Retry-After": "30"}))

    with pytest.raises(TransportError) as excinfo:
        transport(server, rate_limit_max_wait_seconds=10).list_models()

    message = str(excinfo.value)
    assert server.clock.sleeps == []
    assert "30" in message
    assert "10s of 10s" in message
    assert "rate_limit_wait" in message


def test_get_retry_after_equal_to_the_remaining_budget_gets_one_final_request() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "10"}),
            httpx.Response(200, json=models_payload()),
        ]
    )
    server = Server(lambda _request: next(responses))

    transport(server, rate_limit_max_wait_seconds=10).list_models()

    assert server.clock.sleeps == [10.0]
    assert len(server.requests) == 2


def test_headerless_get_429_backoff_uses_jitter() -> None:
    responses = iter(
        [httpx.Response(429), httpx.Response(429), httpx.Response(200, json=models_payload())]
    )
    server = Server(lambda _request: next(responses))

    transport(server, jitter=lambda: 1.0).list_models()

    assert server.clock.sleeps == [0.75, 1.5]


def test_get_429_diagnostics_are_scrubbed_and_body_free() -> None:
    body = f"body echo: Bearer {API_KEY}"
    responses = iter(
        [
            httpx.Response(
                429,
                json={"detail": body},
                headers={"X-Omni-Waf-Action": "block", "Retry-After": "7"},
            ),
            httpx.Response(
                429,
                json={"detail": body},
                headers={"X-Omni-Waf-Action": "block", "Retry-After": "7"},
            ),
        ]
    )
    server = Server(lambda _request: next(responses))

    with pytest.raises(TransportError) as excinfo:
        transport(server, rate_limit_max_wait_seconds=7).list_models()

    message = str(excinfo.value)
    assert "GET /api/v1/models" in message
    assert "2 429 response(s)" in message
    assert "after waiting 7s of its 7s rate-limit budget" in message
    assert "Retry-After: 7" in message
    assert "block" in message
    assert API_KEY not in message
    assert "Bearer " not in message
    assert body not in message


def test_get_429_diagnostics_keep_the_last_raw_retry_after_value() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(429, headers={"Retry-After": "not-a-number"}),
        ]
    )
    server = Server(lambda _request: next(responses))

    with pytest.raises(TransportError, match="not-a-number"):
        transport(server, rate_limit_max_wait_seconds=1).list_models()

    absent = Server(lambda _request: httpx.Response(429))
    with pytest.raises(TransportError, match="Retry-After: absent"):
        transport(absent, rate_limit_max_wait_seconds=0).list_models()

    malformed = Server(lambda _request: httpx.Response(429, headers={"Retry-After": "bad"}))
    with pytest.raises(TransportError, match="Retry-After: bad"):
        transport(malformed, rate_limit_max_wait_seconds=0).list_models()


def test_get_retry_after_zero_has_a_bounded_no_progress_guard() -> None:
    server = Server(lambda _request: httpx.Response(429, headers={"Retry-After": "0"}))

    with pytest.raises(TransportError) as excinfo:
        transport(server).whoami()

    assert len(server.requests) == 9
    assert server.clock.sleeps == [0.0] * 8
    assert excinfo.value.status == 429


def test_get_retry_after_fractional_seconds_is_honoured_exactly() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "0.25"}),
            httpx.Response(200, json=models_payload()),
        ]
    )
    server = Server(lambda _request: next(responses))

    transport(server).list_models()

    assert server.clock.sleeps == [0.25]


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf"), -1])
def test_rate_limit_wait_validation_rejects_invalid_transport_values(value: object) -> None:
    server = json_server({"user": {}})

    with pytest.raises(TransportError, match="rate_limit_max_wait_seconds"):
        transport(server, rate_limit_max_wait_seconds=value)


def test_rate_limit_wait_is_configurable_on_the_transport() -> None:
    server = json_server({"user": {}})

    assert transport(server).rate_limit_max_wait_seconds == DEFAULT_RATE_LIMIT_WAIT_SECONDS
    assert transport(server, rate_limit_max_wait_seconds=42).rate_limit_max_wait_seconds == 42


def test_post_429_then_connect_error_is_not_replayed_after_the_connect_error() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429)
        raise httpx.ConnectError("connection reset by peer")

    server = Server(handler)

    with pytest.raises(TransportError, match="1 attempt"):
        transport(server).run(TOPIC_ENVELOPE)

    assert attempts == 2
    assert server.clock.sleeps == [0.5]


def test_post_429_remains_attempt_bounded_with_the_old_capped_delays() -> None:
    server = counting_server([429], headers={"Retry-After": "60"})

    with pytest.raises(TransportError):
        transport(server).run(TOPIC_ENVELOPE)

    assert len(server.requests) == 4
    assert server.clock.sleeps == [8.0, 8.0, 8.0]


def test_connect_errors_are_retried_for_idempotent_gets():
    attempts = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("connection refused")

    server = Server(handler)

    with pytest.raises(TransportError, match="could not reach"):
        transport(server).whoami()

    assert attempts["count"] == 4
    assert server.clock.sleeps == [0.5, 1.0, 2.0]


def test_read_timeouts_are_retried_for_idempotent_gets():
    attempts = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ReadTimeout("read timed out")

    server = Server(handler)

    with pytest.raises(TransportError):
        transport(server, max_retries=2).list_topics(MODEL_ID)

    assert attempts["count"] == 3


def test_run_posts_are_never_retried_on_connect_errors():
    # A replayed POST /query/run would submit a second job; we cannot tell a lost request from
    # a lost response, so it is not retried at all.
    attempts = {"count": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("connection reset by peer")

    server = Server(handler)

    with pytest.raises(TransportError, match="1 attempt"):
        transport(server).run(TOPIC_ENVELOPE)

    assert attempts["count"] == 1
    assert server.clock.sleeps == []


def test_server_errors_are_not_retried():
    server = counting_server([500])

    with pytest.raises(TransportError):
        transport(server).whoami()

    assert len(server.requests) == 1


# --------------------------------------------------------------------------------------------
# Catalog endpoints
# --------------------------------------------------------------------------------------------


def json_server(payload: Any) -> Server:
    return Server(lambda _request: httpx.Response(200, json=payload))


def test_whoami_hits_the_preflight_endpoint():
    server = json_server({"user": {"id": "u1", "membershipId": "m1"}, "keyScope": "organization"})

    result = transport(server).whoami()

    assert result["keyScope"] == "organization"
    assert server.paths == ["/api/v1/whoami"]
    assert "modelId" not in server.requests[0].url.params


def test_whoami_passes_model_ids_as_csv():
    server = json_server({"user": {}, "rolesByModel": {}})

    transport(server).whoami(("m-1", "m-2"))

    assert server.requests[0].url.params["modelId"] == "m-1,m-2"


def test_list_models_forwards_params_and_drops_nones():
    server = json_server({"pageInfo": {"hasNextPage": False}, "records": []})

    transport(server).list_models(pageSize=100, cursor=None, name="bench")

    params = server.requests[0].url.params
    assert params["pageSize"] == "100"
    assert params["name"] == "bench"
    assert "cursor" not in params


def test_list_topics_and_get_topic_build_the_right_paths():
    server = json_server({"success": True, "topics": []})
    client = transport(server)

    client.list_topics(MODEL_ID)
    client.get_topic(MODEL_ID, "order_items_topic")

    assert server.paths == [
        f"/api/v1/models/{MODEL_ID}/topic",
        f"/api/v1/models/{MODEL_ID}/topic/order_items_topic",
    ]


def test_list_views_builds_the_right_path():
    server = json_server({"success": True, "views": []})

    transport(server).list_views(MODEL_ID)

    assert server.paths == [f"/api/v1/models/{MODEL_ID}/view"]


def test_list_models_passes_the_exact_match_filters_through():
    server = json_server({"records": [], "pageInfo": {"hasNextPage": False}})
    client = transport(server)

    client.list_models(pageSize=100, modelId=MODEL_ID)
    client.list_models(pageSize=100, name="bench_ecommerce", cursor=None)

    first = dict(server.requests[0].url.params)
    second = dict(server.requests[1].url.params)
    assert first["modelId"] == MODEL_ID
    assert second["name"] == "bench_ecommerce"
    assert "cursor" not in second, "a None param is dropped, not sent to a strict schema"


def test_document_queries_builds_the_right_path():
    server = json_server({"queries": []})

    transport(server).document_queries("abc123")

    assert server.paths == ["/api/v1/documents/abc123/queries"]


def test_a_name_with_url_syntax_stays_one_path_segment():
    """A topic name or document id is caller data, not a piece of URL grammar.

    Interpolated raw, ``../../whoami`` walked up to a different endpoint, ``x?foo=1`` moved the
    rest of the path into the query string, and ``abc?tab=1`` put ``/queries`` *inside* the query
    string so the request hit the document endpoint instead.
    """
    server = json_server({"success": True})
    client = transport(server)

    client.get_topic(MODEL_ID, "../../whoami")
    client.get_topic(MODEL_ID, "a/b")
    client.get_topic(MODEL_ID, "x?foo=1")
    client.get_topic(MODEL_ID, "my topic %20")
    client.document_queries("abc123?tab=1")

    # `URL.path` is the *decoded* view; `raw_path` is what actually goes on the wire.
    assert [request.url.raw_path.decode() for request in server.requests] == [
        f"/api/v1/models/{MODEL_ID}/topic/..%2F..%2Fwhoami",
        f"/api/v1/models/{MODEL_ID}/topic/a%2Fb",
        f"/api/v1/models/{MODEL_ID}/topic/x%3Ffoo%3D1",
        f"/api/v1/models/{MODEL_ID}/topic/my%20topic%20%2520",
        "/api/v1/documents/abc123%3Ftab%3D1/queries",
    ]
    assert all(not request.url.params for request in server.requests), "nothing leaked into ?"


def test_generate_query_always_disables_server_side_execution():
    server = json_server({"query": {}, "topic": "order_items_topic"})

    transport(server).generate_query({"modelId": MODEL_ID, "prompt": "sales by state"})

    assert server.paths == ["/api/v1/ai/generate-query"]
    assert server.body() == {
        "modelId": MODEL_ID,
        "prompt": "sales by state",
        "runQuery": False,
    }


def test_generate_query_overrides_a_caller_supplied_run_query():
    server = json_server({"query": {}})

    transport(server).generate_query({"modelId": MODEL_ID, "prompt": "x", "runQuery": True})

    assert server.body()["runQuery"] is False


def test_a_non_json_catalog_body_is_a_transport_error():
    server = Server(lambda _request: httpx.Response(200, text="<html>nope</html>"))

    with pytest.raises(TransportError, match="JSON"):
        transport(server).whoami()


def test_a_json_array_catalog_body_is_a_transport_error():
    server = json_server([1, 2, 3])

    with pytest.raises(TransportError, match="JSON object"):
        transport(server).whoami()


# --------------------------------------------------------------------------------------------
# Client ownership
# --------------------------------------------------------------------------------------------


def test_close_leaves_an_injected_client_open():
    server = wire_server("happy_single_job.ndjson")
    client = httpx.Client(transport=httpx.MockTransport(server))
    owner = HttpTransport(BASE_URL, API_KEY, client=client)

    owner.close()

    assert not client.is_closed
    client.close()


def test_close_closes_a_client_the_transport_created():
    owner = HttpTransport(BASE_URL, API_KEY)

    with owner:
        assert not owner._client.is_closed

    assert owner._client.is_closed


def test_an_owned_client_carries_the_auth_headers():
    owner = HttpTransport(BASE_URL, API_KEY)

    try:
        assert owner._client.headers["Authorization"] == f"Bearer {API_KEY}"
        assert owner._client.headers["User-Agent"] == USER_AGENT
    finally:
        owner.close()


def test_the_arrow_payload_round_trips_through_a_hand_built_body():
    # Guards the assumption the fixtures encode: `result` is base64 of an Arrow IPC *stream*.
    body = ndjson(
        [
            header_line({JOB_HAPPY: None}),
            {
                "job_id": JOB_HAPPY,
                "status": "COMPLETE",
                "summary": summary(HAPPY_FIELDS),
                "result": arrow_b64(happy_table()),
            },
            footer_line(),
        ]
    )
    server = single_body_server(body)

    result = transport(server).run(TOPIC_ENVELOPE)

    assert result.table.equals(happy_table())
