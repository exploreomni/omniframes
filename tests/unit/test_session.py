"""Unit tests for the session and its builder (docs/INTERNALS.md §5, DESIGN §3)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from tests.fakes import BENCH_MODEL_ID, BENCH_TOPIC_NAME, DEFAULT_TOKEN, FakeOmniAPI

from omniframes import OmniSession
from omniframes.compile.querymodel import CachePolicy
from omniframes.errors import (
    CompileError,
    FeatureFlagError,
    ModelPermissionError,
    OmniframesError,
    TransportError,
)
from omniframes.session import API_KEY_ENV, BASE_URL_ENV, SessionBuilder, _status_of
from omniframes.transport import HttpTransport

BASE_URL = "https://bench.omniapp.co"
BRANCH_ID = "9c3b1e5e-2f4a-4d1b-9a7e-6b0f2d8c4a11"


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


def make_session(handler: FakeOmniAPI, **options: Any) -> OmniSession:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    transport = HttpTransport(
        base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
    )
    builder = OmniSession.builder.transport(transport)
    for name, value in options.items():
        getattr(builder, name)(value)
    return builder.get_or_create()


# --------------------------------------------------------------------------------------
# Building (no I/O)
# --------------------------------------------------------------------------------------


def test_building_a_session_performs_no_network_calls(handler: FakeOmniAPI) -> None:
    session = make_session(handler)

    assert handler.requests == [], "the whoami preflight is lazy (docs/DESIGN.md §3)"
    assert session.catalog is not None
    assert session.read is not None


def test_the_builder_is_fresh_every_time() -> None:
    first = OmniSession.builder
    second = OmniSession.builder

    assert isinstance(first, SessionBuilder)
    assert first is not second


def test_host_is_an_alias_of_base_url_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, DEFAULT_TOKEN)
    session = OmniSession.builder.host("acme.omniapp.co/api/v1").api_key_from_env().get_or_create()

    assert repr(session).count("https://acme.omniapp.co") == 1
    session.close()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.api_key_from_env()

    monkeypatch.setenv(API_KEY_ENV, DEFAULT_TOKEN)
    assert isinstance(OmniSession.builder.api_key_from_env(), SessionBuilder)


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BASE_URL_ENV, BASE_URL)
    monkeypatch.setenv(API_KEY_ENV, DEFAULT_TOKEN)
    session = OmniSession.builder.base_url_from_env().api_key_from_env().get_or_create()

    assert BASE_URL in repr(session)
    session.close()


def test_missing_configuration_is_reported_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BASE_URL_ENV, raising=False)
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    with pytest.raises(CompileError, match=BASE_URL_ENV):
        OmniSession.builder.get_or_create()
    with pytest.raises(CompileError, match=API_KEY_ENV):
        OmniSession.builder.host("acme.omniapp.co").get_or_create()


def test_cache_accepts_the_real_policy_values_only(handler: FakeOmniAPI) -> None:
    session = make_session(handler, cache="SkipCache")

    assert session.cache is CachePolicy.SKIP_CACHE
    with pytest.raises(CompileError, match="SkipRequery"):
        OmniSession.builder.cache("normal")


def test_the_builder_has_no_sql_dialect_knob(handler: FakeOmniAPI) -> None:
    """Removed with the v1 tier-2 mechanism (docs/SQLTIER.md §6).

    The server parses an OmniSQL statement and re-renders it per warehouse, so the dialect the
    client emits in is not observable downstream — there is nothing left for the knob to fix.
    """
    assert not hasattr(OmniSession.builder, "sql_dialect")
    assert not hasattr(make_session(handler), "sql_dialect")


def test_session_options_are_exposed_and_reach_the_envelope(handler: FakeOmniAPI) -> None:
    session = make_session(
        handler,
        branch=BRANCH_ID,
        timezone="America/Los_Angeles",
        cache=CachePolicy.STANDARD,
        user_id=BENCH_MODEL_ID,
    )
    options = session.envelope_options()

    assert (session.branch, session.timezone, session.user_id) == (
        BRANCH_ID,
        "America/Los_Angeles",
        BENCH_MODEL_ID,
    )
    assert options.branch_id == BRANCH_ID
    assert options.cache is CachePolicy.STANDARD


def test_camel_case_get_or_create_alias(handler: FakeOmniAPI) -> None:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    transport = HttpTransport(base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client)

    assert isinstance(OmniSession.builder.transport(transport).getOrCreate(), OmniSession)


def test_rate_limit_wait_reaches_a_transport_built_by_the_builder() -> None:
    session = (
        OmniSession.builder.host(BASE_URL)
        .api_key(DEFAULT_TOKEN)
        .rate_limit_wait(42)
        .get_or_create()
    )

    assert isinstance(session._transport, HttpTransport)
    assert session._transport.rate_limit_max_wait_seconds == 42
    session.close()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1, "x"])
def test_rate_limit_wait_rejects_invalid_builder_values(value: object) -> None:
    with pytest.raises(CompileError, match="rate_limit_wait"):
        OmniSession.builder.rate_limit_wait(value)  # type: ignore[arg-type]


def test_rate_limit_wait_cannot_be_combined_with_an_injected_transport(
    handler: FakeOmniAPI,
) -> None:
    injected = HttpTransport(
        BASE_URL, DEFAULT_TOKEN, client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(CompileError, match="configure rate-limit waiting"):
        OmniSession.builder.transport(injected).rate_limit_wait(42)
    with pytest.raises(CompileError, match="configure rate-limit waiting"):
        OmniSession.builder.rate_limit_wait(42).transport(injected)


def test_read_topic_recovers_from_a_rate_limited_catalog_request_and_caches_it() -> None:
    fake = FakeOmniAPI()
    sleeps: list[float] = []
    requests: list[httpx.Request] = []
    model_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_attempts
        requests.append(request)
        if request.url.path == "/api/v1/models":
            model_attempts += 1
            if model_attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "1"})
        return fake(request)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    injected = HttpTransport(
        BASE_URL,
        DEFAULT_TOKEN,
        client=client,
        sleep=sleeps.append,
        jitter=lambda: 0.0,
        rate_limit_max_wait_seconds=2,
    )
    session = OmniSession.builder.transport(injected).get_or_create()

    session.read.topic("bench_ecommerce", BENCH_TOPIC_NAME)
    request_count = len(requests)
    session.read.topic("bench_ecommerce", BENCH_TOPIC_NAME)

    assert sleeps == [1.0]
    assert [request.url.path for request in requests] == [
        "/api/v1/whoami",
        "/api/v1/models",
        "/api/v1/models",
        f"/api/v1/models/{BENCH_MODEL_ID}/topic",
    ]
    assert len(requests) == request_count
    session.close()
    fake.close()


# --------------------------------------------------------------------------------------
# Secrecy
# --------------------------------------------------------------------------------------


def test_the_api_key_never_appears_in_a_repr(handler: FakeOmniAPI) -> None:
    session = make_session(handler)

    assert DEFAULT_TOKEN not in repr(session)
    assert "omni_osk_***" in repr(session)


# --------------------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------------------


def test_verify_runs_whoami_and_reports_permissions(handler: FakeOmniAPI) -> None:
    session = make_session(handler)
    payload = session.verify()

    assert handler.paths == ["GET /api/v1/whoami"]
    assert payload["rolesByModel"][BENCH_MODEL_ID]["permissions"][0] == "QUERY_TOPICS"


def test_the_preflight_is_cached(handler: FakeOmniAPI) -> None:
    session = make_session(handler)
    session.verify()
    assert session.whoami is session.whoami

    session.catalog.models()

    assert handler.paths.count("GET /api/v1/whoami") == 1


def test_the_first_catalog_call_triggers_the_preflight(handler: FakeOmniAPI) -> None:
    session = make_session(handler)
    session.catalog.models()

    assert handler.paths[0] == "GET /api/v1/whoami"


def test_whoami_works_even_when_the_query_api_flag_is_off() -> None:
    fake = FakeOmniAPI(feature_flag_off=True)
    session = make_session(fake)

    assert session.verify()["keyScope"] == "organization"
    fake.close()


# --------------------------------------------------------------------------------------
# Error framing
# --------------------------------------------------------------------------------------


def test_a_disabled_feature_flag_names_the_administrative_remedy() -> None:
    fake = FakeOmniAPI(feature_flag_off=True)
    session = make_session(fake)
    envelope = {"query": {"modelId": BENCH_MODEL_ID, "fields": ["users.state"], "limit": 1}}

    with pytest.raises(FeatureFlagError) as caught:
        session.run(envelope)

    message = str(caught.value)
    assert "admin" in message
    assert "query-api" in message
    fake.close()


def test_a_missing_model_permission_names_the_administrative_remedy() -> None:
    fake = FakeOmniAPI(permissions=("QUERY_FULL_MODEL", "VIEW_SQL"))
    session = make_session(fake)
    envelope = {
        "query": {
            "modelId": BENCH_MODEL_ID,
            "join_paths_from_topic_name": BENCH_TOPIC_NAME,
            "fields": ["users.state"],
            "limit": 1,
        }
    }

    with pytest.raises(ModelPermissionError) as caught:
        session.run(envelope)

    assert caught.value.permission == "QUERY_TOPICS"
    assert "admin" in str(caught.value)
    assert "QUERY_TOPICS" in str(caught.value)
    fake.close()


# --------------------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------------------


def test_read_topic_resolves_the_model_and_the_topic(handler: FakeOmniAPI) -> None:
    session = make_session(handler)
    df = session.read.topic("bench_ecommerce", BENCH_TOPIC_NAME)
    scan = df.logical_plan

    assert scan.source.model_id == BENCH_MODEL_ID  # type: ignore[attr-defined]
    assert scan.source.topic == BENCH_TOPIC_NAME  # type: ignore[attr-defined]
    assert scan.source.base_view == "order_items"  # type: ignore[attr-defined]


def test_read_view_resolves_through_the_flattened_view_list(handler: FakeOmniAPI) -> None:
    session = make_session(handler)
    df = session.read.view("bench_ecommerce", "users")

    assert df.logical_plan.source.view == "users"  # type: ignore[attr-defined]


def test_read_view_accepts_a_view_no_topic_reaches(handler: FakeOmniAPI) -> None:
    """A bare view is read *outside* any topic, so topic reachability must not gate it."""
    session = make_session(handler)
    df = session.read.view("bench_ecommerce", "inventory_snapshots")

    assert df.logical_plan.source.view == "inventory_snapshots"  # type: ignore[attr-defined]


def test_read_topic_costs_three_requests_on_a_cold_session(handler: FakeOmniAPI) -> None:
    """No cursor walk, whatever the catalog size (issue #4)."""
    session = make_session(handler)
    session.read.topic(BENCH_MODEL_ID, BENCH_TOPIC_NAME)

    assert handler.paths == [
        "GET /api/v1/whoami",
        "GET /api/v1/models",
        f"GET /api/v1/models/{BENCH_MODEL_ID}/topic",
    ]
    assert handler.requests[1].params["modelId"] == [BENCH_MODEL_ID]


def test_read_view_costs_three_requests_independent_of_topic_count(handler: FakeOmniAPI) -> None:
    """No per-topic detail fan-out (issue #5)."""
    session = make_session(handler)
    session.read.view(BENCH_MODEL_ID, "users")

    assert handler.paths == [
        "GET /api/v1/whoami",
        "GET /api/v1/models",
        f"GET /api/v1/models/{BENCH_MODEL_ID}/view",
    ]
    assert not any("/topic" in path for path in handler.paths)


def test_unknown_topics_and_views_fail_at_read_time_with_the_alternatives(
    handler: FakeOmniAPI,
) -> None:
    session = make_session(handler)

    with pytest.raises(CompileError, match=BENCH_TOPIC_NAME):
        session.read.topic("bench_ecommerce", "nope")
    with pytest.raises(CompileError, match="products"):
        session.read.view("bench_ecommerce", "nope")


def test_close_only_closes_a_transport_the_session_created(handler: FakeOmniAPI) -> None:
    injected = make_session(handler)
    injected.close()

    with make_session(handler) as session:
        assert session.read is not None


# --------------------------------------------------------------------------------------
# The HTTP status a transport error carries (M5 hygiene)
# --------------------------------------------------------------------------------------


def test_the_status_comes_off_the_attribute_not_out_of_the_message() -> None:
    """Two endpoints branch on the status; the wording of the message is not their contract."""
    assert _status_of(TransportError("nothing here at all", status=404)) == 404
    assert _status_of(TransportError("the Omni API returned 500 for GET /x: boom", status=402)) == (
        402
    ), "the attribute wins over anything the text happens to say"


def test_the_message_regex_survives_as_a_fallback_for_an_error_without_one() -> None:
    """A re-raised error rebuilt from a message — or an older transport — still resolves."""
    assert _status_of(TransportError("the Omni API returned 404 for GET /x: nope")) == 404
    assert _status_of(TransportError("no status anywhere")) is None
    assert _status_of(OmniframesError("not a transport error at all")) is None


def test_a_model_permission_error_forwards_the_status_to_its_base() -> None:
    error = ModelPermissionError("denied", permission="QUERY_TOPICS", status=403)

    assert (error.status, error.permission) == (403, "QUERY_TOPICS")


def test_the_advice_wrapper_keeps_the_status_it_was_handed(handler: FakeOmniAPI) -> None:
    """``_crisp_permissions`` rebuilds the error to add advice; the status must survive that."""
    handler.permissions = ("QUERY_FULL_MODEL",)
    session = make_session(handler)

    with pytest.raises(ModelPermissionError) as excinfo:
        session.run({"query": {"modelId": BENCH_MODEL_ID, "fields": ["users.state"]}})

    assert excinfo.value.status == 403
    assert "admin" in str(excinfo.value)
