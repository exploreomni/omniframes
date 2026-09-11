"""Unit tests for the catalog (docs/INTERNALS.md §5, CONTRACT_NOTES §4).

Driven through the real :class:`~omniframes.transport.http.HttpTransport` against
:class:`~tests.fakes.FakeOmniAPI`, so the payload shapes are the documented ones rather than a
second guess at them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from tests.fakes import (
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

from omniframes.catalog import Catalog, _is_uuid
from omniframes.errors import CompileError
from omniframes.transport import HttpTransport
from omniframes.types import OmniDataType

BASE_URL = "https://bench.omniapp.co"


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


@pytest.fixture
def catalog(handler: FakeOmniAPI) -> Iterator[Catalog]:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    with client:
        yield Catalog(HttpTransport(base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client))


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def test_models_walks_the_whole_cursor(catalog: Catalog, handler: FakeOmniAPI) -> None:
    models = catalog.models()

    assert [model.name for model in models] == [
        "bench_ecommerce",
        "bench_marketing",
        "bench_finance",
        "bench_support",
        "bench_ecommerce_branch",
    ]
    assert handler.last_request.param("pageSize") == "100"


def test_models_are_typed_and_keep_the_raw_payload(catalog: Catalog) -> None:
    model = catalog.model(BENCH_MODEL_NAME)

    assert model.id == BENCH_MODEL_ID
    assert model.model_kind == "SHARED"
    assert model.raw["createdAt"], "the raw payload stays reachable"


def test_a_model_resolves_by_name_or_id(catalog: Catalog) -> None:
    assert catalog.model(BENCH_MODEL_NAME) == catalog.model(BENCH_MODEL_ID)
    assert catalog.model_id(BENCH_MODEL_NAME) == BENCH_MODEL_ID


def test_an_unknown_model_names_the_ones_that_exist(catalog: Catalog) -> None:
    with pytest.raises(CompileError, match="bench_ecommerce"):
        catalog.model("nope")


# --------------------------------------------------------------------------------------
# Topics and views
# --------------------------------------------------------------------------------------


def test_topics_come_from_the_list_endpoint_without_field_metadata(catalog: Catalog) -> None:
    topics = catalog.topics(BENCH_MODEL_NAME)

    assert [topic.name for topic in topics] == [BENCH_TOPIC_NAME]
    assert topics[0].base_view_name == "order_items"
    assert topics[0].is_detailed is False, "the list endpoint is lossy by design"


def test_topic_detail_carries_views_relationships_and_typed_fields(catalog: Catalog) -> None:
    topic = catalog.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)

    assert topic.is_detailed
    assert [view.name for view in topic.views] == ["order_items", "users", "products"]
    assert topic.field("users.state").data_type is OmniDataType.STRING
    assert topic.field("order_items.total_sale_price").is_dimension is False
    assert [(edge.left_view_name, edge.right_view_name) for edge in topic.relationships] == [
        ("order_items", "users"),
        ("order_items", "products"),
    ]
    assert {edge.relationship_type for edge in topic.relationships} == {"many_to_one"}


def test_views_are_reachable_and_split_dimensions_from_measures(catalog: Catalog) -> None:
    views = {view.name: view for view in catalog.views(BENCH_MODEL_NAME)}

    assert set(views) == {"order_items", "users", "products"}
    order_items = views["order_items"]
    assert {measure.name for measure in order_items.measures} == {
        "order_items.total_sale_price",
        "order_items.count",
        "order_items.total_quantity",
        "order_items.average_sale_price",
    }
    assert order_items.field("status").data_type is OmniDataType.STRING
    assert order_items.field("order_items.status").name == "order_items.status"


def test_unknown_names_are_reported_with_what_is_available(catalog: Catalog) -> None:
    topic = catalog.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)

    with pytest.raises(CompileError, match="no view"):
        topic.view("nope")
    with pytest.raises(CompileError, match="no field"):
        topic.field("users.nope")
    with pytest.raises(CompileError, match="no field"):
        topic.view("users").field("nope")


def test_an_unknown_topic_surfaces_the_transport_error(catalog: Catalog) -> None:
    from omniframes.errors import TransportError

    with pytest.raises(TransportError, match="404"):
        catalog.topic(BENCH_MODEL_NAME, "nope")


# --------------------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------------------


def test_lookups_are_read_through_cached(catalog: Catalog, handler: FakeOmniAPI) -> None:
    catalog.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)
    catalog.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)
    catalog.topics(BENCH_MODEL_NAME)
    catalog.models()

    assert handler.paths == [
        "GET /api/v1/models",
        f"GET /api/v1/models/{BENCH_MODEL_ID}/topic/{BENCH_TOPIC_NAME}",
        f"GET /api/v1/models/{BENCH_MODEL_ID}/topic",
        "GET /api/v1/models",
    ], "each endpoint is hit exactly once, in call order"
    # The first `/models` is the filtered resolution, the last is the full walk `models()` asked
    # for: a cheap resolution must not stand in for the complete catalog.
    assert handler.requests[0].params["name"] == ["bench_ecommerce"]
    assert "name" not in handler.requests[-1].params


# --------------------------------------------------------------------------------------
# Cheap resolution (CONTRACT_NOTES §4 server-side filters)
# --------------------------------------------------------------------------------------


def test_resolving_by_id_uses_the_model_id_filter(catalog: Catalog, handler: FakeOmniAPI) -> None:
    info = catalog.model(BENCH_MODEL_ID)

    assert info.id == BENCH_MODEL_ID
    assert handler.paths == ["GET /api/v1/models"], "one filtered request, no cursor walk"
    assert handler.last_request.param("modelId") == BENCH_MODEL_ID
    assert handler.last_request.param("cursor") is None


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ("3f2b1a0c-9d8e-4c7b-a6f5-000000000001", True),
        ("00000000-0000-0000-0000-000000000000", True),  # the nil UUID, allowed explicitly
        ("ffffffff-ffff-ffff-ffff-ffffffffffff", True),  # the max UUID, allowed explicitly
        ("3f2b1a0c9d8e4c7ba6f5000000000001", False),  # un-hyphenated
        ("{3f2b1a0c-9d8e-4c7b-a6f5-000000000001}", False),  # braced
        ("urn:uuid:3f2b1a0c-9d8e-4c7b-a6f5-000000000001", False),  # URN
        ("3f2b1a0c-9d8e-9c7b-a6f5-000000000001", False),  # version nibble out of 1-8
        ("3f2b1a0c-9d8e-4c7b-c6f5-000000000001", False),  # variant nibble not 8/9/a/b
        ("bench_ecommerce", False),
    ],
)
def test_is_uuid_matches_the_servers_grammar(value: str, accepted: bool) -> None:
    """`uuid.UUID()` is looser than `z.uuid()`; being looser here costs the whole fallback chain.

    A value this accepts and the server rejects is a 400 that pre-empts the `?name=` filter and
    the catalog walk, so a resolvable name fails with a transport error instead.
    """
    assert _is_uuid(value) is accepted


def test_a_dashless_id_still_resolves_by_falling_through_to_the_name_filter(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    """The server would 400 on it as `?modelId=`, so it must never be sent as one."""
    with pytest.raises(CompileError, match="no model named"):
        catalog.model(BENCH_MODEL_ID.replace("-", ""))

    assert all("modelId" not in request.params for request in handler.requests)
    assert any("name" in request.params for request in handler.requests), "it tried ?name= instead"


def test_resolving_by_name_uses_the_name_filter(catalog: Catalog, handler: FakeOmniAPI) -> None:
    info = catalog.model(BENCH_MODEL_NAME)

    assert info.id == BENCH_MODEL_ID
    assert handler.paths == ["GET /api/v1/models"]
    assert handler.last_request.param("name") == BENCH_MODEL_NAME
    assert handler.last_request.param("modelId") is None, "a non-UUID name would 400 on modelId"


def test_an_unknown_name_falls_back_to_the_walk_for_the_error(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    with pytest.raises(CompileError, match="bench_marketing"):
        catalog.model("nope")

    # The filtered miss, then the walk that supplies the available-model list.
    assert handler.paths.count("GET /api/v1/models") > 1


def test_an_empty_name_does_not_resolve_an_arbitrary_model(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    """The server drops a filter it considers empty, so the reply is the whole first page.

    Resolution must report no match rather than taking whatever model happened to sort first.
    """
    with pytest.raises(CompileError, match="no model named ''"):
        catalog.model("")

    assert all("name" not in request.params for request in handler.requests), (
        "a filter the server would drop cannot identify anything; don't spend the request"
    )


def test_a_filtered_reply_that_does_not_match_is_rejected() -> None:
    """A filtered list endpoint is not a lookup-by-key: verify the reply, don't trust position."""
    calls: list[dict[str, Any]] = []

    class WrongRecordTransport:
        def list_models(self, **params: Any) -> dict[str, Any]:
            calls.append(params)
            # A truthy filter, and the server answers with a model that is not the one asked for.
            return {
                "records": [{"id": BENCH_MODEL_ID, "name": "some_other_model"}],
                "pageInfo": {"hasNextPage": False},
            }

    catalog = Catalog(WrongRecordTransport())  # type: ignore[arg-type]

    with pytest.raises(CompileError, match="no model named 'bench_ecommerce'"):
        catalog.model(BENCH_MODEL_NAME)
    assert calls, "the filtered lookup was attempted"


def test_resolution_does_not_poison_the_full_listing(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    catalog.model(BENCH_MODEL_NAME)

    assert [model.name for model in catalog.models()] == [
        "bench_ecommerce",
        "bench_marketing",
        "bench_finance",
        "bench_support",
        "bench_ecommerce_branch",
    ], "a filtered lookup must not stand in for the whole catalog"


def test_a_cached_catalog_resolves_without_a_request(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    catalog.models()
    before = len(handler.paths)

    assert catalog.model(BENCH_MODEL_NAME).id == BENCH_MODEL_ID
    assert catalog.model(BENCH_MODEL_ID).name == BENCH_MODEL_NAME
    assert handler.paths[before:] == [], "the full catalog already answers both forms"


def test_a_cached_catalog_makes_a_miss_free(catalog: Catalog, handler: FakeOmniAPI) -> None:
    """The full catalog lists everything the key can see, so a miss needs no confirmation.

    Re-asking the filters would spend the rate-limit budget this whole path exists to conserve,
    once per retyped typo.
    """
    catalog.models()
    before = len(handler.paths)

    for _ in range(3):
        with pytest.raises(CompileError, match="bench_ecommerce"):
            catalog.model("bench_ecommerc")

    assert handler.paths[before:] == []


@pytest.mark.parametrize("preload_catalog", [False, True], ids=["cold", "cached"])
def test_model_names_are_matched_case_sensitively(catalog: Catalog, preload_catalog: bool) -> None:
    """The client is deliberately stricter than the server here, and stays consistent.

    `?name=` is case-insensitive server-side (CITEXT, CONTRACT_NOTES §4), but resolution compares
    exactly — on the filtered reply *and* on the cursor walk. Matching CITEXT on only one of
    those would make the answer depend on whether the catalog happened to be cached.
    """
    if preload_catalog:
        catalog.models()

    with pytest.raises(CompileError, match="no model named 'BENCH_ECOMMERCE'"):
        catalog.model("BENCH_ECOMMERCE")


def test_a_resolution_is_cached_under_both_name_and_id(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    catalog.model(BENCH_MODEL_NAME)
    before = len(handler.paths)

    catalog.model(BENCH_MODEL_NAME)
    catalog.model(BENCH_MODEL_ID)

    assert handler.paths[before:] == []


# --------------------------------------------------------------------------------------
# Flattened view names
# --------------------------------------------------------------------------------------


def test_view_names_are_one_request_and_span_the_composed_model(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    names = catalog.view_names(BENCH_MODEL_NAME)

    assert set(names) == {"order_items", "users", "products", "inventory_snapshots"}
    assert "inventory_snapshots" not in {view.name for view in catalog.views(BENCH_MODEL_NAME)}, (
        "the flattened list is wider than the topic-reachable one — that is the point"
    )
    assert handler.paths.count(f"GET /api/v1/models/{BENCH_MODEL_ID}/view") == 1


def test_a_view_list_without_a_views_array_raises_instead_of_caching_empty() -> None:
    """One malformed response must not become a session-long "Views: (none)"."""

    class NoViewsTransport:
        def list_models(self, **params: Any) -> dict[str, Any]:
            return {
                "records": [{"id": BENCH_MODEL_ID, "name": BENCH_MODEL_NAME}],
                "pageInfo": {"hasNextPage": False},
            }

        def list_views(self, model_id: str) -> dict[str, Any]:
            return {"success": True}

    catalog = Catalog(NoViewsTransport())  # type: ignore[arg-type]

    with pytest.raises(CompileError, match="no 'views' array"):
        catalog.view_names(BENCH_MODEL_NAME)


def test_a_model_with_no_views_caches_the_empty_result() -> None:
    """An empty array is a real answer, unlike a missing key — cache it, don't re-ask."""
    calls: list[str] = []

    class EmptyViewsTransport:
        def list_models(self, **params: Any) -> dict[str, Any]:
            return {
                "records": [{"id": BENCH_MODEL_ID, "name": BENCH_MODEL_NAME}],
                "pageInfo": {"hasNextPage": False},
            }

        def list_views(self, model_id: str) -> dict[str, Any]:
            calls.append(model_id)
            return {"success": True, "views": []}

    catalog = Catalog(EmptyViewsTransport())  # type: ignore[arg-type]

    assert catalog.view_names(BENCH_MODEL_NAME) == ()
    assert catalog.view_names(BENCH_MODEL_NAME) == ()
    assert len(calls) == 1


def test_view_names_are_cached_and_dropped_by_refresh(
    catalog: Catalog, handler: FakeOmniAPI
) -> None:
    catalog.view_names(BENCH_MODEL_NAME)
    catalog.view_names(BENCH_MODEL_NAME)
    assert handler.paths.count(f"GET /api/v1/models/{BENCH_MODEL_ID}/view") == 1

    catalog.refresh()
    catalog.view_names(BENCH_MODEL_NAME)
    assert handler.paths.count(f"GET /api/v1/models/{BENCH_MODEL_ID}/view") == 2


def test_refresh_drops_the_cache(catalog: Catalog, handler: FakeOmniAPI) -> None:
    catalog.models()
    catalog.refresh()
    catalog.models()

    assert handler.paths.count("GET /api/v1/models") == 2


def test_a_preflight_hook_runs_before_the_first_request(handler: FakeOmniAPI) -> None:
    calls: list[str] = []
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    with client:
        transport = HttpTransport(base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client)
        catalog = Catalog(transport, preflight=lambda: calls.append("preflight"))
        catalog.models()
        catalog.models()

    assert calls == ["preflight"], "cached lookups do not re-run the preflight"
