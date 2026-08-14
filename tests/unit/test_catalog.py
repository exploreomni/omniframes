"""Unit tests for the catalog (docs/INTERNALS.md §5, CONTRACT_NOTES §4).

Driven through the real :class:`~omniframes.transport.http.HttpTransport` against
:class:`~tests.fakes.FakeOmniAPI`, so the payload shapes are the documented ones rather than a
second guess at them.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest
from tests.fakes import (
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

from omniframes.catalog import Catalog
from omniframes.errors import CompileError
from omniframes.transport import HttpTransport
from omniframes.types import OmniDataType

BASE_URL = "https://bench.example.omni.co"


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
    ], "each endpoint is hit exactly once, in call order"


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
