"""The governed Omni model the :class:`~tests.fakes.fake_omni.FakeOmniAPI` serves.

This is the offline twin of the model that ``docs/bench_omni_model.md`` describes for the live
org: model ``bench_ecommerce``, topic ``order_items`` (base view ``order_items``, joined to
``users`` on ``user_id`` and to ``products`` on ``product_id``), over the checked-in bench
dataset (``docs/BENCH_DATASET.md``).  Keeping the two in lockstep is the whole point — an
expectation written against the fake must hold against the live org.

Field payloads follow the object documented in CONTRACT_NOTES §4 (``field_name``,
``fully_qualified_name``, ``view_name``, ``data_type``, ``is_dimension``, ``label``,
``aggregate_type``, ``date_type``, ``format``, ``sql``, ``hidden``, ``filter_only_field``); the
same shape is reused for ``summary.fields`` (§2.3), which additionally carries ``is_calc``.

Nothing here performs I/O.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "AVERAGE_SCALE",
    "BENCH_BASE_VIEW",
    "BENCH_CONNECTION_ID",
    "BENCH_DATA_DIR",
    "BENCH_MODEL_ID",
    "BENCH_MODEL_NAME",
    "BENCH_TOPIC",
    "BENCH_TOPIC_NAME",
    "DEFAULT_PERMISSIONS",
    "OTHER_MODELS",
    "TABLE_NAMES",
    "FakeField",
    "FakeModel",
    "FakeRelationship",
    "FakeTopic",
    "FakeView",
    "LiteralKind",
]

#: ``tests/data/bench`` — the parquet tables the fake queries with DuckDB.
BENCH_DATA_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "tests" / "data" / "bench"

BENCH_MODEL_ID: Final = "3f2b1a0c-9d8e-4c7b-a6f5-000000000001"
BENCH_MODEL_NAME: Final = "bench_ecommerce"
BENCH_CONNECTION_ID: Final = "5c7d9e11-4444-4aaa-8bbb-000000000009"
BENCH_TOPIC_NAME: Final = "order_items"
BENCH_BASE_VIEW: Final = "order_items"

#: Warehouse tables registered into DuckDB, in dependency order.
TABLE_NAMES: Final = ("users", "products", "order_items")

#: Model permissions a full-access key reports through ``/whoami`` (CONTRACT_NOTES §1).
DEFAULT_PERMISSIONS: Final = ("QUERY_TOPICS", "QUERY_FULL_MODEL", "QUERY_SQL", "VIEW_SQL")

#: How a date/timestamp filter literal must be rendered for this column in DuckDB.
LiteralKind = str
_TIMESTAMPTZ: Final = "timestamptz"
_DATE: Final = "date"

#: ``AVG`` over a ``decimal(12,2)`` column is a ``DOUBLE`` in DuckDB, and a float cannot be
#: compared exactly against ``known_answers.json``.  The known answers cast the average to
#: ``DECIMAL(38,10)`` for exactness, so ``order_items.average_sale_price`` does the same —
#: see docs/bench_omni_model.md ("offline-only rendering choices").
AVERAGE_SCALE: Final = "DECIMAL(38, 10)"


@dataclass(frozen=True)
class FakeField:
    """One dimension or measure of a view.

    ``column`` is the backing warehouse column and is set for dimensions only; a measure carries
    ``duckdb_sql`` instead — the aggregate expression :mod:`tests.fakes.engine` inlines when the
    measure is selected.  ``sql`` is always the Omni-flavored definition the catalog reports.
    """

    view_name: str
    field_name: str
    data_type: str
    is_dimension: bool
    column: str | None = None
    sql: str = ""
    label: str | None = None
    aggregate_type: str | None = None
    date_type: str | None = None
    literal_kind: LiteralKind | None = None
    duckdb_sql: str | None = None

    @property
    def name(self) -> str:
        """The fully qualified wire name, e.g. ``users.state``."""
        return f"{self.view_name}.{self.field_name}"

    @property
    def display_label(self) -> str:
        return self.label if self.label is not None else self.field_name.replace("_", " ").title()

    def to_wire(self, *, redact_sql: bool = False) -> dict[str, Any]:
        """The Field object as CONTRACT_NOTES §4 / §2.3 spell it.

        Without ``VIEW_SQL`` the ``sql`` key is blanked rather than dropped, matching the way the
        server redacts SQL for callers who may not see it.
        """
        return {
            "field_name": self.field_name,
            "fully_qualified_name": self.name,
            "view_name": self.view_name,
            "data_type": self.data_type,
            "is_dimension": self.is_dimension,
            "is_calc": False,
            "label": self.display_label,
            "format": None,
            "date_type": self.date_type,
            "aggregate_type": self.aggregate_type,
            "filter_only_field": False,
            "hidden": False,
            "sql": "" if redact_sql else self.sql,
        }


@dataclass(frozen=True)
class FakeView:
    """A view inside the topic: its dimensions, measures, and filter-only fields."""

    name: str
    label: str
    dimensions: tuple[FakeField, ...]
    measures: tuple[FakeField, ...] = ()

    @property
    def fields(self) -> tuple[FakeField, ...]:
        return self.dimensions + self.measures

    def to_wire(self, *, redact_sql: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "dimensions": [f.to_wire(redact_sql=redact_sql) for f in self.dimensions],
            "measures": [f.to_wire(redact_sql=redact_sql) for f in self.measures],
            "filter_only_fields": [],
        }


@dataclass(frozen=True)
class FakeRelationship:
    """One join edge of the topic.

    ``sql`` is the Omni-flavored join condition (``${view.field}`` references) that the topic
    endpoint reports; ``on_sql`` is the same condition rendered for DuckDB, used when the fake
    actually executes a query.
    """

    left_view_name: str
    right_view_name: str
    join_type: str
    relationship_type: str
    sql: str
    on_sql: str

    def to_wire(self, *, redact_sql: bool = False) -> dict[str, Any]:
        return {
            "left_view_name": self.left_view_name,
            "right_view_name": self.right_view_name,
            "join_type": self.join_type,
            "relationship_type": self.relationship_type,
            "sql": "" if redact_sql else self.sql,
            "join_from_base_view": True,
            "reversible": False,
        }


@dataclass(frozen=True)
class FakeTopic:
    """The topic: base view, the views reachable through it, and the join edges."""

    name: str
    base_view_name: str
    label: str
    description: str
    group_label: str
    views: tuple[FakeView, ...]
    relationships: tuple[FakeRelationship, ...]
    hidden: bool = False

    def view(self, name: str) -> FakeView | None:
        for candidate in self.views:
            if candidate.name == name:
                return candidate
        return None

    def field(self, name: str) -> FakeField | None:
        """Look a field up by its fully qualified ``view.field`` name."""
        return self._index.get(name)

    @property
    def _index(self) -> Mapping[str, FakeField]:
        return {f.name: f for f in self}

    def __iter__(self) -> Iterator[FakeField]:
        for view in self.views:
            yield from view.fields

    def relationship_to(self, view_name: str) -> FakeRelationship | None:
        for relationship in self.relationships:
            if relationship.right_view_name == view_name:
                return relationship
        return None

    def summary_wire(self) -> dict[str, Any]:
        """The entry the topic *list* endpoint returns (CONTRACT_NOTES §4)."""
        return {
            "name": self.name,
            "base_view_name": self.base_view_name,
            "label": self.label,
            "description": self.description,
            "group_label": self.group_label,
            "hidden": self.hidden,
        }

    def to_wire(self, *, redact_sql: bool = False) -> dict[str, Any]:
        """The topic *detail* payload — the field-metadata endpoint."""
        return {
            **self.summary_wire(),
            "views": [v.to_wire(redact_sql=redact_sql) for v in self.views],
            "relationships": [r.to_wire(redact_sql=redact_sql) for r in self.relationships],
        }


@dataclass(frozen=True)
class FakeModel:
    """A row of ``GET /api/v1/models``."""

    id: str
    name: str
    model_kind: str = "SHARED"
    connection_id: str = BENCH_CONNECTION_ID
    base_model_id: str | None = None
    created_at: str = "2026-06-30T00:00:00.000Z"
    updated_at: str = "2026-06-30T00:00:00.000Z"

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "modelKind": self.model_kind,
            "connectionId": self.connection_id,
            "baseModelId": self.base_model_id,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "deletedAt": None,
        }


# --------------------------------------------------------------------------------------
# Field definitions — one per column of docs/BENCH_DATASET.md, plus the governed measures.
# --------------------------------------------------------------------------------------


def _dimension(
    view: str,
    column: str,
    data_type: str,
    *,
    date_type: str | None = None,
    literal_kind: LiteralKind | None = None,
) -> FakeField:
    return FakeField(
        view_name=view,
        field_name=column,
        data_type=data_type,
        is_dimension=True,
        column=column,
        sql=f"${{TABLE}}.{column}",
        date_type=date_type,
        literal_kind=literal_kind,
    )


def _measure(view: str, name: str, sql: str, aggregate_type: str, duckdb_sql: str) -> FakeField:
    return FakeField(
        view_name=view,
        field_name=name,
        data_type="NUMBER",
        is_dimension=False,
        sql=sql,
        aggregate_type=aggregate_type,
        duckdb_sql=duckdb_sql,
    )


def _qualified(view: str, column: str) -> str:
    return f'"{view}"."{column}"'


USERS_VIEW: Final = FakeView(
    name="users",
    label="Users",
    dimensions=(
        _dimension("users", "id", "NUMBER"),
        _dimension("users", "name", "STRING"),
        _dimension("users", "email", "STRING"),
        _dimension("users", "state", "STRING"),
        _dimension("users", "country", "STRING"),
        _dimension(
            "users", "created_at", "TIMESTAMP", date_type="timestamp", literal_kind=_TIMESTAMPTZ
        ),
        _dimension("users", "age", "NUMBER"),
        _dimension("users", "is_business", "BOOLEAN"),
        _dimension("users", "signup_source", "STRING"),
        _dimension("users", "lifetime_value", "NUMBER"),
    ),
    measures=(
        _measure(
            "users",
            "count",
            "COUNT(DISTINCT ${users.id})",
            "count_distinct",
            f"COUNT(DISTINCT {_qualified('users', 'id')})",
        ),
    ),
)

PRODUCTS_VIEW: Final = FakeView(
    name="products",
    label="Products",
    dimensions=(
        _dimension("products", "id", "NUMBER"),
        _dimension("products", "name", "STRING"),
        _dimension("products", "category", "STRING"),
        _dimension("products", "brand", "STRING"),
        _dimension("products", "cost", "NUMBER"),
        _dimension("products", "price", "NUMBER"),
        _dimension("products", "introduced_on", "TIMESTAMP", date_type="date", literal_kind=_DATE),
    ),
    measures=(
        _measure(
            "products",
            "count",
            "COUNT(DISTINCT ${products.id})",
            "count_distinct",
            f"COUNT(DISTINCT {_qualified('products', 'id')})",
        ),
    ),
)

ORDER_ITEMS_VIEW: Final = FakeView(
    name="order_items",
    label="Order Items",
    dimensions=(
        _dimension("order_items", "id", "NUMBER"),
        _dimension("order_items", "order_id", "NUMBER"),
        _dimension("order_items", "user_id", "NUMBER"),
        _dimension("order_items", "product_id", "NUMBER"),
        _dimension(
            "order_items",
            "created_at",
            "TIMESTAMP",
            date_type="timestamp",
            literal_kind=_TIMESTAMPTZ,
        ),
        _dimension("order_items", "status", "STRING"),
        _dimension("order_items", "quantity", "NUMBER"),
        _dimension("order_items", "sale_price", "NUMBER"),
        _dimension("order_items", "discount", "NUMBER"),
        _dimension("order_items", "returned", "BOOLEAN"),
        _dimension("order_items", "notes", "STRING"),
    ),
    measures=(
        _measure(
            "order_items",
            "total_sale_price",
            "SUM(${order_items.sale_price})",
            "sum",
            f"SUM({_qualified('order_items', 'sale_price')})",
        ),
        _measure("order_items", "count", "COUNT(*)", "count", "COUNT(*)"),
        _measure(
            "order_items",
            "total_quantity",
            "SUM(${order_items.quantity})",
            "sum",
            f"SUM({_qualified('order_items', 'quantity')})",
        ),
        _measure(
            "order_items",
            "average_sale_price",
            "AVG(${order_items.sale_price})",
            "average",
            f"CAST(AVG({_qualified('order_items', 'sale_price')}) AS {AVERAGE_SCALE})",
        ),
    ),
)

#: The one topic the bench model exposes.  Both joins fan in from the fact table, so they are
#: LEFT joins: order_items rows with an orphan ``user_id`` survive with NULL user columns.
BENCH_TOPIC: Final = FakeTopic(
    name=BENCH_TOPIC_NAME,
    base_view_name=BENCH_BASE_VIEW,
    label="Order Items",
    description="Order items joined to the buying user and the purchased product.",
    group_label="Ecommerce",
    views=(ORDER_ITEMS_VIEW, USERS_VIEW, PRODUCTS_VIEW),
    relationships=(
        FakeRelationship(
            left_view_name="order_items",
            right_view_name="users",
            join_type="always_left",
            relationship_type="many_to_one",
            sql="${order_items.user_id} = ${users.id}",
            on_sql='"users"."id" = "order_items"."user_id"',
        ),
        FakeRelationship(
            left_view_name="order_items",
            right_view_name="products",
            join_type="always_left",
            relationship_type="many_to_one",
            sql="${order_items.product_id} = ${products.id}",
            on_sql='"products"."id" = "order_items"."product_id"',
        ),
    ),
)

#: Decoy models so ``GET /models`` pagination has something to page through.  None of them is
#: queryable — a query against one of these ids gets a PLAN job error.
OTHER_MODELS: Final = (
    FakeModel(id="3f2b1a0c-9d8e-4c7b-a6f5-000000000002", name="bench_marketing"),
    FakeModel(id="3f2b1a0c-9d8e-4c7b-a6f5-000000000003", name="bench_finance"),
    FakeModel(id="3f2b1a0c-9d8e-4c7b-a6f5-000000000004", name="bench_support"),
    FakeModel(
        id="3f2b1a0c-9d8e-4c7b-a6f5-000000000005",
        name="bench_ecommerce_branch",
        model_kind="BRANCH",
        base_model_id=BENCH_MODEL_ID,
    ),
)
