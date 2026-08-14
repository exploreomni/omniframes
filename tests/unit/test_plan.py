"""Unit tests for the logical plan nodes and the visitors (docs/INTERNALS.md §2)."""

from __future__ import annotations

from typing import Any

import pytest

from omniframes import functions as F
from omniframes.column import Column, FieldRef, SortKey
from omniframes.compile.querymodel import UNSET
from omniframes.errors import CompileError
from omniframes.plan import (
    Aggregate,
    Filter,
    Join,
    JoinHow,
    Limit,
    MapPandas,
    PlanNode,
    Project,
    SavedQueryScan,
    Scan,
    Sort,
    SqlScan,
    TopicScan,
    Union,
    ViewScan,
    WithColumn,
    collect,
    transform,
    transform_expr,
    walk,
    walk_expr,
)

MODEL_ID = "3f2b1a0c-9d8e-4c7b-a6f5-000000000001"
SCAN = Scan(TopicScan("bench_ecommerce", MODEL_ID, "order_items", "order_items"))


def pipeline() -> PlanNode:
    """Scan → Filter → Project → Sort → Limit, the shape tier 1 accepts."""
    return Limit(
        Sort(
            Project(
                Filter(SCAN, (F.col("users.state") == "California").expr),
                (F.col("users.state"),),
            ),
            (SortKey(FieldRef("users.state")),),
        ),
        10,
    )


# --------------------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------------------


def test_a_scan_is_a_leaf() -> None:
    assert SCAN.children == ()


def test_unary_nodes_expose_their_child() -> None:
    node = Project(SCAN, (F.col("users.state"),))

    assert node.child is SCAN
    assert node.children == (SCAN,)


def test_binary_nodes_expose_both_children() -> None:
    node = Join(SCAN, SCAN, ("users.id",), JoinHow.LEFT)

    assert node.children == (SCAN, SCAN)
    assert Union(SCAN, SCAN).children == (SCAN, SCAN)


def test_nodes_are_frozen() -> None:
    node = Project(SCAN, (F.col("users.state"),))

    with pytest.raises(AttributeError):
        node.child = SCAN  # type: ignore[misc]


def test_sequence_fields_are_normalized_to_tuples() -> None:
    node = Project(SCAN, [F.col("users.state")])  # type: ignore[arg-type]

    assert isinstance(node.columns, tuple)


@pytest.mark.parametrize(
    "source",
    [
        TopicScan("m", MODEL_ID, "order_items", "order_items"),
        ViewScan("m", MODEL_ID, "users"),
        SqlScan(MODEL_ID, "SELECT 1"),
        SavedQueryScan("doc", "saved", {}),
    ],
)
def test_every_scan_source_has_an_explain_label(source: object) -> None:
    assert source.label  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Limit trichotomy
# --------------------------------------------------------------------------------------


def test_limit_defaults_to_unset() -> None:
    """UNSET (no user limit), None (unlimited) and an int are three different states."""
    assert Limit(SCAN).n is UNSET
    assert Limit(SCAN, None).n is None
    assert Limit(SCAN, 10).n == 10


@pytest.mark.parametrize("value", [0, -1, True])
def test_limit_rejects_non_positive_and_boolean_values(value: object) -> None:
    with pytest.raises(CompileError):
        Limit(SCAN, value)  # type: ignore[arg-type]


def test_offset_must_not_be_negative() -> None:
    with pytest.raises(CompileError, match="negative"):
        Limit(SCAN, 10, -1)


# --------------------------------------------------------------------------------------
# Visitors
# --------------------------------------------------------------------------------------


def test_walk_is_pre_order() -> None:
    types = [type(node).__name__ for node in walk(pipeline())]

    assert types == ["Limit", "Sort", "Project", "Filter", "Scan"]


def test_collect_finds_matching_nodes() -> None:
    found = collect(pipeline(), lambda node: isinstance(node, Filter))

    assert len(found) == 1


def test_transform_rebuilds_bottom_up() -> None:
    def drop_limit(node: PlanNode) -> PlanNode:
        return node.child if isinstance(node, Limit) else node

    rewritten = transform(pipeline(), drop_limit)

    assert not any(isinstance(node, Limit) for node in walk(rewritten))
    assert [type(node).__name__ for node in walk(rewritten)] == [
        "Sort",
        "Project",
        "Filter",
        "Scan",
    ]


def test_transform_preserves_identity_when_nothing_changes() -> None:
    plan = pipeline()

    assert transform(plan, lambda node: node) is plan


def test_with_children_checks_arity() -> None:
    node = Project(SCAN, (F.col("users.state"),))

    assert node.with_children((SCAN,)) == node
    with pytest.raises(CompileError, match="exactly one child"):
        node.with_children((SCAN, SCAN))
    with pytest.raises(CompileError, match="takes no children"):
        SCAN.with_children((SCAN,))


def test_expression_walk_and_transform() -> None:
    predicate = ((F.col("v.a") == 1) & (F.col("v.b") == 2)).expr
    names = [node.name for node in walk_expr(predicate) if isinstance(node, FieldRef)]

    assert names == ["v.a", "v.b"]

    renamed = transform_expr(
        predicate,
        lambda expr: FieldRef("v.c") if expr == FieldRef("v.a") else expr,
    )

    assert [node.name for node in walk_expr(renamed) if isinstance(node, FieldRef)] == [
        "v.c",
        "v.b",
    ]


@pytest.mark.parametrize(
    "build",
    [
        lambda udf: udf,
        lambda udf: ~udf,
        lambda udf: SortKey(udf.expr),
        lambda udf: udf.is_null(),
    ],
)
def test_every_node_that_reports_children_can_be_rebuilt_around_them(build: Any) -> None:
    """A ``Udf``'s operands *are* its children, so a rewrite must not fall off the dispatch.

    An identity transform hides the gap — the children compare equal, so the rebuild never
    runs — which is why only a real rename catches it.
    """
    expr = build(F.udf(str.upper)("v.a"))
    expr = expr.expr if isinstance(expr, Column) else expr

    renamed = transform_expr(
        expr, lambda node: FieldRef("v.c") if node == FieldRef("v.a") else node
    )

    assert [node.name for node in walk_expr(renamed) if isinstance(node, FieldRef)] == ["v.c"]


# --------------------------------------------------------------------------------------
# Nodes M1 declares but does not build yet
# --------------------------------------------------------------------------------------


def test_later_milestone_nodes_exist_with_their_documented_shape() -> None:
    """Declared now so M2/M3/M4 add operations rather than reshaping this module."""
    aggregate = Aggregate(SCAN, (F.col("users.state"),), (F.measure("order_items.count"),))
    with_column = WithColumn(SCAN, "doubled", (F.col("users.age") * 2).expr)
    mapped = MapPandas(SCAN, lambda frame: frame)

    assert aggregate.children == (SCAN,)
    assert with_column.name == "doubled"
    assert mapped.schema_hint is None
    assert mapped.with_children((SCAN,)).children == (SCAN,)
