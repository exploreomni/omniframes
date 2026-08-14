"""Generic walking and rewriting for plans and expressions (docs/INTERNALS.md §2).

Two pairs of utilities, each generic over the node types:

* :func:`walk` / :func:`transform` for :class:`~omniframes.plan.nodes.PlanNode` trees;
* :func:`walk_expr` / :func:`transform_expr` for :class:`~omniframes.column.Expr` trees.

``transform`` rebuilds bottom-up: children are transformed first, then the (possibly rebuilt)
node is handed to ``fn``.  Nodes whose children did not change are returned untouched, so a
no-op transform is free and preserves object identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

from omniframes.column import (
    AdHocAgg,
    Arithmetic,
    Between,
    BooleanOp,
    Comparison,
    Expr,
    IsIn,
    IsNull,
    Not,
    SortKey,
    StringPredicate,
    Udf,
)
from omniframes.plan.nodes import PlanNode

__all__ = ["collect", "transform", "transform_expr", "walk", "walk_expr"]


def walk(node: PlanNode) -> Iterator[PlanNode]:
    """Yield every node of the plan, parents before children (pre-order)."""
    yield node
    for child in node.children:
        yield from walk(child)


def collect(node: PlanNode, predicate: Callable[[PlanNode], bool]) -> tuple[PlanNode, ...]:
    """Every node of the plan matching ``predicate``, in pre-order."""
    return tuple(candidate for candidate in walk(node) if predicate(candidate))


def transform(node: PlanNode, fn: Callable[[PlanNode], PlanNode]) -> PlanNode:
    """Rebuild the plan bottom-up, replacing each node with ``fn(node)``."""
    children = node.children
    if children:
        rebuilt = tuple(transform(child, fn) for child in children)
        if rebuilt != children:
            node = node.with_children(rebuilt)
    return fn(node)


def walk_expr(expr: Expr) -> Iterator[Expr]:
    """Yield every sub-expression, parents before children (pre-order)."""
    yield expr
    for child in expr.children:
        yield from walk_expr(child)


def transform_expr(expr: Expr, fn: Callable[[Expr], Expr]) -> Expr:
    """Rebuild an expression bottom-up, replacing each sub-expression with ``fn(expr)``.

    The generic counterpart of the compiler's own rewriters — the splitter's ``_rebind`` and the
    semantic compiler's alias resolution both do this shape of work over the same node set, but
    each does it inline; this is the reusable form, exported from :mod:`omniframes.plan`.

    Every node type that reports ``children`` can be rebuilt, so ``fn`` may replace anything it
    is handed.  A no-op transform returns the original object.
    """
    children = expr.children
    if children:
        rebuilt = tuple(transform_expr(child, fn) for child in children)
        if rebuilt != children:
            expr = _with_expr_children(expr, rebuilt)
    return fn(expr)


def _with_expr_children(expr: Expr, children: tuple[Expr, ...]) -> Expr:
    """Rebuild one expression node around new children (arity is fixed per node type)."""
    if isinstance(expr, AdHocAgg):
        raise TypeError("AdHocAgg operands are field references and are never rewritten")
    if isinstance(expr, Comparison):
        return Comparison(expr.op, children[0], children[1])
    if isinstance(expr, Arithmetic):
        return Arithmetic(expr.op, children[0], children[1])
    if isinstance(expr, BooleanOp):
        return BooleanOp(expr.op, children)
    if isinstance(expr, Not):
        return Not(children[0])
    if isinstance(expr, IsNull):
        return IsNull(children[0])
    if isinstance(expr, IsIn):
        return IsIn(children[0], expr.values)
    if isinstance(expr, StringPredicate):
        return StringPredicate(expr.kind, children[0], expr.value, expr.case_insensitive)
    if isinstance(expr, Between):
        return Between(children[0], expr.low, expr.high)
    if isinstance(expr, SortKey):
        return SortKey(children[0], expr.descending)
    if isinstance(expr, Udf):
        # A UDF's operands ARE its children, so a node that reports them has to be rebuildable
        # around them; the splitter's `_rebind` has always done exactly this.
        return Udf(expr.fn, children, expr.name)
    raise TypeError(f"{type(expr).__name__} has no children to replace")
