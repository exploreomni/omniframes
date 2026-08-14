"""The ``F`` namespace (docs/INTERNALS.md §1)::

    from omniframes import functions as F

    df.filter(F.col("users.state") == "California")

String arguments are accepted anywhere a Column is — they are auto-wrapped through :func:`col`,
so ``df.sort("users.state")`` and ``df.sort(F.col("users.state"))`` are the same thing.

The aggregation helpers build :class:`~omniframes.column.AdHocAgg` nodes.  Those are *ad-hoc*
aggregations (computed over raw columns) and are deliberately distinct from :func:`measure`,
which references a governed model measure: a measure always executes remotely, under its
server-side definition, while an ad-hoc aggregation is written as SQL over a governed reference
core and executed in the warehouse (tier 2, docs/SQLTIER.md) — or, when that is not expressible,
computed here over a raw scan (docs/HYBRID.md §2.1).  Mixing the two in one ``agg()`` is fine:
that is the case the splitter decomposes, and ``explain()`` shows which tier took which half.

:func:`udf` is the escape hatch: any Python function, applied row by row.  Nothing about a UDF
is expressible on the wire, so everything from it up executes locally — ``explain()`` shows
exactly where that starts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omniframes.column import (
    AdHocAgg,
    AggFn,
    Column,
    FieldRef,
    Literal,
    LiteralValue,
    MeasureRef,
    Udf,
)
from omniframes.errors import CompileError

__all__ = [
    "avg",
    "col",
    "count",
    "count_distinct",
    "lit",
    "max",
    "measure",
    "min",
    "sum",
    "udf",
]


def col(name: str | Column) -> Column:
    """A reference to a model field, by fully qualified name (``"users.state"``)."""
    if isinstance(name, Column):
        return name
    if not isinstance(name, str):
        raise CompileError(f"col() takes a field name or a Column, got {type(name).__name__}")
    return Column(FieldRef(name))


def lit(value: LiteralValue) -> Column:
    """A constant."""
    return Column(Literal(value))


def measure(name: str) -> Column:
    """A governed model measure (``F.measure("order_items.total_sale_price")``).

    Measure definitions live server-side; omniframes never emulates one locally.
    """
    if not isinstance(name, str) or not name:
        raise CompileError("measure() takes the fully qualified measure name, e.g. 'view.measure'")
    return Column(MeasureRef(name))


def _agg(fn: AggFn, column: str | Column, *, distinct: bool = False) -> Column:
    operand = col(column).expr
    if isinstance(operand, MeasureRef):
        raise CompileError(
            f"{operand.name!r} is a governed measure and is already aggregated; select it with "
            "F.measure(...) instead of wrapping it in an ad-hoc aggregation"
        )
    if not isinstance(operand, FieldRef):
        raise CompileError(
            f"{fn.value}() aggregates a field reference; got {operand!r}. Expressions inside an "
            "aggregate are not supported yet."
        )
    return Column(AdHocAgg(fn, operand, distinct))


def sum(column: str | Column) -> Column:
    """Ad-hoc ``SUM`` over a raw column.

    Shadows the builtin inside this module on purpose: ``F.sum`` is the name PySpark users
    reach for, and the module is meant to be imported as ``F``, never star-imported.
    """
    return _agg(AggFn.SUM, column)


def avg(column: str | Column) -> Column:
    """Ad-hoc ``AVG`` over a raw column."""
    return _agg(AggFn.AVG, column)


def min(column: str | Column) -> Column:
    """Ad-hoc ``MIN`` over a raw column (shadows the builtin — see :func:`sum`)."""
    return _agg(AggFn.MIN, column)


def max(column: str | Column) -> Column:
    """Ad-hoc ``MAX`` over a raw column (shadows the builtin — see :func:`sum`)."""
    return _agg(AggFn.MAX, column)


def count(column: str | Column) -> Column:
    """Ad-hoc ``COUNT`` over a raw column."""
    return _agg(AggFn.COUNT, column)


def count_distinct(column: str | Column) -> Column:
    """Ad-hoc ``COUNT(DISTINCT …)``; its default column name is ``count_distinct(<field>)``."""
    return _agg(AggFn.COUNT, column, distinct=True)


def udf(fn: Callable[..., Any]) -> Callable[..., Column]:
    """Wrap a Python function so it can be applied to columns (docs/HYBRID.md §5)::

        upper = F.udf(str.upper)
        df.with_column("shout", upper("users.state"))

    The function is **scalar**: it is called once per row with that row's operand values, and
    its return value becomes the cell.  Use :meth:`~omniframes.dataframe.DataFrame.map_pandas`
    for a vectorized function over the whole frame.

    A UDF is never expressible on the wire, so everything from the UDF up executes locally; the
    query below it is still pushed down as far as it goes, and ``explain()`` shows the split.
    """
    if not callable(fn):
        raise CompileError(f"udf() takes a callable, got {type(fn).__name__}")

    def apply(*columns: str | Column) -> Column:
        if not columns:
            raise CompileError("a UDF needs at least one column argument, e.g. my_udf('users.id')")
        # Imported here rather than at module scope: the display-name rules belong to the
        # compiler, and `F` must stay importable without pulling the compiler in.
        from omniframes.compile.semantic import display_name

        operands = tuple(col(column).expr for column in columns)
        label = getattr(fn, "__name__", None) or type(fn).__name__
        rendered = ", ".join(display_name(operand) for operand in operands)
        return Column(Udf(fn, operands, f"{label}({rendered})"))

    return apply
