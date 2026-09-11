"""Tier 3 — the local engine (docs/HYBRID.md §3).

A handful of pure operators over ``pyarrow`` Tables, plus the expression evaluator they share.
Every operator is ``(inputs: tuple[pa.Table, ...]) -> pa.Table`` with no session, no I/O and no
hidden state, so the whole tier is testable from a literal table.

**Why Arrow and not pandas.**  ``pyarrow.compute`` gives SQL parity for free: Kleene
three-valued ``and_kleene``/``or_kleene``/``invert``, null-propagating comparisons, a hash
``group_by`` that keeps the NULL-key group (SQL ``GROUP BY`` semantics), ``count_distinct``
that excludes NULL, decimal128 arithmetic and ``sort_indices(null_placement=…)``.  pandas would
need a per-dtype workaround for every one of those.  pandas therefore appears in exactly two
places here — inside :class:`LocalMapPandas` and :class:`Udf` evaluation (the user's function
wants a frame) and inside :class:`AlignJoin` (its factorization matches NA keys, which is
precisely the alignment a decomposed aggregate needs).

**Governed measures are never emulated.**  A :class:`~omniframes.column.MeasureRef` is legal in
a local expression only as a *reference to a column a remote step already computed*; if the name
is not on the table, that is an error, not an invitation to compute an average.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import pyarrow as pa
import pyarrow.compute as pc

from omniframes.column import (
    AdHocAgg,
    AggFn,
    Arithmetic,
    ArithOp,
    Between,
    BooleanOp,
    BoolOp,
    CmpOp,
    Comparison,
    Expr,
    FieldRef,
    IsIn,
    IsNull,
    Literal,
    MeasureRef,
    Not,
    StringPredicate,
    StrPredKind,
    Udf,
)
from omniframes.compile.semantic import CannotCompile, display_name
from omniframes.errors import CompileError
from omniframes.plan.nodes import JoinHow

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omniframes.types import OmniSchema

__all__ = [
    "DECIMAL128_MAX_PRECISION",
    "AlignJoin",
    "LocalAgg",
    "LocalAggregate",
    "LocalFilter",
    "LocalJoin",
    "LocalLimit",
    "LocalMapPandas",
    "LocalOp",
    "LocalProject",
    "LocalSort",
    "LocalUnion",
    "LocalWithColumn",
    "describe_expr",
    "eval_expr",
    "join_collision_message",
    "promote_type",
    "run_local_op",
]

#: The widest ``decimal128`` Arrow has.  ``sum`` over a decimal promotes to this precision
#: (docs/HYBRID.md §3.3) — enforced here rather than left to Arrow, which only started widening
#: a grouped decimal sum in pyarrow 21.
DECIMAL128_MAX_PRECISION: Final = 38

# pyarrow's compute stubs describe results loosely (``Array`` vs ``ChunkedArray`` vs
# ``Scalar``), so evaluated values flow untyped inside this module; the public signatures are
# precise about the table in and the table out.
Value: TypeAlias = Any

_RELATIVE_DATES = (
    "relative date literals are evaluated by Omni: a string compared to a date column only has "
    'a meaning server-side ("30 days ago", "last quarter"), so this predicate cannot be '
    "evaluated locally. Compare to a date/datetime, or keep the filter inside the pushed-down "
    "part of the query."
)


# --------------------------------------------------------------------------------------
# Expression evaluation (docs/HYBRID.md §3.1)
# --------------------------------------------------------------------------------------


def eval_expr(expr: Expr, table: pa.Table) -> Value:
    """Evaluate ``expr`` against ``table``, returning an Arrow array (or a scalar for literals).

    Null semantics are SQL's, because they are Arrow's: comparisons propagate NULL, ``AND``/
    ``OR`` are Kleene, and ``NOT NULL`` stays NULL (a row a :class:`LocalFilter` then drops).

    Raises:
        CompileError: the expression names a column the table does not have.
        CannotCompile: the expression is only meaningful server-side (relative date literals).
    """
    if isinstance(expr, Literal):
        return _scalar(expr.value)
    if isinstance(expr, FieldRef | MeasureRef | AdHocAgg):
        return _column(expr, table)
    if isinstance(expr, Comparison):
        return _comparison(expr, table)
    if isinstance(expr, BooleanOp):
        return _boolean(expr, table)
    if isinstance(expr, Not):
        return pc.invert(eval_expr(expr.operand, table))
    if isinstance(expr, IsNull):
        return pc.is_null(eval_expr(expr.operand, table))
    if isinstance(expr, IsIn):
        return _is_in(expr, table)
    if isinstance(expr, StringPredicate):
        return _string_predicate(expr, table)
    if isinstance(expr, Between):
        return _between(expr, table)
    if isinstance(expr, Arithmetic):
        return _arithmetic(expr, table)
    if isinstance(expr, Udf):
        return _udf(expr, table)
    raise CompileError(f"{display_name(expr)} is not an expression the local engine evaluates")


def _scalar(value: object) -> Value:
    """A Python literal as an Arrow scalar; Arrow infers the type, decimals included."""
    make: Any = pa.scalar
    return make(value)


def _column(expr: Expr, table: pa.Table) -> Value:
    """Resolve a reference against the table's columns: exact name, then wire name.

    Tables reaching a local operator are already normalized and aliased, so the name a user
    wrote is the name on the table — the splitter rewrites wire names to output names before
    the expression ever gets here.
    """
    names = table.column_names
    for candidate in _candidates(expr):
        if candidate in names:
            return table.column(candidate)
    available = ", ".join(names) or "(none)"
    if isinstance(expr, MeasureRef):
        raise CompileError(
            f"{expr.name!r} is not a column of this intermediate result: governed measures only "
            f"exist remotely, so a local expression can only reference one that a remote step "
            f"already computed. This result has: {available}"
        )
    if isinstance(expr, AdHocAgg):
        raise CompileError(
            f"{display_name(expr)} is not a column of this intermediate result; an aggregation "
            f"can only be referenced above the aggregate that produced it. This result has: "
            f"{available}"
        )
    raise CompileError(
        f"{display_name(expr)} is not a column of this intermediate result. It has: {available}"
    )


def _candidates(expr: Expr) -> tuple[str, ...]:
    if isinstance(expr, FieldRef):
        if expr.grain is None:
            return (expr.name,)
        return (f"{expr.name}[{expr.grain}]", expr.name)
    if isinstance(expr, MeasureRef):
        return (expr.name,)
    return (display_name(expr),)


_COMPARISONS: Final[Mapping[CmpOp, str]] = {
    CmpOp.EQ: "equal",
    CmpOp.NE: "not_equal",
    CmpOp.LT: "less",
    CmpOp.LE: "less_equal",
    CmpOp.GT: "greater",
    CmpOp.GE: "greater_equal",
}


def _comparison(expr: Comparison, table: pa.Table) -> Value:
    left = eval_expr(expr.left, table)
    right = eval_expr(expr.right, table)
    _refuse_date_grammar(left, right)
    _refuse_date_grammar(right, left)
    return getattr(pc, _COMPARISONS[expr.op])(left, right)


def _refuse_date_grammar(column: Value, literal: Value) -> None:
    """A string compared to a temporal column is Omni's date grammar, not a local comparison."""
    if _is_temporal(column) and _is_string(literal):
        raise CannotCompile(_RELATIVE_DATES)


def _is_in(expr: IsIn, table: pa.Table) -> Value:
    """``IN (...)`` with SQL's three-valued result: a NULL operand is NULL, not ``False``.

    ``pc.is_in`` answers a *set membership* question and returns ``False`` for a NULL operand,
    which is right for the positive form and wrong the moment the predicate is negated:
    ``NOT FALSE`` is ``True``, so ``~col.isin(...)`` would resurrect every NULL row that SQL —
    and tiers 1 and 2 — drop (docs/mental-model.md §"A negated predicate never resurrects
    NULLs", docs/HYBRID.md §3.1).  Masking the null positions back to NULL restores Kleene
    parity in both directions.
    """
    operand = eval_expr(expr.operand, table)
    matched = pc.is_in(operand, value_set=pa.array(list(expr.values)))
    return pc.if_else(pc.is_null(operand), pa.scalar(None, pa.bool_()), matched)


def _boolean(expr: BooleanOp, table: pa.Table) -> Value:
    combine = pc.and_kleene if expr.op is BoolOp.AND else pc.or_kleene
    operands = [eval_expr(operand, table) for operand in expr.operands]
    result = operands[0]
    for operand in operands[1:]:
        result = combine(result, operand)
    return result


_STRING_PREDICATES: Final[Mapping[StrPredKind, str]] = {
    StrPredKind.CONTAINS: "match_substring",
    StrPredKind.STARTS_WITH: "starts_with",
    StrPredKind.ENDS_WITH: "ends_with",
    StrPredKind.LIKE: "match_like",
}


def _string_predicate(expr: StringPredicate, table: pa.Table) -> Value:
    operand = eval_expr(expr.operand, table)
    function = getattr(pc, _STRING_PREDICATES[expr.kind])
    # Case-SENSITIVE by default, matching the wire (CONTRACT_NOTES §3.1).
    return function(operand, expr.value, ignore_case=expr.case_insensitive)


def _between(expr: Between, table: pa.Table) -> Value:
    operand = eval_expr(expr.operand, table)
    low = _scalar(expr.low)
    high = _scalar(expr.high)
    if _is_temporal(operand) or _is_temporal(low) or _is_temporal(high):
        if _is_string(low) or _is_string(high):
            raise CannotCompile(_RELATIVE_DATES)
        # Half-open [low, high), exactly what the compiled date filter sends.
        return pc.and_kleene(pc.greater_equal(operand, low), pc.less(operand, high))
    # Inclusive at both ends over numbers, matching Column.between and the compiled composite.
    return pc.and_kleene(pc.greater_equal(operand, low), pc.less_equal(operand, high))


_ARITHMETIC: Final[Mapping[ArithOp, str]] = {
    ArithOp.ADD: "add",
    ArithOp.SUB: "subtract",
    ArithOp.MUL: "multiply",
    ArithOp.DIV: "divide",
}


def _arithmetic(expr: Arithmetic, table: pa.Table) -> Value:
    left = eval_expr(expr.left, table)
    right = eval_expr(expr.right, table)
    if expr.op is ArithOp.DIV and _is_integer(left) and _is_integer(right):
        # SQL's `/` over integers is not integer division here: promote once, like every
        # analytics engine a user of this library has met.
        left = pc.cast(left, pa.float64())
        right = pc.cast(right, pa.float64())
    return getattr(pc, _ARITHMETIC[expr.op])(left, right)


def _udf(expr: Udf, table: pa.Table) -> Value:
    """Apply a scalar Python function row-wise, through pandas (the only place pandas appears)."""
    import pandas as pd

    if table.num_rows == 0:
        return pa.array([], type=pa.null())
    columns = {
        f"a{index}": _as_array(eval_expr(operand, table), table.num_rows)
        for index, operand in enumerate(expr.operands)
    }
    frame = pa.table(columns).to_pandas(types_mapper=pd.ArrowDtype)
    # ``dtype=object, na_value=None`` is load-bearing: the default numpy view of a nullable
    # integer column is float64, which would hand the function 40.0 where the data says 40 —
    # and a missing value reaches it as ``None``, because a UDF is plain Python and Python's
    # absent value is None (this is also what PySpark hands a scalar udf).
    series: Any = frame
    values = [series[name].to_numpy(dtype=object, na_value=None) for name in frame.columns]
    if len(values) == 1:
        applied = pd.Series(values[0], dtype=object).apply(expr.fn)
    else:
        operands = pd.DataFrame({index: column for index, column in enumerate(values)})
        applied = operands.apply(lambda row: expr.fn(*row), axis=1)
    try:
        return pa.Array.from_pandas(applied)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise CompileError(
            f"{expr.name} returned values Arrow cannot hold in one column ({exc}); return a "
            "single consistent type per row"
        ) from exc


# --------------------------------------------------------------------------------------
# Type helpers
# --------------------------------------------------------------------------------------


def _type_of(value: Value) -> pa.DataType | None:
    dtype = getattr(value, "type", None)
    return dtype if isinstance(dtype, pa.DataType) else None


def _is_temporal(value: Value) -> bool:
    dtype = _type_of(value)
    return dtype is not None and (
        pa.types.is_timestamp(dtype) or pa.types.is_date(dtype) or pa.types.is_time(dtype)
    )


def _is_string(value: Value) -> bool:
    dtype = _type_of(value)
    return dtype is not None and (pa.types.is_string(dtype) or pa.types.is_large_string(dtype))


def _is_integer(value: Value) -> bool:
    dtype = _type_of(value)
    return dtype is not None and pa.types.is_integer(dtype)


def _as_array(value: Value, rows: int) -> Value:
    """Broadcast a scalar to a full column so an operator can always work column-wise."""
    if isinstance(value, pa.Scalar):
        return pa.array([value.as_py()] * rows, type=value.type)
    return value


def promote_type(left: pa.DataType, right: pa.DataType, *, column: str) -> pa.DataType:
    """The type both sides of a union (or a join key) can be read as — docs/HYBRID.md §3.3.

    Widening only, and only where SQL would widen: an integer next to a float is a float, a
    decimal keeps its scale, and a string next to a number is a :class:`CompileError` rather
    than a stringified column nobody asked for.

    Raises:
        CompileError: the two types have no common type (the message names the column).
    """
    if left.equals(right):
        return left
    if pa.types.is_null(left):
        return right
    if pa.types.is_null(right):
        return left

    promoted = _promote(left, right) or _promote(right, left)
    if promoted is None:
        raise CompileError(
            f"the column {column!r} is {left} on one side and {right} on the other, and those "
            "two types have no common type. Cast (or re-select) one side so both agree."
        )
    return promoted


def _promote(left: pa.DataType, right: pa.DataType) -> pa.DataType | None:
    """One direction of :func:`promote_type`; the caller tries both orders."""
    if pa.types.is_integer(left) and pa.types.is_integer(right):
        return pa.int64()
    if pa.types.is_floating(left) and (
        pa.types.is_floating(right) or pa.types.is_integer(right) or pa.types.is_decimal(right)
    ):
        # A decimal next to a float loses exactness either way; float is the only type that can
        # hold both, and it is what the local average already produces (§3.3).
        return pa.float64()
    if pa.types.is_decimal(left) and pa.types.is_decimal(right):
        return pa.decimal128(38, max(left.scale, right.scale))
    if pa.types.is_decimal(left) and pa.types.is_integer(right):
        return pa.decimal128(38, left.scale)
    if _is_text(left) and _is_text(right):
        return pa.large_string() if _is_large_text(left) or _is_large_text(right) else pa.string()
    if pa.types.is_timestamp(left) and pa.types.is_date(right):
        # A date is a timestamp at midnight; two *timestamps* that disagree on unit or zone are
        # deliberately not reconciled here — guessing a zone would move values.
        return left
    return None


def _is_text(dtype: pa.DataType) -> bool:
    return bool(pa.types.is_string(dtype) or pa.types.is_large_string(dtype))


def _is_large_text(dtype: pa.DataType) -> bool:
    return bool(pa.types.is_large_string(dtype))


def _mask(value: Value, rows: int) -> Value:
    """A predicate's value as a boolean row mask (NULL keeps the row out — SQL's WHERE)."""
    column = _as_array(value, rows)
    dtype = _type_of(column)
    if dtype is not None and pa.types.is_null(dtype):
        return pa.array([None] * rows, type=pa.bool_())
    if dtype is not None and not pa.types.is_boolean(dtype):
        raise CompileError(
            f"a filter predicate must be boolean; this one produced {dtype}. Compare the column "
            "to a value, e.g. F.col('users.age') > 30"
        )
    return column


# --------------------------------------------------------------------------------------
# Operators (docs/HYBRID.md §3)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalOp:
    """One local operator: a pure function from its inputs to a table."""

    @property
    def kind(self) -> str:
        """The operator's name in ``explain()`` — also how an operator above refers to it."""
        raise NotImplementedError

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        """Apply the operator."""
        raise NotImplementedError

    def detail(self) -> str:
        """What this operator does, rendered for ``explain()``."""
        raise NotImplementedError

    def describe(self, inputs: Sequence[str]) -> str:
        """One ``explain()`` line: ``"filter over step 1: …"`` when the input is a query,
        ``"filter: …"`` when it reads the operator on the line above."""
        source = inputs[0] if inputs else ""
        head = f"{self.kind} over {source}" if source.startswith("step ") else self.kind
        return f"{head}: {self.detail()}"

    @staticmethod
    def _one(inputs: tuple[pa.Table, ...]) -> pa.Table:
        if len(inputs) != 1:
            raise CompileError("this local operator takes exactly one input")
        return inputs[0]


@dataclass(frozen=True)
class LocalFilter(LocalOp):
    """``WHERE`` over an already-materialized table. A NULL predicate drops the row."""

    predicate: Expr

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        table = self._one(inputs)
        return table.filter(_mask(eval_expr(self.predicate, table), table.num_rows))

    kind = "filter"

    def detail(self) -> str:
        return describe_expr(self.predicate)


@dataclass(frozen=True)
class LocalProject(LocalOp):
    """Select columns in order and rename them — this is where widened columns are dropped."""

    names: tuple[str, ...]
    renames: Mapping[str, str] = field(default_factory=dict)

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        table = self._one(inputs)
        missing = [name for name in self.names if name not in table.column_names]
        if missing:
            available = ", ".join(table.column_names) or "(none)"
            raise CompileError(
                f"{', '.join(missing)} is not available in the result being projected; it has: "
                f"{available}"
            )
        selected = table.select(list(self.names))
        if not self.renames:
            return selected
        return selected.rename_columns([self.renames.get(name, name) for name in self.names])

    kind = "project"

    def detail(self) -> str:
        return f"[{', '.join(self.renames.get(name, name) for name in self.names)}]"


@dataclass(frozen=True)
class LocalWithColumn(LocalOp):
    """Append (or replace) one derived column."""

    name: str
    expr: Expr

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        table = self._one(inputs)
        column = _as_array(eval_expr(self.expr, table), table.num_rows)
        if self.name in table.column_names:
            return table.set_column(table.column_names.index(self.name), self.name, column)
        return table.append_column(self.name, column)

    kind = "with column"

    def detail(self) -> str:
        return f"{self.name} = {describe_expr(self.expr)}"


@dataclass(frozen=True)
class LocalAgg:
    """One ad-hoc aggregation of a :class:`LocalAggregate`."""

    name: str
    fn: AggFn
    operand: str
    distinct: bool = False

    @property
    def label(self) -> str:
        kind = "count_distinct" if self.distinct and self.fn is AggFn.COUNT else self.fn.value
        return f"{kind}({self.operand}) AS {self.name}"


#: ``AggFn`` → the pyarrow hash-aggregate function that implements it.
_AGGREGATES: Final[Mapping[AggFn, str]] = {
    AggFn.SUM: "sum",
    AggFn.COUNT: "count",
    AggFn.AVG: "mean",
    AggFn.MIN: "min",
    AggFn.MAX: "max",
}


@dataclass(frozen=True)
class LocalAggregate(LocalOp):
    """``GROUP BY`` with SQL null semantics: a NULL key is a group, like any other value.

    Aggregate null handling matches SQL too (docs/HYBRID.md §3.2): ``sum``/``min``/``max``/
    ``avg`` skip nulls and return NULL for an all-null group, ``count(col)`` counts non-null
    values, and ``count_distinct`` excludes NULL.
    """

    keys: tuple[str, ...]
    aggs: tuple[LocalAgg, ...]

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        table = self._one(inputs)
        for key in self.keys:
            if key not in table.column_names:
                raise CompileError(f"the group key {key!r} is not a column of the scanned rows")

        work = table
        requests: list[tuple[str, str, Any]] = []
        produced: list[str] = []
        for index, agg in enumerate(self.aggs):
            # One private copy per aggregate: two aggregates over the same column with the same
            # function would otherwise collide on pyarrow's `<column>_<function>` output name.
            temporary = f"__agg_{index}"
            work = work.append_column(temporary, _agg_operand(table, agg))
            function = _AGGREGATES[agg.fn]
            options: Any = None
            if agg.fn is AggFn.COUNT:
                # "only_valid" is both pyarrow's default and SQL's: COUNT(col) and
                # COUNT(DISTINCT col) ignore NULL. Spelled out because it is load-bearing.
                function = "count_distinct" if agg.distinct else "count"
                options = pc.CountOptions(mode="only_valid")
            requests.append((temporary, function, options))
            produced.append(f"{temporary}_{function}")

        # use_threads=False keeps the group order deterministic, which every snapshot relies on.
        group_by: Any = pa.TableGroupBy(work, list(self.keys), use_threads=False)
        grouped: pa.Table = group_by.aggregate(requests)
        wanted = [*self.keys, *produced]
        return grouped.select(wanted).rename_columns([*self.keys, *(agg.name for agg in self.aggs)])

    kind = "aggregate"

    def detail(self) -> str:
        keys = ", ".join(self.keys)
        return f"keys=[{keys}], aggs=[{', '.join(agg.label for agg in self.aggs)}]"


def _agg_operand(table: pa.Table, agg: LocalAgg) -> Value:
    """The column an aggregate consumes, cast where the promotion table says so (§3.3)."""
    if agg.operand not in table.column_names:
        raise CompileError(
            f"{agg.operand!r} is not a column of the scanned rows, so {agg.name} cannot be computed"
        )
    column = table.column(agg.operand)
    dtype = column.type
    numeric = (
        pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)
    )
    if agg.fn in (AggFn.SUM, AggFn.AVG) and not numeric:
        raise CompileError(
            f"{agg.fn.value}() needs a numeric column; {agg.operand} is {dtype}. Count it, or "
            "take min/max instead."
        )
    if agg.fn in (AggFn.MIN, AggFn.MAX) and pa.types.is_boolean(dtype):
        raise CompileError(f"{agg.fn.value}() has no meaning over the boolean column {agg.operand}")
    if agg.fn is AggFn.AVG and pa.types.is_decimal(dtype):
        # Arrow's mean over decimal128 rounds to the operand's scale; a float average is the
        # honest answer and is what the differential comparator compares with a tolerance.
        return pc.cast(column, pa.float64())
    if (
        agg.fn is AggFn.SUM
        and pa.types.is_decimal128(dtype)
        and dtype.precision < DECIMAL128_MAX_PRECISION
    ):
        # docs/HYBRID.md §3.3 pins `decimal128(p, s) | sum -> decimal128(38, s)`.  Arrow only
        # widens the *result* of a grouped sum from pyarrow 21; on the older releases this
        # project still supports it keeps the operand's precision, so a group sum that outgrows
        # it is silently wrong — and at precision <= 18 parquet stores the result as an INT64
        # that wraps on read.  Widening the operand produces the documented type on every
        # supported version, and is a no-op where Arrow already does it.
        return pc.cast(column, pa.decimal128(DECIMAL128_MAX_PRECISION, dtype.scale))
    return column


@dataclass(frozen=True)
class LocalSort(LocalOp):
    """``ORDER BY`` with nulls last in both directions — deterministic and stable."""

    keys: tuple[tuple[str, bool], ...]

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        table = self._one(inputs)
        missing = [name for name, _ in self.keys if name not in table.column_names]
        if missing:
            raise CompileError(f"cannot sort by {', '.join(missing)}: not in the result")
        return table.take(_sort_indices(table, self.keys))

    kind = "sort"

    def detail(self) -> str:
        return ", ".join(f"{name} {'desc' if desc else 'asc'}" for name, desc in self.keys)


def _sort_indices(table: pa.Table, keys: Sequence[tuple[str, bool]]) -> Value:
    """``sort_indices`` with nulls at the end, across the pyarrow versions the project allows."""
    ordered = [(name, "descending" if descending else "ascending") for name, descending in keys]
    sort_indices: Any = pc.sort_indices
    try:
        # pyarrow ≥ 25 wants the null placement per key and deprecates the keyword form; older
        # releases only understand the keyword. Both spell the same order.
        return sort_indices(table, sort_keys=[(*key, "at_end") for key in ordered])
    except (TypeError, ValueError, pa.ArrowInvalid):  # pragma: no cover - older pyarrow
        return sort_indices(table, sort_keys=ordered, null_placement="at_end")


@dataclass(frozen=True)
class LocalLimit(LocalOp):
    """``LIMIT n OFFSET k``. ``n=None`` is "everything after the offset"."""

    n: int | None
    offset: int = 0
    #: True when this limit is ``decomposition_row_cap`` applied here because the raw scan under
    #: a local aggregate could not carry it on the wire.  It is not a page the user asked for:
    #: it trims the *input* to an aggregate, so hitting it makes the answer wrong rather than
    #: short — and the executor therefore warns, exactly as a capped remote scan does
    #: (docs/HYBRID.md §2.1: "opt-in cap, never a silent default").
    decomposition_cap: bool = False

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        table = self._one(inputs)
        if self.n is None:
            return table.slice(self.offset)
        return table.slice(self.offset, self.n)

    kind = "limit"

    def detail(self) -> str:
        rendered = "unlimited" if self.n is None else str(self.n)
        if self.decomposition_cap:
            rendered = f"{rendered} (decomposition cap)"
        return f"{rendered}   offset: {self.offset}" if self.offset else rendered


@dataclass(frozen=True)
class LocalMapPandas(LocalOp):
    """Hand the materialized frame to a user function and take a frame back."""

    fn: Callable[..., Any]
    schema_hint: OmniSchema | None = None

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        import pandas as pd

        table = self._one(inputs)
        result = self.fn(table.to_pandas(types_mapper=pd.ArrowDtype))
        if not isinstance(result, pd.DataFrame):
            raise CompileError(
                f"map_pandas({_function_name(self.fn)}) must return a pandas DataFrame; it "
                f"returned {type(result).__name__}"
            )
        return pa.Table.from_pandas(result, preserve_index=False).replace_schema_metadata(None)

    kind = "map_pandas"

    def detail(self) -> str:
        return _function_name(self.fn)


def _function_name(fn: Callable[..., Any]) -> str:
    return str(getattr(fn, "__name__", None) or type(fn).__name__)


@dataclass(frozen=True)
class AlignJoin(LocalOp):
    """Re-assemble the two halves of a decomposed aggregate (docs/HYBRID.md §3.2).

    Both inputs aggregate **the same underlying rows**, one group per row, so their group keys
    line up one-to-one — including the NULL-key group, which must produce ONE output row.  That
    is the opposite of SQL join semantics, which is exactly why this operator exists and why the
    user-facing ``Join`` must not reuse it: pandas' outer merge matches NA keys, giving the
    alignment for free with no sentinel values.
    """

    on: tuple[str, ...]

    kind = "align-join"

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        import pandas as pd

        if len(inputs) != 2:
            raise CompileError("an align-join takes exactly two inputs")
        left, right = inputs
        frames = [side.to_pandas(types_mapper=pd.ArrowDtype) for side in (left, right)]
        if not self.on:
            # A group-less aggregate: one row on each side, so alignment is concatenation.
            extra = [name for name in frames[1].columns if name not in frames[0].columns]
            merged = pd.concat([frames[0], frames[1][extra]], axis=1)
        else:
            merged = frames[0].merge(frames[1], how="outer", on=list(self.on))
        return pa.Table.from_pandas(merged, preserve_index=False).replace_schema_metadata(None)

    def detail(self) -> str:
        return f"on [{', '.join(self.on)}]"

    def describe(self, inputs: Sequence[str]) -> str:
        left, right = (inputs[0], inputs[1]) if len(inputs) == 2 else ("?", "?")
        return f"{self.kind}: {left} \u2a1d {right} {self.detail()}"


#: ``JoinHow`` \u2192 the ``how`` pandas' merge takes for the **non-null-keyed** rows.  The null-keyed
#: rows never take part in the merge; they are re-attached afterwards by :class:`LocalJoin`.
_MERGE_HOW: Final[Mapping[JoinHow, str]] = {
    JoinHow.INNER: "inner",
    JoinHow.LEFT: "left",
    JoinHow.RIGHT: "right",
    JoinHow.OUTER: "outer",
}

#: Which side's unmatched NULL-key rows survive each join kind (SQL: a NULL key matches nothing,
#: so those rows are exactly as unmatched as a key with no counterpart).
_KEEPS_NULL_KEYS: Final[Mapping[JoinHow, tuple[bool, bool]]] = {
    JoinHow.INNER: (False, False),
    JoinHow.LEFT: (True, False),
    JoinHow.RIGHT: (False, True),
    JoinHow.OUTER: (True, True),
}


@dataclass(frozen=True)
class LocalJoin(LocalOp):
    """The **user-facing** join: an equi-join on shared column names, with SQL null semantics.

    The one rule that makes this operator different from :class:`AlignJoin`: **a NULL key never
    matches anything**, not even another NULL.  pandas' ``merge`` matches NA keys to each other,
    which is right for aligning a decomposed aggregate and wrong for a join, so the null-keyed
    rows are held out of the merge entirely and re-attached as unmatched rows afterwards \u2014
    dropped by an inner join, kept (padded with NULLs) by the side an outer join preserves.

    Overlapping non-key column names are refused by the splitter before the plan ever runs, so
    the output is simply: the join keys, then the left frame's other columns, then the right
    frame's.
    """

    on: tuple[str, ...]
    how: JoinHow = JoinHow.INNER

    kind = "join"

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        if len(inputs) != 2:
            raise CompileError("a join takes exactly two inputs")
        left, right = inputs
        if not self.on:
            raise CompileError("a join needs at least one key column")
        if self.how not in _MERGE_HOW:
            raise CompileError(
                f"{self.how.value} joins are not supported: omniframes joins on column names, "
                "so every join kind it runs has keys. Use inner, left, right or outer."
            )
        self._check(left, "left")
        self._check(right, "right")
        overlap = sorted((set(left.column_names) & set(right.column_names)) - set(self.on))
        if overlap:
            raise CompileError(join_collision_message(overlap))

        keys = {
            name: promote_type(
                left.schema.field(name).type, right.schema.field(name).type, column=name
            )
            for name in self.on
        }
        left = _cast_columns(left, keys)
        right = _cast_columns(right, keys)
        schema = self._schema(left, right)

        left_null = _null_key_mask(left, self.on)
        right_null = _null_key_mask(right, self.on)
        merged = _merge(
            _rows(left, pc.invert(left_null)),
            _rows(right, pc.invert(right_null)),
            self.on,
            _MERGE_HOW[self.how],
            schema,
        )

        keep_left, keep_right = _KEEPS_NULL_KEYS[self.how]
        parts = [merged]
        if keep_left:
            parts.append(_padded(_rows(left, left_null), schema))
        if keep_right:
            parts.append(_padded(_rows(right, right_null), schema))
        return parts[0] if len(parts) == 1 else pa.concat_tables(parts)

    def _check(self, table: pa.Table, side: str) -> None:
        missing = [name for name in self.on if name not in table.column_names]
        if missing:
            available = ", ".join(table.column_names) or "(none)"
            raise CompileError(
                f"the join key {', '.join(missing)} is not a column of the {side} frame; it "
                f"has: {available}"
            )

    def _schema(self, left: pa.Table, right: pa.Table) -> pa.Schema:
        fields = [left.schema.field(name) for name in self.on]
        fields.extend(left.schema.field(n) for n in left.column_names if n not in self.on)
        fields.extend(right.schema.field(n) for n in right.column_names if n not in self.on)
        return pa.schema(fields)

    def detail(self) -> str:
        return f"[{self.how.value}] on [{', '.join(self.on)}]"

    def describe(self, inputs: Sequence[str]) -> str:
        left, right = (inputs[0], inputs[1]) if len(inputs) == 2 else ("?", "?")
        keys = ", ".join(self.on)
        return f"{self.kind} [{self.how.value}]: {left} \u2a1d {right} on [{keys}]"


def join_collision_message(overlap: Sequence[str]) -> str:
    """Why a join whose two sides share a non-key column name is refused rather than suffixed."""
    names = ", ".join(repr(name) for name in overlap)
    return (
        f"both sides of the join produce a column called {names}, and omniframes will not guess "
        "which one you meant (nor invent a `_left`/`_right` suffix you did not ask for). Alias "
        "or drop the duplicate before joining, e.g. "
        "right.select(F.col('users.state').alias('right_state'), ...)."
    )


def _cast_columns(table: pa.Table, types: Mapping[str, pa.DataType]) -> pa.Table:
    """Cast the named columns, leaving every other column exactly as it arrived."""
    fields = [
        pa.field(name, types.get(name, table.schema.field(name).type))
        for name in table.column_names
    ]
    schema = pa.schema(fields)
    return table if schema.equals(table.schema) else table.cast(schema)


def _null_key_mask(table: pa.Table, on: Sequence[str]) -> Value:
    """True for every row whose join key is NULL in at least one position."""
    is_null: Any = pc.is_null
    combine: Any = pc.or_
    mask = is_null(table.column(on[0]))
    for name in on[1:]:
        mask = combine(mask, is_null(table.column(name)))
    return mask


def _rows(table: pa.Table, mask: Value) -> pa.Table:
    """``table.filter(mask)``, past pyarrow's loose compute stubs."""
    select: Any = table.filter
    result: pa.Table = select(mask)
    return result


def _merge(
    left: pa.Table,
    right: pa.Table,
    on: Sequence[str],
    how: str,
    schema: pa.Schema,
) -> pa.Table:
    """The equi-join itself, over rows whose keys are all non-NULL."""
    import pandas as pd

    frames = [side.to_pandas(types_mapper=pd.ArrowDtype) for side in (left, right)]
    merge: Any = frames[0].merge
    merged = merge(frames[1], how=how, on=list(on))
    table = pa.Table.from_pandas(merged, preserve_index=False).replace_schema_metadata(None)
    return table.select(list(schema.names)).cast(schema)


def _padded(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """``table``'s rows in the join's output shape, NULL wherever it has no column of its own."""
    columns = [
        table.column(field.name).cast(field.type)
        if field.name in table.column_names
        else pa.nulls(table.num_rows, type=field.type)
        for field in schema
    ]
    return pa.table(columns, schema=schema)


@dataclass(frozen=True)
class LocalUnion(LocalOp):
    """Stack two results by position (``UNION ALL``: nothing is de-duplicated).

    Column *names* have to agree as well as their count \u2014 omniframes' columns are named wire
    outputs, so borrowing the left side's names for a differently named right side would
    relabel data rather than stack it.  Types widen per docs/HYBRID.md \u00a73.3.
    """

    kind = "union"

    def run(self, inputs: tuple[pa.Table, ...]) -> pa.Table:
        if len(inputs) != 2:
            raise CompileError("a union takes exactly two inputs")
        left, right = inputs
        if left.column_names != right.column_names:
            raise CompileError(
                "union() stacks two results by position, so both sides must produce the same "
                f"columns in the same order: left has [{', '.join(left.column_names)}], right "
                f"has [{', '.join(right.column_names)}]"
            )
        schema = pa.schema(
            [
                pa.field(
                    name,
                    promote_type(
                        left.schema.field(name).type,
                        right.schema.field(name).type,
                        column=name,
                    ),
                )
                for name in left.column_names
            ]
        )
        return pa.concat_tables([left.cast(schema), right.cast(schema)])

    def detail(self) -> str:
        return "by position"

    def describe(self, inputs: Sequence[str]) -> str:
        left, right = (inputs[0], inputs[1]) if len(inputs) == 2 else ("?", "?")
        return f"{self.kind}: {left} \u228e {right} ({self.detail()})"


def run_local_op(op: LocalOp, inputs: tuple[pa.Table, ...]) -> pa.Table:
    """Apply one local operator. Pure: same inputs, same output, no session involved."""
    return op.run(inputs)


# --------------------------------------------------------------------------------------
# Rendering (used by explain())
# --------------------------------------------------------------------------------------


def describe_expr(expr: Expr) -> str:
    """Render an expression the way a reader of ``explain()`` would write it."""
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, FieldRef | MeasureRef | AdHocAgg | Udf):
        return display_name(expr)
    if isinstance(expr, Comparison):
        return f"{describe_expr(expr.left)} {expr.op.value} {describe_expr(expr.right)}"
    if isinstance(expr, Arithmetic):
        return f"({describe_expr(expr.left)} {expr.op.value} {describe_expr(expr.right)})"
    if isinstance(expr, BooleanOp):
        joined = f" {expr.op.value} ".join(describe_expr(operand) for operand in expr.operands)
        return f"({joined})"
    if isinstance(expr, Not):
        return f"NOT ({describe_expr(expr.operand)})"
    if isinstance(expr, IsNull):
        return f"{describe_expr(expr.operand)} IS NULL"
    if isinstance(expr, IsIn):
        values = ", ".join(repr(value) for value in expr.values)
        return f"{describe_expr(expr.operand)} IN ({values})"
    if isinstance(expr, StringPredicate):
        suffix = " (case-insensitive)" if expr.case_insensitive else ""
        kind = expr.kind.value
        return f"{describe_expr(expr.operand)} {kind} {expr.value!r}{suffix}"
    if isinstance(expr, Between):
        operator = "<" if isinstance(expr.high, date | datetime) else "<="
        return f"{expr.low!r} <= {describe_expr(expr.operand)} {operator} {expr.high!r}"
    return display_name(expr)
