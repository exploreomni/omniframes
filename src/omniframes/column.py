"""Expressions and the user-facing :class:`Column` wrapper (docs/INTERNALS.md §1).

Every expression is an immutable frozen dataclass under :class:`Expr`; ``Column`` is the thin
user-facing wrapper that builds them through Python operators.  Nothing here knows about the
wire — the mapping from an expression to a field name or a filter lives in
:mod:`omniframes.compile.semantic`, so this module performs no I/O and imports no transport.

Two ergonomics decisions worth knowing:

* ``__eq__`` builds a :class:`Comparison` instead of comparing, which is what makes
  ``df.filter(F.col("users.state") == "California")`` read the way it does.  ``__hash__`` is
  therefore restored explicitly to ``object.__hash__`` so Columns stay usable as dict keys and
  in identity sets.
* ``__bool__`` raises.  ``a == 1 and b == 2`` would otherwise silently evaluate to ``b == 2``;
  the raised :class:`TypeError` names ``&`` / ``|`` / ``~`` instead.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, TypeAlias

from omniframes.errors import CompileError

__all__ = [
    "AdHocAgg",
    "AggFn",
    "ArithOp",
    "Arithmetic",
    "Between",
    "BoolOp",
    "BooleanOp",
    "CmpOp",
    "Column",
    "Comparison",
    "Expr",
    "FieldRef",
    "IsIn",
    "IsNull",
    "Literal",
    "LiteralValue",
    "MeasureRef",
    "Not",
    "SortKey",
    "StrPredKind",
    "StringPredicate",
    "Udf",
]

#: Everything a :class:`Literal` may carry.  ``None`` is legal but only meaningful through
#: :meth:`Column.is_null` — comparing to ``None`` is rejected at compile time.
LiteralValue: TypeAlias = "str | int | float | bool | Decimal | date | datetime | None"


class AggFn(enum.Enum):
    """Ad-hoc aggregation functions (``count_distinct`` is ``COUNT`` + ``distinct=True``)."""

    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class CmpOp(enum.Enum):
    """Comparison operators built by the Column dunders."""

    EQ = "="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="

    @property
    def flipped(self) -> CmpOp:
        """The operator with its operands swapped (``1 < x`` → ``x > 1``)."""
        return _FLIPPED[self]


_FLIPPED: dict[CmpOp, CmpOp] = {
    CmpOp.EQ: CmpOp.EQ,
    CmpOp.NE: CmpOp.NE,
    CmpOp.LT: CmpOp.GT,
    CmpOp.LE: CmpOp.GE,
    CmpOp.GT: CmpOp.LT,
    CmpOp.GE: CmpOp.LE,
}


class ArithOp(enum.Enum):
    """Arithmetic operators. Never expressible in tier 1 — the query API has no calculations."""

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"


class BoolOp(enum.Enum):
    """The n-ary boolean connectives."""

    AND = "AND"
    OR = "OR"


class StrPredKind(enum.Enum):
    """String predicates. ``LIKE`` maps to the wire's ``SQL_LIKE``."""

    CONTAINS = "CONTAINS"
    STARTS_WITH = "STARTS_WITH"
    ENDS_WITH = "ENDS_WITH"
    LIKE = "LIKE"


# --------------------------------------------------------------------------------------
# The expression tree
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Expr:
    """Base of the immutable expression tree."""

    @property
    def children(self) -> tuple[Expr, ...]:
        """Sub-expressions, for generic walking (see :mod:`omniframes.plan.visitor`)."""
        return ()


@dataclass(frozen=True)
class FieldRef(Expr):
    """A model field (``"users.state"``), optionally at a time grain (``created_at[month]``)."""

    name: str
    grain: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise CompileError("a field reference needs a name, e.g. F.col('users.state')")
        if self.name == "*":
            # PySpark's select-everything.  The wire has no wildcard — `*` would go out as a
            # field name and come back as a server-side "unknown field", so say it here instead,
            # with the same advice the no-projection error gives (compile/semantic.py).
            raise CompileError(
                "omniframes never expands a topic to all of its fields (topics are wide, and the "
                "wire needs an explicit field list), so select('*') has nothing to expand. Use "
                "session.catalog.topic(...) to browse what is available, then name the columns."
            )


@dataclass(frozen=True)
class MeasureRef(Expr):
    """A governed model measure. Always executes remotely — never emulated locally."""

    name: str


@dataclass(frozen=True)
class Literal(Expr):
    """A constant value."""

    value: LiteralValue


@dataclass(frozen=True)
class AdHocAgg(Expr):
    """An aggregation the user wrote (``F.sum("order_items.sale_price")``).

    Not expressible in tier 1: the query API's ``calculations`` need a serialized parse tree and
    are post-limit row operations anyway (CONTRACT_NOTES §3.3), so these ride tier 2/3.
    """

    fn: AggFn
    operand: FieldRef
    distinct: bool = False

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)


@dataclass(frozen=True)
class Comparison(Expr):
    """``left <op> right``."""

    op: CmpOp
    left: Expr
    right: Expr

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Arithmetic(Expr):
    """``left <op> right`` over values. Never tier 1."""

    op: ArithOp
    left: Expr
    right: Expr

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.left, self.right)


@dataclass(frozen=True)
class BooleanOp(Expr):
    """``AND``/``OR`` over n operands, flattened at construction."""

    op: BoolOp
    operands: tuple[Expr, ...]

    def __post_init__(self) -> None:
        flat: list[Expr] = []
        for operand in self.operands:
            if isinstance(operand, BooleanOp) and operand.op is self.op:
                flat.extend(operand.operands)
            else:
                flat.append(operand)
        object.__setattr__(self, "operands", tuple(flat))

    @property
    def children(self) -> tuple[Expr, ...]:
        return self.operands


@dataclass(frozen=True)
class Not(Expr):
    """Logical negation."""

    operand: Expr

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)


@dataclass(frozen=True)
class IsNull(Expr):
    """``operand IS NULL``."""

    operand: Expr

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)


@dataclass(frozen=True)
class IsIn(Expr):
    """``operand IN (values)`` — compiles to a multi-value ``EQUALS`` filter."""

    operand: Expr
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)


@dataclass(frozen=True)
class StringPredicate(Expr):
    """CONTAINS / STARTS_WITH / ENDS_WITH / LIKE over a string field."""

    kind: StrPredKind
    operand: Expr
    value: str
    case_insensitive: bool = False

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)


@dataclass(frozen=True)
class Between(Expr):
    """``operand BETWEEN low AND high``.

    Two different bounds, because the wire offers two different vocabularies
    (CONTRACT_NOTES §3.1):

    * **numbers** — inclusive at both ends (``low <= x <= high``, PySpark-consistent).  The
      wire's ``BETWEEN`` kind has an *exclusive* upper bound, so this compiles instead to a
      composite ``AND`` of ``GREATER_THAN``/``LESS_THAN`` with ``is_inclusive: true``.
    * **dates/datetimes** — half-open (``low <= x < high``).  The date arm has only
      ``ON_OR_AFTER`` (``>=``) and ``BEFORE``/``BETWEEN`` (``<``), so an inclusive upper bound
      would need date arithmetic on a literal grammar that also accepts relative expressions
      ("last quarter"); omniframes refuses to invent it.

    See :meth:`Column.between`, which documents the same split for users.
    """

    operand: Expr
    low: Any
    high: Any

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.operand,)


@dataclass(frozen=True)
class Udf(Expr):
    """A Python function applied to values, evaluated by the local engine (docs/HYBRID.md §5).

    A ``Udf`` anywhere in a projection, a filter or a ``with_column`` pins the pushdown frontier
    below it: it is simply an expression the query API cannot express, so the generic splitter
    rules apply and everything from that point up runs locally.  ``name`` is what ``explain()``
    and the default column name show (``"my_fn(users.state)"``).
    """

    fn: Callable[..., Any]
    operands: tuple[Expr, ...]
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "operands", tuple(self.operands))

    @property
    def children(self) -> tuple[Expr, ...]:
        return self.operands


@dataclass(frozen=True)
class SortKey(Expr):
    """One entry of a ``sort()`` — an expression plus its direction."""

    expr: Expr
    descending: bool = False

    @property
    def children(self) -> tuple[Expr, ...]:
        return (self.expr,)


# --------------------------------------------------------------------------------------
# Column
# --------------------------------------------------------------------------------------


def _to_expr(value: object) -> Expr:
    """Coerce an operand into an expression: Columns unwrap, everything else is a literal."""
    if isinstance(value, Column):
        return value.expr
    if isinstance(value, Expr):
        return value
    if value is None or isinstance(value, str | bool | int | float | Decimal | date | datetime):
        return Literal(value)
    raise CompileError(
        f"cannot use {type(value).__name__} in an omniframes expression; expected a Column, a "
        "column name, or a scalar (str, int, float, Decimal, bool, date, datetime, None)"
    )


class Column:
    """A lazy column expression with an optional client-side alias.

    Every method returns a **new** Column; nothing mutates.  Aliases never reach the wire (the
    query API has no aliasing) — they are applied as a rename after the result comes back, and
    sorts/filters written against an alias are reverse-resolved at compile time.
    """

    __slots__ = ("_alias", "_descending", "_expr")

    def __init__(
        self,
        expr: Expr,
        alias: str | None = None,
        *,
        descending: bool | None = None,
    ) -> None:
        self._expr = expr
        self._alias = alias
        self._descending = descending

    # -- identity ---------------------------------------------------------------------

    @property
    def expr(self) -> Expr:
        """The underlying expression."""
        return self._expr

    @property
    def alias_name(self) -> str | None:
        """The client-side alias, if :meth:`alias` was called."""
        return self._alias

    @property
    def descending(self) -> bool | None:
        """Sort direction recorded by :meth:`desc`/:meth:`asc`; ``None`` when unspecified."""
        return self._descending

    def __repr__(self) -> str:
        """``Column(users.state = 'California')`` — the predicate, not its dataclass tree.

        Building predicates in a REPL is the workflow the quickstart describes, and a nested
        ``Comparison(op=<CmpOp.EQ…>, left=FieldRef(name=…))`` dump is unreadable the moment two
        clauses are combined.  ``describe_expr`` is the renderer ``explain()`` already uses, so
        one expression reads the same wherever it is printed.  The import is function-local
        because :mod:`omniframes.compile.local` imports this module (the same pattern
        :mod:`omniframes.compile.sqlgen` uses).
        """
        from omniframes.compile.local import describe_expr

        suffix = f" AS {self._alias}" if self._alias else ""
        direction = "" if self._descending is None else (" DESC" if self._descending else " ASC")
        return f"Column({describe_expr(self._expr)}{suffix}{direction})"

    # Restored explicitly: defining __eq__ (below) would otherwise set __hash__ to None and make
    # Columns unusable as dict keys / set members.  Identity hashing is the only coherent choice
    # when __eq__ does not answer a question about equality.
    __hash__ = object.__hash__

    def __bool__(self) -> bool:
        raise TypeError(
            "a Column has no truth value: Python's `and`, `or` and `not` cannot be overloaded. "
            "Use `&` (and), `|` (or) and `~` (not), and parenthesize every operand: "
            "(F.col('users.age') > 21) & (F.col('users.state') == 'California')"
        )

    # -- naming -----------------------------------------------------------------------

    def alias(self, name: str) -> Column:
        """Rename this column client-side. Collisions are a build-time :class:`CompileError`."""
        if not name:
            raise CompileError("alias() needs a non-empty name")
        return Column(self._expr, name, descending=self._descending)

    def grain(self, grain: str) -> Column:
        """Select a time grain (``created_at[month]``). Only valid on a bare field reference."""
        if not isinstance(self._expr, FieldRef):
            raise CompileError(
                f"grain() applies to a bare field reference, not to {self._expr!r}; "
                "write F.col('order_items.created_at').grain('month')"
            )
        if self._expr.grain is not None:
            raise CompileError(
                f"{self._expr.name!r} already has the grain {self._expr.grain!r}; "
                "a field carries at most one grain"
            )
        # Imported here rather than at module scope: `compile` depends on this module, and the
        # grain list is the one thing an expression needs from the wire contract.
        from omniframes.compile.querymodel import is_valid_grain

        if not is_valid_grain(grain):
            raise CompileError(
                f"{grain!r} is not an Omni time grain. An invalid grain does not error "
                "server-side — the field silently lands in summary.missing_fields."
            )
        return Column(
            FieldRef(self._expr.name, grain.lower()), self._alias, descending=self._descending
        )

    # -- sorting ----------------------------------------------------------------------

    def desc(self) -> Column:
        """Sort this column descending (used inside ``sort()``)."""
        return Column(self._expr, self._alias, descending=True)

    def asc(self) -> Column:
        """Sort this column ascending (used inside ``sort()``)."""
        return Column(self._expr, self._alias, descending=False)

    def to_sort_key(self) -> SortKey:
        """This column as a :class:`SortKey`, defaulting to ascending."""
        return SortKey(self._expr, bool(self._descending))

    # -- comparisons ------------------------------------------------------------------

    def __eq__(self, other: object) -> Column:  # type: ignore[override]
        return Column(Comparison(CmpOp.EQ, self._expr, _to_expr(other)))

    def __ne__(self, other: object) -> Column:  # type: ignore[override]
        return Column(Comparison(CmpOp.NE, self._expr, _to_expr(other)))

    def __lt__(self, other: object) -> Column:
        return Column(Comparison(CmpOp.LT, self._expr, _to_expr(other)))

    def __le__(self, other: object) -> Column:
        return Column(Comparison(CmpOp.LE, self._expr, _to_expr(other)))

    def __gt__(self, other: object) -> Column:
        return Column(Comparison(CmpOp.GT, self._expr, _to_expr(other)))

    def __ge__(self, other: object) -> Column:
        return Column(Comparison(CmpOp.GE, self._expr, _to_expr(other)))

    # -- boolean algebra --------------------------------------------------------------

    def __and__(self, other: object) -> Column:
        return Column(BooleanOp(BoolOp.AND, (self._expr, _to_expr(other))))

    def __rand__(self, other: object) -> Column:
        return Column(BooleanOp(BoolOp.AND, (_to_expr(other), self._expr)))

    def __or__(self, other: object) -> Column:
        return Column(BooleanOp(BoolOp.OR, (self._expr, _to_expr(other))))

    def __ror__(self, other: object) -> Column:
        return Column(BooleanOp(BoolOp.OR, (_to_expr(other), self._expr)))

    def __invert__(self) -> Column:
        return Column(Not(self._expr))

    # -- arithmetic (tier 2/3 only) ---------------------------------------------------

    def __add__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.ADD, self._expr, _to_expr(other)))

    def __radd__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.ADD, _to_expr(other), self._expr))

    def __sub__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.SUB, self._expr, _to_expr(other)))

    def __rsub__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.SUB, _to_expr(other), self._expr))

    def __mul__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.MUL, self._expr, _to_expr(other)))

    def __rmul__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.MUL, _to_expr(other), self._expr))

    def __truediv__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.DIV, self._expr, _to_expr(other)))

    def __rtruediv__(self, other: object) -> Column:
        return Column(Arithmetic(ArithOp.DIV, _to_expr(other), self._expr))

    # -- predicates -------------------------------------------------------------------

    def is_null(self) -> Column:
        """``IS NULL``."""
        return Column(IsNull(self._expr))

    def is_not_null(self) -> Column:
        """``IS NOT NULL``."""
        return Column(Not(IsNull(self._expr)))

    def isin(self, *values: Any) -> Column:
        """``IN (...)``. Accepts either varargs or a single iterable."""
        flat = _flatten_values(values)
        if not flat:
            raise CompileError("isin() needs at least one value")
        return Column(IsIn(self._expr, tuple(flat)))

    def contains(self, value: str, *, case_insensitive: bool = False) -> Column:
        return Column(StringPredicate(StrPredKind.CONTAINS, self._expr, value, case_insensitive))

    def starts_with(self, value: str, *, case_insensitive: bool = False) -> Column:
        return Column(StringPredicate(StrPredKind.STARTS_WITH, self._expr, value, case_insensitive))

    def ends_with(self, value: str, *, case_insensitive: bool = False) -> Column:
        return Column(StringPredicate(StrPredKind.ENDS_WITH, self._expr, value, case_insensitive))

    def like(self, pattern: str, *, case_insensitive: bool = False) -> Column:
        """SQL ``LIKE`` (wire ``SQL_LIKE``); ``case_insensitive=True`` is ``ILIKE``."""
        return Column(StringPredicate(StrPredKind.LIKE, self._expr, pattern, case_insensitive))

    def between(self, low: Any, high: Any) -> Column:
        """Range predicate. **Numbers include both ends; dates do not include the upper one.**

        * ``F.col("order_items.quantity").between(3, 5)`` → ``3 <= quantity <= 5``, matching
          PySpark.  It compiles to a composite of ``>=``/``<=`` rather than to the wire's
          ``BETWEEN`` kind, whose upper bound is exclusive (CONTRACT_NOTES §3.1).
        * ``F.col("order_items.created_at").between(date(2025, 7, 1), date(2026, 7, 1))`` →
          ``created_at >= 2025-07-01 AND created_at < 2026-07-01``, i.e. **half-open**.  The
          date filter arm has no inclusive upper bound and omniframes will not synthesize one
          by adding "the smallest unit" to a literal that may be relative ("last quarter") or
          truncated ("2026-01").  Half-open is also what a month/quarter window wants; write
          ``(col >= low) & (col < high)`` to say it explicitly, and remember that
          ``<=`` on a date is not expressible either (use the next boundary instead).

        Which arm applies is decided by the *literal type* (or by ``.grain()``), exactly as for
        comparisons: the compiler has no field types.
        """
        return Column(Between(self._expr, low, high))


def _flatten_values(values: tuple[Any, ...]) -> list[Any]:
    """``isin("a", "b")`` and ``isin(["a", "b"])`` mean the same thing."""
    if len(values) == 1 and isinstance(values[0], Iterable) and not isinstance(values[0], str):
        return list(values[0])
    return list(values)
