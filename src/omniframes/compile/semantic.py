"""Tier 1 — the semantic compiler (docs/INTERNALS.md §3/§4).

Turns a logical plan into a governed :class:`~omniframes.compile.querymodel.Query`: the whole
computation runs inside Omni, under the model's joins, access grants and measure definitions.

The shape tier 1 accepts is
``Scan → [Filter] → (Project | Aggregate) → [Filter] → [Sort] → [Limit]`` in any build order
that is *semantically* the same as that pipeline.  Anything else — a join, a derived column, an
ad-hoc aggregation, a cross-field ``OR`` — raises :class:`CannotCompile` carrying the reason.
In M2 the DataFrame turns that into a ``CompileError("not yet supported: <reason>")``; from M3
the splitter catches it instead and pushes down what it can, finishing the rest locally.

Five rules deserve their own paragraph:

**Selection is the group-by.**  An :class:`~omniframes.plan.nodes.Aggregate` compiles to exactly
the projection of its keys followed by its measures, because in Omni asking for dimensions next
to governed measures *is* the aggregate (docs/DESIGN.md §2).  ``group_by().agg()`` and the
equivalent ``select()`` therefore produce byte-identical envelopes — the golden lane pins it.

**Measure filters are a genuine HAVING** (CONTRACT_NOTES §3.1).  A ``filters`` entry keyed by a
governed measure is translated against that measure's aggregate expression and applied after the
group-by, so post-aggregation filtering stays in tier 1.  The rules the compiler must respect:
one entry per measure (several conditions merge into a ``composite`` inside it), the number arm
only, and a measure that is filtered but never selected is force-added server-side and projected
away.  Ad-hoc aggregations get no such treatment — they have no model definition to pin a HAVING
to, so they route to tier 2/3.

**The grain-filter rule** (docs/DESIGN.md §3, CONTRACT_NOTES §3.1).  A grain that produces a
timestamp (``month``, ``week``, …) is *filtered on the bare field name* with a ``date`` filter,
while a grain that produces a number (``hour_of_day``, ``month_num``, …) is filtered on the
*bracketed* name with a ``number`` filter.  Getting this backwards does not error server-side —
the field silently lands in ``summary.missing_fields``.

**Filters AND across fields, compose within one.**  ``query.filters`` is keyed by field name, so
a top-level ``AND`` splits per field; two predicates on the same field merge into a
``composite`` AND; an ``OR`` whose operands all touch the same field becomes a ``composite`` OR.
A cross-field ``OR`` needs the ``controls`` array and is out of scope for 0.1 (DESIGN §6), so it
is refused rather than silently mistranslated.

**The limit is always explicit** (DESIGN §3).  No ``.limit()`` sends
:data:`DEFAULT_FETCH_LIMIT`; ``.limit(None)`` sends ``null``.

One consequence of compiling without a schema: the *literal* picks the filter arm.  The compiler
has no field types (the catalog is discovery-only, and ``summary.fields`` only exists after a
plan round trip), so ``created_at == "2026-03"`` becomes a **string** filter while
``created_at == date(2026, 3, 1)`` and ``created_at.grain("month") == "2026-03"`` become date
filters.  Comparing a timestamp field to a real ``date``/``datetime`` — or naming the grain — is
what makes the intent unambiguous.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final, TypeAlias

from omniframes.column import (
    AdHocAgg,
    AggFn,
    Arithmetic,
    Between,
    BooleanOp,
    BoolOp,
    CmpOp,
    Column,
    Comparison,
    Expr,
    FieldRef,
    IsIn,
    IsNull,
    Literal,
    MeasureRef,
    Not,
    SortKey,
    StringPredicate,
    StrPredKind,
    Udf,
)
from omniframes.compile.querymodel import (
    DEFAULT_FETCH_LIMIT,
    DEFAULT_SERVER_LIMIT,
    DURATION_GRAINS,
    GRAINS,
    GRAND_TOTAL_KEY,
    UNSET,
    BooleanFilter,
    CachePolicy,
    CompositeFilter,
    DateFilter,
    DateFilterKind,
    Filter,
    FilterConjunction,
    NullFilter,
    NullSort,
    NumberFilter,
    NumberFilterKind,
    Query,
    RunRequest,
    Sort,
    StringFilter,
    StringFilterKind,
    Unset,
    WireDict,
)
from omniframes.errors import CompileError
from omniframes.plan import nodes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omniframes.compile.local import LocalOp

__all__ = [
    "DEFAULT_FETCH_LIMIT",
    "DEFAULT_SERVER_LIMIT",
    "NUMBER_GRAINS",
    "STRING_GRAINS",
    "TIMESTAMP_GRAINS",
    "CannotCompile",
    "EnvelopeOptions",
    "ExecutionPlan",
    "FilterSet",
    "LocalStep",
    "RemoteStep",
    "SemanticCompilation",
    "Step",
    "adhoc_filter_reason",
    "alias_map",
    "build_envelope",
    "column_key",
    "compile_filters",
    "compile_plan",
    "compile_semantic",
    "display_name",
    "envelope_limit",
    "predicate_to_filters",
    "split_grain",
    "try_semantic",
    "wire_name",
]


class CannotCompile(Exception):
    """This plan is not expressible as a tier-1 semantic query.

    Not a user-facing error on its own: today's DataFrame reports it as
    ``CompileError("not yet supported: <reason>")``, and M3's splitter uses it as the signal to
    cut the plan and finish locally.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------------------
# Grain classification (CONTRACT_NOTES §3.1/§3.2)
# --------------------------------------------------------------------------------------

#: Grains whose value is a timestamp — filtered on the BARE field name with a ``date`` filter.
TIMESTAMP_GRAINS: Final[frozenset[str]] = frozenset(
    {
        "year",
        "quarter",
        "month",
        "week",
        "date",
        "hour",
        "minute",
        "second",
        "millisecond",
        "fiscal_year",
        "fiscal_quarter",
    }
)

#: Grains whose value is a number — filtered on the BRACKETED name with a ``number`` filter.
NUMBER_GRAINS: Final[frozenset[str]] = (
    frozenset(
        {
            "quarter_of_year",
            "week_of_year",
            "day_of_week_num",
            "month_num",
            "hour_of_day",
            "day_of_month",
            "day_of_year",
            "day_of_quarter",
            "epoch",
        }
    )
    | DURATION_GRAINS
)

#: Grains whose value is text — filtered on the BRACKETED name with a ``string`` filter.
STRING_GRAINS: Final[frozenset[str]] = GRAINS - TIMESTAMP_GRAINS - NUMBER_GRAINS

_GRAIN_SUFFIX: Final = re.compile(r"^(?P<base>.+)\[(?P<grain>[^\[\]]+)\]$")

_BOOLEAN: Final = "boolean"
_DATE: Final = "date"
_NUMBER: Final = "number"
_STRING: Final = "string"

#: ``alias -> wire name`` lookup used to reverse-resolve filters and sorts written against an
#: alias.  Returns ``None`` for a name that is not an alias.
Resolver: TypeAlias = Callable[[str], "str | None"]


# --------------------------------------------------------------------------------------
# Names (docs/INTERNALS.md §3)
# --------------------------------------------------------------------------------------


def wire_name(expr: Expr) -> str:
    """The exact field name this expression selects on the wire.

    Raises:
        CannotCompile: the expression has no wire name (ad-hoc aggregation, arithmetic, …).
    """
    if isinstance(expr, FieldRef):
        return expr.name if expr.grain is None else f"{expr.name}[{expr.grain}]"
    if isinstance(expr, MeasureRef):
        return expr.name
    raise CannotCompile(
        f"{display_name(expr)} has no field name on the wire "
        "(the query API cannot express it as a selected field)"
    )


def display_name(expr: Expr) -> str:
    """The column name a user sees for this expression when it is not aliased."""
    if isinstance(expr, FieldRef):
        return expr.name if expr.grain is None else f"{expr.name}[{expr.grain}]"
    if isinstance(expr, MeasureRef):
        return expr.name
    if isinstance(expr, AdHocAgg):
        name = "count_distinct" if expr.distinct and expr.fn is AggFn.COUNT else expr.fn.value
        return f"{name}({display_name(expr.operand)})"
    if isinstance(expr, Arithmetic):
        return f"({display_name(expr.left)} {expr.op.value} {display_name(expr.right)})"
    if isinstance(expr, Literal):
        return repr(expr.value)
    if isinstance(expr, Udf):
        return expr.name
    return f"{type(expr).__name__.lower()}({', '.join(display_name(c) for c in expr.children)})"


def column_key(expr: Expr) -> str:
    """The un-aliased output name of a projected expression (wire name when there is one)."""
    return display_name(expr)


def split_grain(name: str) -> tuple[str, str | None]:
    """``"created_at[month]"`` → ``("created_at", "month")``; a bare name keeps ``None``."""
    match = _GRAIN_SUFFIX.match(name)
    if match is None:
        return name, None
    grain = match.group("grain").lower()
    if grain not in GRAINS:
        return name, None
    return match.group("base"), grain


# --------------------------------------------------------------------------------------
# Aliases (docs/INTERNALS.md §2)
# --------------------------------------------------------------------------------------


def alias_map(columns: Sequence[Column]) -> dict[str, str]:
    """``{alias: wire name}`` for a projection, validating collisions.

    Aliases are client-side renames applied after the result comes back, so an ambiguous map
    would silently drop or duplicate a column.  All three ambiguities are build-time errors:
    a repeated alias, two aliases for the same field, and an alias shadowing another selected
    field.
    """
    mapping: dict[str, str] = {}
    alias_by_key: dict[str, str] = {}
    keys = [column_key(column.expr) for column in columns]

    for column, key in zip(columns, keys, strict=True):
        alias = column.alias_name
        if alias is None:
            continue
        if alias in mapping:
            raise CompileError(
                f"alias {alias!r} is used twice in one select(); aliases must be unique"
            )
        existing = alias_by_key.get(key)
        if existing is not None and existing != alias:
            raise CompileError(
                f"{key!r} is aliased twice, to {existing!r} and {alias!r}; a field can carry at "
                "most one alias in a projection"
            )
        mapping[alias] = key
        alias_by_key[key] = alias

    for alias, key in mapping.items():
        if alias != key and alias in keys:
            raise CompileError(
                f"alias {alias!r} shadows the selected field {alias!r}; pick a different name"
            )
    return mapping


def _resolver(mapping: Mapping[str, str]) -> Resolver:
    def resolve(name: str) -> str | None:
        return mapping.get(name)

    return resolve


# --------------------------------------------------------------------------------------
# Filters (CONTRACT_NOTES §3.1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    """The field one predicate is about, plus how the wire wants it filtered."""

    #: The key of the ``query.filters`` entry (bracketed or bare per the grain-filter rule).
    key: str
    #: ``"date"``/``"number"``/``"string"`` when the grain pins the filter type, else ``None``.
    kind: str | None
    #: True when the key names a governed measure, i.e. the entry lands in the HAVING.
    is_measure: bool = False


@dataclass(frozen=True)
class _Entry:
    """One compiled predicate before same-field entries are merged."""

    key: str
    filter: Filter
    is_measure: bool = False


@dataclass(frozen=True)
class FilterSet:
    """Compiled ``query.filters``, remembering which entries are measure-keyed.

    Both halves travel in the same ``filters`` map on the wire — the server decides where each
    entry lands from the *model field type* of its key (CONTRACT_NOTES §3.1) — but ``explain()``
    has to tell a WHERE from a HAVING, and the totals rule needs to know a query aggregates.
    """

    entries: Mapping[str, Filter]
    measure_keys: frozenset[str] = frozenset()

    @property
    def where(self) -> dict[str, Filter]:
        """The pre-aggregation entries (dimension-keyed)."""
        return {k: v for k, v in self.entries.items() if k not in self.measure_keys}

    @property
    def having(self) -> dict[str, Filter]:
        """The post-aggregation entries (measure-keyed)."""
        return {k: v for k, v in self.entries.items() if k in self.measure_keys}


def compile_filters(predicate: Expr, *, resolve: Resolver | None = None) -> FilterSet:
    """Normalize a boolean expression into ``query.filters``, WHERE and HAVING alike.

    ``resolve`` maps an alias back to its wire name (see :func:`alias_map`).

    Raises:
        CannotCompile: the predicate is not expressible as typed filters.
        CompileError: the predicate is malformed (comparison against ``None``, …).
    """
    entries: dict[str, list[Filter]] = {}
    measure_keys: set[str] = set()
    for conjunct in _flatten(predicate, BoolOp.AND):
        compiled = _compile_predicate(conjunct, resolve)
        entries.setdefault(compiled.key, []).append(compiled.filter)
        if compiled.is_measure:
            measure_keys.add(compiled.key)
    return FilterSet(
        entries={
            # Exactly one entry per field — several conditions on one measure (or one column)
            # merge into a composite AND inside that single entry.
            key: (
                filters[0] if len(filters) == 1 else CompositeFilter(FilterConjunction.AND, filters)
            )
            for key, filters in entries.items()
        },
        measure_keys=frozenset(measure_keys),
    )


def predicate_to_filters(predicate: Expr, *, resolve: Resolver | None = None) -> dict[str, Filter]:
    """:func:`compile_filters`, keeping only the ``{field: filter}`` map."""
    return dict(compile_filters(predicate, resolve=resolve).entries)


def _flatten(expr: Expr, op: BoolOp) -> tuple[Expr, ...]:
    if isinstance(expr, BooleanOp) and expr.op is op:
        return expr.operands
    return (expr,)


def _compile_predicate(expr: Expr, resolve: Resolver | None) -> _Entry:
    """One predicate → the field it filters and the filter entry, or :class:`CannotCompile`."""
    if isinstance(expr, BooleanOp):
        conjunction = FilterConjunction.AND if expr.op is BoolOp.AND else FilterConjunction.OR
        compiled = [_compile_predicate(operand, resolve) for operand in expr.operands]
        keys = {entry.key for entry in compiled}
        if len(keys) != 1:
            raise _cross_field(expr.op, compiled)
        return _Entry(
            key=keys.pop(),
            filter=CompositeFilter(conjunction, [entry.filter for entry in compiled]),
            is_measure=any(entry.is_measure for entry in compiled),
        )

    if isinstance(expr, Not):
        inner = _compile_predicate(expr.operand, resolve)
        return replace(inner, filter=_negated(inner.filter))

    if isinstance(expr, IsNull):
        target = _target(expr.operand, resolve)
        if target.is_measure:
            raise CannotCompile(
                f"{target.key} is a governed measure; a HAVING entry carries the number arm "
                "(compare it to a value instead of asking for NULL)"
            )
        return _Entry(target.key, NullFilter())

    if isinstance(expr, Comparison):
        return _comparison(expr, resolve)

    if isinstance(expr, IsIn):
        target = _target(expr.operand, resolve)
        return _Entry(target.key, _isin_filter(target, expr.values), target.is_measure)

    if isinstance(expr, StringPredicate):
        target = _target(expr.operand, resolve)
        return _Entry(target.key, _string_predicate_filter(target, expr), target.is_measure)

    if isinstance(expr, Between):
        target = _target(expr.operand, resolve)
        return _Entry(target.key, _between_filter(target, expr), target.is_measure)

    if isinstance(expr, FieldRef):
        # A bare boolean field used as a predicate: `df.filter(F.col("order_items.returned"))`.
        return _Entry(_target(expr, resolve).key, BooleanFilter(is_negative=False))

    if isinstance(expr, MeasureRef):
        raise CannotCompile(
            f"{display_name(expr)} is a governed measure, not a boolean: compare it to a value "
            "(the HAVING entry it compiles to carries the number arm)"
        )

    if isinstance(expr, AdHocAgg):
        raise CannotCompile(adhoc_filter_reason(expr))

    if isinstance(expr, Arithmetic):
        raise CannotCompile(
            f"filtering on the computed expression {display_name(expr)}: the query API has no "
            "usable calculations (CONTRACT_NOTES §3.3), so this routes to tier 2/3"
        )

    raise CannotCompile(f"{display_name(expr)} is not a filter omniframes can express")


def _cross_field(op: BoolOp, compiled: Sequence[_Entry]) -> CannotCompile:
    """The reason an n-ary connective spanning two fields cannot be one ``filters`` entry."""
    keys = ", ".join(sorted({entry.key for entry in compiled}))
    if any(entry.is_measure for entry in compiled):
        return CannotCompile(
            f"{op.value} across different measures/fields ({keys}): query.filters holds exactly "
            "one entry per field, so a HAVING that spans two aggregates has to be written in SQL "
            "— that routes to tier 2/3"
        )
    return CannotCompile(
        f"{op.value} across different fields ({keys}): query.filters is keyed by field, and "
        "cross-field OR needs the `controls` array, which is out of scope for 0.1"
    )


def _negated(flt: Filter) -> Filter:
    """Flip a filter's ``is_negative``; every arm supports it, composites included."""
    return replace(flt, is_negative=not bool(flt.is_negative))


def adhoc_filter_reason(expr: Expr) -> str:
    """Why a predicate over an ad-hoc aggregation is not a tier-1 (or M3) filter.

    Public because the splitter reports the same reason: an ad-hoc aggregation inside a
    predicate has no aggregate to hang the comparison on until tier 2 emits the SQL (M5).
    """
    return (
        f"filtering on {display_name(expr)}: an ad-hoc aggregation has no model definition for "
        "the server to translate a HAVING against (only governed measures do), so this routes "
        "to tier 2/3"
    )


def _target(expr: Expr, resolve: Resolver | None) -> _Target:
    """Resolve the field a predicate is about and apply the grain-filter rule."""
    if isinstance(expr, MeasureRef):
        # A measure-keyed entry IS a HAVING over the aggregate (CONTRACT_NOTES §3.1), and every
        # governed measure is a number, so the arm is pinned without consulting a schema.
        return _Target(expr.name, _NUMBER, is_measure=True)
    if isinstance(expr, AdHocAgg):
        raise CannotCompile(adhoc_filter_reason(expr))
    if isinstance(expr, Arithmetic):
        raise CannotCompile(
            f"filtering on the computed expression {display_name(expr)}: the query API has no "
            "usable calculations (CONTRACT_NOTES §3.3), so this routes to tier 2/3"
        )
    if not isinstance(expr, FieldRef):
        raise CannotCompile(f"{display_name(expr)} cannot key a filter entry")

    name, grain = expr.name, expr.grain
    resolved = resolve(name) if resolve is not None else None
    if resolved is not None:
        if grain is not None:
            raise CompileError(
                f"{name!r} is an alias; apply .grain() to the field itself in select(), not to "
                "the alias"
            )
        name, grain = split_grain(resolved)
    elif grain is None:
        # The bracketed spelling is the name a user *sees* in the result, and `select()` and
        # `sorts[*].column_name` both take it verbatim — so a filter written against it has to
        # mean the same thing as `.grain(...)`.  The grain-filter rule keys off the grain, not
        # off which spelling produced it, and `split_grain` leaves a non-grain suffix
        # (``users.array[0]``) alone.
        name, grain = split_grain(name)

    if grain is None:
        return _Target(name, None)
    if grain in TIMESTAMP_GRAINS:
        # Timestamp grains filter on the BARE field with a date filter.
        return _Target(name, _DATE)
    if grain in NUMBER_GRAINS:
        return _Target(f"{name}[{grain}]", _NUMBER)
    return _Target(f"{name}[{grain}]", _STRING)


def _value_kind(value: object) -> str:
    if isinstance(value, bool):
        return _BOOLEAN
    if isinstance(value, datetime | date):
        return _DATE
    if isinstance(value, int | float | Decimal):
        return _NUMBER
    if isinstance(value, str):
        return _STRING
    raise CannotCompile(
        f"{type(value).__name__} is not a filter value omniframes knows how to send"
    )


def _kind(target: _Target, value: object) -> str:
    return target.kind or _value_kind(value)


def _literal(expr: Expr, where: str) -> object:
    if not isinstance(expr, Literal):
        raise CannotCompile(
            f"{where} compares two expressions ({display_name(expr)}); tier 1 filters compare a "
            "field to a constant"
        )
    return expr.value


def _date_side(value: object) -> str:
    """Render one side of a date filter using the literal grammar of CONTRACT_NOTES §3.1."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    raise CannotCompile(
        f"{value!r} is not a date literal; use a date/datetime or one of the documented literal "
        'forms ("2026-01", "last quarter", "30 days ago")'
    )


def _comparison(expr: Comparison, resolve: Resolver | None) -> _Entry:
    left, right, op = expr.left, expr.right, expr.op
    if isinstance(right, FieldRef | MeasureRef | AdHocAgg) and isinstance(left, Literal):
        left, right, op = right, left, op.flipped
    target = _target(left, resolve)
    value = _literal(right, f"the comparison on {target.key}")

    if value is None:
        raise CompileError(
            f"comparing {target.key} to None is not a filter; use .is_null() / .is_not_null()"
        )

    kind = _kind(target, value)
    if kind == _BOOLEAN:
        if op is CmpOp.EQ:
            return _Entry(target.key, BooleanFilter(is_negative=not value))
        if op is CmpOp.NE:
            return _Entry(target.key, BooleanFilter(is_negative=bool(value)))
        raise CannotCompile(f"{op.value} against a boolean value has no wire filter")

    if kind == _STRING:
        if op is CmpOp.EQ:
            return _Entry(target.key, StringFilter(StringFilterKind.EQUALS, [str(value)]))
        if op is CmpOp.NE:
            return _Entry(
                target.key, StringFilter(StringFilterKind.EQUALS, [str(value)], is_negative=True)
            )
        raise CannotCompile(
            f"{op.value} on the string field {target.key}: the wire's string filter has no "
            "ordering kinds (use == / != / contains / starts_with / ends_with / like)"
        )

    if kind == _NUMBER:
        return _Entry(target.key, _number_comparison(op, value), target.is_measure)

    return _Entry(target.key, _date_comparison(target, op, value))


def _number_comparison(op: CmpOp, value: object) -> Filter:
    number: Any = value
    if op is CmpOp.EQ:
        return NumberFilter(NumberFilterKind.EQUALS, [number])
    if op is CmpOp.NE:
        return NumberFilter(NumberFilterKind.EQUALS, [number], is_negative=True)
    if op in (CmpOp.LT, CmpOp.LE):
        return NumberFilter(NumberFilterKind.LESS_THAN, [number], is_inclusive=op is CmpOp.LE)
    return NumberFilter(NumberFilterKind.GREATER_THAN, [number], is_inclusive=op is CmpOp.GE)


def _date_comparison(target: _Target, op: CmpOp, value: object) -> Filter:
    side = _date_side(value)
    if op is CmpOp.EQ:
        return DateFilter(DateFilterKind.TIME_FOR_UNIT_DURATION, left_side=side)
    if op is CmpOp.NE:
        return DateFilter(DateFilterKind.TIME_FOR_UNIT_DURATION, left_side=side, is_negative=True)
    if op is CmpOp.GE:
        return DateFilter(DateFilterKind.ON_OR_AFTER, left_side=side)
    if op is CmpOp.LT:
        return DateFilter(DateFilterKind.BEFORE, right_side=side)
    raise CannotCompile(
        f"{op.value} on the date field {target.key}: the wire has ON_OR_AFTER (>=) and BEFORE "
        "(<) only — use >= or <, or .between(low, high)"
    )


def _isin_filter(target: _Target, values: tuple[Any, ...]) -> Filter:
    kinds = {_kind(target, value) for value in values}
    if len(kinds) != 1:
        raise CannotCompile(
            f"isin() on {target.key} mixes value types ({', '.join(sorted(kinds))}); one filter "
            "entry carries one type"
        )
    kind = kinds.pop()
    if kind == _STRING:
        return StringFilter(StringFilterKind.EQUALS, [str(value) for value in values])
    if kind == _NUMBER:
        return NumberFilter(NumberFilterKind.EQUALS, list(values))
    raise CannotCompile(
        f"isin() on {target.key} over {kind} values has no multi-value wire filter; combine the "
        "cases with |"
    )


_STRING_PREDICATE_KINDS: Final[Mapping[StrPredKind, StringFilterKind]] = {
    StrPredKind.CONTAINS: StringFilterKind.CONTAINS,
    StrPredKind.STARTS_WITH: StringFilterKind.STARTS_WITH,
    StrPredKind.ENDS_WITH: StringFilterKind.ENDS_WITH,
    StrPredKind.LIKE: StringFilterKind.SQL_LIKE,
}


def _string_predicate_filter(target: _Target, expr: StringPredicate) -> Filter:
    if target.is_measure:
        raise CannotCompile(
            f"{expr.kind.value.lower()}() on the governed measure {target.key}: a HAVING entry "
            "carries the number arm, not the string one"
        )
    if target.kind is not None and target.kind != _STRING:
        raise CannotCompile(
            f"{expr.kind.value.lower()}() on {target.key}: that grain produces a {target.kind}, "
            "not text"
        )
    return StringFilter(
        _STRING_PREDICATE_KINDS[expr.kind],
        [expr.value],
        case_insensitive=True if expr.case_insensitive else None,
    )


def _between_filter(target: _Target, expr: Between) -> Filter:
    kinds = {_kind(target, expr.low), _kind(target, expr.high)}
    if len(kinds) != 1:
        raise CannotCompile(
            f"between() on {target.key} mixes value types ({', '.join(sorted(kinds))})"
        )
    kind = kinds.pop()
    if kind == _NUMBER:
        # Inclusive at BOTH ends (PySpark-consistent, see Column.between).  The wire's BETWEEN
        # kind has an exclusive upper bound, so the inclusive form is a composite of two
        # inclusive comparisons inside this one field's entry.
        return CompositeFilter(
            FilterConjunction.AND,
            [
                NumberFilter(NumberFilterKind.GREATER_THAN, [expr.low], is_inclusive=True),
                NumberFilter(NumberFilterKind.LESS_THAN, [expr.high], is_inclusive=True),
            ],
        )
    if kind == _DATE:
        # Half-open [low, high): the date arm has no inclusive upper bound and omniframes does
        # not invent date arithmetic over a literal grammar that includes relative expressions.
        return DateFilter(
            DateFilterKind.BETWEEN,
            left_side=_date_side(expr.low),
            right_side=_date_side(expr.high),
        )
    raise CannotCompile(
        f"between() on {target.key} over {kind} values has no wire filter; the BETWEEN arms are "
        "number and date"
    )


# --------------------------------------------------------------------------------------
# Plan matching
# --------------------------------------------------------------------------------------


@dataclass
class _Shape:
    """The tier-1 pipeline recovered from a plan."""

    scan: nodes.Scan
    projects: list[nodes.Project] = field(default_factory=list)
    predicates: list[Expr] = field(default_factory=list)
    sort: nodes.Sort | None = None
    limit: nodes.Limit | None = None


def _match(plan: nodes.PlanNode) -> _Shape:
    """Walk the plan root-down, checking that the operator order is tier-1 expressible."""
    projects: list[nodes.Project] = []
    predicates: list[Expr] = []
    sort: nodes.Sort | None = None
    limit: nodes.Limit | None = None

    node = plan
    while not isinstance(node, nodes.Scan):
        if isinstance(node, nodes.Limit):
            if limit is not None:
                raise CannotCompile("limit()/offset() applied more than once")
            if projects or predicates or sort is not None:
                raise CannotCompile(
                    "select(), filter() or sort() after limit()/offset(): the wire applies "
                    "filters and sorts before the limit, so this is a different query"
                )
            limit = node
        elif isinstance(node, nodes.Sort):
            if sort is not None:
                raise CannotCompile("sort() applied more than once with other operations between")
            sort = node
        elif isinstance(node, nodes.Project):
            projects.append(node)
        elif isinstance(node, nodes.Filter):
            predicates.append(node.predicate)
        elif isinstance(node, nodes.Aggregate):
            projects.append(_aggregate_as_projection(node))
        elif isinstance(node, nodes.Join | nodes.Union):
            raise CannotCompile(
                f"{type(node).__name__.lower()}() is a local operation: the query API takes one "
                "query, so each side compiles on its own and the two results are combined here"
            )
        elif isinstance(node, nodes.WithColumn):
            raise CannotCompile("with_column() computes client-side or in a SQL job (M3)")
        elif isinstance(node, nodes.MapPandas):
            raise CannotCompile("map_pandas() always runs locally (M3)")
        else:
            raise CannotCompile(f"{type(node).__name__} is not a tier-1 operation")
        node = node.children[0]

    return _Shape(scan=node, projects=projects, predicates=predicates, sort=sort, limit=limit)


def _aggregate_as_projection(node: nodes.Aggregate) -> nodes.Project:
    """``group_by(keys).agg(measures)`` → the projection ``select(*keys, *measures)``.

    That is not a simplification: in Omni, selecting dimensions next to governed measures *is*
    the group-by (docs/DESIGN.md §2), so both spellings must produce the same envelope.  An
    ad-hoc aggregation has no such server-side identity and stops here.
    """
    for column in (*node.keys, *node.aggs):
        if isinstance(column.expr, AdHocAgg):
            raise CannotCompile(
                f"{display_name(column.expr)}: ad-hoc aggregations require the hybrid engine (M3)"
            )
    return nodes.Project(node.child, (*node.keys, *node.aggs))


def _compose_projects(projects: Sequence[nodes.Project]) -> tuple[Column, ...]:
    """Collapse chained ``select()`` calls into the single projection the wire carries.

    The outermost projection wins; each inner one only has to be able to supply it.  A name the
    inner select aliased resolves back to the underlying field, keeping the alias as the output
    name — ``df.select(F.col("users.state").alias("s")).select("s")`` selects ``users.state`` and
    hands back a column called ``s``.

    The narrowing is lossless in **both** directions: a bare string in the outer select is just a
    reference, so the *inner* expression is what survives.  Rebuilding it as a plain ``FieldRef``
    would file a governed measure as a group key, and the query would stop reading as an
    aggregate — with ``with_totals()`` then refusing a plan it accepts when written in one
    ``select()`` (docs/DESIGN.md §2: the two spellings must agree).
    """
    columns = tuple(projects[0].columns)
    for inner in projects[1:]:
        alias_map(inner.columns)  # the same collision rules apply to every projection in a chain
        sources: dict[str, Column] = {}
        for produced in inner.columns:
            sources.setdefault(column_key(produced.expr), produced)
        for produced in inner.columns:
            if produced.alias_name is not None:
                sources[produced.alias_name] = produced
        composed: list[Column] = []
        for column in columns:
            key = column_key(column.expr)
            source = sources.get(key)
            if source is None:
                raise CompileError(
                    f"{key!r} is not available after the previous select(); that projection "
                    f"produced {', '.join(sorted(sources))}"
                )
            if isinstance(column.expr, FieldRef):
                composed.append(
                    Column(
                        source.expr,
                        column.alias_name or key,
                        descending=column.descending,
                    )
                )
            else:
                composed.append(column)
        columns = tuple(composed)
    return columns


# --------------------------------------------------------------------------------------
# The compilation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticCompilation:
    """A plan compiled to one governed query."""

    #: The typed query object; ``to_wire()`` is what goes inside the run envelope.
    query: Query
    #: Where the rows come from — kept for ``explain()``.
    scan: nodes.ScanSource
    #: ``wire name -> alias``, ready to hand to :func:`omniframes.transport.normalize.normalize`.
    aliases: Mapping[str, str]
    #: The user-facing column names, in result order.
    columns: tuple[str, ...]
    #: Sort keys as ``(column name, descending)`` — for ``explain()``.
    sorts: tuple[tuple[str, bool], ...] = ()
    #: The selected dimensions, in wire order: the GROUP BY when :attr:`measures` is non-empty.
    group_keys: tuple[str, ...] = ()
    #: The selected governed measures, in wire order.
    measures: tuple[str, ...] = ()
    #: ``filters`` keys that are measure-keyed, i.e. a HAVING rather than a WHERE.
    measure_filters: tuple[str, ...] = ()
    tier: int = 1
    #: A stored query object to put on the wire **verbatim** instead of ``query.to_wire()``.
    #: Set for saved queries and ``session.ask()`` — omniframes did not write those blobs and
    #: re-serializing them would quietly normalize keys it was handed (CONTRACT_NOTES §4).
    #: :attr:`query` then only *describes* the payload (fields, limit) for ``explain()``.
    envelope_query: WireDict | None = None

    @property
    def fields(self) -> tuple[str, ...]:
        """The field names the query selects, in wire order."""
        return tuple(self.query.fields)

    @property
    def is_aggregate(self) -> bool:
        """Whether this query groups: a selected measure, or a HAVING that force-groups it."""
        return bool(self.measures or self.measure_filters)

    @property
    def opaque(self) -> bool:
        """Whether the *server* decides this step's payload: raw SQL, or a stored query.

        An opaque step is a wall for the compiler in both directions — nothing can be pushed
        into it (its columns are whatever the SQL or the stored query produces) and nothing
        above it compiles, so every operation written on top runs in the local engine.
        """
        return isinstance(self.scan, nodes.SqlScan | nodes.SavedQueryScan)

    @property
    def is_sql(self) -> bool:
        """Whether omniframes *wrote* this step's SQL — a tier-2 job (docs/SQLTIER.md §3).

        The opposite of :attr:`opaque`, which marks SQL omniframes was *handed*.  Both put
        ``userEditedSQL`` on the wire; only this one is OmniSQL the compiler wrote — parsed
        against the model, with output columns it chose — which is why ``explain()`` renders
        them differently.
        """
        return bool(self.query.user_edited_sql) and not self.opaque

    @property
    def role(self) -> str:
        """What this step is, for ``explain()`` and the truncation warning."""
        scan = self.scan
        if self.is_sql:
            return "sql"
        if isinstance(scan, nodes.SqlScan):
            return "raw SQL job"
        if isinstance(scan, nodes.SavedQueryScan):
            return "generated query" if scan.origin == "ask" else "saved query"
        return "semantic"


def compile_semantic(plan: nodes.PlanNode, *, totals: bool = False) -> SemanticCompilation:
    """Compile ``plan`` into a tier-1 query.

    Args:
        plan: the logical plan.
        totals: ``with_totals()`` — ask the server for the grand-total row (CONTRACT_NOTES §2.7).

    Raises:
        CannotCompile: the plan is outside tier 1 (the reason names the operation).
        CompileError: the plan is tier-1 shaped but invalid (unknown alias, bad grain, …).
    """
    shape = _match(plan)

    if isinstance(shape.scan.source, nodes.SqlScan | nodes.SavedQueryScan):
        return _compile_opaque(shape, totals=totals)

    if not shape.projects:
        raise CompileError(
            "select() at least one column before running a query: omniframes never expands a "
            "topic to all of its fields (topics are wide, and the wire needs an explicit field "
            "list). Use session.catalog.topic(...) to browse what is available."
        )
    columns = _compose_projects(shape.projects)
    aliases_by_alias = alias_map(columns)
    resolve = _resolver(aliases_by_alias)

    fields: list[str] = []
    outputs: list[str] = []
    renames: dict[str, str] = {}
    group_keys: list[str] = []
    measures: list[str] = []
    output_by_field: dict[str, str] = {}
    for column in columns:
        name = wire_name(column.expr)
        alias = column.alias_name
        output = alias or name
        previous = output_by_field.get(name)
        if previous is not None:
            # One `fields` entry is one column back.  Asking for the very same column twice is a
            # no-op, so it collapses; asking for it under two different names is a *copy*, and
            # de-duplicating the wire request while keeping both output names would hand back a
            # single column under whichever name the alias rename won.  Tier 2 writes
            # `x, x AS y` and gets it right, so this refuses rather than lying about the shape.
            if previous == output:
                continue
            raise CannotCompile(
                f"{name} is selected twice under different names ({previous}, {output}): the "
                "query API returns one column per field, so the copy has to be made outside the "
                "semantic query"
            )
        output_by_field[name] = output
        fields.append(name)
        # Dimensions + measures IS the group-by, so the split is read straight off the
        # projection — a Project of dims+measures and the equivalent Aggregate agree here.
        (measures if isinstance(column.expr, MeasureRef) else group_keys).append(name)
        outputs.append(output)
        if alias is not None:
            renames[name] = alias

    filter_set = FilterSet({})
    if shape.predicates:
        # _match walks root-down, so the LAST filter() written is the first one collected;
        # reversing puts the conjuncts (and the composites they merge into) in the order the
        # user wrote them, which is what explain() and the snapshots read back.
        predicates = list(reversed(shape.predicates))
        predicate: Expr = (
            predicates[0] if len(predicates) == 1 else BooleanOp(BoolOp.AND, tuple(predicates))
        )
        filter_set = compile_filters(predicate, resolve=resolve)
    filters = dict(filter_set.entries)

    column_totals: tuple[str, ...] = ()
    if totals:
        if not measures:
            raise CompileError(
                "with_totals() needs at least one governed measure in the query: a totals row "
                "re-aggregates the measures over every row the query touched, and a query "
                "without a measure has nothing to total. Select F.measure(...) — or use "
                "group_by(...).agg(F.measure(...))."
            )
        column_totals = (GRAND_TOTAL_KEY,)

    sorts: list[Sort] = []
    sort_summary: list[tuple[str, bool]] = []
    if shape.sort is not None:
        for key in shape.sort.keys:
            column_name = _sort_column(key, resolve)
            sorts.append(Sort(column_name, key.descending, NullSort.OMNI_DEFAULT))
            sort_summary.append((column_name, key.descending))

    limit = UNSET if shape.limit is None else shape.limit.n
    offset = 0 if shape.limit is None else shape.limit.offset

    query = Query(
        model_id=_model_id(shape.scan.source),
        fields=fields,
        table=_table(shape.scan.source),
        join_paths_from_topic_name=_topic(shape.scan.source),
        filters=filters,
        sorts=sorts,
        limit=limit,
        offset=offset,
        column_totals=column_totals,
    )
    query.validate()

    return SemanticCompilation(
        query=query,
        scan=shape.scan.source,
        aliases=renames,
        columns=tuple(outputs),
        sorts=tuple(sort_summary),
        group_keys=tuple(group_keys),
        measures=tuple(measures),
        # A filter written against a measure's ALIAS resolves to a plain field name, so the
        # selected measures are the second way an entry is known to be a HAVING.  Both spellings
        # put the same bytes on the wire; this is only about reading the plan back.
        measure_filters=tuple(
            key for key in filters if key in filter_set.measure_keys or key in set(measures)
        ),
    )


# --------------------------------------------------------------------------------------
# Opaque sources: raw SQL and stored queries (CONTRACT_NOTES §3.4 / §4)
# --------------------------------------------------------------------------------------


def _compile_opaque(shape: _Shape, *, totals: bool) -> SemanticCompilation:
    """Compile a scan whose payload omniframes does not write: raw SQL, or a stored query.

    Neither can absorb an operation: a raw-SQL job is a string the server runs as-is, and a
    stored query goes back exactly as it arrived.  So the *only* shape that compiles is the
    bare scan — everything written on top of it is the splitter's problem, and lands local.
    """
    source = shape.scan.source
    kind = "a raw-SQL job" if isinstance(source, nodes.SqlScan) else "a stored query"
    if shape.projects or shape.predicates or shape.sort is not None or shape.limit is not None:
        raise CannotCompile(
            f"{kind} is sent to Omni exactly as written, so it cannot absorb a select(), "
            "filter(), sort() or limit() — those run in the local engine, over its result"
        )
    if totals:
        raise CompileError(
            f"with_totals() asks the server to total the query's measures, and {kind} carries "
            "no measures omniframes knows about. Put column_totals in the SQL (or in the stored "
            "query) instead."
        )

    if isinstance(source, nodes.SqlScan):
        # `Query.for_sql` pins `rewriteSql: false`; without it the server parses the SQL as
        # OmniSQL instead of running it verbatim (CONTRACT_NOTES §3.5).
        query = Query.for_sql(source.model_id, source.sql)
        query.validate()
        return SemanticCompilation(query=query, scan=source, aliases={}, columns=(), tier=2)

    if not isinstance(source, nodes.SavedQueryScan):  # pragma: no cover - the union is closed
        raise CannotCompile(f"{type(source).__name__} is not a source omniframes can run")
    return _compile_stored(source)


def _compile_stored(source: nodes.SavedQueryScan) -> SemanticCompilation:
    """Describe a stored query well enough for ``explain()``; the blob itself goes on the wire."""
    blob = dict(source.query)
    model_id = blob.get("modelId")
    if not isinstance(model_id, str) or not model_id:
        raise CompileError(
            f"the stored query {source.name!r} carries no modelId, so Omni cannot tell which "
            "model (and therefore which connection) to run it against. Pass model=... when "
            "reading it."
        )
    fields = tuple(str(name) for name in _sequence(blob.get("fields")))
    query = Query(
        model_id=model_id,
        fields=fields,
        table=str(blob.get("table") or ""),
        join_paths_from_topic_name=_optional_str(blob.get("join_paths_from_topic_name")),
        limit=_blob_limit(blob),
        offset=_blob_offset(blob),
    )
    return SemanticCompilation(
        query=query,
        scan=source,
        aliases={},
        # A semantic query's result columns are its `fields`, verbatim (brackets included), so
        # a stored query does know its own columns — unlike raw SQL.
        columns=fields,
        envelope_query=blob,
    )


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _blob_offset(blob: Mapping[str, Any]) -> int:
    offset = blob.get("offset")
    return offset if isinstance(offset, int) and not isinstance(offset, bool) else 0


def _blob_limit(blob: Mapping[str, Any]) -> int | Unset | None:
    """The stored query's limit, carried through the same trichotomy the wire uses.

    An **absent** key is not the library default and not "unlimited": the blob rides the wire
    verbatim, so the server fills in :data:`DEFAULT_SERVER_LIMIT` (CONTRACT_NOTES §3).  Describing
    it as anything else makes ``explain()`` claim rows the query will never return.
    """
    if "limit" not in blob:
        return DEFAULT_SERVER_LIMIT
    limit = blob["limit"]
    if limit is None:
        return None
    return limit if isinstance(limit, int) and not isinstance(limit, bool) else UNSET


def envelope_limit(blob: Mapping[str, Any]) -> int | None:
    """The limit the **server** applies to a query blob sent verbatim; ``None`` = unlimited.

    Omniframes always writes an explicit ``limit`` (docs/DESIGN.md §3), but a stored or
    AI-generated blob may simply not carry the key.  ``createQuery`` then applies
    :data:`DEFAULT_SERVER_LIMIT` (CONTRACT_NOTES §3) — reading the absent key as ``None`` would
    report a 1000-row truncation as an unlimited query and suppress the ``TruncationWarning``
    that goes with it.
    """
    if "limit" not in blob:
        return DEFAULT_SERVER_LIMIT
    limit = blob["limit"]
    return limit if isinstance(limit, int) and not isinstance(limit, bool) else None


def try_semantic(plan: nodes.PlanNode, *, totals: bool = False) -> SemanticCompilation | None:
    """:func:`compile_semantic`, returning ``None`` instead of raising :class:`CannotCompile`."""
    try:
        return compile_semantic(plan, totals=totals)
    except CannotCompile:
        return None


def _sort_column(key: SortKey, resolve: Resolver) -> str:
    """The exact ``sorts[*].column_name``, with aliases reverse-resolved (brackets included)."""
    expr = key.expr
    if isinstance(expr, FieldRef):
        resolved = resolve(expr.name)
        if resolved is not None:
            if expr.grain is not None:
                raise CompileError(
                    f"{expr.name!r} is an alias; apply .grain() in select(), not in sort()"
                )
            return resolved
    return wire_name(expr)


def _model_id(source: nodes.ScanSource) -> str:
    if isinstance(source, nodes.TopicScan | nodes.ViewScan | nodes.SqlScan):
        return source.model_id
    raise CannotCompile(f"{type(source).__name__} is not a tier-1 source yet")


def _table(source: nodes.ScanSource) -> str:
    """``query.table`` — the base view (CONTRACT_NOTES §3)."""
    if isinstance(source, nodes.TopicScan):
        return source.base_view
    if isinstance(source, nodes.ViewScan):
        return source.view
    return ""


def _topic(source: nodes.ScanSource) -> str | None:
    """``join_paths_from_topic_name`` — set for topics, absent for bare views."""
    return source.topic if isinstance(source, nodes.TopicScan) else None


# --------------------------------------------------------------------------------------
# The execution plan (docs/INTERNALS.md §4)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeOptions:
    """Session-level knobs that live on the run envelope rather than in the query (§2.1).

    ``branchId`` is top-level only — inside ``query`` it is a hard 400 — which is why it lives
    here and not on :class:`~omniframes.compile.querymodel.Query`.
    """

    branch_id: str | None = None
    cache: CachePolicy | None = None
    timezone: str | None = None
    user_id: str | None = None


@dataclass(frozen=True)
class RemoteStep:
    """One query omniframes sends to Omni, with everything needed to interpret its result.

    ``label`` names the step's role for ``explain()`` and for the truncation warning: a plain
    query is ``"semantic"``, while the decomposed half of a mixed aggregate (docs/HYBRID.md
    §2.1) is a ``"raw scan"``.  ``note`` is an explain-only suffix (``"(unlimited)"``), and
    ``user_limit`` records that the applied limit is the user's own ``.limit()`` — the one
    ``show()``/``first()`` may suppress a warning for, unlike an intermediate scan.
    """

    envelope: WireDict
    compilation: SemanticCompilation
    tier: int = 1
    label: str = "semantic"
    note: str = ""
    user_limit: bool = False

    def __post_init__(self) -> None:
        """Refuse, by construction, to put SQL on the wire under the wrong reading.

        The server picks between the two ``userEditedSQL`` paths from the ``rewriteSql`` key
        alone (CONTRACT_NOTES §3.5/§3.6): ``false`` runs the text verbatim on the warehouse,
        an absent key parses it as OmniSQL and plans a governed model job.  Taking the wrong
        one is the failure mode the server does not report — verbatim SQL that got parsed comes
        back as a well-formed answer to a question nobody asked, and OmniSQL sent verbatim ships
        ``${...}`` refs to the warehouse.  So the bytes must agree with what the compilation
        says it wrote, and every remote step passes through here.
        """
        query = self.envelope.get("query")
        if not isinstance(query, Mapping):
            return
        sql = query.get("userEditedSQL")
        if not (isinstance(sql, str) and sql.strip()):
            return
        if self.compilation.query.omnisql:
            if "rewriteSql" in query:
                raise CompileError(
                    "this query was compiled as OmniSQL but carries a rewriteSql key "
                    f"({query['rewriteSql']!r}); only an ABSENT key selects the parsed path, "
                    "so the server would send the statement to the warehouse verbatim "
                    "(CONTRACT_NOTES §3.6). Refusing to send it."
                )
            return
        if query.get("rewriteSql") is not False:
            raise CompileError(
                "this query carries userEditedSQL without rewriteSql: false, and the server "
                "would parse the text as OmniSQL instead of running it verbatim "
                "(CONTRACT_NOTES §3.5). Refusing to send it."
            )

    @property
    def query(self) -> Query:
        return self.compilation.query

    @property
    def applied_limit(self) -> int | None:
        """The ``limit`` this step actually sends; ``None`` means unlimited.

        Read off the envelope for a verbatim stored query, because there the bytes — not the
        typed :class:`Query` that merely describes them — are what the server applies.  A blob
        with no ``limit`` key at all reads as :data:`DEFAULT_SERVER_LIMIT`, the value the server
        fills in for it (CONTRACT_NOTES §3) — see :func:`envelope_limit`.
        """
        override = self.compilation.envelope_query
        if override is None:
            return self.query.effective_limit
        return envelope_limit(override)

    @property
    def alias_map(self) -> Mapping[str, str]:
        """``wire name -> alias``, ready for :func:`omniframes.transport.normalize.normalize`."""
        return self.compilation.aliases

    @property
    def aliases(self) -> Mapping[str, str]:
        return self.compilation.aliases

    @property
    def columns(self) -> tuple[str, ...]:
        return self.compilation.columns

    @property
    def sorts(self) -> tuple[tuple[str, bool], ...]:
        return self.compilation.sorts

    @property
    def group_keys(self) -> tuple[str, ...]:
        return self.compilation.group_keys

    @property
    def measures(self) -> tuple[str, ...]:
        return self.compilation.measures

    @property
    def measure_filters(self) -> tuple[str, ...]:
        """``filters`` keys the server applies as a HAVING (CONTRACT_NOTES §3.1)."""
        return self.compilation.measure_filters

    @property
    def source_label(self) -> str:
        return self.compilation.scan.label


@dataclass(frozen=True)
class LocalStep:
    """One local operator in the execution DAG, with the steps that feed it.

    ``op`` is a :class:`~omniframes.compile.local.LocalOp`; the annotation is deferred so that
    tier 1 keeps no runtime dependency on tier 3 (docs/HYBRID.md §1).
    """

    op: LocalOp
    inputs: tuple[Step, ...]

    def __post_init__(self) -> None:
        if not self.inputs:
            raise CompileError("a local step needs at least one input")


#: A node of the execution DAG: a query sent to Omni, or an operator run here.
Step: TypeAlias = "RemoteStep | LocalStep"


@dataclass(frozen=True)
class ExecutionPlan:
    """What running an action actually does.

    M1/M2 always produce exactly one remote step and no local work; that stays the degenerate
    case — ``root is None`` means "just run ``steps[0]``".  M3's splitter builds a DAG instead
    (docs/HYBRID.md §1): ``steps`` lists every remote query in DFS order (which is how
    ``explain()`` numbers them) and ``root`` is the operator whose output the user receives.
    """

    steps: tuple[RemoteStep, ...]
    #: The DAG root. ``None`` ⇒ the degenerate single-remote-step plan.
    root: Step | None = None
    #: The final user-facing columns when :attr:`root` is local; ``None`` when they cannot be
    #: known without running (``map_pandas()`` with no schema hint).
    output_columns: tuple[str, ...] | None = None

    @property
    def remote(self) -> RemoteStep:
        """The single remote step. Raises as soon as there is local work or a second query."""
        if self.root is not None or len(self.steps) != 1:
            raise CompileError(
                "this execution plan is a DAG (several remote steps and/or local operators); "
                "there is no single remote query to read off it — see explain()"
            )
        return self.steps[0]

    @property
    def envelope(self) -> WireDict:
        return self.remote.envelope

    @property
    def alias_map(self) -> Mapping[str, str]:
        return self.remote.alias_map

    @property
    def columns(self) -> tuple[str, ...]:
        """The user-facing column names, in result order."""
        if self.output_columns is not None:
            return self.output_columns
        if self.root is not None:
            raise CompileError(
                "this frame's columns are only known once it runs: map_pandas() hands the frame "
                "to a Python function, and a raw-SQL job decides its own result columns — "
                "neither can be planned. Pass schema_hint=OmniSchema(...) to map_pandas(), or "
                "select() the columns you want out of a SQL scan."
            )
        step = self.remote
        if not step.columns and step.compilation.opaque:
            raise CompileError(
                "a raw-SQL job decides its own result columns, and omniframes only learns them "
                "from the server. Use .schema (one planOnly round trip), select() the columns "
                "you want, or collect() and read the Arrow schema of the result."
            )
        return step.columns

    @property
    def tier(self) -> int:
        """The LOWEST tier involved — local work makes the whole plan tier 3."""
        if self.root is not None:
            return 3
        return self.remote.tier


def compile_plan(
    plan: nodes.PlanNode,
    *,
    options: EnvelopeOptions | None = None,
    totals: bool = False,
) -> ExecutionPlan:
    """Compile a logical plan into the steps that execute it.

    M2 knows one tier: a non-tier-1 plan raises ``CompileError("not yet supported: …")``
    carrying the :class:`CannotCompile` reason rather than falling back silently to pandas.
    ``totals`` is the frame-level ``with_totals()`` marker (docs/INTERNALS.md §5).
    """
    try:
        compilation = compile_semantic(plan, totals=totals)
    except CannotCompile as exc:
        raise CompileError(f"not yet supported: {exc.reason}") from exc

    return ExecutionPlan(
        (
            RemoteStep(
                build_envelope(compilation, options),
                compilation,
                tier=compilation.tier,
                label=compilation.role,
            ),
        )
    )


def build_envelope(compilation: SemanticCompilation, options: EnvelopeOptions | None) -> WireDict:
    """The exact body ``POST /api/v1/query/run`` receives for one compiled step.

    A stored query rides through untouched: omniframes did not write that blob, and
    round-tripping it through :class:`~omniframes.compile.querymodel.Query` would normalize keys
    it was handed (CONTRACT_NOTES §4).  Everything omniframes *did* write is validated first.
    """
    request = RunRequest(
        compilation.query,
        branch_id=None if options is None else options.branch_id,
        cache=None if options is None else options.cache,
        timezone=None if options is None else options.timezone,
        user_id=None if options is None else options.user_id,
    )
    if compilation.envelope_query is None:
        request.validate()
        return request.to_wire()
    wire = request.to_wire()
    wire["query"] = dict(compilation.envelope_query)
    return wire
