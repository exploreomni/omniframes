"""Tier 2 — the SQL-job compiler (docs/SQLTIER.md).

Tier 2 exists to kill tier 3's biggest cost: the **unlimited raw scan** feeding a local
aggregate.  A ``GROUP BY`` that runs in the warehouse moves five orders of magnitude less data,
and everything else this tier picks up — a cross-field ``OR`` in ``WHERE``, a computed ``SELECT``
expression, a ``HAVING`` over an ad-hoc aggregate, ``ORDER BY``/``LIMIT`` on top — rides the same
envelope for free.

A tier-2 job is **one** run envelope whose query is a raw-SQL job (``userEditedSQL`` +
``rewriteSql: false``) over a **reference core**: a governed tier-1 query passed in
``staticQueryReferences`` under the key ``ref_1`` and named as a table by the SQL.  So the rows
the SQL sees are still the rows Omni's model would have handed back — same joins, same access
grants, same filters — and only the shape of the computation is written by omniframes.

Three rules make that safe, and each one is enforced here rather than hoped for:

* **The reference core is built with the tier-1 compiler.**  This module constructs plan nodes
  and calls :func:`~omniframes.compile.semantic.compile_semantic`; it never hand-assembles a
  :class:`~omniframes.compile.querymodel.Query`.  A filter that tier 1 can express is pushed
  *into* the reference (governed, pre-aggregation); one it cannot moves to the outer ``WHERE``.
* **Values are never string-formatted into SQL.**  Every literal becomes a typed
  :mod:`sqlglot` node, so a filter value containing a quote — or a whole statement — arrives at
  the warehouse as a literal and nothing else.  The injection test pins it.  The rendering of a
  literal is nonetheless dialect-dependent where a backslash escapes inside a string (Snowflake,
  BigQuery, Redshift, Databricks/Spark, MySQL), and omniframes is never told the connection's
  dialect — so a backslash-bearing value is *refused* unless ``sql_dialect`` names the warehouse
  (:func:`_refuse_dialect_sensitive_literals`), and the constructed ``LIKE`` patterns escape with
  ``!`` rather than ``\\`` (:data:`_LIKE_ESCAPE`), which does not even tokenize there.
* **Governed measures never appear.**  A measure has no client-side definition, so a plan that
  mentions one is refused here and the splitter keeps the governed half remote (docs/HYBRID.md
  §2.1).  Only the ad-hoc half is ever rewritten as SQL.

Anything this module cannot express raises :class:`~omniframes.compile.semantic.CannotCompile`
internally and surfaces as ``None`` from :func:`try_sql`; the splitter then falls through to the
local engine.  A tier-2 construction failure is never a user error — the chosen tier is always
visible in ``explain()``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Final

from sqlglot import exp

from omniframes.column import (
    AdHocAgg,
    AggFn,
    Arithmetic,
    ArithOp,
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
)
from omniframes.compile.querymodel import DEFAULT_FETCH_LIMIT, UNSET, Query, Unset
from omniframes.compile.semantic import (
    TIMESTAMP_GRAINS,
    CannotCompile,
    SemanticCompilation,
    column_key,
    compile_filters,
    compile_semantic,
    display_name,
    split_grain,
    wire_name,
)
from omniframes.errors import CompileError
from omniframes.plan import nodes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omniframes.compile.splitter import SplitOptions

__all__ = [
    "REFERENCE_PREFIX",
    "SQL_EXPLAIN_LINES",
    "compile_sql",
    "reference_summary",
    "render_expr",
    "sql_lines",
    "try_sql",
]

#: ``staticQueryReferences`` keys are ``ref_1``, ``ref_2``, … in first-use order: deterministic,
#: and a bare SQL identifier so the outer statement can name one as a table (CONTRACT_NOTES §3.5).
REFERENCE_PREFIX: Final = "ref"

#: How many lines of generated SQL ``explain()`` prints before eliding the rest.
SQL_EXPLAIN_LINES: Final = 8

#: The escape character the constructed ``LIKE`` patterns use for ``%``, ``_`` and itself.
#: Deliberately **not** a backslash: ``ESCAPE '\'`` does not even tokenize on Snowflake,
#: BigQuery, Redshift, Databricks/Spark or MySQL, where a backslash escapes inside a string
#: literal, so ``'\'`` swallows the closing quote.  ``!`` has no meaning inside a string literal
#: in any dialect, and the pattern builder escapes it like any other special character.
_LIKE_ESCAPE: Final = "!"

#: Omni's relative date grammar (CONTRACT_NOTES §3.1).  Those literals only have a meaning
#: *server-side*, so a predicate that would have to render one as SQL is refused here exactly as
#: the local engine refuses to evaluate one (docs/HYBRID.md §3.1).
_RELATIVE_DATE: Final = re.compile(
    r"^\s*(?:"
    r"today|yesterday|tomorrow"
    r"|\d+\s+(?:complete\s+)?\w+\s+(?:ago|from\s+now)"
    r"|(?:this|last|next)\s+\w+"
    r")\s*$",
    re.IGNORECASE,
)

_AGGREGATES: Final[Mapping[AggFn, type[exp.AggFunc]]] = {
    AggFn.SUM: exp.Sum,
    AggFn.COUNT: exp.Count,
    AggFn.AVG: exp.Avg,
    AggFn.MIN: exp.Min,
    AggFn.MAX: exp.Max,
}

_COMPARISONS: Final[Mapping[CmpOp, type[exp.Binary]]] = {
    CmpOp.EQ: exp.EQ,
    CmpOp.NE: exp.NEQ,
    CmpOp.LT: exp.LT,
    CmpOp.LE: exp.LTE,
    CmpOp.GT: exp.GT,
    CmpOp.GE: exp.GTE,
}

_ARITHMETIC: Final[Mapping[ArithOp, type[exp.Binary]]] = {
    ArithOp.ADD: exp.Add,
    ArithOp.SUB: exp.Sub,
    ArithOp.MUL: exp.Mul,
    ArithOp.DIV: exp.Div,
}


# --------------------------------------------------------------------------------------
# Expr -> sqlglot (docs/SQLTIER.md §3)
# --------------------------------------------------------------------------------------


def render_expr(expr: Expr) -> exp.Expr:
    """Translate one omniframes expression into a sqlglot AST node.

    Public for the unit lane, which pins the translation table row by row.  Identifiers are
    always quoted (fields keep their dotted wire names) and values are always typed literal
    nodes — this function never formats a user value into a string of SQL.

    Raises:
        CannotCompile: the expression has no tier-2 rendering (a UDF, a governed measure, a
            relative date literal that would have to be compared as SQL).
    """
    if isinstance(expr, FieldRef):
        return exp.column(wire_name(expr), quoted=True)
    if isinstance(expr, MeasureRef):
        raise CannotCompile(
            f"{expr.name} is a governed measure: its definition lives in the model, so tier 2 "
            "cannot write the SQL for it (the governed half of an aggregate stays a tier-1 step)"
        )
    if isinstance(expr, Literal):
        return _literal(expr.value)
    if isinstance(expr, AdHocAgg):
        return _aggregate(expr)
    if isinstance(expr, Arithmetic):
        return _ARITHMETIC[expr.op](this=render_expr(expr.left), expression=render_expr(expr.right))
    if isinstance(expr, Comparison):
        return _comparison(expr)
    if isinstance(expr, BooleanOp):
        operands = [render_expr(operand) for operand in expr.operands]
        combined = exp.and_(*operands) if expr.op is BoolOp.AND else exp.or_(*operands)
        return exp.paren(combined) if len(operands) > 1 else combined
    if isinstance(expr, Not):
        return exp.not_(exp.paren(render_expr(expr.operand)))
    if isinstance(expr, IsNull):
        return exp.Is(this=render_expr(expr.operand), expression=exp.Null())
    if isinstance(expr, IsIn):
        return _isin(expr)
    if isinstance(expr, StringPredicate):
        return _string_predicate(expr)
    if isinstance(expr, Between):
        return _between(expr)
    raise CannotCompile(f"{display_name(expr)} has no SQL rendering omniframes can write")


def _literal(value: object) -> exp.Expr:
    """A Python value as a **typed** literal node. The only way a value enters the statement."""
    if value is None:
        return exp.Null()
    if isinstance(value, bool):
        # Before the int check: bool is a subclass of int, and TRUE is not 1 in SQL either.
        return exp.Boolean(this=value)
    if isinstance(value, datetime):
        # Before the date check, for the same reason.
        return exp.Cast(
            this=exp.Literal.string(value.strftime("%Y-%m-%d %H:%M:%S")),
            to=exp.DataType.build("TIMESTAMP"),
        )
    if isinstance(value, date):
        return exp.Cast(this=exp.Literal.string(value.isoformat()), to=exp.DataType.build("DATE"))
    if isinstance(value, int | float | Decimal):
        return exp.Literal.number(str(value))
    if isinstance(value, str):
        return exp.Literal.string(value)
    raise CannotCompile(f"{type(value).__name__} is not a value omniframes can render as SQL")


def _aggregate(expr: AdHocAgg) -> exp.Expr:
    operand = render_expr(expr.operand)
    if expr.fn is AggFn.COUNT and expr.distinct:
        return exp.Count(this=exp.Distinct(expressions=[operand]))
    return _AGGREGATES[expr.fn](this=operand)


def _comparison(expr: Comparison) -> exp.Expr:
    left, right = expr.left, expr.right
    _refuse_null_literal(left, expr)
    _refuse_null_literal(right, expr)
    _refuse_grained_date_literal(left, right)
    _refuse_grained_date_literal(right, left)
    _refuse_relative_date(left)
    _refuse_relative_date(right)
    return _COMPARISONS[expr.op](this=render_expr(left), expression=render_expr(right))


def _refuse_null_literal(side: Expr, expr: Comparison) -> None:
    if isinstance(side, Literal) and side.value is None:
        raise CannotCompile(
            f"comparing {display_name(expr)} to None is not a predicate; use "
            ".is_null() / .is_not_null()"
        )


def _refuse_relative_date(side: Expr) -> None:
    """Refuse a literal only Omni's date grammar can evaluate (docs/SQLTIER.md §1).

    Tier 1 sends those to the server, which understands them; the local engine refuses them by
    name.  Tier 2 would have to write them into SQL, where they mean nothing, so this is the
    third place the same rule is applied — conservatively, since falling through to tier 3 is
    always a correct answer and rendering "30 days ago" as a string never is.  That includes
    ``==``/``!=``: the grammar is the grammar whichever operator it sits next to, and a string
    value that merely *looks* relative still gets the right answer one tier down.
    """
    if (
        isinstance(side, Literal)
        and isinstance(side.value, str)
        and _RELATIVE_DATE.match(side.value)
    ):
        raise CannotCompile(
            f"{side.value!r} is a relative date literal, which only Omni evaluates: it cannot be "
            "written into SQL (CONTRACT_NOTES §3.1)"
        )


def _refuse_grained_date_literal(field: Expr, value: Expr) -> None:
    """Refuse a string compared against a field whose grain produces a timestamp (§3.1).

    ``.grain("month")`` pins the operand's type without a schema, which is exactly what tier 1
    reads to compile ``created_at.grain("month") == "2026-03"`` into a **date** filter.  Tier 2
    has no way to express Omni's date grammar in SQL: rendering it as a text comparison sends
    ``'2026-03'`` to the warehouse as a string (a conversion error on a strict dialect, a
    silently different row set on a coercing one), and ``between()`` would additionally turn
    tier 1's half-open upper bound into an inclusive one.  Tier 3 answers it correctly, so this
    declines instead (docs/SQLTIER.md §5).

    A **bare** field compared to a string stays as it is: with no grain there is no type to read,
    and tier 1 deliberately treats that as a string filter too (docs/INTERNALS.md §2).
    """
    if not isinstance(field, FieldRef) or not isinstance(value, Literal):
        return
    if not isinstance(value.value, str):
        return
    _, grain = split_grain(wire_name(field))
    if grain in TIMESTAMP_GRAINS:
        raise CannotCompile(
            f"{display_name(field)} carries the timestamp grain {grain!r}, so {value.value!r} is "
            "a date literal in Omni's grammar (CONTRACT_NOTES §3.1) and has no SQL rendering — "
            "compare it to a date/datetime instead, or let it run one tier down"
        )


def _isin(expr: IsIn) -> exp.Expr:
    for value in expr.values:
        # Same rule as ==/!=: one `IN` list is n equality tests, and each of them has to be a
        # value SQL can compare, not a phrase only Omni's date grammar understands.
        _refuse_grained_date_literal(expr.operand, Literal(value))
        _refuse_relative_date(Literal(value))
    return exp.In(
        this=render_expr(expr.operand),
        expressions=[_literal(value) for value in expr.values],
    )


def _string_predicate(expr: StringPredicate) -> exp.Expr:
    """CONTAINS / STARTS_WITH / ENDS_WITH / LIKE, all as a portable ``LIKE``.

    The three positional predicates build their pattern from the **escaped** value, so a value
    containing ``%`` or ``_`` matches those characters literally rather than acting as a
    wildcard; ``like()`` passes the user's pattern through, because there the wildcards are the
    point.  ``case_insensitive`` wraps both sides in ``LOWER()`` — ``ILIKE`` is not portable.
    """
    operand = render_expr(expr.operand)
    if expr.kind is StrPredKind.LIKE:
        pattern: exp.Expr = exp.Literal.string(expr.value)
        escaped = False
    else:
        pattern = exp.Literal.string(_like_pattern(expr.kind, expr.value))
        escaped = True
    if expr.case_insensitive:
        operand = exp.Lower(this=operand)
        pattern = exp.Lower(this=pattern)
    like: exp.Expr = exp.Like(this=operand, expression=pattern)
    if escaped:
        like = exp.Escape(this=like, expression=exp.Literal.string(_LIKE_ESCAPE))
    return like


def _like_pattern(kind: StrPredKind, value: str) -> str:
    escaped = value
    for character in (_LIKE_ESCAPE, "%", "_"):
        escaped = escaped.replace(character, _LIKE_ESCAPE + character)
    if kind is StrPredKind.STARTS_WITH:
        return f"{escaped}%"
    if kind is StrPredKind.ENDS_WITH:
        return f"%{escaped}"
    return f"%{escaped}%"


def _between(expr: Between) -> exp.Expr:
    """Numbers are inclusive at both ends; dates are half-open — matching M2 exactly."""
    _refuse_relative_date(Literal(expr.low))
    _refuse_relative_date(Literal(expr.high))
    _refuse_grained_date_literal(expr.operand, Literal(expr.low))
    _refuse_grained_date_literal(expr.operand, Literal(expr.high))
    operand = render_expr(expr.operand)
    lower = exp.GTE(this=operand, expression=_literal(expr.low))
    if isinstance(expr.high, date | datetime):
        upper: exp.Expr = exp.LT(this=operand.copy(), expression=_literal(expr.high))
    else:
        upper = exp.LTE(this=operand.copy(), expression=_literal(expr.high))
    return exp.paren(exp.and_(lower, upper))


# --------------------------------------------------------------------------------------
# The plan shape tier 2 accepts (docs/SQLTIER.md §1)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Select:
    """One output column of the generated statement."""

    name: str
    expr: Expr
    is_key: bool = False


@dataclass
class _Shape:
    """The pipeline recovered from a plan, before anything is rendered."""

    scan: nodes.Scan
    core: nodes.Aggregate | nodes.Project
    below: list[Expr] = field(default_factory=list)
    above: list[Expr] = field(default_factory=list)
    derived: list[tuple[str, Expr]] = field(default_factory=list)
    sort: nodes.Sort | None = None
    limit: nodes.Limit | None = None


def _match(plan: nodes.PlanNode) -> _Shape:
    """Walk the plan root-down, checking the operator order is one tier 2 can write."""
    sort: nodes.Sort | None = None
    limit: nodes.Limit | None = None
    core: nodes.Aggregate | nodes.Project | None = None
    above: list[Expr] = []
    below: list[Expr] = []
    derived: list[tuple[str, Expr]] = []

    node = plan
    while not isinstance(node, nodes.Scan):
        if isinstance(node, nodes.Limit):
            if limit is not None:
                raise CannotCompile("limit()/offset() applied more than once")
            if core is not None or above or below or derived or sort is not None:
                raise CannotCompile(
                    "select(), filter() or sort() after limit()/offset(): the limit pins the "
                    "pushdown frontier, so what is written above it runs on its result"
                )
            limit = node
        elif isinstance(node, nodes.Sort):
            if sort is not None or core is not None:
                raise CannotCompile("sort() applied more than once, or below an aggregate")
            sort = node
        elif isinstance(node, nodes.WithColumn):
            if core is not None:
                raise CannotCompile("with_column() below the projection tier 2 writes")
            derived.append((node.name, node.expr))
        elif isinstance(node, nodes.Filter):
            (below if core is not None else above).append(node.predicate)
        elif isinstance(node, nodes.Aggregate | nodes.Project):
            if core is not None:
                raise CannotCompile(
                    "two projections/aggregates in one tier-2 statement; the inner one is the "
                    "reference core and cannot also be the outer SELECT"
                )
            core = node
        else:
            raise CannotCompile(f"{type(node).__name__} is not a tier-2 operation")
        node = node.children[0]

    if core is None:
        raise CannotCompile("a tier-2 statement needs a select() or a group_by().agg()")
    if isinstance(node.source, nodes.SqlScan | nodes.SavedQueryScan):
        raise CannotCompile(
            "a raw-SQL job and a stored query decide their own result columns, so omniframes "
            "never re-derives SQL over one; everything above it runs in the local engine"
        )
    # `_match` collects root-down, so reversing puts every list back into the order the user
    # wrote it — which is what the golden `.sql` files read back.
    return _Shape(
        scan=node,
        core=core,
        below=list(reversed(below)),
        above=list(reversed(above)),
        derived=list(reversed(derived)),
        sort=sort,
        limit=limit,
    )


def _core_selects(core: nodes.Aggregate | nodes.Project) -> tuple[tuple[_Select, ...], bool]:
    """The core's output columns, in order, and whether the statement aggregates."""
    if isinstance(core, nodes.Aggregate):
        for key in core.keys:
            if not isinstance(key.expr, FieldRef):
                raise CannotCompile(
                    f"{display_name(key.expr)} is not a group key tier 2 can write; group by "
                    "dimensions (optionally at a grain)"
                )
        for agg in core.aggs:
            if not isinstance(agg.expr, AdHocAgg):
                raise CannotCompile(
                    f"{display_name(agg.expr)} is not an ad-hoc aggregation; a governed measure "
                    "stays a tier-1 step"
                )
        selects = [_Select(_name(column), column.expr, is_key=True) for column in core.keys]
        selects.extend(_Select(_name(column), column.expr) for column in core.aggs)
        return tuple(selects), True

    aggregated = any(isinstance(column.expr, AdHocAgg) for column in core.columns)
    selects = []
    for column in core.columns:
        expr = column.expr
        if isinstance(expr, MeasureRef):
            raise CannotCompile(
                f"{expr.name} is a governed measure; tier 2 never writes a measure's definition"
            )
        if aggregated and not isinstance(expr, AdHocAgg | FieldRef):
            raise CannotCompile(
                f"{display_name(expr)} is neither a group key nor an aggregation, so it has no "
                "value in an aggregated SELECT"
            )
        selects.append(
            _Select(_name(column), expr, is_key=aggregated and not isinstance(expr, AdHocAgg))
        )
    return tuple(selects), aggregated


def _name(column: Column) -> str:
    return column.alias_name or column_key(column.expr)


# --------------------------------------------------------------------------------------
# Substitution: an output column referenced from above is its defining expression
# --------------------------------------------------------------------------------------


def _substitute(expr: Expr, definitions: Mapping[str, Expr]) -> Expr:
    """Replace references to output columns with the expressions that produce them.

    ANSI ``HAVING`` and ``WHERE`` cannot see ``SELECT`` aliases, so a predicate written against
    ``buyers`` has to become the full ``COUNT(DISTINCT "users.id")``.  The replacement is
    single-pass by construction: derived columns are substituted against the map built so far,
    so a chain resolves without ever re-walking its own result.
    """
    if isinstance(expr, FieldRef | MeasureRef | AdHocAgg):
        return definitions.get(display_name(expr), expr)
    if isinstance(expr, Comparison):
        return Comparison(
            expr.op, _substitute(expr.left, definitions), _substitute(expr.right, definitions)
        )
    if isinstance(expr, Arithmetic):
        return Arithmetic(
            expr.op, _substitute(expr.left, definitions), _substitute(expr.right, definitions)
        )
    if isinstance(expr, BooleanOp):
        return BooleanOp(expr.op, tuple(_substitute(o, definitions) for o in expr.operands))
    if isinstance(expr, Not):
        return Not(_substitute(expr.operand, definitions))
    if isinstance(expr, IsNull):
        return IsNull(_substitute(expr.operand, definitions))
    if isinstance(expr, IsIn):
        return IsIn(_substitute(expr.operand, definitions), expr.values)
    if isinstance(expr, StringPredicate):
        return StringPredicate(
            expr.kind, _substitute(expr.operand, definitions), expr.value, expr.case_insensitive
        )
    if isinstance(expr, Between):
        return Between(_substitute(expr.operand, definitions), expr.low, expr.high)
    return expr


def _walk(expr: Expr) -> Iterator[Expr]:
    yield expr
    for child in expr.children:
        yield from _walk(child)


def _has_aggregate(expr: Expr) -> bool:
    return any(isinstance(sub, AdHocAgg) for sub in _walk(expr))


def _conjuncts(predicates: Sequence[Expr]) -> list[Expr]:
    """Flatten a stack of ``filter()`` calls into the conditions they AND together."""
    flat: list[Expr] = []
    for predicate in predicates:
        if isinstance(predicate, BooleanOp) and predicate.op is BoolOp.AND:
            flat.extend(predicate.operands)
        else:
            flat.append(predicate)
    return flat


def _fields(expr: Expr, seen: dict[str, FieldRef]) -> None:
    """Collect the bare fields an expression needs the reference to fetch, in first-use order."""
    for sub in _walk(expr):
        if isinstance(sub, FieldRef):
            seen.setdefault(wire_name(sub), sub)
        elif isinstance(sub, MeasureRef):
            raise CannotCompile(
                f"{sub.name} is a governed measure; tier 2 selects only bare fields from its "
                "reference"
            )


# --------------------------------------------------------------------------------------
# The compilation
# --------------------------------------------------------------------------------------


def try_sql(
    plan: nodes.PlanNode, *, options: SplitOptions | None = None
) -> SemanticCompilation | None:
    """:func:`compile_sql`, returning ``None`` for anything tier 2 cannot express.

    A tier-2 construction failure is never a user error: tier 3 can express everything tier 2
    can, so the splitter simply falls through to the local engine and ``explain()`` says which
    tier ran (docs/SQLTIER.md §5).
    """
    try:
        return compile_sql(plan, options=options)
    except (CannotCompile, CompileError):
        return None


def compile_sql(
    plan: nodes.PlanNode, *, options: SplitOptions | None = None
) -> SemanticCompilation:
    """Compile ``plan`` into a tier-2 SQL job over a governed reference core.

    Raises:
        CannotCompile: the plan is outside tier 2 (the reason names the operation).
        CompileError: the plan is tier-2 shaped but invalid.
    """
    shape = _match(plan)
    selects, aggregated = _core_selects(shape.core)

    definitions: dict[str, Expr] = {}
    for select in selects:
        definitions[select.name] = select.expr
        definitions.setdefault(column_key(select.expr), select.expr)

    outputs = list(selects)
    for name, raw in shape.derived:
        resolved = _substitute(raw, definitions)
        if aggregated and not _has_aggregate(resolved) and not _references_keys(resolved, outputs):
            raise CannotCompile(
                f"{name} is computed from rows the aggregate consumed, so it has no value in an "
                "aggregated SELECT"
            )
        replaced = [index for index, out in enumerate(outputs) if out.name == name]
        if replaced:
            if outputs[replaced[0]].is_key:
                raise CannotCompile(
                    f"{name} is a group key, and replacing it with a derived column would leave "
                    "the GROUP BY naming something the SELECT no longer produces"
                )
            outputs[replaced[0]] = _Select(name, resolved)
        else:
            outputs.append(_Select(name, resolved))
        definitions[name] = resolved

    where, having = _partition(shape, definitions, outputs, aggregated=aggregated)
    pushed, outer_where = _split_pushable(where)

    ordered = _order_by(shape.sort, definitions, outputs, aggregated=aggregated)

    # Fields in first-use order: the SELECT list, then WHERE, then HAVING, then ORDER BY — which
    # is the order a reader of the statement meets them.
    seen: dict[str, FieldRef] = {}
    for out in outputs:
        _fields(out.expr, seen)
    for predicate in (*outer_where, *having):
        _fields(predicate, seen)
    for sort_key in shape.sort.keys if shape.sort is not None else ():
        _fields(_substitute(sort_key.expr, definitions), seen)
    if not seen:
        raise CannotCompile("a tier-2 statement needs at least one field from its reference")

    reference = _reference(shape, tuple(seen.values()), pushed)
    reference_key = f"{REFERENCE_PREFIX}_1"

    statement = _statement(
        outputs,
        reference=reference_key,
        where=outer_where,
        group_keys=[out for out in outputs if out.is_key] if aggregated else [],
        having=having,
        ordered=ordered,
        limit=_sql_limit(shape.limit),
        offset=shape.limit.offset if shape.limit is not None else 0,
    )
    dialect = None if options is None else options.sql_dialect
    _refuse_dialect_sensitive_literals(statement, dialect)
    sql = statement.sql(dialect=dialect, pretty=True)

    query = Query.for_sql(
        reference.query.model_id,
        sql,
        sql_sorts_enabled=False,
        limit=_wire_limit(shape.limit),
        offset=shape.limit.offset if shape.limit is not None else 0,
        static_query_references={reference_key: reference.query},
    )
    query.validate()

    return SemanticCompilation(
        query=query,
        scan=shape.scan.source,
        # The statement already selects each column AS its final user-facing name, so there is
        # nothing left for normalize() to rename (docs/SQLTIER.md §2).
        aliases={},
        columns=tuple(out.name for out in outputs),
        sorts=tuple(
            (_sort_label(entry, outputs), entry.descending)
            for entry in (shape.sort.keys if shape.sort is not None else ())
        ),
        group_keys=tuple(out.name for out in outputs if out.is_key) if aggregated else (),
        tier=2,
    )


def _refuse_dialect_sensitive_literals(statement: exp.Expression, dialect: str | None) -> None:
    """Refuse a string literal whose SQL *text* depends on the warehouse's escaping rules.

    This module's promise is that "a value is a value, never a fragment of statement".  With
    sqlglot's default (ANSI-ish) dialect that promise holds only where a backslash is an ordinary
    character — ANSI, Postgres, DuckDB (``standard_conforming_strings``).  On every
    backslash-escaping warehouse (Snowflake, BigQuery, Redshift, Databricks/Spark, MySQL) the
    default rendering of ``x\\'`` closes the literal one character early and the rest of the
    value is parsed as SQL: a filter value can become a tautology, or a subquery against a table
    outside the governed model.  Benign values corrupt just as quietly (``C:\\Users\\dan``,
    ``a\\nb``, a trailing backslash).

    Omniframes cannot learn the connection's dialect — neither ``whoami`` nor ``/models`` reports
    it (CONTRACT_NOTES §4) — so when the caller has not named one with
    ``SessionBuilder.sql_dialect(...)``, a backslash-bearing literal is refused and the predicate
    is evaluated one tier down instead of rendered.  Falling through to tier 3 is always a
    correct answer (docs/SQLTIER.md §5); guessing the escaping rules is not.  With an explicit
    dialect sqlglot escapes for that dialect and the value rides through untouched.
    """
    if dialect is not None:
        return
    for literal in statement.find_all(exp.Literal):
        if literal.is_string and "\\" in str(literal.this):
            raise CannotCompile(
                "a filter value containing a backslash cannot be written into SQL without "
                "knowing the warehouse's string-escaping rules, and omniframes is never told "
                "them. Name the warehouse with OmniSession.builder.sql_dialect(...) to push this "
                "down, or let it run in the local engine"
            )


def _references_keys(expr: Expr, outputs: Sequence[_Select]) -> bool:
    """Whether every field this expression names is one of the aggregate's group keys.

    The line between "the same question, asked earlier" and "a different question": a predicate
    over group keys means the same thing before or after the ``GROUP BY``, and a predicate over
    a column the aggregate consumed means nothing at all above it.

    Fields *inside* an aggregate call are not counted: ``COUNT("order_items.id")`` is legal in a
    ``HAVING`` precisely because the aggregate, not the raw column, survives the ``GROUP BY``.
    """
    keys = {display_name(out.expr) for out in outputs if out.is_key}
    return _bare_fields(expr) <= keys


def _bare_fields(expr: Expr) -> set[str]:
    """The fields an expression names outside any aggregate call."""
    if isinstance(expr, AdHocAgg):
        return set()
    if isinstance(expr, FieldRef):
        return {display_name(expr)}
    fields: set[str] = set()
    for child in expr.children:
        fields |= _bare_fields(child)
    return fields


def _partition(
    shape: _Shape,
    definitions: Mapping[str, Expr],
    outputs: Sequence[_Select],
    *,
    aggregated: bool,
) -> tuple[list[Expr], list[Expr]]:
    """Split every predicate into the row half (``WHERE``) and the group half (``HAVING``)."""
    where: list[Expr] = list(_conjuncts(shape.below))
    having: list[Expr] = []
    for predicate in _conjuncts(shape.above):
        resolved = _substitute(predicate, definitions)
        if not _has_aggregate(resolved):
            if aggregated and not _references_keys(resolved, outputs):
                # Not tier 2's to reinterpret: above an aggregate that never produced the column,
                # this predicate is an error the splitter reports by name (docs/HYBRID.md §2.2),
                # and quietly moving it below the GROUP BY would answer a different question.
                raise CannotCompile(
                    f"filtering on {display_name(predicate)} above an aggregate that does not "
                    "produce it: only the group keys still exist there"
                )
            # Filtering on a group key before or after the GROUP BY is the same question, so a
            # row predicate written above the aggregate rides the WHERE half either way.
            where.append(resolved)
            continue
        if not aggregated:
            raise CannotCompile(
                f"filtering on {display_name(predicate)}: there is no aggregate above which it "
                "resolves to a column, so HAVING has nothing to filter"
            )
        if not _references_keys(resolved, outputs):
            # The same rule as the WHERE half, and the reason it cannot be skipped just because
            # an aggregate appears somewhere in the tree: `(n > 10) OR (users.age = 30)` is one
            # conjunct, so `users.age` would ride into the HAVING naming a column that is
            # neither grouped nor aggregated — a binder error on every ONLY_FULL_GROUP_BY engine
            # and an arbitrary-row answer on the rest.  Declining is always correct
            # (docs/SQLTIER.md §5); the splitter reports it by name.
            raise CannotCompile(
                f"filtering on {display_name(predicate)} above an aggregate that does not "
                "produce it: only the group keys and the aggregates still exist there"
            )
        having.append(resolved)
    return where, having


def _split_pushable(predicates: Sequence[Expr]) -> tuple[list[Expr], list[Expr]]:
    """Which row predicates ride *inside* the governed reference, and which stay in the SQL."""
    pushed: list[Expr] = []
    outer: list[Expr] = []
    for predicate in predicates:
        (pushed if _is_pushable(predicate) else outer).append(predicate)
    return pushed, outer


def _is_pushable(predicate: Expr) -> bool:
    """Whether tier 1 can express this predicate as a governed ``query.filters`` entry."""
    if any(isinstance(sub, MeasureRef) for sub in _walk(predicate)):
        return False
    try:
        compile_filters(predicate)
    except (CannotCompile, CompileError):
        return False
    return True


def _reference(
    shape: _Shape, fields: Sequence[FieldRef], pushed: Sequence[Expr]
) -> SemanticCompilation:
    """Build the reference core with the tier-1 compiler — never by hand (docs/SQLTIER.md §2)."""
    plan: nodes.PlanNode = shape.scan
    if pushed:
        predicate: Expr = pushed[0] if len(pushed) == 1 else BooleanOp(BoolOp.AND, tuple(pushed))
        plan = nodes.Filter(plan, predicate)
    plan = nodes.Project(plan, tuple(Column(_bare(ref)) for ref in fields))
    # Unlimited: a reference that silently paged would make the outer aggregate wrong, which is
    # the same reason the tier-3 raw scan is unlimited (docs/HYBRID.md §2.1).
    plan = nodes.Limit(plan, None)
    return compile_semantic(plan)


def _bare(ref: FieldRef) -> FieldRef:
    """The field as the reference must project it — grain included, exactly as it is named."""
    base, grain = split_grain(wire_name(ref))
    return FieldRef(base, grain)


def _order_by(
    sort: nodes.Sort | None,
    definitions: Mapping[str, Expr],
    outputs: Sequence[_Select],
    *,
    aggregated: bool,
) -> list[exp.Ordered]:
    if sort is None:
        return []
    lookup: dict[str, str] = {}
    for out in outputs:
        lookup[out.name] = out.name
        lookup.setdefault(column_key(out.expr), out.name)

    ordered: list[exp.Ordered] = []
    for key in sort.keys:
        name = lookup.get(display_name(key.expr))
        if name is not None:
            target: exp.Expr = exp.column(name, quoted=True)
        elif aggregated:
            raise CannotCompile(
                f"sorting by {display_name(key.expr)}: an aggregated query can only order by a "
                "column it selects"
            )
        else:
            target = render_expr(_substitute(key.expr, definitions))
        # Nulls last in both directions, exactly like the local engine (docs/HYBRID.md §3.4), so
        # the tier-2 and tier-3 answers are the same rows in the same order.
        ordered.append(exp.Ordered(this=target, desc=key.descending, nulls_first=False))
    return ordered


def _sort_label(key: SortKey, outputs: Sequence[_Select]) -> str:
    """The name ``explain()`` shows for one ORDER BY term — the output column it targets."""
    lookup: dict[str, str] = {}
    for out in outputs:
        lookup[out.name] = out.name
        lookup.setdefault(column_key(out.expr), out.name)
    return lookup.get(display_name(key.expr), display_name(key.expr))


def _sql_limit(limit: nodes.Limit | None) -> int | None:
    """The ``LIMIT`` the statement carries — the same value the envelope sends."""
    rows = _wire_limit(limit)
    return DEFAULT_FETCH_LIMIT if rows is UNSET else rows


def _wire_limit(limit: nodes.Limit | None) -> int | Unset | None:
    """``UNSET`` when the user never called ``.limit()``, so ``user_limit`` reads correctly."""
    return UNSET if limit is None else limit.n


def _statement(
    outputs: Sequence[_Select],
    *,
    reference: str,
    where: Sequence[Expr],
    group_keys: Sequence[_Select],
    having: Sequence[Expr],
    ordered: Sequence[exp.Ordered],
    limit: int | None,
    offset: int,
) -> exp.Select:
    """Assemble the statement. Every part is an AST node; nothing here concatenates SQL text."""
    projections: list[exp.Expr] = []
    for out in outputs:
        rendered = render_expr(out.expr)
        if isinstance(rendered, exp.Column) and rendered.name == out.name:
            projections.append(rendered)
        else:
            projections.append(exp.alias_(rendered, out.name, quoted=True))

    select = exp.Select().select(*projections).from_(exp.to_table(reference))
    if where:
        select = select.where(_all(where))
    if group_keys:
        select = select.group_by(*(render_expr(key.expr) for key in group_keys))
    if having:
        select = select.having(_all(having))
    if ordered:
        select = select.order_by(*ordered)
    if limit is not None:
        select = select.limit(limit)
    if offset:
        select = select.offset(offset)
    return select


def _all(predicates: Sequence[Expr]) -> exp.Expr:
    rendered = [render_expr(predicate) for predicate in predicates]
    return rendered[0] if len(rendered) == 1 else exp.and_(*rendered)


# --------------------------------------------------------------------------------------
# explain() support (docs/SQLTIER.md §4)
# --------------------------------------------------------------------------------------


def sql_lines(sql: str, *, budget: int = SQL_EXPLAIN_LINES) -> list[str]:
    """The generated statement as ``explain()`` prints it: at most ``budget`` lines, then a tail."""
    lines = sql.strip().splitlines()
    if len(lines) <= budget:
        return lines
    return [*lines[:budget], f"… (+{len(lines) - budget} more lines)"]


def reference_summary(query: Query) -> str:
    """One ``explain()`` line for a ``staticQueryReferences`` entry."""
    from omniframes.compile.explain import describe_filters

    parts = [f"fields [{', '.join(query.fields)}]"]
    filters = describe_filters(dict(query.filters))
    if filters:
        parts.append(f"filters: {filters}")
    limit = query.effective_limit
    parts.append("(unlimited)" if limit is None else f"(limit {limit})")
    return "  ".join(parts)
