"""Tier 2 — the OmniSQL compiler (docs/SQLTIER.md §3).

Tier 2 exists to kill tier 3's biggest cost: the **unlimited raw scan** feeding a local
aggregate.  A ``GROUP BY`` that runs in the warehouse moves five orders of magnitude less data,
and everything else this tier picks up — a cross-field ``OR`` in ``WHERE``, a computed ``SELECT``
expression, a governed measure sitting next to an ad-hoc aggregate, a ``HAVING``,
``ORDER BY``/``LIMIT`` on top — rides the same statement for free.

A tier-2 job is **one OmniSQL statement**: ``userEditedSQL`` with ``rewriteSql`` *absent*, which
is what makes the server parse the text against the model rather than hand it to the warehouse
(CONTRACT_NOTES §3.5/§3.6).  Joins come from the topic's relationships, ``${view.measure}``
expands to its governed SQL, row-level policies apply — the statement *is* the governed plan::

    SELECT ${users.state}, ${order_items.sale_price_sum},
        COUNT(DISTINCT ${users.id}) AS of_expr_1
    FROM ${order_items}
    GROUP BY 1
    HAVING COUNT(DISTINCT ${users.id}) > 10
    ORDER BY 3 DESC
    LIMIT 50000

Four rules make that safe, and each one is enforced here rather than hoped for:

* **Values are never string-formatted into SQL.**  Every literal becomes a typed :mod:`sqlglot`
  node, so a filter value containing a quote — or a whole statement — arrives at the warehouse as
  a literal and nothing else.  The injection tests pin it.
* **Model references never travel as text either.**  ``${users.state}`` is not a sqlglot
  identifier (the ``.`` splits into table/column, the ``$``/``{`` invite quoting), so every
  reference renders as an opaque ``__OF_REF_<n>__`` sentinel and is substituted *after*
  ``.sql()`` — see :class:`_References`.  A name that fails the sentinel charset is refused, so a
  hostile field name can never smuggle text into the statement, and a value that merely *looks*
  like a sentinel is refused too.
* **Only what the server guarantees is predicted** (docs/SQLTIER.md §3.2).  Bare refs come back
  under their canonical ``view.field`` name with aliases ignored and duplicates collapsed, so
  they are emitted unaliased and de-duplicated here; every other select item gets a generated
  ``of_expr_<n>`` alias that :func:`~omniframes.transport.normalize.normalize` matches by
  suffix, because the scope prefix the server prepends is not predictable.
* **The limit lives in the text.**  The server ignores the query object's ``limit`` on this path
  and a statement without ``LIMIT`` runs unlimited (§3.6), so the always-explicit-limit invariant
  is a ``LIMIT`` clause, not an envelope field.

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
    display_name,
    split_grain,
    wire_name,
)
from omniframes.errors import CompileError
from omniframes.plan import nodes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omniframes.compile.splitter import SplitOptions

__all__ = [
    "EXPR_ALIAS_PREFIX",
    "SQL_EXPLAIN_LINES",
    "compile_sql",
    "render_expr",
    "sql_lines",
    "try_sql",
]

#: How many lines of generated OmniSQL ``explain()`` prints before eliding the rest.
SQL_EXPLAIN_LINES: Final = 8

#: Prefix of the aliases generated for **expression** select items (docs/SQLTIER.md §3.2).  The
#: server honors the alias but prefixes it with a scope view no client can predict, so the result
#: column is matched by ``.endswith("." + alias)`` — which is unambiguous exactly because these
#: names are generated, globally unique within one statement, and never chosen by a user.
EXPR_ALIAS_PREFIX: Final = "of_expr_"

#: The escape character the constructed ``LIKE`` patterns use for ``%``, ``_`` and itself.
#: Deliberately **not** a backslash: the parser re-renders ``ESCAPE`` with the warehouse's own
#: escape character (CONTRACT_NOTES §3.6, probe L2), and ``ESCAPE '\'`` does not even tokenize on
#: Snowflake, BigQuery, Redshift, Databricks/Spark or MySQL.  ``!`` has no meaning inside a string
#: literal in any dialect, and the pattern builder escapes it like any other special character.
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

#: The opaque stand-in a model reference renders as inside the sqlglot AST.  ``[A-Z0-9_]`` only,
#: so no dialect quotes it, rewrites it or changes its case.
_SENTINEL_PREFIX: Final = "__OF_REF_"
_SENTINEL: Final = re.compile(r"__OF_REF_\d+__")

#: What a substituted name may contain: ``view.field``, ``view.field[grain]``, or a bare name.
#: Anything else refuses to tier 3 rather than becoming text inside the statement.
_REF_CHARSET: Final = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)?(\[[A-Za-z0-9_]+\])?$")

#: What a ``FROM ${…}`` target may contain — a topic or view name, so the first group alone.
_SOURCE_CHARSET: Final = re.compile(r"^[A-Za-z0-9_]+$")

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
# Sentinel substitution (docs/SQLTIER.md §3.1)
# --------------------------------------------------------------------------------------


class _References:
    """The model references one statement names, each behind an opaque sentinel.

    sqlglot cannot carry ``${users.state}`` as an identifier, so the AST holds
    ``__OF_REF_<n>__`` instead and the tokens are replaced *after* rendering.  The pass is a
    single regex sweep over the finished text: nothing it inserts is re-scanned, so a field
    legitimately named like a sentinel cannot chain into the next replacement — and it is refused
    at registration anyway, together with every name outside the charset.

    No user **value** ever travels through this: values are typed literal nodes in the AST, and a
    literal that merely *contains* the sentinel prefix refuses the whole statement to tier 3
    (:func:`_refuse_sentinel_literals`).
    """

    def __init__(self) -> None:
        self._tokens: dict[str, str] = {}
        self._names: dict[str, str] = {}

    def field(self, name: str) -> exp.Expr:
        """The sqlglot node standing in for ``${name}`` — a field, grain or measure ref."""
        return exp.column(self._sentinel(name, _REF_CHARSET, "field"), quoted=False)

    def source(self, name: str) -> exp.Expr:
        """The sqlglot table standing in for ``FROM ${name}`` — a topic, or a bare view."""
        return exp.to_table(self._sentinel(name, _SOURCE_CHARSET, "topic/view"))

    def _sentinel(self, name: str, charset: re.Pattern[str], kind: str) -> str:
        existing = self._tokens.get(name)
        if existing is not None:
            return existing
        if _SENTINEL_PREFIX in name or not charset.match(name):
            raise CannotCompile(
                f"{name!r} is not a {kind} name omniframes will write into an OmniSQL statement: "
                "a reference is substituted as text after the statement is rendered, so only "
                "letters, digits, underscores, one dot and one bracketed grain are accepted "
                "(docs/SQLTIER.md §3.1)"
            )
        token = f"{_SENTINEL_PREFIX}{len(self._tokens) + 1}__"
        self._tokens[name] = token
        self._names[token] = name
        return token

    def substitute(self, sql: str) -> str:
        """Replace every sentinel with its ``${…}`` reference, in one pass over the text."""

        def replace(match: re.Match[str]) -> str:
            name = self._names.get(match.group(0))
            if name is None:  # pragma: no cover - the literal guard makes this unreachable
                raise CannotCompile(
                    "the rendered statement contains a reference sentinel omniframes did not "
                    "write; refusing to substitute it"
                )
            return "${" + name + "}"

        return _SENTINEL.sub(replace, sql)


def _refuse_sentinel_literals(statement: exp.Expr) -> None:
    """Refuse a *value* that looks like a sentinel, so substitution can never touch one.

    The sentinel pass is textual by necessity, and the one thing that could put sentinel-shaped
    text into the rendered statement without going through :class:`_References` is a string
    literal a user chose.  Declining costs one tier and keeps the guarantee absolute.
    """
    for literal in statement.find_all(exp.Literal):
        if literal.is_string and _SENTINEL_PREFIX in str(literal.this):
            raise CannotCompile(
                f"a value containing {_SENTINEL_PREFIX!r} collides with the reference sentinels "
                "omniframes substitutes into an OmniSQL statement; it runs one tier down instead"
            )


# --------------------------------------------------------------------------------------
# Expr -> sqlglot (docs/SQLTIER.md §3.1)
# --------------------------------------------------------------------------------------


def render_expr(expr: Expr) -> str:
    """Translate one omniframes expression into the OmniSQL fragment it becomes.

    Public for the unit lane, which pins the translation table row by row.  Model references come
    back as ``${view.field}`` / ``${view.field[grain]}`` / ``${view.measure}``; values are always
    typed literal nodes — this function never formats a user value into a string of SQL.

    Raises:
        CannotCompile: the expression has no tier-2 rendering (a UDF, a relative date literal
            that would have to be compared as SQL, a name outside the sentinel charset).
    """
    refs = _References()
    rendered = _render(expr, refs)
    _refuse_sentinel_literals(rendered)
    return refs.substitute(rendered.sql())


def _render(expr: Expr, refs: _References) -> exp.Expr:
    if isinstance(expr, FieldRef | MeasureRef):
        return refs.field(wire_name(expr))
    if isinstance(expr, Literal):
        return _literal(expr.value)
    if isinstance(expr, AdHocAgg):
        return _aggregate(expr, refs)
    if isinstance(expr, Arithmetic):
        return _ARITHMETIC[expr.op](
            this=_operand(expr.left, refs), expression=_operand(expr.right, refs)
        )
    if isinstance(expr, Comparison):
        return _comparison(expr, refs)
    if isinstance(expr, BooleanOp):
        operands = [_render(operand, refs) for operand in expr.operands]
        combined = exp.and_(*operands) if expr.op is BoolOp.AND else exp.or_(*operands)
        return exp.paren(combined) if len(operands) > 1 else combined
    if isinstance(expr, Not):
        return exp.not_(exp.paren(_render(expr.operand, refs)))
    if isinstance(expr, IsNull):
        return exp.Is(this=_render(expr.operand, refs), expression=exp.Null())
    if isinstance(expr, IsIn):
        return _isin(expr, refs)
    if isinstance(expr, StringPredicate):
        return _string_predicate(expr, refs)
    if isinstance(expr, Between):
        return _between(expr, refs)
    raise CannotCompile(f"{display_name(expr)} has no SQL rendering omniframes can write")


def _operand(expr: Expr, refs: _References) -> exp.Expr:
    """Render one side of an arithmetic node, parenthesized if it is itself arithmetic.

    SQLGlot's generator prints the tree it is given and never re-derives grouping: an
    ``exp.Div(this=exp.Sub(a, b), expression=c)`` prints as ``a - b / c``, which the warehouse
    then reads by *its* precedence — silently the wrong number.  Grouping has to be explicit in
    the tree, so every nested operand gets its own parens.  Redundant ones are harmless; a
    missing one is a wrong answer.
    """
    rendered = _render(expr, refs)
    return exp.paren(rendered) if isinstance(expr, Arithmetic) else rendered


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


def _aggregate(expr: AdHocAgg, refs: _References) -> exp.Expr:
    operand = _render(expr.operand, refs)
    if expr.fn is AggFn.COUNT and expr.distinct:
        return exp.Count(this=exp.Distinct(expressions=[operand]))
    return _AGGREGATES[expr.fn](this=operand)


def _comparison(expr: Comparison, refs: _References) -> exp.Expr:
    left, right = expr.left, expr.right
    _refuse_null_literal(left, expr)
    _refuse_null_literal(right, expr)
    _refuse_grained_date_literal(left, right)
    _refuse_grained_date_literal(right, left)
    _refuse_relative_date(left)
    _refuse_relative_date(right)
    return _COMPARISONS[expr.op](this=_render(left, refs), expression=_render(right, refs))


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
    declines instead (docs/SQLTIER.md §1).

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


def _isin(expr: IsIn, refs: _References) -> exp.Expr:
    for value in expr.values:
        # Same rule as ==/!=: one `IN` list is n equality tests, and each of them has to be a
        # value SQL can compare, not a phrase only Omni's date grammar understands.
        _refuse_grained_date_literal(expr.operand, Literal(value))
        _refuse_relative_date(Literal(value))
    return exp.In(
        this=_render(expr.operand, refs),
        expressions=[_literal(value) for value in expr.values],
    )


def _string_predicate(expr: StringPredicate, refs: _References) -> exp.Expr:
    """CONTAINS / STARTS_WITH / ENDS_WITH / LIKE, all as a portable ``LIKE``.

    The three positional predicates build their pattern from the **escaped** value, so a value
    containing ``%`` or ``_`` matches those characters literally rather than acting as a
    wildcard; ``like()`` passes the user's pattern through, because there the wildcards are the
    point.  ``case_insensitive`` wraps both sides in ``LOWER()`` — ``ILIKE`` is not portable.
    """
    operand = _render(expr.operand, refs)
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


def _between(expr: Between, refs: _References) -> exp.Expr:
    """One canonical parenthesized range per column — numbers inclusive, dates half-open.

    The compound form is emitted as a single conjunct rather than two loose ones because that is
    the shape the OmniSQL parser was probed with: ``(${f} >= x AND ${f} < y)`` survives with both
    bounds intact (CONTRACT_NOTES §3.6, probe L1).
    """
    _refuse_relative_date(Literal(expr.low))
    _refuse_relative_date(Literal(expr.high))
    _refuse_grained_date_literal(expr.operand, Literal(expr.low))
    _refuse_grained_date_literal(expr.operand, Literal(expr.high))
    operand = _render(expr.operand, refs)
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
                    "two projections/aggregates in one tier-2 statement; one OmniSQL statement "
                    "writes one SELECT"
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
    """The core's output columns, in order, and whether the statement aggregates.

    A governed measure is a legal select item here (docs/SQLTIER.md §1): ``${view.measure}``
    expands server-side and mixes freely with ad-hoc aggregates in one statement, which is what
    lets a mixed ``agg()`` collapse into a single tier-2 job instead of decomposing.
    """
    if isinstance(core, nodes.Aggregate):
        for key in core.keys:
            if not isinstance(key.expr, FieldRef):
                raise CannotCompile(
                    f"{display_name(key.expr)} is not a group key tier 2 can write; group by "
                    "dimensions (optionally at a grain)"
                )
        for agg in core.aggs:
            if not _aggregates_rows(agg.expr):
                raise CannotCompile(
                    f"{display_name(agg.expr)} is neither an aggregation nor a governed measure, "
                    "so it has no value beside a GROUP BY"
                )
        selects = [_Select(_name(column), column.expr, is_key=True) for column in core.keys]
        selects.extend(_Select(_name(column), column.expr) for column in core.aggs)
        return tuple(selects), True

    aggregated = any(_is_aggregation(column.expr) for column in core.columns)
    selects = []
    for column in core.columns:
        expr = column.expr
        if aggregated and not isinstance(expr, FieldRef) and not _aggregates_rows(expr):
            raise CannotCompile(
                f"{display_name(expr)} is neither a group key nor an aggregation, so it has no "
                "value in an aggregated SELECT"
            )
        selects.append(
            _Select(_name(column), expr, is_key=aggregated and isinstance(expr, FieldRef))
        )
    return tuple(selects), aggregated


def _name(column: Column) -> str:
    return column.alias_name or column_key(column.expr)


def _is_aggregation(expr: Expr) -> bool:
    """Whether this expression collapses rows — an ad-hoc aggregate, or a governed measure."""
    return any(isinstance(sub, AdHocAgg | MeasureRef) for sub in _walk(expr))


def _aggregates_rows(expr: Expr) -> bool:
    """Whether this expression is legal beside a ``GROUP BY`` without being a key.

    Aggregates and measures qualify, and so does arithmetic over them
    (``${m1} / COUNT(DISTINCT ${f})`` — CONTRACT_NOTES §3.6, probe L3).  A bare field outside an
    aggregate call disqualifies the whole item: it is neither grouped nor aggregated, which is a
    binder error on every ``ONLY_FULL_GROUP_BY`` engine.
    """
    return _is_aggregation(expr) and not _bare_fields(expr)


# --------------------------------------------------------------------------------------
# Substitution: an output column referenced from above is its defining expression
# --------------------------------------------------------------------------------------


def _substitute(expr: Expr, definitions: Mapping[str, Expr]) -> Expr:
    """Replace references to output columns with the expressions that produce them.

    ANSI ``HAVING`` and ``WHERE`` cannot see ``SELECT`` aliases, so a predicate written against
    ``buyers`` has to become the full ``COUNT(DISTINCT ${users.id})``.  The replacement is
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


def _conjuncts(predicates: Sequence[Expr]) -> list[Expr]:
    """Flatten a stack of ``filter()`` calls into the conditions they AND together."""
    flat: list[Expr] = []
    for predicate in predicates:
        if isinstance(predicate, BooleanOp) and predicate.op is BoolOp.AND:
            flat.extend(predicate.operands)
        else:
            flat.append(predicate)
    return flat


# --------------------------------------------------------------------------------------
# The compilation
# --------------------------------------------------------------------------------------


def try_sql(
    plan: nodes.PlanNode, *, options: SplitOptions | None = None
) -> SemanticCompilation | None:
    """:func:`compile_sql`, returning ``None`` for anything tier 2 cannot express.

    A tier-2 construction failure is never a user error: tier 3 can express everything tier 2
    can, so the splitter simply falls through to the local engine and ``explain()`` says which
    tier ran (docs/SQLTIER.md §1).
    """
    try:
        return compile_sql(plan, options=options)
    except (CannotCompile, CompileError):
        return None


def compile_sql(
    plan: nodes.PlanNode, *, options: SplitOptions | None = None
) -> SemanticCompilation:
    """Compile ``plan`` into one OmniSQL statement the server parses against the model.

    Raises:
        CannotCompile: the plan is outside tier 2 (the reason names the operation).
        CompileError: the plan is tier-2 shaped but invalid.
    """
    # Nothing outside the plan changes the emission any more: the server re-renders the parsed
    # statement per warehouse, so there is no dialect to be told (docs/SQLTIER.md §6).
    del options

    shape = _match(plan)
    selects, aggregated = _core_selects(shape.core)

    definitions: dict[str, Expr] = {}
    for select in selects:
        definitions[select.name] = select.expr
        definitions.setdefault(column_key(select.expr), select.expr)

    outputs = list(selects)
    for name, raw in shape.derived:
        resolved = _substitute(raw, definitions)
        if (
            aggregated
            and not _aggregates_rows(resolved)
            and not _references_keys(resolved, outputs)
        ):
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
    _refuse_where_hazards(where)

    refs = _References()
    emission = _emit_selects(outputs, refs)
    ordered = _order_by(shape.sort, definitions, emission, refs, aggregated=aggregated)

    statement = _statement(
        emission,
        source=refs.source(_from_ref(shape.scan.source)),
        refs=refs,
        where=where,
        group_positions=(
            [emission.positions[out.name] for out in outputs if out.is_key] if aggregated else []
        ),
        having=having,
        ordered=ordered,
        limit=_sql_limit(shape.limit),
        offset=shape.limit.offset if shape.limit is not None else 0,
    )
    _refuse_sentinel_literals(statement)
    sql = refs.substitute(statement.sql(pretty=True))

    query = Query.for_omnisql(
        _model_id(shape.scan.source),
        sql,
        limit=_wire_limit(shape.limit),
        offset=shape.limit.offset if shape.limit is not None else 0,
    )
    query.validate()

    return SemanticCompilation(
        query=query,
        scan=shape.scan.source,
        aliases=emission.aliases,
        columns=emission.columns,
        sorts=tuple(
            (_sort_label(entry, outputs), entry.descending)
            for entry in (shape.sort.keys if shape.sort is not None else ())
        ),
        group_keys=tuple(out.name for out in outputs if out.is_key) if aggregated else (),
        measures=tuple(wire_name(out.expr) for out in outputs if isinstance(out.expr, MeasureRef)),
        tier=2,
    )


def _model_id(source: nodes.ScanSource) -> str:
    if isinstance(source, nodes.TopicScan | nodes.ViewScan):
        return source.model_id
    raise CannotCompile(f"{type(source).__name__} is not a tier-2 source")


def _from_ref(source: nodes.ScanSource) -> str:
    """The ``FROM ${…}`` target: the topic (which brings its join graph), or a bare view."""
    if isinstance(source, nodes.TopicScan):
        return source.topic
    if isinstance(source, nodes.ViewScan):
        return source.view
    raise CannotCompile(f"{type(source).__name__} has no OmniSQL FROM reference")


def _references_keys(expr: Expr, outputs: Sequence[_Select]) -> bool:
    """Whether every field this expression names is one of the aggregate's group keys.

    The line between "the same question, asked earlier" and "a different question": a predicate
    over group keys means the same thing before or after the ``GROUP BY``, and a predicate over
    a column the aggregate consumed means nothing at all above it.

    Fields *inside* an aggregate call are not counted: ``COUNT(${order_items.id})`` is legal in a
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
        if not _is_aggregation(resolved):
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
            # and an arbitrary-row answer on the rest.  Declining is always correct; the splitter
            # reports it by name.
            raise CannotCompile(
                f"filtering on {display_name(predicate)} above an aggregate that does not "
                "produce it: only the group keys and the aggregates still exist there"
            )
        having.append(resolved)
    return where, having


def _refuse_where_hazards(where: Sequence[Expr]) -> None:
    """The two shapes a ``WHERE`` conjunct may not contain (docs/SQLTIER.md §3.3).

    * A **grain ref**.  Mixing a bare ref and a grain ref of one field in a ``WHERE`` is the
      one live-observed predicate loss on the OmniSQL path — the tighter bound is silently
      dropped (CONTRACT_NOTES §3.6).  Rather than reason about which combinations are safe, no
      grain ever reaches the ``WHERE``: a grain predicate tier 1 can express never gets here
      (tier 1 wins first), and one it cannot refuses.  Grains stay legal as select items and
      group keys.
    * A **governed measure**.  Its definition is an aggregate, so it belongs in the ``HAVING``
      the partition already routes it to; naming one in a ``WHERE`` is a different query.
    """
    for predicate in where:
        for sub in _walk(predicate):
            if isinstance(sub, MeasureRef):
                raise CannotCompile(
                    f"{sub.name} is a governed measure and expands to an aggregate, so it "
                    "filters groups (HAVING), never rows (WHERE)"
                )
            if isinstance(sub, FieldRef) and split_grain(wire_name(sub))[1] is not None:
                raise CannotCompile(
                    f"{wire_name(sub)} is a grain reference, and the OmniSQL parser merges a "
                    "grain ref with the bare field in one WHERE with predicate loss "
                    "(CONTRACT_NOTES §3.6); this filter runs one tier down"
                )


# --------------------------------------------------------------------------------------
# Emission (docs/SQLTIER.md §3.1/§3.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Emission:
    """The rendered SELECT list, plus everything the rest of the statement reads off it."""

    #: The select items, in emitted order.
    projections: tuple[exp.Expr, ...]
    #: The user-facing output names, in result order.
    columns: tuple[str, ...]
    #: ``result column key -> user-facing name``: an exact wire name for a bare ref, an
    #: ``of_expr_<n>`` suffix key for an expression item.
    aliases: Mapping[str, str]
    #: ``output name`` and ``column_key`` alike → the item's 1-based position, for the
    #: positional GROUP BY / ORDER BY that is the only verified form (CONTRACT_NOTES §3.6).
    positions: Mapping[str, int]


def _emit_selects(outputs: Sequence[_Select], refs: _References) -> _Emission:
    """Render the select list under the two naming regimes of docs/SQLTIER.md §3.2.

    Bare refs (fields, grains, measures) are emitted **unaliased and de-duplicated**: the server
    ignores their aliases and collapses duplicates to one column, so emitting them twice would
    shift every position after them.  Everything else is an expression item and carries a
    generated ``of_expr_<n>`` alias, matched back by suffix because the scope prefix the server
    prepends is not predictable.
    """
    projections: list[exp.Expr] = []
    columns: list[str] = []
    aliases: dict[str, str] = {}
    positions: dict[str, int] = {}
    bare: dict[str, tuple[int, str]] = {}
    expressions = 0

    for select in outputs:
        if isinstance(select.expr, FieldRef | MeasureRef):
            wire = wire_name(select.expr)
            previous = bare.get(wire)
            if previous is None:
                projections.append(_render(select.expr, refs))
                position = len(projections)
                bare[wire] = (position, select.name)
                if select.name != wire:
                    aliases[wire] = select.name
                columns.append(select.name)
            else:
                position, name = previous
                if name != select.name:
                    # The server returns ONE column per distinct bare ref whatever the SQL says,
                    # so two output names over one field cannot both be produced remotely.  Tier
                    # 1 refuses the same shape; tier 3 makes the copy.
                    raise CannotCompile(
                        f"{wire} is selected twice under different names ({name}, {select.name}): "
                        "the OmniSQL parser de-duplicates a repeated field to one result column, "
                        "so the copy has to be made outside the statement"
                    )
        else:
            expressions += 1
            alias = f"{EXPR_ALIAS_PREFIX}{expressions}"
            projections.append(exp.alias_(_render(select.expr, refs), alias, quoted=False))
            position = len(projections)
            aliases[alias] = select.name
            columns.append(select.name)
        positions[select.name] = position
        positions.setdefault(column_key(select.expr), position)

    if not projections:
        raise CannotCompile("a tier-2 statement needs at least one selected column")

    return _Emission(
        projections=tuple(projections),
        columns=tuple(columns),
        aliases=aliases,
        positions=positions,
    )


def _order_by(
    sort: nodes.Sort | None,
    definitions: Mapping[str, Expr],
    emission: _Emission,
    refs: _References,
    *,
    aggregated: bool,
) -> list[exp.Ordered]:
    """``ORDER BY <position>`` for a selected column; the expression inline for anything else.

    Positional ordering is the only verified form, and over a formatted grain it sorts by the
    ``__raw`` timestamp — i.e. chronologically (CONTRACT_NOTES §3.6, probe P4).  A non-aggregated
    sort over a column the statement does not select renders inline: the server wraps the
    statement in a subquery whose ``omni_sort_expr_<n>`` sidecars never reach the result (L5).
    """
    if sort is None:
        return []
    ordered: list[exp.Ordered] = []
    for key in sort.keys:
        position = emission.positions.get(display_name(key.expr))
        if position is not None:
            target: exp.Expr = exp.Literal.number(position)
        elif aggregated:
            raise CannotCompile(
                f"sorting by {display_name(key.expr)}: an aggregated query can only order by a "
                "column it selects"
            )
        else:
            target = _render(_substitute(key.expr, definitions), refs)
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
    """The ``LIMIT`` the statement carries — and the only limit the server applies (§3.6)."""
    rows = _wire_limit(limit)
    return DEFAULT_FETCH_LIMIT if rows is UNSET else rows


def _wire_limit(limit: nodes.Limit | None) -> int | Unset | None:
    """``UNSET`` when the user never called ``.limit()``, so ``user_limit`` reads correctly."""
    return UNSET if limit is None else limit.n


def _statement(
    emission: _Emission,
    *,
    source: exp.Expr,
    refs: _References,
    where: Sequence[Expr],
    group_positions: Sequence[int],
    having: Sequence[Expr],
    ordered: Sequence[exp.Ordered],
    limit: int | None,
    offset: int,
) -> exp.Select:
    """Assemble the statement. Every part is an AST node; nothing here concatenates SQL text.

    There is deliberately no ``SELECT DISTINCT`` branch: the parser strips one silently
    (CONTRACT_NOTES §3.6), so de-duplication stays in the local engine.
    """
    select = exp.Select().select(*emission.projections).from_(source)
    if where:
        select = select.where(_all(where, refs))
    if group_positions:
        select = select.group_by(*(exp.Literal.number(n) for n in group_positions))
    if having:
        select = select.having(_all(having, refs))
    if ordered:
        select = select.order_by(*ordered)
    if limit is not None:
        select = select.limit(limit)
    if offset:
        select = select.offset(offset)
    return select


def _all(predicates: Sequence[Expr], refs: _References) -> exp.Expr:
    rendered = [_render(predicate, refs) for predicate in predicates]
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
