"""The parsed-OmniSQL path of the FakeOmniAPI — CONTRACT_NOTES §3.6, docs/SQLTIER.md §7.

``userEditedSQL`` **without** ``rewriteSql`` (the key absent, not ``false``) neither runs the
caller's SQL verbatim nor ignores it: the server parses the text as *OmniSQL*, binds every
``${…}`` reference against the model, and plans a governed model job — joins from the topic's
relationships pruned to the views the statement mentions, measures expanded to their governed
SQL.  This module is that path, executed offline over the bench dataset:

* ``${topic}`` in FROM position → the bench base table plus the topic's LEFT JOINs, pruned;
* ``${view.field}`` → the qualified DuckDB column, ``${view.field[grain]}`` → its ``date_trunc``
  (plus the §2.7 formatted sidecar when the grain is formatted);
* ``${view.measure}`` → the measure's aggregate expression from the bench model.

**Two naming regimes, reproduced exactly** (CONTRACT_NOTES §3.6, probe-pinned):

* a **bare ref** select item surfaces under its canonical ``view.field`` name — a SQL alias on it
  is *ignored*, which is tier-1 semantics: renames are the client's job;
* every other select item is an **expression item**: its SQL alias is honored, prefixed with a
  scope view.  The live prefix is not predictable (first-ref-wins is refuted), so the fake picks
  one **arbitrarily** — see :func:`scope_view`.  Only matching an expression column by its alias
  *suffix* can work against this fake, which is the only thing that works live either.

**Stricter than the server, on purpose.**  The server silently rewrites a handful of shapes; a
silent rewrite offline is a divergence that only shows up live, so each one is a loud
:class:`~tests.fakes.engine.PlanFailure` here instead — every message starts with
:data:`REJECTION`.

Substitution failures carry the **server's own text** (:data:`SUBSTITUTION_ERROR`), so the
client's tier-2 error mapping (docs/SQLTIER.md §8) is exercised offline.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import pyarrow as pa
import sqlglot
from sqlglot import exp

from tests.fakes.engine import (
    BenchEngine,
    PlanFailure,
    ResolvedField,
    arrow_data_type,
    fetch_arrow,
    grain_pair,
    sql_ident,
)
from tests.fakes.sqljobs import SQL_WRAPPER_ALIAS, SqlJob

__all__ = [
    "REJECTION",
    "SUBSTITUTION_ERROR",
    "is_omnisql_job",
    "no_such_field",
    "no_such_view",
    "run_omnisql_job",
    "scope_view",
]

#: The server's substitution-failure prefix (CONTRACT_NOTES §3.6).  docs/SQLTIER.md §8 makes this
#: the string the client's executor keys its tier-2 error mapping off, so it is spelled out here
#: rather than imported from the client — a drift between the two has to be a test failure.
SUBSTITUTION_ERROR: Final = "Could not substitute Omni SQL"

#: Prefix of every *loud* rejection this resolver adds on top of the server's own behavior.
#: Distinctive by design: "FakeOmniAPI is being stricter than the server" and "the server said
#: no" have to be tellable apart at a glance.
REJECTION: Final = "FakeOmniAPI rejects this OmniSQL statement"

#: ``${…}`` reference, as the server's OmniSQL parser sees it.
_REFERENCE = re.compile(r"\$\{([^{}]*)\}")

#: A single-quoted string literal, ``''`` doubling included — a capturing split pattern, so the
#: literals come back as their own chunks and stay untouched.
_STRING_LITERAL = re.compile(r"('(?:[^']|'')*')")

#: A resolvable field reference: ``view.field`` with an optional ``[grain]`` suffix.  A name
#: outside this charset cannot name a model field, so it gets the server's not-found error
#: instead of a chance to smuggle text into the statement.
_FIELD_REFERENCE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*(\[[A-Za-z0-9_]+\])?$"
)

#: A resolvable topic/view reference (FROM position).
_VIEW_REFERENCE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Query-object keys that have no meaning on the parsed path.  The statement IS the plan, so the
#: server has nowhere to put these and quietly drops or reinterprets them; an envelope that still
#: carries one is an emission bug (docs/SQLTIER.md §2 sends empty collections for all of them).
_REFUSED_KEYS: Final = (
    "filters",
    "sorts",
    "calculations",
    "column_totals",
    "row_totals",
    "pivots",
    "fill_fields",
    "staticQueryReferences",
    "sqlSortsEnabled",
)


def is_omnisql_job(query: Mapping[str, Any]) -> bool:
    """Whether this query rides the parsed-OmniSQL path (§3.6).

    ``userEditedSQL`` present and non-empty, with **no** "do not rewrite" marker next to it.  The
    marker's absence is the whole selector: with one, the text goes to the warehouse verbatim
    (:func:`tests.fakes.sqljobs.is_raw_sql_job`); without one it is parsed as OmniSQL.
    """
    sql = query.get("userEditedSQL")
    if not isinstance(sql, str) or not sql.strip():
        return False
    return not (
        query.get("rewriteSql") is False
        or query.get("parsed") is False
        or query.get("dbtMode") is True
    )


def no_such_view(name: str) -> str:
    """The server's error for an unresolvable FROM reference (CONTRACT_NOTES §3.6)."""
    return f'{SUBSTITUTION_ERROR}: No such view "{name}"'


def no_such_field(name: str) -> str:
    """The server's error for an unresolvable ``${view.field}`` reference (§3.6)."""
    return f'{SUBSTITUTION_ERROR}: Field "{name}" not found: No such field "{name}"'


def scope_view(views: Sequence[str]) -> str:
    """The prefix an *expression* select item comes back under.

    There is no live rule to copy: ``${users.age} + ${products.cost}`` and
    ``COALESCE(${products.brand}, ${users.country})`` both scope to ``users``, which refutes
    first-ref-wins and leaves no positional rule standing (CONTRACT_NOTES §3.6).  The fake
    therefore picks the lexicographically greatest referenced view — deterministic enough to
    assert on, arbitrary enough that a client which learned it would be learning a lie.
    """
    return max(views)


# --------------------------------------------------------------------------------------
# Bound references, select items, output columns
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ref:
    """One ``${…}`` occurrence, bound to what it names."""

    sentinel: str
    text: str
    #: The model field for a ``${view.field}`` reference; ``None`` for a topic/view reference.
    field: ResolvedField | None = None


@dataclass(frozen=True)
class _Item:
    """A select item as written, before substitution."""

    #: Set when the item is a lone ``${view.field}`` — naming regime (a).
    ref: _Ref | None
    #: The name an *expression* item is published under, before the scope prefix.
    alias: str


@dataclass(frozen=True)
class _Output:
    """One result column: its wire name and the DuckDB expression that produces it."""

    name: str
    expression: exp.Expr
    #: The bound model field, when the column came from a bare ref (regime a).
    field: ResolvedField | None = None
    #: Whether an expression item aggregates — the fake's stand-in for "is not a dimension".
    aggregate: bool = False
    #: A formatted-grain sidecar, appended after every positional column (§2.7).
    appended: bool = False


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run_omnisql_job(
    engine: BenchEngine,
    query: Mapping[str, Any],
    *,
    model_id: str,
    plan_only: bool = False,
) -> tuple[SqlJob, pa.Table | None]:
    """Compile and (unless ``plan_only``) run one parsed-OmniSQL job.

    ``model_id`` is taken for signature parity with the verbatim path; the statement carries no
    model reference of its own — it binds against the one topic this fake serves.
    """
    del model_id
    _refuse_semantic_baggage(query)
    statement_text = _statement_text(query)
    sentinel_sql, refs = _extract_references(statement_text)
    parsed = _parse(sentinel_sql)
    _refuse_rewritten_shapes(parsed)
    _bind(refs, engine)
    base_view = _from_view(parsed, refs, engine)
    items = _classify(parsed, refs)
    _refuse_where_hazard(parsed, refs)

    views = sorted({base_view, *(r.field.base.view_name for r in refs.values() if r.field)})
    bound = _substitute(parsed, refs)
    outputs = _outputs(bound, items, scope_view(views))
    sql = _render(bound, outputs, base_view, views, engine)

    schema = _probe(engine, sql)
    job = SqlJob(
        sql=sql,
        user_sql=statement_text,
        schema=schema,
        summary_fields=_summary_fields(outputs, schema),
        omni_sql=statement_text,
    )
    if plan_only:
        return job, None
    return job, _execute(engine, sql)


# --------------------------------------------------------------------------------------
# The envelope around the statement
# --------------------------------------------------------------------------------------


def _statement_text(query: Mapping[str, Any]) -> str:
    raw = query.get("userEditedSQL")
    if not isinstance(raw, str):  # pragma: no cover - guarded by is_omnisql_job
        raise PlanFailure("query.userEditedSQL must be a string")
    return raw.strip().rstrip(";").strip()


def _refuse_semantic_baggage(query: Mapping[str, Any]) -> None:
    """Loud reject: semantic keys riding along with a parsed-OmniSQL statement (SQLTIER §7).

    The statement is the whole plan — sorts, filters, totals and references have nowhere to go,
    and the server quietly drops or reinterprets them.  Failing here turns that into a red test
    instead of a plausible wrong answer live.
    """
    carried = [key for key in _REFUSED_KEYS if query.get(key)]
    if carried:
        raise PlanFailure(
            f"{REJECTION}: the query object carries {', '.join(carried)}; on the parsed path "
            "(rewriteSql absent) the statement is the whole plan, so the server has no place to "
            "put them (CONTRACT_NOTES §3.6, docs/SQLTIER.md §2)"
        )


def _parse(text: str) -> exp.Select:
    try:
        statement = sqlglot.parse_one(text, read="duckdb")
    except Exception as exc:  # sqlglot raises a family of its own parse errors
        raise PlanFailure(f"{SUBSTITUTION_ERROR}: {type(exc).__name__}: {exc}") from None
    if not isinstance(statement, exp.Select):
        raise PlanFailure(
            f"{REJECTION}: FakeOmniAPI models a single SELECT statement, got "
            f"{type(statement).__name__}"
        )
    # find() instead of an args key: the WITH arg is "with" at the sqlglot 25.0 floor and
    # "with_" at 30.x, and a miss here silently ACCEPTS the CTE this check exists to reject.
    if statement.find(exp.With):
        raise PlanFailure(
            f"{REJECTION}: the server accepts CTEs and FLATTENS them into one statement "
            "(CONTRACT_NOTES §3.6), which FakeOmniAPI does not model; emit the flat statement"
        )
    return statement


def _refuse_rewritten_shapes(statement: exp.Select) -> None:
    """Loud reject: the shapes the server silently rewrites (CONTRACT_NOTES §3.6)."""
    if statement.args.get("distinct"):
        raise PlanFailure(
            f"{REJECTION}: SELECT DISTINCT is silently STRIPPED by the server "
            "(CONTRACT_NOTES §3.6), so a pushed-down dedup comes back with duplicate rows; "
            "dedup stays local"
        )
    if statement.args.get("joins"):
        raise PlanFailure(
            f"{REJECTION}: joins come from the topic's relationships, pruned to the views the "
            "statement references; an explicit JOIN in the text is not modeled"
        )


# --------------------------------------------------------------------------------------
# ${…} extraction and binding
# --------------------------------------------------------------------------------------


def _extract_references(text: str) -> tuple[str, dict[str, _Ref]]:
    """Replace every ``${…}`` with a parseable sentinel identifier, keeping the mapping.

    Substituting before parsing is what makes the statement parseable at all — ``${users.state}``
    is an identifier in no dialect.  The sentinels are ``[a-z0-9_]``-only, so nothing quotes or
    rewrites them on the way back out.

    **Quoted text is data, never a reference.**  A filter value that happens to contain
    ``${users.state}`` is a string the client rendered as a literal, and a resolver that bound it
    would answer a question nobody asked — so the scan tokenizes ``'…'`` (``''`` doubling
    included) first and only substitutes outside it.
    """
    refs: dict[str, _Ref] = {}

    def sentinel(match: re.Match[str]) -> str:
        name = f"__omni_ref_{len(refs)}"
        refs[name] = _Ref(sentinel=name, text=match.group(1).strip())
        return name

    return "".join(
        chunk if _is_literal(chunk) else _REFERENCE.sub(sentinel, chunk)
        for chunk in _STRING_LITERAL.split(text)
    ), refs


def _is_literal(chunk: str) -> bool:
    return chunk.startswith("'") and chunk.endswith("'") and len(chunk) >= 2


def _bind(refs: dict[str, _Ref], engine: BenchEngine) -> None:
    """Resolve every reference against the model, in place; an unresolvable one is a hard error."""
    for name, ref in refs.items():
        if "." not in ref.text:
            if not _VIEW_REFERENCE.match(ref.text):
                raise PlanFailure(no_such_view(ref.text))
            continue
        if not _FIELD_REFERENCE.match(ref.text):
            raise PlanFailure(no_such_field(ref.text))
        resolved = engine.resolve(ref.text)
        if resolved is None:
            raise PlanFailure(no_such_field(ref.text))
        refs[name] = _Ref(sentinel=name, text=ref.text, field=resolved)


def _from_view(statement: exp.Select, refs: Mapping[str, _Ref], engine: BenchEngine) -> str:
    """The base view the statement selects from — a ``${topic}`` / ``${view}`` reference (§3.6)."""
    source = statement.find(exp.From)
    table = None if source is None else source.this
    if not isinstance(table, exp.Table):
        raise PlanFailure(
            f"{REJECTION}: the FROM target of an OmniSQL statement is a ${{topic}} reference "
            "(CONTRACT_NOTES §3.6); FakeOmniAPI does not model a bare warehouse table here"
        )
    if table.args.get("alias"):
        raise PlanFailure(
            f"{REJECTION}: an alias on the ${{topic}} reference is not modeled — a model ref is "
            "already fully qualified"
        )
    ref = refs.get(table.name)
    if ref is None:
        raise PlanFailure(no_such_view(table.name))
    if ref.field is not None:
        raise PlanFailure(no_such_view(ref.text))
    topic = engine.topic
    if ref.text in {topic.name, topic.base_view_name}:
        return topic.base_view_name
    if topic.view(ref.text) is not None:
        raise PlanFailure(
            f"{REJECTION}: FakeOmniAPI serves the {topic.name!r} topic's join graph rooted at "
            f"{topic.base_view_name!r}; FROM ${{{ref.text}}} would need a different root, and the "
            "topic-vs-view precedence is unverified (CONTRACT_NOTES §6 item 11)"
        )
    raise PlanFailure(no_such_view(ref.text))


# --------------------------------------------------------------------------------------
# The two naming regimes (CONTRACT_NOTES §3.6)
# --------------------------------------------------------------------------------------


def _classify(statement: exp.Select, refs: Mapping[str, _Ref]) -> tuple[_Item, ...]:
    """Split select items into bare refs and expression items — before substitution.

    After substitution a bare ``${users.state}`` is indistinguishable from a hand-written column
    expression, and the two regimes name their columns differently, so the split has to happen
    while the sentinels are still visible.
    """
    items: list[_Item] = []
    for position, item in enumerate(statement.expressions, start=1):
        inner = item.this if isinstance(item, exp.Alias) else item
        alias = item.alias if isinstance(item, exp.Alias) else ""
        ref = None
        if isinstance(inner, exp.Column) and not inner.table:
            candidate = refs.get(inner.name)
            if candidate is not None and candidate.field is None:
                raise PlanFailure(no_such_field(candidate.text))
            ref = candidate
        if ref is None and not alias:
            alias = item.output_name or f"expr_{position}"
        items.append(_Item(ref=ref, alias=alias))
    _refuse_duplicate_bare_items(items)
    return tuple(items)


def _refuse_duplicate_bare_items(items: Sequence[_Item]) -> None:
    """Loud reject: the same bare ref twice — the server DEDUPS it and shifts every position."""
    seen: set[str] = set()
    for item in items:
        if item.ref is None or item.ref.field is None:
            continue
        name = item.ref.field.name
        if name in seen:
            raise PlanFailure(
                f"{REJECTION}: {name!r} is selected twice as a bare ref, and the server "
                "DEDUPLICATES duplicate bare select items (CONTRACT_NOTES §3.6) — which silently "
                "shifts every later positional GROUP BY / ORDER BY; dedup at emission instead"
            )
        seen.add(name)


def _substitute(statement: exp.Select, refs: Mapping[str, _Ref]) -> exp.Select:
    """Replace every sentinel column with the DuckDB expression the model binds it to."""

    def bind(node: exp.Expr) -> exp.Expr:
        if not isinstance(node, exp.Column) or node.table:
            return node
        ref = refs.get(node.name)
        if ref is None:
            return node
        if ref.field is None:
            raise PlanFailure(no_such_field(ref.text))
        return sqlglot.parse_one(ref.field.expr, read="duckdb")

    bound = statement.transform(bind)
    assert isinstance(bound, exp.Select)
    return bound


def _outputs(statement: exp.Select, items: Sequence[_Item], scope: str) -> tuple[_Output, ...]:
    """Name every result column, expanding a formatted grain into its §2.7 PAIR.

    Bare refs take their canonical ``view.field`` name and drop whatever alias the SQL gave them;
    expression items keep their alias under ``scope``.  A formatted grain becomes the pair tier 1
    returns: ``X__raw`` at the item's own position, the formatted string appended last.
    """
    positional: list[_Output] = []
    appended: list[_Output] = []
    for item, rendered in zip(items, statement.expressions, strict=True):
        expression = rendered.this if isinstance(rendered, exp.Alias) else rendered
        field_def = None if item.ref is None else item.ref.field
        if field_def is None:
            positional.append(
                _Output(
                    name=f"{scope}.{item.alias}",
                    expression=expression,
                    aggregate=expression.find(exp.AggFunc) is not None,
                )
            )
        elif field_def.grain_format is None:
            positional.append(_Output(name=field_def.name, expression=expression, field=field_def))
        else:
            raw, formatted = grain_pair(field_def)
            positional.append(_Output(name=raw.name, expression=expression, field=raw))
            appended.append(
                _Output(
                    name=formatted.name,
                    expression=exp.func(
                        "strftime",
                        expression.copy(),
                        exp.Literal.string(field_def.grain_format.strftime),
                        dialect="duckdb",
                    ),
                    field=formatted,
                    appended=True,
                )
            )
    return tuple(positional + appended)


# --------------------------------------------------------------------------------------
# WHERE discipline (CONTRACT_NOTES §3.6)
# --------------------------------------------------------------------------------------


def _refuse_where_hazard(statement: exp.Select, refs: Mapping[str, _Ref]) -> None:
    """Loud reject: a bare ref and a grain ref of the SAME field inside one WHERE.

    This is the one shape where the live server merges predicates **with loss** — the tighter
    bound simply disappears and the query answers plausible, wrong rows.  Plain same-column
    conjuncts and compound ranges are safe and stay legal.
    """
    where = statement.args.get("where")
    if where is None:
        return
    variants: dict[str, set[str]] = {}
    for column in where.find_all(exp.Column):
        ref = refs.get(column.name)
        if column.table or ref is None or ref.field is None:
            continue
        grain = "" if ref.field.grain is None else ref.field.grain.name
        variants.setdefault(ref.field.base.name, set()).add(grain)
    for base_name, grains in variants.items():
        if "" in grains and len(grains) > 1:
            raise PlanFailure(
                f"{REJECTION}: WHERE references both the bare {base_name!r} and a grain variant "
                "of it, and the server MERGES those predicates with loss (CONTRACT_NOTES §3.6) — "
                "keep grain refs out of WHERE"
            )


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def _render(
    statement: exp.Select,
    outputs: Sequence[_Output],
    base_view: str,
    views: Sequence[str],
    engine: BenchEngine,
) -> str:
    """The governed DuckDB statement: the named select list, the pruned join graph, the rest."""
    statement.set(
        "expressions",
        [exp.alias_(output.expression.copy(), output.name, quoted=True) for output in outputs],
    )
    # Select's FROM arg key differs by sqlglot version ("from" at the 25.0 floor, "from_" at
    # 30.x) and set() stores an unknown key inertly — the builder method resolves it on both.
    statement.from_(exp.to_table(sql_ident(base_view), dialect="duckdb"), copy=False)
    statement.set(
        "joins",
        [
            exp.Join(
                this=exp.to_table(sql_ident(relationship.right_view_name), dialect="duckdb"),
                on=sqlglot.parse_one(relationship.on_sql, read="duckdb"),
                side="LEFT",
            )
            for relationship in engine.topic.relationships
            if relationship.right_view_name in views
        ],
    )
    _extend_group_by(statement, outputs)
    return statement.sql(dialect="duckdb", pretty=True)


def _extend_group_by(statement: exp.Select, outputs: Sequence[_Output]) -> None:
    """Group by the appended formatted-grain columns too.

    Each is a deterministic function of the ``__raw`` value already in the GROUP BY, so the group
    set does not change — but DuckDB still wants every non-aggregated select item named.
    """
    group = statement.args.get("group")
    if group is None:
        return
    trailing = [
        exp.Literal.number(position)
        for position, output in enumerate(outputs, start=1)
        if output.appended
    ]
    if trailing:
        group.set("expressions", [*group.expressions, *trailing])


# --------------------------------------------------------------------------------------
# summary.fields (CONTRACT_NOTES §2.3)
# --------------------------------------------------------------------------------------


def _summary_fields(outputs: Sequence[_Output], schema: pa.Schema) -> dict[str, Any]:
    """``summary.fields`` for an OmniSQL result — the scoped names, bound back to the model.

    A bare-ref column reports the model's own metadata (label, ``date_type``, ``aggregate_type``);
    an expression column is synthesized, a dimension unless it aggregates.  ``sql`` is blank
    throughout, exactly as on the verbatim path (docs/bench_omni_model.md §6.2): the fake does not
    model per-column SQL redaction on a SQL job.
    """
    types = {name: schema.field(name).type for name in schema.names}
    fields: dict[str, Any] = {}
    for output in outputs:
        if output.field is not None:
            payload = output.field.to_wire()
        else:
            view_name, _, field_name = output.name.partition(".")
            payload = {
                "field_name": field_name or output.name,
                "fully_qualified_name": output.name,
                "view_name": view_name,
                "data_type": "UNKNOWN",
                "is_dimension": not output.aggregate,
                "is_calc": False,
                "label": (field_name or output.name).replace("_", " ").title(),
                "format": None,
                "date_type": None,
                "aggregate_type": None,
                "filter_only_field": False,
                "hidden": False,
            }
        payload["sql"] = ""
        dtype = types.get(output.name)
        if dtype is not None:
            payload["data_type"] = arrow_data_type(dtype)
        fields[output.name] = payload
    return fields


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def _probe(engine: BenchEngine, sql: str) -> pa.Schema:
    """The statement's result schema without running it — also how ``planOnly`` is served."""
    wrapped = f"SELECT * FROM (\n{sql}\n) AS {sql_ident(SQL_WRAPPER_ALIAS)} LIMIT 0"
    return _execute(engine, wrapped).schema


def _execute(engine: BenchEngine, sql: str) -> pa.Table:
    try:
        return fetch_arrow(engine.connection.execute(sql))
    except Exception as exc:  # duckdb raises a family of its own error types
        raise PlanFailure(
            f"OmniSQL failed: {type(exc).__name__}: {exc}", error_type="QUERY"
        ) from None
