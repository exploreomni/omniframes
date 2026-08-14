"""``explain()`` rendering (docs/INTERNALS.md §4).

The output names every remote step with its tier and payload, and every local operator::

    == Physical plan ==
    Remote [tier 1 · semantic → POST /api/v1/query/run]
      topic: order_items   model: bench_ecommerce
      fields: [users.state, order_items.total_sale_price]
      group by: [users.state]
      measures: [order_items.total_sale_price]
      filters: users.state = 'California' AND NOT order_items.returned
      having: order_items.total_sale_price >= 50000
      totals: column_totals [::total::]
      sort: users.state ASC   limit: 50000   version: 9
    Local [pandas]
      (none — fully pushed down)

``filters`` and ``having`` are the same ``query.filters`` map on the wire — the server routes
each entry by the model field type of its key (CONTRACT_NOTES §3.1) — but reading a plan means
knowing which rows a predicate removes and which groups, so they are rendered apart.

There is deliberately no "silent local fallback" line to hide behind: if something ran locally,
``explain()`` says so (docs/DESIGN.md §2).

A plan the splitter had to cut renders as a DAG instead (docs/HYBRID.md §7): every remote query
is numbered in execution order, and every local operator names the inputs it reads::

    == Physical plan ==
    Remote step 1 [tier 1 · semantic → POST /api/v1/query/run]
      ...
    Remote step 2 [tier 2 · sql → POST /api/v1/query/run]
      topic: order_items   model: bench_ecommerce
      sql:
        SELECT ${users.state}, COUNT(DISTINCT ${users.id}) AS of_expr_1
        ...
        … (+3 more lines)
    Local [arrow compute]
      align-join: step 1 ⨝ step 2 on [users.state]
      sort: buyers desc
      project: [users.state, revenue, buyers]

A tier-2 step is rendered as the OmniSQL statement omniframes wrote (docs/SQLTIER.md §4), elided
after eight lines because a DAG of them still has to be readable.  The ``${…}`` refs make it
self-documenting: the joins, the measure definitions and the access grants all come from the
model the statement is parsed against.  Single-remote tier-1 plans keep the M1/M2 rendering byte
for byte — a plan that did not change must not read as if it had.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from omniframes.compile.querymodel import (
    BooleanFilter,
    CompositeFilter,
    DateFilter,
    DateFilterKind,
    Filter,
    NullFilter,
    NumberFilter,
    NumberFilterKind,
    Query,
    StringFilter,
    StringFilterKind,
)
from omniframes.compile.semantic import ExecutionPlan, LocalStep, RemoteStep, Step
from omniframes.compile.sqlgen import sql_lines

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omniframes.plan.nodes import PlanNode

__all__ = ["describe_filter", "describe_filters", "explain_text"]

_REDACTED_SQL = "(SQL redacted — the key's user lacks the VIEW_SQL permission)"


def explain_text(
    plan: PlanNode,
    execution: ExecutionPlan,
    *,
    analyze: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
) -> str:
    """Render the physical plan.

    Args:
        plan: the logical plan (unused: everything ``explain()`` shows is read off the compiled
            execution plan, so the two can never disagree).
        execution: the compiled execution plan.
        analyze: the ``summary`` of a ``planOnly`` round trip per remote step, when
            ``explain(analyze=True)`` asked for the server's own plan — one mapping, or one per
            step in ``execution.steps``.  ``display_sql`` and ``omni_sql`` are blanked by the
            server for callers without ``VIEW_SQL``, which is rendered as such.
    """
    del plan
    multi = execution.root is not None
    lines = ["== Physical plan =="]
    for index, step in enumerate(execution.steps, start=1):
        lines.extend(_remote_step(step, index if multi else None))
    if execution.root is None:
        lines.append("Local [pandas]")
        lines.append("  (none — fully pushed down)")
    else:
        lines.append("Local [arrow compute]")
        lines.extend(_local_lines(execution, execution.root))
    for summary in _summaries(analyze):
        lines.extend(_analyze_section(summary))
    return "\n".join(lines)


def _summaries(
    analyze: Mapping[str, object] | Sequence[Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], ...]:
    if analyze is None:
        return ()
    if isinstance(analyze, Mapping):
        return (analyze,)
    return tuple(analyze)


def _local_lines(execution: ExecutionPlan, root: Step) -> list[str]:
    """One line per local operator, deepest first, each naming what it reads."""
    numbers = {id(step): index for index, step in enumerate(execution.steps, start=1)}
    ordered: list[LocalStep] = []
    seen: set[int] = set()

    def visit(step: Step) -> None:
        if isinstance(step, RemoteStep) or id(step) in seen:
            return
        seen.add(id(step))
        for source in step.inputs:
            visit(source)
        ordered.append(step)

    visit(root)

    def name(step: Step) -> str:
        if isinstance(step, RemoteStep):
            return f"step {numbers.get(id(step), 0)}"
        return step.op.kind

    return [f"  {step.op.describe([name(source) for source in step.inputs])}" for step in ordered]


def _heading(step: RemoteStep, number: int | None) -> str:
    where = "Remote" if number is None else f"Remote step {number}"
    role = step.label if not step.note else f"{step.label} {step.note}"
    return f"{where} [tier {step.tier} · {role} → POST /api/v1/query/run]"


def _opaque_step(step: RemoteStep, number: int | None) -> list[str]:
    """A payload omniframes did not write: raw SQL, or a stored query sent verbatim.

    Rendered off the **envelope**, not off a typed query, because for these steps the bytes are
    the truth — including ``rewriteSql: false``, whose absence would make the server parse the
    SQL as OmniSQL rather than run it verbatim (CONTRACT_NOTES §3.5), and which therefore
    belongs where a reader can see it.
    """
    query = step.envelope.get("query")
    payload: Mapping[str, object] = query if isinstance(query, Mapping) else {}
    lines = [_heading(step, number), f"  {step.source_label}"]

    sql = payload.get("userEditedSQL")
    if isinstance(sql, str) and sql.strip():
        lines.append("  userEditedSQL:")
        lines.extend(f"    {line}" for line in sql.strip().splitlines())
        lines.append(
            f"  rewriteSql: {_flag(payload.get('rewriteSql'))}   "
            f"sqlSortsEnabled: {_flag(payload.get('sqlSortsEnabled'))}"
        )
    else:
        lines.append(f"  fields: [{', '.join(_strings(payload.get('fields')))}]")
        filters = payload.get("filters")
        if isinstance(filters, Mapping) and filters:
            keys = ", ".join(sorted(str(key) for key in filters))
            lines.append(f"  filters: [{keys}] (stored — sent verbatim, not recompiled)")
        sorts = _stored_sorts(payload.get("sorts"))
        if sorts:
            lines.append(f"  sort: {sorts}")
    lines.append(f"  limit: {_stored_limit(step, payload)}   version: {payload.get('version')}")
    return lines


def _stored_limit(step: RemoteStep, payload: Mapping[str, object]) -> str:
    """The limit line of a verbatim payload, telling an absent key from an explicit ``null``.

    A blob omniframes was handed may carry no ``limit`` at all, and the server then applies its
    own 1000-row default (CONTRACT_NOTES §3).  Rendering that as ``unlimited (null)`` — the same
    text a genuine ``limit: null`` gets — is how a silently truncated query reads as complete.
    """
    if "limit" not in payload:
        return f"{step.applied_limit} (server default — the stored query carries no limit)"
    return _limit_text(step.applied_limit)


def _flag(value: object) -> str:
    return "true" if value is True else "false" if value is False else str(value)


def _strings(value: object) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [str(entry) for entry in value]
    return []


def _stored_sorts(value: object) -> str:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ""
    rendered: list[str] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            continue
        direction = "DESC" if entry.get("sort_descending") else "ASC"
        rendered.append(f"{entry.get('column_name')} {direction}")
    return ", ".join(rendered)


def _sql_step(step: RemoteStep, number: int | None) -> list[str]:
    """A tier-2 job: the one OmniSQL statement omniframes wrote (docs/SQLTIER.md §4).

    The SQL is elided after :data:`~omniframes.compile.sqlgen.SQL_EXPLAIN_LINES` lines — long
    enough to see the shape of the computation, short enough that a DAG of them still reads.
    There is nothing else to show: the statement *is* the plan, and its ``${…}`` refs say which
    model it binds against.
    """
    lines = [_heading(step, number), f"  {step.source_label}", "  sql:"]
    lines.extend(f"    {line}" for line in sql_lines(step.query.user_edited_sql))
    return lines


def _remote_step(step: RemoteStep, number: int | None) -> list[str]:
    if step.compilation.is_sql:
        return _sql_step(step, number)
    if step.compilation.opaque:
        return _opaque_step(step, number)
    query = step.query
    lines = [_heading(step, number)]
    lines.append(f"  {step.source_label}")
    lines.append(f"  fields: [{', '.join(query.fields)}]")
    lines.extend(_aggregate_lines(step))
    having_keys = frozenset(step.measure_filters)
    where = {name: flt for name, flt in query.filters.items() if name not in having_keys}
    having = {name: flt for name, flt in query.filters.items() if name in having_keys}
    filters = describe_filters(where)
    if filters:
        lines.append(f"  filters: {filters}")
    if having:
        lines.append(f"  having: {describe_filters(having)}")
    if query.column_totals:
        lines.append(f"  totals: column_totals [{', '.join(query.column_totals)}]")
    tail = [f"sort: {_describe_sorts(step.sorts) or '(none)'}", f"limit: {_limit(query)}"]
    if query.offset:
        tail.append(f"offset: {query.offset}")
    tail.append(f"version: {query.version}")
    lines.append("  " + "   ".join(tail))
    if step.aliases:
        renames = ", ".join(f"{wire} -> {alias}" for wire, alias in step.aliases.items())
        lines.append(f"  aliases: {renames}")
    return lines


def _aggregate_lines(step: RemoteStep) -> list[str]:
    """The group-by, when there is one. Selecting a measure IS the group-by (DESIGN §2)."""
    if not step.measures and not step.measure_filters:
        return []
    keys = ", ".join(step.group_keys)
    lines = [f"  group by: [{keys}]" if keys else "  group by: (none — one aggregate row)"]
    if step.measures:
        lines.append(f"  measures: [{', '.join(step.measures)}]")
    return lines


def _analyze_section(summary: Mapping[str, object]) -> list[str]:
    display_sql = summary.get("display_sql")
    text = str(display_sql).strip() if display_sql else ""
    lines = ["Analyzed [planOnly round trip]"]
    if not text:
        lines.append(f"  {_REDACTED_SQL}")
        return lines
    lines.extend(f"  {line}" for line in text.splitlines())
    return lines


def _limit(query: Query) -> str:
    return _limit_text(query.effective_limit)


def _limit_text(limit: int | None) -> str:
    return "unlimited (null)" if limit is None else str(limit)


def _describe_sorts(sorts: Sequence[tuple[str, bool]]) -> str:
    return ", ".join(f"{name} {'DESC' if descending else 'ASC'}" for name, descending in sorts)


def describe_filters(filters: Mapping[str, Filter]) -> str:
    """Render ``query.filters`` as one readable conjunction (they AND across fields)."""
    return " AND ".join(describe_filter(name, flt) for name, flt in filters.items())


def describe_filter(name: str, flt: Filter) -> str:
    """Render one filter entry the way a human would read it."""
    text = _describe(name, flt)
    if flt.is_negative and not isinstance(flt, BooleanFilter | NullFilter):
        return f"NOT ({text})"
    return text


def _describe(name: str, flt: Filter) -> str:
    if isinstance(flt, CompositeFilter):
        joined = f" {flt.conjunction.value} ".join(
            _describe(name, child) if not child.is_negative else f"NOT ({_describe(name, child)})"
            for child in flt.filters
        )
        return f"({joined})"
    if isinstance(flt, StringFilter):
        return _describe_string(name, flt)
    if isinstance(flt, NumberFilter):
        return _describe_number(name, flt)
    if isinstance(flt, DateFilter):
        if flt.kind is DateFilterKind.BETWEEN and flt.left_side and flt.right_side:
            # Spelled out because the upper bound is EXCLUSIVE, which "BETWEEN" hides.
            return f"{flt.left_side!r} <= {name} < {flt.right_side!r}"
        sides = " ".join(repr(side) for side in (flt.left_side, flt.right_side) if side is not None)
        return f"{name} {flt.kind.value} {sides}".strip()
    if isinstance(flt, BooleanFilter):
        if flt.is_negative is None:
            return f"{name} (boolean placeholder — no-op)"
        return f"NOT {name}" if flt.is_negative else name
    if isinstance(flt, NullFilter):
        return f"{name} IS NOT NULL" if flt.is_negative else f"{name} IS NULL"
    return f"{name} {type(flt).__name__}"


def _describe_string(name: str, flt: StringFilter) -> str:
    values = ", ".join(repr(value) for value in flt.values)
    suffix = " (case-insensitive)" if flt.case_insensitive else ""
    if flt.kind is StringFilterKind.EQUALS:
        body = f"{name} = {values}" if len(flt.values) == 1 else f"{name} IN ({values})"
        return f"{body}{suffix}"
    if flt.kind is StringFilterKind.IS_EMPTY:
        return f"{name} IS EMPTY"
    return f"{name} {flt.kind.value} {values}{suffix}"


_NUMBER_OPERATORS: dict[tuple[NumberFilterKind, bool], str] = {
    (NumberFilterKind.LESS_THAN, False): "<",
    (NumberFilterKind.LESS_THAN, True): "<=",
    (NumberFilterKind.GREATER_THAN, False): ">",
    (NumberFilterKind.GREATER_THAN, True): ">=",
}


def _describe_number(name: str, flt: NumberFilter) -> str:
    values = [str(value) for value in flt.values]
    if flt.kind is NumberFilterKind.BETWEEN:
        low, high = values
        return f"{low} <= {name} < {high}"
    if flt.kind is NumberFilterKind.EQUALS:
        joined = ", ".join(values)
        return f"{name} = {joined}" if len(values) == 1 else f"{name} IN ({joined})"
    operator = _NUMBER_OPERATORS[(flt.kind, bool(flt.is_inclusive))]
    return " OR ".join(f"{name} {operator} {value}" for value in values)
