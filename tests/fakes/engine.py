"""Query planning + execution for the FakeOmniAPI, backed by DuckDB over the bench parquet files.

The fake is only useful if the numbers it returns are *real*, so a wire query object
(CONTRACT_NOTES §3) is compiled to DuckDB SQL and executed against ``tests/data/bench/*.parquet``.
The SQL produced here is what the fake reports as ``summary.display_sql``.

**Query scope.**  ``fields`` are dimensions (``view.column``), time grains
(``view.column[grain]``, §3.2) and the six governed measures of the bench model.  Selecting
dimensions *and* measures together **is** the group-by (Omni semantics): every dimension field
becomes a group key.  Measures on their own produce the single aggregate row.  ``column_totals``
(§2.7) appends a totals row carrying ``$omni_column_total_indicator``.  The join graph is exactly
the topic's: ``order_items`` LEFT JOINed to ``users`` and ``products`` (a fact-table row with an
orphan ``user_id`` therefore survives with NULL user columns, which is what
``known_answers.json`` encodes as the NULL ``users.state`` group).

**Filters split by field kind** (§3.1).  A dimension-keyed entry is pre-aggregation and lands in
``WHERE``; an entry keyed by a governed measure is translated against that measure's *aggregate*
expression and lands in ``HAVING``, post-``GROUP BY`` — the server's source-pinned behavior.  A
measure that is filtered but not selected is still aggregated for the ``HAVING`` and never
appears as a result column, and a measure filter in a query with no other aggregation context
forces the group-by (dimensions-only ``fields`` + a measure filter ⇒ ``GROUP BY`` those
dimensions + ``HAVING``).

Anything the planner refuses raises :class:`PlanFailure`, which the HTTP layer turns into a job
error line with ``error_type: "PLAN"`` — never an HTTP error, because that is what the server
does: a query the planner cannot handle still returns ``200`` with an in-band error line.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, cast

import pyarrow as pa

from tests.fakes.bench_model import (
    BENCH_DATA_DIR,
    BENCH_TOPIC,
    TABLE_NAMES,
    FakeField,
    FakeTopic,
)

__all__ = [
    "COLUMN_TOTAL_INDICATOR",
    "GRAND_TOTAL_INDICATOR",
    "GRAND_TOTAL_KEY",
    "NULL_SORT_SQL",
    "TIME_GRAINS",
    "TOTAL_INDICATOR_COLUMN",
    "BenchEngine",
    "ColumnTotals",
    "Grain",
    "PlanFailure",
    "PlannedQuery",
    "ResolvedField",
    "arrow_data_type",
    "fetch_arrow",
    "sql_ident",
]

#: ``limit`` the server substitutes when the key is absent (CONTRACT_NOTES §3).
DEFAULT_SERVER_LIMIT: Final = 1000

#: The reserved indicator column that flags totals rows (CONTRACT_NOTES §2.7).
TOTAL_INDICATOR_COLUMN: Final = "$omni_column_total_indicator"

#: ``column_totals`` key meaning "the grand total over every measure in the query" (§3).
GRAND_TOTAL_KEY: Final = "::total::"

#: Indicator values (§2.7): the grand total announces itself as ``::total::``, a totals row
#: requested per measure column as ``column_total``.
GRAND_TOTAL_INDICATOR: Final = "::total::"
COLUMN_TOTAL_INDICATOR: Final = "column_total"

_TIMESTAMP_FORMATS: Final = (
    "%Y",
    "%Y-%m",
    "%Y-%m-%d",
    "%Y-%m-%d %H",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
)

NULL_SORT_SQL: Final[Mapping[str, str]] = {
    "LAST": " NULLS LAST",
    "FIRST": " NULLS FIRST",
}

#: How one filter arm is compiled: ``(field, payload, where) -> predicate``.  A composite recurses
#: through whichever compiler its entry belongs to — the WHERE one or the HAVING one.
_ArmCompiler = Callable[["ResolvedField", Mapping[str, Any], str], "str | None"]


class PlanFailure(Exception):
    """The planner refused the query — surfaces as a job ``ERROR`` line, ``error_type: PLAN``."""

    def __init__(self, message: str, *, error_type: str = "PLAN") -> None:
        super().__init__(message)
        self.error_type = error_type


# --------------------------------------------------------------------------------------
# Time grains (CONTRACT_NOTES §3.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Grain:
    """One ``field[grain]`` suffix: how it is rendered and what type it produces.

    ``date_types`` is the set of ``FakeField.date_type`` values the grain applies to — a grain
    asked of a field it does not fit (``introduced_on[hour_of_day]``: a date has no time of day)
    is *missing*, not an error, exactly like an unknown grain (§3.2).
    """

    name: str
    kind: str  # trunc | extract | name
    unit: str  # date_trunc/EXTRACT unit, or the DuckDB function for ``name`` grains
    data_type: str
    date_types: frozenset[str]

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()

    def render(self, expr: str, *, date_type: str) -> str:
        if self.kind == "trunc":
            truncated = f"date_trunc('{self.unit}', {expr})"
            # DuckDB widens ``date_trunc`` over a DATE to TIMESTAMP; a date column keeps a
            # date-shaped result instead (docs/bench_omni_model.md §4, products.introduced_on).
            return f"CAST({truncated} AS DATE)" if date_type == "date" else truncated
        if self.kind == "extract":
            return f"EXTRACT({self.unit} FROM {expr})"
        return f"{self.unit}({expr})"


_DATE_OR_TIMESTAMP: Final = frozenset({"timestamp", "date"})
_TIMESTAMP_ONLY: Final = frozenset({"timestamp"})

#: The grains the fake executes, canonical (lowercase) name → definition.  Requested grains are
#: matched case-insensitively (§3.2); anything absent here lands in ``summary.missing_fields``.
TIME_GRAINS: Final[Mapping[str, Grain]] = {
    grain.name: grain
    for grain in (
        Grain("year", "trunc", "year", "TIMESTAMP", _DATE_OR_TIMESTAMP),
        Grain("quarter", "trunc", "quarter", "TIMESTAMP", _DATE_OR_TIMESTAMP),
        Grain("month", "trunc", "month", "TIMESTAMP", _DATE_OR_TIMESTAMP),
        Grain("week", "trunc", "week", "TIMESTAMP", _DATE_OR_TIMESTAMP),
        Grain("date", "trunc", "day", "TIMESTAMP", _DATE_OR_TIMESTAMP),
        Grain("hour", "trunc", "hour", "TIMESTAMP", _TIMESTAMP_ONLY),
        Grain("minute", "trunc", "minute", "TIMESTAMP", _TIMESTAMP_ONLY),
        Grain("second", "trunc", "second", "TIMESTAMP", _TIMESTAMP_ONLY),
        Grain("day_of_week_num", "extract", "dow", "NUMBER", _DATE_OR_TIMESTAMP),
        Grain("day_of_month", "extract", "day", "NUMBER", _DATE_OR_TIMESTAMP),
        Grain("day_of_year", "extract", "dayofyear", "NUMBER", _DATE_OR_TIMESTAMP),
        Grain("month_num", "extract", "month", "NUMBER", _DATE_OR_TIMESTAMP),
        Grain("quarter_of_year", "extract", "quarter", "NUMBER", _DATE_OR_TIMESTAMP),
        Grain("week_of_year", "extract", "week", "NUMBER", _DATE_OR_TIMESTAMP),
        Grain("hour_of_day", "extract", "hour", "NUMBER", _TIMESTAMP_ONLY),
        Grain("month_name", "name", "monthname", "STRING", _DATE_OR_TIMESTAMP),
        Grain("day_of_week_name", "name", "dayname", "STRING", _DATE_OR_TIMESTAMP),
    )
}


# --------------------------------------------------------------------------------------
# The compiled query
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedField:
    """A requested field bound to the SQL that produces it.

    ``name`` is the field name **exactly as the query asked for it**, bracket included — that is
    the key ``summary.fields`` uses and the name the result column carries (§2.3, §3.2).
    """

    name: str
    base: FakeField
    expr: str
    data_type: str
    grain: Grain | None = None

    @property
    def is_dimension(self) -> bool:
        return self.base.is_dimension

    def to_wire(self, *, redact_sql: bool = False) -> dict[str, Any]:
        """The ``summary.fields`` entry, adjusted for the grain when there is one."""
        payload = self.base.to_wire(redact_sql=redact_sql)
        if self.grain is None:
            return payload
        payload["field_name"] = self.name.split(".", 1)[1]
        payload["fully_qualified_name"] = self.name
        payload["data_type"] = self.data_type
        payload["label"] = f"{self.base.display_label} {self.grain.label}"
        # Only the truncating grains still produce a date/timestamp; the numeric and name
        # grains are plain numbers and strings, so they carry no ``date_type``.
        payload["date_type"] = self.base.date_type if self.grain.kind == "trunc" else None
        return payload


@dataclass(frozen=True)
class ColumnTotals:
    """The compiled ``column_totals`` request (§2.7/§3)."""

    sql: str
    indicator: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class PlannedQuery:
    """A compiled query: the SQL to run plus the schema bookkeeping the summary needs."""

    sql: str
    fields: tuple[ResolvedField, ...] = ()
    missing_fields: tuple[str, ...] = ()
    row_limit: int | None = None
    totals: ColumnTotals | None = None

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)


# --------------------------------------------------------------------------------------
# SQL literal helpers
# --------------------------------------------------------------------------------------


def sql_ident(name: str) -> str:
    """Quote ``name`` as a SQL identifier, doubling any embedded quote."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _text(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _column_expr(field_def: FakeField) -> str:
    if field_def.column is None:  # pragma: no cover - guarded by the caller
        raise PlanFailure(f"{field_def.name} has no backing column")
    return f"{sql_ident(field_def.view_name)}.{sql_ident(field_def.column)}"


def _number_literal(value: object, where: str) -> str:
    """Number-filter values arrive as strings (CONTRACT_NOTES §3.1); validate before inlining."""
    text = value if isinstance(value, str) else str(value)
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        raise PlanFailure(f"{where}: {text!r} is not a number") from None
    if not parsed.is_finite():
        raise PlanFailure(f"{where}: {text!r} is not a finite number")
    return text


def _date_literal(value: object, field_def: FakeField, where: str) -> str:
    """Render one side of a date filter as a DuckDB literal.

    Only the absolute half of the CONTRACT_NOTES §3.1 grammar is implemented — the truncatable
    ``"YYYY-MM-DD HH:MM:SS"`` form.  Relative literals (``"30 days ago"``, ``"last quarter"``)
    are rejected loudly instead of being silently mis-evaluated.
    """
    if not isinstance(value, str) or not value.strip():
        raise PlanFailure(f"{where}: expected a date literal, got {value!r}")
    text = value.strip().replace("T", " ").removesuffix("Z").removesuffix("+00:00").strip()
    parsed: datetime | None = None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            # Naive by design: date-filter literals are UTC by contract (CONTRACT_NOTES §3.1).
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        break
    if parsed is None:
        raise PlanFailure(
            f"{where}: {value!r} is not an absolute date literal; FakeOmniAPI supports the "
            '"YYYY-MM-DD HH:MM:SS" form (truncatable), not relative expressions'
        )
    if field_def.literal_kind == "date":
        return f"DATE '{parsed.date().isoformat()}'"
    return f"TIMESTAMPTZ '{parsed.strftime('%Y-%m-%d %H:%M:%S')}+00'"


# --------------------------------------------------------------------------------------
# Arrow -> Omni data_type
# --------------------------------------------------------------------------------------


def arrow_data_type(dtype: pa.DataType) -> str:
    """Map an Arrow type onto the ``summary.fields[*].data_type`` enum (CONTRACT_NOTES §2.3)."""
    if pa.types.is_boolean(dtype):
        return "BOOLEAN"
    if pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype):
        return "NUMBER"
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "STRING"
    if pa.types.is_timestamp(dtype) or pa.types.is_date(dtype):
        return "TIMESTAMP"
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return "ARRAY"
    if pa.types.is_duration(dtype) or pa.types.is_interval(dtype):
        return "INTERVAL"
    return "UNKNOWN"


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------


@dataclass
class _QueryContext:
    """Scratch state for one compilation: which views the query touched."""

    used_views: set[str] = field(default_factory=set)


class BenchEngine:
    """Compiles wire query objects to DuckDB SQL and runs them over the bench dataset."""

    def __init__(self, *, data_dir: Path | None = None, topic: FakeTopic = BENCH_TOPIC) -> None:
        self.data_dir = BENCH_DATA_DIR if data_dir is None else data_dir
        self.topic = topic
        self._connection: Any | None = None

    # -- connection ------------------------------------------------------------------

    @property
    def connection(self) -> Any:
        """The lazily created DuckDB connection with the bench tables registered."""
        if self._connection is None:
            import duckdb

            con = duckdb.connect()
            con.execute("SET threads TO 1")
            con.execute("SET TimeZone='UTC'")
            for name in TABLE_NAMES:
                path = str(self.data_dir / f"{name}.parquet").replace("'", "''")
                con.execute(
                    f"CREATE OR REPLACE VIEW {sql_ident(name)} AS SELECT * FROM read_parquet('{path}')"
                )
            self._connection = con
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # -- planning --------------------------------------------------------------------

    def plan(self, query: Mapping[str, Any]) -> PlannedQuery:
        """Compile a wire query object into :class:`PlannedQuery`, or raise :class:`PlanFailure`."""
        context = _QueryContext()
        base_view = self._resolve_base_view(query)
        context.used_views.add(base_view)

        selected, missing = self._resolve_fields(query.get("fields"), context)
        if not selected:
            raise PlanFailure(
                "query has no resolvable fields"
                + (f"; unknown: {', '.join(missing)}" if missing else "")
            )

        where, having = self._compile_filters(query.get("filters"), context)
        # A measure-keyed filter aggregates even when nothing else does: the server force-adds
        # the filtered measure to the aggregate, so dimensions-only ``fields`` still group (§3.1).
        aggregated = any(not f.is_dimension for f in selected) or bool(having)
        order_by = self._compile_sorts(query.get("sorts"), selected, context, aggregated=aggregated)
        totaled, indicator = self._compile_column_totals(query.get("column_totals"), selected)
        if totaled and having:
            raise PlanFailure(
                "query.column_totals alongside a measure-keyed filter (HAVING) is not modeled by "
                "FakeOmniAPI: what a total over a HAVING-restricted group set aggregates is not "
                "pinned by the contract"
            )
        limit, offset = _limit_and_offset(query)

        # FROM/JOIN/WHERE is shared verbatim with the totals query, so a totals row aggregates
        # exactly the rows the data query grouped (post-filter, pre-limit).
        body = [f"FROM {sql_ident(base_view)}", *self._join_clauses(base_view, context)]
        if where:
            body.append(f"WHERE {where}")

        select_list = ",\n       ".join(f"{f.expr} AS {sql_ident(f.name)}" for f in selected)
        lines = [f"SELECT {select_list}", *body]
        # Dimensions + measures IS the group-by (Omni semantics, DESIGN.md §2).
        group_by = ", ".join(
            str(position) for position, f in enumerate(selected, start=1) if f.is_dimension
        )
        if aggregated and group_by:
            lines.append(f"GROUP BY {group_by}")
        # Post-aggregation, exactly where the server puts a measure filter (§3.1).
        if having:
            lines.append(f"HAVING {having}")
        if order_by:
            lines.append(f"ORDER BY {order_by}")
        if limit is not None:
            lines.append(f"LIMIT {limit}")
        if offset:
            lines.append(f"OFFSET {offset}")

        totals: ColumnTotals | None = None
        if totaled:
            totals_select = ",\n       ".join(f"{f.expr} AS {sql_ident(f.name)}" for f in totaled)
            totals = ColumnTotals(
                sql="\n".join([f"SELECT {totals_select}", *body]),
                indicator=indicator,
                fields=tuple(f.name for f in totaled),
            )

        return PlannedQuery(
            sql="\n".join(lines),
            fields=tuple(selected),
            missing_fields=tuple(missing),
            row_limit=limit,
            totals=totals,
        )

    def _resolve_base_view(self, query: Mapping[str, Any]) -> str:
        topic_name = query.get("join_paths_from_topic_name")
        if topic_name is not None and topic_name != self.topic.name:
            raise PlanFailure(f"Topic {topic_name} not found")
        table = query.get("table") or ""
        if table and table != self.topic.base_view_name:
            raise PlanFailure(
                f"table {table!r} is not queryable; FakeOmniAPI serves the "
                f"{self.topic.name!r} topic (base view {self.topic.base_view_name!r})"
            )
        if not table and topic_name is None:
            raise PlanFailure("query needs a table (base view) or join_paths_from_topic_name")
        return self.topic.base_view_name

    def _resolve_fields(
        self, raw: object, context: _QueryContext
    ) -> tuple[list[ResolvedField], list[str]]:
        """Split requested fields into executable ones and ``summary.missing_fields`` entries.

        Unknown names — and unknown or inapplicable ``[grain]`` suffixes — are *dropped*, not
        rejected: the server treats them as missing fields and still reports success
        (CONTRACT_NOTES §2.3/§3.2).
        """
        selected: list[ResolvedField] = []
        missing: list[str] = []
        seen: set[str] = set()
        for entry in _as_sequence(raw, "query.fields"):
            name = str(entry)
            if name in seen:
                continue
            seen.add(name)
            resolved = self._resolve(name)
            if resolved is None:
                missing.append(name)
                continue
            context.used_views.add(resolved.base.view_name)
            selected.append(resolved)
        return selected, missing

    def resolve(self, name: str) -> ResolvedField | None:
        """Bind a wire field name to the model, or ``None`` when the model has no such field.

        Public because the raw-SQL planner (:mod:`tests.fakes.sqljobs`) has to know whether a
        ``filters`` key names a governed *measure* — those are silently skipped on a SQL job
        (CONTRACT_NOTES §3.1) — without duplicating the model lookup.
        """
        return self._resolve(name)

    def _resolve(self, name: str) -> ResolvedField | None:
        """Bind a wire field name (with an optional ``[grain]``) to its SQL, or ``None``."""
        base_name, grain_name = _split_grain(name)
        base = self.topic.field(base_name)
        if base is None:
            return None
        if grain_name is None:
            if base.is_dimension:
                if base.column is None:  # pragma: no cover - every dimension has a column
                    return None
                return ResolvedField(name, base, _column_expr(base), base.data_type)
            if base.duckdb_sql is None:  # pragma: no cover - every measure has an expression
                return None
            return ResolvedField(name, base, base.duckdb_sql, base.data_type)

        grain = TIME_GRAINS.get(grain_name.lower())
        if grain is None or not base.is_dimension or base.column is None:
            return None
        if base.date_type not in grain.date_types:
            return None
        expression = grain.render(_column_expr(base), date_type=base.date_type)
        return ResolvedField(name, base, expression, grain.data_type, grain=grain)

    def _reference(self, name: str, where: str, context: _QueryContext) -> ResolvedField:
        """Resolve a field named by ``filters``/``sorts``; unknown names are hard failures."""
        resolved = self._resolve(name)
        if resolved is None:
            raise PlanFailure(f"{where}: unknown field {name!r}")
        context.used_views.add(resolved.base.view_name)
        return resolved

    def _join_clauses(self, base_view: str, context: _QueryContext) -> list[str]:
        clauses: list[str] = []
        for relationship in self.topic.relationships:
            if relationship.right_view_name not in context.used_views:
                continue
            if relationship.left_view_name != base_view:  # pragma: no cover - defensive
                raise PlanFailure(
                    f"no join path from {base_view} to {relationship.right_view_name}"
                )
            clauses.append(
                f"LEFT JOIN {sql_ident(relationship.right_view_name)} ON {relationship.on_sql}"
            )
        unjoined = (
            context.used_views - {base_view} - {r.right_view_name for r in self.topic.relationships}
        )
        if unjoined:  # pragma: no cover - defensive
            raise PlanFailure(f"no join path from {base_view} to {', '.join(sorted(unjoined))}")
        return clauses

    # -- filters ---------------------------------------------------------------------

    def _compile_filters(self, raw: object, context: _QueryContext) -> tuple[str, str]:
        """Split ``query.filters`` into its WHERE half and its HAVING half (§3.1).

        Entries are implicitly ANDed across fields.  Which side an entry lands on is decided by
        the *model field type* of its key, not by the filter arm: a dimension filters rows before
        aggregation, a governed measure filters groups after it.  Exactly one entry per measure is
        possible — the wire keys ``filters`` by field name — so several conditions on one measure
        arrive as a ``composite`` inside that single entry.
        """
        if raw is None:
            return "", ""
        if not isinstance(raw, Mapping):
            raise PlanFailure("query.filters must be an object keyed by field name")
        predicates: list[str] = []
        having: list[str] = []
        for name, payload in raw.items():
            where = f"query.filters[{str(name)!r}]"
            field_def = self._reference(str(name), where, context)
            if not isinstance(payload, Mapping):
                raise PlanFailure(f"{where} must be an object")
            if field_def.is_dimension:
                predicate = self._compile_filter(field_def, payload, where)
                target = predicates
            else:
                predicate = self._compile_measure_filter(field_def, payload, where)
                target = having
            if predicate is not None:
                target.append(predicate)
        return "\n  AND ".join(predicates), "\n  AND ".join(having)

    def _compile_filter(
        self, field_def: ResolvedField, payload: Mapping[str, Any], where: str
    ) -> str | None:
        """Compile one filter arm against ``field_def``'s expression.

        The expression is the **grain-filter rule** in action (CONTRACT_NOTES §3.1): a filter
        keyed on the bare name lands on the underlying column even when the projection is
        grained, while one keyed on a bracketed numeric grain lands on that grain's extract.
        """
        filter_type = payload.get("type")
        expr = field_def.expr
        is_negative = payload.get("is_negative")

        if filter_type == "string":
            return _negate(self._string_filter(expr, payload, where), is_negative)
        if filter_type == "number":
            return _negate(self._number_filter(expr, payload, where), is_negative)
        if filter_type == "date":
            return _negate(self._date_filter(expr, field_def.base, payload, where), is_negative)
        if filter_type == "boolean":
            return _boolean_filter(expr, payload)
        if filter_type == "null":
            return f"{expr} IS NOT NULL" if is_negative else f"{expr} IS NULL"
        if filter_type == "composite":
            return self._composite_filter(field_def, payload, where, self._compile_filter)
        raise PlanFailure(
            f"{where}: filter type {filter_type!r} is not supported by FakeOmniAPI "
            "(string, number, date, boolean, null, composite are)"
        )

    def _compile_measure_filter(
        self, field_def: ResolvedField, payload: Mapping[str, Any], where: str
    ) -> str | None:
        """Compile one filter arm against a governed measure's **aggregate** expression (§3.1).

        This is the HAVING side.  Only the number kinds and composites of them are implemented;
        every other arm is refused by name rather than translated into a predicate whose live
        meaning FakeOmniAPI cannot vouch for.
        """
        filter_type = payload.get("type")
        if filter_type == "number":
            return _negate(
                self._number_filter(field_def.expr, payload, where), payload.get("is_negative")
            )
        if filter_type == "composite":
            return self._composite_filter(field_def, payload, where, self._compile_measure_filter)
        raise PlanFailure(
            f"{where}: filter type {filter_type!r} on the governed measure {field_def.name!r} is "
            "not supported by FakeOmniAPI; a measure filter compiles to HAVING for the number "
            "kinds (LESS_THAN, GREATER_THAN, EQUALS, BETWEEN) and composite AND/OR of those"
        )

    def _composite_filter(
        self,
        field_def: ResolvedField,
        payload: Mapping[str, Any],
        where: str,
        compile_child: _ArmCompiler,
    ) -> str | None:
        conjunction = payload.get("conjunction")
        if conjunction not in {"AND", "OR"}:
            raise PlanFailure(f"{where}: composite conjunction must be AND or OR")
        children = payload.get("filters")
        if not isinstance(children, Sequence) or isinstance(children, str) or not children:
            raise PlanFailure(f"{where}: composite filter needs a non-empty filters array")
        parts: list[str] = []
        for index, child in enumerate(children):
            if not isinstance(child, Mapping):
                raise PlanFailure(f"{where}.filters[{index}] must be an object")
            compiled = compile_child(field_def, child, f"{where}.filters[{index}]")
            if compiled is not None:
                parts.append(compiled)
        if not parts:
            return None
        joined = f" {conjunction} ".join(parts)
        return _negate(f"({joined})", payload.get("is_negative"))

    def _string_filter(self, expr: str, payload: Mapping[str, Any], where: str) -> str:
        kind = payload.get("kind")
        values = [str(v) for v in _as_sequence(payload.get("values"), f"{where}.values")]
        insensitive = bool(payload.get("case_insensitive"))
        target = f"lower({expr})" if insensitive else expr

        if kind == "IS_EMPTY":
            return f"({expr} IS NULL OR {expr} = '')"
        if not values:
            raise PlanFailure(f"{where}: string filter {kind} needs at least one value")

        def literal(value: str) -> str:
            return _text(value.lower() if insensitive else value)

        if kind == "EQUALS":
            if len(values) == 1:
                return f"{target} = {literal(values[0])}"
            joined = ", ".join(literal(v) for v in values)
            return f"{target} IN ({joined})"
        function = {"CONTAINS": "contains", "STARTS_WITH": "starts_with", "ENDS_WITH": "ends_with"}
        if kind in function:
            parts = [f"{function[str(kind)]}({target}, {literal(v)})" for v in values]
            return _any_of(parts)
        if kind == "SQL_LIKE":
            operator = "ILIKE" if insensitive else "LIKE"
            return _any_of([f"{expr} {operator} {_text(v)}" for v in values])
        raise PlanFailure(f"{where}: unsupported string filter kind {kind!r}")

    def _number_filter(self, expr: str, payload: Mapping[str, Any], where: str) -> str:
        kind = payload.get("kind")
        raw_values = _as_sequence(payload.get("values"), f"{where}.values")
        values = [_number_literal(v, where) for v in raw_values]
        inclusive = bool(payload.get("is_inclusive"))

        if kind == "BETWEEN":
            if len(values) != 2:
                raise PlanFailure(f"{where}: number BETWEEN needs exactly two values")
            lower, upper = values
            return f"({expr} >= {lower} AND {expr} < {upper})"
        if not values:
            raise PlanFailure(f"{where}: number filter {kind} needs at least one value")
        if kind == "EQUALS":
            if len(values) == 1:
                return f"{expr} = {values[0]}"
            return f"{expr} IN ({', '.join(values)})"
        if kind == "LESS_THAN":
            operator = "<=" if inclusive else "<"
        elif kind == "GREATER_THAN":
            operator = ">=" if inclusive else ">"
        else:
            raise PlanFailure(f"{where}: unsupported number filter kind {kind!r}")
        return _any_of([f"{expr} {operator} {value}" for value in values])

    def _date_filter(
        self, expr: str, field_def: FakeField, payload: Mapping[str, Any], where: str
    ) -> str:
        kind = payload.get("kind")
        left = payload.get("left_side")
        right = payload.get("right_side")
        if kind == "BETWEEN":
            lower = _date_literal(left, field_def, f"{where}.left_side")
            upper = _date_literal(right, field_def, f"{where}.right_side")
            return f"({expr} >= {lower} AND {expr} < {upper})"
        if kind == "ON_OR_AFTER":
            return f"{expr} >= {_date_literal(left, field_def, f'{where}.left_side')}"
        if kind == "BEFORE":
            return f"{expr} < {_date_literal(right, field_def, f'{where}.right_side')}"
        raise PlanFailure(
            f"{where}: date filter kind {kind!r} is not supported by FakeOmniAPI "
            "(BETWEEN, ON_OR_AFTER, BEFORE are)"
        )

    # -- sorts -----------------------------------------------------------------------

    def _compile_sorts(
        self,
        raw: object,
        selected: Sequence[ResolvedField],
        context: _QueryContext,
        *,
        aggregated: bool,
    ) -> str:
        """``sorts[].column_name`` is the EXACT field name, bracket included (§3).

        A selected field sorts by its output alias; measures are sortable exactly like
        dimensions.  In an aggregate query a sort key that is not one of ``fields`` has no
        defined value, so it is refused rather than guessed.
        """
        if raw is None:
            return ""
        projected = {f.name: f for f in selected}
        terms: list[str] = []
        for index, entry in enumerate(_as_sequence(raw, "query.sorts")):
            where = f"query.sorts[{index}]"
            if not isinstance(entry, Mapping):
                raise PlanFailure(f"{where} must be an object")
            column_name = entry.get("column_name")
            if not isinstance(column_name, str) or not column_name:
                raise PlanFailure(f"{where}: column_name is required")
            if column_name in projected:
                expr = sql_ident(column_name)
            else:
                field_def = self._reference(column_name, where, context)
                if aggregated or not field_def.is_dimension:
                    raise PlanFailure(
                        f"{where}: {column_name!r} must also appear in query.fields to be "
                        "sortable in an aggregated query"
                    )
                expr = field_def.expr
            direction = "DESC" if entry.get("sort_descending") else "ASC"
            nulls = NULL_SORT_SQL.get(str(entry.get("null_sort") or ""), "")
            terms.append(f"{expr} {direction}{nulls}")
        return ", ".join(terms)

    # -- column totals ---------------------------------------------------------------

    def _compile_column_totals(
        self, raw: object, selected: Sequence[ResolvedField]
    ) -> tuple[tuple[ResolvedField, ...], str]:
        """Resolve ``column_totals`` into the measures to total and the indicator value.

        ``{"::total::": {...}}`` totals every measure in the query; a measure-keyed entry totals
        just that column.  Dimension columns on the totals row are NULL either way (§2.7).
        """
        if raw is None:
            return (), ""
        if not isinstance(raw, Mapping):
            raise PlanFailure(
                "query.column_totals must be an object keyed by field name or '::total::'"
            )
        if not raw:
            return (), ""

        measures = {f.name: f for f in selected if not f.is_dimension}
        grand = False
        keyed: list[ResolvedField] = []
        for key, payload in raw.items():
            name = str(key)
            where = f"query.column_totals[{name!r}]"
            if not isinstance(payload, Mapping) or payload.get("type") != "aggregation":
                raise PlanFailure(f'{where}: expected {{"type": "aggregation"}}')
            if name == GRAND_TOTAL_KEY:
                grand = True
                continue
            measure = measures.get(name)
            if measure is None:
                raise PlanFailure(
                    f"{where}: column totals aggregate a measure listed in query.fields; "
                    f"{name!r} is not one"
                )
            if measure not in keyed:
                keyed.append(measure)

        totaled = tuple(measures.values()) if grand else tuple(keyed)
        if not totaled:
            raise PlanFailure(
                "query.column_totals needs at least one measure in query.fields to aggregate"
            )
        return totaled, GRAND_TOTAL_INDICATOR if grand else COLUMN_TOTAL_INDICATOR

    # -- execution -------------------------------------------------------------------

    def execute(self, planned: PlannedQuery) -> pa.Table:
        """Run the compiled SQL and return the Arrow table the ``result`` payload carries."""
        table = fetch_arrow(self.connection.execute(planned.sql))
        if planned.totals is None:
            return table
        totals = fetch_arrow(self.connection.execute(planned.totals.sql))
        return _append_totals(table, totals, planned.totals)

    def schema(self, planned: PlannedQuery) -> pa.Schema:
        """Result schema without executing the query — how ``planOnly`` learns its types."""
        sql = f"SELECT * FROM (\n{planned.sql}\n) AS omni_plan LIMIT 0"
        schema = fetch_arrow(self.connection.execute(sql)).schema
        if planned.totals is None:
            return schema
        return schema.append(pa.field(TOTAL_INDICATOR_COLUMN, pa.string()))


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _split_grain(name: str) -> tuple[str, str | None]:
    """``"order_items.created_at[month]"`` -> ``("order_items.created_at", "month")``."""
    if not name.endswith("]"):
        return name, None
    head, bracket, grain = name[:-1].partition("[")
    if not bracket:
        return name, None
    return head, grain


def _append_totals(data: pa.Table, totals: pa.Table, spec: ColumnTotals) -> pa.Table:
    """Append the totals row and the ``$omni_column_total_indicator`` column (§2.7).

    The totals row carries the aggregated measures that ``column_totals`` asked for and NULL for
    every other column — dimensions included.  The indicator is NULL on the data rows, which is
    exactly how a client tells them apart (there is no ``row_type`` column on ``/query/run``).
    """
    columns: list[Any] = []
    for output in data.schema:
        if output.name in spec.fields:
            value = totals.column(output.name).combine_chunks()
            columns.append(value if value.type == output.type else value.cast(output.type))
        else:
            columns.append(pa.nulls(totals.num_rows, type=output.type))
    combined = pa.concat_tables([data, pa.table(columns, schema=data.schema)])
    indicator = pa.array(
        [None] * data.num_rows + [spec.indicator] * totals.num_rows, type=pa.string()
    )
    return combined.append_column(TOTAL_INDICATOR_COLUMN, indicator)


def fetch_arrow(result: Any) -> pa.Table:
    """Materialize a DuckDB result as Arrow across the versions ``pyproject`` allows.

    ``fetch_arrow_table`` is deprecated in favor of ``to_arrow_table`` in recent DuckDB releases,
    but the floor is ``duckdb>=1.1`` where only the former exists.
    """
    fetch = getattr(result, "to_arrow_table", None)
    if fetch is None:  # pragma: no cover - depends on the installed duckdb
        fetch = result.fetch_arrow_table
    return cast("pa.Table", fetch())


def _as_sequence(raw: object, where: str) -> Sequence[Any]:
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise PlanFailure(f"{where} must be an array")
    return raw


def _any_of(parts: Sequence[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return "(" + " OR ".join(parts) + ")"


def _negate(predicate: str, is_negative: object) -> str:
    return f"NOT ({predicate})" if is_negative else predicate


def _boolean_filter(expr: str, payload: Mapping[str, Any]) -> str | None:
    """``is_negative`` is the whole semantic: False = is true, True = is false, absent = no-op."""
    is_negative = payload.get("is_negative")
    if is_negative is None:
        return None
    target = f"COALESCE({expr}, FALSE)" if payload.get("treat_nulls_as_false") else expr
    return f"{target} = FALSE" if is_negative else f"{target} = TRUE"


def _limit_and_offset(query: Mapping[str, Any]) -> tuple[int | None, int]:
    """``limit`` is a trichotomy: absent -> server default, ``null`` -> unlimited, int -> itself."""
    limit: int | None
    if "limit" not in query:
        limit = DEFAULT_SERVER_LIMIT
    else:
        raw_limit = query["limit"]
        limit = None if raw_limit is None else int(raw_limit)
    raw_offset = query.get("offset") or 0
    return limit, int(raw_offset)
