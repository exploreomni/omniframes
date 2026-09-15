"""Raw-SQL jobs for the FakeOmniAPI — CONTRACT_NOTES §3.4, §3.5 and the §2.7 sidecars.

A job whose ``query`` carries ``userEditedSQL`` **plus** one of the "do not rewrite" markers runs
the caller's SQL verbatim instead of compiling the semantic query object.  Everything else about
the exchange is unchanged: the same NDJSON framing, the same base64 Arrow ``result``, the same
``summary`` keys.

**The fake's warehouse schema.**  The SQL runs against DuckDB with the three bench tables
registered under their **bare** names — ``users``, ``products`` and ``order_items`` — with no
schema qualifier, so ``SELECT ... FROM order_items oi LEFT JOIN users u ON u.id = oi.user_id``
is the shape a tier-2 query takes offline.  A live org puts them behind whatever schema the
connection uses (``OMNIFRAMES_BENCH`` in internal-docs/bench_omni_model.md §1), so a SQL string that must
run in both lanes has to be schema-qualified by the caller, not by the fake.

Three server behaviors are reproduced literally because they are the ones that bite clients:

* **``userEditedSQL`` alone is a different job kind.**  Without ``rewriteSql: false`` (or
  ``parsed: false`` / ``dbtMode: true``) the text is not run verbatim and not ignored either: it
  is parsed as **OmniSQL** and planned as a governed model job (CONTRACT_NOTES §3.6), which is
  :mod:`tests.fakes.omnisql`.  The marker is the whole selector between the two paths.
* **``sqlSortsEnabled``** gates ``sorts`` and ``column_totals``: truthy applies them on top of
  the SQL result, falsy strips them **silently** (the server forces them to ``[]``).
* **Measure-keyed ``filters`` entries are silently SKIPPED** — the view Omni wraps around the
  SQL has an empty measures map (§3.1, ``OmniJobPlanner.kt``).  Dimension-keyed entries would
  need Omni's mustache templating, which the fake refuses loudly instead of approximating.

``staticQueryReferences`` are ignored on this path (§3.5). Their keys do not become tables;
only the warehouse tables named in the SQL are available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import pyarrow as pa

from tests.fakes.engine import (
    COLUMN_TOTAL_INDICATOR,
    GRAND_TOTAL_KEY,
    NULL_SORT_SQL,
    TOTAL_INDICATOR_COLUMN,
    BenchEngine,
    PlanFailure,
    arrow_data_type,
    fetch_arrow,
    sql_ident,
)

__all__ = [
    "SQL_WRAPPER_ALIAS",
    "SUMM_SIDECAR_SUFFIX",
    "SqlJob",
    "is_raw_sql_job",
    "run_sql_job",
    "sidecar_name",
    "synthesize_fields",
]

#: Suffix of the totals sidecar column a raw-SQL ``column_totals`` job emits (§2.7).  The prefix
#: is the **lowercased** result-column name, which is why the client's normalizer lowercases
#: before matching a sidecar back to its field.
SUMM_SIDECAR_SUFFIX: Final = "__omni_summ"

#: Alias the fake gives the sub-select it wraps ``userEditedSQL`` in for sorts and totals.
SQL_WRAPPER_ALIAS: Final = "omni_sql_wrapper"


def is_raw_sql_job(query: Mapping[str, Any]) -> bool:
    """Whether this query runs the caller's SQL **verbatim** (§3.4).

    ``userEditedSQL`` on its own is **not** enough — that is the whole point.  One of the three
    "do not rewrite this" markers has to sit next to it; without one the text takes the parsed
    OmniSQL path instead (:func:`tests.fakes.omnisql.is_omnisql_job`), which is a governed model
    job rather than a warehouse passthrough.
    """
    sql = query.get("userEditedSQL")
    if not isinstance(sql, str) or not sql.strip():
        return False
    return (
        query.get("rewriteSql") is False
        or query.get("parsed") is False
        or query.get("dbtMode") is True
    )


# --------------------------------------------------------------------------------------
# The compiled SQL job
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _SqlTotals:
    """The compiled ``column_totals`` request of a raw-SQL job: which columns, and the SUM SQL."""

    columns: tuple[str, ...]
    sql: str


@dataclass(frozen=True)
class SqlJob:
    """A compiled raw-SQL job: what ran, what it returns, and what was dropped on the way."""

    #: The SQL actually executed — ``user_sql``, wrapped in a sort sub-select when
    #: ``sqlSortsEnabled`` applied envelope sorts.  This is what ``summary.display_sql`` reports.
    sql: str
    #: ``query.userEditedSQL`` as the caller wrote it.
    user_sql: str
    #: The result schema.  ``summary.fields`` is synthesized from it (:func:`synthesize_fields`);
    #: it never includes the sidecars or the indicator column the fake appends afterwards.
    schema: pa.Schema
    #: Result columns that got a ``__omni_summ`` sidecar and a totals row.
    totaled_columns: tuple[str, ...] = ()
    #: ``filters`` entries keyed by a governed measure — silently skipped, per §3.1.
    skipped_filters: tuple[str, ...] = ()
    #: Whether ``sqlSortsEnabled`` was on *and* ``sorts`` were non-empty.
    sorts_applied: bool = False
    #: ``summary.fields`` when the path computed them itself — the OmniSQL resolver binds result
    #: columns back to model fields (:mod:`tests.fakes.omnisql`).  ``None`` on the verbatim path,
    #: where there is no model to bind to and they are synthesized from the Arrow schema.
    summary_fields: Mapping[str, Any] | None = None
    #: ``summary.omni_sql`` — the OmniSQL text on the parsed path, empty on the verbatim one
    #: (hand-written SQL has no Omni-flavored form).
    omni_sql: str = ""

    @property
    def result_fields(self) -> Mapping[str, Any]:
        """``summary.fields`` for this job, however this path arrived at them."""
        if self.summary_fields is not None:
            return self.summary_fields
        return synthesize_fields(self.schema)


# --------------------------------------------------------------------------------------
# summary.fields for a SQL result
# --------------------------------------------------------------------------------------


def sidecar_name(column: str) -> str:
    """The ``__omni_summ`` sidecar column for ``column`` — the prefix is **lowercased** (§2.7)."""
    return f"{column.lower()}{SUMM_SIDECAR_SUFFIX}"


def synthesize_fields(schema: pa.Schema) -> dict[str, Any]:
    """``summary.fields`` for a raw-SQL result, synthesized from the Arrow schema.

    Every column is reported as a **dimension**: the view Omni wraps around ``userEditedSQL``
    has an empty measures map (§3.1 — the same fact that makes measure-keyed filters a no-op on
    a SQL job), so there is nothing for the fake to call a measure.  ``data_type`` comes from the
    Arrow type, and the key is the result column name exactly as the SQL spelled it, which is
    what makes ``summary.fields`` usable as the schema authority for the decoded Arrow (§2.3).
    """
    fields: dict[str, Any] = {}
    for column in schema:
        view_name, _, field_name = column.name.partition(".")
        fields[column.name] = {
            "field_name": field_name or column.name,
            "fully_qualified_name": column.name,
            "view_name": view_name,
            "data_type": arrow_data_type(column.type),
            "is_dimension": True,
            "is_calc": False,
            "label": (field_name or column.name).replace("_", " ").title(),
            "format": None,
            "date_type": None,
            "aggregate_type": None,
            "filter_only_field": False,
            "hidden": False,
            "sql": "",
        }
    return fields


# --------------------------------------------------------------------------------------
# Running a SQL job
# --------------------------------------------------------------------------------------


def run_sql_job(
    engine: BenchEngine,
    query: Mapping[str, Any],
    *,
    plan_only: bool = False,
) -> tuple[SqlJob, pa.Table | None]:
    """Compile and (unless ``plan_only``) run a raw-SQL job.

    Returns the compiled job and its result table (``None`` for ``planOnly``).  Every refusal is
    a :class:`~tests.fakes.engine.PlanFailure`, which the HTTP layer turns into an in-band job
    error line — a SQL job the warehouse rejects reports ``error_type: "QUERY"`` instead of
    ``"PLAN"``, matching the server's split between planner and query failures (§2.2).
    """
    user_sql = _statement(query)
    skipped = _partition_filters(engine, query)
    schema = _probe_schema(engine, user_sql)
    # `sqlSortsEnabled` falsy strips sorts AND column_totals, silently: the server forces
    # both to empty for a SQL job that did not opt in (§3.4).
    sorts_enabled = bool(query.get("sqlSortsEnabled"))
    order_by = _order_by(query.get("sorts"), schema) if sorts_enabled else ""
    totals = (
        _resolve_totals(query.get("column_totals"), schema, user_sql) if sorts_enabled else None
    )

    sql = _with_order_by(user_sql, order_by) if order_by else user_sql
    job = SqlJob(
        sql=sql,
        user_sql=user_sql,
        schema=schema,
        totaled_columns=() if totals is None else totals.columns,
        skipped_filters=skipped,
        sorts_applied=bool(order_by),
    )
    if plan_only:
        return job, None
    table = _execute(engine, sql)
    if totals is not None:
        table = _append_sidecar_totals(engine, table, totals)
    return job, table


def _statement(query: Mapping[str, Any]) -> str:
    """``userEditedSQL``, trimmed of the trailing semicolon a sub-select cannot carry."""
    raw = query.get("userEditedSQL")
    if not isinstance(raw, str):  # pragma: no cover - guarded by is_raw_sql_job
        raise PlanFailure("query.userEditedSQL must be a string")
    return raw.strip().rstrip(";").strip()


def _execute(engine: BenchEngine, sql: str) -> pa.Table:
    try:
        return fetch_arrow(engine.connection.execute(sql))
    except Exception as exc:  # duckdb raises a family of its own error types
        raise PlanFailure(_sql_error(exc), error_type="QUERY") from None


def _probe_schema(engine: BenchEngine, user_sql: str) -> pa.Schema:
    """The SQL's result schema, without running it for real — also how ``planOnly`` is served."""
    return _execute(
        engine, f"SELECT * FROM (\n{user_sql}\n) AS {sql_ident(SQL_WRAPPER_ALIAS)} LIMIT 0"
    ).schema


def _sql_error(exc: Exception) -> str:
    return f"userEditedSQL failed: {type(exc).__name__}: {exc}".strip()


def _with_order_by(user_sql: str, order_by: str) -> str:
    return f"SELECT * FROM (\n{user_sql}\n) AS {sql_ident(SQL_WRAPPER_ALIAS)}\nORDER BY {order_by}"


# --------------------------------------------------------------------------------------
# filters (§3.1) — measures skipped, dimensions refused
# --------------------------------------------------------------------------------------


def _partition_filters(engine: BenchEngine, query: Mapping[str, Any]) -> tuple[str, ...]:
    """Decide what happens to each ``filters`` entry on a SQL job; return the skipped measures.

    Source-pinned (§3.1): a measure-keyed entry is a **silent no-op** because the SQL wrapper
    view has no measures.  A dimension-keyed entry is where Omni would splice a mustache
    template into the SQL text; the fake will not guess at that, so it refuses loudly rather
    than return rows that look filtered but are not.
    """
    raw = query.get("filters")
    if not raw:
        return ()
    if not isinstance(raw, Mapping):
        raise PlanFailure("query.filters must be an object keyed by field name")
    skipped: list[str] = []
    for key in raw:
        name = str(key)
        resolved = engine.resolve(name)
        if resolved is None:
            raise PlanFailure(f'query.filters[{name!r}]: No such field "{name}"')
        if resolved.is_dimension:
            raise PlanFailure(
                f"query.filters[{name!r}]: a dimension-keyed filter on a raw-SQL job is applied "
                "by templating it into the SQL text, which FakeOmniAPI does not model; express "
                "the predicate in query.userEditedSQL itself"
            )
        skipped.append(name)
    return tuple(skipped)


# --------------------------------------------------------------------------------------
# sorts (§3.4)
# --------------------------------------------------------------------------------------


def _order_by(raw: object, schema: pa.Schema) -> str:
    """``sorts`` rendered as an ORDER BY over the SQL result — only when ``sqlSortsEnabled``."""
    terms: list[str] = []
    for index, entry in enumerate(_sequence(raw, "query.sorts")):
        where = f"query.sorts[{index}]"
        if not isinstance(entry, Mapping):
            raise PlanFailure(f"{where} must be an object")
        column_name = entry.get("column_name")
        if not isinstance(column_name, str) or not column_name:
            raise PlanFailure(f"{where}: column_name is required")
        if column_name not in schema.names:
            raise PlanFailure(
                f"{where}: {column_name!r} is not a column of the userEditedSQL result "
                f"({', '.join(schema.names)}); a SQL job sorts by result column, not by field"
            )
        direction = "DESC" if entry.get("sort_descending") else "ASC"
        nulls = NULL_SORT_SQL.get(str(entry.get("null_sort") or ""), "")
        terms.append(f"{sql_ident(column_name)} {direction}{nulls}")
    return ", ".join(terms)


# --------------------------------------------------------------------------------------
# column_totals → __omni_summ sidecars (§2.7)
# --------------------------------------------------------------------------------------


def _resolve_totals(raw: object, schema: pa.Schema, user_sql: str) -> _SqlTotals | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PlanFailure("query.column_totals must be an object keyed by result column name")
    if not raw:
        return None

    names: list[str] = []
    for key, payload in raw.items():
        name = str(key)
        where = f"query.column_totals[{name!r}]"
        if not isinstance(payload, Mapping) or payload.get("type") != "aggregation":
            raise PlanFailure(f'{where}: expected {{"type": "aggregation"}}')
        if name == GRAND_TOTAL_KEY:
            raise PlanFailure(
                f"{where}: {GRAND_TOTAL_KEY!r} totals every MEASURE of the query, and the view "
                "Omni wraps around userEditedSQL has an empty measures map (CONTRACT_NOTES "
                "§3.1); name the result columns to total instead"
            )
        if name not in schema.names:
            raise PlanFailure(
                f"{where}: {name!r} is not a column of the userEditedSQL result "
                f"({', '.join(schema.names)})"
            )
        if not _is_numeric(schema.field(name).type):
            raise PlanFailure(
                f"{where}: only numeric columns can be totaled; {name!r} is "
                f"{schema.field(name).type}"
            )
        if name not in names:
            names.append(name)

    select = ", ".join(f"SUM({sql_ident(name)}) AS {sql_ident(name)}" for name in names)
    return _SqlTotals(
        columns=tuple(names),
        sql=f"SELECT {select}\nFROM (\n{user_sql}\n) AS {sql_ident(SQL_WRAPPER_ALIAS)}",
    )


def _append_sidecar_totals(engine: BenchEngine, data: pa.Table, spec: _SqlTotals) -> pa.Table:
    """Append the §2.7 raw-SQL totals framing: one totals row, sidecars, and the indicator.

    The shape is the one ``tests/wire/fixtures/totals_with_sidecars.ndjson`` models, because
    that fixture is what the client's normalizer was written against:

    * every base column is **NULL** on the appended row — the value lives in the sidecar;
    * ``<column-lowercased>__omni_summ`` is NULL on the data rows and carries ``SUM(column)`` on
      the totals row, in the base column's own Arrow type;
    * ``$omni_column_total_indicator`` is NULL on data rows and ``column_total`` on the totals
      row, since the totals were requested per column rather than as ``::total::``.
    """
    totals = _execute(engine, spec.sql)
    rows = data.num_rows

    # pyarrow's stubs type array/field construction by concrete Arrow type, which a generic
    # column loop cannot satisfy; the public signature above stays precise.
    arrays: list[Any] = []
    fields: list[Any] = []
    for column in data.schema:
        dtype: Any = column.type
        fields.append(column)
        arrays.append(_with_null_row(data.column(column.name), dtype))
        if column.name not in spec.columns:
            continue
        value: Any = totals.column(column.name).combine_chunks()
        try:
            value = value.cast(dtype)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError) as exc:
            raise PlanFailure(
                f"query.column_totals[{column.name!r}]: the total does not fit the column's "
                f"type ({dtype}): {exc}"
            ) from None
        fields.append(pa.field(sidecar_name(column.name), dtype))
        arrays.append(pa.chunked_array([pa.nulls(rows, type=dtype), value], type=dtype))

    text: Any = pa.string()
    fields.append(pa.field(TOTAL_INDICATOR_COLUMN, text))
    arrays.append(
        pa.chunked_array(
            [pa.nulls(rows, type=text), pa.array([COLUMN_TOTAL_INDICATOR], type=text)],
            type=text,
        )
    )
    return pa.table(arrays, schema=pa.schema(fields))


def _with_null_row(column: Any, dtype: Any) -> Any:
    """``column`` plus one trailing NULL — the totals row's value for every base column."""
    return pa.chunked_array([*column.chunks, pa.nulls(1, type=dtype)], type=dtype)


def _is_numeric(dtype: pa.DataType) -> bool:
    return bool(
        pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)
    )


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _sequence(raw: object, where: str) -> Sequence[Any]:
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise PlanFailure(f"{where} must be an array")
    return raw
