"""The DataFrame: a lazy, immutable handle on a logical plan (docs/INTERNALS.md §5).

Every transformation returns a **new** DataFrame wrapping a new plan; nothing touches the
network until an action (:meth:`~DataFrame.collect`, :meth:`~DataFrame.to_pandas`,
:meth:`~DataFrame.to_arrow`, :meth:`~DataFrame.show`, :meth:`~DataFrame.count`,
:meth:`~DataFrame.first`, :attr:`~DataFrame.schema`, ``explain(analyze=True)``).

Four behaviors are worth reading before the code:

* **The limit is always explicit** (docs/DESIGN.md §3).  Without ``.limit()`` omniframes sends
  :data:`DEFAULT_FETCH_LIMIT`; every action warns with
  :class:`~omniframes.errors.TruncationWarning` when the number of returned rows equals the
  applied limit, because that is exactly the case where rows may be missing.
* **Aliases are client-side.**  ``.alias()`` never reaches the wire: it is recorded in the plan,
  reverse-resolved when a filter or sort mentions it, and applied as a rename after the result
  comes back.  Collisions fail at build time, not at action time.
* **Selecting a measure is the group-by.**  ``select("users.state", F.measure("…"))`` returns one
  row per state, and :meth:`~DataFrame.group_by` + :meth:`GroupedData.agg` is sugar that compiles
  to the identical envelope (docs/DESIGN.md §2).  Filtering on a measure — before or after the
  ``group_by`` — is a genuine ``HAVING``, not a client-side pass.
* **Totals are opt-in.**  :meth:`~DataFrame.with_totals` asks Omni for its grand-total row and
  hands it back appended to the data with a trailing ``row_type`` column; without it, totals rows
  and Omni's reserved columns are stripped before a frame ever reaches the user
  (CONTRACT_NOTES §2.7).
* **What Omni cannot express runs here, visibly.**  A derived column, a UDF, an ad-hoc
  aggregation next to a governed measure — the splitter pushes down the largest governed query
  it can (tier 1), writes SQL over a governed reference core for what is left when it can
  (tier 2, docs/SQLTIER.md), and finishes the rest locally (docs/HYBRID.md).  There is no silent
  fallback: :meth:`~DataFrame.explain` prints every query sent, the tier it ran at, and every
  operator run here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa

from omniframes.column import AdHocAgg, Column, Expr, FieldRef, MeasureRef
from omniframes.compile.executor import execute, remote_errors
from omniframes.compile.explain import explain_text
from omniframes.compile.querymodel import DEFAULT_FETCH_LIMIT, UNSET, Unset
from omniframes.compile.semantic import ExecutionPlan, RemoteStep, alias_map, display_name
from omniframes.compile.splitter import SplitOptions, split
from omniframes.errors import CompileError, OmniframesError, TruncationWarning
from omniframes.functions import col as _col
from omniframes.io.writers import DataFrameWriter
from omniframes.plan import nodes
from omniframes.transport.arrow import schema_from_summary
from omniframes.transport.normalize import (
    GRAND_TOTAL_VALUE,
    NormalizedResult,
    normalize,
    resolve_aliases,
)
from omniframes.types import OmniDataType, OmniField, OmniSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from omniframes.session import OmniSession

__all__ = ["DEFAULT_FETCH_LIMIT", "ROW_TYPE_COLUMN", "DataFrame", "GroupedData"]

_SHOW_DEFAULT_ROWS = 20

#: The column ``with_totals()`` appends to the materialized frame.  ``/query/run`` has no
#: ``row_type`` of its own — it flags totals rows with reserved indicator columns that never
#: reach a user (CONTRACT_NOTES §2.7) — so omniframes derives one.
ROW_TYPE_COLUMN = "row_type"

#: The value :data:`ROW_TYPE_COLUMN` carries on ordinary rows and on the grand-total row.
_DATA_ROW_TYPE = "data"
_TOTAL_ROW_TYPE = "total"

#: ``row_type`` as a schema field.  Derived client-side, so ``summary.fields`` never mentions it.
_ROW_TYPE_FIELD = OmniField(name=ROW_TYPE_COLUMN, data_type=OmniDataType.STRING, is_dimension=True)

#: What :meth:`DataFrame.to_polars` raises without the optional extra.  Pinned by a test: the
#: install command is the whole point of the message.
_POLARS_MISSING = "to_polars() needs the optional dependency: pip install 'omniframes[polars]'"


class DataFrame:
    """A lazy table of rows produced by an Omni query."""

    __slots__ = ("_execution", "_plan", "_schema", "_session", "_totals")

    def __init__(self, session: OmniSession, plan: nodes.PlanNode, *, totals: bool = False) -> None:
        self._session = session
        self._plan = plan
        self._totals = totals
        self._execution: ExecutionPlan | None = None
        self._schema: OmniSchema | None = None

    # -- identity ----------------------------------------------------------------------

    @property
    def session(self) -> OmniSession:
        """The session this frame runs against."""
        return self._session

    @property
    def logical_plan(self) -> nodes.PlanNode:
        """The immutable plan this frame wraps."""
        return self._plan

    def __repr__(self) -> str:
        try:
            return f"DataFrame[{', '.join(self.columns)}]"
        except CompileError:
            return f"DataFrame(<{type(self._plan).__name__} plan>)"

    def _derive(self, plan: nodes.PlanNode) -> DataFrame:
        return DataFrame(self._session, plan, totals=self._totals)

    # -- transformations ---------------------------------------------------------------

    def select(self, *columns: str | Column | Iterable[str | Column]) -> DataFrame:
        """Choose the columns to return.

        In Omni, selecting dimensions together with governed measures *is* the group-by, so
        ``select("users.state", F.measure("order_items.total_sale_price"))`` returns one row per
        state.  Strings are field names; :meth:`~omniframes.column.Column.alias` renames
        client-side.
        """
        chosen = _as_columns(columns)
        if not chosen:
            raise CompileError("select() needs at least one column")
        alias_map(chosen)  # fail fast on alias collisions, at build time
        return self._derive(nodes.Project(self._plan, chosen))

    def group_by(self, *columns: str | Column | Iterable[str | Column]) -> GroupedData:
        """Group by dimensions (and grains), then call :meth:`GroupedData.agg`.

        Sugar, not a different query: ``df.group_by("users.state").agg(F.measure("m"))`` compiles
        to exactly the same envelope as ``df.select("users.state", F.measure("m"))``, because in
        Omni the selection *is* the group-by (docs/DESIGN.md §2).  Use whichever reads better.

        ``group_by()`` with no keys aggregates the whole frame into a single row.
        """
        keys = _as_columns(columns)
        for key in keys:
            if not isinstance(key.expr, FieldRef):
                raise CompileError(
                    f"group_by() takes dimensions (optionally at a grain); {_name(key)} is not "
                    "one. Governed measures and aggregations belong in .agg(...)."
                )
        return GroupedData(self, keys)

    def filter(self, condition: str | Column | Expr) -> DataFrame:
        """Keep the rows matching ``condition``.

        Use ``&``, ``|`` and ``~`` — never ``and``/``or``/``not``, which Python cannot overload
        (:class:`~omniframes.column.Column` raises a ``TypeError`` explaining this).  A bare
        column name filters a boolean field.

        A predicate on a governed measure (``F.measure("order_items.count") > 10``) is a genuine
        ``HAVING``: the server applies it after the group-by, whether the measure is selected or
        not, and whether the filter is written before or after the ``group_by()``
        (CONTRACT_NOTES §3.1).
        """
        return self._derive(nodes.Filter(self._plan, _as_predicate(condition)))

    def sort(self, *columns: str | Column | Iterable[str | Column]) -> DataFrame:
        """Order rows. ``F.col("x").desc()`` sorts descending; a later ``sort()`` replaces it."""
        keys = tuple(column.to_sort_key() for column in _as_columns(columns))
        if not keys:
            raise CompileError("sort() needs at least one column")
        plan = self._plan
        if isinstance(plan, nodes.Sort):
            # The last sort wins, exactly as if the earlier one had never been applied.
            return self._derive(nodes.Sort(plan.child, keys))
        return self._derive(nodes.Sort(plan, keys))

    def limit(self, n: int | None) -> DataFrame:
        """Cap the number of rows. ``limit(None)`` asks for everything (wire ``limit: null``).

        Limits compose to the tightest of the two: ``df.limit(50).limit(10)`` returns 10 rows.
        """
        if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n <= 0):
            raise CompileError(f"limit() takes a positive integer or None (unlimited); got {n!r}")
        plan = self._plan
        if isinstance(plan, nodes.Limit):
            return self._derive(nodes.Limit(plan.child, _tightest(plan.n, n), plan.offset))
        return self._derive(nodes.Limit(plan, n))

    def offset(self, n: int) -> DataFrame:
        """Skip ``n`` rows.

        ``limit`` and ``offset`` travel together as the wire's ``LIMIT n OFFSET k`` pair, so
        ``df.limit(10).offset(5)`` returns rows 6..15 of the sorted result — SQL semantics.
        Repeated calls accumulate.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise CompileError(f"offset() takes a non-negative integer; got {n!r}")
        plan = self._plan
        if isinstance(plan, nodes.Limit):
            return self._derive(nodes.Limit(plan.child, plan.n, plan.offset + n))
        return self._derive(nodes.Limit(plan, UNSET, n))

    def with_column(self, name: str, column: str | Column | Expr) -> DataFrame:
        """Add (or replace) a derived column.

        SQL-expressible expressions, such as arithmetic over selected fields, push down to
        an OmniSQL job. A :func:`~omniframes.functions.udf` or an expression that cannot run
        remotely is evaluated locally over the largest remote sub-plan. ``explain()`` shows
        which tier runs the expression and where any local work begins.
        """
        if not isinstance(name, str) or not name:
            raise CompileError("with_column() needs a non-empty column name")
        return self._derive(nodes.WithColumn(self._plan, name, _as_predicate(column)))

    def map_pandas(
        self,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        schema_hint: OmniSchema | None = None,
    ) -> DataFrame:
        """Hand the materialized frame to a Python function and take a frame back.

        This is the escape hatch: whatever ``fn`` does, it runs here, over the result of the
        largest query omniframes could push down.

        A Python function's output schema cannot be planned, so without ``schema_hint`` the
        frame no longer knows its own columns: :attr:`schema` and :attr:`columns` raise, and the
        only operations allowed above it are the ones that need no schema (``limit``, ``offset``,
        another ``map_pandas``).  Pass ``schema_hint=OmniSchema(...)`` to declare what comes back
        and everything works normally again.
        """
        if not callable(fn):
            raise CompileError(f"map_pandas() takes a callable, got {type(fn).__name__}")
        if schema_hint is not None and not isinstance(schema_hint, OmniSchema):
            raise CompileError(
                "map_pandas(schema_hint=...) takes an OmniSchema describing the columns the "
                f"function returns; got {type(schema_hint).__name__}"
            )
        return self._derive(nodes.MapPandas(self._plan, fn, schema_hint))

    def join(
        self,
        other: DataFrame,
        on: str | Sequence[str],
        how: str = "inner",
    ) -> DataFrame:
        """Join this frame to ``other`` on shared column names, with **SQL** semantics.

        ``on`` names columns both frames *output* — so a field aliased in ``select()`` is named
        by its alias, and an un-aliased one by its wire name.  ``how`` is ``inner`` (default),
        ``left``, ``right`` or ``outer`` (``full`` / ``full_outer`` mean the same).

        **NULL keys never match** — not even another NULL, exactly as in SQL.  A row whose key
        is NULL takes no part in the matching and reappears only as an unmatched row: dropped by
        an inner join, kept with the other side's columns NULL by the join type that preserves
        its side.  (This is deliberately *not* the alignment a decomposed aggregate uses
        internally, where the two NULL groups are the same group — docs/HYBRID.md §3.2.)

        Each side compiles on its own, so joining across two models, or a governed topic to a
        raw-SQL job, needs nothing special; the join itself always runs locally, because the
        query API takes one query. ``explain()`` shows both sub-plans and the join between them.

        Non-key columns that would collide are a :class:`~omniframes.errors.CompileError` rather
        than a silently suffixed pair: alias or drop one of them first.
        """
        if not isinstance(other, DataFrame):
            raise CompileError(f"join() takes another DataFrame; got {type(other).__name__}")
        if other._session is not self._session:
            raise CompileError(
                "join() needs both frames to come from the same OmniSession: results are "
                "combined in this process, but each side is fetched by its own session"
            )
        if self._totals or other._totals:
            raise CompileError(
                "with_totals() asks Omni for a totals row over one query, and a join is two. "
                "Join first, or total each side and combine the numbers yourself."
            )
        return DataFrame(
            self._session, nodes.Join(self._plan, other._plan, _join_on(on), _how(how))
        )

    def union(self, other: DataFrame) -> DataFrame:
        """Stack ``other``'s rows under this frame's (``UNION ALL`` — nothing is de-duplicated).

        By position, like PySpark — but omniframes additionally insists the column **names**
        line up, because its columns are named wire outputs: quietly relabelling the right
        side's data with the left side's names is the one outcome nobody wants.  Types widen
        where SQL widens (``int64`` + ``float64`` → ``float64``); types with no common type are
        a :class:`~omniframes.errors.CompileError`.
        """
        if not isinstance(other, DataFrame):
            raise CompileError(f"union() takes another DataFrame; got {type(other).__name__}")
        if other._session is not self._session:
            raise CompileError(
                "union() needs both frames to come from the same OmniSession: results are "
                "combined in this process, but each side is fetched by its own session"
            )
        if self._totals or other._totals:
            raise CompileError(
                "with_totals() asks Omni for a totals row over one query, and a union is two. "
                "Union first, or total each side separately."
            )
        return DataFrame(self._session, nodes.Union(self._plan, other._plan))

    def with_totals(self) -> DataFrame:
        """Ask Omni for the grand-total row and keep it in the result.

        The query gains ``column_totals: {"::total::": {"type": "aggregation"}}`` and the
        materialized frame gains a trailing ``row_type`` column: ``"data"`` on the rows the
        query grouped, ``"total"`` on the appended totals row. The totals row **re-aggregates
        the measures over every row the query touched** — post-filter, pre-limit — so it is not
        the sum of the values above it, which is exactly the point of asking the server for it.

        Only meaningful when the query has a measure: without one there is nothing to total, and
        compiling raises :class:`~omniframes.errors.CompileError`.  The marker rides along
        through further transformations, so ``df.with_totals().sort(...)`` still totals.
        """
        return DataFrame(self._session, self._plan, totals=True)

    # -- actions -----------------------------------------------------------------------

    def collect(self) -> pa.Table:
        """Run the query and return the normalized result as an Arrow table."""
        return self._collect()

    def to_arrow(self) -> pa.Table:
        """Run the query and return an Arrow table."""
        return self._collect()

    def to_pandas(self) -> pd.DataFrame:
        """Run the query and return a pandas DataFrame."""
        return self._collect().to_pandas()

    def to_polars(self) -> Any:
        """Run the query and return a polars DataFrame.

        polars is an **optional** dependency — ``pip install 'omniframes[polars]'`` — because
        most users already have pandas and nobody should pay for a second dataframe library they
        did not ask for.  The conversion is zero-copy through Arrow either way.
        """
        try:
            import polars
        except ImportError as exc:
            raise ImportError(_POLARS_MISSING) from exc
        return polars.from_arrow(self._collect())

    @property
    def write(self) -> DataFrameWriter:
        """``df.write.csv(path)`` / ``df.write.parquet(path)`` — runs the query, writes the file."""
        return DataFrameWriter(self)

    def omni_url(self) -> str:
        """Run this query and return the Omni workbook URL that opens it in the browser.

        The handoff from a notebook to the UI: the same governed query, explorable by whoever
        you send the link to.  Two constraints come straight from the wire (CONTRACT_NOTES §2.1):
        ``workbookUrl`` is rejected alongside ``staticQueryReferences``, and a plan that is a DAG
        has no single query for a workbook to be *of*.  Both are refused here, before any
        request, naming which one bit.

        Note this **executes** the query: the URL is minted by the same run that returns the
        rows, and the rows are then thrown away.
        """
        execution = self._compiled()
        step = _single_remote(execution)
        result = self._session.run({**step.envelope, "workbookUrl": True})
        if not result.workbook_url:
            raise OmniframesError(
                "Omni ran the query but returned no workbook URL. Workbook links may not be "
                "enabled for this organization; the query itself succeeded."
            )
        return result.workbook_url

    def count(self) -> int:
        """The number of rows in the materialized frame.

        PySpark-consistent: this counts what ``collect()`` would return — i.e. *after* the
        applied limit — and inherits its truncation warning.  It is not ``COUNT(*)`` over the
        underlying table; select ``F.measure(...)`` for a governed count.
        """
        return self._collect().num_rows

    def first(self) -> dict[str, Any] | None:
        """The first row as a dict, or ``None`` when the result is empty.

        No truncation warning: a limit of one row is what was asked for.
        """
        table = self.limit(1)._collect(warn=False)
        if table.num_rows == 0:
            return None
        row: dict[str, Any] = table.slice(0, 1).to_pylist()[0]
        return row

    def show(self, n: int = _SHOW_DEFAULT_ROWS) -> None:
        """Print up to ``n`` rows as a text table.

        Fetches one row more than it prints, purely to know whether to say "only showing top n
        rows".  That extra row is also why ``show()`` never raises
        :class:`~omniframes.errors.TruncationWarning`: the footer says the same thing without
        crying wolf on every preview.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise CompileError(f"show() takes a positive number of rows; got {n!r}")
        print(self._render(self.limit(n + 1)._collect(warn=False), n))

    @property
    def schema(self) -> OmniSchema:
        """The result schema, from a cached ``planOnly`` round trip.

        ``summary.fields`` is the only schema authority (docs/DESIGN.md §3) — never the catalog:
        it is what the planner will actually return, grains and all.  The plan job runs once per
        DataFrame instance.  The one field omniframes adds itself is ``row_type``, which
        :meth:`with_totals` derives client-side and the server therefore never describes.
        """
        if self._schema is None:
            self._schema = self._resolve_schema()
        return self._schema

    def _resolve_schema(self) -> OmniSchema:
        execution = self._compiled()
        if execution.root is not None:
            plan = self._plan
            if isinstance(plan, nodes.MapPandas):
                if plan.schema_hint is not None:
                    return plan.schema_hint
                raise CompileError(
                    "map_pandas() hands the frame to a Python function, whose output schema "
                    "cannot be planned. Pass schema_hint=OmniSchema(...) to declare it, or call "
                    "collect() and read the Arrow schema of the result."
                )
            raise CompileError(
                "this frame finishes locally, so its result schema is only known once it runs "
                "(explain() shows the split). Use .columns for the names, or collect() and read "
                "the Arrow schema of the result."
            )
        step = execution.remote
        with remote_errors(step):
            result = self._session.plan(step.envelope)
        schema = _rename_schema(schema_from_summary(result.summary.get("fields")), step.alias_map)
        if self._totals:
            schema = OmniSchema((*schema.fields, _ROW_TYPE_FIELD))
        return schema

    @property
    def columns(self) -> tuple[str, ...]:
        """The output column names, aliases applied. Compiles the plan; performs no I/O."""
        names = self._compiled().columns
        return (*names, ROW_TYPE_COLUMN) if self._totals else names

    def explain(self, analyze: bool = False) -> str:
        """Describe how this frame will run.

        With ``analyze=True`` the plan is sent to Omni with ``planOnly: true`` and the server's
        own SQL is appended — blanked out for callers without the ``VIEW_SQL`` permission, which
        the output says explicitly rather than pretending there is no SQL.
        """
        execution = self._compiled()
        summaries: list[Mapping[str, Any]] | None = None
        if analyze:
            summaries = [self._session.plan(step.envelope).summary for step in execution.steps]
        return explain_text(self._plan, execution, analyze=summaries)

    # -- camelCase aliases (PySpark muscle memory, docs/DESIGN.md §3) --------------------

    where = filter
    orderBy = sort
    groupBy = group_by
    toPandas = to_pandas
    toArrow = to_arrow
    toPolars = to_polars
    withColumn = with_column
    mapInPandas = map_pandas
    #: ``union`` is already ``UNION ALL``; the alias exists for PySpark muscle memory only.
    unionAll = union

    # -- internals ---------------------------------------------------------------------

    def _compiled(self) -> ExecutionPlan:
        if self._execution is None:
            self._execution = split(
                self._plan,
                options=SplitOptions(
                    envelope=self._session.envelope_options(),
                    totals=self._totals,
                    decomposition_row_cap=self._session.decomposition_row_cap,
                ),
            )
        return self._execution

    def _collect(self, *, warn: bool = True) -> pa.Table:
        execution = self._compiled()
        if execution.root is not None:
            return execute(execution, self._run_remote, warn=warn, stacklevel=4)
        step = execution.remote
        with remote_errors(step):
            result = self._session.run(step.envelope)
        normalized = normalize(
            result.table,
            result.summary.get("fields"),
            keep_totals=self._totals,
            aliases=step.alias_map,
        )
        if warn:
            # The totals row is not a data row, so it never signals truncation.
            _warn_if_truncated(normalized.data.num_rows, step.applied_limit)
        return _with_row_type(normalized) if self._totals else normalized.data

    def _run_remote(self, step: RemoteStep) -> pa.Table:
        """Run one step of a split plan. Totals never reach here — they are tier 1 only."""
        result = self._session.run(step.envelope)
        return normalize(result.table, result.summary.get("fields"), aliases=step.alias_map).data

    @staticmethod
    def _render(table: pa.Table, limit: int) -> str:
        names = list(table.column_names)
        rows = [
            [_cell(value) for value in row.values()] for row in table.slice(0, limit).to_pylist()
        ]
        widths = [
            max(len(name), *(len(row[index]) for row in rows)) if rows else len(name)
            for index, name in enumerate(names)
        ]
        rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

        def line(cells: Sequence[str]) -> str:
            padded = (f" {cell:<{width}} " for cell, width in zip(cells, widths, strict=True))
            return "|" + "|".join(padded) + "|"

        body = [rule, line(names), rule, *(line(row) for row in rows), rule]
        if table.num_rows > limit:
            body.append(f"only showing top {limit} rows")
        return "\n".join(body)


# --------------------------------------------------------------------------------------
# GroupedData
# --------------------------------------------------------------------------------------


class GroupedData:
    """The result of :meth:`DataFrame.group_by` — call :meth:`agg` to get a DataFrame back.

    Deliberately tiny: it holds the group keys and nothing else.  ``agg()`` validates what it is
    handed *at build time* (a group key must be a dimension, an aggregate must be a measure or an
    ad-hoc aggregation) so that a typo fails where it was written rather than at action time.
    """

    __slots__ = ("_frame", "_keys")

    def __init__(self, frame: DataFrame, keys: tuple[Column, ...]) -> None:
        self._frame = frame
        self._keys = keys

    def __repr__(self) -> str:
        return f"GroupedData[{', '.join(_name(key) for key in self._keys)}]"

    @property
    def keys(self) -> tuple[Column, ...]:
        """The group keys, in order."""
        return self._keys

    def agg(self, *columns: str | Column | Iterable[str | Column]) -> DataFrame:
        """Aggregate each group.

        Takes governed measures (``F.measure("order_items.total_sale_price")``) and ad-hoc
        aggregations (``F.count_distinct("users.id")``). Governed measures always execute
        remotely. Ad-hoc and mixed aggregations use an OmniSQL job when expressible;
        otherwise the splitter uses local aggregation, with a separate remote query for
        governed measures when needed. ``explain()`` shows the resulting plan.
        """
        aggs = _as_columns(columns)
        if not aggs:
            raise CompileError(
                "agg() needs at least one aggregate, e.g. F.measure('order_items.total_sale_price')"
            )
        for agg in aggs:
            if not isinstance(agg.expr, MeasureRef | AdHocAgg):
                raise CompileError(
                    f"agg() takes governed measures (F.measure(...)) and ad-hoc aggregations "
                    f"(F.sum/count/count_distinct/...); {_name(agg)} is neither. Plain "
                    "dimensions belong in group_by(...)."
                )
        alias_map((*self._keys, *aggs))  # fail fast on alias collisions, at build time
        return self._frame._derive(nodes.Aggregate(self._frame.logical_plan, self._keys, aggs))


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _name(column: Column) -> str:
    """A column's user-facing name, for error messages."""
    return column.alias_name or display_name(column.expr)


def _single_remote(execution: ExecutionPlan) -> RemoteStep:
    """The one governed query ``omni_url()`` can open, or a :class:`CompileError` saying why not.

    Three separate constraints, reported separately because the fix differs: a DAG has no single
    query to be a workbook *of*; a query carrying ``staticQueryReferences`` is rejected outright
    when ``workbookUrl`` is set (CONTRACT_NOTES §2.1), so that refusal happens here rather than
    as a 400 nobody can act on; and a workbook explores model *fields*, which a step whose
    payload is a SQL statement — tier 2 or ``read.sql`` — has none of.
    """
    if execution.root is not None or len(execution.steps) != 1:
        raise CompileError(
            "omni_url() opens one query in a workbook, and this frame compiles to several "
            "(explain() shows the split). Open the governed part on its own, or push the whole "
            "frame into one query."
        )
    step = execution.steps[0]
    if step.query.static_query_references:
        raise CompileError(
            "omni_url() is not available for this frame: its query carries query references, "
            "and the query API rejects workbookUrl together with staticQueryReferences "
            "(CONTRACT_NOTES §2.1). Only a query Omni's model can express on its own has a "
            "workbook to open."
        )
    if step.tier != 1:
        raise CompileError(
            "omni_url() opens a governed query in a workbook, and this frame compiles to a "
            f"{step.label} step instead. A workbook explores model fields; SQL omniframes did "
            "not write has none to show."
        )
    return step


def _with_row_type(result: NormalizedResult) -> pa.Table:
    """Append the derived ``row_type`` column and the totals rows (CONTRACT_NOTES §2.7)."""
    table = result.data
    labels = [_DATA_ROW_TYPE] * table.num_rows
    if result.totals is not None and result.totals.num_rows:
        table = pa.concat_tables([table, result.totals])
        labels.extend(_row_type(value) for value in result.total_row_types)
    return table.append_column(ROW_TYPE_COLUMN, pa.array(labels, type=pa.string()))


def _row_type(indicator: str | None) -> str:
    """The indicator value of a totals row as the label a user reads."""
    if indicator is None or indicator == GRAND_TOTAL_VALUE:
        return _TOTAL_ROW_TYPE
    return indicator


def _as_columns(args: tuple[str | Column | Iterable[str | Column], ...]) -> tuple[Column, ...]:
    """Flatten ``select("a", "b")`` and ``select(["a", "b"])`` into Columns."""
    flat: list[str | Column] = []
    for arg in args:
        if isinstance(arg, str | Column):
            flat.append(arg)
        elif isinstance(arg, Iterable):
            flat.extend(arg)
        else:
            raise CompileError(
                f"expected a column name, a Column or an iterable of them; got {type(arg).__name__}"
            )
    return tuple(_col(entry) for entry in flat)


#: Every spelling of a join kind omniframes accepts.  The synonyms are PySpark's; the canonical
#: names are the four the error message lists.
_JOIN_HOWS: Mapping[str, nodes.JoinHow] = {
    "inner": nodes.JoinHow.INNER,
    "left": nodes.JoinHow.LEFT,
    "leftouter": nodes.JoinHow.LEFT,
    "left_outer": nodes.JoinHow.LEFT,
    "right": nodes.JoinHow.RIGHT,
    "rightouter": nodes.JoinHow.RIGHT,
    "right_outer": nodes.JoinHow.RIGHT,
    "outer": nodes.JoinHow.OUTER,
    "full": nodes.JoinHow.OUTER,
    "fullouter": nodes.JoinHow.OUTER,
    "full_outer": nodes.JoinHow.OUTER,
}


def _how(how: str) -> nodes.JoinHow:
    if not isinstance(how, str):
        raise CompileError(f"join(how=...) takes a string; got {type(how).__name__}")
    kind = _JOIN_HOWS.get(how.strip().lower())
    if kind is None:
        raise CompileError(
            f"{how!r} is not a join kind omniframes knows; use 'inner', 'left', 'right' or "
            "'outer' ('full'/'full_outer' are accepted spellings of 'outer')"
        )
    return kind


def _join_on(on: str | Sequence[str]) -> tuple[str, ...]:
    """``on`` as the tuple of column names both frames must produce."""
    if isinstance(on, str):
        names: tuple[str, ...] = (on,)
    elif isinstance(on, Sequence):
        names = tuple(on)
    else:
        raise CompileError(
            f"join(on=...) takes a column name or a list of column names; got {type(on).__name__}"
        )
    if not names:
        raise CompileError("join() needs at least one column to join on")
    wrong = [name for name in names if not isinstance(name, str) or not name]
    if wrong:
        raise CompileError(
            "join(on=...) takes the names of columns both frames produce (aliases included); "
            f"got {wrong!r}"
        )
    return names


def _as_predicate(condition: str | Column | Expr) -> Expr:
    if isinstance(condition, Column):
        return condition.expr
    if isinstance(condition, Expr):
        return condition
    if isinstance(condition, str):
        # A bare field name is a boolean-field filter, matching select()'s auto-wrapping.
        _refuse_expression_string(condition)
        return _col(condition).expr
    raise CompileError(
        "filter() takes a Column expression, e.g. F.col('users.state') == 'California'; "
        f"got {type(condition).__name__}"
    )


#: Tokens that give a ``filter()`` string away as a SQL expression rather than a field name.
_SQL_OPERATORS: Final = frozenset("<>=!'\"()")
_SQL_KEYWORDS: Final = frozenset({"and", "or", "not", "like", "in", "is", "between"})


def _refuse_expression_string(condition: str) -> None:
    """Refuse ``filter("users.age > 21")`` at build time, where it was written.

    PySpark's ``filter()`` takes a SQL-expression string; omniframes has no SQL parser, so the
    string would become a *field name* — a filter keyed by ``"users.age > 21"``, which
    ``explain()`` renders byte for byte like a correctly compiled predicate and the server
    rejects only at action time (CONTRACT_NOTES §3.1: nonexistent filter key → hard error).
    This is a shape check, not a schema check: it needs no catalog, so it does not reintroduce
    I/O at build time (docs/DESIGN.md §3) and a plain misspelled field still compiles.
    """
    tokens = {word.lower() for word in condition.split()}
    if not (_SQL_OPERATORS & set(condition) or _SQL_KEYWORDS & tokens):
        return
    raise CompileError(
        f"filter({condition!r}) looks like a SQL expression, and omniframes has no "
        "SQL-expression parser: a string here is a boolean field name and nothing else. Write it "
        "as a Column expression — F.col('users.age') > 21 — combining operands with & | ~ and "
        "parenthesizing each one."
    )


def _tightest(current: int | Unset | None, requested: int | None) -> int | Unset | None:
    """Compose two limits: the smaller wins, and an existing limit beats ``None``."""
    if current is UNSET:
        return requested
    if requested is None:
        return current
    if current is None:
        return requested
    return min(current, requested)


def _warn_if_truncated(rows: int, applied_limit: int | None) -> None:
    if applied_limit is None or rows != applied_limit:
        return
    warnings.warn(
        f"the result has exactly {rows} rows, which is the applied limit — rows are probably "
        "missing. Raise .limit(n), use .limit(None) for everything, or narrow the query.",
        TruncationWarning,
        stacklevel=4,
    )


def _rename_schema(schema: OmniSchema, aliases: Mapping[str, str]) -> OmniSchema:
    """Apply the step's renames to a planned schema, by the same rules the result columns get.

    ``summary.fields`` carries the server's names, so a tier-2 expression item is described
    under the scoped ``<view>.of_expr_<n>`` the client cannot predict — matched by suffix here
    exactly as in :func:`~omniframes.transport.normalize.normalize` (docs/SQLTIER.md §3.2).
    """
    if not aliases:
        return schema
    renamed = resolve_aliases(schema.names, aliases)
    return OmniSchema(
        tuple(replace(f, name=name) for f, name in zip(schema.fields, renamed, strict=True))
    )


def _cell(value: object) -> str:
    return "NULL" if value is None else str(value)
