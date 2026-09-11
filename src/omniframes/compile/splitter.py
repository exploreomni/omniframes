"""The splitter: one logical plan in, a DAG of remote queries and local operators out.

The whole algorithm is one invariant, applied top-down (docs/HYBRID.md §2, docs/SQLTIER.md §4):

    **At every node, try the tiers in order over the entire subtree: the governed query
    (tier 1), then one OmniSQL statement (tier 2).**  If either compiles, that subtree *is* a
    remote step and the recursion stops there.  Only when both refuse does the node itself
    become a local operator over the split of its children.

Because the attempt happens at every level, the remote frontier is automatically maximal — no
separate analysis pass, and no way for the two to disagree.  A tier-2 refusal is never a user
error: tier 3 can express everything tier 2 can, so the fallback is silent and the tier that ran
is always visible in ``explain()``.  Three consequences are worth spelling out, because they are
what makes the result predictable:

* **Column widening** (§2.2).  A local operator can only reference columns the query below it
  fetched, so the splitter adds the missing bare fields to that query's projection and drops
  them again in the final projection.  A field that cannot be widened in — because it would
  have to come out of an aggregate, or out of a ``map_pandas()`` whose schema nobody declared —
  is an error naming the column, never a silent NULL.
* **A ``Limit`` pins the frontier** (§2.3).  Operations *below* a limit ride the limited remote
  query; operations *above* one run locally over its result.  Both directions are honest about
  what the limit means; the compiler preserves that operation order.
* **Mixed aggregation collapses, and only decomposes as a fallback** (docs/SQLTIER.md §4).  A
  governed measure is a legal select item on the OmniSQL path and mixes freely with ad-hoc
  aggregates in one statement, so ``agg(F.measure(...), F.count_distinct(...))`` is ONE tier-2
  job whenever tier 2 can write it.  When it cannot, aggregate decomposition answers instead: a
  governed tier-1 query for the measures, a tier-2 ``GROUP BY`` (or an unlimited raw scan plus
  a local aggregate) for the ad-hoc half, and an :class:`~omniframes.compile.local.AlignJoin`
  re-assembling the two on the group keys.  That join stays local on purpose: matching NULL keys
  to each other needs ``IS NOT DISTINCT FROM``, which is not portable across warehouse dialects,
  and both sides are already aggregated (docs/HYBRID.md §2.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from omniframes.column import (
    AdHocAgg,
    Arithmetic,
    Between,
    BooleanOp,
    Column,
    Comparison,
    Expr,
    FieldRef,
    IsIn,
    IsNull,
    MeasureRef,
    Not,
    SortKey,
    StringPredicate,
    Udf,
)
from omniframes.compile.local import (
    AlignJoin,
    LocalAgg,
    LocalAggregate,
    LocalFilter,
    LocalJoin,
    LocalLimit,
    LocalMapPandas,
    LocalOp,
    LocalProject,
    LocalSort,
    LocalUnion,
    LocalWithColumn,
    join_collision_message,
)
from omniframes.compile.querymodel import UNSET
from omniframes.compile.semantic import (
    CannotCompile,
    EnvelopeOptions,
    ExecutionPlan,
    LocalStep,
    RemoteStep,
    SemanticCompilation,
    Step,
    adhoc_filter_reason,
    build_envelope,
    column_key,
    compile_semantic,
    display_name,
    split_grain,
    try_semantic,
    wire_name,
)
from omniframes.compile.sqlgen import try_sql
from omniframes.errors import CompileError
from omniframes.plan import nodes

__all__ = ["SplitOptions", "split"]

#: Nodes a widening walk may pass through on its way down to the projection it has to grow.
_PASS_THROUGH: Final = (nodes.Filter, nodes.Sort, nodes.Limit)


@dataclass(frozen=True)
class SplitOptions:
    """Everything outside the plan that changes what the splitter produces."""

    #: Session knobs that ride the run envelope (branch, cache, timezone, impersonation).
    envelope: EnvelopeOptions | None = None
    #: ``with_totals()`` — legal only when the whole plan compiles to one governed query.
    totals: bool = False
    #: Opt-in cap on the decomposition raw scan (``None`` = unlimited, docs/HYBRID.md §2.1).
    decomposition_row_cap: int | None = None
    #: Skip tier 2 entirely and go straight from tier 1 to the local engine.  The differential
    #: lane's headline check runs the same plan both ways (docs/SQLTIER.md §5/§7); there is
    #: deliberately no public builder knob for it, because a user has no reason to want one.
    disable_sql: bool = False


def split(plan: nodes.PlanNode, *, options: SplitOptions | None = None) -> ExecutionPlan:
    """Compile ``plan`` into the steps that execute it.

    A plan that either remote tier can express entirely produces one
    :class:`~omniframes.compile.semantic.RemoteStep` with ``root=None``. Otherwise it becomes
    a DAG whose local part ``explain()`` prints in full.

    Raises:
        CompileError: the plan is invalid, or needs something no tier implements yet (the
            message carries tier 1's own reason, prefixed with ``"not yet supported: "``).
    """
    settings = options or SplitOptions()
    compilation = try_semantic(plan, totals=settings.totals)
    if compilation is not None:
        # The whole thing is one governed query: no local work, no DAG, nothing to explain.
        return ExecutionPlan((_step(compilation, settings),))
    if settings.totals:
        raise CompileError(
            "with_totals() needs a query Omni can total server-side, and this one does not "
            "compile to a single governed query. Totals re-aggregate the measures over every "
            "row the query touched, which only the server can do — the local engine never "
            "computes governed measures."
        )

    splitter = _Splitter(settings)
    frame = splitter.split(plan, frozenset())
    outputs = _outputs(plan)
    if outputs is None:
        return ExecutionPlan(tuple(splitter.steps), root=frame.step, output_columns=None)

    names = tuple(name for _, name in outputs)
    if isinstance(frame.step, RemoteStep) and len(splitter.steps) == 1 and frame.columns == names:
        # One query, no local work: the degenerate shape again, which is how a tier-2 plan that
        # swallowed the whole thing reports `tier 2` rather than "some local work happened".
        return ExecutionPlan((frame.step,))

    # The final projection drops the columns widening added and fixes the output order, so
    # `explain()` ends by naming the columns the user gets back — unless the plan already ends
    # in exactly that projection, in which case saying it twice would only be noise.
    if not _ends_with_projection(frame, names):
        frame = frame.then(LocalProject(names), names)
    return ExecutionPlan(tuple(splitter.steps), root=frame.step, output_columns=names)


def _ends_with_projection(frame: _Frame, names: tuple[str, ...]) -> bool:
    step = frame.step
    return (
        isinstance(step, LocalStep) and isinstance(step.op, LocalProject) and frame.columns == names
    )


def _step(
    compilation: SemanticCompilation,
    options: SplitOptions,
    *,
    label: str = "semantic",
    note: str = "",
    user_limit: bool | None = None,
) -> RemoteStep:
    if user_limit is None:
        user_limit = compilation.query.limit is not UNSET
    return RemoteStep(
        build_envelope(compilation, options.envelope),
        compilation,
        tier=compilation.tier,
        # A raw-SQL job or a stored query names itself; only a compiled query is "semantic".
        label=compilation.role if label == "semantic" else label,
        note=note,
        user_limit=user_limit,
    )


# --------------------------------------------------------------------------------------
# The recursion
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Frame:
    """The split of one subtree: the step that produces it and what its columns are called."""

    step: Step
    #: The result's column names, in order (aliases already applied).
    columns: tuple[str, ...]
    #: ``wire name -> output name`` for everything below, so a local expression written against
    #: a wire name can be rewritten to the column that actually arrives.
    renames: Mapping[str, str]

    def then(
        self,
        op: LocalOp,
        columns: tuple[str, ...],
        renames: Mapping[str, str] | None = None,
    ) -> _Frame:
        """Stack one more local operator on top of this frame."""
        return _Frame(
            LocalStep(op, (self.step,)), columns, self.renames if renames is None else renames
        )


class _Splitter:
    """One split in progress: the options, plus the remote steps discovered so far."""

    def __init__(self, options: SplitOptions) -> None:
        self.options = options
        self.steps: list[RemoteStep] = []

    # -- the invariant ----------------------------------------------------------------

    def split(self, node: nodes.PlanNode, required: frozenset[str]) -> _Frame:
        """Split ``node``, making sure the result also carries every name in ``required``."""
        widened = _widen(node, required)
        if widened is not None:
            compilation = try_semantic(widened)
            if compilation is not None:
                return self.remote(compilation)
            # Tier 2 sits exactly here: between the governed query and the local engine, tried at
            # every node so the remote frontier stays maximal (docs/SQLTIER.md §4).  Only over a
            # subtree that *could* be widened, for the same reason tier 1 is: a step that quietly
            # dropped a column an operator above needs would fail at runtime instead of naming it.
            sql = self.sql(widened)
            if sql is not None:
                return sql
        return self._local(node, required)

    def sql(self, node: nodes.PlanNode) -> _Frame | None:
        """One tier-2 attempt: the subtree as a single OmniSQL statement, or ``None``."""
        if self.options.disable_sql:
            return None
        compilation = try_sql(node, options=self.options)
        if compilation is None:
            return None
        # `label` falls out of `compilation.role`, which is "sql" for a statement omniframes
        # wrote; `user_limit` falls out of the query's own limit trichotomy, exactly as tier 1.
        return self.remote(compilation)

    def remote(
        self,
        compilation: SemanticCompilation,
        *,
        label: str = "semantic",
        note: str = "",
        user_limit: bool | None = None,
        renames: Mapping[str, str] | None = None,
    ) -> _Frame:
        """Record one governed query as a step of the DAG and hand back its frame."""
        step = _step(compilation, self.options, label=label, note=note, user_limit=user_limit)
        self.steps.append(step)
        return _Frame(
            step,
            compilation.columns,
            dict(compilation.aliases) if renames is None else renames,
        )

    # -- per-node rules ---------------------------------------------------------------

    def _local(self, node: nodes.PlanNode, required: frozenset[str]) -> _Frame:
        if isinstance(node, nodes.Filter):
            return self._filter(node, required)
        if isinstance(node, nodes.Project):
            return self._project(node, required)
        if isinstance(node, nodes.Aggregate):
            return self._aggregate(node, required)
        if isinstance(node, nodes.Sort):
            return self._sort(node, required)
        if isinstance(node, nodes.Limit):
            return self._limit(node, required)
        if isinstance(node, nodes.WithColumn):
            return self._with_column(node, required)
        if isinstance(node, nodes.MapPandas):
            return self._map_pandas(node, required)
        if isinstance(node, nodes.Join):
            return self._join(node, required)
        if isinstance(node, nodes.Union):
            return self._union(node, required)
        raise CompileError(f"not yet supported: {_reason(node)}")

    def _filter(self, node: nodes.Filter, required: frozenset[str]) -> _Frame:
        available = _names(node.child)
        _refuse_unresolvable_aggregation(node.predicate, available)
        child = self.split(node.child, required | _referenced(node.predicate, available))
        predicate = _rebind(node.predicate, child.renames)
        return child.then(LocalFilter(predicate), child.columns)

    def _project(self, node: nodes.Project, required: frozenset[str]) -> _Frame:
        if any(isinstance(column.expr, AdHocAgg) for column in node.columns):
            # `select(dim, F.measure(...), F.count_distinct(...))` is the same query as the
            # equivalent `group_by(dim).agg(...)` — decompose it the same way (docs/HYBRID.md §2.1;
            # DESIGN §2's "selection is the group-by" cuts both ways).
            keys = tuple(c for c in node.columns if not isinstance(c.expr, MeasureRef | AdHocAgg))
            aggs = tuple(c for c in node.columns if isinstance(c.expr, MeasureRef | AdHocAgg))
            return self._aggregate(nodes.Aggregate(node.child, keys, aggs), required)

        available = _names(node.child)
        for column in node.columns:
            if not isinstance(column.expr, FieldRef | MeasureRef):
                # Computing *with* an aggregate only makes sense above the aggregate that
                # produced it; say so here rather than at runtime.
                _refuse_unresolvable_aggregation(column.expr, available)
        needed = frozenset[str]().union(
            *(_referenced(column.expr, available) for column in node.columns)
        )
        child = self.split(node.child, required | needed)

        # A field selected more than once arrives as ONE column whatever the query said (tier 1
        # by its `fields` list, tier 2 by the OmniSQL parser's bare-ref dedup — docs/SQLTIER.md
        # §3.2), and a rename map keyed by the source name cannot say "this copy, not that one".
        # Every copy of such a field is therefore materialized here, by name.
        sources = [
            _resolve(column.expr, child.renames)
            for column in node.columns
            if isinstance(column.expr, FieldRef | MeasureRef)
        ]
        copied = {name for name in sources if sources.count(name) > 1}

        frame = child
        names: list[str] = []
        renames: dict[str, str] = {}
        for column in node.columns:
            output = column.alias_name or column_key(column.expr)
            if isinstance(column.expr, FieldRef | MeasureRef):
                source = _resolve(column.expr, child.renames)
                if source not in copied:
                    names.append(source)
                    if source != output:
                        renames[source] = output
                    continue
                if output != source:
                    frame = frame.then(
                        LocalWithColumn(output, FieldRef(source)),
                        frame.columns if output in frame.columns else (*frame.columns, output),
                    )
                names.append(output)
            else:
                frame = frame.then(
                    LocalWithColumn(output, _rebind(column.expr, child.renames)),
                    (*frame.columns, output),
                )
                names.append(output)
        # Names this projection does not mention but an operator above still needs ride along
        # and are dropped by the final projection.
        carried = sorted(name for name in required if name not in names and name in frame.columns)
        selected = (*names, *carried)
        outputs = tuple(renames.get(name, name) for name in selected)
        produced = {key: name for key, name in _column_outputs(node.columns) if key != name}
        return frame.then(LocalProject(selected, renames), outputs, produced)

    def _aggregate(self, node: nodes.Aggregate, required: frozenset[str]) -> _Frame:
        measures = tuple(column for column in node.aggs if isinstance(column.expr, MeasureRef))
        adhoc = tuple(column for column in node.aggs if isinstance(column.expr, AdHocAgg))
        if len(measures) + len(adhoc) != len(node.aggs):
            raise CompileError(f"not yet supported: {_reason(node)}")
        outputs = _column_outputs((*node.keys, *node.aggs))
        names = tuple(name for _, name in outputs)
        missing = sorted(required - set(names) - {key for key, _ in outputs})
        if missing:
            raise CompileError(
                f"{', '.join(missing)} is not available after group_by(...).agg(...): an "
                "aggregate only produces its group keys and its aggregates. Select the field "
                "before aggregating, or group by it."
            )
        if not adhoc:
            # Every aggregate is governed, so tier 1 owns this node; getting here means the
            # rows underneath it do not compile to one query.
            raise CompileError(
                "governed measures always execute remotely, and this aggregate's input does not "
                f"compile to a single governed query: {_reason(node)}"
            )

        keys = tuple(node.keys)
        for key in keys:
            if not isinstance(key.expr, FieldRef):
                raise CompileError(
                    f"{display_name(key.expr)} is not a group key omniframes can decompose; "
                    "group by dimensions (optionally at a grain)"
                )
        key_names = tuple(key.alias_name or column_key(key.expr) for key in keys)
        produced = {key: name for key, name in outputs if key != name}

        # A governed measure is a legal select item on the OmniSQL path and mixes freely with
        # ad-hoc aggregates in one statement (CONTRACT_NOTES §3.6), so the whole node — keys,
        # measures and ad-hoc aggregates together — is tried as ONE tier-2 job first.  When it
        # compiles there is nothing to decompose: no measure step, no align-join, and no raw
        # scan for `decomposition_row_cap` to cap (docs/SQLTIER.md §4).
        whole = self.sql(node)
        if whole is not None:
            return _Frame(whole.step, whole.columns, produced)

        # The governed half goes first so that `explain()` numbers the steps the way the
        # decomposition reads: step 1 computes the measures, step 2 scans the rows behind the
        # ad-hoc aggregates (docs/HYBRID.md §2.1).
        remote: _Frame | None = None
        if measures:
            governed_plan = nodes.Aggregate(node.child, keys, measures)
            governed = try_semantic(governed_plan)
            if governed is None:
                raise CompileError(
                    "governed measures always execute remotely, and this aggregate's input does "
                    f"not compile to a single governed query: {_reason(governed_plan)}"
                )
            remote = self.remote(governed)

        agg_names = tuple(column.alias_name or column_key(column.expr) for column in adhoc)
        computed = self._adhoc_half(node.child, keys, adhoc, key_names, agg_names, produced)

        if remote is None:
            return computed
        return _Frame(
            LocalStep(AlignJoin(key_names), (remote.step, computed.step)),
            (*remote.columns, *agg_names),
            produced,
        )

    def _adhoc_half(
        self,
        child: nodes.PlanNode,
        keys: Sequence[Column],
        adhoc: Sequence[Column],
        key_names: tuple[str, ...],
        agg_names: tuple[str, ...],
        produced: Mapping[str, str],
    ) -> _Frame:
        """The ad-hoc half alone: one tier-2 ``GROUP BY`` when expressible, else a raw scan.

        Reached only after the whole node refused tier 2, so the measures are running as their
        own governed query and this half has to be produced beside them.  Even here it is worth
        one more tier-2 attempt: only the *scan* cost ever mattered (docs/HYBRID.md §2.1), and a
        warehouse-side GROUP BY removes it entirely even when the measures could not ride along.
        """
        sql = self.sql(nodes.Aggregate(child, tuple(keys), tuple(adhoc)))
        if sql is not None:
            return _Frame(sql.step, sql.columns, produced)

        raw = self._raw_scan(child, keys, adhoc)
        local_aggs = tuple(
            LocalAgg(
                name=name,
                fn=_agg_expr(column).fn,
                operand=raw.renames.get(
                    wire_name(_agg_expr(column).operand), wire_name(_agg_expr(column).operand)
                ),
                distinct=_agg_expr(column).distinct,
            )
            for name, column in zip(agg_names, adhoc, strict=True)
        )
        return raw.then(LocalAggregate(key_names, local_aggs), (*key_names, *agg_names), produced)

    def _raw_scan(
        self,
        child: nodes.PlanNode,
        keys: Sequence[Column],
        adhoc: Sequence[Column],
    ) -> _Frame:
        """The unlimited scan of raw rows the local aggregation consumes (docs/HYBRID.md §2.1)."""
        columns: list[Column] = list(keys)
        seen = {column_key(key.expr) for key in keys}
        for column in adhoc:
            operand = _agg_expr(column).operand
            key = column_key(operand)
            if key not in seen:
                seen.add(key)
                columns.append(Column(operand))

        projection: nodes.PlanNode = nodes.Project(child, tuple(columns))
        cap = self.options.decomposition_row_cap
        # The user already pinned a limit under the aggregate; that limit is part of the
        # question they asked, so it stays rather than being widened to everything — and the
        # cap has nothing left to cap.
        user_limited = _has_limit(child)
        note = "" if user_limited else "(unlimited)"
        plan: nodes.PlanNode = projection if user_limited else nodes.Limit(projection, cap)
        if cap is not None and not user_limited:
            note = f"(capped at {cap})"

        compilation = try_semantic(plan)
        if compilation is not None:
            return self.remote(compilation, label="raw scan", note=note, user_limit=False)

        # The rows behind the aggregate need local work of their own: an aggregate written above
        # a `.limit()`, but also a UDF, a raw-SQL scan or a join.  The scan cannot carry the cap
        # on the wire, so it is applied here — marked, because a `LocalLimit` the executor knows
        # nothing about would trim an aggregate's input in silence, and a capped aggregate is a
        # wrong answer rather than a short page (docs/HYBRID.md §2.1).
        if cap is None or user_limited:
            return self.split(plan, frozenset())
        frame = self.split(projection, frozenset())
        return frame.then(LocalLimit(cap, decomposition_cap=True), frame.columns)

    def _sort(self, node: nodes.Sort, required: frozenset[str]) -> _Frame:
        available = _names(node.child)
        needed = frozenset[str]().union(*(_referenced(key.expr, available) for key in node.keys))
        child = self.split(node.child, required | needed)
        keys = tuple((_resolve(key.expr, child.renames), key.descending) for key in node.keys)
        return child.then(LocalSort(keys), child.columns)

    def _limit(self, node: nodes.Limit, required: frozenset[str]) -> _Frame:
        child = self.split(node.child, required)
        # UNSET means "the user only asked for an offset": the fetch limit that the sentinel
        # stands for was already applied by the query below, so nothing more is capped here.
        rows = node.n if isinstance(node.n, int) else None
        return child.then(LocalLimit(rows, node.offset), child.columns)

    def _with_column(self, node: nodes.WithColumn, required: frozenset[str]) -> _Frame:
        available = _names(node.child)
        _refuse_unresolvable_aggregation(node.expr, available)
        child = self.split(node.child, required | _referenced(node.expr, available))
        columns = child.columns if node.name in child.columns else (*child.columns, node.name)
        return child.then(LocalWithColumn(node.name, _rebind(node.expr, child.renames)), columns)

    def _join(self, node: nodes.Join, required: frozenset[str]) -> _Frame:
        """Two sub-plans, split independently, combined here with SQL join semantics.

        Neither side constrains the other: each is a whole plan in its own right, so a join
        across two models — or across a governed topic and a raw-SQL job — is just two remote
        steps and one local operator.  Nothing about the join can ride the wire: the query API
        takes one query.
        """
        if node.how is nodes.JoinHow.CROSS:
            raise CompileError(
                "cross joins are not supported: every row of one frame against every row of the "
                "other is a result size nobody meant to ask for, and omniframes joins on column "
                "names. Use inner, left, right or outer with an `on`."
            )
        keys = _join_keys(node)
        left = self.split(node.left, frozenset())
        right = self.split(node.right, frozenset())
        _check_join_keys(keys, "left", left.columns)
        _check_join_keys(keys, "right", right.columns)
        overlap = sorted((set(left.columns) & set(right.columns)) - set(keys))
        if overlap:
            raise CompileError(join_collision_message(overlap))

        columns = _join_columns(keys, left.columns, right.columns)
        if left.columns and right.columns:
            # Only when *both* sides know their columns is this list complete; an opaque side
            # (raw SQL) is checked by the operator, against the table that actually arrives.
            _refuse_missing(
                required,
                columns,
                "join(...)",
                "a join produces only the columns of the two frames it joined",
            )
        return _Frame(
            LocalStep(LocalJoin(keys, node.how), (left.step, right.step)),
            columns,
            {**dict(left.renames), **dict(right.renames)},
        )

    def _union(self, node: nodes.Union, required: frozenset[str]) -> _Frame:
        left = self.split(node.left, frozenset())
        right = self.split(node.right, frozenset())
        if left.columns and right.columns:
            # A raw-SQL side knows its columns only once it runs, so the operator repeats this
            # check against the real tables; here it is made as early as it can be made.
            if left.columns != right.columns:
                raise CompileError(
                    "union() stacks two frames by position, so both must produce the same "
                    f"columns in the same order: the left frame has [{', '.join(left.columns)}], "
                    f"the right one has [{', '.join(right.columns)}]. Select the columns apart, "
                    "or alias them so the names line up."
                )
            _refuse_missing(
                required,
                left.columns,
                "union(...)",
                "a union produces only the columns both frames carry",
            )
        return _Frame(
            LocalStep(LocalUnion(), (left.step, right.step)),
            left.columns,
            {**dict(left.renames), **dict(right.renames)},
        )

    def _map_pandas(self, node: nodes.MapPandas, required: frozenset[str]) -> _Frame:
        if node.schema_hint is None:
            if required:
                raise CompileError(
                    f"{', '.join(sorted(required))} cannot be read out of map_pandas(): a Python "
                    "function's output schema cannot be planned, so nothing above it may name a "
                    "column. Pass schema_hint=OmniSchema(...) to declare what it returns."
                )
            columns: tuple[str, ...] = ()
        else:
            columns = node.schema_hint.names
            missing = sorted(required - set(columns))
            if missing:
                raise CompileError(
                    f"{', '.join(missing)} is not in the schema_hint map_pandas() declared "
                    f"({', '.join(columns) or '(no fields)'})"
                )
        child = self.split(node.child, frozenset())
        return _Frame(
            LocalStep(LocalMapPandas(node.fn, node.schema_hint), (child.step,)), columns, {}
        )


# --------------------------------------------------------------------------------------
# Column widening (docs/HYBRID.md §2.2)
# --------------------------------------------------------------------------------------


def _join_keys(node: nodes.Join) -> tuple[str, ...]:
    """The equi-join columns, or a :class:`CompileError` naming what was passed instead."""
    on = node.on
    if isinstance(on, Expr):
        raise CompileError(
            "join(on=...) takes a column name or a list of column names that both frames "
            "produce; an arbitrary join expression has no equivalent here (compute the key as "
            "a column on each side and join on that)."
        )
    keys = tuple(on)
    if not keys:
        raise CompileError("join() needs at least one column to join on")
    wrong = [key for key in keys if not isinstance(key, str) or not key]
    if wrong:
        raise CompileError(f"join(on=...) takes non-empty column names; got {wrong!r}")
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise CompileError(f"join(on=...) repeats {', '.join(duplicates)}; name each key once")
    return keys


def _check_join_keys(keys: Sequence[str], side: str, columns: Sequence[str]) -> None:
    if not columns:
        # This side's columns are not knowable without running it (a raw-SQL job, or a
        # `map_pandas()` with no schema hint).  The operator itself names a missing key at
        # runtime, against the table that actually arrived — a guess here could only be wrong.
        return
    missing = [key for key in keys if key not in columns]
    if not missing:
        return
    available = ", ".join(columns) or "(none)"
    raise CompileError(
        f"{', '.join(missing)} is not a column of the {side} frame, so the join has nothing to "
        f"match on. That frame produces: {available}. (Join keys are matched against each "
        "side's OUTPUT columns, so an alias is named by its alias.)"
    )


def _join_columns(
    keys: Sequence[str], left: Sequence[str], right: Sequence[str]
) -> tuple[str, ...]:
    """The join's output order: the keys once, then the left's own columns, then the right's."""
    return (
        *keys,
        *(name for name in left if name not in keys),
        *(name for name in right if name not in keys),
    )


def _refuse_missing(
    required: frozenset[str], available: Sequence[str], operation: str, because: str
) -> None:
    missing = sorted(required - set(available))
    if not missing:
        return
    raise CompileError(
        f"{', '.join(missing)} is not available after {operation}: {because}. It produced: "
        f"{', '.join(available) or '(none)'}. Select the column before combining the frames."
    )


def _is_opaque(source: nodes.ScanSource) -> bool:
    """Whether the *server* decides this scan's columns (raw SQL, or a stored query)."""
    return isinstance(source, nodes.SqlScan | nodes.SavedQueryScan)


def _widen(node: nodes.PlanNode, required: frozenset[str]) -> nodes.PlanNode | None:
    """``node`` with ``required`` added to the projection underneath it, or ``None``.

    ``None`` means "this subtree cannot be made to carry those columns" — never an error on its
    own, because the local branch may still be able to satisfy them (and will say so by name if
    it cannot).
    """
    if not required:
        return node

    chain: list[nodes.PlanNode] = []
    current = node
    while True:
        if isinstance(current, nodes.Project):
            available = _projected_names(current.columns)
            missing = sorted(name for name in required if name not in available)
            if not missing:
                return node
            grown = nodes.Project(current.child, (*current.columns, *_columns_for(missing)))
            return _rebuild(chain, grown)
        if isinstance(current, nodes.Aggregate):
            available = _projected_names((*current.keys, *current.aggs))
            return node if all(name in available for name in required) else None
        if isinstance(current, nodes.Scan):
            if _is_opaque(current.source):
                # A raw-SQL job or a stored query already carries whatever it carries; adding a
                # projection would either be ignored or change the query the user handed over.
                # The columns are simply expected to be there, and named if they are not.
                return node
            # Nothing projects yet: the projection belongs directly above the scan, so a limit
            # or a filter written above it still applies to the widened query.
            return _rebuild(chain, nodes.Project(current, _columns_for(sorted(required))))
        if isinstance(current, _PASS_THROUGH):
            chain.append(current)
            current = current.child
            continue
        return None


def _projected_names(columns: Sequence[Column]) -> set[str]:
    names = {column_key(column.expr) for column in columns}
    names.update(column.alias_name for column in columns if column.alias_name is not None)
    return names


def _columns_for(names: Sequence[str]) -> tuple[Column, ...]:
    columns: list[Column] = []
    for name in names:
        base, grain = split_grain(name)
        columns.append(Column(FieldRef(base, grain)))
    return tuple(columns)


def _rebuild(chain: Sequence[nodes.PlanNode], bottom: nodes.PlanNode) -> nodes.PlanNode:
    """Re-apply the pass-through nodes the widening walk descended past."""
    rebuilt = bottom
    for node in reversed(chain):
        rebuilt = node.with_children((rebuilt,))
    return rebuilt


def _has_limit(node: nodes.PlanNode) -> bool:
    current = node
    while True:
        if isinstance(current, nodes.Limit):
            return True
        children = current.children
        if len(children) != 1:
            return False
        current = children[0]


# --------------------------------------------------------------------------------------
# What a plan node produces
# --------------------------------------------------------------------------------------


def _outputs(node: nodes.PlanNode) -> tuple[tuple[str, str], ...] | None:
    """``(un-aliased key, output name)`` per column, or ``None`` when the shape is unknowable."""
    if isinstance(node, nodes.Project):
        return _column_outputs(node.columns)
    if isinstance(node, nodes.Aggregate):
        return _column_outputs((*node.keys, *node.aggs))
    if isinstance(node, nodes.Filter | nodes.Sort | nodes.Limit):
        return _outputs(node.child)
    if isinstance(node, nodes.WithColumn):
        base = _outputs(node.child)
        if base is None:
            return None
        if any(name == node.name for _, name in base):
            return base
        return (*base, (node.name, node.name))
    if isinstance(node, nodes.MapPandas):
        if node.schema_hint is None:
            return None
        return tuple((name, name) for name in node.schema_hint.names)
    if isinstance(node, nodes.Join):
        return _join_outputs(node)
    if isinstance(node, nodes.Union):
        return _outputs(node.left)
    if isinstance(node, nodes.Scan) and isinstance(node.source, nodes.SavedQueryScan):
        # A stored query is opaque to the compiler but not to the reader: a semantic query's
        # result columns are its `fields`, verbatim, brackets included.
        fields = node.source.query.get("fields")
        if isinstance(fields, Sequence) and not isinstance(fields, str):
            return tuple((str(name), str(name)) for name in fields)
    return None


def _join_outputs(node: nodes.Join) -> tuple[tuple[str, str], ...] | None:
    left = _outputs(node.left)
    right = _outputs(node.right)
    if left is None or right is None or isinstance(node.on, Expr):
        return None
    keys = tuple(node.on)
    names = _join_columns(keys, [name for _, name in left], [name for _, name in right])
    return tuple((name, name) for name in names)


def _column_outputs(columns: Sequence[Column]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (column_key(column.expr), column.alias_name or column_key(column.expr))
        for column in columns
    )


def _names(node: nodes.PlanNode) -> tuple[str, ...]:
    """Every name a local expression may legally reference from ``node``: keys and aliases."""
    outputs = _outputs(node)
    if outputs is None:
        return ()
    names: list[str] = []
    for key, output in outputs:
        names.append(output)
        if key != output:
            names.append(key)
    return tuple(names)


# --------------------------------------------------------------------------------------
# Expressions inside local operators
# --------------------------------------------------------------------------------------


def _referenced(expr: Expr, available: Sequence[str]) -> frozenset[str]:
    """The bare fields a local expression needs the query below it to fetch (§2.2).

    Names the child already produces are not requested again — that is what keeps a filter
    written against an alias from widening the query with a field called ``"revenue"``.
    """
    if isinstance(expr, MeasureRef | AdHocAgg):
        # An aggregate reference is a whole column, not a request for its operand: the operand
        # was consumed by the aggregate that already ran.
        return frozenset()
    if isinstance(expr, FieldRef):
        wire = wire_name(expr)
        if wire in available or expr.name in available:
            return frozenset()
        return frozenset({wire})
    names: set[str] = set()
    for child in expr.children:
        names |= _referenced(child, available)
    return frozenset(names)


def _refuse_unresolvable_aggregation(expr: Expr, available: Sequence[str]) -> None:
    """An aggregate inside a predicate is only a column reference, never a computation."""
    for sub in _walk(expr):
        if isinstance(sub, AdHocAgg) and display_name(sub) not in available:
            raise CompileError(f"not yet supported: {adhoc_filter_reason(sub)}")
        if isinstance(sub, MeasureRef) and sub.name not in available:
            raise CompileError(
                f"{sub.name} is a governed measure: it only exists in a result a remote step "
                "computed, and the local engine never emulates one. Select it (or aggregate it) "
                "before referring to it here."
            )


def _walk(expr: Expr) -> list[Expr]:
    found = [expr]
    for child in expr.children:
        found.extend(_walk(child))
    return found


def _resolve(expr: Expr, renames: Mapping[str, str]) -> str:
    """The column name an already-materialized result carries for this reference."""
    if isinstance(expr, FieldRef | MeasureRef):
        wire = wire_name(expr)
        return renames.get(wire, renames.get(expr.name, wire))
    key = display_name(expr)
    return renames.get(key, key)


def _rebind(expr: Expr, renames: Mapping[str, str]) -> Expr:
    """Rewrite field/measure/aggregate references to the names the arriving table uses.

    Aliases are applied by ``normalize`` before a local operator sees a table, so an expression
    written against a wire name has to follow.  An aggregation reference becomes a plain field
    reference: above the aggregate that produced it, ``count_distinct(users.id)`` *is* a column.
    """
    if isinstance(expr, FieldRef | MeasureRef | AdHocAgg):
        return FieldRef(_resolve(expr, renames))
    if isinstance(expr, Comparison):
        return Comparison(expr.op, _rebind(expr.left, renames), _rebind(expr.right, renames))
    if isinstance(expr, Arithmetic):
        return Arithmetic(expr.op, _rebind(expr.left, renames), _rebind(expr.right, renames))
    if isinstance(expr, BooleanOp):
        return BooleanOp(expr.op, tuple(_rebind(o, renames) for o in expr.operands))
    if isinstance(expr, Not):
        return Not(_rebind(expr.operand, renames))
    if isinstance(expr, IsNull):
        return IsNull(_rebind(expr.operand, renames))
    if isinstance(expr, IsIn):
        return IsIn(_rebind(expr.operand, renames), expr.values)
    if isinstance(expr, StringPredicate):
        return StringPredicate(
            expr.kind, _rebind(expr.operand, renames), expr.value, expr.case_insensitive
        )
    if isinstance(expr, Between):
        return Between(_rebind(expr.operand, renames), expr.low, expr.high)
    if isinstance(expr, Udf):
        return replace(expr, operands=tuple(_rebind(o, renames) for o in expr.operands))
    if isinstance(expr, SortKey):
        return SortKey(_rebind(expr.expr, renames), expr.descending)
    return expr


def _agg_expr(column: Column) -> AdHocAgg:
    expr = column.expr
    if not isinstance(expr, AdHocAgg):  # pragma: no cover - guarded by the caller
        raise CompileError(f"{display_name(expr)} is not an ad-hoc aggregation")
    return expr


def _reason(node: nodes.PlanNode) -> str:
    """Tier 1's own explanation for refusing ``node`` — the message users actually read."""
    try:
        compile_semantic(node)
    except CannotCompile as exc:
        return exc.reason
    return "this plan needs an operation no tier implements yet"
