"""Running an :class:`~omniframes.compile.semantic.ExecutionPlan` (docs/HYBRID.md §1).

The executor knows three things: how to walk the DAG, when a remote step came back full, and
which job errors are omniframes' own SQL rather than the user's (:func:`remote_errors`).  It
does **not** know what a session is — the caller passes a ``run_remote`` callable that turns one
:class:`~omniframes.compile.semantic.RemoteStep` into a normalized Arrow table.  That makes the
whole hybrid engine testable from a dictionary of canned tables, and it is the seam an
in-product broker plugs into later.

Each step is evaluated once and memoized by identity, so a step feeding two operators (or the
same query reached twice) runs once.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

import pyarrow as pa

from omniframes.compile.local import LocalLimit, run_local_op
from omniframes.compile.semantic import (
    CannotCompile,
    ExecutionPlan,
    LocalStep,
    RemoteStep,
    Step,
)
from omniframes.errors import CompileError, QueryError, TruncationWarning

__all__ = ["execute", "remote_errors"]

#: Turns one remote step into the table it produced (normalized, aliases applied).
RunRemote = Callable[[RemoteStep], pa.Table]

#: The server's own prefix when an OmniSQL statement names something the model does not bind
#: (CONTRACT_NOTES §3.6).  Matched verbatim: it is the only signal that separates "omniframes
#: emitted a statement this model cannot bind" from every other job failure.
_SUBSTITUTION_ERROR: Final = "Could not substitute Omni SQL"

#: What the server says is missing, in either of its two spellings (CONTRACT_NOTES §3.6):
#: ``No such view "x"`` for a FROM ref, ``No such field "v.f"`` for an expression ref.
_MISSING: Final = re.compile(r'No such (view|field)\s+"([^"]*)"')


def execute(
    execution: ExecutionPlan,
    run_remote: RunRemote,
    *,
    warn: bool = True,
    stacklevel: int = 4,
) -> pa.Table:
    """Run ``execution`` and return the user-facing table.

    Args:
        execution: the compiled plan.
        run_remote: runs one remote step; called at most once per step.
        warn: when false, suppress the truncation warning for a limit the *user* asked for —
            what ``show()``/``first()`` want.  An intermediate scan always warns: nobody asked
            for its limit, and a full one means the local result may simply be wrong.
        stacklevel: passed through to :func:`warnings.warn` so the warning points at the action.
    """
    root = execution.root
    if root is None:
        with remote_errors(execution.remote):
            return run_remote(execution.remote)

    numbers = {id(step): index for index, step in enumerate(execution.steps, start=1)}
    results: dict[int, pa.Table] = {}
    # Collected during the walk and emitted from here: the recursion's depth varies with the
    # DAG, and a warning that points at a random frame of the executor helps nobody.
    truncated: list[str] = []

    def evaluate(step: Step) -> pa.Table:
        key = id(step)
        cached = results.get(key)
        if cached is not None:
            return cached
        if isinstance(step, RemoteStep):
            with remote_errors(step):
                table = run_remote(step)
            message = _truncation(step, table, numbers.get(key, 0), warn=warn)
            if message is not None:
                truncated.append(message)
        elif isinstance(step, LocalStep):
            table = _run(step, tuple(evaluate(source) for source in step.inputs))
            local_message = _local_truncation(step, table)
            if local_message is not None:
                truncated.append(local_message)
        else:  # pragma: no cover - the union is closed
            raise CompileError(f"{type(step).__name__} is not an execution step")
        results[key] = table
        return table

    result = evaluate(root)
    for message in truncated:
        warnings.warn(message, TruncationWarning, stacklevel=stacklevel)
    return result


@contextmanager
def remote_errors(step: RemoteStep) -> Iterator[None]:
    """Run one remote step's request, re-reading an OmniSQL binding failure as omniframes' own.

    A ``Could not substitute Omni SQL`` job error on a tier-2 step is not a user error: the
    statement was written here, from a plan built against this model, so the model drifted
    between the plan and the run — or the emission is wrong (docs/SQLTIER.md §8).  The server's
    text names what it could not bind and nothing else, which reads as a mystery next to
    dataframe code that mentions no SQL at all; this says whose SQL it is, what is missing, and
    where to read the statement.  Every other job error passes through untouched, ``read.sql``
    included: that SQL is the user's own.
    """
    try:
        yield
    except QueryError as exc:
        mapped = _omnisql_error(step, exc)
        if mapped is None:
            raise
        raise mapped from exc


def _omnisql_error(step: RemoteStep, exc: QueryError) -> QueryError | None:
    statement = step.query.user_edited_sql
    if not step.query.omnisql or not statement:
        return None
    text = str(exc)
    if _SUBSTITUTION_ERROR not in text:
        return None
    match = _MISSING.search(text)
    missing = (
        f"the model has no {match.group(1)} {match.group(2)!r}"
        if match is not None
        else "the model rejected a reference in it"
    )
    return QueryError(
        f"Omni rejected the tier-2 statement omniframes generated for this frame: {missing}. "
        "The model changed under the query, or omniframes emitted a reference it should not "
        f"have — explain() prints the statement. Server said: {text}",
        error_type=exc.error_type,
        job_id=exc.job_id,
        statement=statement,
    )


def _run(step: LocalStep, inputs: tuple[pa.Table, ...]) -> pa.Table:
    try:
        return run_local_op(step.op, inputs)
    except CannotCompile as exc:
        # Something only Omni can evaluate reached the local engine (a relative date literal,
        # today).  Report it the way every other unsupported shape is reported.
        raise CompileError(f"not yet supported: {exc.reason}") from exc


def _truncation(step: RemoteStep, table: pa.Table, number: int, *, warn: bool) -> str | None:
    """The warning this step earned, if any: rows returned == the limit it applied."""
    limit = step.applied_limit
    if limit is None or table.num_rows != limit:
        return None
    if not warn and step.user_limit:
        return None
    return _truncation_message(step, number, limit)


def _local_truncation(step: LocalStep, table: pa.Table) -> str | None:
    """``decomposition_row_cap`` applied here rather than on the wire, and hit.

    The cap normally rides the raw scan's own ``limit``, where :func:`_truncation` reports it.
    When the rows under the aggregate need local work of their own (a UDF, a raw-SQL scan, a
    join) the scan cannot carry it, and the cap becomes a :class:`LocalLimit` instead — the same
    opt-in cap, trimming the same aggregate's input, so it owes the same warning.  Without it a
    capped aggregate is simply a wrong answer with no signal (docs/HYBRID.md §2.1).
    """
    op = step.op
    if not isinstance(op, LocalLimit) or not op.decomposition_cap:
        return None
    if op.n is None or table.num_rows != op.n:
        return None
    return (
        f"the local aggregate's input was cut to exactly its {op.n}-row decomposition cap, so "
        "the aggregate is computed over part of the data — set decomposition_row_cap(None), or "
        "raise it"
    )


def _truncation_message(step: RemoteStep, number: int, limit: int) -> str:
    if step.label == "raw scan":
        return (
            f"remote step {number} (raw scan) returned exactly its {limit}-row limit; the local "
            "aggregate may be wrong — set decomposition_row_cap(None) or use .limit()"
        )
    return (
        f"remote step {number} ({step.label}) returned exactly its {limit}-row limit, which is "
        "where rows go missing. Raise .limit(n), use .limit(None) for everything, or narrow the "
        "query."
    )
