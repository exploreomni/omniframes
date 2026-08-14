"""Running an :class:`~omniframes.compile.semantic.ExecutionPlan` (docs/HYBRID.md §1).

The executor knows two things: how to walk the DAG, and when a remote step came back full.
It does **not** know what a session is — the caller passes a ``run_remote`` callable that turns
one :class:`~omniframes.compile.semantic.RemoteStep` into a normalized Arrow table.  That makes
the whole hybrid engine testable from a dictionary of canned tables, and it is the seam an
in-product broker plugs into later.

Each step is evaluated once and memoized by identity, so a step feeding two operators (or the
same query reached twice) runs once.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pyarrow as pa

from omniframes.compile.local import LocalLimit, run_local_op
from omniframes.compile.semantic import (
    CannotCompile,
    ExecutionPlan,
    LocalStep,
    RemoteStep,
    Step,
)
from omniframes.errors import CompileError, TruncationWarning

__all__ = ["execute"]

#: Turns one remote step into the table it produced (normalized, aliases applied).
RunRemote = Callable[[RemoteStep], pa.Table]


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
