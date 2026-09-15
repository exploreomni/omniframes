"""The compiler: logical plan in, executable steps out.

* :mod:`omniframes.compile.querymodel` — the typed wire contract (CONTRACT_NOTES §2/§3).
* :mod:`omniframes.compile.semantic` — tier 1, the governed semantic query, plus the
  :class:`ExecutionPlan` an action runs.
* :mod:`omniframes.compile.sqlgen` — tier 2, one governed OmniSQL statement (docs/SQLTIER.md).
* :mod:`omniframes.compile.splitter` — the pushdown frontier: one plan in, a DAG of remote
  queries and local operators out (docs/HYBRID.md §2).  This is what a DataFrame action calls,
  and the seam the three tiers are tried at, in order.
* :mod:`omniframes.compile.local` — tier 3, the pyarrow operator interpreter (§3).
* :mod:`omniframes.compile.executor` — walks the DAG, given something that runs one query.
* :mod:`omniframes.compile.explain` — how a plan renders for ``df.explain()``.

:func:`compile_plan` remains the tier-1-only entry point — it refuses anything that needs
another tier rather than splitting it.
"""

from __future__ import annotations

from omniframes.compile.executor import execute, remote_errors
from omniframes.compile.explain import describe_filter, describe_filters, explain_text
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
    eval_expr,
    promote_type,
    run_local_op,
)
from omniframes.compile.semantic import (
    DEFAULT_FETCH_LIMIT,
    CannotCompile,
    EnvelopeOptions,
    ExecutionPlan,
    LocalStep,
    RemoteStep,
    SemanticCompilation,
    Step,
    alias_map,
    build_envelope,
    compile_plan,
    compile_semantic,
    display_name,
    predicate_to_filters,
    try_semantic,
    wire_name,
)
from omniframes.compile.splitter import SplitOptions, split
from omniframes.compile.sqlgen import compile_sql, render_expr, try_sql

__all__ = [
    "DEFAULT_FETCH_LIMIT",
    "AlignJoin",
    "CannotCompile",
    "EnvelopeOptions",
    "ExecutionPlan",
    "LocalAgg",
    "LocalAggregate",
    "LocalFilter",
    "LocalJoin",
    "LocalLimit",
    "LocalMapPandas",
    "LocalOp",
    "LocalProject",
    "LocalSort",
    "LocalStep",
    "LocalUnion",
    "LocalWithColumn",
    "RemoteStep",
    "SemanticCompilation",
    "SplitOptions",
    "Step",
    "alias_map",
    "build_envelope",
    "compile_plan",
    "compile_semantic",
    "compile_sql",
    "describe_filter",
    "describe_filters",
    "display_name",
    "eval_expr",
    "execute",
    "explain_text",
    "predicate_to_filters",
    "promote_type",
    "remote_errors",
    "render_expr",
    "run_local_op",
    "split",
    "try_semantic",
    "try_sql",
    "wire_name",
]
