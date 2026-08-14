"""The logical plan: immutable relational nodes plus generic walking utilities.

See :mod:`omniframes.plan.nodes` for the node set (docs/INTERNALS.md §2) and
:mod:`omniframes.plan.visitor` for ``walk``/``transform``.
"""

from __future__ import annotations

from omniframes.plan.nodes import (
    Aggregate,
    Filter,
    Join,
    JoinHow,
    Limit,
    MapPandas,
    PlanNode,
    Project,
    SavedQueryScan,
    Scan,
    ScanSource,
    Sort,
    SqlScan,
    TopicScan,
    Union,
    ViewScan,
    WithColumn,
)
from omniframes.plan.visitor import collect, transform, transform_expr, walk, walk_expr

__all__ = [
    "Aggregate",
    "Filter",
    "Join",
    "JoinHow",
    "Limit",
    "MapPandas",
    "PlanNode",
    "Project",
    "SavedQueryScan",
    "Scan",
    "ScanSource",
    "Sort",
    "SqlScan",
    "TopicScan",
    "Union",
    "ViewScan",
    "WithColumn",
    "collect",
    "transform",
    "transform_expr",
    "walk",
    "walk_expr",
]
