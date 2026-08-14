"""The logical plan (docs/INTERNALS.md §2).

Immutable relational nodes.  A :class:`~omniframes.dataframe.DataFrame` is a thin wrapper around
one of these; every transformation returns a new node, and nothing here knows how a node is
executed — that is the compiler's job (:mod:`omniframes.compile`).

Every node exposes ``children`` and ``with_children`` so the visitors in
:mod:`omniframes.plan.visitor` can walk and rebuild a plan without knowing the node types.

The full node set is declared here even though today's DataFrame only *builds* Scan / Project /
Filter / Aggregate / Sort / Limit: later milestones add operations, not shapes, so the module
does not have to be reshaped underneath the compiler.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from omniframes.column import Column, Expr, SortKey
from omniframes.compile.querymodel import UNSET, Unset
from omniframes.errors import CompileError
from omniframes.types import OmniSchema

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
]


class JoinHow(enum.Enum):
    """Join kinds a local (tier-3) join supports (M4)."""

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    CROSS = "cross"


# --------------------------------------------------------------------------------------
# Scan sources
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanSource:
    """Where rows come from. One of the four concrete sources below."""

    @property
    def label(self) -> str:
        """A short human description used by ``explain()``."""
        raise NotImplementedError


@dataclass(frozen=True)
class TopicScan(ScanSource):
    """A governed topic: the query carries ``join_paths_from_topic_name`` (CONTRACT_NOTES §3)."""

    model_name: str
    model_id: str
    topic: str
    base_view: str = ""

    @property
    def label(self) -> str:
        return f"topic: {self.topic}   model: {self.model_name}"


@dataclass(frozen=True)
class ViewScan(ScanSource):
    """A bare view: the query carries ``table`` and no topic. Needs ``QUERY_FULL_MODEL``."""

    model_name: str
    model_id: str
    view: str

    @property
    def label(self) -> str:
        return f"view: {self.view}   model: {self.model_name}"


@dataclass(frozen=True)
class SqlScan(ScanSource):
    """A raw-SQL job (``userEditedSQL`` + ``rewriteSql: false``) — M4.

    The SQL is opaque to omniframes: the server decides what columns come back, so nothing can
    be pushed *into* it and every operation written on top of it runs in the local engine.
    """

    model_id: str
    sql: str
    model_name: str = ""

    @property
    def label(self) -> str:
        suffix = f"   model: {self.model_name}" if self.model_name else ""
        return f"sql: <raw SQL job>{suffix}"


@dataclass(frozen=True)
class SavedQueryScan(ScanSource):
    """A stored query, hydrated at read time and sent back **verbatim** — M4.

    Two endpoints hand out a query object omniframes did not write (CONTRACT_NOTES §4):
    ``GET /documents/{id}/queries`` (``origin="saved query"``) and ``POST /ai/generate-query``
    (``origin="ask"``, where :attr:`document_id` carries the model the prompt was answered
    against).  Both blobs execute unchanged, so the node keeps the mapping as it arrived.
    """

    document_id: str
    name: str
    query: Mapping[str, Any]
    origin: str = "saved query"

    @property
    def label(self) -> str:
        if self.origin == "ask":
            return f'ask("{_shorten(self.name)}")   model: {self.document_id}'
        return f"saved query: {self.name} ({self.document_id})"


#: Longest prompt ``explain()`` prints in full before eliding the tail.
_PROMPT_LABEL_LIMIT = 60


def _shorten(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _PROMPT_LABEL_LIMIT:
        return collapsed
    return collapsed[: _PROMPT_LABEL_LIMIT - 1] + "…"


# --------------------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanNode:
    """Base of the logical plan."""

    @property
    def children(self) -> tuple[PlanNode, ...]:
        """Child nodes, in evaluation order."""
        return ()

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        """Return a copy of this node with its children replaced (arity must match)."""
        if children:
            raise CompileError(f"{type(self).__name__} takes no children")
        return self


def _one(node: PlanNode, children: tuple[PlanNode, ...]) -> PlanNode:
    if len(children) != 1:
        raise CompileError(f"{type(node).__name__} takes exactly one child")
    return children[0]


def _two(node: PlanNode, children: tuple[PlanNode, ...]) -> tuple[PlanNode, PlanNode]:
    if len(children) != 2:
        raise CompileError(f"{type(node).__name__} takes exactly two children")
    return children[0], children[1]


@dataclass(frozen=True)
class Scan(PlanNode):
    """Leaf: the rows a topic, view, SQL job or saved query produces."""

    source: ScanSource


@dataclass(frozen=True)
class Project(PlanNode):
    """``select()`` — dimensions, grains, governed measures, and (tier 2/3) ad-hoc aggregates."""

    child: PlanNode
    columns: tuple[Column, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        return Project(_one(self, children), self.columns)


@dataclass(frozen=True)
class Filter(PlanNode):
    """``filter()`` / ``where()`` — a boolean predicate over the child's rows."""

    child: PlanNode
    predicate: Expr

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        return Filter(_one(self, children), self.predicate)


@dataclass(frozen=True)
class Aggregate(PlanNode):
    """``group_by().agg()``.

    In Omni, selecting dimensions plus measures *is* the group-by, so this node compiles to the
    same tier-1 query as the equivalent :class:`Project` when every aggregate is a governed
    measure; an ad-hoc aggregation needs the hybrid engine, and mixed aggregates decompose
    (docs/DESIGN.md §2) — both land in M3.
    """

    child: PlanNode
    keys: tuple[Column, ...]
    aggs: tuple[Column, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))
        object.__setattr__(self, "aggs", tuple(self.aggs))

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        return Aggregate(_one(self, children), self.keys, self.aggs)


@dataclass(frozen=True)
class Sort(PlanNode):
    """``sort()`` / ``orderBy()``."""

    child: PlanNode
    keys: tuple[SortKey, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        return Sort(_one(self, children), self.keys)


@dataclass(frozen=True)
class Limit(PlanNode):
    """``limit()`` / ``offset()``.

    ``n`` carries the same trichotomy as the wire's ``limit`` (docs/DESIGN.md §3):
    :data:`~omniframes.compile.querymodel.UNSET` (the user never asked, so the default fetch
    limit applies), ``None`` (the user asked for unlimited → wire ``null``), or an ``int``.
    The sentinel exists because ``offset()`` can be called without ``limit()``.
    """

    child: PlanNode
    n: int | Unset | None = UNSET
    offset: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.n, bool):
            raise CompileError("limit() takes a positive integer or None (unlimited), not a bool")
        if isinstance(self.n, int) and self.n <= 0:
            raise CompileError(f"limit() takes a positive integer or None; got {self.n}")
        if self.offset < 0:
            raise CompileError(f"offset() must not be negative; got {self.offset}")

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        return Limit(_one(self, children), self.n, self.offset)


@dataclass(frozen=True)
class Join(PlanNode):
    """A local join of two sub-plans, with **SQL** semantics — M4.

    ``on`` is an equi-join over column names both sides produce; NULL keys never match, which
    is the whole difference between this node and the internal
    :class:`~omniframes.compile.local.AlignJoin` that re-assembles a decomposed aggregate.
    """

    left: PlanNode
    right: PlanNode
    on: tuple[str, ...] | Expr
    how: JoinHow = JoinHow.INNER

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.left, self.right)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        left, right = _two(self, children)
        return Join(left, right, self.on, self.how)


@dataclass(frozen=True)
class Union(PlanNode):
    """A local union of two sub-plans, by position — M4.

    Both sides must produce the same number of columns *with the same names*: omniframes'
    columns are named wire outputs, so silently taking the left side's names for a differently
    named right side would relabel data rather than stack it.
    """

    left: PlanNode
    right: PlanNode

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.left, self.right)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        left, right = _two(self, children)
        return Union(left, right)


@dataclass(frozen=True)
class WithColumn(PlanNode):
    """``with_column()`` — a derived column, evaluated in tier 2/3 — M3."""

    child: PlanNode
    name: str
    expr: Expr

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        return WithColumn(_one(self, children), self.name, self.expr)


@dataclass(frozen=True)
class MapPandas(PlanNode):
    """A user function applied to the materialized frame — always local — M3."""

    child: PlanNode
    fn: Callable[..., Any]
    schema_hint: OmniSchema | None = None

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def with_children(self, children: tuple[PlanNode, ...]) -> PlanNode:
        return MapPandas(_one(self, children), self.fn, self.schema_hint)
