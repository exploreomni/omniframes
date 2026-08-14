"""Discovery: models, topics, views and their fields (docs/INTERNALS.md §5).

The catalog is for *browsing* — "what can I query?".  It is deliberately not the schema
authority: the only authority for the shape of a result is ``summary.fields`` from a
``planOnly`` run (docs/DESIGN.md §3), which is what ``df.schema`` uses.  Catalog metadata can
be stale relative to a branch, and the topic list endpoint is lossy by design.

Every lookup is read-through cached on the instance; :meth:`Catalog.refresh` drops the cache.
Payloads are wrapped in small typed records that keep a ``raw`` escape hatch, so nothing the
server sends is ever lost — a new key in the API is reachable through ``.raw`` without a
release here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from omniframes.errors import CompileError
from omniframes.transport.base import QueryTransport
from omniframes.types import OmniField

__all__ = ["Catalog", "ModelInfo", "RelationshipInfo", "TopicInfo", "ViewInfo"]

#: ``GET /models`` is cursor-paginated with a maximum page size of 100 (CONTRACT_NOTES §4).
_PAGE_SIZE = 100

#: Guard against a server that keeps handing back cursors.
_MAX_PAGES = 1_000


@dataclass(frozen=True)
class ModelInfo:
    """One row of ``GET /api/v1/models``."""

    id: str
    name: str
    model_kind: str | None = None
    connection_id: str | None = None
    base_model_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> ModelInfo:
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            model_kind=payload.get("modelKind"),
            connection_id=payload.get("connectionId"),
            base_model_id=payload.get("baseModelId"),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class ViewInfo:
    """A view inside a topic, with its fields split the way the API reports them."""

    name: str
    label: str | None = None
    dimensions: tuple[OmniField, ...] = ()
    measures: tuple[OmniField, ...] = ()
    filter_only_fields: tuple[OmniField, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> ViewInfo:
        return cls(
            name=str(payload.get("name", "")),
            label=payload.get("label"),
            dimensions=_fields(payload.get("dimensions")),
            measures=_fields(payload.get("measures")),
            filter_only_fields=_fields(payload.get("filter_only_fields")),
            raw=dict(payload),
        )

    @property
    def fields(self) -> tuple[OmniField, ...]:
        """Dimensions, then measures, then filter-only fields."""
        return self.dimensions + self.measures + self.filter_only_fields

    def field(self, name: str) -> OmniField:
        """One field by fully qualified (``view.field``) or bare name."""
        for candidate in self.fields:
            if name in (candidate.name, candidate.name.split(".")[-1]):
                return candidate
        raise CompileError(f"view {self.name!r} has no field {name!r}")


@dataclass(frozen=True)
class RelationshipInfo:
    """One join edge of a topic."""

    left_view_name: str
    right_view_name: str
    join_type: str | None = None
    relationship_type: str | None = None
    sql: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> RelationshipInfo:
        return cls(
            left_view_name=str(payload.get("left_view_name", "")),
            right_view_name=str(payload.get("right_view_name", "")),
            join_type=payload.get("join_type"),
            relationship_type=payload.get("relationship_type"),
            sql=payload.get("sql"),
            raw=dict(payload),
        )


@dataclass(frozen=True)
class TopicInfo:
    """A topic.

    Entries returned by :meth:`Catalog.topics` come from the *list* endpoint and carry no
    field metadata (``views`` is empty, :attr:`is_detailed` is false); :meth:`Catalog.topic`
    fetches the detail payload — the only endpoint with full field metadata (CONTRACT_NOTES §4).
    """

    name: str
    base_view_name: str = ""
    label: str | None = None
    description: str | None = None
    group_label: str | None = None
    hidden: bool = False
    views: tuple[ViewInfo, ...] = ()
    relationships: tuple[RelationshipInfo, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> TopicInfo:
        raw_views = payload.get("views")
        raw_relationships = payload.get("relationships")
        return cls(
            name=str(payload.get("name", "")),
            base_view_name=str(payload.get("base_view_name", "")),
            label=payload.get("label"),
            description=payload.get("description"),
            group_label=payload.get("group_label"),
            hidden=bool(payload.get("hidden", False)),
            views=tuple(
                ViewInfo.from_wire(view) for view in raw_views or () if isinstance(view, Mapping)
            ),
            relationships=tuple(
                RelationshipInfo.from_wire(edge)
                for edge in raw_relationships or ()
                if isinstance(edge, Mapping)
            ),
            raw=dict(payload),
        )

    @property
    def is_detailed(self) -> bool:
        """Whether this record came from the topic *detail* endpoint (has field metadata)."""
        return bool(self.views)

    @property
    def fields(self) -> tuple[OmniField, ...]:
        """Every field of every view in the topic."""
        return tuple(f for view in self.views for f in view.fields)

    def view(self, name: str) -> ViewInfo:
        for candidate in self.views:
            if candidate.name == name:
                return candidate
        available = ", ".join(v.name for v in self.views)
        raise CompileError(f"topic {self.name!r} has no view {name!r}; it joins: {available}")

    def field(self, name: str) -> OmniField:
        """One field by fully qualified name (``users.state``)."""
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        raise CompileError(f"topic {self.name!r} has no field {name!r}")


def _fields(raw: object) -> tuple[OmniField, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return ()
    parsed: list[OmniField] = []
    for payload in raw:
        if not isinstance(payload, Mapping):
            continue
        name = payload.get("fully_qualified_name") or payload.get("field_name") or ""
        parsed.append(OmniField.from_wire(str(name), dict(payload)))
    return tuple(parsed)


class Catalog:
    """Read-through cached discovery over a :class:`~omniframes.transport.base.QueryTransport`."""

    def __init__(
        self, transport: QueryTransport, *, preflight: Callable[[], None] | None = None
    ) -> None:
        self._transport = transport
        #: Run before the first request of any lookup — the session hooks its cached ``whoami``
        #: preflight in here so a bad key fails crisply at the first catalog call.
        self._preflight = preflight
        self._models: tuple[ModelInfo, ...] | None = None
        self._topics: dict[str, tuple[TopicInfo, ...]] = {}
        self._topic_details: dict[tuple[str, str], TopicInfo] = {}

    def __repr__(self) -> str:
        cached = "cached" if self._models is not None else "empty"
        return f"Catalog({cached}, topics={len(self._topics)})"

    # -- models ------------------------------------------------------------------------

    def models(self) -> tuple[ModelInfo, ...]:
        """Every model the key can see, walking the whole cursor (CONTRACT_NOTES §4)."""
        if self._models is None:
            self._before_call()
            self._models = tuple(self._paginate_models())
        return self._models

    def _paginate_models(self) -> list[ModelInfo]:
        collected: list[ModelInfo] = []
        cursor: str | None = None
        for _ in range(_MAX_PAGES):
            payload = self._transport.list_models(pageSize=_PAGE_SIZE, cursor=cursor)
            records = payload.get("records")
            if isinstance(records, Sequence) and not isinstance(records, str):
                collected.extend(
                    ModelInfo.from_wire(record) for record in records if isinstance(record, Mapping)
                )
            page_info = payload.get("pageInfo")
            if not isinstance(page_info, Mapping) or not page_info.get("hasNextPage"):
                return collected
            next_cursor = page_info.get("nextCursor")
            if not next_cursor:
                return collected
            cursor = str(next_cursor)
        raise CompileError(  # pragma: no cover - a server that never stops paginating
            "GET /models kept returning cursors; refusing to page forever"
        )

    def model(self, name_or_id: str) -> ModelInfo:
        """Resolve a model by name or id (cached)."""
        for candidate in self.models():
            if name_or_id in (candidate.id, candidate.name):
                return candidate
        names = ", ".join(sorted(candidate.name for candidate in self.models())) or "(none)"
        raise CompileError(
            f"no model named {name_or_id!r} is visible to this API key. Available models: {names}"
        )

    def model_id(self, name_or_id: str) -> str:
        """The model id for a name or id."""
        return self.model(name_or_id).id

    # -- topics ------------------------------------------------------------------------

    def topics(self, model: str) -> tuple[TopicInfo, ...]:
        """The topics of a model, from the list endpoint (no field metadata)."""
        model_id = self.model_id(model)
        cached = self._topics.get(model_id)
        if cached is None:
            self._before_call()
            payload = self._transport.list_topics(model_id)
            raw = payload.get("topics")
            entries = raw if isinstance(raw, Sequence) and not isinstance(raw, str) else ()
            cached = tuple(
                TopicInfo.from_wire(entry) for entry in entries if isinstance(entry, Mapping)
            )
            self._topics[model_id] = cached
        return cached

    def topic(self, model: str, name: str) -> TopicInfo:
        """One topic with its full field metadata (the topic *detail* endpoint)."""
        model_id = self.model_id(model)
        key = (model_id, name)
        cached = self._topic_details.get(key)
        if cached is None:
            self._before_call()
            payload = self._transport.get_topic(model_id, name)
            raw = payload.get("topic")
            if not isinstance(raw, Mapping):
                raise CompileError(
                    f"the topic detail response for {name!r} carried no 'topic' object"
                )
            cached = TopicInfo.from_wire(raw)
            self._topic_details[key] = cached
        return cached

    def views(self, model: str) -> tuple[ViewInfo, ...]:
        """Every view reachable through the model's topics, de-duplicated by name.

        There is no per-view detail endpoint and the flattened view list is lossy
        (CONTRACT_NOTES §4), so views are read out of the topic detail payloads — which is also
        the only place their fields come with types.
        """
        seen: dict[str, ViewInfo] = {}
        for summary in self.topics(model):
            for view in self.topic(model, summary.name).views:
                seen.setdefault(view.name, view)
        return tuple(seen.values())

    def _before_call(self) -> None:
        if self._preflight is not None:
            self._preflight()

    def refresh(self) -> None:
        """Drop every cached payload; the next lookup re-reads from the API."""
        self._models = None
        self._topics.clear()
        self._topic_details.clear()
