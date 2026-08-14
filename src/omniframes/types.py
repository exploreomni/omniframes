"""Public schema types.

The only schema authority is ``summary.fields`` from a ``planOnly`` run (see
docs/CONTRACT_NOTES.md §2.3/§2.4). Catalog metadata is discovery-only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

__all__ = ["OmniDataType", "OmniField", "OmniSchema"]


class OmniDataType(enum.Enum):
    """Omni's field data types as reported in ``summary.fields[*].data_type``."""

    ARRAY = "ARRAY"
    BOOLEAN = "BOOLEAN"
    INTERVAL = "INTERVAL"
    JSON = "JSON"
    NUMBER = "NUMBER"
    OTHER_UNGROUPABLE = "OTHER_UNGROUPABLE"
    STRING = "STRING"
    TIMESTAMP = "TIMESTAMP"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_wire(cls, value: str | None) -> OmniDataType:
        try:
            return cls(value) if value is not None else cls.UNKNOWN
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class OmniField:
    """One field of a result schema (from ``summary.fields``) or the catalog."""

    name: str
    data_type: OmniDataType
    view_name: str | None = None
    label: str | None = None
    is_dimension: bool | None = None
    aggregate_type: str | None = None
    date_type: str | None = None
    is_calc: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_wire(cls, name: str, payload: dict[str, Any]) -> OmniField:
        return cls(
            name=name,
            data_type=OmniDataType.from_wire(payload.get("data_type")),
            view_name=payload.get("view_name"),
            label=payload.get("label"),
            is_dimension=payload.get("is_dimension"),
            aggregate_type=payload.get("aggregate_type"),
            date_type=payload.get("date_type"),
            is_calc=bool(payload.get("is_calc", False)),
            raw=payload,
        )


@dataclass(frozen=True)
class OmniSchema:
    """An ordered collection of fields."""

    fields: tuple[OmniField, ...]

    def __iter__(self) -> Any:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def __getitem__(self, name: str) -> OmniField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)
