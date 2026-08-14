"""Result normalization — CONTRACT_NOTES §2.7.

Every frame handed to a user goes through :func:`normalize` first:

* **reserved columns** (``$omni_*``, ``__omni_*``, ``*__omni_sort[_N]``, ``*__omni_summ[_N]``)
  are stripped;
* **totals rows** — flagged by the four indicator columns, since ``/query/run`` has no
  ``row_type`` column — are removed by default and returned separately for ``with_totals()``;
* ``<field>__omni_summ`` **sidecars** (keyed by the *lowercased* field name) supply the totals
  values for raw-SQL ``column_totals`` queries, falling back to the base column; when
  ``_N``-suffixed duplicates exist only the first occurrence wins;
* client-side **aliases** are applied last, as a plain rename (the wire has no aliasing).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import pyarrow as pa
import pyarrow.compute as pc

from omniframes.errors import CompileError

__all__ = [
    "COLUMN_SUBTOTAL_PREFIX",
    "GRAND_TOTAL_VALUE",
    "RESERVED_COLUMN_PATTERNS",
    "TOTAL_INDICATOR_COLUMNS",
    "NormalizedResult",
    "is_reserved_column",
    "normalize",
]

#: Exact patterns from CONTRACT_NOTES §2.7.  The first two are prefixes, the last two suffixes
#: (with an optional ``_N`` disambiguator the server appends for duplicates).
RESERVED_COLUMN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\$omni_"),
    re.compile(r"^__omni_"),
    re.compile(r"__omni_sort(_\d+)?$"),
    re.compile(r"__omni_summ(_\d+)?$"),
)

#: Columns whose non-empty value marks the row as a totals row.  Both the ``$omni_`` and the
#: ``__omni_`` spelling occur, for column and row totals alike.
TOTAL_INDICATOR_COLUMNS: tuple[str, ...] = (
    "$omni_column_total_indicator",
    "__omni_column_total_indicator",
    "$omni_row_total_indicator",
    "__omni_row_total_indicator",
)

#: Indicator value for the grand-total row.
GRAND_TOTAL_VALUE = "::total::"

#: Indicator prefix for a per-field subtotal row (``column_subtotal::<fieldName>``).
COLUMN_SUBTOTAL_PREFIX = "column_subtotal::"

_SIDECAR_PATTERN = re.compile(r"^(?P<base>.+?)__omni_summ(?:_\d+)?$")

# pyarrow's compute stubs describe results loosely (``Array`` vs ``ChunkedArray`` vs
# ``ArrayLike``), so column/mask values flow untyped inside this module; every public signature
# above is precise.
_Column: TypeAlias = Any
_Mask: TypeAlias = Any


def is_reserved_column(name: str) -> bool:
    """Whether ``name`` is one of Omni's internal columns and must never reach the user."""
    return any(pattern.search(name) for pattern in RESERVED_COLUMN_PATTERNS)


@dataclass(frozen=True, slots=True)
class NormalizedResult:
    """The outcome of :func:`normalize`."""

    #: The user-facing rows: totals removed, reserved columns stripped, aliases applied.
    data: pa.Table
    #: The totals rows in the same shape as :attr:`data`, or ``None`` when they were dropped or
    #: the result carried no totals information at all.
    totals: pa.Table | None = None
    #: Reserved column names that were removed, in wire order.
    dropped_columns: tuple[str, ...] = ()
    #: The indicator value of each totals row (``column_total``, ``row_total``, ``::total::``,
    #: ``column_subtotal::<field>``), aligned with :attr:`totals`.
    total_row_types: tuple[str | None, ...] = ()

    @property
    def has_totals(self) -> bool:
        return self.totals is not None and self.totals.num_rows > 0


def normalize(
    table: pa.Table,
    summary_fields: Mapping[str, Any] | None = None,
    *,
    keep_totals: bool = False,
    aliases: Mapping[str, str] | None = None,
) -> NormalizedResult:
    """Turn a decoded Arrow result into the frame the user asked for.

    Args:
        table: the table decoded from a job line's ``result``.
        summary_fields: ``summary.fields`` — used to resolve ``__omni_summ`` sidecars back to
            their field (the sidecar name is the *lowercased* field name).  The table's own
            column names are the fallback, so this may be omitted.
        keep_totals: return the totals rows in :attr:`NormalizedResult.totals` (with sidecar
            values merged in) instead of discarding them.
        aliases: wire-name → alias renames, applied last.  Unknown keys are ignored; a rename
            that collides with another output column is a :class:`~omniframes.errors.CompileError`.
    """
    totals_mask = _totals_mask(table)
    row_types = _total_row_types(table)

    if keep_totals and totals_mask is not None:
        table = _merge_summ_sidecars(table, summary_fields, totals_mask)

    if totals_mask is None:
        data = table
        totals = None
        kept_row_types: tuple[str | None, ...] = ()
    else:
        data_mask: _Mask = pc.invert(totals_mask)
        data = table.filter(data_mask)
        if keep_totals:
            totals = table.filter(totals_mask)
            kept_row_types = tuple(
                value
                for value, is_total in zip(row_types, totals_mask.to_pylist(), strict=True)
                if is_total
            )
        else:
            totals = None
            kept_row_types = ()

    dropped = tuple(name for name in table.column_names if is_reserved_column(name))
    kept = [name for name in table.column_names if not is_reserved_column(name)]
    data = data.select(kept)
    if totals is not None:
        totals = totals.select(kept)

    if aliases:
        renamed = _apply_aliases(kept, aliases)
        data = data.rename_columns(renamed)
        if totals is not None:
            totals = totals.rename_columns(renamed)

    return NormalizedResult(
        data=data,
        totals=totals,
        dropped_columns=dropped,
        total_row_types=kept_row_types,
    )


def _indicator_columns(table: pa.Table) -> list[str]:
    present = set(table.column_names)
    return [name for name in TOTAL_INDICATOR_COLUMNS if name in present]


def _text_column(column: _Column) -> _Column | None:
    """``column`` as text, decoding a dictionary column first; ``None`` when it is not text."""
    if pa.types.is_dictionary(column.type):
        try:
            column = column.cast(column.type.value_type)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError):
            return None
    if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
        return column
    return None


def _flag(column: _Column) -> _Mask:
    """A boolean column: true where the indicator carries a value (empty string does not count).

    Kleene logic keeps the result null-free: ``is_valid`` never yields null, and
    ``false AND null`` is ``false``, so the mask is safe to use directly as a row filter.
    """
    valid = pc.is_valid(column)
    text = _text_column(column)
    if text is None:
        return valid
    non_empty = pc.not_equal(text, pa.scalar("", type=text.type))
    return pc.and_kleene(valid, non_empty)


def _totals_mask(table: pa.Table) -> _Mask | None:
    """A row mask for totals rows, or ``None`` when the result has no indicator columns."""
    indicators = _indicator_columns(table)
    if not indicators:
        return None
    flags = [_flag(table.column(name)) for name in indicators]
    mask = flags[0]
    for flag in flags[1:]:
        mask = pc.or_kleene(mask, flag)
    return mask


def _total_row_types(table: pa.Table) -> tuple[str | None, ...]:
    """The indicator value of every row (``None`` for data rows), in table order."""
    indicators = _indicator_columns(table)
    if not indicators:
        return ()
    columns = [table.column(name) for name in indicators]
    coalesced: _Column = columns[0] if len(columns) == 1 else pc.coalesce(*columns)
    return tuple(value or None for value in coalesced.to_pylist())


def _merge_summ_sidecars(
    table: pa.Table,
    summary_fields: Mapping[str, Any] | None,
    totals_mask: _Mask,
) -> pa.Table:
    """On totals rows, replace each base column with its ``__omni_summ`` sidecar when present."""
    sidecars: dict[str, str] = {}
    for name in table.column_names:
        match = _SIDECAR_PATTERN.match(name)
        if match is not None:
            # First occurrence wins: `x__omni_summ` beats a later `x__omni_summ_1`.
            sidecars.setdefault(match.group("base").lower(), name)
    if not sidecars:
        return table

    # summary.fields is authoritative for the field names; the table's own columns cover raw-SQL
    # results whose column names are not model fields.
    by_lowercase: dict[str, str] = {}
    for name in summary_fields or {}:
        by_lowercase.setdefault(str(name).lower(), str(name))
    for name in table.column_names:
        by_lowercase.setdefault(name.lower(), name)

    for base, sidecar_name in sidecars.items():
        target = by_lowercase.get(base)
        if target is None or target == sidecar_name or target not in table.column_names:
            continue
        index = table.column_names.index(target)
        base_column = table.column(target)
        sidecar = table.column(sidecar_name)
        if sidecar.type != base_column.type:
            try:
                sidecar = sidecar.cast(base_column.type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, ValueError):
                continue
        merged: _Column = pc.if_else(
            pc.and_kleene(totals_mask, pc.is_valid(sidecar)), sidecar, base_column
        )
        table = table.set_column(index, target, merged)
    return table


def _apply_aliases(columns: list[str], aliases: Mapping[str, str]) -> list[str]:
    renamed = [aliases.get(name, name) for name in columns]
    seen: set[str] = set()
    collisions: list[str] = []
    for name in renamed:
        if name in seen:
            collisions.append(name)
        seen.add(name)
    if collisions:
        joined = ", ".join(sorted(set(collisions)))
        raise CompileError(
            f"alias rename produces duplicate column name(s): {joined}. "
            "Aliases must be unique and must not shadow another selected field."
        )
    return renamed
