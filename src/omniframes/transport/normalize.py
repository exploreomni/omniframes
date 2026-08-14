"""Result normalization — CONTRACT_NOTES §2.7, docs/SQLTIER.md §5/§3.2.

Every frame handed to a user goes through :func:`normalize` first:

* **reserved columns** (``$omni_*``, ``__omni_*``, ``*__omni_sort[_N]``, ``*__omni_summ[_N]``)
  are stripped;
* **totals rows** — flagged by the four indicator columns, since ``/query/run`` has no
  ``row_type`` column — are removed by default and returned separately for ``with_totals()``;
* ``<field>__omni_summ`` **sidecars** (keyed by the *lowercased* field name) supply the totals
  values for raw-SQL ``column_totals`` queries, falling back to the base column; when
  ``_N``-suffixed duplicates exist only the first occurrence wins;
* a **formatted grain pair** (``field[grain]__raw`` + the formatted ``field[grain]``) collapses
  to the ``__raw`` timestamp under the plain name — the same on the semantic and the OmniSQL
  path, because both tiers emit the pair;
* client-side **aliases** are applied last, as a plain rename (the wire has no aliasing), by
  exact wire name for a bare ref and by ``.of_expr_<n>`` suffix for a tier-2 expression item,
  whose scope prefix the server chooses and no client can predict.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import pyarrow as pa
import pyarrow.compute as pc

from omniframes.errors import CompileError

__all__ = [
    "COLUMN_SUBTOTAL_PREFIX",
    "GRAIN_RAW_SUFFIX",
    "GRAND_TOTAL_VALUE",
    "RESERVED_COLUMN_PATTERNS",
    "TOTAL_INDICATOR_COLUMNS",
    "NormalizedResult",
    "collapse_grain_names",
    "is_reserved_column",
    "normalize",
    "resolve_aliases",
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

#: Suffix of the ``DATE_TRUNC`` half of a formatted grain pair (CONTRACT_NOTES §2.7).
GRAIN_RAW_SUFFIX: Final = "__raw"

#: The ``__raw`` half of a grain pair, recognized *structurally*: only a bracketed grain item
#: (``view.field[grain]__raw``) is one, so an ordinary column that merely ends in ``__raw`` — a
#: raw-SQL result may well have one — is left alone.  ``__raw`` matches none of the reserved
#: patterns above, which is why the pair has to be reconciled here rather than stripped.
_GRAIN_RAW_PATTERN: Final = re.compile(
    rf"^(?P<base>.+\[[A-Za-z0-9_]+\]){re.escape(GRAIN_RAW_SUFFIX)}$"
)

#: A tier-2 alias key: the generated name of an expression select item (docs/SQLTIER.md §3.2).
#: The server honors the alias but prefixes it with a scope view no client can predict, so such
#: a key is matched against the result columns by ``.<key>`` suffix instead of by equality.
#: Kept in step with ``compile.sqlgen.EXPR_ALIAS_PREFIX`` — the transport layer never imports
#: the compiler — and pinned against it by test.
_EXPR_ALIAS_KEY: Final = re.compile(r"^of_expr_\d+$")

# pyarrow's compute stubs describe results loosely (``Array`` vs ``ChunkedArray`` vs
# ``ArrayLike``), so column/mask values flow untyped inside this module; every public signature
# above is precise.
_Column: TypeAlias = Any
_Mask: TypeAlias = Any


def is_reserved_column(name: str) -> bool:
    """Whether ``name`` is one of Omni's internal columns and must never reach the user."""
    return any(pattern.search(name) for pattern in RESERVED_COLUMN_PATTERNS)


def collapse_grain_names(names: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Collapse the formatted-grain pairs in ``names`` (docs/SQLTIER.md §5).

    A model that formats a grain returns ``field[grain]`` as TWO columns on both the semantic
    and the OmniSQL path: ``field[grain]__raw`` — the ``DATE_TRUNC`` timestamp, at the item's
    own select position — and the formatted string under ``field[grain]``, appended after every
    other column (CONTRACT_NOTES §2.7).  ``__raw`` wins: the values are type-stable and sort,
    group and join chronologically, which the display string only does by accident.

    Returns:
        the names to keep, in order, and what each is called afterwards.  The result is the
        input unchanged when there is no pair to collapse.  Both halves must be present for a
        collapse: a lone ``__raw`` column, and a ``__raw``-suffixed name that is not a grain
        item, pass through untouched.
    """
    present = set(names)
    collapsed: dict[str, str] = {}
    for name in names:
        match = _GRAIN_RAW_PATTERN.match(name)
        if match is not None and match.group("base") in present:
            collapsed[name] = match.group("base")
    if not collapsed:
        return tuple(names), tuple(names)
    formatted = set(collapsed.values())
    keep = tuple(name for name in names if name not in formatted)
    return keep, tuple(collapsed.get(name, name) for name in keep)


def resolve_aliases(columns: Sequence[str], aliases: Mapping[str, str]) -> tuple[str, ...]:
    """Apply the client-side rename map to ``columns`` (docs/SQLTIER.md §3.2).

    Alias keys come in two kinds, because the server names result columns in two regimes: a
    **bare ref** arrives under its exact wire name, so its key matches by equality, while a
    tier-2 **expression item** arrives as ``<scope_view>.of_expr_<n>`` with a prefix that is not
    predictable, so its key matches the one column ending in ``.of_expr_<n>``.  Keys that match
    nothing are ignored, exactly as a stale wire name is.

    Raises:
        CompileError: two output columns would end up with the same name, or a generated alias
            is ambiguous (which its construction rules out).
    """
    if not aliases:
        return tuple(columns)
    renamed = list(columns)
    for key, alias in aliases.items():
        index = _alias_target(columns, key)
        if index is not None:
            renamed[index] = alias
    _refuse_collisions(renamed)
    return tuple(renamed)


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
        aliases: renames applied last — see :func:`resolve_aliases` for the two key regimes.
            Unknown keys are ignored; a rename that collides with another output column is a
            :class:`~omniframes.errors.CompileError`.
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
    # The grain collapse runs before the renames, since an alias names the plain `field[grain]`
    # the pair collapses into, never its `__raw` half.
    selected, columns = collapse_grain_names(kept)
    data = data.select(list(selected))
    if totals is not None:
        totals = totals.select(list(selected))

    renamed = resolve_aliases(columns, aliases or {})
    if renamed != tuple(selected):
        data = data.rename_columns(list(renamed))
        if totals is not None:
            totals = totals.rename_columns(list(renamed))

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


def _alias_target(columns: Sequence[str], key: str) -> int | None:
    """The column ``key`` renames: an exact wire name, else a generated ``of_expr_<n>`` suffix."""
    if key in columns:
        return columns.index(key)
    if _EXPR_ALIAS_KEY.match(key) is None:
        return None
    suffix = f".{key}"
    matches = [index for index, name in enumerate(columns) if name.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    if matches:
        joined = ", ".join(columns[index] for index in matches)
        raise CompileError(
            f"the generated alias {key} matches {len(matches)} result columns ({joined}); "
            "an expression item's alias is unique within one statement, so this result did not "
            "come from the query that was sent."
        )
    return None


def _refuse_collisions(names: Sequence[str]) -> None:
    seen: set[str] = set()
    collisions: list[str] = []
    for name in names:
        if name in seen:
            collisions.append(name)
        seen.add(name)
    if collisions:
        joined = ", ".join(sorted(set(collisions)))
        raise CompileError(
            f"alias rename produces duplicate column name(s): {joined}. "
            "Aliases must be unique and must not shadow another selected field."
        )
