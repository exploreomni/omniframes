"""The differential comparator (docs/HYBRID.md §4).

``assert_frames_agree`` is deliberately strict about *values* and deliberately relaxed about
everything that is not pinned by a contract:

* **column names and order** must match exactly — a differential test that silently reorders
  columns would hide a real projection bug;
* **decimals compare as exact strings**, so a scale that quietly drifts from ``12,2`` to
  ``12,4`` fails here rather than surfacing as a rounding complaint from a user;
* **floats compare with a relative tolerance** — the only float columns are averages, which two
  engines are entitled to accumulate in different orders;
* **row order is normalized away by default**.  Null ordering across engines is dialect
  dependent and unpinned (``OMNI_DEFAULT``), so a test that has not asked for an order must not
  depend on one.  Pass ``sort=False`` when the ordering *is* the thing under test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pyarrow as pa

__all__ = ["assert_frames_agree", "canonical_key", "values_agree"]

#: Averages are the only float columns either side produces; 1e-9 is far tighter than any
#: difference a summation order can create, and far looser than a real disagreement.
REL_TOL = 1e-9


def assert_frames_agree(
    pushdown: pa.Table,
    reference: pa.Table,
    *,
    sort: bool = True,
) -> None:
    """Fail unless the two tables carry the same rows.

    Args:
        pushdown: what omniframes produced (fake org + local engine).
        reference: what the independent pandas reference produced.
        sort: re-sort both sides by every column before comparing (the default).  Set it to
            ``False`` only when the test asserts an ordering it made deterministic itself.
    """
    assert pushdown.column_names == reference.column_names, (
        f"columns differ\n  pushdown:  {pushdown.column_names}\n  reference: {reference.column_names}"
    )

    left = [_normalize_row(row) for row in pushdown.to_pylist()]
    right = [_normalize_row(row) for row in reference.to_pylist()]
    assert len(left) == len(right), (
        f"row counts differ: pushdown {len(left)}, reference {len(right)}{_sample(left, right)}"
    )

    if sort:
        names = tuple(pushdown.column_names)
        left.sort(key=lambda row: canonical_key(row, names))
        right.sort(key=lambda row: canonical_key(row, names))

    for index, (actual, expected) in enumerate(zip(left, right, strict=True)):
        if not _rows_agree(actual, expected):
            raise AssertionError(
                f"row {index} differs\n  pushdown:  {actual}\n  reference: {expected}"
            )


def canonical_key(row: Mapping[str, Any], names: Sequence[str]) -> tuple[tuple[bool, str], ...]:
    """A total order over rows that cannot be upset by null placement or by dtype."""
    return tuple((row[name] is None, _text(row[name])) for name in names)


def values_agree(actual: object, expected: object) -> bool:
    """Whether two cell values are the same answer (§4 step 2)."""
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, Decimal) or isinstance(expected, Decimal):
        # Exact, as strings: a decimal is exact by construction, so any difference is a bug.
        return str(actual) == str(expected)
    if isinstance(actual, bool) or isinstance(expected, bool):
        return bool(actual) is bool(expected)
    if isinstance(actual, float) or isinstance(expected, float):
        left, right = float(actual), float(expected)  # type: ignore[arg-type]
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return math.isclose(left, right, rel_tol=REL_TOL, abs_tol=0.0)
    return bool(actual == expected)


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _normalize(value) for name, value in row.items()}


def _normalize(value: object) -> Any:
    """Make the two engines' spellings of the same value comparable."""
    if value is None:
        return None
    if isinstance(value, datetime):
        # Naive timestamps are UTC by contract (CONTRACT_NOTES §3.1), so say so explicitly.
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return value


def _rows_agree(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(values_agree(actual[name], expected[name]) for name in actual)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        # Enough digits to order distinct values, few enough that two engines' last bits agree.
        return f"{value:.9g}"
    return str(value)


def _sample(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> str:
    head = "\n  pushdown[0]:  " + (str(left[0]) if left else "(empty)")
    return head + "\n  reference[0]: " + (str(right[0]) if right else "(empty)")
