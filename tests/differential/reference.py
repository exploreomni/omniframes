"""The independent reference the differential lane compares against (docs/HYBRID.md §4).

Plain ``pandas.read_parquet`` over ``tests/data/bench/*.parquet``, joined the way the bench
topic joins (``order_items`` LEFT JOIN ``users`` LEFT JOIN ``products``), and then **explicit
Python** for every predicate and every aggregate.

That explicitness is the point.  If the reference used ``Series != value`` it would keep NULLs
(pandas' two-valued comparison) and quietly agree with a local engine that made the same
mistake; writing the predicate as a function that returns ``True``/``False``/``None`` and
keeping only ``True`` states SQL's rule instead of inheriting somebody's.  Likewise the
aggregates are computed from Python values — ``Decimal`` arithmetic stays exact, an average is
``sum / count`` as a float, and ``count_distinct`` builds a set of the non-null values — so the
lane really does cross-check three implementations: DuckDB (the fake), Arrow compute (the local
engine) and this.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

__all__ = [
    "BENCH_DIR",
    "Row",
    "aggregate",
    "and_",
    "bench_rows",
    "keep",
    "known",
    "not_",
    "or_",
    "project",
    "sql_join",
    "stack",
    "table_of",
    "truth",
]

BENCH_DIR = Path(__file__).resolve().parents[1] / "data" / "bench"

#: One joined fact row, keyed by wire name (``"users.state"``), with missing values as ``None``.
Row = Mapping[str, Any]

_TABLES = ("order_items", "users", "products")


def _frame() -> pd.DataFrame:
    """``order_items`` LEFT JOIN ``users`` LEFT JOIN ``products``, columns named as the wire does.

    The joins are the topic's (tests/fakes/bench_model.py): an order item whose ``user_id`` has
    no user survives with NULL user columns, which is exactly the NULL-``users.state`` group the
    known answers encode.
    """
    # ``numpy_nullable`` keeps an integer column an integer when the join finds no match; the
    # default backend widens it to float64, which would make the reference disagree with both
    # engines about whether an age is 40 or 40.0.
    frames = {
        name: pd.read_parquet(BENCH_DIR / f"{name}.parquet", dtype_backend="numpy_nullable")
        for name in _TABLES
    }
    joined = frames["order_items"].add_prefix("order_items.")
    joined = joined.merge(
        frames["users"].add_prefix("users."),
        how="left",
        left_on="order_items.user_id",
        right_on="users.id",
    )
    return joined.merge(
        frames["products"].add_prefix("products."),
        how="left",
        left_on="order_items.product_id",
        right_on="products.id",
    )


_CACHED: list[dict[str, Any]] | None = None


def bench_rows() -> list[dict[str, Any]]:
    """The joined bench fact table as plain Python rows (read once, then copied per call)."""
    global _CACHED
    if _CACHED is None:
        frame = _frame()
        _CACHED = [
            {name: _value(row[name]) for name in frame.columns} for _, row in frame.iterrows()
        ]
    return list(_CACHED)


def _value(value: Any) -> Any:
    """One cell as a plain Python value; every flavor of missing becomes ``None``."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, float) and value != value:
        # NaN, which is the shape a missing group key still takes when pandas groups on it.
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, Decimal | str):
        return value
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


# --------------------------------------------------------------------------------------
# Three-valued logic, spelled out
# --------------------------------------------------------------------------------------


def truth(value: Any, check: Callable[[Any], bool]) -> bool | None:
    """``check(value)``, except that a missing operand makes the answer unknown, not false."""
    return None if value is None else bool(check(value))


def not_(operand: bool | None) -> bool | None:
    """``NOT UNKNOWN`` is UNKNOWN — the case a two-valued implementation gets wrong."""
    return None if operand is None else not operand


def and_(*operands: bool | None) -> bool | None:
    if any(operand is False for operand in operands):
        return False
    return None if any(operand is None for operand in operands) else True


def or_(*operands: bool | None) -> bool | None:
    if any(operand is True for operand in operands):
        return True
    return None if any(operand is None for operand in operands) else False


def known(key: str) -> Any:
    """The ``tests/data/bench/known_answers.json`` entry called ``key``, as a list of dicts."""
    import json

    payload = json.loads((BENCH_DIR / "known_answers.json").read_text("utf-8"))
    entry = payload["answers"][key]
    return [dict(zip(entry["columns"], row, strict=True)) for row in entry["rows"]]


# --------------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------------


def keep(rows: Iterable[Row], predicate: Callable[[Row], bool | None]) -> list[Row]:
    """SQL's ``WHERE``: a row survives only when the predicate is **exactly** true.

    ``None`` (unknown) is not true, so a NULL never passes — including through a negation,
    which is where a two-valued implementation goes wrong.
    """
    return [row for row in rows if predicate(row) is True]


def project(rows: Iterable[Row], columns: Mapping[str, Callable[[Row], Any] | str]) -> pa.Table:
    """Build the reference result: ``{output name: source column or function}``, in order."""
    materialized = list(rows)
    payload: list[dict[str, Any]] = []
    for row in materialized:
        record: dict[str, Any] = {}
        for name, source in columns.items():
            record[name] = row[source] if isinstance(source, str) else source(row)
        payload.append(record)
    return table_of(payload, tuple(columns))


def aggregate(
    rows: Iterable[Row],
    keys: Sequence[str],
    aggs: Mapping[str, tuple[str, str]],
) -> pa.Table:
    """``GROUP BY`` with ``dropna=False`` and SQL's null-skipping aggregates.

    ``aggs`` maps an output name to ``(function, column)`` where function is one of
    ``sum``/``count``/``count_distinct``/``avg``/``min``/``max``.
    """
    materialized = list(rows)
    if keys:
        frame = pd.DataFrame({key: [row[key] for row in materialized] for key in keys})
        # dropna=False is the whole reason this is written with pandas: a NULL key is a group.
        grouped = frame.groupby(list(keys), dropna=False, sort=False).indices
        groups = [
            (tuple(_value(part) for part in _as_tuple(key)), [materialized[i] for i in indices])
            for key, indices in grouped.items()
        ]
    else:
        groups = [((), materialized)]

    payload: list[dict[str, Any]] = []
    for key, members in groups:
        record: dict[str, Any] = dict(zip(keys, key, strict=True))
        for name, (function, column) in aggs.items():
            record[name] = _apply(function, [row[column] for row in members])
        payload.append(record)
    return table_of(payload, (*keys, *aggs))


def _as_tuple(key: Any) -> tuple[Any, ...]:
    return key if isinstance(key, tuple) else (key,)


def _apply(function: str, values: Sequence[Any]) -> Any:
    present = [value for value in values if value is not None]
    if function == "count":
        return len(present)
    if function == "count_distinct":
        return len({str(value) for value in present})
    if not present:
        # sum/min/max/avg over an empty or all-null group is NULL, not zero.
        return None
    if function == "sum":
        return _sum(present)
    if function == "min":
        return min(present)
    if function == "max":
        return max(present)
    if function == "avg":
        # An average is a float on both sides (docs/HYBRID.md §3.3): Arrow's decimal mean would
        # round to the operand's scale, so the local engine casts once and so does this.
        return float(_sum(present)) / len(present)
    raise AssertionError(f"the reference does not implement {function}()")


def _sum(values: Sequence[Any]) -> Any:
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def sql_join(
    left: Sequence[Row],
    right: Sequence[Row],
    *,
    on: Sequence[str],
    how: str,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
) -> pa.Table:
    """An equi-join with **SQL** null semantics, written out rather than delegated.

    ``pandas.merge`` matches NA keys to each other, so a reference built on it would agree with
    a local engine that made the same mistake and the differential lane would prove nothing.
    Here the rule is stated instead: a row whose key is NULL in any position **matches nothing**
    — not even another NULL — and therefore reappears only where an unmatched row would, which
    is nowhere for an inner join and on its own side for a left/right/outer one.

    Output order is the join's: the keys once, then the left frame's other columns, then the
    right frame's.
    """
    keys = tuple(on)
    left_rest = [name for name in left_columns if name not in keys]
    right_rest = [name for name in right_columns if name not in keys]
    columns = (*keys, *left_rest, *right_rest)

    def key_of(row: Row) -> tuple[Any, ...]:
        return tuple(row[name] for name in keys)

    matchable: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(right):
        candidate = key_of(row)
        if any(value is None for value in candidate):
            continue  # a NULL key is not a value; it joins to nothing
        matchable.setdefault(candidate, []).append(index)

    matched: set[int] = set()
    rows: list[dict[str, Any]] = []
    for row in left:
        candidate = key_of(row)
        hits = () if any(value is None for value in candidate) else matchable.get(candidate, ())
        for index in hits:
            matched.add(index)
            partner = right[index]
            rows.append(
                {
                    **{name: row[name] for name in keys},
                    **{name: row[name] for name in left_rest},
                    **{name: partner[name] for name in right_rest},
                }
            )
        if not hits and how in ("left", "outer"):
            rows.append(
                {
                    **{name: row[name] for name in keys},
                    **{name: row[name] for name in left_rest},
                    **dict.fromkeys(right_rest),
                }
            )
    if how in ("right", "outer"):
        for index, row in enumerate(right):
            if index in matched:
                continue
            rows.append(
                {
                    **{name: row[name] for name in keys},
                    **dict.fromkeys(left_rest),
                    **{name: row[name] for name in right_rest},
                }
            )
    return table_of(rows, columns)


def stack(*tables: pa.Table) -> pa.Table:
    """``UNION ALL``: the rows of each table in turn, keeping duplicates and column order."""
    columns = list(tables[0].column_names)
    rows: list[Mapping[str, Any]] = []
    for table in tables:
        assert list(table.column_names) == columns, "the reference stacks matching shapes only"
        rows.extend(table.to_pylist())
    return table_of(rows, columns)


def table_of(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pa.Table:
    """Rows to an Arrow table with the given column order, decimals and all."""
    data = {name: [row[name] for row in rows] for name in columns}
    return pa.table({name: pa.array(values) for name, values in data.items()})


def decimal(value: str) -> Decimal:
    """A literal in the reference, spelled to match the dataset's ``decimal(12, 2)`` scale."""
    return Decimal(value)
