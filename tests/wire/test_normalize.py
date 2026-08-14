"""Result normalization against checked-in wire bodies (CONTRACT_NOTES §2.7)."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from omniframes.errors import CompileError
from omniframes.transport.arrow import decode_result
from omniframes.transport.ndjson import parse_response
from omniframes.transport.normalize import (
    TOTAL_INDICATOR_COLUMNS,
    is_reserved_column,
    normalize,
)
from tests.wire import read_fixture

RESERVED_IN_TOTALS_FIXTURE = (
    "total_sale_price__omni_summ",
    "total_sale_price__omni_summ_1",
    "order_count__omni_summ",
    "PRODUCT__omni_sort",
    "$omni_id",
    "$omni_column_total_indicator",
    "$omni_row_total_indicator",
)


def fixture_table(name: str) -> pa.Table:
    (job,) = parse_response(read_fixture(name)).jobs
    assert job.result is not None
    return decode_result(job.result)


def fixture_summary_fields(name: str) -> dict[str, Any]:
    (job,) = parse_response(read_fixture(name)).jobs
    assert job.summary is not None
    return dict(job.summary["fields"])


# --------------------------------------------------------------------------------------------
# Reserved columns
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "$omni_id",
        "$omni_column_total_indicator",
        "__omni_column_total_indicator",
        "__omni_anything",
        "PRODUCT__omni_sort",
        "PRODUCT__omni_sort_12",
        "total_sale_price__omni_summ",
        "total_sale_price__omni_summ_3",
    ],
)
def test_reserved_columns_are_recognized(name):
    assert is_reserved_column(name)


@pytest.mark.parametrize(
    "name",
    [
        "users.state",
        "order_items.total_sale_price",
        "omni_id",
        "_omni_thing",
        "x__omni_sorted",
        "x__omni_summary",
        "$omni",
    ],
)
def test_ordinary_columns_are_not_reserved(name):
    assert not is_reserved_column(name)


def test_reserved_columns_are_stripped_and_reported():
    table = fixture_table("totals_with_sidecars.ndjson")

    result = normalize(table, fixture_summary_fields("totals_with_sidecars.ndjson"))

    assert result.dropped_columns == RESERVED_IN_TOTALS_FIXTURE
    assert result.data.column_names == ["STATE", "TOTAL_SALE_PRICE", "ORDER_COUNT"]
    for name in TOTAL_INDICATOR_COLUMNS:
        assert name not in result.data.column_names


def test_a_clean_result_passes_through_unchanged():
    table = fixture_table("happy_single_job.ndjson")

    result = normalize(table, fixture_summary_fields("happy_single_job.ndjson"))

    assert result.data.equals(table)
    assert result.totals is None
    assert result.dropped_columns == ()
    assert result.total_row_types == ()
    assert not result.has_totals


# --------------------------------------------------------------------------------------------
# Totals rows
# --------------------------------------------------------------------------------------------


def test_totals_rows_are_dropped_by_default():
    table = fixture_table("totals_with_sidecars.ndjson")
    assert table.num_rows == 6

    result = normalize(table, fixture_summary_fields("totals_with_sidecars.ndjson"))

    assert result.totals is None
    assert result.data.num_rows == 3
    assert result.data.column("STATE").to_pylist() == ["CA", "NY", None]
    assert result.data.column("TOTAL_SALE_PRICE").to_pylist() == [10.0, 20.0, 5.0]
    assert result.data.column("ORDER_COUNT").to_pylist() == [2, 3, 1]


def test_an_empty_indicator_value_is_not_a_totals_row():
    # Row 2 carries "" in $omni_column_total_indicator and must stay in the data frame.
    table = fixture_table("totals_with_sidecars.ndjson")

    result = normalize(table, keep_totals=True)

    assert result.data.num_rows == 3
    assert result.data.column("STATE").to_pylist()[2] is None


def test_keep_totals_returns_the_totals_rows_separately():
    table = fixture_table("totals_with_sidecars.ndjson")

    result = normalize(
        table, fixture_summary_fields("totals_with_sidecars.ndjson"), keep_totals=True
    )

    assert result.totals is not None
    assert result.has_totals
    assert result.totals.num_rows == 3
    assert result.totals.column_names == result.data.column_names
    assert result.total_row_types == ("column_subtotal::STATE", "::total::", "row_total")


def test_sidecars_supply_the_totals_values_with_fallback_to_the_base_column():
    table = fixture_table("totals_with_sidecars.ndjson")

    result = normalize(
        table, fixture_summary_fields("totals_with_sidecars.ndjson"), keep_totals=True
    )

    assert result.totals is not None
    # Rows 0/1: sidecar value wins. Row 2: sidecar is null, so the base column is kept.
    assert result.totals.column("TOTAL_SALE_PRICE").to_pylist() == [30.0, 35.0, 1.5]
    # The string sidecar is cast to the base column's type.
    assert result.totals.column("ORDER_COUNT").to_pylist() == [6, 6, 1]
    assert result.totals.column("ORDER_COUNT").type == pa.int64()


def test_only_the_first_sidecar_occurrence_wins():
    table = fixture_table("totals_with_sidecars.ndjson")

    result = normalize(
        table, fixture_summary_fields("totals_with_sidecars.ndjson"), keep_totals=True
    )

    assert result.totals is not None
    merged = result.totals.column("TOTAL_SALE_PRICE").to_pylist()
    # `total_sale_price__omni_summ_1` holds 111.0/222.0/333.0 and must be ignored entirely.
    assert 111.0 not in merged
    assert 222.0 not in merged
    assert 333.0 not in merged


def test_sidecar_merge_does_not_touch_data_rows():
    table = fixture_table("totals_with_sidecars.ndjson")

    result = normalize(
        table, fixture_summary_fields("totals_with_sidecars.ndjson"), keep_totals=True
    )

    assert result.data.column("TOTAL_SALE_PRICE").to_pylist() == [10.0, 20.0, 5.0]
    assert result.data.column("ORDER_COUNT").to_pylist() == [2, 3, 1]


def test_summary_fields_are_optional_for_the_sidecar_lookup():
    table = fixture_table("totals_with_sidecars.ndjson")

    with_summary = normalize(
        table, fixture_summary_fields("totals_with_sidecars.ndjson"), keep_totals=True
    )
    without_summary = normalize(table, keep_totals=True)

    assert without_summary.totals is not None
    assert with_summary.totals is not None
    assert without_summary.totals.equals(with_summary.totals)


def test_indicator_columns_without_totals_rows_yield_an_empty_totals_table():
    table = pa.table(
        {
            "users.state": pa.array(["CA", "NY"], pa.string()),
            "__omni_column_total_indicator": pa.array([None, None], pa.string()),
        }
    )

    result = normalize(table, keep_totals=True)

    assert result.data.num_rows == 2
    assert result.totals is not None
    assert result.totals.num_rows == 0
    assert not result.has_totals
    assert result.total_row_types == ()


def test_totals_detected_via_the_double_underscore_indicator_spelling():
    table = pa.table(
        {
            "users.state": pa.array(["CA", None], pa.string()),
            "__omni_row_total_indicator": pa.array([None, "row_total"], pa.string()),
        }
    )

    result = normalize(table, keep_totals=True)

    assert result.data.num_rows == 1
    assert result.totals is not None
    assert result.totals.num_rows == 1
    assert result.total_row_types == ("row_total",)
    assert result.dropped_columns == ("__omni_row_total_indicator",)


# --------------------------------------------------------------------------------------------
# Aliases
# --------------------------------------------------------------------------------------------


def test_aliases_are_applied_last_to_data_and_totals():
    table = fixture_table("totals_with_sidecars.ndjson")

    result = normalize(
        table,
        fixture_summary_fields("totals_with_sidecars.ndjson"),
        keep_totals=True,
        aliases={"TOTAL_SALE_PRICE": "revenue", "not_a_column": "ignored"},
    )

    assert result.data.column_names == ["STATE", "revenue", "ORDER_COUNT"]
    assert result.totals is not None
    assert result.totals.column_names == ["STATE", "revenue", "ORDER_COUNT"]
    assert result.data.column("revenue").to_pylist() == [10.0, 20.0, 5.0]
    # Renames never resurrect reserved columns.
    assert result.dropped_columns == RESERVED_IN_TOTALS_FIXTURE


def test_alias_collision_is_a_compile_error():
    table = fixture_table("totals_with_sidecars.ndjson")

    with pytest.raises(CompileError, match="duplicate column name"):
        normalize(table, aliases={"TOTAL_SALE_PRICE": "STATE"})


def test_alias_can_swap_names_without_colliding():
    table = pa.table(
        {
            "a": pa.array([1], pa.int64()),
            "b": pa.array([2], pa.int64()),
        }
    )

    result = normalize(table, aliases={"a": "b", "b": "a"})

    assert result.data.column_names == ["b", "a"]
    assert result.data.column("b").to_pylist() == [1]


def test_dictionary_encoded_indicator_columns_are_understood():
    indicator = pa.array(["", "column_total", None], pa.string()).dictionary_encode()
    table = pa.table(
        {
            "users.state": pa.array(["CA", None, "NY"], pa.string()),
            "$omni_column_total_indicator": indicator,
        }
    )

    result = normalize(table, keep_totals=True)

    # The empty string is not a total; the dictionary value is decoded before comparing.
    assert result.data.num_rows == 2
    assert result.data.column("users.state").to_pylist() == ["CA", "NY"]
    assert result.totals is not None
    assert result.totals.num_rows == 1
    assert result.total_row_types == ("column_total",)
