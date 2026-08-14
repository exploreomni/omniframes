"""Result normalization against checked-in wire bodies (CONTRACT_NOTES §2.7, SQLTIER §5/§3.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pytest

from omniframes.compile.sqlgen import EXPR_ALIAS_PREFIX
from omniframes.errors import CompileError
from omniframes.transport.arrow import decode_result
from omniframes.transport.ndjson import parse_response
from omniframes.transport.normalize import (
    GRAIN_RAW_SUFFIX,
    TOTAL_INDICATOR_COLUMNS,
    collapse_grain_names,
    is_reserved_column,
    normalize,
    resolve_aliases,
)
from tests.wire import read_fixture

MONTH = "order_items.created_at[month]"
MONTH_RAW = f"{MONTH}{GRAIN_RAW_SUFFIX}"
MONTHS = [datetime(2026, m, 1, tzinfo=UTC) for m in (1, 2, 3)]

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


def test_expression_aliases_match_by_suffix_under_an_unpredictable_scope_prefix():
    # The fixture scopes `of_expr_1` to `products` and `of_expr_2` to `inventory_items` — neither
    # is a view the expression mentions, which is exactly the point (docs/SQLTIER.md §3.2).
    table = fixture_table("omnisql_expressions.ndjson")

    result = normalize(
        table,
        fixture_summary_fields("omnisql_expressions.ndjson"),
        aliases={MONTH: "month", "of_expr_1": "buyers", "of_expr_2": "revenue_per_buyer"},
    )

    assert result.data.column_names == ["month", "users.state", "buyers", "revenue_per_buyer"]
    assert result.data.column("buyers").to_pylist() == [12, 7, 3]
    assert result.data.column("revenue_per_buyer").to_pylist() == [102.875, 127.175, 14.0]


def test_an_unaliased_expression_column_keeps_its_scoped_wire_name():
    table = fixture_table("omnisql_expressions.ndjson")

    result = normalize(table)

    assert result.data.column_names == [
        MONTH,
        "users.state",
        "products.of_expr_1",
        "inventory_items.of_expr_2",
    ]


def test_a_generated_alias_that_matches_no_column_is_ignored():
    table = pa.table({"users.state": pa.array(["CA"], pa.string())})

    result = normalize(table, aliases={"of_expr_9": "gone"})

    assert result.data.column_names == ["users.state"]


def test_an_ambiguous_generated_alias_is_refused():
    table = pa.table(
        {
            "users.of_expr_1": pa.array([1], pa.int64()),
            "products.of_expr_1": pa.array([2], pa.int64()),
        }
    )

    with pytest.raises(CompileError, match="matches 2 result columns"):
        normalize(table, aliases={"of_expr_1": "buyers"})


def test_suffix_matching_never_touches_a_bare_wire_name():
    # A bare ref is renamed by equality only: `state` must not suffix-match `users.state`.
    table = pa.table({"users.state": pa.array(["CA"], pa.string())})

    result = normalize(table, aliases={"state": "s"})

    assert result.data.column_names == ["users.state"]


def test_the_generated_alias_shape_matches_what_the_compiler_emits():
    """The transport does not import the compiler, so the two spellings are pinned here."""
    assert resolve_aliases(
        (f"users.{EXPR_ALIAS_PREFIX}1",), {f"{EXPR_ALIAS_PREFIX}1": "buyers"}
    ) == ("buyers",)


# --------------------------------------------------------------------------------------------
# Formatted grain pairs (CONTRACT_NOTES §2.7, docs/SQLTIER.md §5)
# --------------------------------------------------------------------------------------------


def test_a_grain_pair_collapses_to_the_raw_timestamp():
    table = fixture_table("grain_pair.ndjson")
    assert table.column_names == [MONTH_RAW, "order_items.total_sale_price", MONTH]

    result = normalize(table, fixture_summary_fields("grain_pair.ndjson"))

    # The `__raw` half keeps the item's position and loses the suffix; the formatted string,
    # appended last on the wire, is gone.
    assert result.data.column_names == [MONTH, "order_items.total_sale_price"]
    assert result.data.column(MONTH).to_pylist() == MONTHS
    assert result.data.column(MONTH).type == pa.timestamp("us", tz="UTC")


def test_a_grain_pair_collapses_on_the_omnisql_path_too():
    table = fixture_table("omnisql_expressions.ndjson")

    result = normalize(table, fixture_summary_fields("omnisql_expressions.ndjson"))

    assert MONTH_RAW not in result.data.column_names
    assert result.data.column(MONTH).to_pylist() == [MONTHS[0], MONTHS[1], MONTHS[0]]


def test_an_alias_over_a_grain_names_the_collapsed_column():
    table = fixture_table("grain_pair.ndjson")

    result = normalize(table, aliases={MONTH: "month"})

    assert result.data.column_names == ["month", "order_items.total_sale_price"]
    assert result.data.column("month").to_pylist() == MONTHS


def test_the_collapse_applies_to_totals_rows_as_well():
    table = pa.table(
        {
            MONTH_RAW: pa.array([MONTHS[0], None], pa.timestamp("us", tz="UTC")),
            "order_items.total_sale_price": pa.array([10.0, 30.0], pa.float64()),
            MONTH: pa.array(["2026-01", None], pa.string()),
            "$omni_column_total_indicator": pa.array([None, "::total::"], pa.string()),
        }
    )

    result = normalize(table, keep_totals=True)

    assert result.data.column_names == [MONTH, "order_items.total_sale_price"]
    assert result.totals is not None
    assert result.totals.column_names == result.data.column_names
    assert result.total_row_types == ("::total::",)


def test_a_lone_raw_column_is_left_alone():
    table = pa.table({MONTH_RAW: pa.array(MONTHS, pa.timestamp("us", tz="UTC"))})

    result = normalize(table)

    assert result.data.column_names == [MONTH_RAW]


def test_a_user_column_that_merely_ends_in_raw_passes_through():
    # No `[grain]` bracket, so this is an ordinary pair of raw-SQL columns, not a §2.7 sidecar.
    table = pa.table(
        {
            "revenue__raw": pa.array([1.0], pa.float64()),
            "revenue": pa.array(["$1.00"], pa.string()),
        }
    )

    result = normalize(table)

    assert result.data.column_names == ["revenue__raw", "revenue"]


def test_collapse_grain_names_is_a_no_op_without_a_pair():
    names = ("users.state", MONTH, "order_items.total_sale_price")

    assert collapse_grain_names(names) == (names, names)


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
