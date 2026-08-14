"""Unit tests for the typed query-contract port (docs/CONTRACT_NOTES.md §2.1 and §3)."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import pytest

from omniframes.compile.querymodel import (
    DATE_GRAINS,
    DEFAULT_FETCH_LIMIT,
    DURATION_GRAINS,
    GRAINS,
    GRAND_TOTAL_KEY,
    HIGH_LIMIT_THRESHOLD,
    QUERY_VERSION,
    UNSET,
    BooleanFilter,
    CachePolicy,
    Calculation,
    CompositeFilter,
    DateFilter,
    DateFilterKind,
    Filter,
    FilterConjunction,
    NullFilter,
    NullSort,
    NumberFilter,
    NumberFilterKind,
    Query,
    QueryFilter,
    ResultType,
    RunRequest,
    Sort,
    StringFilter,
    StringFilterKind,
    UserAttributeFilter,
    is_valid_grain,
)
from omniframes.errors import CompileError

MODEL_ID = "123e4567-e89b-12d3-a456-426614174000"
OTHER_UUID = "0f8fad5b-d9cb-469f-a165-70867728950e"


def make_query(**overrides: Any) -> Query:
    """A valid topic query; overrides let a test poke exactly one thing."""
    base: dict[str, Any] = {
        "model_id": MODEL_ID,
        "fields": ["users.state", "order_items.total_sale_price"],
        "join_paths_from_topic_name": "order_items",
    }
    base.update(overrides)
    return Query(**base)


# ---------------------------------------------------------------------------------------
# Construction / serialization basics
# ---------------------------------------------------------------------------------------


def test_minimal_query_wire_shape() -> None:
    wire = make_query().to_wire()

    assert wire == {
        "modelId": MODEL_ID,
        "table": "",
        "fields": ["users.state", "order_items.total_sale_price"],
        "filters": {},
        "sorts": [],
        "limit": DEFAULT_FETCH_LIMIT,
        "offset": 0,
        "pivots": [],
        "calculations": [],
        "fill_fields": [],
        "column_totals": {},
        "row_totals": {},
        "userEditedSQL": "",
        "default_group_by": True,
        "version": QUERY_VERSION,
        "join_paths_from_topic_name": "order_items",
    }


def test_version_is_always_nine() -> None:
    assert QUERY_VERSION == 9
    assert make_query().to_wire()["version"] == 9


@pytest.mark.parametrize("key", ["rewriteSql", "sqlSortsEnabled", "staticQueryReferences"])
def test_unset_optionals_are_omitted(key: str) -> None:
    assert key not in make_query().to_wire()


def test_join_paths_omitted_when_unset() -> None:
    wire = Query(model_id=MODEL_ID, fields=["users.state"], table="users").to_wire()
    assert "join_paths_from_topic_name" not in wire
    assert wire["table"] == "users"


def test_sequences_are_normalized_and_query_is_frozen() -> None:
    mutable_fields = ["users.state"]
    query = Query(model_id=MODEL_ID, fields=mutable_fields, table="users")
    mutable_fields.append("users.country")

    assert query.fields == ("users.state",)
    assert query == Query(model_id=MODEL_ID, fields=("users.state",), table="users")
    with pytest.raises(dataclasses.FrozenInstanceError):
        query.table = "order_items"  # type: ignore[misc]


def test_filters_mapping_is_copied() -> None:
    filters: dict[str, Filter] = {"users.state": NullFilter()}
    query = make_query(filters=filters)
    filters["users.country"] = NullFilter()

    assert set(query.filters) == {"users.state"}


def test_totals_render_as_aggregation_records() -> None:
    wire = make_query(
        column_totals=["order_items.total_sale_price", GRAND_TOTAL_KEY],
        row_totals=["order_items.total_sale_price"],
    ).to_wire()

    assert wire["column_totals"] == {
        "order_items.total_sale_price": {"type": "aggregation"},
        "::total::": {"type": "aggregation"},
    }
    assert wire["row_totals"] == {"order_items.total_sale_price": {"type": "aggregation"}}


# ---------------------------------------------------------------------------------------
# limit: unset vs None vs int
# ---------------------------------------------------------------------------------------


def test_limit_unset_sends_the_default_fetch_limit() -> None:
    query = make_query()

    assert query.limit is UNSET
    assert query.limit_is_unset is True
    assert query.effective_limit == DEFAULT_FETCH_LIMIT
    assert query.to_wire()["limit"] == DEFAULT_FETCH_LIMIT


def test_limit_none_sends_null() -> None:
    query = make_query(limit=None)

    assert query.limit_is_unset is False
    assert query.effective_limit is None
    assert query.to_wire()["limit"] is None


def test_limit_int_sends_that_int() -> None:
    query = make_query(limit=25)

    assert query.limit_is_unset is False
    assert query.effective_limit == 25
    assert query.to_wire()["limit"] == 25


def test_limit_trichotomy_is_three_distinct_states() -> None:
    unset, explicit_none, explicit_int = make_query(), make_query(limit=None), make_query(limit=1)

    assert unset != explicit_none
    assert unset != explicit_int
    assert explicit_none != explicit_int


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (UNSET, False),
        (10, False),
        (HIGH_LIMIT_THRESHOLD, False),
        (HIGH_LIMIT_THRESHOLD + 1, True),
        (None, True),
    ],
)
def test_high_limit_detection(limit: Any, expected: bool) -> None:
    assert make_query(limit=limit).is_high_limit is expected


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_is_rejected(limit: int) -> None:
    with pytest.raises(CompileError, match="positive integer"):
        make_query(limit=limit).validate()


def test_bool_limit_is_rejected() -> None:
    with pytest.raises(CompileError, match="not a bool"):
        make_query(limit=True).validate()


# ---------------------------------------------------------------------------------------
# Filter arms
# ---------------------------------------------------------------------------------------


def test_string_filter_wire() -> None:
    flt = StringFilter(StringFilterKind.EQUALS, ["CA", "NY"], case_insensitive=True)

    assert flt.to_wire() == {
        "type": "string",
        "kind": "EQUALS",
        "values": ["CA", "NY"],
        "case_insensitive": True,
    }


def test_string_filter_is_empty_takes_no_values() -> None:
    flt = StringFilter(StringFilterKind.IS_EMPTY)
    flt.validate()

    assert flt.to_wire() == {"type": "string", "kind": "IS_EMPTY", "values": []}
    with pytest.raises(CompileError, match="IS_EMPTY takes no values"):
        StringFilter(StringFilterKind.IS_EMPTY, ["x"]).validate()


def test_string_filter_requires_values() -> None:
    with pytest.raises(CompileError, match="at least one value"):
        StringFilter(StringFilterKind.CONTAINS).validate()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (42, "42"),
        (-7, "-7"),
        (99.5, "99.5"),
        (Decimal("10.50"), "10.50"),
        ("42", "42"),
        ("1e3", "1e3"),
    ],
)
def test_number_filter_values_are_coerced_to_strings(value: Any, expected: str) -> None:
    flt = NumberFilter(NumberFilterKind.EQUALS, [value])

    assert flt.to_wire()["values"] == [expected]
    assert all(isinstance(v, str) for v in flt.to_wire()["values"])


def test_number_filter_wire() -> None:
    flt = NumberFilter(NumberFilterKind.BETWEEN, [10, 100], is_inclusive=True)

    assert flt.to_wire() == {
        "type": "number",
        "kind": "BETWEEN",
        "values": ["10", "100"],
        "is_inclusive": True,
    }


def test_number_filter_rejects_bool() -> None:
    with pytest.raises(CompileError, match="boolean"):
        NumberFilter(NumberFilterKind.EQUALS, [True])


def test_number_filter_rejects_the_string_null() -> None:
    with pytest.raises(CompileError, match="NullFilter"):
        NumberFilter(NumberFilterKind.EQUALS, ["null"])


def test_number_filter_rejects_non_numeric_strings() -> None:
    with pytest.raises(CompileError, match="is not a number"):
        NumberFilter(NumberFilterKind.EQUALS, ["abc"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "NaN", "-Infinity"])
def test_number_filter_rejects_non_finite_values(value: Any) -> None:
    with pytest.raises(CompileError, match="not finite"):
        NumberFilter(NumberFilterKind.EQUALS, [value])


def test_number_filter_rejects_unsupported_types() -> None:
    unsupported: list[Any] = [object()]
    with pytest.raises(CompileError, match="int, float, Decimal or str"):
        NumberFilter(NumberFilterKind.EQUALS, unsupported)


def test_number_filter_between_needs_two_values() -> None:
    with pytest.raises(CompileError, match="exactly two values"):
        NumberFilter(NumberFilterKind.BETWEEN, [1]).validate()


def test_number_filter_needs_a_value() -> None:
    with pytest.raises(CompileError, match="at least one value"):
        NumberFilter(NumberFilterKind.GREATER_THAN).validate()


def test_date_filter_wire_omits_absent_sides() -> None:
    flt = DateFilter(DateFilterKind.ON_OR_AFTER, left_side="30 days ago")
    flt.validate()

    assert flt.to_wire() == {"type": "date", "kind": "ON_OR_AFTER", "left_side": "30 days ago"}


def test_date_filter_between_wire() -> None:
    flt = DateFilter(DateFilterKind.BETWEEN, "2024-01-01", "2024-02-01")
    flt.validate()

    assert flt.to_wire() == {
        "type": "date",
        "kind": "BETWEEN",
        "left_side": "2024-01-01",
        "right_side": "2024-02-01",
    }


@pytest.mark.parametrize(
    ("kind", "side"),
    [
        (DateFilterKind.BETWEEN, "left_side"),
        (DateFilterKind.ON_OR_AFTER, "left_side"),
        (DateFilterKind.BEFORE, "right_side"),
        (DateFilterKind.TIME_FOR_INTERVAL_DURATION, "left_side"),
        (DateFilterKind.TIME_FOR_UNIT_DURATION, "left_side"),
    ],
)
def test_date_filter_requires_its_sides(kind: DateFilterKind, side: str) -> None:
    with pytest.raises(CompileError, match=side):
        DateFilter(kind).validate()


@pytest.mark.parametrize(
    ("is_negative", "expected"),
    [
        (False, {"type": "boolean", "is_negative": False}),
        (True, {"type": "boolean", "is_negative": True}),
    ],
)
def test_boolean_filter_wire(is_negative: bool, expected: dict[str, Any]) -> None:
    assert BooleanFilter(is_negative=is_negative).to_wire() == expected


def test_boolean_filter_without_is_negative_is_a_placeholder() -> None:
    assert BooleanFilter().to_wire() == {"type": "boolean"}
    assert BooleanFilter(treat_nulls_as_false=True).to_wire() == {
        "type": "boolean",
        "treat_nulls_as_false": True,
    }


def test_null_filter_wire() -> None:
    assert NullFilter().to_wire() == {"type": "null"}
    assert NullFilter(is_negative=True).to_wire() == {"type": "null", "is_negative": True}


def test_user_attribute_filter_wire() -> None:
    assert UserAttributeFilter("sales_region").to_wire() == {
        "type": "user_attribute",
        "user_attribute_name": "sales_region",
    }
    with pytest.raises(CompileError, match="user_attribute_name"):
        UserAttributeFilter("").validate()


def test_query_filter_wire() -> None:
    flt = QueryFilter("users.id", "top_users", disregard_limit=True)

    assert flt.to_wire() == {
        "type": "query",
        "field_name": "users.id",
        "query_id": "top_users",
        "disregard_limit": True,
    }


@pytest.mark.parametrize(
    ("flt", "message"),
    [
        (QueryFilter("", "top_users"), "field_name"),
        (QueryFilter("users.id", ""), "query_id"),
    ],
)
def test_query_filter_requires_both_names(flt: QueryFilter, message: str) -> None:
    with pytest.raises(CompileError, match=message):
        flt.validate()


def test_composite_filter_nests_recursively() -> None:
    inner = CompositeFilter(
        FilterConjunction.AND,
        [
            NumberFilter(NumberFilterKind.GREATER_THAN, [10]),
            NumberFilter(NumberFilterKind.LESS_THAN, [100]),
        ],
    )
    outer = CompositeFilter(
        FilterConjunction.OR, [inner, NullFilter(is_negative=True)], is_negative=False
    )
    outer.validate()

    assert outer.to_wire() == {
        "type": "composite",
        "is_negative": False,
        "conjunction": "OR",
        "filters": [
            {
                "type": "composite",
                "conjunction": "AND",
                "filters": [
                    {"type": "number", "kind": "GREATER_THAN", "values": ["10"]},
                    {"type": "number", "kind": "LESS_THAN", "values": ["100"]},
                ],
            },
            {"type": "null", "is_negative": True},
        ],
    }


def test_composite_filter_has_no_depth_cap() -> None:
    flt: Filter = StringFilter(StringFilterKind.EQUALS, ["deep"])
    for _ in range(8):
        flt = CompositeFilter(FilterConjunction.AND, [flt])
    flt.validate()

    wire = flt.to_wire()
    depth = 0
    while wire["type"] == "composite":
        depth += 1
        wire = wire["filters"][0]
    assert depth == 8


def test_composite_filter_needs_children() -> None:
    with pytest.raises(CompileError, match="at least one child"):
        CompositeFilter(FilterConjunction.AND).validate()


def test_composite_filter_validates_children() -> None:
    flt = CompositeFilter(
        FilterConjunction.AND,
        [CompositeFilter(FilterConjunction.OR, [NumberFilter(NumberFilterKind.BETWEEN, [1])])],
    )
    with pytest.raises(CompileError, match="exactly two values"):
        flt.validate()


ALL_ARMS: list[tuple[str, Filter]] = [
    ("string", StringFilter(StringFilterKind.CONTAINS, ["a"])),
    ("number", NumberFilter(NumberFilterKind.EQUALS, [1])),
    ("date", DateFilter(DateFilterKind.ON_OR_AFTER, left_side="today")),
    ("boolean", BooleanFilter()),
    ("null", NullFilter()),
    ("composite", CompositeFilter(FilterConjunction.AND, [NullFilter()])),
    ("query", QueryFilter("users.id", "top_users")),
    ("user_attribute", UserAttributeFilter("region")),
]


@pytest.mark.parametrize(("expected_type", "flt"), ALL_ARMS)
def test_every_arm_reports_its_type(expected_type: str, flt: Filter) -> None:
    assert flt.to_wire()["type"] == expected_type


@pytest.mark.parametrize(("expected_type", "flt"), ALL_ARMS)
def test_every_arm_carries_the_common_keys(expected_type: str, flt: Filter) -> None:
    wire = dataclasses.replace(
        flt, is_negative=True, cancel_query_filter=True, ignore_if_unjoinable=True
    ).to_wire()

    assert wire["type"] == expected_type
    assert wire["is_negative"] is True
    assert wire["cancel_query_filter"] is True
    assert wire["ignore_if_unjoinable"] is True


@pytest.mark.parametrize(("expected_type", "flt"), ALL_ARMS)
def test_common_keys_are_omitted_when_unset(expected_type: str, flt: Filter) -> None:
    wire = flt.to_wire()

    assert "is_negative" not in wire
    assert "cancel_query_filter" not in wire
    assert "ignore_if_unjoinable" not in wire


# ---------------------------------------------------------------------------------------
# Sorts and calculations
# ---------------------------------------------------------------------------------------


def test_sort_wire_defaults_to_omni_default_null_sort() -> None:
    assert Sort("order_items.created_at[month]").to_wire() == {
        "column_name": "order_items.created_at[month]",
        "sort_descending": False,
        "null_sort": "OMNI_DEFAULT",
    }


@pytest.mark.parametrize("null_sort", list(NullSort))
def test_sort_null_sort_round_trips(null_sort: NullSort) -> None:
    wire = Sort("users.state", True, null_sort).to_wire()

    assert wire["null_sort"] == null_sort.value
    assert wire["sort_descending"] is True


def test_sort_requires_a_column_name() -> None:
    with pytest.raises(CompileError, match="column_name"):
        make_query(sorts=[Sort("")]).validate()


def test_calculation_wire_and_validation() -> None:
    calc = Calculation("margin", {"type": "field", "field_name": "order_items.sale_price"})

    assert calc.to_wire() == {
        "calc_name": "margin",
        "sql_expression": {"type": "field", "field_name": "order_items.sale_price"},
    }
    with pytest.raises(CompileError, match="sql_expression"):
        Calculation("margin").validate()


def test_calculation_name_must_be_selected() -> None:
    calc = Calculation("margin", {"type": "field"})
    with pytest.raises(CompileError, match=r"must also appear in query\.fields"):
        make_query(calculations=[calc]).validate()

    make_query(fields=["users.state", "margin"], calculations=[calc]).validate()


# ---------------------------------------------------------------------------------------
# Query.validate
# ---------------------------------------------------------------------------------------


def test_valid_query_passes() -> None:
    make_query(
        filters={"users.state": StringFilter(StringFilterKind.EQUALS, ["CA"])},
        sorts=[Sort("users.state")],
        limit=100,
        offset=10,
    ).validate()


def test_model_id_must_be_a_uuid() -> None:
    with pytest.raises(CompileError, match=r"query\.modelId must be a UUID"):
        make_query(model_id="bench_ecommerce").validate()


def test_fields_must_not_be_empty() -> None:
    with pytest.raises(CompileError, match=r"query\.fields must not be empty"):
        make_query(fields=[]).validate()


def test_negative_offset_is_rejected() -> None:
    with pytest.raises(CompileError, match="offset"):
        make_query(offset=-1).validate()


def test_version_must_be_nine() -> None:
    with pytest.raises(CompileError, match=r"query\.version must be 9"):
        make_query(version=8).validate()


def test_query_needs_a_table_or_a_topic() -> None:
    with pytest.raises(CompileError, match="join_paths_from_topic_name"):
        Query(model_id=MODEL_ID, fields=["users.state"]).validate()


def test_pivots_must_be_selected() -> None:
    with pytest.raises(CompileError, match="pivot fields must also appear"):
        make_query(pivots=["users.country"]).validate()

    make_query(pivots=["users.state"]).validate()


def test_empty_filter_key_is_rejected() -> None:
    with pytest.raises(CompileError, match="non-empty field names"):
        make_query(filters={"": NullFilter()}).validate()


def test_filters_are_validated_through_the_query() -> None:
    with pytest.raises(CompileError, match="at least one value"):
        make_query(filters={"users.state": StringFilter(StringFilterKind.EQUALS)}).validate()


# ---------------------------------------------------------------------------------------
# Raw SQL jobs and static query references
# ---------------------------------------------------------------------------------------


def test_for_sql_sets_the_raw_sql_trio() -> None:
    query = Query.for_sql(MODEL_ID, "SELECT 1 AS n")
    query.validate()
    wire = query.to_wire()

    assert wire["userEditedSQL"] == "SELECT 1 AS n"
    assert wire["rewriteSql"] is False
    assert wire["sqlSortsEnabled"] is True
    assert wire["fields"] == []


def test_sql_without_rewrite_sql_false_is_rejected() -> None:
    with pytest.raises(CompileError, match="rewriteSql=False"):
        make_query(user_edited_sql="SELECT 1").validate()

    with pytest.raises(CompileError, match="rewriteSql=False"):
        make_query(user_edited_sql="SELECT 1", rewrite_sql=True).validate()


def test_sql_sorts_enabled_can_be_turned_off() -> None:
    wire = Query.for_sql(MODEL_ID, "SELECT 1", sql_sorts_enabled=False).to_wire()

    assert wire["sqlSortsEnabled"] is False


# ---------------------------------------------------------------------------------------
# OmniSQL jobs (CONTRACT_NOTES §3.6, docs/SQLTIER.md §2)
# ---------------------------------------------------------------------------------------

OMNISQL = (
    "SELECT ${users.state}, ${order_items.sale_price_sum}\n"
    "FROM ${order_items}\n"
    "GROUP BY 1\n"
    "LIMIT 50000"
)


def test_for_omnisql_wire_shape() -> None:
    """The parsed path is selected by what is ABSENT, so pin the whole object."""
    query = Query.for_omnisql(MODEL_ID, OMNISQL)
    query.validate()

    assert query.to_wire() == {
        "modelId": MODEL_ID,
        "table": "",
        "fields": [],
        "filters": {},
        "sorts": [],
        "limit": DEFAULT_FETCH_LIMIT,
        "offset": 0,
        "pivots": [],
        "calculations": [],
        "fill_fields": [],
        "column_totals": {},
        "row_totals": {},
        "userEditedSQL": OMNISQL,
        "default_group_by": True,
        "version": QUERY_VERSION,
    }


@pytest.mark.parametrize("key", ["rewriteSql", "staticQueryReferences", "sqlSortsEnabled"])
def test_omnisql_omits_the_keys_that_would_change_the_path(key: str) -> None:
    """``rewriteSql: false`` — even ``true`` — takes the statement off the parsed path."""
    assert key not in Query.for_omnisql(MODEL_ID, OMNISQL).to_wire()


def test_the_omnisql_flag_is_not_a_wire_key() -> None:
    query = Query.for_omnisql(MODEL_ID, OMNISQL)

    assert query.omnisql is True
    assert not [key for key in query.to_wire() if key.lower() == "omnisql"]
    assert not [key for key in query.to_reference_wire() if key.lower() == "omnisql"]


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(UNSET, DEFAULT_FETCH_LIMIT), (250, 250), (None, None), (HIGH_LIMIT_THRESHOLD + 1, 50_001)],
)
def test_for_omnisql_mirrors_the_statements_limit(limit: Any, expected: int | None) -> None:
    """Bookkeeping only — the server ignores it here — but the truncation warning reads it."""
    query = Query.for_omnisql(MODEL_ID, OMNISQL, limit=limit)
    query.validate()

    assert query.to_wire()["limit"] == expected
    assert query.effective_limit == expected
    assert query.limit_is_unset is (limit is UNSET)


def test_for_omnisql_mirrors_the_offset() -> None:
    query = Query.for_omnisql(MODEL_ID, OMNISQL, limit=10, offset=20)
    query.validate()

    assert query.to_wire()["offset"] == 20


def test_for_sql_is_never_the_parsed_path() -> None:
    """The structural half of the guarantee: user SQL is built here and cannot be OmniSQL."""
    assert Query.for_sql(MODEL_ID, "SELECT 1 AS n").omnisql is False


@pytest.mark.parametrize("rewrite_sql", [True, False])
def test_omnisql_with_a_rewrite_sql_key_is_rejected(rewrite_sql: bool) -> None:
    query = Query(model_id=MODEL_ID, user_edited_sql=OMNISQL, omnisql=True, rewrite_sql=rewrite_sql)
    with pytest.raises(CompileError, match="must leave rewriteSql unset"):
        query.validate()


def test_the_omnisql_flag_without_sql_is_rejected() -> None:
    with pytest.raises(CompileError, match="carries no SQL"):
        make_query(omnisql=True).validate()


def test_omnisql_with_sql_sorts_enabled_is_rejected() -> None:
    query = Query(model_id=MODEL_ID, user_edited_sql=OMNISQL, omnisql=True, sql_sorts_enabled=False)
    with pytest.raises(CompileError, match="sqlSortsEnabled belongs to verbatim SQL jobs"):
        query.validate()


def test_omnisql_with_static_query_references_is_rejected() -> None:
    """A refKey is not a table on this path (CONTRACT_NOTES §3.5, live-refuted)."""
    reference = Query(model_id=OTHER_UUID, fields=["users.id"], table="users")
    query = Query(
        model_id=MODEL_ID,
        user_edited_sql=OMNISQL,
        omnisql=True,
        static_query_references={"ref_1": reference},
    )
    with pytest.raises(CompileError, match="cannot reference staticQueryReferences"):
        query.validate()


def test_static_query_reference_adds_snake_case_model_id() -> None:
    reference = Query(model_id=OTHER_UUID, fields=["users.id"], table="users", limit=10)
    query = make_query(
        filters={"users.id": QueryFilter("users.id", "top_users")},
        static_query_references={"top_users": reference},
    )
    query.validate()
    wire = query.to_wire()

    assert set(wire["staticQueryReferences"]) == {"top_users"}
    reference_wire = wire["staticQueryReferences"]["top_users"]
    assert reference_wire["model_id"] == OTHER_UUID
    assert reference_wire["modelId"] == OTHER_UUID
    assert reference_wire["limit"] == 10
    assert reference_wire["version"] == QUERY_VERSION


def test_query_filter_must_reference_a_known_key() -> None:
    query = make_query(filters={"users.id": QueryFilter("users.id", "missing")})
    with pytest.raises(CompileError, match="key of staticQueryReferences"):
        query.validate()


def test_nested_query_filter_reference_is_checked() -> None:
    query = make_query(
        filters={
            "users.id": CompositeFilter(
                FilterConjunction.OR, [NullFilter(), QueryFilter("users.id", "missing")]
            )
        }
    )
    with pytest.raises(CompileError, match="staticQueryReferences"):
        query.validate()


def test_referenced_queries_are_validated() -> None:
    reference = Query(model_id="not-a-uuid", fields=["users.id"], table="users")
    query = make_query(static_query_references={"top_users": reference})
    with pytest.raises(CompileError, match="must be a UUID"):
        query.validate()


# ---------------------------------------------------------------------------------------
# The request envelope
# ---------------------------------------------------------------------------------------


def test_envelope_minimal_wire() -> None:
    request = RunRequest(make_query())
    request.validate()

    assert request.to_wire() == {"query": make_query().to_wire()}


def test_envelope_full_wire() -> None:
    request = RunRequest(
        make_query(),
        branch_id=OTHER_UUID,
        cache=CachePolicy.SKIP_CACHE_AND_REBUILD_EXTRACTS,
        result_type=ResultType.CSV,
        format_results=True,
        plan_only=False,
        timezone="America/Los_Angeles",
        user_id=MODEL_ID,
        workbook_url=True,
    )
    request.validate()
    wire = request.to_wire()

    assert wire["branchId"] == OTHER_UUID
    assert wire["cache"] == "SkipCacheAndRebuildExtracts"
    assert wire["resultType"] == "csv"
    assert wire["formatResults"] is True
    assert wire["planOnly"] is False
    assert wire["timezone"] == "America/Los_Angeles"
    assert wire["userId"] == MODEL_ID
    assert wire["workbookUrl"] is True


@pytest.mark.parametrize("policy", list(CachePolicy))
def test_cache_policy_values(policy: CachePolicy) -> None:
    assert RunRequest(make_query(), cache=policy).to_wire()["cache"] == policy.value


def test_plan_only_with_result_type_is_rejected() -> None:
    request = RunRequest(make_query(), plan_only=True, result_type=ResultType.JSON)
    with pytest.raises(CompileError, match="planOnly and resultType"):
        request.validate()


def test_plan_only_with_workbook_url_is_rejected() -> None:
    request = RunRequest(make_query(), plan_only=True, workbook_url=True)
    with pytest.raises(CompileError, match="planOnly and workbookUrl"):
        request.validate()


def test_workbook_url_with_static_query_references_is_rejected() -> None:
    reference = Query(model_id=OTHER_UUID, fields=["users.id"], table="users")
    request = RunRequest(
        make_query(static_query_references={"top_users": reference}), workbook_url=True
    )
    with pytest.raises(CompileError, match="workbookUrl is not supported"):
        request.validate()


@pytest.mark.parametrize("format_results", [True, False])
def test_format_results_without_result_type_is_rejected(format_results: bool) -> None:
    request = RunRequest(make_query(), format_results=format_results)
    with pytest.raises(CompileError, match="formatResults cannot be provided without resultType"):
        request.validate()

    RunRequest(make_query(), format_results=format_results, result_type=ResultType.JSON).validate()


def test_null_limit_with_pivots_is_rejected() -> None:
    query = make_query(limit=None, pivots=["users.state"])
    with pytest.raises(CompileError, match="Unlimited limit"):
        RunRequest(query).validate()

    RunRequest(query, result_type=ResultType.CSV).validate()


def test_null_limit_without_pivots_is_fine() -> None:
    RunRequest(make_query(limit=None)).validate()


def test_pivots_with_a_real_limit_are_fine() -> None:
    RunRequest(make_query(pivots=["users.state"], limit=100)).validate()
    RunRequest(make_query(pivots=["users.state"])).validate()


def test_user_id_in_both_places_is_rejected() -> None:
    request = RunRequest(make_query(), user_id=MODEL_ID)
    with pytest.raises(CompileError, match="but not both"):
        request.validate(user_id_query_param=MODEL_ID)

    request.validate()
    RunRequest(make_query()).validate(user_id_query_param=MODEL_ID)


@pytest.mark.parametrize("kwargs", [{"branch_id": "nope"}, {"user_id": "nope"}])
def test_envelope_uuid_fields_are_checked(kwargs: dict[str, Any]) -> None:
    with pytest.raises(CompileError, match="must be a UUID"):
        RunRequest(make_query(), **kwargs).validate()


def test_envelope_validates_the_query() -> None:
    with pytest.raises(CompileError, match=r"query\.fields must not be empty"):
        RunRequest(make_query(fields=[])).validate()


def test_envelope_is_frozen() -> None:
    request = RunRequest(make_query())
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.plan_only = True  # type: ignore[misc]


# ---------------------------------------------------------------------------------------
# Grains (§3.2)
# ---------------------------------------------------------------------------------------


def test_grains_are_the_union_of_date_and_duration_grains() -> None:
    assert GRAINS == DATE_GRAINS | DURATION_GRAINS
    assert isinstance(GRAINS, frozenset)
    assert len(DATE_GRAINS) == 23
    assert len(DURATION_GRAINS) == 8


@pytest.mark.parametrize(
    "grain",
    ["year", "month", "week", "date", "hour_of_day", "day_of_week_num", "fiscal_quarter", "days"],
)
def test_known_grains_are_valid(grain: str) -> None:
    assert is_valid_grain(grain)


@pytest.mark.parametrize("grain", ["MONTH", "Month", "Day_Of_Week_Num"])
def test_grain_matching_is_case_insensitive(grain: str) -> None:
    assert is_valid_grain(grain)


@pytest.mark.parametrize("grain", ["monthly", "", "day", "week_num", "quarter_of_the_year"])
def test_unknown_grains_are_invalid(grain: str) -> None:
    assert not is_valid_grain(grain)
