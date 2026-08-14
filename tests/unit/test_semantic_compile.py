"""Unit tests for the tier-1 compiler (docs/INTERNALS.md §3/§4, CONTRACT_NOTES §3)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest

from omniframes import functions as F
from omniframes.column import Column, Expr, FieldRef, SortKey
from omniframes.compile.querymodel import (
    DEFAULT_FETCH_LIMIT,
    DEFAULT_SERVER_LIMIT,
    GRAINS,
    UNSET,
    CachePolicy,
    Query,
)
from omniframes.compile.semantic import (
    NUMBER_GRAINS,
    STRING_GRAINS,
    TIMESTAMP_GRAINS,
    CannotCompile,
    EnvelopeOptions,
    RemoteStep,
    SemanticCompilation,
    alias_map,
    build_envelope,
    compile_filters,
    compile_plan,
    compile_semantic,
    display_name,
    predicate_to_filters,
    split_grain,
    try_semantic,
    wire_name,
)
from omniframes.errors import CompileError
from omniframes.plan import (
    Aggregate,
    Filter,
    Join,
    JoinHow,
    Limit,
    MapPandas,
    PlanNode,
    Project,
    SavedQueryScan,
    Scan,
    Sort,
    SqlScan,
    TopicScan,
    Union,
    ViewScan,
    WithColumn,
)

MODEL_ID = "3f2b1a0c-9d8e-4c7b-a6f5-000000000001"
BRANCH_ID = "9c3b1e5e-2f4a-4d1b-9a7e-6b0f2d8c4a11"
SCAN = Scan(TopicScan("bench_ecommerce", MODEL_ID, "order_items", "order_items"))


def filters_of(predicate: Column, **kwargs: Any) -> dict[str, Any]:
    """The wire payload of every filter entry a predicate compiles to."""
    compiled = predicate_to_filters(predicate.expr, **kwargs)
    return {name: flt.to_wire() for name, flt in compiled.items()}


def one_filter(predicate: Column, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    compiled = filters_of(predicate, **kwargs)
    assert len(compiled) == 1, compiled
    return next(iter(compiled.items()))


def compiled(plan: PlanNode) -> Any:
    return compile_semantic(plan)


def selected(*columns: str | Column) -> PlanNode:
    return Project(SCAN, tuple(F.col(column) for column in columns))


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------


def test_wire_names() -> None:
    assert wire_name(FieldRef("users.state")) == "users.state"
    assert wire_name(FieldRef("order_items.created_at", "month")) == (
        "order_items.created_at[month]"
    )
    assert wire_name(F.measure("order_items.count").expr) == "order_items.count"


def test_an_adhoc_aggregation_has_no_wire_name_but_has_a_display_name() -> None:
    agg = F.count_distinct("users.id").expr

    with pytest.raises(CannotCompile, match="no field name on the wire"):
        wire_name(agg)
    assert display_name(agg) == "count_distinct(users.id)"
    assert display_name(F.sum("order_items.sale_price").expr) == "sum(order_items.sale_price)"


def test_split_grain_round_trips_and_ignores_unknown_suffixes() -> None:
    assert split_grain("order_items.created_at[month]") == ("order_items.created_at", "month")
    assert split_grain("users.state") == ("users.state", None)
    assert split_grain("users.array[0]") == ("users.array[0]", None)


def test_the_grain_classes_partition_the_wire_grain_list() -> None:
    """Every grain the wire accepts must have a documented filter shape."""
    assert TIMESTAMP_GRAINS | NUMBER_GRAINS | STRING_GRAINS == GRAINS
    assert not TIMESTAMP_GRAINS & NUMBER_GRAINS
    assert not TIMESTAMP_GRAINS & STRING_GRAINS
    assert not NUMBER_GRAINS & STRING_GRAINS


# --------------------------------------------------------------------------------------
# Aliases
# --------------------------------------------------------------------------------------


def test_alias_map_is_alias_to_wire_name() -> None:
    columns = (F.col("users.state").alias("state"), F.col("users.age"))

    assert alias_map(columns) == {"state": "users.state"}


def test_duplicate_aliases_are_a_build_time_error() -> None:
    with pytest.raises(CompileError, match="used twice"):
        alias_map((F.col("users.state").alias("x"), F.col("users.age").alias("x")))


def test_two_aliases_for_one_field_are_a_build_time_error() -> None:
    with pytest.raises(CompileError, match="aliased twice"):
        alias_map((F.col("users.state").alias("a"), F.col("users.state").alias("b")))


def test_an_alias_may_not_shadow_another_selected_field() -> None:
    with pytest.raises(CompileError, match="shadows"):
        alias_map((F.col("users.state"), F.col("users.age").alias("users.state")))


def test_aliasing_a_column_to_its_own_name_is_harmless() -> None:
    assert alias_map((F.col("users.state").alias("users.state"),)) == {"users.state": "users.state"}


# --------------------------------------------------------------------------------------
# Filters: comparisons
# --------------------------------------------------------------------------------------


def test_string_equality() -> None:
    assert one_filter(F.col("users.state") == "California") == (
        "users.state",
        {"type": "string", "kind": "EQUALS", "values": ["California"]},
    )


def test_string_inequality_is_negated_equals() -> None:
    _, payload = one_filter(F.col("users.state") != "California")

    assert payload == {
        "type": "string",
        "kind": "EQUALS",
        "values": ["California"],
        "is_negative": True,
    }


def test_number_values_go_on_the_wire_as_strings() -> None:
    _, payload = one_filter(F.col("order_items.sale_price") == Decimal("10.50"))

    assert payload["values"] == ["10.50"], "the raw path has no number->string transform"


@pytest.mark.parametrize(
    ("build", "kind", "inclusive"),
    [
        (lambda c: c < 10, "LESS_THAN", False),
        (lambda c: c <= 10, "LESS_THAN", True),
        (lambda c: c > 10, "GREATER_THAN", False),
        (lambda c: c >= 10, "GREATER_THAN", True),
    ],
)
def test_number_ordering_uses_is_inclusive(build: Any, kind: str, inclusive: bool) -> None:
    _, payload = one_filter(build(F.col("users.age")))

    assert payload["kind"] == kind
    assert payload["is_inclusive"] is inclusive


def test_literal_on_the_left_flips_the_operator() -> None:
    _, payload = one_filter(F.lit(10) < F.col("users.age"))

    assert payload["kind"] == "GREATER_THAN"
    assert payload["is_inclusive"] is False


def test_boolean_fields_compile_to_the_boolean_arm() -> None:
    assert one_filter(F.col("order_items.returned")) == (
        "order_items.returned",
        {"type": "boolean", "is_negative": False},
    )
    assert one_filter(~F.col("order_items.returned")) == (
        "order_items.returned",
        {"type": "boolean", "is_negative": True},
    )
    assert one_filter(F.col("order_items.returned") == False)[1] == (  # noqa: E712
        {"type": "boolean", "is_negative": True}
    )


def test_null_filters_both_ways() -> None:
    assert one_filter(F.col("users.state").is_null())[1] == {"type": "null"}
    assert one_filter(F.col("users.state").is_not_null())[1] == {
        "type": "null",
        "is_negative": True,
    }


def test_comparing_to_none_points_at_is_null() -> None:
    with pytest.raises(CompileError, match="is_null"):
        predicate_to_filters((F.col("users.state") == None).expr)  # noqa: E711


def test_date_comparisons_map_to_the_two_open_ended_kinds() -> None:
    created = F.col("order_items.created_at")

    assert one_filter(created >= date(2026, 1, 1))[1] == {
        "type": "date",
        "kind": "ON_OR_AFTER",
        "left_side": "2026-01-01",
    }
    assert one_filter(created < datetime(2026, 7, 1, 6, 30, 15))[1] == {
        "type": "date",
        "kind": "BEFORE",
        "right_side": "2026-07-01 06:30:15",
    }
    assert one_filter(created == datetime(2026, 3, 1))[1] == {
        "type": "date",
        "kind": "TIME_FOR_UNIT_DURATION",
        "left_side": "2026-03-01 00:00:00",
    }


def test_a_grain_pins_the_filter_type_so_string_date_literals_work() -> None:
    """With a grain the compiler knows it is a date, so the documented literal grammar applies."""
    month = F.col("order_items.created_at").grain("month")

    assert one_filter(month == "2026-03")[1] == {
        "type": "date",
        "kind": "TIME_FOR_UNIT_DURATION",
        "left_side": "2026-03",
    }


def test_the_bracketed_spelling_of_a_grain_obeys_the_same_grain_filter_rule() -> None:
    """``F.col("x[month]")`` is the name a user sees, and it must filter like ``.grain("month")``.

    ``select()`` and ``sorts[*].column_name`` both take the bracketed spelling verbatim, so a
    filter written against it has to land on the bare key with a date filter (CONTRACT_NOTES
    §3.1) rather than inventing a string filter keyed by a field that does not exist.
    """
    bracketed = F.col("order_items.created_at[month]")

    assert one_filter(bracketed == "2026-03") == (
        "order_items.created_at",
        {"type": "date", "kind": "TIME_FOR_UNIT_DURATION", "left_side": "2026-03"},
    )
    assert one_filter(bracketed >= date(2026, 3, 1))[0] == "order_items.created_at"
    # A number grain still keys off the bracketed name with the number arm.
    assert one_filter(F.col("order_items.created_at[month_num]") == 3) == (
        "order_items.created_at[month_num]",
        {"type": "number", "kind": "EQUALS", "values": ["3"]},
    )
    # A suffix that is not a grain is part of the field name, untouched.
    assert one_filter(F.col("users.array[0]") == 3)[0] == "users.array[0]"


def test_a_bare_field_compared_to_a_string_is_a_string_filter() -> None:
    """No catalog types at compile time: the literal decides.

    ``created_at == "2026-03"`` therefore compiles to a *string* filter.  Compare against a
    ``date``/``datetime`` (or add a ``.grain()``) to get a date filter — this is why the
    ``between()``/comparison docs say so.
    """
    _, payload = one_filter(F.col("order_items.created_at") == "2026-03")

    assert payload["type"] == "string"


def test_date_comparisons_the_wire_cannot_express_are_refused() -> None:
    created = F.col("order_items.created_at")

    with pytest.raises(CannotCompile, match="ON_OR_AFTER"):
        predicate_to_filters((created > date(2026, 1, 1)).expr)
    with pytest.raises(CannotCompile, match="ON_OR_AFTER"):
        predicate_to_filters((created <= date(2026, 1, 1)).expr)


def test_string_ordering_comparisons_are_refused() -> None:
    with pytest.raises(CannotCompile, match="no ordering kinds"):
        predicate_to_filters((F.col("users.state") > "M").expr)


# --------------------------------------------------------------------------------------
# Filters: the other arms
# --------------------------------------------------------------------------------------


def test_isin_becomes_a_multi_value_equals() -> None:
    assert one_filter(F.col("users.state").isin("Ohio", "Texas"))[1] == {
        "type": "string",
        "kind": "EQUALS",
        "values": ["Ohio", "Texas"],
    }
    assert one_filter(F.col("users.age").isin(21, 22))[1] == {
        "type": "number",
        "kind": "EQUALS",
        "values": ["21", "22"],
    }


def test_isin_over_mixed_types_is_refused() -> None:
    with pytest.raises(CannotCompile, match="mixes value types"):
        predicate_to_filters(F.col("users.state").isin("Ohio", 3).expr)


def test_string_predicates_and_case_insensitivity() -> None:
    assert one_filter(F.col("users.state").contains("cal", case_insensitive=True))[1] == {
        "type": "string",
        "kind": "CONTAINS",
        "values": ["cal"],
        "case_insensitive": True,
    }
    assert one_filter(F.col("users.state").like("%cal%"))[1] == {
        "type": "string",
        "kind": "SQL_LIKE",
        "values": ["%cal%"],
    }


def test_between_over_numbers_includes_both_ends() -> None:
    """PySpark-consistent — and NOT the wire's BETWEEN kind, whose upper bound is exclusive."""
    assert one_filter(F.col("users.age").between(21, 65))[1] == {
        "type": "composite",
        "conjunction": "AND",
        "filters": [
            {"type": "number", "kind": "GREATER_THAN", "values": ["21"], "is_inclusive": True},
            {"type": "number", "kind": "LESS_THAN", "values": ["65"], "is_inclusive": True},
        ],
    }


def test_between_over_dates_stays_half_open() -> None:
    """The date arm has ``>=`` and ``<`` only; omniframes will not invent date arithmetic."""
    assert one_filter(F.col("order_items.created_at").between(date(2026, 1, 1), date(2026, 7, 1)))[
        1
    ] == {
        "type": "date",
        "kind": "BETWEEN",
        "left_side": "2026-01-01",
        "right_side": "2026-07-01",
    }


def test_between_over_strings_is_refused() -> None:
    with pytest.raises(CannotCompile, match="no wire filter"):
        predicate_to_filters(F.col("users.state").between("a", "m").expr)


# --------------------------------------------------------------------------------------
# Filters: composition
# --------------------------------------------------------------------------------------


def test_top_level_and_splits_per_field() -> None:
    compiled_filters = filters_of(
        (F.col("users.state") == "California") & (F.col("order_items.status") == "complete")
    )

    assert set(compiled_filters) == {"users.state", "order_items.status"}


def test_two_predicates_on_one_field_merge_into_a_composite_and() -> None:
    age = F.col("users.age")
    _, payload = one_filter((age >= 21) & (age < 65))

    assert payload["type"] == "composite"
    assert payload["conjunction"] == "AND"
    assert [child["kind"] for child in payload["filters"]] == ["GREATER_THAN", "LESS_THAN"]


def test_or_within_one_field_becomes_a_composite_or() -> None:
    state = F.col("users.state")
    _, payload = one_filter((state == "California") | (state == "Texas"))

    assert payload["conjunction"] == "OR"
    assert len(payload["filters"]) == 2


def test_cross_field_or_is_refused_with_the_reason() -> None:
    predicate = (F.col("users.state") == "California") | (F.col("order_items.status") == "complete")

    with pytest.raises(CannotCompile, match="controls"):
        predicate_to_filters(predicate.expr)


def test_negating_a_composite_sets_is_negative_on_it() -> None:
    state = F.col("users.state")
    _, payload = one_filter(~((state == "California") | (state == "Texas")))

    assert payload["type"] == "composite"
    assert payload["is_negative"] is True


def test_double_negation_cancels() -> None:
    _, payload = one_filter(~~(F.col("users.state") == "California"))

    assert payload.get("is_negative") is False


def test_post_aggregation_filters_on_ad_hoc_aggregations_route_to_a_later_tier() -> None:
    """A governed measure has a server-side definition to hang a HAVING on; an ad-hoc agg does not."""
    with pytest.raises(CannotCompile, match="tier 2/3"):
        predicate_to_filters((F.sum("order_items.sale_price") > 10).expr)
    with pytest.raises(CannotCompile, match="tier 2/3"):
        predicate_to_filters(F.count_distinct("users.id").between(2, 5).expr)


def test_computed_expressions_route_to_a_later_tier() -> None:
    with pytest.raises(CannotCompile, match="tier 2/3"):
        predicate_to_filters(((F.col("users.age") + 1) > 21).expr)


def test_filters_reverse_resolve_aliases() -> None:
    aliases = {"state": "users.state", "month": "order_items.created_at[month]"}
    resolve = aliases.get

    assert set(filters_of(F.col("state") == "California", resolve=resolve)) == {"users.state"}
    # The alias points at a grained field, so the grain-filter rule still applies.
    assert set(filters_of(F.col("month") >= date(2026, 1, 1), resolve=resolve)) == {
        "order_items.created_at"
    }


def test_a_grain_cannot_be_applied_to_an_alias() -> None:
    predicate = F.col("state").grain("month") == "2026-01"

    with pytest.raises(CompileError, match="alias"):
        predicate_to_filters(predicate.expr, resolve={"state": "users.state"}.get)


# --------------------------------------------------------------------------------------
# Measure filters → HAVING (CONTRACT_NOTES §3.1)
# --------------------------------------------------------------------------------------

REVENUE = "order_items.total_sale_price"


def test_a_measure_comparison_is_a_measure_keyed_number_filter() -> None:
    """The wire keys it by the measure name; the server turns that into a HAVING."""
    assert one_filter(F.measure(REVENUE) > 50000) == (
        REVENUE,
        {"type": "number", "kind": "GREATER_THAN", "values": ["50000"], "is_inclusive": False},
    )


def test_measure_filter_values_are_strings_like_every_number_filter() -> None:
    _, payload = one_filter(F.measure(REVENUE) >= Decimal("50150.60"))

    assert payload["values"] == ["50150.60"]
    assert payload["is_inclusive"] is True


def test_several_conditions_on_one_measure_share_a_single_entry() -> None:
    """``filters`` is a map keyed by field, so one measure gets exactly one entry."""
    revenue = F.measure(REVENUE)
    name, payload = one_filter((revenue > 20000) & (revenue < 80000))

    assert name == REVENUE
    assert payload["type"] == "composite"
    assert payload["conjunction"] == "AND"
    assert [child["kind"] for child in payload["filters"]] == ["GREATER_THAN", "LESS_THAN"]


def test_an_or_over_one_measure_is_a_composite_or() -> None:
    revenue = F.measure(REVENUE)
    _, payload = one_filter((revenue < 20000) | (revenue > 100000))

    assert payload["conjunction"] == "OR"


def test_negating_a_measure_filter_sets_is_negative() -> None:
    _, payload = one_filter(~(F.measure(REVENUE) > 50000))

    assert payload["is_negative"] is True


def test_isin_and_between_on_a_measure_stay_on_the_number_arm() -> None:
    assert one_filter(F.measure("order_items.count").isin(10, 20))[1] == {
        "type": "number",
        "kind": "EQUALS",
        "values": ["10", "20"],
    }
    assert one_filter(F.measure("order_items.count").between(10, 20))[1]["type"] == "composite"


def test_a_measure_filter_is_tracked_as_a_having() -> None:
    compiled_set = compile_filters(
        ((F.col("users.state") == "Ohio") & (F.measure(REVENUE) > 10)).expr
    )

    assert set(compiled_set.entries) == {"users.state", REVENUE}
    assert compiled_set.measure_keys == frozenset({REVENUE})
    assert set(compiled_set.where) == {"users.state"}
    assert set(compiled_set.having) == {REVENUE}


def test_an_or_across_two_measures_is_refused() -> None:
    predicate = (F.measure(REVENUE) > 10) | (F.measure("order_items.count") > 10)

    with pytest.raises(CannotCompile, match="tier 2/3"):
        predicate_to_filters(predicate.expr)


def test_an_or_between_a_measure_and_a_dimension_is_refused() -> None:
    predicate = (F.measure(REVENUE) > 10) | (F.col("users.state") == "Ohio")

    with pytest.raises(CannotCompile, match="tier 2/3"):
        predicate_to_filters(predicate.expr)


@pytest.mark.parametrize(
    ("predicate", "reason"),
    [
        (F.measure(REVENUE).is_null(), "number arm"),
        (F.measure(REVENUE).contains("10"), "number arm"),
        (F.measure(REVENUE), "not a boolean"),
    ],
)
def test_measure_filters_the_wire_cannot_express_are_refused(
    predicate: Column, reason: str
) -> None:
    with pytest.raises(CannotCompile, match=reason):
        predicate_to_filters(predicate.expr)


# --------------------------------------------------------------------------------------
# The grain-filter rule
# --------------------------------------------------------------------------------------


def test_timestamp_grains_filter_the_bare_field_with_a_date_filter() -> None:
    month = F.col("order_items.created_at").grain("month")
    name, payload = one_filter(month >= date(2026, 1, 1))

    assert name == "order_items.created_at", "no bracket: the grain is a timestamp"
    assert payload["type"] == "date"


def test_number_grains_filter_the_bracketed_field_with_a_number_filter() -> None:
    hour = F.col("order_items.created_at").grain("hour_of_day")
    name, payload = one_filter(hour >= 8)

    assert name == "order_items.created_at[hour_of_day]"
    assert payload["type"] == "number"
    assert payload["values"] == ["8"]


def test_name_grains_filter_the_bracketed_field_with_a_string_filter() -> None:
    day = F.col("order_items.created_at").grain("day_of_week_name")
    name, payload = one_filter(day == "Monday")

    assert name == "order_items.created_at[day_of_week_name]"
    assert payload["type"] == "string"


def test_a_string_predicate_on_a_number_grain_is_refused() -> None:
    hour = F.col("order_items.created_at").grain("hour_of_day")

    with pytest.raises(CannotCompile, match="not text"):
        predicate_to_filters(hour.contains("8").expr)


# --------------------------------------------------------------------------------------
# Plan compilation
# --------------------------------------------------------------------------------------


def test_the_minimal_plan_compiles_to_a_topic_query() -> None:
    result = compiled(selected("users.state"))

    assert result.tier == 1
    assert result.fields == ("users.state",)
    assert result.columns == ("users.state",)
    assert result.query.join_paths_from_topic_name == "order_items"
    assert result.query.table == "order_items"


def test_a_view_scan_sends_table_and_no_topic() -> None:
    plan = Project(Scan(ViewScan("bench_ecommerce", MODEL_ID, "users")), (F.col("users.state"),))
    result = compiled(plan)

    assert result.query.table == "users"
    assert result.query.join_paths_from_topic_name is None


def test_duplicate_field_selection_is_de_duplicated() -> None:
    result = compiled(selected("users.state", "users.state"))

    assert result.fields == ("users.state",)
    assert result.columns == ("users.state",), (
        "one field asked for twice is one column back; the output shape must say so"
    )


def test_selecting_one_field_bare_and_aliased_is_not_a_tier_1_query() -> None:
    """That asks for a *copy*, and one ``fields`` entry only ever returns one column.

    De-duplicating the wire request while keeping both output names used to hand back a single
    column renamed to the alias, with ``.columns`` still claiming two — so the bare column
    vanished silently.  Tier 2 writes ``x, x AS y`` and gets it right, so this refuses.
    """
    plan = Project(SCAN, (F.col("users.state"), F.col("users.state").alias("s")))

    with pytest.raises(CannotCompile, match="selected twice under different names"):
        compile_semantic(plan)


def test_the_limit_trichotomy() -> None:
    assert compiled(selected("users.state")).query.effective_limit == DEFAULT_FETCH_LIMIT
    assert compiled(Limit(selected("users.state"), 10)).query.effective_limit == 10
    assert compiled(Limit(selected("users.state"), None)).query.effective_limit is None


def test_offset_without_a_limit_keeps_the_default_limit() -> None:
    result = compiled(Limit(selected("users.state"), UNSET, 25))

    assert result.query.offset == 25
    assert result.query.effective_limit == DEFAULT_FETCH_LIMIT


def test_sorts_carry_the_exact_field_name_and_direction() -> None:
    plan = Sort(
        selected("order_items.created_at"),
        (SortKey(FieldRef("order_items.created_at", "month"), True),),
    )
    sorts = compiled(plan).query.sorts

    assert sorts[0].column_name == "order_items.created_at[month]"
    assert sorts[0].sort_descending is True


def test_sorts_reverse_resolve_aliases() -> None:
    plan = Sort(
        Project(SCAN, (F.col("users.state").alias("state"),)),
        (SortKey(FieldRef("state")),),
    )

    assert compiled(plan).query.sorts[0].column_name == "users.state"


def test_aliases_become_the_normalize_rename_map() -> None:
    result = compiled(Project(SCAN, (F.col("users.state").alias("state"),)))

    assert result.aliases == {"users.state": "state"}
    assert result.columns == ("state",)


def test_chained_selects_compose() -> None:
    inner = Project(SCAN, (F.col("users.state").alias("state"), F.col("users.age")))
    result = compiled(Project(inner, (F.col("state"),)))

    assert result.fields == ("users.state",)
    assert result.columns == ("state",)


def test_chained_selects_keep_a_measure_a_measure() -> None:
    """Re-selecting a measure by alias (or by wire name) must not file it as a group key.

    Composition is a narrowing, not a rewrite: the outer name is a *reference*, so the inner
    expression is what survives.  Rebuilding it as a plain ``FieldRef`` made ``is_aggregate``
    false, which then made ``with_totals()`` refuse a plan it accepts written in one select().
    """
    inner = Project(SCAN, (F.measure(REVENUE).alias("rev"), F.col("users.state")))

    for outer in (F.col("rev"), F.col(REVENUE)):
        result = compiled(Project(inner, (outer, F.col("users.state"))))
        assert result.measures == (REVENUE,)
        assert result.group_keys == ("users.state",)
        assert result.is_aggregate is True
    # …and the composed envelope is still byte-identical to the single-select form.
    single = compiled(inner)
    chained = compiled(Project(inner, (F.col("rev"), F.col("users.state"))))
    assert chained.query.to_wire() == single.query.to_wire()


def test_chained_selects_keep_a_measure_filter_a_having() -> None:
    inner = Project(SCAN, (F.measure(REVENUE).alias("rev"), F.col("users.state")))
    plan = Filter(Project(inner, (F.col("rev"), F.col("users.state"))), (F.col("rev") > 100).expr)

    assert compiled(plan).measure_filters == (REVENUE,)


def test_selecting_a_column_the_previous_select_dropped_is_an_error() -> None:
    inner = Project(SCAN, (F.col("users.state"),))

    with pytest.raises(CompileError, match="not available"):
        compiled(Project(inner, (F.col("users.age"),)))


def test_a_plan_without_a_projection_says_what_to_do() -> None:
    with pytest.raises(CompileError, match="select"):
        compiled(SCAN)


# --------------------------------------------------------------------------------------
# Aggregates (docs/DESIGN.md §2: the selection IS the group-by)
# --------------------------------------------------------------------------------------


def test_an_aggregate_compiles_exactly_like_the_equivalent_projection() -> None:
    keys = (F.col("users.state"),)
    aggs = (F.measure(REVENUE), F.measure("order_items.count"))
    aggregate = compiled(Aggregate(SCAN, keys, aggs))
    projection = compiled(Project(SCAN, keys + aggs))

    assert aggregate.query.to_wire() == projection.query.to_wire()
    assert aggregate.columns == projection.columns


def test_an_aggregate_reports_its_group_keys_and_measures() -> None:
    month = F.col("order_items.created_at").grain("month")
    result = compiled(Aggregate(SCAN, (month, F.col("users.state")), (F.measure(REVENUE),)))

    assert result.group_keys == ("order_items.created_at[month]", "users.state")
    assert result.measures == (REVENUE,)
    assert result.is_aggregate is True
    assert result.fields == ("order_items.created_at[month]", "users.state", REVENUE)


def test_measures_alone_are_one_aggregate_row_with_no_group_keys() -> None:
    result = compiled(Aggregate(SCAN, (), (F.measure(REVENUE),)))

    assert result.group_keys == ()
    assert result.measures == (REVENUE,)


def test_a_projection_of_dimensions_only_is_not_an_aggregate() -> None:
    assert compiled(selected("users.state")).is_aggregate is False


def test_a_measure_filter_makes_a_dimension_query_an_aggregate() -> None:
    """The server force-adds the filtered measure to the aggregate (CONTRACT_NOTES §3.1)."""
    plan = Filter(selected("users.state"), (F.measure(REVENUE) > 50000).expr)
    result = compiled(plan)

    assert result.fields == ("users.state",), "a filtered-but-unselected measure is not a field"
    assert result.measure_filters == (REVENUE,)
    assert result.is_aggregate is True


def test_a_measure_filter_compiles_the_same_above_and_below_the_aggregate() -> None:
    keys = (F.col("users.state"),)
    aggs = (F.measure(REVENUE),)
    predicate = (F.measure(REVENUE) > 50000).expr
    above = compiled(Filter(Aggregate(SCAN, keys, aggs), predicate))
    below = compiled(Aggregate(Filter(SCAN, predicate), keys, aggs))

    assert above.query.to_wire() == below.query.to_wire()
    assert above.measure_filters == (REVENUE,)


def test_ad_hoc_aggregations_in_an_aggregate_name_the_hybrid_engine() -> None:
    plan = Aggregate(SCAN, (F.col("users.state"),), (F.count_distinct("users.id"),))

    with pytest.raises(
        CannotCompile, match=r"ad-hoc aggregations require the hybrid engine \(M3\)"
    ):
        compile_semantic(plan)


def test_an_ad_hoc_group_key_is_refused_too() -> None:
    plan = Aggregate(SCAN, (F.sum("order_items.sale_price"),), (F.measure(REVENUE),))

    with pytest.raises(CannotCompile, match="hybrid engine"):
        compile_semantic(plan)


def test_selecting_a_subset_of_an_aggregate_composes_like_chained_selects() -> None:
    aggregate = Aggregate(SCAN, (F.col("users.state"),), (F.measure(REVENUE),))
    result = compiled(Project(aggregate, (F.col("users.state"),)))

    assert result.fields == ("users.state",)


# --------------------------------------------------------------------------------------
# with_totals (CONTRACT_NOTES §2.7)
# --------------------------------------------------------------------------------------


def test_totals_ask_for_the_grand_total_over_every_measure() -> None:
    plan = Aggregate(SCAN, (F.col("users.state"),), (F.measure(REVENUE),))
    query = compile_semantic(plan, totals=True).query

    assert query.column_totals == ("::total::",)
    assert query.to_wire()["column_totals"] == {"::total::": {"type": "aggregation"}}


def test_totals_without_a_measure_say_why() -> None:
    with pytest.raises(CompileError, match="at least one governed measure"):
        compile_semantic(selected("users.state"), totals=True)


def test_totals_are_absent_unless_asked_for() -> None:
    plan = Aggregate(SCAN, (F.col("users.state"),), (F.measure(REVENUE),))

    assert compile_semantic(plan).query.to_wire()["column_totals"] == {}


# --------------------------------------------------------------------------------------
# What tier 1 refuses
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        (
            Aggregate(SCAN, (F.col("users.state"),), (F.count_distinct("users.id"),)),
            "hybrid engine",
        ),
        # A join is not a tier-1 shape and never will be — the query API takes one query — so
        # tier 1 refuses it and the splitter (M4) picks it up as a local operator.
        (Join(selected("users.state"), SCAN, ("users.id",), JoinHow.LEFT), "local operation"),
        (Union(selected("users.state"), selected("users.state")), "local operation"),
        (WithColumn(selected("users.state"), "x", F.lit(1).expr), "M3"),
        (MapPandas(selected("users.state"), lambda frame: frame), "M3"),
        (Project(SCAN, (F.count_distinct("users.id"),)), "no field name on the wire"),
        (Project(SCAN, (F.col("users.age") + 1,)), "no field name on the wire"),
        (Limit(Sort(selected("users.state"), (SortKey(FieldRef("users.state")),)), 5), None),
    ],
)
def test_tier_one_boundaries(plan: PlanNode, reason: str | None) -> None:
    if reason is None:
        assert try_semantic(plan) is not None  # the control case: this one does compile
        return
    with pytest.raises(CannotCompile, match=reason):
        compile_semantic(plan)
    assert try_semantic(plan) is None


def test_operations_applied_after_a_limit_cannot_be_one_query() -> None:
    """The wire filters and sorts *before* the limit, so this would be a different query."""
    limited = Limit(selected("users.state"), 10)

    with pytest.raises(CannotCompile, match="after limit"):
        compile_semantic(Filter(limited, (F.col("users.state") == "California").expr))
    with pytest.raises(CannotCompile, match="after limit"):
        compile_semantic(Sort(limited, (SortKey(FieldRef("users.state")),)))
    with pytest.raises(CannotCompile, match="after limit"):
        compile_semantic(Project(limited, (F.col("users.state"),)))


def test_limit_applied_twice_is_refused() -> None:
    with pytest.raises(CannotCompile, match="more than once"):
        compile_semantic(Limit(Limit(selected("users.state"), 10), 5))


# --------------------------------------------------------------------------------------
# compile_plan / the execution plan
# --------------------------------------------------------------------------------------


def test_compile_plan_reports_tier_limits_as_not_yet_supported() -> None:
    plan = Aggregate(SCAN, (F.col("users.state"),), (F.count_distinct("users.id"),))

    with pytest.raises(CompileError, match=r"not yet supported: .*M3"):
        compile_plan(plan)


def test_compile_plan_produces_one_remote_step_with_the_envelope() -> None:
    execution = compile_plan(Project(SCAN, (F.col("users.state").alias("state"),)))

    assert len(execution.steps) == 1
    assert execution.tier == 1
    assert execution.alias_map == {"users.state": "state"}
    assert execution.columns == ("state",)
    assert execution.envelope["query"]["fields"] == ["users.state"]


def test_session_options_ride_the_envelope_not_the_query() -> None:
    execution = compile_plan(
        selected("users.state"),
        options=EnvelopeOptions(
            branch_id=BRANCH_ID, cache=CachePolicy.SKIP_CACHE, timezone="America/Los_Angeles"
        ),
    )
    envelope = execution.envelope

    assert envelope["branchId"] == BRANCH_ID
    assert envelope["cache"] == "SkipCache"
    assert envelope["timezone"] == "America/Los_Angeles"
    assert "branchId" not in envelope["query"], "branchId inside query is a hard 400"


def test_the_envelope_is_validated_before_it_is_sent() -> None:
    plan = Project(Scan(TopicScan("m", "not-a-uuid", "t", "t")), (F.col("users.state"),))

    with pytest.raises(CompileError, match="UUID"):
        compile_plan(plan)


def test_predicate_to_filters_accepts_a_bare_expression() -> None:
    predicate: Expr = (F.col("users.state") == "California").expr

    assert list(predicate_to_filters(predicate)) == ["users.state"]


# --------------------------------------------------------------------------------------
# Opaque sources: raw SQL and stored queries (CONTRACT_NOTES §3.4 / §4)
# --------------------------------------------------------------------------------------

SQL = "SELECT state, SUM(sale_price) AS revenue FROM order_items GROUP BY 1"
SQL_SCAN = Scan(SqlScan(MODEL_ID, SQL, "bench_ecommerce"))
STORED = {
    "modelId": MODEL_ID,
    "table": "order_items",
    "join_paths_from_topic_name": "order_items",
    "fields": ["users.state", "order_items.total_sale_price"],
    "filters": {},
    "sorts": [],
    "limit": 1000,
    "offset": 0,
    "version": 9,
}


def test_a_sql_scan_always_carries_the_do_not_rewrite_marker() -> None:
    """Without ``rewriteSql: false`` the server parses the SQL as OmniSQL (CONTRACT_NOTES §3.5)."""
    query = compile_plan(SQL_SCAN).envelope["query"]

    assert query["userEditedSQL"] == SQL
    assert query["rewriteSql"] is False
    assert query["fields"] == [], "a SQL job selects nothing through the semantic planner"


def test_a_remote_step_refuses_to_carry_sql_without_the_marker() -> None:
    """The guarantee is structural: every remote step is built here, and this one cannot be."""
    compilation = compile_semantic(SQL_SCAN)
    envelope = {"query": {**compilation.query.to_wire(), "rewriteSql": True}}

    with pytest.raises(CompileError, match="parse the text as OmniSQL"):
        RemoteStep(envelope, compilation)


def test_a_remote_step_refuses_verbatim_sql_with_the_marker_stripped() -> None:
    """The other direction: an absent key is what selects the parsed path (CONTRACT_NOTES §3.6)."""
    compilation = compile_semantic(SQL_SCAN)
    stripped = {
        key: value for key, value in compilation.query.to_wire().items() if key != "rewriteSql"
    }

    with pytest.raises(CompileError, match="parse the text as OmniSQL"):
        RemoteStep({"query": stripped}, compilation)


# ---------------------------------------------------------------------------------------
# Compiled OmniSQL steps (docs/SQLTIER.md §2, CONTRACT_NOTES §3.6)
# ---------------------------------------------------------------------------------------

OMNISQL = "SELECT ${users.state}\nFROM ${order_items}\nLIMIT 50000"


def omnisql_compilation() -> SemanticCompilation:
    """What tier 2 hands the executor: one statement, over the topic omniframes compiled from."""
    return SemanticCompilation(
        query=Query.for_omnisql(MODEL_ID, OMNISQL),
        scan=SCAN.source,
        aliases={},
        columns=("users.state",),
        tier=2,
    )


def test_a_compiled_omnisql_step_leaves_rewrite_sql_off_the_wire() -> None:
    compilation = omnisql_compilation()
    step = RemoteStep(build_envelope(compilation, None), compilation, tier=2, label="sql")
    query = step.envelope["query"]

    assert query["userEditedSQL"] == OMNISQL
    assert "rewriteSql" not in query
    assert "staticQueryReferences" not in query
    assert "sqlSortsEnabled" not in query
    assert query["limit"] == DEFAULT_FETCH_LIMIT, "the envelope mirrors the statement's LIMIT"
    assert step.applied_limit == DEFAULT_FETCH_LIMIT
    assert compilation.is_sql is True
    assert compilation.role == "sql"


@pytest.mark.parametrize("rewrite_sql", [True, False])
def test_a_remote_step_refuses_omnisql_carrying_a_rewrite_sql_key(rewrite_sql: bool) -> None:
    """``rewriteSql: false`` would ship ``${...}`` refs to the warehouse verbatim."""
    compilation = omnisql_compilation()
    envelope = {"query": {**compilation.query.to_wire(), "rewriteSql": rewrite_sql}}

    with pytest.raises(CompileError, match="compiled as OmniSQL but carries a rewriteSql key"):
        RemoteStep(envelope, compilation, tier=2, label="sql")


def test_a_sql_job_is_tier_two_and_names_itself() -> None:
    execution = compile_plan(SQL_SCAN)

    assert execution.tier == 2
    assert execution.steps[0].label == "raw SQL job"
    assert execution.steps[0].compilation.opaque is True


def test_nothing_compiles_on_top_of_a_sql_scan() -> None:
    """A SQL job is a string the server runs; it cannot absorb a select or a limit."""
    for plan in (
        Project(SQL_SCAN, (F.col("state"),)),
        Filter(SQL_SCAN, (F.col("state") == "California").expr),
        Sort(SQL_SCAN, (SortKey(FieldRef("state")),)),
        Limit(SQL_SCAN, 10),
    ):
        with pytest.raises(CannotCompile, match="exactly as written"):
            compile_semantic(plan)


def test_a_sql_scan_cannot_be_totalled_server_side() -> None:
    with pytest.raises(CompileError, match="column_totals in the SQL"):
        compile_semantic(SQL_SCAN, totals=True)


def test_the_columns_of_a_raw_sql_scan_are_only_known_from_the_server() -> None:
    with pytest.raises(CompileError, match="planOnly"):
        _ = compile_plan(SQL_SCAN).columns


def test_a_stored_query_rides_the_wire_verbatim() -> None:
    """Omniframes did not write this blob; re-serializing it would normalize what it was given."""
    stored = {**STORED, "an_unknown_key": "kept"}
    execution = compile_plan(Scan(SavedQueryScan("bench_dashboard", "Revenue by state", stored)))

    assert execution.envelope["query"] == stored
    assert execution.envelope["query"] is not stored, "the blob is copied, never mutated"
    assert execution.columns == ("users.state", "order_items.total_sale_price")
    assert execution.steps[0].applied_limit == 1000


def test_a_stored_query_with_no_limit_key_reads_as_the_servers_own_default() -> None:
    """An absent ``limit`` is 1000 rows (CONTRACT_NOTES §3), never "unlimited".

    The blob rides the wire verbatim, so nothing omniframes writes fills the key in.  Reporting
    ``None`` here would print ``limit: unlimited (null)`` in ``explain()`` and suppress the
    TruncationWarning on a query the server truncated at 1000 rows.
    """
    blob = {key: value for key, value in STORED.items() if key != "limit"}
    execution = compile_plan(Scan(SavedQueryScan("bench_dashboard", "No limit", blob)))

    assert "limit" not in execution.envelope["query"], "still verbatim — nothing is injected"
    assert execution.steps[0].applied_limit == DEFAULT_SERVER_LIMIT
    assert execution.steps[0].query.effective_limit == DEFAULT_SERVER_LIMIT, (
        "the typed description and the applied limit must not disagree"
    )
    assert (
        compile_plan(Scan(SavedQueryScan("d", "Null", {**STORED, "limit": None})))
        .steps[0]
        .applied_limit
        is None
    ), "an explicit null is still genuinely unlimited"


def test_a_stored_query_needs_a_model_id() -> None:
    blob = {key: value for key, value in STORED.items() if key != "modelId"}

    with pytest.raises(CompileError, match="carries no modelId"):
        compile_semantic(Scan(SavedQueryScan("bench_dashboard", "Revenue by state", blob)))


def test_a_generated_query_explains_itself_as_the_prompt_that_produced_it() -> None:
    scan = SavedQueryScan(MODEL_ID, "revenue by state", STORED, origin="ask")

    assert scan.label.startswith('ask("revenue by state")')
    assert compile_semantic(Scan(scan)).role == "generated query"
