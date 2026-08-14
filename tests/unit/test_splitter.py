"""Unit tests for the splitter (docs/HYBRID.md §2).

These build plans directly and compile them — no session, no network — because what is under
test is *where the cut falls*, not what comes back.  The invariant every case circles is
**maximality**: whatever tier 1 can express stays one governed query, and only what it cannot
becomes a local operator.

Since M5 there are three tiers to cut between, so the cases come in pairs: :func:`split` shows
where the frontier falls *today* (tier 1 → tier 2 → local), and :func:`local_split` — tier 2
switched off with ``disable_sql`` — pins the local engine's own behavior, which is still what
runs whenever tier 2 cannot express a plan.  Both halves matter: the differential lane's headline
check is that the two agree on every answer (docs/SQLTIER.md §7).
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.fakes import BENCH_MODEL_ID, BENCH_MODEL_NAME, BENCH_TOPIC_NAME

from omniframes import functions as F
from omniframes.column import Column
from omniframes.compile.local import (
    AlignJoin,
    LocalAggregate,
    LocalFilter,
    LocalJoin,
    LocalLimit,
    LocalMapPandas,
    LocalOp,
    LocalProject,
    LocalSort,
    LocalUnion,
    LocalWithColumn,
)
from omniframes.compile.querymodel import DEFAULT_FETCH_LIMIT
from omniframes.compile.semantic import ExecutionPlan, LocalStep, RemoteStep, Step, compile_plan
from omniframes.compile.splitter import SplitOptions, split
from omniframes.errors import CompileError
from omniframes.plan import nodes
from omniframes.types import OmniDataType, OmniField, OmniSchema

REVENUE = "order_items.total_sale_price"
SCAN = nodes.Scan(
    nodes.TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)


def columns(*names: str | Column) -> tuple[Column, ...]:
    return tuple(F.col(name) if isinstance(name, str) else name for name in names)


def selected(*names: str | Column) -> nodes.Project:
    return nodes.Project(SCAN, columns(*names))


def local_split(plan: nodes.PlanNode, **options: Any) -> ExecutionPlan:
    """Split with tier 2 switched off — the M3 frontier, which is still the fallback path."""
    return split(plan, options=SplitOptions(disable_sql=True, **options))


def ops(execution: ExecutionPlan) -> list[LocalOp]:
    """Every local operator, deepest first — the order ``explain()`` prints them in."""
    found: list[LocalOp] = []
    seen: set[int] = set()

    def visit(step: Step) -> None:
        if isinstance(step, RemoteStep) or id(step) in seen:
            return
        seen.add(id(step))
        for source in step.inputs:
            visit(source)
        found.append(step.op)

    if execution.root is not None:
        visit(execution.root)
    return found


def kinds(execution: ExecutionPlan) -> list[type[LocalOp]]:
    return [type(op) for op in ops(execution)]


def fields(execution: ExecutionPlan, index: int = 0) -> list[str]:
    payload: list[str] = execution.steps[index].envelope["query"]["fields"]
    return payload


def limit(execution: ExecutionPlan, index: int = 0) -> Any:
    return execution.steps[index].envelope["query"]["limit"]


# --------------------------------------------------------------------------------------
# Maximality: a plan tier 1 can express does not change at all
# --------------------------------------------------------------------------------------


def test_a_fully_compilable_plan_is_one_remote_step_and_no_local_work() -> None:
    plan = nodes.Limit(
        nodes.Filter(selected("users.state"), (F.col("users.state") == "California").expr), 10
    )
    execution = split(plan)

    assert execution.root is None, "the degenerate case: nothing to run locally"
    assert len(execution.steps) == 1
    assert execution.tier == 1
    assert execution.envelope == compile_plan(plan).envelope, "byte-identical to M1/M2"


def test_the_frontier_is_maximal_the_local_part_is_only_what_tier_one_refused() -> None:
    """Filter + sort + limit all compile; only the derived column does not."""
    plan = nodes.WithColumn(
        nodes.Limit(
            nodes.Sort(
                nodes.Filter(
                    selected("users.state", "users.age"),
                    (F.col("users.state") == "California").expr,
                ),
                (F.col("users.age").desc().to_sort_key(),),
            ),
            10,
        ),
        "doubled",
        (F.col("users.age") * 2).expr,
    )
    execution = split(plan)
    query = execution.steps[0].envelope["query"]

    assert len(execution.steps) == 1
    assert kinds(execution) == [LocalWithColumn, LocalProject]
    assert query["filters"], "the compilable filter rode remote"
    assert query["sorts"], "and so did the sort"
    assert query["limit"] == 10, "and so did the limit"


def test_a_split_plan_reports_the_lowest_tier_and_refuses_to_name_one_query() -> None:
    """A UDF is permanently local, so this plan is tier 3 no matter what tier 2 learns."""
    execution = split(
        nodes.WithColumn(selected("users.state"), "loud", F.udf(shout)("users.state").expr)
    )

    assert execution.tier == 3
    with pytest.raises(CompileError, match="DAG"):
        _ = execution.remote


# --------------------------------------------------------------------------------------
# Per-node rules and column widening (§2.2)
# --------------------------------------------------------------------------------------


def test_an_uncompilable_filter_becomes_a_local_filter_over_a_widened_scan() -> None:
    plan = nodes.Filter(selected("users.state"), (F.col("users.age") * 2 > 100).expr)
    execution = local_split(plan)

    assert fields(execution) == ["users.state", "users.age"], "widened with the field it needs"
    assert kinds(execution) == [LocalFilter, LocalProject]
    assert execution.columns == ("users.state",), "and the widened column is dropped again"


def test_a_cross_field_or_runs_locally_rather_than_being_mistranslated() -> None:
    predicate = ((F.col("users.state") == "California") | (F.col("users.age") > 60)).expr
    execution = local_split(nodes.Filter(selected("order_items.status"), predicate))

    assert fields(execution) == ["order_items.status", "users.age", "users.state"]
    assert kinds(execution) == [LocalFilter, LocalProject]
    assert execution.steps[0].envelope["query"]["filters"] == {}, "nothing was pushed as a filter"


def test_widening_reaches_through_a_filter_and_a_sort_to_the_projection() -> None:
    plan = nodes.Filter(
        nodes.Sort(
            nodes.Filter(selected("users.state"), (F.col("users.state") == "Texas").expr),
            (F.col("users.state").to_sort_key(),),
        ),
        (F.col("users.age") + 1 > 20).expr,
    )
    execution = local_split(plan)
    query = execution.steps[0].envelope["query"]

    assert query["fields"] == ["users.state", "users.age"]
    assert list(query["filters"]) == ["users.state"], "the compilable filter still rides remote"
    assert query["sorts"], "and so does the sort"


def test_a_bare_scan_gets_the_projection_widening_needs() -> None:
    """No ``select()`` yet: the projection is created directly above the scan."""
    execution = local_split(nodes.Project(SCAN, columns(F.col("users.age") * 2)))

    assert fields(execution) == ["users.age"]
    assert execution.columns == ("(users.age * 2)",)


def test_a_computed_column_in_select_is_projected_locally_in_the_written_order() -> None:
    plan = nodes.Project(
        SCAN,
        columns(
            (F.col("order_items.sale_price") * 2).alias("double"),
            "users.state",
        ),
    )
    execution = local_split(plan)

    assert fields(execution) == ["order_items.sale_price", "users.state"]
    assert kinds(execution) == [LocalWithColumn, LocalProject]
    assert execution.columns == ("double", "users.state")


def test_a_field_selected_bare_and_aliased_is_copied_locally() -> None:
    """One column arrives however many times a field is selected (docs/SQLTIER.md §3.2).

    Neither tier can produce the second copy — tier 1 sends `fields` once and the OmniSQL parser
    de-duplicates a repeated bare ref — so the projection makes it, rather than renaming both
    copies to the alias and losing the bare one.
    """
    execution = split(selected("users.state", F.col("users.state").alias("s")))

    assert fields(execution) == ["users.state"], "the wire still fetches the field once"
    assert execution.columns == ("users.state", "s")
    assert kinds(execution) == [LocalWithColumn, LocalProject]


def test_the_alias_may_be_written_before_the_bare_field() -> None:
    execution = split(selected(F.col("users.state").alias("s"), "users.state"))

    assert execution.columns == ("s", "users.state")
    assert kinds(execution) == [LocalWithColumn, LocalProject]


def test_a_field_that_cannot_be_widened_in_is_an_error_naming_it() -> None:
    aggregate = nodes.Aggregate(SCAN, columns("users.state"), columns(F.measure(REVENUE)))
    plan = nodes.Filter(aggregate, (F.col("users.age") * 2 > 1).expr)

    with pytest.raises(CompileError, match=r"users\.age is not available after group_by"):
        split(plan)


def test_tier_two_never_answers_a_question_the_widening_rules_would_refuse() -> None:
    """Tier 2 is tried only over a subtree that could carry the columns above it.

    Without that gate a SQL job could quietly drop a column an operator above needs — or, worse,
    reinterpret a filter on a consumed column as a pre-aggregation WHERE — instead of failing
    with the message §2.2 promises.
    """
    aggregate = nodes.Aggregate(SCAN, columns("users.state"), columns(F.count("users.id")))
    plan = nodes.Filter(aggregate, (F.col("users.age") * 2 > 1).expr)

    with pytest.raises(CompileError, match=r"users\.age is not available after group_by"):
        split(plan)


def test_a_governed_measure_is_never_emulated_locally() -> None:
    plan = nodes.Filter(selected("users.state"), (F.measure(REVENUE) + 1 > 2).expr)

    with pytest.raises(CompileError, match="never emulates one"):
        split(plan)


# --------------------------------------------------------------------------------------
# Limit pins the frontier (§2.3)
# --------------------------------------------------------------------------------------


def test_a_limit_below_local_work_rides_remote_and_the_local_ops_see_its_result() -> None:
    """M1 refused this shape; M3 executes it, with exactly M1's reading of what it means."""
    plan = nodes.Filter(
        nodes.Limit(selected("users.state", "order_items.status"), 5),
        (F.col("users.state") == "California").expr,
    )
    execution = split(plan)

    assert limit(execution) == 5, "the limit is part of the query, not applied afterwards"
    assert kinds(execution) == [LocalFilter, LocalProject]


def test_a_limit_above_local_work_is_a_local_limit() -> None:
    plan = nodes.Limit(
        nodes.WithColumn(selected("users.age"), "d", (F.col("users.age") * 2).expr), 5
    )
    execution = local_split(plan)

    assert limit(execution) == DEFAULT_FETCH_LIMIT, "the query still sends an explicit limit"
    assert kinds(execution) == [LocalWithColumn, LocalLimit, LocalProject]
    assert ops(execution)[1] == LocalLimit(5, 0)


def test_an_offset_above_local_work_does_not_invent_a_second_cap() -> None:
    plan = nodes.Limit(
        nodes.WithColumn(selected("users.age"), "d", (F.col("users.age") * 2).expr), offset=3
    )

    assert LocalLimit(None, 3) in ops(local_split(plan))


def test_widening_lands_below_the_limit_not_above_it() -> None:
    """The extra column has to come from the same limited query, or it would be a different one."""
    plan = nodes.Filter(nodes.Limit(selected("users.state"), 5), (F.col("users.age") * 2 > 1).expr)
    execution = split(plan)

    assert fields(execution) == ["users.state", "users.age"]
    assert limit(execution) == 5


# --------------------------------------------------------------------------------------
# Mixed aggregation: one tier-2 statement (SQLTIER §4), decomposing only when tier 2 refuses
# --------------------------------------------------------------------------------------


def mixed() -> nodes.Aggregate:
    return nodes.Aggregate(
        SCAN,
        columns("users.state"),
        columns(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers")),
    )


def refused_mixed() -> nodes.Aggregate:
    """The same aggregate over a filter tier 2 cannot write.

    Omni's relative-date grammar only means something server-side, so a `last month` predicate
    has no OmniSQL rendering and the whole statement refuses (docs/SQLTIER.md §1) — while tier 1
    carries it happily, which is exactly the shape the decomposition still exists for.
    """
    return nodes.Aggregate(
        nodes.Filter(SCAN, (F.col("order_items.created_at") == "last month").expr),
        columns("users.state"),
        columns(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("buyers")),
    )


def test_a_mixed_aggregate_is_one_tier_two_statement() -> None:
    """The measure and the ad-hoc aggregate ride the same SELECT; nothing is decomposed."""
    execution = split(mixed())
    sql = execution.steps[0].envelope["query"]["userEditedSQL"]

    assert execution.root is None, "no align-join, no local aggregate, no second query"
    assert [(step.label, step.tier) for step in execution.steps] == [("sql", 2)]
    assert "${order_items.total_sale_price}" in sql, "the measure expands server-side"
    assert "COUNT(DISTINCT ${users.id})" in sql
    assert execution.columns == ("users.state", "revenue", "buyers")


def test_the_row_cap_has_nothing_to_cap_when_the_mixed_aggregate_collapses() -> None:
    """`decomposition_row_cap` caps the decomposition's raw scan, and there is no raw scan."""
    execution = split(mixed(), options=SplitOptions(decomposition_row_cap=1000))

    assert len(execution.steps) == 1
    assert limit(execution) == DEFAULT_FETCH_LIMIT, "the statement's own limit, not the cap"


def test_a_mixed_aggregate_tier_two_refuses_still_decomposes() -> None:
    """The M3 decomposition is the fallback, not the dead code path (docs/SQLTIER.md §4)."""
    execution = split(refused_mixed())

    assert [step.label for step in execution.steps] == ["semantic", "raw scan"]
    assert kinds(execution) == [LocalAggregate, AlignJoin, LocalProject]
    assert execution.columns == ("users.state", "revenue", "buyers")
    assert list(execution.steps[0].envelope["query"]["filters"]) == ["order_items.created_at"], (
        "the predicate tier 2 could not write is exactly the one tier 1 pushes"
    )


def test_a_measure_only_aggregate_is_still_tier_one() -> None:
    """Tier order is untouched: tier 2's reach grew, tier 1 still wins what it can express."""
    execution = split(nodes.Aggregate(SCAN, columns("users.state"), columns(F.measure(REVENUE))))

    assert execution.root is None
    assert [(step.label, step.tier) for step in execution.steps] == [("semantic", 1)]


def test_a_measure_only_aggregate_over_local_work_is_refused_by_name() -> None:
    """Governed measures never run locally, so an input no query can express is an error."""
    plan = nodes.Aggregate(
        nodes.MapPandas(selected("users.state"), lambda frame: frame),
        columns("users.state"),
        columns(F.measure(REVENUE)),
    )

    with pytest.raises(CompileError, match="governed measures always execute remotely"):
        split(plan)


def test_a_mixed_aggregate_decomposes_into_measures_then_raw_scan() -> None:
    execution = local_split(mixed())

    assert [step.label for step in execution.steps] == ["semantic", "raw scan"]
    assert fields(execution, 0) == ["users.state", REVENUE]
    assert fields(execution, 1) == ["users.state", "users.id"]
    assert kinds(execution) == [LocalAggregate, AlignJoin, LocalProject]
    assert execution.columns == ("users.state", "revenue", "buyers")


def test_the_raw_scan_is_unlimited_on_the_wire() -> None:
    """A silently 50k-capped input to a local aggregation is a wrong answer, not a short page."""
    execution = local_split(mixed())

    assert limit(execution, 1) is None
    assert limit(execution, 0) == DEFAULT_FETCH_LIMIT, "the governed half is limited as usual"
    assert execution.steps[1].note == "(unlimited)"


def test_the_row_cap_turns_the_raw_scan_into_a_limited_query() -> None:
    execution = local_split(mixed(), decomposition_row_cap=1000)

    assert limit(execution, 1) == 1000
    assert execution.steps[1].note == "(capped at 1000)"
    assert execution.steps[1].user_limit is False, "nobody asked for it, so it always warns"


def test_the_align_join_keys_on_the_user_facing_group_key_names() -> None:
    aggregate = nodes.Aggregate(
        SCAN,
        columns(F.col("users.state").alias("state")),
        columns(F.measure(REVENUE), F.count_distinct("users.id")),
    )
    execution = local_split(aggregate)
    join = next(op for op in ops(execution) if isinstance(op, AlignJoin))

    assert join.on == ("state",), "both halves carry the same alias, so the keys line up by name"
    assert execution.columns == ("state", REVENUE, "count_distinct(users.id)")


def test_an_aggregate_of_only_ad_hoc_aggregations_needs_no_join() -> None:
    aggregate = nodes.Aggregate(
        SCAN, columns("users.state"), columns(F.count_distinct("users.id"), F.sum("users.age"))
    )
    execution = local_split(aggregate)

    assert len(execution.steps) == 1
    assert execution.steps[0].label == "raw scan"
    assert kinds(execution) == [LocalAggregate, LocalProject]
    assert fields(execution) == ["users.state", "users.id", "users.age"]


def test_the_raw_scan_fetches_each_operand_once() -> None:
    aggregate = nodes.Aggregate(
        SCAN, columns("users.state"), columns(F.sum("users.age"), F.max("users.age"))
    )

    assert fields(local_split(aggregate)) == ["users.state", "users.age"]


def test_an_operand_that_is_also_a_group_key_is_read_under_its_alias() -> None:
    aggregate = nodes.Aggregate(
        SCAN, columns(F.col("users.state").alias("s")), columns(F.count("users.state"))
    )
    execution = local_split(aggregate)
    local = next(op for op in ops(execution) if isinstance(op, LocalAggregate))

    assert fields(execution) == ["users.state"]
    assert local.aggs[0].operand == "s", "the scan renamed it, so the aggregate follows"


def test_a_global_ad_hoc_aggregate_groups_by_nothing() -> None:
    aggregate = nodes.Aggregate(SCAN, (), columns(F.sum("users.age")))
    execution = local_split(aggregate)
    local = next(op for op in ops(execution) if isinstance(op, LocalAggregate))

    assert local.keys == ()
    assert execution.columns == ("sum(users.age)",)


def stacked() -> nodes.PlanNode:
    """A HAVING, a sort over the measure and a limit, all written above the mixed aggregate."""
    return nodes.Limit(
        nodes.Sort(
            nodes.Filter(mixed(), (F.col("buyers") > 10).expr),
            (F.col("revenue").desc().to_sort_key(),),
        ),
        5,
    )


def test_everything_above_a_collapsed_mixed_aggregate_rides_the_same_statement() -> None:
    execution = split(stacked())
    query = execution.steps[0].envelope["query"]

    assert execution.root is None
    assert "HAVING\n  COUNT(DISTINCT ${users.id}) > 10" in query["userEditedSQL"]
    assert "ORDER BY\n  2 DESC" in query["userEditedSQL"], "positional, over the measure item"
    assert query["limit"] == 5


def test_everything_above_a_decomposed_aggregate_runs_locally() -> None:
    execution = local_split(stacked())

    assert kinds(execution) == [
        LocalAggregate,
        AlignJoin,
        LocalFilter,
        LocalSort,
        LocalLimit,
        LocalProject,
    ], "with tier 2 off nothing above the align-join can ride the wire"
    assert all(step.envelope["query"]["sorts"] == [] for step in execution.steps)


def test_a_filter_on_an_ad_hoc_aggregate_column_is_a_having() -> None:
    """Spelled as a column reference, which is what it is once the aggregate has run."""
    plan = nodes.Filter(mixed(), (F.count_distinct("users.id") > 10).expr)

    assert kinds(split(plan)) == [], "tier 2 writes it as a HAVING on the one statement"
    assert kinds(local_split(plan)) == [LocalAggregate, AlignJoin, LocalFilter, LocalProject]


def test_a_measure_filter_above_a_decomposition_reads_the_joined_column() -> None:
    """The measure column exists once step 1 ran, alias and all — so the filter is local."""
    plan = nodes.Filter(mixed(), (F.measure(REVENUE) > 50000).expr)
    execution = local_split(plan)
    local = next(op for op in ops(execution) if isinstance(op, LocalFilter))

    assert kinds(execution) == [LocalAggregate, AlignJoin, LocalFilter, LocalProject]
    assert all(step.envelope["query"]["filters"] == {} for step in execution.steps)
    assert "revenue > 50000" in local.detail(), "rewritten to the alias the result carries"


def test_a_measure_filter_above_a_collapsed_aggregate_is_a_having_over_the_measure_ref() -> None:
    """``${measure}`` expands inline in a HAVING, so tier 2 keeps the whole thing (§3.6, L4)."""
    execution = split(nodes.Filter(mixed(), (F.measure(REVENUE) > 50000).expr))

    assert execution.root is None
    assert (
        "HAVING\n  ${order_items.total_sale_price} > 50000"
        in (execution.steps[0].envelope["query"]["userEditedSQL"])
    )


def test_a_filter_on_an_ad_hoc_aggregate_that_nothing_computed_still_routes_to_tier_two() -> None:
    plan = nodes.Filter(selected("users.state"), (F.count_distinct("users.id") > 10).expr)

    with pytest.raises(CompileError, match=r"not yet supported: .*tier 2/3"):
        split(plan)


def test_select_with_an_ad_hoc_aggregation_compiles_like_the_group_by() -> None:
    """Selection IS the group-by, so the two spellings must produce the same plan."""
    project = nodes.Project(
        SCAN,
        columns("users.state", F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id")),
    )
    aggregate = nodes.Aggregate(
        SCAN,
        columns("users.state"),
        columns(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id")),
    )
    from_project = split(project)
    from_aggregate = split(aggregate)

    assert [s.envelope for s in from_project.steps] == [s.envelope for s in from_aggregate.steps]
    assert from_project.columns == from_aggregate.columns
    assert kinds(from_project) == kinds(from_aggregate)


def test_totals_over_a_decomposed_aggregate_are_refused() -> None:
    """Totals are a tier-1 server feature; emulating them would mean emulating the measures."""
    with pytest.raises(CompileError, match="with_totals\\(\\) needs a query Omni can total"):
        split(mixed(), options=SplitOptions(totals=True))


def test_totals_still_work_when_the_whole_query_compiles() -> None:
    aggregate = nodes.Aggregate(SCAN, columns("users.state"), columns(F.measure(REVENUE)))
    execution = split(aggregate, options=SplitOptions(totals=True))

    assert execution.root is None
    assert execution.envelope["query"]["column_totals"] == {"::total::": {"type": "aggregation"}}


# --------------------------------------------------------------------------------------
# The UDF / map_pandas boundary (§5)
# --------------------------------------------------------------------------------------


def shout(value: str) -> str:
    return value.upper()


def test_a_udf_in_a_filter_pins_the_frontier() -> None:
    plan = nodes.Filter(selected("users.state"), (F.udf(shout)("users.state") == "TEXAS").expr)
    execution = split(plan)

    assert execution.steps[0].envelope["query"]["filters"] == {}
    assert kinds(execution) == [LocalFilter, LocalProject]


def test_a_udf_in_a_projection_pins_the_frontier_and_widens() -> None:
    plan = nodes.Project(SCAN, columns(F.udf(shout)("users.state").alias("loud")))
    execution = split(plan)

    assert fields(execution) == ["users.state"]
    assert execution.columns == ("loud",)


def test_map_pandas_pins_the_frontier_structurally() -> None:
    plan = nodes.MapPandas(selected("users.state"), lambda frame: frame)
    execution = split(plan)

    assert kinds(execution) == [LocalMapPandas]
    assert execution.output_columns is None


def test_without_a_schema_hint_the_columns_are_unknowable() -> None:
    execution = split(nodes.MapPandas(selected("users.state"), lambda frame: frame))

    with pytest.raises(CompileError, match="map_pandas"):
        _ = execution.columns


def test_a_schema_hint_makes_the_columns_knowable_again() -> None:
    hint = OmniSchema((OmniField(name="region", data_type=OmniDataType.STRING),))
    execution = split(nodes.MapPandas(selected("users.state"), lambda frame: frame, hint))

    assert execution.columns == ("region",)


def test_an_operation_above_an_unhinted_map_pandas_names_the_column_it_cannot_get() -> None:
    plan = nodes.Sort(
        nodes.MapPandas(selected("users.state"), lambda frame: frame),
        (F.col("region").to_sort_key(),),
    )

    with pytest.raises(CompileError, match="region cannot be read out of map_pandas"):
        split(plan)


def test_an_operation_above_a_hinted_map_pandas_compiles_against_the_hint() -> None:
    hint = OmniSchema((OmniField(name="region", data_type=OmniDataType.STRING),))
    plan = nodes.Sort(
        nodes.MapPandas(selected("users.state"), lambda frame: frame, hint),
        (F.col("region").to_sort_key(),),
    )
    execution = split(plan)

    assert kinds(execution) == [LocalMapPandas, LocalSort, LocalProject]


def test_a_column_outside_the_hint_is_refused_by_name() -> None:
    hint = OmniSchema((OmniField(name="region", data_type=OmniDataType.STRING),))
    plan = nodes.Sort(
        nodes.MapPandas(selected("users.state"), lambda frame: frame, hint),
        (F.col("nope").to_sort_key(),),
    )

    with pytest.raises(CompileError, match="nope is not in the schema_hint"):
        split(plan)


def test_limit_needs_no_schema_so_it_is_allowed_above_an_unhinted_map_pandas() -> None:
    plan = nodes.Limit(nodes.MapPandas(selected("users.state"), lambda frame: frame), 3)
    execution = split(plan)

    assert kinds(execution) == [LocalMapPandas, LocalLimit]
    assert execution.output_columns is None


# --------------------------------------------------------------------------------------
# What the splitter still refuses
# --------------------------------------------------------------------------------------


def test_acting_on_a_frame_with_no_projection_still_says_to_select_something() -> None:
    with pytest.raises(CompileError, match="select\\(\\) at least one column"):
        split(SCAN)


def test_session_options_still_ride_every_envelope_of_a_split_plan() -> None:
    from omniframes.compile.querymodel import CachePolicy
    from omniframes.compile.semantic import EnvelopeOptions

    options = EnvelopeOptions(cache=CachePolicy.SKIP_CACHE)
    collapsed = split(mixed(), options=SplitOptions(envelope=options))
    decomposed = local_split(mixed(), envelope=options)

    assert [step.envelope["cache"] for step in collapsed.steps] == ["SkipCache"]
    assert [step.envelope["cache"] for step in decomposed.steps] == ["SkipCache", "SkipCache"]


def test_a_local_step_needs_an_input() -> None:
    with pytest.raises(CompileError, match="at least one input"):
        LocalStep(LocalProject(("a",)), ())


# --------------------------------------------------------------------------------------
# Joins and unions (M4): each side splits on its own, the combination is local
# --------------------------------------------------------------------------------------


def revenue_by_state() -> nodes.PlanNode:
    return nodes.Aggregate(SCAN, (F.col("users.state"),), (F.measure(REVENUE).alias("revenue"),))


def buyers_by_state() -> nodes.PlanNode:
    return nodes.Aggregate(SCAN, (F.col("users.state"),), (F.measure("users.count").alias("n"),))


def test_a_join_is_two_independent_remote_steps_and_one_local_operator() -> None:
    execution = split(nodes.Join(revenue_by_state(), buyers_by_state(), ("users.state",)))

    assert len(execution.steps) == 2, "each side compiles on its own; neither constrains the other"
    assert [fields(execution, i) for i in (0, 1)] == [
        ["users.state", REVENUE],
        ["users.state", "users.count"],
    ]
    assert kinds(execution) == [LocalJoin, LocalProject]
    assert execution.columns == ("users.state", "revenue", "n")
    assert execution.tier == 3


def test_the_join_keys_are_matched_against_each_side_s_output_columns() -> None:
    """An aliased field is named by its alias; an un-aliased one by its wire name."""
    left = nodes.Project(SCAN, columns(F.col("users.id").alias("uid"), F.col("users.state")))
    right = nodes.Aggregate(
        SCAN, (F.col("order_items.user_id").alias("uid"),), (F.measure(REVENUE),)
    )
    execution = split(nodes.Join(left, right, ("uid",)))

    assert execution.columns == ("uid", "users.state", REVENUE)
    assert fields(execution, 0) == ["users.id", "users.state"], "the alias never reaches the wire"


def test_a_join_key_that_is_not_an_output_column_names_the_side_and_its_columns() -> None:
    plan = nodes.Join(revenue_by_state(), selected("users.age"), ("users.state",))

    with pytest.raises(CompileError, match="not a column of the right frame"):
        split(plan)


def test_overlapping_non_key_columns_are_a_compile_error_not_a_suffix() -> None:
    """PySpark would hand back two columns of the same name; omniframes refuses to guess."""
    plan = nodes.Join(revenue_by_state(), revenue_by_state(), ("users.state",))

    with pytest.raises(CompileError, match="will not guess which one you meant"):
        split(plan)


@pytest.mark.parametrize(
    "how",
    [nodes.JoinHow.INNER, nodes.JoinHow.LEFT, nodes.JoinHow.RIGHT, nodes.JoinHow.OUTER],
)
def test_every_join_kind_reaches_the_operator(how: nodes.JoinHow) -> None:
    execution = split(nodes.Join(revenue_by_state(), buyers_by_state(), ("users.state",), how))
    op = ops(execution)[0]

    assert isinstance(op, LocalJoin)
    assert op.how is how


def test_a_cross_join_is_refused_rather_than_silently_sized() -> None:
    plan = nodes.Join(revenue_by_state(), buyers_by_state(), (), nodes.JoinHow.CROSS)

    with pytest.raises(CompileError, match="cross joins are not supported"):
        split(plan)


def test_a_join_on_an_expression_says_what_it_takes_instead() -> None:
    plan = nodes.Join(revenue_by_state(), buyers_by_state(), (F.col("users.state") == "CA").expr)

    with pytest.raises(CompileError, match="column name or a list of column names"):
        split(plan)


def test_a_filter_above_a_join_runs_on_the_joined_result() -> None:
    plan = nodes.Filter(
        nodes.Join(revenue_by_state(), buyers_by_state(), ("users.state",)),
        (F.col("users.state") == "California").expr,
    )
    execution = split(plan)

    assert kinds(execution) == [LocalJoin, LocalFilter, LocalProject]


def test_a_column_a_join_does_not_produce_cannot_be_widened_in() -> None:
    """Which side would it come from? Naming the column beats inventing an answer."""
    plan = nodes.Filter(
        nodes.Join(revenue_by_state(), buyers_by_state(), ("users.state",)),
        (F.col("users.age") > 30).expr,
    )

    with pytest.raises(CompileError, match=r"users\.age is not available after join"):
        split(plan)


def test_a_union_is_two_steps_and_keeps_the_shared_column_names() -> None:
    execution = split(nodes.Union(revenue_by_state(), revenue_by_state()))

    assert len(execution.steps) == 2
    assert kinds(execution) == [LocalUnion, LocalProject]
    assert execution.columns == ("users.state", "revenue")


def test_a_union_whose_names_do_not_line_up_is_refused_at_compile_time() -> None:
    plan = nodes.Union(selected("users.state", "users.age"), selected("users.state", "users.id"))

    with pytest.raises(CompileError, match="same columns in the same order"):
        split(plan)


def test_a_union_of_frames_with_different_widths_is_refused() -> None:
    plan = nodes.Union(selected("users.state"), selected("users.state", "users.age"))

    with pytest.raises(CompileError, match="same columns in the same order"):
        split(plan)


def test_with_totals_over_a_join_is_refused() -> None:
    plan = nodes.Join(revenue_by_state(), buyers_by_state(), ("users.state",))

    with pytest.raises(CompileError, match="with_totals"):
        split(plan, options=SplitOptions(totals=True))


# --------------------------------------------------------------------------------------
# Opaque scans: raw SQL and stored queries (M4)
# --------------------------------------------------------------------------------------

SQL = "SELECT state, SUM(revenue) AS revenue FROM t GROUP BY 1"
SQL_SCAN = nodes.Scan(nodes.SqlScan(BENCH_MODEL_ID, SQL, BENCH_MODEL_NAME))
STORED_SCAN = nodes.Scan(
    nodes.SavedQueryScan(
        "bench_dashboard",
        "Revenue by state",
        {
            "modelId": BENCH_MODEL_ID,
            "fields": ["users.state", REVENUE],
            "table": "order_items",
            "join_paths_from_topic_name": BENCH_TOPIC_NAME,
            "limit": 1000,
            "version": 9,
        },
    )
)


def test_a_bare_sql_scan_is_one_remote_step_with_no_local_work() -> None:
    execution = split(SQL_SCAN)

    assert execution.root is None
    assert execution.envelope["query"]["userEditedSQL"] == SQL
    assert execution.envelope["query"]["rewriteSql"] is False


def test_everything_written_on_top_of_a_sql_scan_runs_locally() -> None:
    """A SQL job is opaque: nothing can be pushed into it, so nothing is."""
    plan = nodes.Limit(
        nodes.Sort(
            nodes.Filter(
                nodes.Project(SQL_SCAN, columns("state", "revenue")),
                (F.col("state") == "California").expr,
            ),
            (F.col("revenue").desc().to_sort_key(),),
        ),
        5,
    )
    execution = split(plan)

    assert len(execution.steps) == 1
    assert kinds(execution) == [LocalProject, LocalFilter, LocalSort, LocalLimit, LocalProject]
    assert execution.columns == ("state", "revenue")
    assert execution.steps[0].envelope["query"]["userEditedSQL"] == SQL, "the SQL is untouched"


def test_a_sql_scan_is_never_widened_with_a_projection() -> None:
    """Adding fields to a raw-SQL job would change nothing on the wire and lie in explain()."""
    execution = split(nodes.Filter(SQL_SCAN, (F.col("state") == "California").expr))

    assert fields(execution) == []


def test_a_stored_query_scan_sends_the_blob_and_finishes_locally() -> None:
    execution = split(nodes.Project(STORED_SCAN, columns("users.state")))

    assert execution.steps[0].envelope["query"]["limit"] == 1000, "verbatim, default limit and all"
    assert kinds(execution) == [LocalProject]
    assert execution.columns == ("users.state",)


def test_a_frame_can_join_a_governed_query_to_a_raw_sql_job() -> None:
    """Neither side knows about the other; the join is the only thing that runs here."""
    sql = nodes.Scan(
        nodes.SqlScan(BENCH_MODEL_ID, "SELECT 'CA' AS \"users.state\", 1 AS rank", BENCH_MODEL_NAME)
    )
    execution = split(nodes.Join(revenue_by_state(), sql, ("users.state",)))

    assert [step.tier for step in execution.steps] == [1, 2]
    assert kinds(execution) == [LocalJoin], "no final projection: the SQL's columns are its own"
    assert execution.output_columns is None
