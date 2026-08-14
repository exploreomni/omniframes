"""Unit tests for the DAG executor (docs/HYBRID.md §1).

``execute`` is pure given ``run_remote``: it never sees a session, so these tests hand it canned
tables and assert on what it does with them — evaluation order, memoization, and the truncation
policy that decides which full page is worth interrupting a user about.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pyarrow as pa
import pytest
from tests.fakes import BENCH_MODEL_ID, BENCH_MODEL_NAME, BENCH_TOPIC_NAME

from omniframes import functions as F
from omniframes.compile.executor import execute
from omniframes.compile.local import AlignJoin, LocalFilter, LocalLimit, LocalProject
from omniframes.compile.semantic import ExecutionPlan, LocalStep, RemoteStep
from omniframes.compile.splitter import SplitOptions, split
from omniframes.errors import CompileError, QueryError, TruncationWarning
from omniframes.plan import nodes

SCAN = nodes.Scan(
    nodes.TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)
STATES = pa.table({"users.state": pa.array(["California", "Texas", None])})


def plan(limit: int | None = None) -> ExecutionPlan:
    """A one-query plan with a local filter on top — the UDF is what pins the frontier."""
    selected: nodes.PlanNode = nodes.Project(SCAN, (F.col("users.state"),))
    if limit is not None:
        selected = nodes.Limit(selected, limit)
    predicate = F.udf(lambda state: state is not None)("users.state")
    return split(nodes.Filter(selected, predicate.expr))


def test_execute_needs_nothing_but_a_way_to_run_one_query() -> None:
    calls: list[RemoteStep] = []

    def run_remote(step: RemoteStep) -> pa.Table:
        calls.append(step)
        return STATES

    result = execute(plan(), run_remote)

    assert len(calls) == 1
    assert result.to_pylist() == [{"users.state": "California"}, {"users.state": "Texas"}]


def test_a_step_feeding_two_operators_runs_once() -> None:
    """The DAG is a graph, not a tree; a shared step must not be fetched twice."""
    calls = 0

    def run_remote(step: RemoteStep) -> pa.Table:
        nonlocal calls
        calls += 1
        return pa.table({"k": pa.array(["a"]), "v": pa.array([1])})

    execution = split(nodes.Project(SCAN, (F.col("users.state"),)))
    shared = execution.steps[0]
    joined = LocalStep(AlignJoin(("k",)), (shared, shared))

    result = execute(ExecutionPlan((shared,), root=joined, output_columns=("k", "v")), run_remote)

    assert calls == 1
    assert result.num_rows == 1


def test_a_degenerate_plan_just_runs_its_one_query() -> None:
    execution = split(nodes.Project(SCAN, (F.col("users.state"),)))

    assert execution.root is None
    assert execute(execution, lambda step: STATES) is STATES


def test_an_expression_only_omni_can_evaluate_is_reported_as_unsupported() -> None:
    """A relative date literal is refused by the local engine; users see the usual message."""
    stamps = pa.table({"d": pa.array([None], type=pa.timestamp("us", tz="UTC"))})
    remote = split(nodes.Project(SCAN, (F.col("users.state"),))).steps[0]
    root = LocalStep(LocalFilter((F.col("d") >= "30 days ago").expr), (remote,))

    with pytest.raises(CompileError, match=r"not yet supported: relative date literals"):
        execute(ExecutionPlan((remote,), root=root, output_columns=("d",)), lambda step: stamps)


# --------------------------------------------------------------------------------------
# Truncation (§2.4)
# --------------------------------------------------------------------------------------


def test_a_full_page_warns_and_names_the_step() -> None:
    with pytest.warns(TruncationWarning, match="remote step 1 \\(semantic\\) returned exactly"):
        execute(plan(limit=3), lambda step: STATES)


def test_a_partial_page_says_nothing() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        execute(plan(limit=10), lambda step: STATES)


def test_suppressing_warnings_only_covers_the_users_own_limit() -> None:
    """What ``show()``/``first()`` want: quiet about the limit they imposed themselves."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        execute(plan(limit=3), lambda step: STATES, warn=False)


def test_an_intermediate_scan_warns_even_when_warnings_are_suppressed() -> None:
    """Nobody asked for the scan's limit, so a full one is not a limit doing its job."""
    aggregate = nodes.Aggregate(SCAN, (F.col("users.state"),), (F.count_distinct("users.id"),))
    # `disable_sql`: since M5 this aggregate rides tier 2 and there is no scan to cap, but the
    # cap still governs the local path tier 2 falls back to (docs/SQLTIER.md §4).
    execution = split(aggregate, options=SplitOptions(decomposition_row_cap=3, disable_sql=True))
    scanned = pa.table({"users.state": ["a"] * 3, "users.id": [1, 2, 3]})

    with pytest.warns(TruncationWarning, match="raw scan.*decomposition_row_cap"):
        execute(execution, lambda step: scanned, warn=False)


def test_an_unlimited_scan_never_warns() -> None:
    aggregate = nodes.Aggregate(SCAN, (F.col("users.state"),), (F.count_distinct("users.id"),))
    execution = split(aggregate, options=SplitOptions(disable_sql=True))

    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        execute(
            execution,
            lambda step: pa.table({"users.state": ["a"] * 9, "users.id": list(range(9))}),
        )


def test_a_local_operator_still_takes_exactly_its_inputs() -> None:
    remote = split(nodes.Project(SCAN, (F.col("users.state"),))).steps[0]
    root = LocalStep(LocalProject(("users.state",)), (remote,))
    execution = ExecutionPlan((remote,), root=root, output_columns=("users.state",))

    assert execute(execution, lambda step: STATES).column_names == ["users.state"]
    assert LocalLimit(1, 0).run((STATES,)).num_rows == 1


# --------------------------------------------------------------------------------------
# A statement the model does not bind (docs/SQLTIER.md §8)
# --------------------------------------------------------------------------------------


def tier_two() -> ExecutionPlan:
    """A one-statement tier-2 plan: an ad-hoc aggregate over the bench topic."""
    aggregate = nodes.Aggregate(SCAN, (F.col("users.state"),), (F.count_distinct("users.id"),))
    return split(aggregate)


def refusing(message: str) -> Callable[[RemoteStep], pa.Table]:
    def run_remote(step: RemoteStep) -> pa.Table:
        raise QueryError(message)

    return run_remote


def test_a_substitution_failure_on_a_tier_two_step_is_re_read_as_omniframes_own() -> None:
    execution = tier_two()
    assert execution.steps[0].tier == 2

    with pytest.raises(QueryError, match="tier-2 statement omniframes generated") as caught:
        execute(execution, refusing('Could not substitute Omni SQL: No such field "users.gone"'))
    error = caught.value
    assert "no field 'users.gone'" in str(error)
    assert "explain()" in str(error)
    assert error.statement is not None
    assert "${users.state}" in error.statement


def test_the_from_ref_arm_names_the_view() -> None:
    with pytest.raises(QueryError, match="no view 'gone'"):
        execute(tier_two(), refusing('Could not substitute Omni SQL: No such view "gone"'))


def test_an_unparseable_substitution_failure_still_says_whose_sql_it_is() -> None:
    """The two spellings above are the documented ones; the wrapper must not need them."""
    with pytest.raises(QueryError, match="rejected a reference in it") as caught:
        execute(tier_two(), refusing("Could not substitute Omni SQL: something new"))
    assert caught.value.statement is not None


def test_every_other_job_error_passes_through_untouched() -> None:
    with pytest.raises(QueryError, match=r"^division by zero$") as caught:
        execute(tier_two(), refusing("division by zero"))
    assert caught.value.statement is None


def test_a_tier_one_step_is_never_re_attributed() -> None:
    """Only a step whose SQL omniframes wrote can be omniframes' emission bug."""
    execution = split(nodes.Project(SCAN, (F.col("users.state"),)))

    with pytest.raises(QueryError, match=r"^Could not substitute Omni SQL: whatever$"):
        execute(execution, refusing("Could not substitute Omni SQL: whatever"))
