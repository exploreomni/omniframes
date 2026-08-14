"""Unit tests for the DataFrame API (docs/INTERNALS.md §5, DESIGN §3)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pyarrow as pa
import pytest
from tests.fakes import (
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

from omniframes import functions as F
from omniframes.compile.querymodel import DEFAULT_FETCH_LIMIT, UNSET
from omniframes.dataframe import DataFrame
from omniframes.errors import CompileError, TruncationWarning
from omniframes.plan import (
    Filter,
    Join,
    JoinHow,
    Limit,
    Project,
    Scan,
    Sort,
    TopicScan,
    Union,
)
from omniframes.session import OmniSession
from omniframes.transport import HttpTransport
from omniframes.types import OmniDataType

BASE_URL = "https://bench.example.omni.co"
SCAN = Scan(
    TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


@pytest.fixture
def session(handler: FakeOmniAPI) -> Iterator[OmniSession]:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    with client:
        transport = HttpTransport(
            base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
        )
        yield OmniSession.builder.transport(transport).get_or_create()


@pytest.fixture
def df(session: OmniSession) -> DataFrame:
    """A frame on the bench topic, built without any catalog round trip."""
    return DataFrame(session, SCAN)


def query_of(frame: DataFrame) -> dict[str, Any]:
    from omniframes.compile.semantic import compile_plan

    envelope = compile_plan(frame.logical_plan, options=frame.session.envelope_options()).envelope
    query: dict[str, Any] = envelope["query"]
    return query


# --------------------------------------------------------------------------------------
# Transformations are immutable
# --------------------------------------------------------------------------------------


def test_every_transformation_returns_a_new_frame(df: DataFrame) -> None:
    selected = df.select("users.state")
    filtered = selected.filter(F.col("users.state") == "California")

    assert selected is not df
    assert filtered is not selected
    assert df.logical_plan is SCAN, "the original frame is untouched"
    assert isinstance(selected.logical_plan, Project)
    assert isinstance(filtered.logical_plan, Filter)


def test_select_accepts_names_columns_and_iterables(df: DataFrame) -> None:
    from_strings = df.select("users.state", "users.age")
    from_columns = df.select(F.col("users.state"), F.col("users.age"))
    from_iterable = df.select(["users.state", "users.age"])

    assert from_strings.columns == from_columns.columns == from_iterable.columns


def test_select_needs_at_least_one_column(df: DataFrame) -> None:
    with pytest.raises(CompileError, match="at least one column"):
        df.select()


def test_alias_collisions_fail_at_build_time_not_at_action_time(df: DataFrame) -> None:
    with pytest.raises(CompileError, match="used twice"):
        df.select(F.col("users.state").alias("x"), F.col("users.age").alias("x"))


def test_filter_auto_wraps_a_bare_column_name(df: DataFrame) -> None:
    query = query_of(df.select("order_items.returned").filter("order_items.returned"))

    assert query["filters"] == {"order_items.returned": {"type": "boolean", "is_negative": False}}


def test_filter_rejects_a_non_expression(df: DataFrame) -> None:
    with pytest.raises(CompileError, match="takes a Column"):
        df.select("users.state").filter(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "condition", ["users.age > 21", "users.state = 'California'", "returned AND shipped"]
)
def test_a_sql_expression_string_is_refused_where_it_was_written(
    df: DataFrame, condition: str
) -> None:
    """PySpark takes a SQL string here; omniframes has no parser, so it would be a *field name*.

    Left alone it became ``filters['users.age > 21']`` — which ``explain()`` renders byte for
    byte like a compiled predicate and only the server rejects, at action time.
    """
    for method in (df.select("users.state").filter, df.select("users.state").where):
        with pytest.raises(CompileError, match="no SQL-expression parser"):
            method(condition)


def test_a_misspelled_field_name_still_compiles(df: DataFrame) -> None:
    """The plausibility check is shape, not schema: there is no catalog at build time."""
    assert df.select("users.state").filter("users.returnd").columns == ("users.state",)


def test_select_star_says_omniframes_does_not_expand_a_topic(df: DataFrame) -> None:
    """``select("*")`` is exactly the request the no-projection message was written for."""
    with pytest.raises(CompileError, match="never expands a topic"):
        df.select("*")
    with pytest.raises(CompileError, match="never expands a topic"):
        F.col("*")


def test_filters_accumulate_and_and_together(df: DataFrame) -> None:
    query = query_of(
        df.select("users.state", "order_items.status")
        .filter(F.col("users.state") == "California")
        .filter(F.col("order_items.status") == "complete")
    )

    assert set(query["filters"]) == {"users.state", "order_items.status"}


def test_a_later_sort_replaces_the_previous_one(df: DataFrame) -> None:
    frame = df.select("users.state", "users.age").sort("users.age").sort("users.state")
    plan = frame.logical_plan

    assert isinstance(plan, Sort)
    assert not isinstance(plan.child, Sort), "sorting twice is one sort, not two"
    assert [sort["column_name"] for sort in query_of(frame)["sorts"]] == ["users.state"]


def test_order_by_is_the_camel_case_alias(df: DataFrame) -> None:
    assert query_of(df.select("users.state").orderBy("users.state")) == query_of(
        df.select("users.state").sort("users.state")
    )


def test_where_is_the_alias_of_filter(df: DataFrame) -> None:
    predicate = F.col("users.state") == "California"

    assert query_of(df.select("users.state").where(predicate)) == query_of(
        df.select("users.state").filter(predicate)
    )


def test_sort_direction_comes_from_the_column(df: DataFrame) -> None:
    sorts = query_of(df.select("users.age").sort(F.col("users.age").desc()))["sorts"]

    assert sorts[0]["sort_descending"] is True


# --------------------------------------------------------------------------------------
# Limit and offset
# --------------------------------------------------------------------------------------


def test_the_default_limit_is_explicit(df: DataFrame) -> None:
    assert query_of(df.select("users.state"))["limit"] == DEFAULT_FETCH_LIMIT


def test_limit_none_is_unlimited(df: DataFrame) -> None:
    assert query_of(df.select("users.state").limit(None))["limit"] is None


def test_limits_compose_to_the_tightest(df: DataFrame) -> None:
    frame = df.select("users.state")

    assert query_of(frame.limit(50).limit(10))["limit"] == 10
    assert query_of(frame.limit(10).limit(50))["limit"] == 10
    assert query_of(frame.limit(None).limit(10))["limit"] == 10
    assert query_of(frame.limit(10).limit(None))["limit"] == 10


@pytest.mark.parametrize("value", [0, -3, True])
def test_limit_rejects_nonsense(df: DataFrame, value: object) -> None:
    with pytest.raises(CompileError, match="positive integer"):
        df.select("users.state").limit(value)  # type: ignore[arg-type]


def test_offset_without_a_limit_keeps_the_default(df: DataFrame) -> None:
    query = query_of(df.select("users.state").offset(10))

    assert query["offset"] == 10
    assert query["limit"] == DEFAULT_FETCH_LIMIT


def test_offset_accumulates_and_keeps_the_limit(df: DataFrame) -> None:
    frame = df.select("users.state").limit(5).offset(2).offset(3)
    plan = frame.logical_plan

    assert isinstance(plan, Limit)
    assert plan.n == 5
    assert plan.offset == 5
    assert not isinstance(plan.child, Limit), "limit and offset live in one node"


def test_offset_rejects_negatives(df: DataFrame) -> None:
    with pytest.raises(CompileError, match="non-negative"):
        df.select("users.state").offset(-1)


def test_a_bare_limit_node_defaults_to_unset(df: DataFrame) -> None:
    assert Limit(SCAN).n is UNSET


# --------------------------------------------------------------------------------------
# Compilation surface (no I/O)
# --------------------------------------------------------------------------------------


def test_columns_are_available_without_a_round_trip(df: DataFrame, handler: FakeOmniAPI) -> None:
    frame = df.select("users.state", F.col("users.age").alias("age"))

    assert frame.columns == ("users.state", "age")
    assert handler.requests == []


def test_repr_shows_the_columns_or_the_plan(df: DataFrame) -> None:
    assert repr(df.select("users.state")) == "DataFrame[users.state]"
    assert "Scan plan" in repr(df), "an un-projected frame cannot name its columns yet"


def test_an_ad_hoc_aggregation_in_select_decomposes_like_the_group_by(df: DataFrame) -> None:
    """M3: ``select()`` with an aggregation IS the group-by, so it splits the same way.

    Until M3 this raised ``not yet supported``; the two spellings now agree on the plan as well
    as on the message, which is the whole point of "selection is the group-by".
    """
    selected = df.select("users.state", F.count_distinct("users.id"))
    grouped = df.group_by("users.state").agg(F.count_distinct("users.id"))

    assert selected.columns == grouped.columns == ("users.state", "count_distinct(users.id)")
    assert selected.explain() == grouped.explain()


def test_filtering_on_a_measure_is_a_measure_keyed_filter(df: DataFrame) -> None:
    """Post-aggregation filters on governed measures ARE tier 1 — a server-side HAVING."""
    query = query_of(df.select("users.state").filter(F.measure("order_items.count") > 10))

    assert query["filters"] == {
        "order_items.count": {
            "type": "number",
            "kind": "GREATER_THAN",
            "values": ["10"],
            "is_inclusive": False,
        }
    }
    assert query["fields"] == ["users.state"], "a filtered measure need not be selected"


def test_filtering_on_an_ad_hoc_aggregation_is_reported_the_same_way(df: DataFrame) -> None:
    frame = df.select("users.state").filter(F.count_distinct("users.id") > 10)

    with pytest.raises(CompileError, match="not yet supported"):
        frame.collect()


def test_explain_names_the_tier_the_source_and_the_filters(df: DataFrame) -> None:
    text = (
        df.select("users.state", "order_items.status")
        .filter(F.col("users.state") == "California")
        .filter(~F.col("order_items.returned"))
        .sort(F.col("users.state").desc())
        .limit(10)
        .explain()
    )

    assert "== Physical plan ==" in text
    assert "Remote [tier 1 · semantic → POST /api/v1/query/run]" in text
    assert f"topic: {BENCH_TOPIC_NAME}   model: {BENCH_MODEL_NAME}" in text
    assert "users.state = 'California'" in text
    assert "NOT order_items.returned" in text
    assert "sort: users.state DESC" in text
    assert "limit: 10" in text
    assert "version: 9" in text
    assert "Local [pandas]\n  (none — fully pushed down)" in text


def test_explain_renders_the_limit_trichotomy(df: DataFrame) -> None:
    frame = df.select("users.state")

    assert "limit: 50000" in frame.explain()
    assert "limit: unlimited (null)" in frame.limit(None).explain()
    assert "offset: 7" in frame.limit(3).offset(7).explain()


# --------------------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------------------


def test_collect_returns_a_normalized_arrow_table(df: DataFrame) -> None:
    table = df.select("users.state", "order_items.status").limit(3).collect()

    assert table.column_names == ["users.state", "order_items.status"]
    assert table.num_rows == 3


def test_aliases_are_applied_as_a_rename_after_the_result_arrives(df: DataFrame) -> None:
    table = df.select(F.col("users.state").alias("state")).limit(2).collect()

    assert table.column_names == ["state"]


def test_selecting_a_field_bare_and_aliased_returns_both_columns(df: DataFrame) -> None:
    """``.columns`` and the collected table must agree — the wire returning one column is not
    a licence to drop one of the two the user asked for (tier 2 writes ``x, x AS y``)."""
    frame = df.select("users.state", F.col("users.state").alias("s")).limit(2)

    assert frame.columns == ("users.state", "s")
    assert frame.collect().column_names == ["users.state", "s"]


def test_to_pandas_and_the_camel_case_aliases(df: DataFrame) -> None:
    frame = df.select("users.state").limit(2)

    assert list(frame.to_pandas().columns) == ["users.state"]
    assert list(frame.toPandas().columns) == ["users.state"]
    assert frame.toArrow().num_rows == 2


def test_count_counts_the_materialized_frame(df: DataFrame) -> None:
    assert df.select("users.state").limit(7).count() == 7


def test_first_returns_a_row_or_none(df: DataFrame) -> None:
    row = df.select("users.state").sort("users.state").first()

    assert row is not None
    assert set(row) == {"users.state"}

    empty = df.select("users.state").filter(F.col("users.state") == "Atlantis").first()
    assert empty is None


def test_first_does_not_warn_about_its_own_limit_of_one(df: DataFrame) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        assert df.select("users.state").first() is not None


def test_show_prints_a_table_and_says_when_more_rows_exist(
    df: DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    df.select("users.state", "order_items.status").show(3)
    printed = capsys.readouterr().out

    assert "users.state" in printed
    assert printed.count("\n") >= 6, "header rule, header, rule, rows, rule"
    assert "only showing top 3 rows" in printed


def test_show_does_not_cry_wolf_about_its_own_limit(df: DataFrame) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", TruncationWarning)
        df.select("users.state").show(2)


def test_show_omits_the_footer_when_the_frame_fits(
    df: DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    df.select("users.state").filter(F.col("users.state") == "Atlantis").show(3)

    assert "only showing" not in capsys.readouterr().out


def test_show_rejects_a_nonsense_row_count(df: DataFrame) -> None:
    with pytest.raises(CompileError, match="positive"):
        df.select("users.state").show(0)


# --------------------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------------------


def test_a_full_page_warns_that_rows_are_probably_missing(df: DataFrame) -> None:
    with pytest.warns(TruncationWarning, match="applied limit"):
        df.select("users.state").limit(3).collect()


def test_a_partial_page_does_not_warn(df: DataFrame, recwarn: pytest.WarningsRecorder) -> None:
    df.select("users.state").filter(F.col("users.state") == "Atlantis").collect()

    assert [w for w in recwarn if issubclass(w.category, TruncationWarning)] == []


def test_count_inherits_the_truncation_warning(df: DataFrame) -> None:
    with pytest.warns(TruncationWarning):
        assert df.select("users.state").limit(5).count() == 5


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


def test_schema_uses_plan_only_and_never_runs_the_query(
    df: DataFrame, handler: FakeOmniAPI
) -> None:
    schema = df.select("users.state", "users.age").schema

    assert schema.names == ("users.state", "users.age")
    assert schema["users.age"].data_type is OmniDataType.NUMBER
    runs = [request for request in handler.requests if request.path.endswith("/query/run")]
    assert len(runs) == 1
    assert runs[0].body["planOnly"] is True, "df.schema plans, it does not execute"


def test_schema_is_cached_per_frame(df: DataFrame, handler: FakeOmniAPI) -> None:
    frame = df.select("users.state")
    assert frame.schema is frame.schema

    runs = [request for request in handler.requests if request.path.endswith("/query/run")]
    assert len(runs) == 1


def test_schema_reports_alias_names(df: DataFrame) -> None:
    schema = df.select(F.col("users.state").alias("state")).schema

    assert schema.names == ("state",)


# --------------------------------------------------------------------------------------
# Formatted grains and generated aliases (docs/SQLTIER.md §5/§3.2)
# --------------------------------------------------------------------------------------

MONTH = "order_items.created_at[month]"


def test_schema_collapses_a_formatted_grain_pair(df: DataFrame) -> None:
    """The wire returns ``…[month]__raw`` + a formatted string; the frame has one TIMESTAMP."""
    schema = df.select(F.col("order_items.created_at").grain("month")).schema

    assert schema.names == (MONTH,)
    assert schema[MONTH].data_type is OmniDataType.TIMESTAMP


def test_a_formatted_grain_collects_as_the_raw_timestamp(df: DataFrame) -> None:
    frame = df.select(F.col("order_items.created_at").grain("month")).limit(5)

    with pytest.warns(TruncationWarning):
        table = frame.collect()

    assert table.column_names == list(frame.columns) == [MONTH]
    assert table.schema.field(MONTH).type == pa.timestamp("us", tz="UTC")


def test_schema_names_a_tier_2_expression_item(df: DataFrame, handler: FakeOmniAPI) -> None:
    """``of_expr_<n>`` comes back under a scope prefix, so the schema matches it by suffix."""
    frame = df.group_by("users.state").agg(F.count_distinct("users.id").alias("buyers"))

    schema = frame.schema

    assert schema.names == ("users.state", "buyers")
    assert schema["buyers"].data_type is OmniDataType.NUMBER
    (plan,) = [request for request in handler.requests if request.path.endswith("/query/run")]
    assert "AS of_expr_1" in plan.body["query"]["userEditedSQL"]


def test_a_tier_2_grain_and_expression_collect_under_their_user_facing_names(
    df: DataFrame,
) -> None:
    frame = df.group_by(F.col("order_items.created_at").grain("month")).agg(
        F.count_distinct("users.id").alias("buyers")
    )

    table = frame.collect()

    assert table.column_names == list(frame.columns) == [MONTH, "buyers"]
    assert table.schema.field(MONTH).type == pa.timestamp("us", tz="UTC")


# --------------------------------------------------------------------------------------
# join() / union() — build-time validation (M4)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("how", "expected"),
    [
        ("inner", JoinHow.INNER),
        ("left", JoinHow.LEFT),
        ("LEFT", JoinHow.LEFT),
        ("left_outer", JoinHow.LEFT),
        ("right", JoinHow.RIGHT),
        ("outer", JoinHow.OUTER),
        ("full", JoinHow.OUTER),
        ("full_outer", JoinHow.OUTER),
        ("fullouter", JoinHow.OUTER),
    ],
)
def test_join_accepts_the_documented_spellings_of_each_kind(
    df: DataFrame, how: str, expected: JoinHow
) -> None:
    joined = df.select("users.state").join(df.select("users.state"), "users.state", how)
    plan = joined.logical_plan

    assert isinstance(plan, Join)
    assert plan.how is expected
    assert plan.on == ("users.state",)


def test_join_takes_one_key_or_several(df: DataFrame) -> None:
    left = df.select("users.state", "users.age")
    plan = left.join(left, ["users.state", "users.age"]).logical_plan

    assert isinstance(plan, Join)
    assert plan.on == ("users.state", "users.age")


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda df: df.join(df, "k", "anti"), "not a join kind"),
        (lambda df: df.join(df, []), "at least one column"),
        (lambda df: df.join(df, [1]), "names of columns"),
        (lambda df: df.join("nope", "k"), "takes another DataFrame"),
        (lambda df: df.union("nope"), "takes another DataFrame"),
    ],
)
def test_join_and_union_refuse_what_they_cannot_mean(
    df: DataFrame, call: Any, message: str
) -> None:
    frame = df.select("users.state")
    with pytest.raises(CompileError, match=message):
        call(frame)


def test_joining_frames_from_two_sessions_is_refused(df: DataFrame, session: OmniSession) -> None:
    """Both sides are fetched separately but combined in one process; one session owns that."""
    other = DataFrame(OmniSession.builder.transport(session._transport).get_or_create(), SCAN)

    with pytest.raises(CompileError, match="same OmniSession"):
        df.select("users.state").join(other.select("users.state"), "users.state")


def test_a_totalled_frame_cannot_be_joined_or_unioned(df: DataFrame) -> None:
    """Totals are a tier-1 server feature over one query; a join is two."""
    totalled = df.select("users.state", F.measure("order_items.count")).with_totals()
    plain = df.select("users.state", F.measure("order_items.count"))

    with pytest.raises(CompileError, match="with_totals"):
        totalled.join(plain, "users.state")
    with pytest.raises(CompileError, match="with_totals"):
        plain.union(totalled)


def test_union_and_its_pyspark_alias_build_the_same_node(df: DataFrame) -> None:
    left = df.select("users.state")
    right = df.select("users.state")

    assert isinstance(left.union(right).logical_plan, Union)
    assert isinstance(left.unionAll(right).logical_plan, Union)


def test_a_join_is_lazy_like_every_other_transformation(
    df: DataFrame, handler: FakeOmniAPI
) -> None:
    df.select("users.state").join(df.select("users.state"), "users.state")

    assert not [r for r in handler.requests if r.path.endswith("/query/run")]
