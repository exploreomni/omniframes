"""Golden lane: DataFrame code compiled to the exact run envelope it puts on the wire.

Each case builds a DataFrame against a stub transport (the golden lane never talks to anything),
compiles it, and compares the envelope to a checked-in snapshot.  The snapshots are meant to be
diff-reviewed: a change to one of these files in a PR is a change to what omniframes sends.

Regenerate after an intentional change::

    OMNIFRAMES_UPDATE_SNAPSHOTS=1 uv run pytest tests/golden -q
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

import pytest
from tests.fakes import BENCH_MODEL_ID, BENCH_MODEL_NAME, BENCH_TOPIC_NAME

from omniframes import functions as F
from omniframes.dataframe import DataFrame
from omniframes.plan import nodes
from omniframes.session import OmniSession

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
UPDATE_ENV_VAR = "OMNIFRAMES_UPDATE_SNAPSHOTS"
PREFIX = "semantic_"

BRANCH_ID = "9c3b1e5e-2f4a-4d1b-9a7e-6b0f2d8c4a11"


class NoNetworkTransport:
    """A transport that refuses every call: compiling must never need the network."""

    def _refuse(self, what: str) -> NoReturn:
        raise AssertionError(f"the golden lane must not call {what}()")

    def run(self, envelope: dict[str, Any], *, deadline_seconds: float | None = None) -> NoReturn:
        self._refuse("run")

    def plan(self, envelope: dict[str, Any]) -> NoReturn:
        self._refuse("plan")

    def whoami(self, model_ids: tuple[str, ...] = ()) -> NoReturn:
        self._refuse("whoami")

    def list_models(self, **params: Any) -> NoReturn:
        self._refuse("list_models")

    def list_topics(self, model_id: str) -> NoReturn:
        self._refuse("list_topics")

    def get_topic(self, model_id: str, topic_name: str) -> NoReturn:
        self._refuse("get_topic")

    def list_views(self, model_id: str) -> NoReturn:
        self._refuse("list_views")

    def document_queries(self, document_identifier: str) -> NoReturn:
        self._refuse("document_queries")

    def generate_query(self, body: dict[str, Any]) -> NoReturn:
        self._refuse("generate_query")

    def close(self) -> None:
        return None


def _session(**options: Any) -> OmniSession:
    builder = OmniSession.builder.transport(NoNetworkTransport())
    for name, value in options.items():
        getattr(builder, name)(value)
    return builder.get_or_create()


SESSION = _session()

TOPIC_SCAN = nodes.Scan(
    nodes.TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)
VIEW_SCAN = nodes.Scan(
    nodes.ViewScan(model_name=BENCH_MODEL_NAME, model_id=BENCH_MODEL_ID, view="users")
)
SQL_SCAN = nodes.Scan(
    nodes.SqlScan(
        model_id=BENCH_MODEL_ID,
        sql=(
            "SELECT u.state AS state, SUM(oi.sale_price) AS revenue\n"
            "FROM order_items oi LEFT JOIN users u ON u.id = oi.user_id\n"
            "GROUP BY 1"
        ),
        model_name=BENCH_MODEL_NAME,
    )
)


def topic() -> DataFrame:
    """A frame over the governed topic (join paths come from the model)."""
    return DataFrame(SESSION, TOPIC_SCAN)


def view() -> DataFrame:
    """A frame over a bare view (no topic, no governed joins)."""
    return DataFrame(SESSION, VIEW_SCAN)


# ---------------------------------------------------------------------------------------
# Snapshot plumbing (same contract as tests/golden/test_query_json.py)
# ---------------------------------------------------------------------------------------


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def assert_matches_snapshot(name: str, payload: dict[str, Any]) -> None:
    path = SNAPSHOT_DIR / f"{name}.json"
    serialized = _serialize(payload)

    if os.environ.get(UPDATE_ENV_VAR) == "1":
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
        return

    if not path.exists():
        pytest.fail(
            f"missing snapshot {path}; regenerate with "
            f"`{UPDATE_ENV_VAR}=1 uv run pytest tests/golden -q` and review the diff"
        )

    committed = path.read_text(encoding="utf-8")
    assert json.loads(committed) == payload, (
        f"wire payload drifted from {path.name}; if the change is intended regenerate with "
        f"{UPDATE_ENV_VAR}=1"
    )
    assert committed == serialized, f"{path.name} is not in canonical formatting"


def envelope(df: DataFrame) -> dict[str, Any]:
    """The exact body ``collect()`` would POST.

    Goes through the frame's own compilation rather than :func:`compile_plan` directly so that
    frame-level state — today ``with_totals()`` — is part of what the snapshots pin.
    """
    return df._compiled().envelope


# ---------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------


def select_bare() -> DataFrame:
    """The smallest frame: a topic scan plus a projection."""
    return topic().select("users.state", "order_items.status")


def view_scan() -> DataFrame:
    """A bare view sends ``table`` and no ``join_paths_from_topic_name``."""
    return view().select("users.state", "users.age")


def filter_string_equals() -> DataFrame:
    return select_bare().filter(F.col("users.state") == "California")


def filter_string_not_equals() -> DataFrame:
    """``!=`` is EQUALS with ``is_negative``; the wire has no NOT_EQUALS kind."""
    return select_bare().filter(F.col("order_items.status") != "complete")


def filter_composite_or() -> DataFrame:
    """An OR within one field becomes a composite; across fields it would not compile."""
    state = F.col("users.state")
    return select_bare().filter((state == "California") | (state == "Texas"))


def filter_isin() -> DataFrame:
    return select_bare().filter(F.col("users.state").isin("Ohio", "Texas", "Georgia"))


def filter_string_predicates() -> DataFrame:
    """CONTAINS / STARTS_WITH / ENDS_WITH / SQL_LIKE, one per field so nothing merges."""
    return (
        topic()
        .select("users.state", "users.name", "users.email", "order_items.notes")
        .filter(
            F.col("users.state").contains("cal", case_insensitive=True)
            & F.col("users.name").starts_with("A")
            & F.col("users.email").ends_with("@example.com")
            & F.col("order_items.notes").like("%rush%")
        )
    )


def filter_between_number() -> DataFrame:
    """``between`` includes both ends over numbers: ``10 <= x <= 99.5``.

    Not the wire's BETWEEN kind — its upper bound is exclusive — so this is a composite of two
    inclusive comparisons inside the one entry that field is allowed.
    """
    return (
        topic()
        .select("order_items.sale_price")
        .filter(F.col("order_items.sale_price").between(10, Decimal("99.5")))
    )


def filter_between_date() -> DataFrame:
    """Dates stay half-open — ``[2025-07-01, 2026-07-01)`` — see :meth:`Column.between`."""
    return (
        topic()
        .select("order_items.created_at")
        .filter(F.col("order_items.created_at").between(date(2025, 7, 1), date(2026, 7, 1)))
    )


def filter_numeric_comparisons() -> DataFrame:
    """Two predicates on one field merge into a composite AND; ``>=``/``<`` set is_inclusive."""
    age = F.col("users.age")
    return topic().select("users.age").filter((age >= 21) & (age < 65))


def filter_date_comparisons() -> DataFrame:
    """``>=`` is ON_OR_AFTER, ``<`` is BEFORE (exclusive) — the only two the wire has."""
    created = F.col("order_items.created_at")
    return (
        topic()
        .select("order_items.created_at")
        .filter((created >= datetime(2026, 1, 1, 12, 30, 0)) & (created < date(2026, 7, 1)))
    )


def filter_is_null() -> DataFrame:
    return (
        topic()
        .select("users.state", "order_items.discount")
        .filter(F.col("users.state").is_null() & F.col("order_items.discount").is_not_null())
    )


def filter_boolean() -> DataFrame:
    """A bare boolean field is a boolean filter; ``~`` flips ``is_negative``."""
    return (
        topic()
        .select("order_items.returned", "users.is_business")
        .filter(F.col("order_items.returned") & ~F.col("users.is_business"))
    )


def filter_negations() -> DataFrame:
    """Every negation route: ~isin, ~contains, ~is_null, ~(OR composite)."""
    status = F.col("order_items.status")
    return (
        topic()
        .select("users.state", "users.name", "order_items.status", "products.category")
        .filter(
            ~F.col("users.state").isin("Ohio", "Texas")
            & ~F.col("users.name").contains("test")
            & ~F.col("products.category").is_null()
            & ~((status == "cancelled") | (status == "returned"))
        )
    )


def grain_projection_timestamp() -> DataFrame:
    """A timestamp grain: bracketed in ``fields``, BARE in ``filters``, as a date filter."""
    month = F.col("order_items.created_at").grain("month")
    return topic().select(month, "order_items.status").filter(month >= date(2026, 1, 1)).sort(month)


def grain_projection_number() -> DataFrame:
    """A number grain: bracketed in ``fields`` AND in ``filters``, as a number filter."""
    hour = F.col("order_items.created_at").grain("hour_of_day")
    return topic().select(hour, "order_items.status").filter(hour >= 8)


def sort_alias_reverse_resolution() -> DataFrame:
    """Sorts and filters written against an alias resolve back to the wire name."""
    return (
        topic()
        .select(
            F.col("users.state").alias("state"),
            F.col("order_items.created_at").grain("month").alias("month"),
        )
        .filter(F.col("state") == "California")
        .sort(F.col("month").desc(), F.col("state"))
    )


def limit_default() -> DataFrame:
    """No ``.limit()`` still sends an explicit limit — DEFAULT_FETCH_LIMIT."""
    return select_bare()


def limit_user() -> DataFrame:
    return select_bare().limit(10)


def limit_unlimited() -> DataFrame:
    """``limit(None)`` is the wire's ``null`` — genuinely unlimited."""
    return select_bare().limit(None)


def limit_offset() -> DataFrame:
    return select_bare().sort("users.state").limit(100).offset(25)


def measure_selection() -> DataFrame:
    """Dimensions + governed measures: in Omni the selection *is* the group-by."""
    return (
        topic()
        .select(
            "users.state",
            F.measure("order_items.total_sale_price").alias("revenue"),
            F.measure("order_items.count"),
        )
        .filter(F.col("order_items.status") == "complete")
        .sort(F.col("revenue").desc())
        .limit(20)
    )


def aggregate_group_by() -> DataFrame:
    """``group_by().agg()`` — sugar over the projection below, byte for byte."""
    return (
        topic()
        .group_by("users.state")
        .agg(
            F.measure("order_items.total_sale_price"),
            F.measure("users.count"),
        )
    )


def aggregate_select_equivalent() -> DataFrame:
    """The same query written as a ``select()``: in Omni the selection IS the group-by."""
    return topic().select(
        "users.state",
        F.measure("order_items.total_sale_price"),
        F.measure("users.count"),
    )


def aggregate_grain_alias_sort() -> DataFrame:
    """A grained group key, aliased client-side, sorted by the alias — descending."""
    month = F.col("order_items.created_at").grain("month").alias("month")
    return (
        topic()
        .group_by(month)
        .agg(F.measure("order_items.total_sale_price").alias("revenue"))
        .sort(F.col("month").desc())
    )


def filter_measure_simple() -> DataFrame:
    """A measure-keyed entry: a genuine HAVING over the aggregate (CONTRACT_NOTES §3.1)."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.measure("order_items.total_sale_price"))
        .filter(F.measure("order_items.total_sale_price") > 50000)
    )


def filter_measure_composite() -> DataFrame:
    """Two conditions on ONE measure share one entry — the wire keys filters by field."""
    revenue = F.measure("order_items.total_sale_price")
    return (
        topic().group_by("users.state").agg(revenue).filter(revenue > 20000).filter(revenue < 80000)
    )


def filter_measure_unselected() -> DataFrame:
    """A measure may be filtered without being selected; the server projects it away."""
    return (
        topic()
        .select("users.state", F.measure("order_items.count"))
        .filter(F.measure("order_items.total_sale_price") > 50000)
    )


def filter_measure_and_dimension() -> DataFrame:
    """WHERE and HAVING in one query: both live in ``filters``, keyed by their own field."""
    month = F.col("order_items.created_at").grain("month")
    return (
        topic()
        .group_by(month)
        .agg(F.measure("order_items.total_sale_price"))
        .filter(F.col("order_items.created_at").between(date(2025, 7, 1), date(2026, 7, 1)))
        .filter(F.measure("order_items.total_sale_price") > 45000)
    )


def with_totals() -> DataFrame:
    """``with_totals()`` asks for the grand total over every measure in the query."""
    return (
        topic()
        .group_by("users.state")
        .agg(F.measure("order_items.total_sale_price"), F.measure("order_items.count"))
        .with_totals()
    )


def sql_scan() -> DataFrame:
    """A raw-SQL job: ``userEditedSQL`` **and** the ``rewriteSql: false`` that makes it run.

    Without the marker the server drops the SQL and plans the (empty) query object instead
    (CONTRACT_NOTES §3.4), which is why this snapshot exists: the two keys have to move together
    or not at all.
    """
    return DataFrame(SESSION, SQL_SCAN)


def envelope_session_options() -> DataFrame:
    """Session-level knobs ride the envelope, never the query object (branchId especially)."""
    session = _session(branch=BRANCH_ID, cache="SkipCache", timezone="America/Los_Angeles")
    return DataFrame(session, TOPIC_SCAN).select("users.state").limit(3)


CASES: dict[str, Callable[[], DataFrame]] = {
    "select_bare": select_bare,
    "view_scan": view_scan,
    "filter_string_equals": filter_string_equals,
    "filter_string_not_equals": filter_string_not_equals,
    "filter_composite_or": filter_composite_or,
    "filter_isin": filter_isin,
    "filter_string_predicates": filter_string_predicates,
    "filter_between_number": filter_between_number,
    "filter_between_date": filter_between_date,
    "filter_numeric_comparisons": filter_numeric_comparisons,
    "filter_date_comparisons": filter_date_comparisons,
    "filter_is_null": filter_is_null,
    "filter_boolean": filter_boolean,
    "filter_negations": filter_negations,
    "grain_projection_timestamp": grain_projection_timestamp,
    "grain_projection_number": grain_projection_number,
    "sort_alias_reverse_resolution": sort_alias_reverse_resolution,
    "limit_default": limit_default,
    "limit_user": limit_user,
    "limit_unlimited": limit_unlimited,
    "limit_offset": limit_offset,
    "measure_selection": measure_selection,
    "aggregate_group_by": aggregate_group_by,
    "aggregate_select_equivalent": aggregate_select_equivalent,
    "aggregate_grain_alias_sort": aggregate_grain_alias_sort,
    "filter_measure_simple": filter_measure_simple,
    "filter_measure_composite": filter_measure_composite,
    "filter_measure_unselected": filter_measure_unselected,
    "filter_measure_and_dimension": filter_measure_and_dimension,
    "with_totals": with_totals,
    "sql_scan": sql_scan,
    "envelope_session_options": envelope_session_options,
}


# ---------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CASES))
def test_compiled_envelope_snapshot(name: str) -> None:
    assert_matches_snapshot(f"{PREFIX}{name}", envelope(CASES[name]()))


def test_no_orphan_semantic_snapshots() -> None:
    expected = {f"{PREFIX}{name}.json" for name in CASES}
    # `semantic_sqltier_*.json` belongs to test_sql_snapshots.py (tier-2 envelopes); that
    # module has its own orphan check over the same directory.
    actual = {
        path.name
        for path in SNAPSHOT_DIR.glob(f"{PREFIX}*.json")
        if not path.name.startswith("semantic_sqltier_")
    }

    assert actual == expected, "the semantic snapshots directory is out of sync with the cases"


def test_every_envelope_sends_an_explicit_limit_and_version_9() -> None:
    """Two invariants no case may break (docs/DESIGN.md §3, CONTRACT_NOTES §3)."""
    for name, build in CASES.items():
        query = envelope(build())["query"]
        assert "limit" in query, f"{name} omitted the limit"
        assert query["version"] == 9, f"{name} sent version {query['version']}"


def test_branch_id_is_top_level_only() -> None:
    """``branchId`` inside ``query`` is a hard 400 (CONTRACT_NOTES §2.1)."""
    payload = envelope(envelope_session_options())

    assert payload["branchId"] == BRANCH_ID
    assert "branchId" not in payload["query"]
    assert "branchId" not in envelope(select_bare())


def test_topic_and_view_scans_differ_only_in_the_source_keys() -> None:
    topic_query = envelope(select_bare())["query"]
    view_query = envelope(view_scan())["query"]

    assert topic_query["join_paths_from_topic_name"] == BENCH_TOPIC_NAME
    assert topic_query["table"] == "order_items"
    assert "join_paths_from_topic_name" not in view_query
    assert view_query["table"] == "users"


def test_grain_filter_rule_goes_both_ways() -> None:
    """Timestamp grains filter the bare field; number grains filter the bracketed one."""
    timestamp = envelope(grain_projection_timestamp())["query"]
    number = envelope(grain_projection_number())["query"]

    assert timestamp["fields"][0] == "order_items.created_at[month]"
    assert list(timestamp["filters"]) == ["order_items.created_at"]
    assert timestamp["filters"]["order_items.created_at"]["type"] == "date"

    assert number["fields"][0] == "order_items.created_at[hour_of_day]"
    assert list(number["filters"]) == ["order_items.created_at[hour_of_day]"]
    assert number["filters"]["order_items.created_at[hour_of_day]"]["type"] == "number"


def test_group_by_agg_and_select_compile_to_the_same_envelope() -> None:
    """The headline M2 invariant: ``group_by().agg()`` is sugar, not a second code path."""
    assert envelope(aggregate_group_by()) == envelope(aggregate_select_equivalent())


def test_a_measure_filter_is_keyed_by_the_measure_and_carries_the_number_arm() -> None:
    query = envelope(filter_measure_simple())["query"]
    entry = query["filters"]["order_items.total_sale_price"]

    assert entry["type"] == "number", "the HAVING side takes the number arm (CONTRACT_NOTES §3.1)"
    assert entry["values"] == ["50000"], "number filter values are STRINGS on the wire"
    assert query["fields"] == ["users.state", "order_items.total_sale_price"]


def test_conditions_on_one_measure_share_one_filters_entry() -> None:
    query = envelope(filter_measure_composite())["query"]

    assert list(query["filters"]) == ["order_items.total_sale_price"]
    entry = query["filters"]["order_items.total_sale_price"]
    assert entry["type"] == "composite"
    assert [child["kind"] for child in entry["filters"]] == ["GREATER_THAN", "LESS_THAN"]


def test_a_filtered_measure_need_not_be_a_selected_field() -> None:
    query = envelope(filter_measure_unselected())["query"]

    assert query["fields"] == ["users.state", "order_items.count"]
    assert list(query["filters"]) == ["order_items.total_sale_price"]


def test_with_totals_sends_the_grand_total_key() -> None:
    query = envelope(with_totals())["query"]

    assert query["column_totals"] == {"::total::": {"type": "aggregation"}}
    assert envelope(aggregate_group_by())["query"]["column_totals"] == {}


def test_between_bounds_differ_between_numbers_and_dates() -> None:
    """Numbers are inclusive at both ends; dates are half-open (see Column.between)."""
    number = envelope(filter_between_number())["query"]["filters"]["order_items.sale_price"]
    dates = envelope(filter_between_date())["query"]["filters"]["order_items.created_at"]

    assert number["type"] == "composite"
    assert [(f["kind"], f["is_inclusive"]) for f in number["filters"]] == [
        ("GREATER_THAN", True),
        ("LESS_THAN", True),
    ]
    assert dates == {
        "type": "date",
        "kind": "BETWEEN",
        "left_side": "2025-07-01",
        "right_side": "2026-07-01",
    }


def test_aliases_never_reach_the_wire() -> None:
    query = envelope(sort_alias_reverse_resolution())["query"]

    assert query["fields"] == ["users.state", "order_items.created_at[month]"]
    assert list(query["filters"]) == ["users.state"]
    assert [sort["column_name"] for sort in query["sorts"]] == [
        "order_items.created_at[month]",
        "users.state",
    ]
    aliased = {"state", "month"}
    assert aliased.isdisjoint(query["fields"])
    assert aliased.isdisjoint({sort["column_name"] for sort in query["sorts"]})
    assert aliased.isdisjoint(query["filters"])
