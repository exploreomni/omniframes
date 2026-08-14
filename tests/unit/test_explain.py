"""Unit tests for ``explain()``'s filter rendering (docs/INTERNALS.md §4).

``explain()`` is the promise that nothing runs in secret, so what it prints is user-facing
output with a contract of its own.  Several arms of ``describe_filter`` were reachable only
through a query shape no test built — a date filter that is not ``BETWEEN``, ``IS_EMPTY``, a
boolean placeholder, a multi-value ``IN`` — so this file exercises the table directly, one row
per wire arm, the way ``tests/unit/test_sqlgen.py`` pins the SQL translation.
"""

from __future__ import annotations

import pytest

from omniframes.compile.explain import describe_filter, describe_filters
from omniframes.compile.querymodel import (
    BooleanFilter,
    CompositeFilter,
    DateFilter,
    DateFilterKind,
    Filter,
    FilterConjunction,
    NullFilter,
    NumberFilter,
    NumberFilterKind,
    StringFilter,
    StringFilterKind,
)

STATE = "users.state"
AGE = "users.age"
CREATED = "order_items.created_at"


@pytest.mark.parametrize(
    ("name", "flt", "expected"),
    [
        # -- string ------------------------------------------------------------------
        (STATE, StringFilter(StringFilterKind.EQUALS, ["Ohio"]), "users.state = 'Ohio'"),
        (
            STATE,
            StringFilter(StringFilterKind.EQUALS, ["Ohio", "Texas"]),
            "users.state IN ('Ohio', 'Texas')",
        ),
        (STATE, StringFilter(StringFilterKind.IS_EMPTY, []), "users.state IS EMPTY"),
        (
            STATE,
            StringFilter(StringFilterKind.CONTAINS, ["cal"]),
            "users.state CONTAINS 'cal'",
        ),
        (
            STATE,
            StringFilter(StringFilterKind.STARTS_WITH, ["New"], case_insensitive=True),
            "users.state STARTS_WITH 'New' (case-insensitive)",
        ),
        (
            STATE,
            StringFilter(StringFilterKind.SQL_LIKE, ["Cal%"]),
            "users.state SQL_LIKE 'Cal%'",
        ),
        # -- number ------------------------------------------------------------------
        (AGE, NumberFilter(NumberFilterKind.EQUALS, [30]), "users.age = 30"),
        (AGE, NumberFilter(NumberFilterKind.EQUALS, [1, 2, 3]), "users.age IN (1, 2, 3)"),
        # BETWEEN's upper bound is EXCLUSIVE on the wire, so it is never rendered as "BETWEEN".
        (AGE, NumberFilter(NumberFilterKind.BETWEEN, [18, 30]), "18 <= users.age < 30"),
        (AGE, NumberFilter(NumberFilterKind.GREATER_THAN, [18]), "users.age > 18"),
        (
            AGE,
            NumberFilter(NumberFilterKind.LESS_THAN, [30], is_inclusive=True),
            "users.age <= 30",
        ),
        # -- date --------------------------------------------------------------------
        (
            CREATED,
            DateFilter(DateFilterKind.ON_OR_AFTER, left_side="2025-07-01"),
            "order_items.created_at ON_OR_AFTER '2025-07-01'",
        ),
        (
            CREATED,
            DateFilter(DateFilterKind.BEFORE, right_side="2026-07-01"),
            "order_items.created_at BEFORE '2026-07-01'",
        ),
        (
            CREATED,
            DateFilter(DateFilterKind.TIME_FOR_UNIT_DURATION, left_side="last quarter"),
            "order_items.created_at TIME_FOR_UNIT_DURATION 'last quarter'",
        ),
        (
            CREATED,
            DateFilter(DateFilterKind.BETWEEN, left_side="2025-07-01", right_side="2026-07-01"),
            "'2025-07-01' <= order_items.created_at < '2026-07-01'",
        ),
        # -- boolean and null --------------------------------------------------------
        ("order_items.returned", BooleanFilter(is_negative=False), "order_items.returned"),
        ("order_items.returned", BooleanFilter(is_negative=True), "NOT order_items.returned"),
        (
            "order_items.returned",
            BooleanFilter(),
            "order_items.returned (boolean placeholder — no-op)",
        ),
        (STATE, NullFilter(), "users.state IS NULL"),
        (STATE, NullFilter(is_negative=True), "users.state IS NOT NULL"),
    ],
)
def test_every_filter_arm_renders_the_way_a_human_reads_it(
    name: str, flt: Filter, expected: str
) -> None:
    assert describe_filter(name, flt) == expected


def test_a_negated_filter_is_wrapped_but_the_two_self_negating_arms_are_not() -> None:
    """``NOT (…)`` around a boolean or a null filter would read as a double negative."""
    negated = StringFilter(StringFilterKind.EQUALS, ["Ohio"], is_negative=True)
    assert describe_filter(STATE, negated) == "NOT (users.state = 'Ohio')"
    assert "NOT (" not in describe_filter(STATE, NullFilter(is_negative=True))
    assert "NOT (" not in describe_filter("order_items.returned", BooleanFilter(is_negative=True))


def test_a_composite_spells_out_its_conjunction_and_its_negated_children() -> None:
    composite = CompositeFilter(
        FilterConjunction.OR,
        [
            StringFilter(StringFilterKind.EQUALS, ["Ohio"]),
            StringFilter(StringFilterKind.EQUALS, ["Texas"], is_negative=True),
        ],
    )

    assert describe_filter(STATE, composite) == (
        "(users.state = 'Ohio' OR NOT (users.state = 'Texas'))"
    )


def test_filters_across_fields_read_as_one_conjunction() -> None:
    """``query.filters`` is keyed by field and ANDs across them (CONTRACT_NOTES §3.1)."""
    rendered = describe_filters(
        {
            STATE: StringFilter(StringFilterKind.EQUALS, ["Ohio"]),
            AGE: NumberFilter(NumberFilterKind.GREATER_THAN, [18]),
        }
    )

    assert rendered == "users.state = 'Ohio' AND users.age > 18"
