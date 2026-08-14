"""Unit tests for the expression tree and the Column wrapper (docs/INTERNALS.md §1)."""

from __future__ import annotations

from datetime import date

import pytest

from omniframes import functions as F
from omniframes.column import (
    AdHocAgg,
    AggFn,
    Arithmetic,
    ArithOp,
    Between,
    BooleanOp,
    BoolOp,
    CmpOp,
    Column,
    Comparison,
    FieldRef,
    IsIn,
    IsNull,
    Literal,
    MeasureRef,
    Not,
    SortKey,
    StringPredicate,
    StrPredKind,
)
from omniframes.dataframe import DataFrame
from omniframes.errors import CompileError

# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def test_col_builds_a_field_reference() -> None:
    assert F.col("users.state").expr == FieldRef("users.state")


def test_col_is_idempotent_over_columns() -> None:
    column = F.col("users.state")

    assert F.col(column) is column


def test_measure_and_literal() -> None:
    assert F.measure("order_items.count").expr == MeasureRef("order_items.count")
    assert F.lit(3).expr == Literal(3)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (F.sum, AggFn.SUM),
        (F.avg, AggFn.AVG),
        (F.min, AggFn.MIN),
        (F.max, AggFn.MAX),
        (F.count, AggFn.COUNT),
    ],
)
def test_aggregations_build_adhoc_nodes(build: object, expected: AggFn) -> None:
    column = build("order_items.sale_price")  # type: ignore[operator]

    assert column.expr == AdHocAgg(expected, FieldRef("order_items.sale_price"))


def test_count_distinct_is_count_plus_distinct() -> None:
    assert F.count_distinct("users.id").expr == AdHocAgg(
        AggFn.COUNT, FieldRef("users.id"), distinct=True
    )


def test_aggregating_a_governed_measure_is_refused() -> None:
    with pytest.raises(CompileError, match="already aggregated"):
        F.sum(F.measure("order_items.total_sale_price"))


def test_expressions_are_frozen() -> None:
    ref = FieldRef("users.state")

    with pytest.raises(AttributeError):
        ref.name = "users.country"  # type: ignore[misc]


def test_a_field_reference_needs_a_name() -> None:
    with pytest.raises(CompileError, match="needs a name"):
        F.col("")


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "op"),
    [
        (lambda c: c == 1, CmpOp.EQ),
        (lambda c: c != 1, CmpOp.NE),
        (lambda c: c < 1, CmpOp.LT),
        (lambda c: c <= 1, CmpOp.LE),
        (lambda c: c > 1, CmpOp.GT),
        (lambda c: c >= 1, CmpOp.GE),
    ],
)
def test_comparison_dunders(build: object, op: CmpOp) -> None:
    column = build(F.col("users.age"))  # type: ignore[operator]

    assert column.expr == Comparison(op, FieldRef("users.age"), Literal(1))


@pytest.mark.parametrize(
    ("build", "op"),
    [
        (lambda c: c + 1, ArithOp.ADD),
        (lambda c: c - 1, ArithOp.SUB),
        (lambda c: c * 1, ArithOp.MUL),
        (lambda c: c / 1, ArithOp.DIV),
    ],
)
def test_arithmetic_dunders(build: object, op: ArithOp) -> None:
    column = build(F.col("users.age"))  # type: ignore[operator]

    assert column.expr == Arithmetic(op, FieldRef("users.age"), Literal(1))


def test_reflected_operators_keep_the_operand_order() -> None:
    assert (1 - F.col("users.age")).expr == Arithmetic(
        ArithOp.SUB, Literal(1), FieldRef("users.age")
    )


def test_boolean_operators_flatten_on_construction() -> None:
    a, b, c = F.col("v.a") == 1, F.col("v.b") == 2, F.col("v.c") == 3
    expr = (a & b & c).expr

    assert isinstance(expr, BooleanOp)
    assert expr.op is BoolOp.AND
    assert len(expr.operands) == 3, "nested ANDs collapse into one n-ary node"


def test_mixed_connectives_do_not_flatten() -> None:
    a, b, c = F.col("v.a") == 1, F.col("v.b") == 2, F.col("v.c") == 3
    expr = ((a & b) | c).expr

    assert isinstance(expr, BooleanOp)
    assert expr.op is BoolOp.OR
    assert len(expr.operands) == 2


def test_invert_builds_not() -> None:
    assert (~(F.col("users.age") == 1)).expr == Not(
        Comparison(CmpOp.EQ, FieldRef("users.age"), Literal(1))
    )


def test_bool_raises_and_names_the_operators() -> None:
    with pytest.raises(TypeError, match=r"`&`.*`\|`.*`~`"):
        bool(F.col("users.state") == "California")


def test_the_bool_message_shows_an_api_omniframes_actually_has() -> None:
    """It used to model ``(df.a == 1) & (df.b == 2)``: omniframes has no column accessor at all.

    A PySpark user believes the message, and ``df.a`` / ``df['a']`` then fail for an unrelated
    reason — twice as far from working code as when they started.
    """
    with pytest.raises(TypeError) as excinfo:
        bool(F.col("users.state") == "California")
    message = str(excinfo.value)

    assert "df.a" not in message
    assert "df[" not in message
    assert "F.col(" in message, "the message must show the only way to name a column"
    assert not hasattr(DataFrame, "__getattr__"), "…because there is no attribute access"
    assert not hasattr(DataFrame, "__getitem__"), "…and no subscripting either"


def test_python_and_would_have_been_silently_wrong() -> None:
    """This is the whole reason ``__bool__`` raises."""
    with pytest.raises(TypeError):
        (F.col("v.a") == 1) and (F.col("v.b") == 2)


def test_columns_stay_hashable_despite_eq_building_a_comparison() -> None:
    column = F.col("users.state")
    other = F.col("users.state")

    assert {column, other} == {column, other}
    assert len({column, other}) == 2, "hashing is by identity, not by expression"
    assert isinstance(column == other, Column)


def test_unsupported_operand_types_are_rejected() -> None:
    with pytest.raises(CompileError, match="cannot use"):
        _ = F.col("users.state") == object()


# --------------------------------------------------------------------------------------
# Predicates
# --------------------------------------------------------------------------------------


def test_is_null_and_is_not_null() -> None:
    assert F.col("users.state").is_null().expr == IsNull(FieldRef("users.state"))
    assert F.col("users.state").is_not_null().expr == Not(IsNull(FieldRef("users.state")))


@pytest.mark.parametrize(
    "values",
    [("Ohio", "Texas"), (["Ohio", "Texas"],), ({"Ohio", "Texas"},)],
    ids=["varargs", "list", "set"],
)
def test_isin_accepts_varargs_or_an_iterable(values: tuple[object, ...]) -> None:
    expr = F.col("users.state").isin(*values).expr

    assert isinstance(expr, IsIn)
    assert set(expr.values) == {"Ohio", "Texas"}


def test_isin_needs_values() -> None:
    with pytest.raises(CompileError, match="at least one value"):
        F.col("users.state").isin()


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("contains", StrPredKind.CONTAINS),
        ("starts_with", StrPredKind.STARTS_WITH),
        ("ends_with", StrPredKind.ENDS_WITH),
        ("like", StrPredKind.LIKE),
    ],
)
def test_string_predicates(method: str, kind: StrPredKind) -> None:
    column = getattr(F.col("users.state"), method)("cal", case_insensitive=True)

    assert column.expr == StringPredicate(kind, FieldRef("users.state"), "cal", True)


def test_between_records_both_bounds() -> None:
    assert F.col("order_items.created_at").between(date(2026, 1, 1), date(2026, 7, 1)).expr == (
        Between(FieldRef("order_items.created_at"), date(2026, 1, 1), date(2026, 7, 1))
    )


# --------------------------------------------------------------------------------------
# Aliases, grains, sorting
# --------------------------------------------------------------------------------------


def test_alias_is_recorded_without_touching_the_expression() -> None:
    column = F.col("users.state").alias("state")

    assert column.alias_name == "state"
    assert column.expr == FieldRef("users.state")


def test_alias_needs_a_name() -> None:
    with pytest.raises(CompileError, match="non-empty"):
        F.col("users.state").alias("")


def test_grain_is_case_insensitive_and_canonicalized() -> None:
    assert F.col("order_items.created_at").grain("MONTH").expr == FieldRef(
        "order_items.created_at", "month"
    )


def test_grain_validates_against_the_wire_grain_list() -> None:
    with pytest.raises(CompileError, match="missing_fields"):
        F.col("order_items.created_at").grain("fortnight")


def test_grain_only_applies_to_a_bare_field() -> None:
    with pytest.raises(CompileError, match="bare field reference"):
        F.measure("order_items.count").grain("month")
    with pytest.raises(CompileError, match="at most one grain"):
        F.col("order_items.created_at").grain("month").grain("year")


def test_alias_survives_grain_and_direction() -> None:
    column = F.col("order_items.created_at").alias("month").grain("month").desc()

    assert column.alias_name == "month"
    assert column.descending is True
    assert column.to_sort_key() == SortKey(FieldRef("order_items.created_at", "month"), True)


def test_asc_and_desc_return_new_columns() -> None:
    column = F.col("users.state")

    assert column.descending is None
    assert column.asc().descending is False
    assert column.desc().descending is True
    assert column.descending is None, "the original column is untouched"


def test_repr_shows_the_alias_and_direction() -> None:
    text = repr(F.col("users.state").alias("state").desc())

    assert "AS state" in text
    assert "DESC" in text


def test_repr_reads_like_the_predicate_not_like_its_dataclass_tree() -> None:
    """Building predicates in a REPL is the documented workflow; a nested dump is unusable."""
    assert repr(F.col("users.state") == "California") == "Column(users.state = 'California')"

    composite = ((F.col("users.state") == "CA") | (F.col("users.state") == "NY")) & ~F.col(
        "users.email"
    ).is_null()
    text = repr(composite)

    assert "Comparison(" not in text
    assert "FieldRef(" not in text
    assert text == (
        "Column(((users.state = 'CA' OR users.state = 'NY') AND NOT (users.email IS NULL)))"
    )
