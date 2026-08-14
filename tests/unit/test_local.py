"""Unit tests for the local engine (docs/HYBRID.md §3).

Every operator here is a pure function of Arrow tables, so these tests build tables literally
and never touch a session.  What they are really pinning is **SQL parity**: three-valued logic,
NULL group keys, ``count_distinct`` ignoring NULL, the dtype promotion table, nulls-last
ordering.  When one of these drifts, a differential test fails somewhere far away and takes an
afternoon to localize — so they are asserted here, one behavior per test.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pa_parquet
import pytest

from omniframes import functions as F
from omniframes.column import AggFn, Column, Expr, MeasureRef
from omniframes.compile.local import (
    AlignJoin,
    LocalAgg,
    LocalAggregate,
    LocalFilter,
    LocalJoin,
    LocalLimit,
    LocalMapPandas,
    LocalProject,
    LocalSort,
    LocalUnion,
    LocalWithColumn,
    describe_expr,
    eval_expr,
    promote_type,
    run_local_op,
)
from omniframes.compile.semantic import CannotCompile
from omniframes.errors import CompileError
from omniframes.plan.nodes import JoinHow


def table(**columns: Any) -> pa.Table:
    return pa.table(columns)


#: One row per truth value, so a predicate over ``flag`` is the Kleene truth table.
TRUTH = pa.table(
    {
        "flag": pa.array([True, False, None], type=pa.bool_()),
        "other": pa.array([True, True, True], type=pa.bool_()),
    }
)

#: A frame with a NULL in every interesting position: group key, operand, and both at once.
NULLY = pa.table(
    {
        "state": pa.array(["CA", "CA", None, "TX", None], type=pa.string()),
        "amount": pa.array(
            [Decimal("1.50"), None, Decimal("2.25"), Decimal("4.00"), None],
            type=pa.decimal128(12, 2),
        ),
        "qty": pa.array([1, 2, None, 2, 3], type=pa.int64()),
        "label": pa.array(["a", "b", None, "b", "c"], type=pa.string()),
    }
)


def predicate(column: Column) -> Expr:
    return column.expr


# --------------------------------------------------------------------------------------
# eval_expr — three-valued logic (§3.1)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: F.col("flag") & F.col("other"), [True, False, None]),
        (lambda: F.col("flag") | F.col("other"), [True, True, True]),
        (lambda: ~F.col("flag"), [False, True, None]),
        (lambda: F.col("flag").is_null(), [False, False, True]),
        (lambda: F.col("flag").is_not_null(), [True, True, False]),
    ],
)
def test_the_kleene_truth_table(build: Any, expected: list[bool | None]) -> None:
    """AND/OR/NOT are three-valued, exactly as SQL defines them."""
    assert eval_expr(predicate(build()), TRUTH).to_pylist() == expected


def test_a_null_predicate_drops_the_row() -> None:
    """``NOT NULL`` is NULL, and a WHERE keeps neither the match nor the unknowns."""
    kept = LocalFilter(predicate(~(F.col("label") == "a"))).run((NULLY,))

    assert kept.column("label").to_pylist() == ["b", "b", "c"], "no 'a', and no NULL either"


def test_comparisons_propagate_null() -> None:
    values = table(x=pa.array([1, None, 5], type=pa.int64()))

    assert eval_expr(predicate(F.col("x") > 2), values).to_pylist() == [False, None, True]


def test_is_in_never_matches_null() -> None:
    kept = LocalFilter(predicate(F.col("label").isin("a", "c"))).run((NULLY,))

    assert kept.column("label").to_pylist() == ["a", "c"]


def test_is_in_is_three_valued_so_negating_it_does_not_resurrect_nulls() -> None:
    """``x IN (...)`` is NULL for a NULL ``x``, so ``NOT`` leaves it NULL and the row is dropped.

    Arrow's ``is_in`` answers set membership and returns FALSE for a NULL operand, which reads
    the same as a genuine non-match until the predicate is negated — the point at which the
    local engine would keep rows tiers 1 and 2 drop (docs/HYBRID.md §3.1).
    """
    assert eval_expr(predicate(F.col("label").isin("a", "c")), NULLY).to_pylist() == [
        True,
        False,
        None,
        False,
        True,
    ]
    kept = LocalFilter(predicate(~F.col("label").isin("a", "c"))).run((NULLY,))

    assert kept.column("label").to_pylist() == ["b", "b"], "no 'a'/'c', and no NULL either"


def test_string_predicates_are_case_sensitive_unless_asked() -> None:
    """Matching the wire, where ``case_insensitive`` is opt-in (CONTRACT_NOTES §3.1)."""
    names = table(name=pa.array(["Cal", "cal", None]))
    sensitive = LocalFilter(predicate(F.col("name").contains("Cal"))).run((names,))
    insensitive = LocalFilter(predicate(F.col("name").contains("Cal", case_insensitive=True))).run(
        (names,)
    )

    assert sensitive.column("name").to_pylist() == ["Cal"]
    assert insensitive.column("name").to_pylist() == ["Cal", "cal"]


def test_like_and_the_anchored_predicates() -> None:
    names = table(name=pa.array(["alpha", "beta", "alpaca"]))

    def kept(column: Column) -> list[Any]:
        return LocalFilter(predicate(column)).run((names,)).column("name").to_pylist()

    assert kept(F.col("name").starts_with("alp")) == ["alpha", "alpaca"]
    assert kept(F.col("name").ends_with("ta")) == ["beta"]
    assert kept(F.col("name").like("al%a")) == ["alpha", "alpaca"]


def test_between_is_inclusive_over_numbers_and_half_open_over_dates() -> None:
    """The same split :meth:`omniframes.column.Column.between` compiles to (docs/HYBRID.md §3.1)."""
    numbers = table(x=pa.array([2, 3, 5, 6], type=pa.int64()))
    dates = table(
        d=pa.array([date(2025, 1, 1), date(2025, 6, 1), date(2026, 1, 1)], type=pa.date32())
    )

    kept = LocalFilter(predicate(F.col("x").between(3, 5))).run((numbers,))
    within = LocalFilter(predicate(F.col("d").between(date(2025, 1, 1), date(2026, 1, 1)))).run(
        (dates,)
    )

    assert kept.column("x").to_pylist() == [3, 5], "both ends included"
    assert within.column("d").to_pylist() == [date(2025, 1, 1), date(2025, 6, 1)], "upper excluded"


def test_a_relative_date_literal_is_refused_locally() -> None:
    """Omni's date grammar is evaluated by Omni; the local engine will not guess at it."""
    stamps = table(d=pa.array([datetime(2026, 1, 1, tzinfo=UTC)]))

    with pytest.raises(CannotCompile, match="relative date literals are evaluated by Omni"):
        LocalFilter(predicate(F.col("d") >= "30 days ago")).run((stamps,))


def test_arithmetic_is_decimal_aware_and_promotes_integer_division() -> None:
    values = table(
        price=pa.array([Decimal("2.50")], type=pa.decimal128(12, 2)),
        n=pa.array([7], type=pa.int64()),
    )
    doubled = eval_expr(predicate(F.col("price") * 2), values)
    divided = eval_expr(predicate(F.col("n") / 2), values)

    assert doubled.to_pylist() == [Decimal("5.00")]
    assert divided.type == pa.float64()
    assert divided.to_pylist() == [3.5], "integer division is not floor division here"


def test_a_measure_reference_is_a_column_reference_or_an_error() -> None:
    computed = table(**{"order_items.total_sale_price": pa.array([Decimal("1.00")])})

    assert eval_expr(MeasureRef("order_items.total_sale_price"), computed).to_pylist() == [
        Decimal("1.00")
    ]
    with pytest.raises(CompileError, match="governed measures only exist remotely"):
        eval_expr(MeasureRef("order_items.total_sale_price"), NULLY)


def test_an_unknown_column_names_itself_and_what_is_available() -> None:
    with pytest.raises(CompileError, match=r"users\.nope.*state, amount"):
        eval_expr(predicate(F.col("users.nope") == 1), NULLY)


# --------------------------------------------------------------------------------------
# UDFs (§5)
# --------------------------------------------------------------------------------------


def test_a_udf_is_scalar_and_sees_none_for_a_null() -> None:
    seen: list[object] = []

    def tag(value: object) -> str:
        seen.append(value)
        return "missing" if value is None else str(value)

    result = LocalWithColumn("tag", predicate(F.udf(tag)("label"))).run((NULLY,))

    assert result.column("tag").to_pylist() == ["a", "b", "missing", "b", "c"]
    assert None in seen, "a UDF is plain Python, so a missing value arrives as None"


def test_a_udf_takes_several_operands_in_order() -> None:
    joined = F.udf(lambda state, qty: f"{state}:{qty}")("state", "qty")
    result = LocalWithColumn("j", predicate(joined)).run((NULLY,))

    assert result.column("j").to_pylist()[:2] == ["CA:1", "CA:2"]


def test_a_udf_names_itself_for_explain() -> None:
    def shout(value: str) -> str:
        return value.upper()

    assert describe_expr(predicate(F.udf(shout)("users.state"))) == "shout(users.state)"


def test_a_udf_needs_a_column() -> None:
    with pytest.raises(CompileError, match="at least one column"):
        F.udf(str.upper)()


def test_udf_takes_a_callable() -> None:
    with pytest.raises(CompileError, match="takes a callable"):
        F.udf("not callable")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# LocalAggregate — NULL keys and null-skipping aggregates (§3.2)
# --------------------------------------------------------------------------------------


def test_a_null_group_key_is_a_group() -> None:
    """SQL's ``GROUP BY`` keeps NULL as a value; pandas would drop it without ``dropna=False``."""
    grouped = LocalAggregate(("state",), (LocalAgg("n", AggFn.COUNT, "qty"),)).run((NULLY,))
    by_state = {row["state"]: row["n"] for row in grouped.to_pylist()}

    assert set(by_state) == {"CA", "TX", None}
    assert by_state[None] == 1, "one NULL-key group, counting only its non-null qty"


def test_aggregates_skip_nulls_and_return_null_for_an_all_null_group() -> None:
    rows = table(
        k=pa.array(["a", "a", "b"]),
        v=pa.array([None, None, 3], type=pa.int64()),
    )
    grouped = LocalAggregate(
        ("k",),
        (
            LocalAgg("total", AggFn.SUM, "v"),
            LocalAgg("smallest", AggFn.MIN, "v"),
            LocalAgg("mean", AggFn.AVG, "v"),
            LocalAgg("n", AggFn.COUNT, "v"),
        ),
    ).run((rows,))
    by_key = {row["k"]: row for row in grouped.to_pylist()}

    assert by_key["a"]["total"] is None
    assert by_key["a"]["smallest"] is None
    assert by_key["a"]["mean"] is None
    assert by_key["a"]["n"] == 0, "count(col) counts non-null values, so an all-null group is 0"
    assert by_key["b"]["total"] == 3


def test_count_distinct_excludes_null() -> None:
    rows = table(k=pa.array(["a"] * 4), v=pa.array([1, 1, None, 2], type=pa.int64()))
    grouped = LocalAggregate(("k",), (LocalAgg("d", AggFn.COUNT, "v", distinct=True),)).run((rows,))

    assert grouped.column("d").to_pylist() == [2]


def test_two_aggregates_over_the_same_column_do_not_collide() -> None:
    grouped = LocalAggregate(
        ("state",),
        (LocalAgg("a", AggFn.SUM, "qty"), LocalAgg("b", AggFn.SUM, "qty")),
    ).run((NULLY,))

    assert grouped.column("a").to_pylist() == grouped.column("b").to_pylist()


def test_a_group_less_aggregate_produces_one_row() -> None:
    grouped = LocalAggregate((), (LocalAgg("total", AggFn.SUM, "qty"),)).run((NULLY,))

    assert grouped.to_pylist() == [{"total": 8}]


@pytest.mark.parametrize(
    ("fn", "distinct", "column", "expected"),
    [
        (AggFn.SUM, False, "i", pa.int64()),
        (AggFn.AVG, False, "i", pa.float64()),
        (AggFn.MIN, False, "i", pa.int64()),
        (AggFn.COUNT, False, "i", pa.int64()),
        (AggFn.COUNT, True, "i", pa.int64()),
        (AggFn.SUM, False, "f", pa.float64()),
        (AggFn.AVG, False, "f", pa.float64()),
        (AggFn.MAX, False, "f", pa.float64()),
        (AggFn.COUNT, False, "f", pa.int64()),
        (AggFn.SUM, False, "d", pa.decimal128(38, 2)),
        (AggFn.AVG, False, "d", pa.float64()),
        (AggFn.MIN, False, "d", pa.decimal128(12, 2)),
        (AggFn.MAX, False, "d", pa.decimal128(12, 2)),
        (AggFn.COUNT, False, "d", pa.int64()),
        (AggFn.MIN, False, "ts", pa.timestamp("us", tz="UTC")),
        (AggFn.COUNT, False, "ts", pa.int64()),
        (AggFn.MIN, False, "s", pa.string()),
        (AggFn.MAX, False, "s", pa.string()),
        (AggFn.COUNT, False, "s", pa.int64()),
        (AggFn.COUNT, False, "b", pa.int64()),
    ],
)
def test_the_dtype_promotion_table(
    fn: AggFn, distinct: bool, column: str, expected: pa.DataType
) -> None:
    """Every allowed cell of docs/HYBRID.md §3.3 — the comparator uses the same table."""
    rows = table(
        k=pa.array(["a"]),
        i=pa.array([1], type=pa.int64()),
        f=pa.array([1.5], type=pa.float64()),
        d=pa.array([Decimal("1.50")], type=pa.decimal128(12, 2)),
        ts=pa.array([datetime(2026, 1, 1, tzinfo=UTC)], type=pa.timestamp("us", tz="UTC")),
        s=pa.array(["x"]),
        b=pa.array([True]),
    )
    grouped = LocalAggregate(("k",), (LocalAgg("out", fn, column, distinct),)).run((rows,))

    assert grouped.schema.field("out").type == expected


def test_a_decimal_sum_widens_before_it_can_overflow() -> None:
    """The promotion to ``decimal128(38, s)`` is enforced here, not left to Arrow's version.

    Arrow only widens a grouped decimal sum from pyarrow 21; the declared floor is 15, where the
    result keeps the operand's precision — and at precision <= 18 parquet stores it as an INT64
    that wraps silently on the way back.
    """
    big = Decimal("9999999999999999.99")
    rows = table(
        k=pa.array(["a"] * 20),
        d=pa.array([big] * 20, type=pa.decimal128(18, 2)),
    )
    grouped = LocalAggregate(("k",), (LocalAgg("total", AggFn.SUM, "d"),)).run((rows,))

    assert grouped.schema.field("total").type == pa.decimal128(38, 2)
    assert grouped.column("total").to_pylist() == [big * 20]

    buffer = io.BytesIO()
    pa_parquet.write_table(grouped, buffer)
    buffer.seek(0)
    assert pa_parquet.read_table(buffer).column("total").to_pylist() == [big * 20], (
        "…and the widened type is what makes the parquet round trip exact"
    )


@pytest.mark.parametrize(
    ("fn", "column", "message"),
    [
        (AggFn.SUM, "ts", "needs a numeric column"),
        (AggFn.AVG, "ts", "needs a numeric column"),
        (AggFn.SUM, "s", "needs a numeric column"),
        (AggFn.AVG, "s", "needs a numeric column"),
        (AggFn.SUM, "b", "needs a numeric column"),
        (AggFn.MIN, "b", "no meaning over the boolean column"),
        (AggFn.MAX, "b", "no meaning over the boolean column"),
    ],
)
def test_the_forbidden_cells_of_the_promotion_table(fn: AggFn, column: str, message: str) -> None:
    rows = table(
        k=pa.array(["a"]),
        ts=pa.array([datetime(2026, 1, 1, tzinfo=UTC)], type=pa.timestamp("us", tz="UTC")),
        s=pa.array(["x"]),
        b=pa.array([True]),
    )

    with pytest.raises(CompileError, match=message):
        LocalAggregate(("k",), (LocalAgg("out", fn, column),)).run((rows,))


def test_an_average_over_decimals_is_a_float_not_a_rounded_decimal() -> None:
    """Arrow's decimal mean rounds to the operand scale; that is a wrong answer, not a type."""
    rows = table(
        k=pa.array(["a", "a", "a"]),
        d=pa.array([Decimal("1.00"), Decimal("1.00"), Decimal("2.00")], type=pa.decimal128(12, 2)),
    )
    grouped = LocalAggregate(("k",), (LocalAgg("avg", AggFn.AVG, "d"),)).run((rows,))

    assert grouped.column("avg").to_pylist() == [pytest.approx(4 / 3)]


def test_an_aggregate_over_a_missing_column_says_so() -> None:
    with pytest.raises(CompileError, match="'nope' is not a column"):
        LocalAggregate(("state",), (LocalAgg("x", AggFn.SUM, "nope"),)).run((NULLY,))


# --------------------------------------------------------------------------------------
# AlignJoin (§3.2)
# --------------------------------------------------------------------------------------


def test_align_join_matches_a_null_key_to_a_null_key() -> None:
    """The opposite of SQL join semantics — and the whole reason this operator exists."""
    measures = table(state=pa.array(["CA", None]), revenue=pa.array([10, 20]))
    aggregated = table(state=pa.array([None, "CA"]), buyers=pa.array([3, 7]))

    joined = AlignJoin(("state",)).run((measures, aggregated))

    assert joined.column_names == ["state", "revenue", "buyers"]
    assert joined.to_pylist() == [
        {"state": "CA", "revenue": 10, "buyers": 7},
        {"state": None, "revenue": 20, "buyers": 3},
    ]


def test_align_join_keeps_decimal_and_timestamp_types() -> None:
    measures = table(
        k=pa.array(["a"]), revenue=pa.array([Decimal("1.50")], type=pa.decimal128(12, 2))
    )
    aggregated = table(k=pa.array(["a"]), first=pa.array([datetime(2026, 1, 1, tzinfo=UTC)]))
    joined = AlignJoin(("k",)).run((measures, aggregated))

    assert joined.schema.field("revenue").type == pa.decimal128(12, 2)
    assert joined.to_pylist()[0]["revenue"] == Decimal("1.50")


def test_align_join_without_keys_pairs_the_two_single_rows() -> None:
    joined = AlignJoin(()).run((table(revenue=pa.array([5])), table(buyers=pa.array([2]))))

    assert joined.to_pylist() == [{"revenue": 5, "buyers": 2}]


def test_align_join_takes_exactly_two_inputs() -> None:
    with pytest.raises(CompileError, match="exactly two inputs"):
        AlignJoin(("k",)).run((NULLY,))


# --------------------------------------------------------------------------------------
# LocalJoin — the user-facing join, with SQL null semantics (M4)
# --------------------------------------------------------------------------------------

#: Two sides that share a key, each with a NULL-keyed row and a row the other cannot match.
JOIN_LEFT = pa.table(
    {
        "state": pa.array(["CA", "TX", None, "OR"], type=pa.string()),
        "revenue": pa.array([10, 20, 30, 40], type=pa.int64()),
    }
)
JOIN_RIGHT = pa.table(
    {
        "state": pa.array(["CA", None, "WA"], type=pa.string()),
        "buyers": pa.array([1, 2, 3], type=pa.int64()),
    }
)


def joined(how: JoinHow) -> list[dict[str, Any]]:
    return LocalJoin(("state",), how).run((JOIN_LEFT, JOIN_RIGHT)).to_pylist()


def test_an_inner_join_drops_every_null_key_and_every_orphan() -> None:
    """SQL's rule: NULL = NULL is UNKNOWN, so neither NULL-keyed row matches the other."""
    assert joined(JoinHow.INNER) == [{"state": "CA", "revenue": 10, "buyers": 1}]


def test_a_left_join_keeps_the_left_null_key_row_with_the_right_columns_null() -> None:
    assert joined(JoinHow.LEFT) == [
        {"state": "CA", "revenue": 10, "buyers": 1},
        {"state": "TX", "revenue": 20, "buyers": None},
        {"state": "OR", "revenue": 40, "buyers": None},
        {"state": None, "revenue": 30, "buyers": None},
    ]


def test_a_right_join_keeps_the_right_null_key_row_instead() -> None:
    assert joined(JoinHow.RIGHT) == [
        {"state": "CA", "revenue": 10, "buyers": 1},
        {"state": "WA", "revenue": None, "buyers": 3},
        {"state": None, "revenue": None, "buyers": 2},
    ]


def test_an_outer_join_keeps_both_null_key_rows_separately() -> None:
    """Two NULL keys, two output rows — an align-join would have produced one."""
    rows = joined(JoinHow.OUTER)

    assert len(rows) == 6, "one match, three unmatched, and the two NULL-keyed rows"
    assert rows.count({"state": None, "revenue": 30, "buyers": None}) == 1
    assert rows.count({"state": None, "revenue": None, "buyers": 2}) == 1
    assert {"state": "CA", "revenue": 10, "buyers": 1} in rows


def test_the_join_output_is_the_keys_then_the_left_columns_then_the_right_ones() -> None:
    result = LocalJoin(("state",), JoinHow.INNER).run((JOIN_LEFT, JOIN_RIGHT))

    assert result.column_names == ["state", "revenue", "buyers"]
    assert result.schema.field("revenue").type == pa.int64()


def test_a_join_on_several_keys_needs_every_key_to_be_non_null() -> None:
    left = table(a=pa.array(["x", "x", None]), b=pa.array([1, None, 1]), v=pa.array([10, 20, 30]))
    right = table(a=pa.array(["x", None]), b=pa.array([1, 1]), w=pa.array([1, 2]))

    result = LocalJoin(("a", "b"), JoinHow.INNER).run((left, right))

    assert result.to_pylist() == [{"a": "x", "b": 1, "v": 10, "w": 1}]


def test_a_join_widens_key_types_that_differ() -> None:
    left = table(k=pa.array([1, 2], type=pa.int32()), v=pa.array(["a", "b"]))
    right = table(k=pa.array([1.0, 3.0], type=pa.float64()), w=pa.array(["x", "y"]))

    result = LocalJoin(("k",), JoinHow.INNER).run((left, right))

    assert result.schema.field("k").type == pa.float64()
    assert result.to_pylist() == [{"k": 1.0, "v": "a", "w": "x"}]


def test_a_join_that_matches_nothing_still_has_the_right_shape() -> None:
    left = table(k=pa.array(["a"]), v=pa.array([1]))
    right = table(k=pa.array(["b"]), w=pa.array([Decimal("1.50")], type=pa.decimal128(12, 2)))

    result = LocalJoin(("k",), JoinHow.INNER).run((left, right))

    assert result.num_rows == 0
    assert result.column_names == ["k", "v", "w"]
    assert result.schema.field("w").type == pa.decimal128(12, 2)


def test_overlapping_non_key_columns_are_refused_rather_than_suffixed() -> None:
    left = table(k=pa.array(["a"]), v=pa.array([1]))
    right = table(k=pa.array(["a"]), v=pa.array([2]))

    with pytest.raises(CompileError, match="will not guess which one you meant"):
        LocalJoin(("k",), JoinHow.INNER).run((left, right))


def test_a_join_key_missing_from_one_side_names_the_side() -> None:
    with pytest.raises(CompileError, match="not a column of the right frame"):
        LocalJoin(("state",), JoinHow.INNER).run((JOIN_LEFT, table(other=pa.array(["a"]))))


def test_a_join_takes_exactly_two_inputs_and_at_least_one_key() -> None:
    with pytest.raises(CompileError, match="exactly two inputs"):
        LocalJoin(("state",), JoinHow.INNER).run((JOIN_LEFT,))
    with pytest.raises(CompileError, match="at least one key"):
        LocalJoin((), JoinHow.INNER).run((JOIN_LEFT, JOIN_RIGHT))


# --------------------------------------------------------------------------------------
# LocalUnion and the promotion table (§3.3)
# --------------------------------------------------------------------------------------


def test_a_union_stacks_rows_and_keeps_duplicates() -> None:
    left = table(k=pa.array(["a", "b"]), v=pa.array([1, 2]))
    right = table(k=pa.array(["a"]), v=pa.array([1]))

    result = LocalUnion().run((left, right))

    assert result.to_pylist() == [
        {"k": "a", "v": 1},
        {"k": "b", "v": 2},
        {"k": "a", "v": 1},
    ]


def test_a_union_promotes_an_integer_column_next_to_a_float_one() -> None:
    left = table(k=pa.array(["a"]), v=pa.array([1], type=pa.int64()))
    right = table(k=pa.array(["b"]), v=pa.array([2.5], type=pa.float64()))

    result = LocalUnion().run((left, right))

    assert result.schema.field("v").type == pa.float64()
    assert result.to_pylist() == [{"k": "a", "v": 1.0}, {"k": "b", "v": 2.5}]


def test_a_union_refuses_columns_that_do_not_line_up_by_name() -> None:
    left = table(k=pa.array(["a"]), v=pa.array([1]))
    right = table(k=pa.array(["a"]), w=pa.array([1]))

    with pytest.raises(CompileError, match="same columns in the same order"):
        LocalUnion().run((left, right))


def test_a_union_takes_exactly_two_inputs() -> None:
    with pytest.raises(CompileError, match="exactly two inputs"):
        LocalUnion().run((NULLY,))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (pa.int32(), pa.int64(), pa.int64()),
        (pa.int64(), pa.float64(), pa.float64()),
        (pa.float64(), pa.decimal128(12, 2), pa.float64()),
        (pa.decimal128(12, 2), pa.decimal128(10, 4), pa.decimal128(38, 4)),
        (pa.decimal128(12, 2), pa.int64(), pa.decimal128(38, 2)),
        (pa.string(), pa.large_string(), pa.large_string()),
        (pa.null(), pa.int64(), pa.int64()),
        (pa.string(), pa.string(), pa.string()),
    ],
)
def test_the_promotion_table(left: Any, right: Any, expected: Any) -> None:
    assert promote_type(left, right, column="v") == expected
    assert promote_type(right, left, column="v") == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (pa.string(), pa.int64()),
        (pa.bool_(), pa.int64()),
        (pa.timestamp("us"), pa.string()),
    ],
)
def test_types_with_no_common_type_name_the_column(left: Any, right: Any) -> None:
    with pytest.raises(CompileError, match="no common type"):
        promote_type(left, right, column="v")


# --------------------------------------------------------------------------------------
# Sort / limit / project / map_pandas
# --------------------------------------------------------------------------------------


def test_sort_puts_nulls_last_in_both_directions() -> None:
    rows = table(v=pa.array([3, None, 1], type=pa.int64()))

    ascending = LocalSort((("v", False),)).run((rows,)).column("v").to_pylist()
    descending = LocalSort((("v", True),)).run((rows,)).column("v").to_pylist()

    assert ascending == [1, 3, None]
    assert descending == [3, 1, None]


def test_sort_is_stable_across_several_keys() -> None:
    rows = table(a=pa.array(["x", "x", "y"]), b=pa.array([2, 1, 0]))
    sorted_rows = LocalSort((("a", False), ("b", False))).run((rows,))

    assert sorted_rows.to_pylist() == [
        {"a": "x", "b": 1},
        {"a": "x", "b": 2},
        {"a": "y", "b": 0},
    ]


def test_sorting_by_a_column_that_is_not_there_says_so() -> None:
    with pytest.raises(CompileError, match="cannot sort by nope"):
        LocalSort((("nope", False),)).run((NULLY,))


def test_limit_and_offset_slice_like_sql() -> None:
    rows = table(v=pa.array([1, 2, 3, 4, 5]))

    assert LocalLimit(2, 1).run((rows,)).column("v").to_pylist() == [2, 3]
    assert LocalLimit(None, 3).run((rows,)).column("v").to_pylist() == [4, 5]
    assert LocalLimit(None, 0).run((rows,)).column("v").to_pylist() == [1, 2, 3, 4, 5]


def test_project_selects_in_order_and_renames() -> None:
    projected = LocalProject(("qty", "state"), {"qty": "n"}).run((NULLY,))

    assert projected.column_names == ["n", "state"]


def test_projecting_a_missing_column_names_it() -> None:
    with pytest.raises(CompileError, match="nope is not available"):
        LocalProject(("nope",)).run((NULLY,))


def test_map_pandas_round_trips_a_frame() -> None:
    def rename(frame: Any) -> Any:
        return frame.rename(columns={"state": "region"})

    result = run_local_op(LocalMapPandas(rename), (NULLY,))

    assert result.column_names == ["region", "amount", "qty", "label"]
    assert result.num_rows == NULLY.num_rows


def test_map_pandas_must_return_a_frame() -> None:
    with pytest.raises(CompileError, match="must return a pandas DataFrame"):
        run_local_op(LocalMapPandas(lambda frame: frame.shape), (NULLY,))


def test_every_operator_names_itself_for_explain() -> None:
    """``explain()`` reads these; a step with no line is a step nobody can audit."""
    ops = [
        LocalFilter(predicate(F.col("state") == "CA")),
        LocalProject(("state",)),
        LocalWithColumn("d", predicate(F.col("qty") * 2)),
        LocalAggregate(("state",), (LocalAgg("n", AggFn.COUNT, "qty"),)),
        LocalSort((("state", True),)),
        LocalLimit(5, 2),
        LocalMapPandas(lambda frame: frame),
        AlignJoin(("state",)),
        LocalJoin(("state",), JoinHow.LEFT),
        LocalUnion(),
    ]

    for op in ops:
        assert op.describe(["step 1", "step 2"]).strip(), f"{type(op).__name__} renders nothing"
    assert LocalFilter(predicate(F.col("state") == "CA")).describe(["step 1"]) == (
        "filter over step 1: state = 'CA'"
    )
    assert LocalSort((("n", True),)).describe(["filter"]) == "sort: n desc"
    assert LocalJoin(("state",), JoinHow.LEFT).describe(["step 1", "step 2"]) == (
        "join [left]: step 1 ⨝ step 2 on [state]"
    )
    assert LocalUnion().describe(["step 1", "step 2"]) == "union: step 1 ⊎ step 2 (by position)"
