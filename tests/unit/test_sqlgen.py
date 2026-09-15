"""Unit tests for tier 2 (docs/SQLTIER.md §3).

Three halves, really.  The first is the translation table, row by row: one expression in, one
OmniSQL fragment out, rendered through the same code path the compiler uses.  The second is the
compiler itself — what shape it accepts, what it refuses, how the two naming regimes of §3.2 are
emitted, and where each predicate ends up.  The third is the safety net: values never become
statement text, and neither do model names the sentinel charset does not recognize.

Everything here is pure — nothing talks to anything.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest
import sqlglot
from tests.fakes import BENCH_MODEL_ID, BENCH_MODEL_NAME, BENCH_TOPIC_NAME

from omniframes import functions as F
from omniframes.column import Column, Expr
from omniframes.compile.querymodel import DEFAULT_FETCH_LIMIT
from omniframes.compile.semantic import CannotCompile
from omniframes.compile.sqlgen import EXPR_ALIAS_PREFIX, compile_sql, render_expr, try_sql
from omniframes.plan import nodes
from omniframes.types import OmniDataType, OmniField, OmniSchema

REVENUE = "order_items.total_sale_price"
STATE = "users.state"
AGE = "users.age"
PRICE = "order_items.sale_price"
CREATED = "order_items.created_at"

SCAN = nodes.Scan(
    nodes.TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)
VIEW_SCAN = nodes.Scan(nodes.ViewScan(BENCH_MODEL_NAME, BENCH_MODEL_ID, "order_items"))
SQL_SCAN = nodes.Scan(nodes.SqlScan(BENCH_MODEL_ID, "SELECT 1 AS x", BENCH_MODEL_NAME))
STORED_SCAN = nodes.Scan(
    nodes.SavedQueryScan("doc", "saved", {"modelId": BENCH_MODEL_ID, "fields": [STATE]})
)


def columns(*names: str | Column) -> tuple[Column, ...]:
    return tuple(F.col(name) if isinstance(name, str) else name for name in names)


def selected(*names: str | Column) -> nodes.Project:
    return nodes.Project(SCAN, columns(*names))


def sql(plan: nodes.PlanNode) -> str:
    return compile_sql(plan).query.user_edited_sql


def fragment(expr: Expr) -> str:
    return render_expr(expr)


def aggregate(*aggs: Column, keys: tuple[Column, ...] = ()) -> nodes.Aggregate:
    return nodes.Aggregate(SCAN, keys, aggs)


# --------------------------------------------------------------------------------------
# The translation table (docs/SQLTIER.md §3.1)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (F.col(STATE).expr, "${users.state}"),
        (F.col(CREATED).grain("month").expr, "${order_items.created_at[month]}"),
        (F.measure(REVENUE).expr, "${order_items.total_sale_price}"),
        (F.lit("California").expr, "'California'"),
        (F.lit(42).expr, "42"),
        (F.lit(3.5).expr, "3.5"),
        (F.lit(Decimal("99.50")).expr, "99.50"),
        (F.lit(True).expr, "TRUE"),
        (F.lit(False).expr, "FALSE"),
        (F.lit(date(2026, 3, 1)).expr, "CAST('2026-03-01' AS DATE)"),
        (F.lit(datetime(2026, 3, 1, 12, 30)).expr, "CAST('2026-03-01 12:30:00' AS TIMESTAMP)"),
        ((F.col(AGE) == 30).expr, "${users.age} = 30"),
        ((F.col(AGE) != 30).expr, "${users.age} <> 30"),
        ((F.col(AGE) < 30).expr, "${users.age} < 30"),
        ((F.col(AGE) <= 30).expr, "${users.age} <= 30"),
        ((F.col(AGE) > 30).expr, "${users.age} > 30"),
        ((F.col(AGE) >= 30).expr, "${users.age} >= 30"),
        ((F.col(AGE) + 1).expr, "${users.age} + 1"),
        ((F.col(AGE) - 1).expr, "${users.age} - 1"),
        ((F.col(AGE) * 2).expr, "${users.age} * 2"),
        ((F.col(AGE) / 2).expr, "${users.age} / 2"),
        # Nested arithmetic keeps its grouping.  SQLGlot's generator prints the tree it is handed
        # and never re-derives precedence, so the parens have to be nodes: without them
        # ``(a - b) / a`` prints as ``a - b / a``, which the warehouse answers — wrongly.
        (
            ((F.col(PRICE) - F.col(AGE)) / F.col(PRICE)).expr,
            "(${order_items.sale_price} - ${users.age}) / ${order_items.sale_price}",
        ),
        (
            ((F.col(AGE) + 1) * (F.col(AGE) - 1)).expr,
            "(${users.age} + 1) * (${users.age} - 1)",
        ),
        ((F.col(AGE) - (F.col(AGE) - 1)).expr, "${users.age} - (${users.age} - 1)"),
        (
            ((F.measure(REVENUE) - F.col(PRICE)) / F.count_distinct("users.id")).expr,
            "(${order_items.total_sale_price} - ${order_items.sale_price})"
            " / COUNT(DISTINCT ${users.id})",
        ),
        (((F.col(AGE) + 1) * 2 > 30).expr, "(${users.age} + 1) * 2 > 30"),
        (
            ((F.col(AGE) > 30) & (F.col(STATE) == "Ohio")).expr,
            "(${users.age} > 30 AND ${users.state} = 'Ohio')",
        ),
        (
            ((F.col(AGE) > 30) | (F.col(STATE) == "Ohio")).expr,
            "(${users.age} > 30 OR ${users.state} = 'Ohio')",
        ),
        ((~(F.col(AGE) > 30)).expr, "NOT (${users.age} > 30)"),
        (F.col(STATE).is_null().expr, "${users.state} IS NULL"),
        (F.col(STATE).is_not_null().expr, "NOT (${users.state} IS NULL)"),
        (F.col(STATE).isin("Ohio", "Texas").expr, "${users.state} IN ('Ohio', 'Texas')"),
        (F.col(STATE).contains("cal").expr, "${users.state} LIKE '%cal%' ESCAPE '!'"),
        (F.col(STATE).starts_with("New").expr, "${users.state} LIKE 'New%' ESCAPE '!'"),
        (F.col(STATE).ends_with("ia").expr, "${users.state} LIKE '%ia' ESCAPE '!'"),
        (F.col(STATE).like("%a_b%").expr, "${users.state} LIKE '%a_b%'"),
        (
            F.col(STATE).contains("cal", case_insensitive=True).expr,
            "LOWER(${users.state}) LIKE LOWER('%cal%') ESCAPE '!'",
        ),
        ((F.col(AGE).between(18, 21)).expr, "(${users.age} >= 18 AND ${users.age} <= 21)"),
        (
            (F.col(CREATED).between(date(2026, 1, 1), date(2026, 2, 1))).expr,
            "(${order_items.created_at} >= CAST('2026-01-01' AS DATE)"
            " AND ${order_items.created_at} < CAST('2026-02-01' AS DATE))",
        ),
        (F.sum(PRICE).expr, "SUM(${order_items.sale_price})"),
        (F.count(PRICE).expr, "COUNT(${order_items.sale_price})"),
        (F.count_distinct(PRICE).expr, "COUNT(DISTINCT ${order_items.sale_price})"),
        (F.avg(PRICE).expr, "AVG(${order_items.sale_price})"),
        (F.min(PRICE).expr, "MIN(${order_items.sale_price})"),
        (F.max(PRICE).expr, "MAX(${order_items.sale_price})"),
        (
            (F.measure(REVENUE) / F.count_distinct("users.id")).expr,
            "${order_items.total_sale_price} / COUNT(DISTINCT ${users.id})",
        ),
        (
            (F.measure(REVENUE) / F.measure("users.count")).expr,
            "${order_items.total_sale_price} / ${users.count}",
        ),
    ],
)
def test_each_expression_renders_to_its_omnisql_fragment(expr: Expr, expected: str) -> None:
    assert fragment(expr) == expected


def test_numbers_include_both_ends_and_dates_do_not() -> None:
    """The M2 split, carried into SQL unchanged (see Column.between)."""
    assert "<=" in fragment(F.col(AGE).between(18, 21).expr)
    assert "<=" not in fragment(F.col(CREATED).between(date(2026, 1, 1), date(2026, 2, 1)).expr)


def test_a_udf_has_no_tier_two_rendering() -> None:
    with pytest.raises(CannotCompile, match="no SQL rendering"):
        fragment(F.udf(str.upper)(STATE).expr)


# --------------------------------------------------------------------------------------
# Sentinel substitution and the charset gate (docs/SQLTIER.md §3.1)
# --------------------------------------------------------------------------------------


HOSTILE_NAMES = [
    'weird"name',
    "users.state} + (SELECT 1) + ${users.state",
    "users.state; DROP TABLE users",
    "users.state[month][week]",
    "a.b.c",
    "users state",
]


@pytest.mark.parametrize("name", HOSTILE_NAMES)
def test_a_name_outside_the_sentinel_charset_refuses_to_tier_three(name: str) -> None:
    """Substitution is textual by necessity, so the charset is the whole guarantee (§3.1)."""
    with pytest.raises(CannotCompile, match="OmniSQL statement"):
        fragment(F.col(name).expr)

    assert try_sql(selected(F.col(name))) is None


@pytest.mark.parametrize("name", ["users.state", "users.created_at[month]", "count"])
def test_the_names_omniframes_actually_writes_pass_the_gate(name: str) -> None:
    assert fragment(F.col(name).expr) == "${" + name + "}"


@pytest.mark.parametrize("base_view", ["", "order items", "orders} UNION SELECT 1"])
def test_a_missing_or_unsafe_topic_base_view_refuses(base_view: str) -> None:
    scan = nodes.Scan(nodes.TopicScan(BENCH_MODEL_NAME, BENCH_MODEL_ID, "sales", base_view))

    assert try_sql(nodes.Project(scan, columns(STATE))) is None


def test_a_topic_with_a_different_name_selects_from_its_base_view() -> None:
    scan = nodes.Scan(
        nodes.TopicScan(BENCH_MODEL_NAME, BENCH_MODEL_ID, "Sales topic", "order_items")
    )
    plan = nodes.Aggregate(scan, columns(STATE), (F.sum(PRICE).alias("sales"),))

    compiled = compile_sql(plan)

    assert "FROM ${order_items}" in compiled.query.user_edited_sql
    assert "Sales topic" not in compiled.query.user_edited_sql
    assert compiled.scan == scan.source


def test_a_bare_view_scan_selects_from_the_view() -> None:
    assert "FROM ${order_items}" in sql(nodes.Project(VIEW_SCAN, columns(STATE)))


def test_the_sentinels_never_survive_into_the_statement() -> None:
    statement = sql(
        nodes.Filter(
            aggregate(F.count_distinct("users.id").alias("n"), keys=columns(STATE)),
            (F.col("n") > 1).expr,
        )
    )

    assert "__OF_REF_" not in statement
    assert statement.count("${") == statement.count("}")


# --------------------------------------------------------------------------------------
# Escaping and injection — the reason this module builds an AST at all
# --------------------------------------------------------------------------------------


MALICIOUS = "California'; DROP TABLE users; --"


def test_a_sql_injection_arrives_as_a_literal_and_nothing_else() -> None:
    """The headline safety property: a value is a value, never a fragment of statement."""
    predicate = ((F.col(STATE) == MALICIOUS) | (F.col(AGE) > 60)).expr
    statement = sql(nodes.Filter(selected("order_items.id", STATE, AGE), predicate))
    parsed = sqlglot.parse(_placeholder_free(statement))

    assert "'California''; DROP TABLE users; --'" in statement, "the payload survives, as data"
    assert len(parsed) == 1, "one statement: nothing the value contained terminated it"
    assert isinstance(parsed[0], sqlglot.exp.Select)
    literals = [
        node.this for node in parsed[0].find_all(sqlglot.exp.Literal) if node.args.get("is_string")
    ]
    assert literals == [MALICIOUS], "the whole payload is one string literal, verbatim"
    assert not parsed[0].find(sqlglot.exp.Drop), "and nothing became a DROP"


def test_an_injected_value_is_a_literal_in_every_arm_that_takes_one() -> None:
    quoted = "'California''; DROP TABLE users; --'"
    assert fragment((F.col(STATE) == MALICIOUS).expr) == f"${{users.state}} = {quoted}"
    assert fragment(F.col(STATE).isin(MALICIOUS).expr) == f"${{users.state}} IN ({quoted})"
    assert fragment(F.lit(MALICIOUS).expr) == quoted


def test_a_value_that_looks_like_a_model_reference_stays_a_value() -> None:
    """``${users.state}`` typed as a *value* is text, and the sentinel pass never sees it."""
    payload = "${users.state}"
    statement = sql(nodes.Filter(selected(STATE, AGE), (F.col(STATE) == payload).expr))

    assert "${users.state} = '${users.state}'" in statement, "the value stayed quoted"
    assert statement.count("'${users.state}'") == 1


def test_a_value_that_looks_like_a_sentinel_refuses_the_statement() -> None:
    """The only text that could reach substitution without going through the reference table."""
    payload = "__OF_REF_1__"

    assert try_sql(nodes.Filter(selected(STATE), (F.col(STATE) == payload).expr)) is None


def test_a_backslash_in_a_value_rides_through_as_a_value() -> None:
    """The server re-renders the parsed statement per warehouse, so escaping is its problem now.

    v1 refused a backslash-bearing literal unless the caller named the warehouse; on the parsed
    path a backslash is not an escape and passes verbatim (CONTRACT_NOTES §3.6, probe P7b), so
    the refusal — and the ``sql_dialect`` knob behind it — is gone (docs/SQLTIER.md §6).
    """
    for payload in ("x\\' OR 1=1 --", "C:\\Users\\dan"):
        statement = sql(nodes.Filter(selected(STATE, AGE), (F.col(STATE) == payload).expr))
        assert payload.replace("'", "''") in statement


def test_like_metacharacters_in_a_value_are_escaped_not_interpreted() -> None:
    """``contains("50%")`` looks for the text ``50%``, not for "50 followed by anything"."""
    assert fragment(F.col(STATE).contains("50%").expr) == "${users.state} LIKE '%50!%%' ESCAPE '!'"
    assert (
        fragment(F.col(STATE).starts_with("a_b").expr) == "${users.state} LIKE 'a!_b%' ESCAPE '!'"
    )
    assert fragment(F.col(STATE).ends_with("c!d").expr) == "${users.state} LIKE '%c!!d' ESCAPE '!'"


def test_a_like_pattern_is_passed_through_because_there_the_wildcards_are_the_point() -> None:
    assert "ESCAPE" not in fragment(F.col(STATE).like("Cal%").expr)


# --------------------------------------------------------------------------------------
# The envelope: one OmniSQL statement, no reference core (docs/SQLTIER.md §2)
# --------------------------------------------------------------------------------------


def test_the_statement_selects_from_the_base_view_and_carries_no_references() -> None:
    compiled = compile_sql(aggregate(F.count("users.id"), keys=columns(STATE)))
    wire = compiled.query.to_wire()

    assert "FROM ${order_items}" in compiled.query.user_edited_sql
    assert compiled.query.omnisql is True
    assert compiled.query.static_query_references == {}
    assert "rewriteSql" not in wire, "an ABSENT key is what selects the parsed path (§3.6)"
    assert "sqlSortsEnabled" not in wire
    assert "staticQueryReferences" not in wire
    assert wire["fields"] == []
    assert wire["userEditedSQL"] == compiled.query.user_edited_sql


def test_the_envelope_limit_mirrors_the_text_because_only_the_text_is_applied() -> None:
    """The server ignores the query object's limit on this path (§3.6); ``user_limit`` reads it."""
    default = compile_sql(aggregate(F.count("users.id"), keys=columns(STATE)))
    assert default.query.effective_limit == DEFAULT_FETCH_LIMIT
    assert f"LIMIT {DEFAULT_FETCH_LIMIT}" in default.query.user_edited_sql
    assert default.query.limit_is_unset is True

    user = compile_sql(nodes.Limit(aggregate(F.count("users.id"), keys=columns(STATE)), 10, 5))
    assert user.query.effective_limit == 10
    assert user.query.offset == 5
    assert "LIMIT 10" in user.query.user_edited_sql
    assert "OFFSET 5" in user.query.user_edited_sql


def test_only_limit_none_emits_a_statement_without_a_limit_clause() -> None:
    unlimited = compile_sql(nodes.Limit(aggregate(F.count("users.id"), keys=columns(STATE)), None))

    assert "LIMIT" not in unlimited.query.user_edited_sql
    assert unlimited.query.effective_limit is None


def test_no_statement_ever_says_select_distinct() -> None:
    """The parser strips ``DISTINCT`` silently (§3.6), so dedup stays local — never emit one."""
    statement = sql(aggregate(F.count_distinct("users.id").alias("n"), keys=columns(STATE)))

    assert "SELECT DISTINCT" not in statement
    assert "COUNT(DISTINCT ${users.id})" in statement


# --------------------------------------------------------------------------------------
# Naming — the two regimes of docs/SQLTIER.md §3.2
# --------------------------------------------------------------------------------------


def test_a_bare_ref_is_emitted_unaliased_and_renamed_client_side() -> None:
    """SQL aliases on a bare ref are IGNORED by the server, so the rename happens here."""
    compiled = compile_sql(
        nodes.Project(SCAN, columns(F.col(STATE).alias("state"), F.col(AGE).alias("age")))
    )

    assert " AS " not in compiled.query.user_edited_sql
    assert compiled.aliases == {STATE: "state", AGE: "age"}
    assert compiled.columns == ("state", "age")


def test_a_measure_and_a_grain_are_bare_refs_too() -> None:
    month = F.col(CREATED).grain("month").alias("month")
    compiled = compile_sql(
        nodes.Aggregate(SCAN, (month,), columns(F.measure(REVENUE).alias("revenue")))
    )

    assert " AS " not in compiled.query.user_edited_sql
    assert compiled.aliases == {
        "order_items.created_at[month]": "month",
        REVENUE: "revenue",
    }
    assert compiled.columns == ("month", "revenue")


def test_an_expression_item_gets_a_generated_alias_matched_by_suffix() -> None:
    """The scope prefix the server prepends is unpredictable, so only the suffix is contracted."""
    compiled = compile_sql(
        nodes.Aggregate(
            SCAN,
            columns(STATE),
            columns(F.count_distinct("users.id").alias("buyers"), F.sum(PRICE).alias("revenue")),
        )
    )

    assert "COUNT(DISTINCT ${users.id}) AS of_expr_1" in compiled.query.user_edited_sql
    assert "SUM(${order_items.sale_price}) AS of_expr_2" in compiled.query.user_edited_sql
    assert compiled.aliases == {"of_expr_1": "buyers", "of_expr_2": "revenue"}
    assert compiled.columns == (STATE, "buyers", "revenue")


def test_generated_aliases_are_numbered_over_expression_items_in_select_order() -> None:
    compiled = compile_sql(
        nodes.Project(
            SCAN,
            columns(F.count("users.id").alias("n"), STATE, F.sum(PRICE).alias("total")),
        )
    )
    statement = compiled.query.user_edited_sql

    assert statement.index("AS of_expr_1") < statement.index("${users.state}")
    assert statement.index("${users.state}") < statement.index("AS of_expr_2")
    assert compiled.columns == ("n", STATE, "total")


def test_every_generated_alias_carries_the_reserved_prefix() -> None:
    compiled = compile_sql(aggregate(F.count("users.id").alias("n"), keys=columns(STATE)))

    for key in compiled.aliases:
        assert key == STATE or key.startswith(EXPR_ALIAS_PREFIX)


def test_a_repeated_bare_ref_is_de_duplicated_at_emission() -> None:
    """The server collapses duplicates and shifts positions, so the statement must not repeat."""
    compiled = compile_sql(nodes.Project(SCAN, columns(STATE, STATE)))

    assert compiled.query.user_edited_sql.count("${users.state}") == 1
    assert compiled.columns == (STATE,)


def test_one_field_under_two_output_names_is_refused_rather_than_half_answered() -> None:
    """One distinct bare ref comes back as one column whatever the SQL says (§3.2)."""
    assert try_sql(nodes.Project(SCAN, columns(STATE, F.col(STATE).alias("s")))) is None


# --------------------------------------------------------------------------------------
# Shapes tier 2 accepts
# --------------------------------------------------------------------------------------


def test_group_by_and_order_by_are_positional() -> None:
    """The only verified forms; over a formatted grain the position sorts by ``__raw`` (§3.1)."""
    plan = nodes.Sort(
        nodes.Aggregate(
            SCAN,
            columns(STATE, F.col(CREATED).grain("month")),
            columns(F.count("users.id").alias("n")),
        ),
        (F.col("n").desc().to_sort_key(), F.col(STATE).to_sort_key()),
    )
    statement = sql(plan)

    assert "GROUP BY\n  1,\n  2" in statement
    assert "ORDER BY\n  3 DESC,\n  1 ASC NULLS LAST" in statement


def test_a_group_less_aggregate_emits_no_group_by() -> None:
    assert "GROUP BY" not in sql(aggregate(F.sum(AGE)))


def test_a_measure_only_select_is_a_grand_total() -> None:
    """Without a GROUP BY a measure select is the grand total (CONTRACT_NOTES §3.6)."""
    compiled = compile_sql(nodes.Project(SCAN, columns(F.measure(REVENUE).alias("revenue"))))

    assert "GROUP BY" not in compiled.query.user_edited_sql
    assert "${order_items.total_sale_price}" in compiled.query.user_edited_sql
    assert compiled.measures == (REVENUE,)


def test_a_mixed_aggregate_is_one_statement() -> None:
    """The M3 decomposition is now the fallback, not the rule (docs/SQLTIER.md §4)."""
    compiled = compile_sql(
        nodes.Aggregate(
            SCAN,
            columns(STATE),
            columns(F.measure(REVENUE).alias("revenue"), F.count_distinct("users.id").alias("b")),
        )
    )
    statement = compiled.query.user_edited_sql

    assert "${order_items.total_sale_price}" in statement
    assert "COUNT(DISTINCT ${users.id}) AS of_expr_1" in statement
    assert compiled.columns == (STATE, "revenue", "b")
    assert compiled.group_keys == (STATE,)


def test_measure_arithmetic_is_a_select_item() -> None:
    """``${m} / COUNT(DISTINCT ${f})`` is expanded server-side (CONTRACT_NOTES §3.6, probe L3)."""
    compiled = compile_sql(
        nodes.Aggregate(
            SCAN,
            columns(STATE),
            columns((F.measure(REVENUE) / F.count_distinct("users.id")).alias("per_buyer")),
        )
    )

    assert (
        "${order_items.total_sale_price} / COUNT(DISTINCT ${users.id}) AS of_expr_1"
        in compiled.query.user_edited_sql
    )
    assert compiled.aliases == {"of_expr_1": "per_buyer"}


def test_having_substitutes_the_full_aggregate_for_its_alias() -> None:
    plan = nodes.Filter(
        aggregate(F.count_distinct("users.id").alias("b"), keys=columns(STATE)),
        (F.col("b") > 25).expr,
    )
    statement = sql(plan)

    assert "HAVING\n  COUNT(DISTINCT ${users.id}) > 25" in statement
    assert "HAVING\n  b" not in statement


def test_having_over_a_governed_measure_expands_the_ref_inline() -> None:
    """Verified live (probe L4): a ``${measure}`` ref in HAVING expands to its aggregate."""
    plan = nodes.Filter(
        nodes.Aggregate(
            SCAN,
            columns(STATE),
            columns(F.measure(REVENUE).alias("revenue"), F.count("users.id").alias("n")),
        ),
        (F.col("revenue") > 1000).expr,
    )

    assert "HAVING\n  ${order_items.total_sale_price} > 1000" in sql(plan)


def test_a_select_with_an_ad_hoc_aggregation_is_the_same_group_by() -> None:
    """Selection IS the group-by, and tier 2 keeps the user's written column order."""
    project = nodes.Project(SCAN, columns(F.count("users.id").alias("n"), STATE))
    statement = sql(project)

    assert statement.index("AS of_expr_1") < statement.index("${users.state}")
    assert compile_sql(project).columns == ("n", STATE)
    assert "GROUP BY\n  2" in statement


def test_a_computed_column_becomes_a_select_expression() -> None:
    plan = nodes.WithColumn(
        selected(PRICE, "order_items.quantity"),
        "unit",
        (F.col(PRICE) / F.col("order_items.quantity")).expr,
    )
    compiled = compile_sql(plan)

    assert (
        "${order_items.sale_price} / ${order_items.quantity} AS of_expr_1"
        in compiled.query.user_edited_sql
    )
    assert compiled.columns == (PRICE, "order_items.quantity", "unit")
    assert compiled.aliases == {"of_expr_1": "unit"}


def test_a_computed_column_replaces_the_one_it_shadows() -> None:
    plan = nodes.WithColumn(selected(STATE, AGE), AGE, (F.col(AGE) * 2).expr)
    compiled = compile_sql(plan)

    assert compiled.columns == (STATE, AGE)
    assert "${users.age} * 2 AS of_expr_1" in compiled.query.user_edited_sql
    assert compiled.aliases == {"of_expr_1": AGE}


def test_a_derived_column_over_an_aggregate_output_substitutes_the_aggregate() -> None:
    plan = nodes.WithColumn(
        aggregate(F.count("users.id").alias("n"), keys=columns(STATE)),
        "doubled",
        (F.col("n") * 2).expr,
    )

    assert "COUNT(${users.id}) * 2 AS of_expr_2" in sql(plan)


def test_a_non_aggregated_sort_over_an_unselected_column_renders_inline() -> None:
    """The server wraps the statement in a subquery whose sort sidecars never leak (probe L5)."""
    plan = nodes.Sort(selected(STATE), (F.col(AGE).desc().to_sort_key(),))

    assert "ORDER BY\n  ${users.age} DESC" in sql(plan)


def test_the_sort_summary_names_the_output_columns() -> None:
    plan = nodes.Sort(
        aggregate(F.count("users.id").alias("n"), keys=columns(STATE)),
        (F.col("n").desc().to_sort_key(),),
    )

    assert compile_sql(plan).sorts == (("n", True),)


# --------------------------------------------------------------------------------------
# WHERE discipline (docs/SQLTIER.md §3.3)
# --------------------------------------------------------------------------------------


def test_every_row_predicate_is_a_where_conjunct_now() -> None:
    """v1 pushed the tier-1-expressible ones into a reference core; there is no core left."""
    plan = nodes.Aggregate(
        nodes.Filter(
            nodes.Filter(SCAN, (F.col("order_items.status") == "complete").expr),
            ((F.col(STATE) == "California") | (F.col(AGE) > 60)).expr,
        ),
        columns(STATE),
        columns(F.count("users.id")),
    )
    statement = sql(plan)

    assert "${order_items.status} = 'complete'" in statement
    assert "${users.state} = 'California' OR ${users.age} > 60" in statement


def test_a_compound_range_emits_as_one_parenthesized_conjunct() -> None:
    """Same-column compound ranges survive the parser intact (probe L1) — so they are emitted."""
    plan = nodes.Filter(selected("order_items.id", AGE), F.col(AGE).between(18, 21).expr)
    statement = sql(plan)

    assert "(\n    ${users.age} >= 18 AND ${users.age} <= 21\n  )" in statement


def test_a_date_range_keeps_its_half_open_upper_bound() -> None:
    plan = nodes.Filter(
        selected("order_items.id", CREATED),
        F.col(CREATED).between(date(2025, 7, 1), date(2026, 7, 1)).expr,
    )
    statement = sql(plan)

    assert "${order_items.created_at} >= CAST('2025-07-01' AS DATE)" in statement
    assert "${order_items.created_at} < CAST('2026-07-01' AS DATE)" in statement


def test_a_grain_ref_never_reaches_the_where() -> None:
    """The one live-observed predicate loss: a grain ref beside the bare field (§3.3)."""
    month = F.col(CREATED).grain("month")
    plan = nodes.Filter(selected("order_items.id", STATE), (month >= date(2026, 3, 1)).expr)

    assert try_sql(plan) is None


@pytest.mark.parametrize("spelling", ["order_items.created_at[month]"])
def test_the_bracketed_spelling_of_a_grain_is_refused_in_where_too(spelling: str) -> None:
    plan = nodes.Filter(selected("order_items.id", STATE), (F.col(spelling) >= 1).expr)

    assert try_sql(plan) is None


def test_a_grain_is_still_a_legal_select_item_and_group_key() -> None:
    month = F.col(CREATED).grain("month").alias("month")
    statement = sql(nodes.Aggregate(SCAN, (month,), columns(F.count("users.id").alias("n"))))

    assert "${order_items.created_at[month]}" in statement
    assert "GROUP BY\n  1" in statement


def test_a_governed_measure_is_refused_in_the_where() -> None:
    plan = nodes.Aggregate(
        nodes.Filter(SCAN, (F.measure(REVENUE) > 10).expr),
        columns(STATE),
        columns(F.count("users.id")),
    )

    assert try_sql(plan) is None


def test_a_filter_on_a_group_key_above_the_aggregate_rides_the_where_half() -> None:
    """``GROUP BY k HAVING k = x`` and ``WHERE k = x GROUP BY k`` are the same question."""
    plan = nodes.Filter(
        aggregate(F.count("users.id"), keys=columns(STATE)),
        (F.col(STATE) == "California").expr,
    )
    statement = sql(plan)

    assert "WHERE\n  ${users.state} = 'California'" in statement
    assert "HAVING" not in statement


# --------------------------------------------------------------------------------------
# Shapes tier 2 refuses (docs/SQLTIER.md §1) — every one of them falls through to tier 3
# --------------------------------------------------------------------------------------


def test_a_raw_sql_scan_is_never_re_derived() -> None:
    assert try_sql(nodes.Project(SQL_SCAN, columns("x"))) is None


def test_a_stored_query_scan_is_never_re_derived() -> None:
    assert try_sql(nodes.Project(STORED_SCAN, columns(STATE))) is None


def test_a_udf_keeps_the_plan_out_of_tier_two() -> None:
    plan = nodes.Filter(selected(STATE), (F.udf(str.upper)(STATE) == "OHIO").expr)

    assert try_sql(plan) is None


def test_map_pandas_keeps_the_plan_out_of_tier_two() -> None:
    hint = OmniSchema((OmniField(name="x", data_type=OmniDataType.STRING),))

    assert try_sql(nodes.MapPandas(selected(STATE), lambda frame: frame, hint)) is None


def test_a_join_or_a_union_keeps_the_plan_out_of_tier_two() -> None:
    assert try_sql(nodes.Join(selected(STATE), selected(STATE), (STATE,))) is None
    assert try_sql(nodes.Union(selected(STATE), selected(STATE))) is None


def test_an_aggregate_above_a_user_limit_stays_tier_three() -> None:
    """The limit pins the frontier: paging then aggregating is the question the user asked."""
    plan = nodes.Aggregate(
        nodes.Limit(selected(STATE, PRICE), 100), columns(STATE), columns(F.sum(PRICE))
    )

    assert try_sql(plan) is None


def test_an_operation_above_a_limit_stays_tier_three() -> None:
    plan = nodes.Filter(nodes.Limit(selected(STATE), 5), (F.col(STATE) == "Ohio").expr)

    assert try_sql(plan) is None


@pytest.mark.parametrize("literal", ["30 days ago", "last quarter", "today", "2 complete days ago"])
def test_a_relative_date_literal_is_refused_rather_than_written_into_sql(literal: str) -> None:
    """Only Omni evaluates that grammar; a warehouse would read it as a string (§1)."""
    predicate = ((F.col(CREATED) >= literal) | (F.col(AGE) > 60)).expr

    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is None


def test_a_relative_date_literal_is_refused_next_to_equals_too() -> None:
    """The grammar is the grammar whichever operator it sits beside — ``==`` was exempt."""
    predicate = ((F.col(CREATED) == "today") | (F.col(AGE) > 60)).expr

    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is None


@pytest.mark.parametrize(
    "build",
    [
        lambda month: month == "2026-03",
        lambda month: month != "2026-03",
        lambda month: month >= "2026-03",
        lambda month: month.between("2026-01", "2026-04"),
        lambda month: month.isin("2026-01", "2026-02"),
    ],
)
def test_a_string_against_a_timestamp_grain_is_refused_rather_than_compared_as_text(
    build: Any,
) -> None:
    """A grain pins the type, so the string is a DATE literal — tier 1 compiles it as one."""
    month = F.col(CREATED).grain("month")
    predicate = (build(month) | (F.col(AGE) > 60)).expr

    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is None


def test_an_absolute_date_literal_is_not_mistaken_for_a_relative_one() -> None:
    predicate = ((F.col(CREATED) >= date(2026, 1, 1)) | (F.col(AGE) > 60)).expr

    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is not None


def test_comparing_to_none_is_refused_rather_than_rendered_as_equals_null() -> None:
    predicate = ((F.col(STATE) == None) | (F.col(AGE) > 60)).expr  # noqa: E711 - a Column
    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is None


def test_a_bare_scan_has_nothing_to_select() -> None:
    assert try_sql(SCAN) is None


def test_a_having_with_nothing_to_group_is_refused() -> None:
    """``select(dim).filter(count_distinct(...) > 10)`` has no aggregate to hang a HAVING on."""
    plan = nodes.Filter(selected(STATE), (F.count_distinct("users.id") > 10).expr)

    assert try_sql(plan) is None


def test_a_filter_above_an_aggregate_on_a_consumed_column_is_refused() -> None:
    """Not tier 2's to reinterpret: moving it below the GROUP BY answers a different question."""
    plan = nodes.Filter(
        aggregate(F.count("users.id").alias("n"), keys=columns(STATE)),
        (F.col(AGE) * 2 > 1).expr,
    )

    assert try_sql(plan) is None


def test_a_having_that_also_names_a_consumed_column_is_refused() -> None:
    """An aggregate somewhere in the predicate does not license the rest of it."""
    agg = aggregate(F.count("users.id").alias("n"), keys=columns(STATE))

    assert try_sql(nodes.Filter(agg, ((F.col("n") > 10) | (F.col(AGE) == 30)).expr)) is None
    assert try_sql(nodes.Filter(agg, (F.col("n") > F.col(AGE)).expr)) is None
    # The legitimate shape — a group key beside the aggregate — still compiles.
    legal = nodes.Filter(agg, ((F.col("n") > 10) | (F.col(STATE) == "California")).expr)
    compiled = try_sql(legal)
    assert compiled is not None
    assert "HAVING" in compiled.query.user_edited_sql


def test_a_derived_column_above_an_aggregate_over_a_consumed_column_is_refused() -> None:
    plan = nodes.WithColumn(
        aggregate(F.count("users.id").alias("n"), keys=columns(STATE)),
        "doubled",
        (F.col(AGE) * 2).expr,
    )

    assert try_sql(plan) is None


def test_replacing_a_group_key_with_a_derived_column_is_refused() -> None:
    """The GROUP BY would name a column the SELECT no longer produces."""
    plan = nodes.WithColumn(
        aggregate(F.count("users.id").alias("n"), keys=columns(AGE)),
        AGE,
        (F.col(AGE) * 2).expr,
    )

    assert try_sql(plan) is None


def test_the_same_filter_over_a_group_key_is_accepted() -> None:
    """The other side of the line: a group key still exists above the aggregate."""
    plan = nodes.Filter(
        nodes.Aggregate(SCAN, columns(STATE, AGE), columns(F.count("users.id").alias("n"))),
        (F.col(AGE) * 2 > 1).expr,
    )

    assert try_sql(plan) is not None


def test_sorting_an_aggregate_by_a_column_it_does_not_select_is_refused() -> None:
    plan = nodes.Sort(
        aggregate(F.count("users.id"), keys=columns(STATE)),
        (F.col(AGE).to_sort_key(),),
    )

    assert try_sql(plan) is None


def test_a_bare_field_beside_an_aggregate_is_refused_rather_than_left_ungrouped() -> None:
    """``users.age * 2`` beside a COUNT is neither grouped nor aggregated — a binder error."""
    plan = nodes.Aggregate(SCAN, columns(STATE), columns((F.col(AGE) * 2).alias("x")))

    assert try_sql(plan) is None


# --------------------------------------------------------------------------------------
# The carrier
# --------------------------------------------------------------------------------------


def test_a_tier_two_compilation_fills_the_semantic_carrier() -> None:
    """``RemoteStep`` reads only these attributes, which is why tier 2 needs no new class."""
    compiled = compile_sql(aggregate(F.count("users.id").alias("n"), keys=columns(STATE)))

    assert compiled.tier == 2
    assert compiled.role == "sql"
    assert compiled.is_sql is True
    assert compiled.opaque is False, "omniframes wrote this SQL; it was not handed it"
    assert compiled.aliases == {"of_expr_1": "n"}
    assert compiled.columns == (STATE, "n")
    assert compiled.group_keys == (STATE,)
    assert compiled.measure_filters == ()
    assert compiled.scan is SCAN.source


def test_the_statement_is_pretty_printed_so_the_golden_files_stay_readable() -> None:
    assert "\n" in sql(aggregate(F.count("users.id"), keys=columns(STATE)))


def test_compiling_the_same_plan_twice_gives_byte_identical_sql() -> None:
    plan = nodes.Aggregate(SCAN, columns(STATE, AGE), columns(F.sum(PRICE), F.count("users.id")))

    assert sql(plan) == sql(plan), "determinism is what makes the golden files reviewable"


def _placeholder_free(statement: str) -> str:
    """``${view.field}`` back to a plain identifier, so sqlglot can re-parse the text.

    OmniSQL is not SQL until the server substitutes it, and the injection tests want to read the
    statement's *structure*: one SELECT, one string literal, no second statement.
    """
    return re.sub(r"\$\{([^}]*)\}", lambda m: '"' + m.group(1) + '"', statement)
