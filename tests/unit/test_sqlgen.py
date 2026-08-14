"""Unit tests for tier 2 (docs/SQLTIER.md §3).

Two halves.  The first is the translation table, row by row: one expression in, one SQL fragment
out, rendered through the same code path the compiler uses.  The second is the compiler itself —
what shape it accepts, what it refuses, how the reference core is built, and where each predicate
ends up.

The injection cases are the reason the translation goes through :mod:`sqlglot` nodes at all: a
filter value is a *value*, and there is no code path that turns one into SQL text.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pytest
import sqlglot
from tests.fakes import BENCH_MODEL_ID, BENCH_MODEL_NAME, BENCH_TOPIC_NAME

from omniframes import functions as F
from omniframes.column import Column, Expr
from omniframes.compile.semantic import CannotCompile, SemanticCompilation
from omniframes.compile.splitter import SplitOptions
from omniframes.compile.sqlgen import compile_sql, render_expr, try_sql
from omniframes.plan import nodes
from omniframes.types import OmniDataType, OmniField, OmniSchema

REVENUE = "order_items.total_sale_price"
STATE = "users.state"
AGE = "users.age"
PRICE = "order_items.sale_price"

SCAN = nodes.Scan(
    nodes.TopicScan(
        model_name=BENCH_MODEL_NAME,
        model_id=BENCH_MODEL_ID,
        topic=BENCH_TOPIC_NAME,
        base_view="order_items",
    )
)
SQL_SCAN = nodes.Scan(nodes.SqlScan(BENCH_MODEL_ID, "SELECT 1 AS x", BENCH_MODEL_NAME))
STORED_SCAN = nodes.Scan(
    nodes.SavedQueryScan("doc", "saved", {"modelId": BENCH_MODEL_ID, "fields": [STATE]})
)


def columns(*names: str | Column) -> tuple[Column, ...]:
    return tuple(F.col(name) if isinstance(name, str) else name for name in names)


def selected(*names: str | Column) -> nodes.Project:
    return nodes.Project(SCAN, columns(*names))


def sql(plan: nodes.PlanNode, **options: Any) -> str:
    return compile_sql(plan, options=SplitOptions(**options)).query.user_edited_sql


def reference_of(compilation: SemanticCompilation) -> Any:
    references = compilation.query.static_query_references
    assert list(references) == ["ref_1"]
    return references["ref_1"]


def fragment(expr: Expr) -> str:
    return render_expr(expr).sql()


# --------------------------------------------------------------------------------------
# The translation table (docs/SQLTIER.md §3)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (F.col(STATE).expr, '"users.state"'),
        (F.col("order_items.created_at").grain("month").expr, '"order_items.created_at[month]"'),
        (F.lit("California").expr, "'California'"),
        (F.lit(42).expr, "42"),
        (F.lit(3.5).expr, "3.5"),
        (F.lit(Decimal("99.50")).expr, "99.50"),
        (F.lit(True).expr, "TRUE"),
        (F.lit(False).expr, "FALSE"),
        (F.lit(date(2026, 3, 1)).expr, "CAST('2026-03-01' AS DATE)"),
        (F.lit(datetime(2026, 3, 1, 12, 30)).expr, "CAST('2026-03-01 12:30:00' AS TIMESTAMP)"),
        ((F.col(AGE) == 30).expr, '"users.age" = 30'),
        ((F.col(AGE) != 30).expr, '"users.age" <> 30'),
        ((F.col(AGE) < 30).expr, '"users.age" < 30'),
        ((F.col(AGE) <= 30).expr, '"users.age" <= 30'),
        ((F.col(AGE) > 30).expr, '"users.age" > 30'),
        ((F.col(AGE) >= 30).expr, '"users.age" >= 30'),
        ((F.col(AGE) + 1).expr, '"users.age" + 1'),
        ((F.col(AGE) - 1).expr, '"users.age" - 1'),
        ((F.col(AGE) * 2).expr, '"users.age" * 2'),
        ((F.col(AGE) / 2).expr, '"users.age" / 2'),
        (
            ((F.col(AGE) > 30) & (F.col(STATE) == "Ohio")).expr,
            '("users.age" > 30 AND "users.state" = \'Ohio\')',
        ),
        (
            ((F.col(AGE) > 30) | (F.col(STATE) == "Ohio")).expr,
            '("users.age" > 30 OR "users.state" = \'Ohio\')',
        ),
        ((~(F.col(AGE) > 30)).expr, 'NOT ("users.age" > 30)'),
        (F.col(STATE).is_null().expr, '"users.state" IS NULL'),
        (F.col(STATE).is_not_null().expr, 'NOT ("users.state" IS NULL)'),
        (F.col(STATE).isin("Ohio", "Texas").expr, "\"users.state\" IN ('Ohio', 'Texas')"),
        (F.col(STATE).contains("cal").expr, "\"users.state\" LIKE '%cal%' ESCAPE '!'"),
        (F.col(STATE).starts_with("New").expr, "\"users.state\" LIKE 'New%' ESCAPE '!'"),
        (F.col(STATE).ends_with("ia").expr, "\"users.state\" LIKE '%ia' ESCAPE '!'"),
        (F.col(STATE).like("%a_b%").expr, "\"users.state\" LIKE '%a_b%'"),
        (
            F.col(STATE).contains("cal", case_insensitive=True).expr,
            "LOWER(\"users.state\") LIKE LOWER('%cal%') ESCAPE '!'",
        ),
        ((F.col(AGE).between(18, 21)).expr, '("users.age" >= 18 AND "users.age" <= 21)'),
        (
            (F.col("order_items.created_at").between(date(2026, 1, 1), date(2026, 2, 1))).expr,
            "(\"order_items.created_at\" >= CAST('2026-01-01' AS DATE)"
            " AND \"order_items.created_at\" < CAST('2026-02-01' AS DATE))",
        ),
        (F.sum(PRICE).expr, 'SUM("order_items.sale_price")'),
        (F.count(PRICE).expr, 'COUNT("order_items.sale_price")'),
        (F.count_distinct(PRICE).expr, 'COUNT(DISTINCT "order_items.sale_price")'),
        (F.avg(PRICE).expr, 'AVG("order_items.sale_price")'),
        (F.min(PRICE).expr, 'MIN("order_items.sale_price")'),
        (F.max(PRICE).expr, 'MAX("order_items.sale_price")'),
    ],
)
def test_each_expression_renders_to_its_sql_fragment(expr: Expr, expected: str) -> None:
    assert fragment(expr) == expected


def test_numbers_include_both_ends_and_dates_do_not() -> None:
    """The M2 split, carried into SQL unchanged (see Column.between)."""
    assert "<=" in fragment(F.col(AGE).between(18, 21).expr)
    assert "<=" not in fragment(
        F.col("order_items.created_at").between(date(2026, 1, 1), date(2026, 2, 1)).expr
    )


def test_a_governed_measure_has_no_tier_two_rendering() -> None:
    with pytest.raises(CannotCompile, match="governed measure"):
        fragment(F.measure(REVENUE).expr)


def test_a_udf_has_no_tier_two_rendering() -> None:
    with pytest.raises(CannotCompile, match="no SQL rendering"):
        fragment(F.udf(str.upper)(STATE).expr)


# --------------------------------------------------------------------------------------
# Escaping and injection — the reason this module builds an AST at all
# --------------------------------------------------------------------------------------


MALICIOUS = "California'; DROP TABLE users; --"


def test_a_sql_injection_arrives_as_a_literal_and_nothing_else() -> None:
    """The headline safety property: a value is a value, never a fragment of statement.

    The predicate is a cross-field OR on purpose — that is the shape tier 2 has to *write* as
    SQL, rather than push into the reference as a typed wire filter.
    """
    predicate = ((F.col(STATE) == MALICIOUS) | (F.col(AGE) > 60)).expr
    statement = sql(nodes.Filter(selected("order_items.id", STATE, AGE), predicate))
    parsed = sqlglot.parse(statement)

    assert "'California''; DROP TABLE users; --'" in statement, "the payload survives, as data"
    assert len(parsed) == 1, "one statement: nothing the value contained terminated it"
    assert isinstance(parsed[0], sqlglot.exp.Select)
    literals = [
        node.this for node in parsed[0].find_all(sqlglot.exp.Literal) if node.args.get("is_string")
    ]
    assert literals == [MALICIOUS], "the whole payload is one string literal, verbatim"
    assert not parsed[0].find(sqlglot.exp.Drop), "and nothing became a DROP"


def test_an_injected_value_pushed_into_the_reference_is_a_typed_filter_value() -> None:
    """The other half of the same guarantee: a pushable predicate never becomes SQL at all."""
    plan = nodes.Aggregate(nodes.Filter(SCAN, (F.col(STATE) == MALICIOUS).expr), columns(STATE), ())
    compiled = compile_sql(plan)

    assert "DROP" not in compiled.query.user_edited_sql
    entry = reference_of(compiled).to_wire()["filters"][STATE]
    assert entry == {"type": "string", "kind": "EQUALS", "values": [MALICIOUS]}


def test_an_injected_value_is_a_literal_in_every_arm_that_takes_one() -> None:
    quoted = "'California''; DROP TABLE users; --'"
    assert fragment((F.col(STATE) == MALICIOUS).expr) == f'"users.state" = {quoted}'
    assert fragment(F.col(STATE).isin(MALICIOUS).expr) == f'"users.state" IN ({quoted})'
    assert fragment(F.lit(MALICIOUS).expr) == quoted


#: Warehouses where a backslash escapes *inside* a string literal, so the ANSI-ish default
#: rendering of one is not the value that arrives.  (Postgres/DuckDB are the other family.)
BACKSLASH_DIALECTS = ["snowflake", "bigquery", "redshift", "spark", "databricks", "mysql"]


@pytest.mark.parametrize("dialect", BACKSLASH_DIALECTS)
def test_the_generated_string_predicates_tokenize_on_backslash_escaping_warehouses(
    dialect: str,
) -> None:
    """``ESCAPE '\\'`` is a *lexer* error there — the backslash swallows the closing quote.

    Not a portability nuance (CONTRACT_NOTES §6 #9): those warehouses all support LIKE/ESCAPE,
    the emitted text simply did not parse, so every tier-2 statement carrying a
    contains/starts_with/ends_with failed outright until the escape character stopped being a
    backslash.
    """
    predicate = (F.col(STATE).contains("cal") | (F.col(AGE) > 60)).expr
    statement = sql(nodes.Filter(selected("order_items.id", STATE, AGE), predicate))

    assert sqlglot.parse(statement, read=dialect), statement


def test_a_backslash_in_a_value_is_refused_until_the_warehouse_is_named() -> None:
    """The default dialect escapes only ``'``, so a backslash is dialect-dependent SQL text.

    On Snowflake/BigQuery/Redshift/Spark/MySQL the default rendering of ``x\\'`` ends the literal
    one character early and the rest of the value is parsed as SQL — a tautology, or a subquery
    against a table outside the governed model.  Omniframes is never told the connection's
    dialect, so without one it declines and the predicate runs one tier down.
    """
    payload = "x\\' OR 1=1 --"
    plan = nodes.Filter(
        selected("order_items.id", STATE, AGE), ((F.col(STATE) == payload) | (F.col(AGE) > 60)).expr
    )

    assert try_sql(plan) is None, "no dialect named: decline rather than guess the escaping"
    statement = sql(plan, sql_dialect="snowflake")
    parsed = sqlglot.parse(statement, read="snowflake")
    assert len(parsed) == 1
    select = parsed[0]
    assert select is not None
    literals = [
        node.this for node in select.find_all(sqlglot.exp.Literal) if node.args.get("is_string")
    ]
    assert payload in literals, "named the warehouse: the value round-trips as one literal"
    assert not select.find(sqlglot.exp.Boolean), "and 1=1 never became a predicate"


def test_a_benign_backslash_is_not_silently_corrupted_either() -> None:
    """``C:\\Users\\dan`` is not an attack and is mangled just the same. Same refusal."""
    plan = nodes.Filter(
        selected("order_items.id", STATE, AGE),
        ((F.col(STATE) == "C:\\Users\\dan") | (F.col(AGE) > 60)).expr,
    )

    assert try_sql(plan) is None
    assert try_sql(plan, options=SplitOptions(sql_dialect="bigquery")) is not None


def test_like_metacharacters_in_a_value_are_escaped_not_interpreted() -> None:
    """``contains("50%")`` looks for the text ``50%``, not for "50 followed by anything"."""
    assert fragment(F.col(STATE).contains("50%").expr) == (
        "\"users.state\" LIKE '%50!%%' ESCAPE '!'"
    )
    assert fragment(F.col(STATE).starts_with("a_b").expr) == (
        "\"users.state\" LIKE 'a!_b%' ESCAPE '!'"
    )
    assert fragment(F.col(STATE).ends_with("c!d").expr) == (
        "\"users.state\" LIKE '%c!!d' ESCAPE '!'"
    )


def test_a_like_pattern_is_passed_through_because_there_the_wildcards_are_the_point() -> None:
    assert "ESCAPE" not in fragment(F.col(STATE).like("Cal%").expr)


def test_a_quote_in_an_identifier_would_be_doubled_too() -> None:
    assert fragment(F.col('weird"name').expr) == '"weird""name"'


# --------------------------------------------------------------------------------------
# The reference core (docs/SQLTIER.md §2)
# --------------------------------------------------------------------------------------


def test_the_reference_key_is_ref_1_and_the_sql_names_it_as_a_bare_table() -> None:
    compiled = compile_sql(nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id"))))

    assert list(compiled.query.static_query_references) == ["ref_1"]
    assert "FROM ref_1" in compiled.query.user_edited_sql


def test_the_reference_projects_only_the_fields_the_sql_names_in_first_use_order() -> None:
    compiled = compile_sql(
        nodes.Filter(
            nodes.Aggregate(SCAN, columns(STATE), columns(F.sum(PRICE).alias("total"))),
            (F.col("total") > 1).expr,
        )
    )

    assert reference_of(compiled).fields == (STATE, PRICE)


def test_the_reference_is_unlimited_even_when_the_outer_select_is_not() -> None:
    """The outer LIMIT caps the *answer*; capping the input would change it."""
    compiled = compile_sql(
        nodes.Limit(nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id"))), 5)
    )

    assert reference_of(compiled).limit is None
    assert compiled.query.effective_limit == 5
    assert "LIMIT 5" in compiled.query.user_edited_sql


def test_a_tier_one_expressible_filter_is_pushed_into_the_reference() -> None:
    plan = nodes.Aggregate(
        nodes.Filter(SCAN, (F.col("order_items.status") == "complete").expr),
        columns(STATE),
        columns(F.count("users.id")),
    )
    compiled = compile_sql(plan)

    assert list(reference_of(compiled).filters) == ["order_items.status"]
    assert "WHERE" not in compiled.query.user_edited_sql, "governed filters never become SQL"


def test_a_filter_tier_one_cannot_express_stays_in_the_outer_where() -> None:
    predicate = ((F.col(STATE) == "California") | (F.col(AGE) > 60)).expr
    compiled = compile_sql(nodes.Aggregate(nodes.Filter(SCAN, predicate), columns(STATE), ()))

    assert reference_of(compiled).filters == {}
    assert "WHERE" in compiled.query.user_edited_sql
    assert reference_of(compiled).fields == (STATE, AGE), "the reference widened for the WHERE"


def test_the_two_kinds_of_filter_split_within_one_query() -> None:
    plan = nodes.Aggregate(
        nodes.Filter(
            nodes.Filter(SCAN, (F.col("order_items.status") == "complete").expr),
            ((F.col(STATE) == "California") | (F.col(AGE) > 60)).expr,
        ),
        columns(STATE),
        columns(F.count("users.id")),
    )
    compiled = compile_sql(plan)

    assert list(reference_of(compiled).filters) == ["order_items.status"]
    assert '"users.state" = \'California\' OR "users.age" > 60' in compiled.query.user_edited_sql


def test_a_filter_on_a_group_key_above_the_aggregate_rides_the_where_half() -> None:
    """``GROUP BY k HAVING k = x`` and ``WHERE k = x GROUP BY k`` are the same question."""
    plan = nodes.Filter(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id"))),
        (F.col(STATE) == "California").expr,
    )
    compiled = compile_sql(plan)

    assert list(reference_of(compiled).filters) == [STATE]
    assert "HAVING" not in compiled.query.user_edited_sql


def test_the_reference_query_is_a_governed_tier_one_query() -> None:
    compiled = compile_sql(nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id"))))
    reference = reference_of(compiled)

    assert reference.join_paths_from_topic_name == BENCH_TOPIC_NAME
    assert reference.table == "order_items"
    assert reference.user_edited_sql == ""
    assert reference.to_reference_wire()["model_id"] == BENCH_MODEL_ID


# --------------------------------------------------------------------------------------
# Shapes tier 2 accepts
# --------------------------------------------------------------------------------------


def test_an_alias_becomes_the_select_alias_and_the_group_by_stays_the_expression() -> None:
    plan = nodes.Aggregate(
        SCAN, columns(F.col(STATE).alias("state")), columns(F.count("users.id").alias("n"))
    )
    statement = sql(plan)

    assert '"users.state" AS "state"' in statement
    assert 'GROUP BY\n  "users.state"' in statement, "ANSI GROUP BY cannot see a SELECT alias"


def test_an_unaliased_column_is_not_aliased_to_itself() -> None:
    assert 'AS "users.state"' not in sql(nodes.Aggregate(SCAN, columns(STATE), ()))


def test_having_substitutes_the_full_aggregate_for_its_alias() -> None:
    plan = nodes.Filter(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count_distinct("users.id").alias("b"))),
        (F.col("b") > 25).expr,
    )
    statement = sql(plan)

    assert 'HAVING\n  COUNT(DISTINCT "users.id") > 25' in statement
    assert 'HAVING\n  "b"' not in statement


def test_a_select_with_an_ad_hoc_aggregation_is_the_same_group_by() -> None:
    """Selection IS the group-by, and tier 2 keeps the user's written column order."""
    project = nodes.Project(SCAN, columns(F.count("users.id").alias("n"), STATE))
    statement = sql(project)

    assert statement.index('AS "n"') < statement.index('"users.state"\nFROM')
    assert compile_sql(project).columns == ("n", STATE)


def test_a_group_less_aggregate_emits_no_group_by() -> None:
    assert "GROUP BY" not in sql(nodes.Aggregate(SCAN, (), columns(F.sum(AGE))))


def test_a_sort_targets_the_output_alias_and_puts_nulls_last() -> None:
    plan = nodes.Sort(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n"))),
        (F.col("n").desc().to_sort_key(), F.col(STATE).to_sort_key()),
    )
    statement = sql(plan)

    assert 'ORDER BY\n  "n" DESC,\n  "users.state" ASC NULLS LAST' in statement


def test_the_sort_summary_names_the_output_columns() -> None:
    plan = nodes.Sort(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n"))),
        (F.col("n").desc().to_sort_key(),),
    )

    assert compile_sql(plan).sorts == (("n", True),)


def test_a_computed_column_becomes_a_select_expression() -> None:
    plan = nodes.WithColumn(
        selected(PRICE, "order_items.quantity"),
        "unit",
        (F.col(PRICE) / F.col("order_items.quantity")).expr,
    )
    statement = sql(plan)

    assert '"order_items.sale_price" / "order_items.quantity" AS "unit"' in statement
    assert compile_sql(plan).columns == (PRICE, "order_items.quantity", "unit")


def test_a_computed_column_replaces_the_one_it_shadows() -> None:
    plan = nodes.WithColumn(selected(STATE, AGE), AGE, (F.col(AGE) * 2).expr)

    assert compile_sql(plan).columns == (STATE, AGE)
    assert '"users.age" * 2 AS "users.age"' in sql(plan)


def test_a_derived_column_over_an_aggregate_output_substitutes_the_aggregate() -> None:
    plan = nodes.WithColumn(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n"))),
        "doubled",
        (F.col("n") * 2).expr,
    )

    assert 'COUNT("users.id") * 2 AS "doubled"' in sql(plan)


# --------------------------------------------------------------------------------------
# Shapes tier 2 refuses (docs/SQLTIER.md §1) — every one of them falls through to tier 3
# --------------------------------------------------------------------------------------


def test_a_raw_sql_scan_is_never_re_derived() -> None:
    assert try_sql(nodes.Project(SQL_SCAN, columns("x"))) is None


def test_a_stored_query_scan_is_never_re_derived() -> None:
    assert try_sql(nodes.Project(STORED_SCAN, columns(STATE))) is None


def test_a_governed_measure_keeps_the_aggregate_out_of_tier_two() -> None:
    plan = nodes.Aggregate(SCAN, columns(STATE), columns(F.measure(REVENUE), F.count("users.id")))

    assert try_sql(plan) is None


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
    predicate = ((F.col("order_items.created_at") >= literal) | (F.col(AGE) > 60)).expr

    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is None


def test_a_relative_date_literal_is_refused_next_to_equals_too() -> None:
    """The grammar is the grammar whichever operator it sits beside — ``==`` was exempt."""
    predicate = ((F.col("order_items.created_at") == "today") | (F.col(AGE) > 60)).expr

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
    """A grain pins the type, so the string is a DATE literal — tier 1 compiles it as one.

    Rendering it as SQL text sends ``'2026-03'`` to the warehouse as a string: a conversion error
    on DuckDB/Snowflake, and for ``between()`` a *different row set* on a coercing dialect, since
    tier 1's date BETWEEN is half-open while the SQL rendering would be inclusive.
    """
    month = F.col("order_items.created_at").grain("month")
    predicate = (build(month) | (F.col(AGE) > 60)).expr

    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is None


def test_a_grained_field_against_a_real_date_still_compiles() -> None:
    """Only the *string* form is ambiguous; a date literal renders as a CAST like any other."""
    month = F.col("order_items.created_at").grain("month")
    predicate = ((month >= date(2026, 3, 1)) | (F.col(AGE) > 60)).expr

    assert try_sql(nodes.Filter(selected(STATE, AGE), predicate)) is not None


def test_an_absolute_date_literal_is_not_mistaken_for_a_relative_one() -> None:
    predicate = ((F.col("order_items.created_at") >= date(2026, 1, 1)) | (F.col(AGE) > 60)).expr

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
    """Not tier 2's to reinterpret: moving it below the GROUP BY answers a different question.

    The splitter reports it by name instead (``users.age is not available after group_by``), and
    it can only do that if tier 2 declines rather than quietly rewriting it as a WHERE.
    """
    plan = nodes.Filter(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n"))),
        (F.col(AGE) * 2 > 1).expr,
    )

    assert try_sql(plan) is None


def test_a_having_that_also_names_a_consumed_column_is_refused() -> None:
    """An aggregate somewhere in the predicate does not license the rest of it.

    ``(n > 10) OR (users.age = 30)`` is a single conjunct, so without this guard ``users.age``
    rode into the HAVING naming a column that is neither grouped nor aggregated — a binder error
    on DuckDB/Postgres/Snowflake and an arbitrary-row answer on a permissive dialect.  Declining
    is always correct (docs/SQLTIER.md §5); the splitter then reports it by name.
    """
    aggregate = nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n")))

    assert try_sql(nodes.Filter(aggregate, ((F.col("n") > 10) | (F.col(AGE) == 30)).expr)) is None
    assert try_sql(nodes.Filter(aggregate, (F.col("n") > F.col(AGE)).expr)) is None
    # The legitimate shape — a group key beside the aggregate — still compiles.
    legal = nodes.Filter(aggregate, ((F.col("n") > 10) | (F.col(STATE) == "California")).expr)
    compiled = try_sql(legal)
    assert compiled is not None
    assert "HAVING" in compiled.query.user_edited_sql


def test_a_derived_column_above_an_aggregate_over_a_consumed_column_is_refused() -> None:
    plan = nodes.WithColumn(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n"))),
        "doubled",
        (F.col(AGE) * 2).expr,
    )

    assert try_sql(plan) is None


def test_replacing_a_group_key_with_a_derived_column_is_refused() -> None:
    """The GROUP BY would name a column the SELECT no longer produces."""
    plan = nodes.WithColumn(
        nodes.Aggregate(SCAN, columns(AGE), columns(F.count("users.id").alias("n"))),
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
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id"))),
        (F.col(AGE).to_sort_key(),),
    )

    assert try_sql(plan) is None


# --------------------------------------------------------------------------------------
# The carrier and the dialect knob
# --------------------------------------------------------------------------------------


def test_a_tier_two_compilation_fills_the_semantic_carrier() -> None:
    """``RemoteStep`` reads only these attributes, which is why tier 2 needs no new class."""
    compiled = compile_sql(
        nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n")))
    )

    assert compiled.tier == 2
    assert compiled.role == "sql"
    assert compiled.is_sql is True
    assert compiled.opaque is False, "omniframes wrote this SQL; it was not handed it"
    assert compiled.aliases == {}
    assert compiled.columns == (STATE, "n")
    assert compiled.group_keys == (STATE,)
    assert compiled.measures == ()
    assert compiled.measure_filters == ()
    assert compiled.scan is SCAN.source


def test_the_default_dialect_is_ansi_ish_and_the_knob_switches_it() -> None:
    plan = nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id").alias("n")))

    assert '"users.state"' in sql(plan)
    assert "`users.state`" in sql(plan, sql_dialect="bigquery")


def test_the_statement_is_pretty_printed_so_the_golden_files_stay_readable() -> None:
    assert "\n" in sql(nodes.Aggregate(SCAN, columns(STATE), columns(F.count("users.id"))))


def test_compiling_the_same_plan_twice_gives_byte_identical_sql() -> None:
    plan = nodes.Aggregate(SCAN, columns(STATE, AGE), columns(F.sum(PRICE), F.count("users.id")))

    assert sql(plan) == sql(plan), "determinism is what makes the golden files reviewable"
