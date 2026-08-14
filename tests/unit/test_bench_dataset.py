"""Invariant tests for the checked-in bench dataset.

Every invariant listed in ``docs/BENCH_DATASET.md`` § "Invariants tests may rely on" is asserted
here against the artifacts in ``tests/data/bench/`` — the other lanes (differential, wire, golden,
live) build on these guarantees, so a violation here should fail loudly and early.

The generator lives at ``tools/bench/generate.py``; it is loaded by path (``tools/`` is not an
importable package) for the determinism and freshness checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import unicodedata
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "tests" / "data" / "bench"
GENERATOR_PATH = REPO_ROOT / "tools" / "bench" / "generate.py"

ANCHOR_DATE = date(2026, 6, 30)
EXPECTED_ROW_COUNTS = {"users": 500, "products": 200, "order_items": 10_000}
DATA_FILES = (
    "known_answers.json",
    "order_items.csv",
    "order_items.parquet",
    "products.csv",
    "products.parquet",
    "users.csv",
    "users.parquet",
)
REQUIRED_ANSWER_KEYS = (
    "grand_totals",
    "revenue_by_state",
    "monthly_revenue_trailing_12m",
    "revenue_by_category_month",
    "distinct_buyers_by_month",
    "revenue_and_buyers_by_state",
)
# Columns whose NULL share the spec guarantees is non-zero at scale=1.
NULLABLE_COLUMNS = (
    ("users", "state"),
    ("users", "age"),
    ("users", "is_business"),
    ("products", "category"),
    ("order_items", "discount"),
    ("order_items", "returned"),
    ("order_items", "notes"),
)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_generate", GENERATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tables() -> dict[str, pa.Table]:
    return {name: pq.read_table(BENCH_DIR / f"{name}.parquet") for name in EXPECTED_ROW_COUNTS}


@pytest.fixture(scope="module")
def users(tables: dict[str, pa.Table]) -> pa.Table:
    return tables["users"]


@pytest.fixture(scope="module")
def products(tables: dict[str, pa.Table]) -> pa.Table:
    return tables["products"]


@pytest.fixture(scope="module")
def order_items(tables: dict[str, pa.Table]) -> pa.Table:
    return tables["order_items"]


@pytest.fixture(scope="module")
def known_answers() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((BENCH_DIR / "known_answers.json").read_text("utf-8"))
    return payload


@pytest.fixture(scope="module")
def regenerated(
    generator: ModuleType, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    """Two independent regenerations at scale=1, for the determinism/freshness checks."""
    first = tmp_path_factory.mktemp("bench_first")
    second = tmp_path_factory.mktemp("bench_second")
    generator.generate(first, scale=1)
    generator.generate(second, scale=1)
    return first, second


def _column(table: pa.Table, name: str) -> list[Any]:
    """Arrow column -> Python list. Typed as list[Any] so comparisons stay readable."""
    return cast("list[Any]", table.column(name).to_pylist())


def _read_bench_csv(path: Path, **convert: Any) -> pa.Table:
    """order_items.notes embeds newlines, so the parser must be told to expect them."""
    return pacsv.read_csv(
        path,
        parse_options=pacsv.ParseOptions(newlines_in_values=True),
        convert_options=pacsv.ConvertOptions(**convert) if convert else None,
    )


# --------------------------------------------------------------------------------------
# Presence, row counts
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("filename", DATA_FILES)
def test_artifact_exists(filename: str) -> None:
    path = BENCH_DIR / filename
    assert path.is_file(), f"missing bench artifact: {path}"
    assert path.stat().st_size > 0


@pytest.mark.parametrize(("name", "expected"), sorted(EXPECTED_ROW_COUNTS.items()))
def test_parquet_row_counts(tables: dict[str, pa.Table], name: str, expected: int) -> None:
    assert tables[name].num_rows == expected


@pytest.mark.parametrize(("name", "expected"), sorted(EXPECTED_ROW_COUNTS.items()))
def test_csv_row_counts_match_parquet(name: str, expected: int) -> None:
    table = _read_bench_csv(BENCH_DIR / f"{name}.csv")
    assert table.num_rows == expected
    assert table.schema.names == pq.read_schema(BENCH_DIR / f"{name}.parquet").names


# --------------------------------------------------------------------------------------
# Schema / Arrow types
# --------------------------------------------------------------------------------------


def test_users_schema(users: pa.Table) -> None:
    schema = users.schema
    assert schema.names == [
        "id",
        "name",
        "email",
        "state",
        "country",
        "created_at",
        "age",
        "is_business",
        "signup_source",
        "lifetime_value",
    ]
    assert schema.field("id").type == pa.int64()
    assert schema.field("name").type == pa.string()
    assert schema.field("age").type == pa.int64()
    assert schema.field("is_business").type == pa.bool_()
    assert schema.field("signup_source").type == pa.string()


def test_products_schema(products: pa.Table) -> None:
    schema = products.schema
    assert schema.names == [
        "id",
        "name",
        "category",
        "brand",
        "cost",
        "price",
        "introduced_on",
    ]
    assert schema.field("introduced_on").type == pa.date32(), (
        "introduced_on must be a date, not a timestamp"
    )


def test_order_items_schema(order_items: pa.Table) -> None:
    schema = order_items.schema
    assert schema.names == [
        "id",
        "order_id",
        "user_id",
        "product_id",
        "created_at",
        "status",
        "quantity",
        "sale_price",
        "discount",
        "returned",
        "notes",
    ]
    assert schema.field("quantity").type == pa.int64()
    assert schema.field("returned").type == pa.bool_()
    assert schema.field("notes").type == pa.large_string(), "notes must be large_string"


@pytest.mark.parametrize(
    ("table_name", "column", "precision", "scale"),
    [
        ("users", "lifetime_value", 12, 2),
        ("products", "cost", 10, 2),
        ("products", "price", 10, 2),
        ("order_items", "sale_price", 12, 2),
        ("order_items", "discount", 12, 2),
    ],
)
def test_decimal_columns_are_decimal128(
    tables: dict[str, pa.Table], table_name: str, column: str, precision: int, scale: int
) -> None:
    field_type = tables[table_name].schema.field(column).type
    assert pa.types.is_decimal128(field_type), f"{table_name}.{column} is {field_type}"
    assert (field_type.precision, field_type.scale) == (precision, scale)
    values = _column(tables[table_name], column)
    assert any(v is not None for v in values)
    assert all(isinstance(v, Decimal) for v in values if v is not None)


@pytest.mark.parametrize("table_name", ["users", "order_items"])
def test_timestamps_are_tz_aware_utc(tables: dict[str, pa.Table], table_name: str) -> None:
    field_type = tables[table_name].schema.field("created_at").type
    assert pa.types.is_timestamp(field_type)
    assert field_type.unit == "us"
    assert field_type.tz == "UTC"
    values = _column(tables[table_name], "created_at")
    assert all(isinstance(v, datetime) for v in values)
    assert all(v.tzinfo is not None for v in values)
    assert all(v.utcoffset() == timedelta(0) for v in values)


# --------------------------------------------------------------------------------------
# NULL / empty-string edge cases
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("table_name", "column"), NULLABLE_COLUMNS)
def test_nullable_columns_have_nulls(
    tables: dict[str, pa.Table], table_name: str, column: str
) -> None:
    table = tables[table_name]
    null_count = table.column(column).null_count
    assert null_count > 0, f"{table_name}.{column} has no NULLs"
    assert null_count < table.num_rows, f"{table_name}.{column} is entirely NULL"


@pytest.mark.parametrize(("table_name", "column"), NULLABLE_COLUMNS)
def test_nullable_column_share_matches_spec(
    tables: dict[str, pa.Table], table_name: str, column: str
) -> None:
    expected = {
        ("users", "state"): 0.06,
        ("users", "age"): 0.08,
        ("users", "is_business"): 0.05,
        ("products", "category"): 0.04,
        ("order_items", "discount"): 0.70,
        ("order_items", "returned"): 0.03,
        ("order_items", "notes"): 0.90,
    }[table_name, column]
    table = tables[table_name]
    share = table.column(column).null_count / table.num_rows
    assert share == pytest.approx(expected, abs=0.01)


def test_non_nullable_columns_have_no_nulls(tables: dict[str, pa.Table]) -> None:
    required = {
        "users": [
            "id",
            "name",
            "email",
            "country",
            "created_at",
            "signup_source",
            "lifetime_value",
        ],
        "products": ["id", "name", "brand", "cost", "price", "introduced_on"],
        "order_items": [
            "id",
            "order_id",
            "user_id",
            "product_id",
            "created_at",
            "status",
            "quantity",
            "sale_price",
        ],
    }
    for table_name, columns in required.items():
        for column in columns:
            assert tables[table_name].column(column).null_count == 0, f"{table_name}.{column}"


def test_signup_source_has_empty_strings_that_are_not_null(users: pa.Table) -> None:
    values = _column(users, "signup_source")
    assert None not in values, "signup_source models empty-string, not NULL"
    assert values.count("") > 0, "signup_source must contain empty strings"
    assert len({v for v in values if v}) > 1, "signup_source should stay low-cardinality but > 1"


def test_csv_preserves_empty_string_versus_null() -> None:
    """The CSV encoding must keep `""` (empty) distinct from an unquoted NULL."""
    raw = (BENCH_DIR / "users.csv").read_text("utf-8")
    header, first_row = raw.split("\n", 2)[:2]
    assert header.startswith('"id","name"'), header
    assert '""' in raw, "no quoted empty string in users.csv"

    table = _read_bench_csv(
        BENCH_DIR / "users.csv", strings_can_be_null=True, quoted_strings_can_be_null=False
    )
    signup_source = _column(table, "signup_source")
    assert None not in signup_source
    assert signup_source.count("") > 0
    assert table.column("state").null_count > 0
    assert "" not in _column(table, "state")
    assert first_row  # sanity: the file has at least one data row


# --------------------------------------------------------------------------------------
# Referential integrity & value ranges
# --------------------------------------------------------------------------------------


def test_orphan_user_ids_exist(users: pa.Table, order_items: pa.Table) -> None:
    known = set(_column(users, "id"))
    referenced = _column(order_items, "user_id")
    orphans = [uid for uid in referenced if uid not in known]
    assert orphans, "the fact table must contain orphan user_ids for join-edge testing"
    assert len(orphans) < order_items.num_rows * 0.05, "orphans should stay a small minority"
    assert set(_column(users, "id")) == set(range(1, users.num_rows + 1))


def test_every_product_id_resolves(products: pa.Table, order_items: pa.Table) -> None:
    known = set(_column(products, "id"))
    referenced = set(_column(order_items, "product_id"))
    assert referenced <= known
    assert referenced == known, "every product should appear in the fact table"


def test_order_ids_duplicate_by_design(order_items: pa.Table) -> None:
    order_ids = _column(order_items, "order_id")
    distinct_orders = len(set(order_ids))
    assert distinct_orders < order_items.num_rows, "order_id must repeat across items"
    items_per_order = order_items.num_rows / distinct_orders
    assert items_per_order == pytest.approx(2.6, abs=0.3)


def test_sale_price_and_discount_bounds(order_items: pa.Table) -> None:
    sale_prices = _column(order_items, "sale_price")
    discounts = _column(order_items, "discount")
    assert all(v > 0 for v in sale_prices), "sale_price must be strictly positive"
    pairs = [(s, d) for s, d in zip(sale_prices, discounts, strict=True) if d is not None]
    assert pairs, "discount must be present on some rows"
    assert all(d >= 0 for _, d in pairs), "discount must be non-negative where present"
    assert all(d < s for s, d in pairs), "discount must be strictly less than sale_price"


def test_quantity_range(order_items: pa.Table) -> None:
    quantities = _column(order_items, "quantity")
    assert min(quantities) == 1
    assert max(quantities) == 8


def test_status_enum(order_items: pa.Table) -> None:
    assert set(_column(order_items, "status")) == {
        "complete",
        "shipped",
        "processing",
        "cancelled",
        "returned",
    }


def test_some_products_have_negative_margin(products: pa.Table) -> None:
    costs = _column(products, "cost")
    prices = _column(products, "price")
    assert any(p < c for c, p in zip(costs, prices, strict=True)), "no negative-margin products"
    assert all(p > 0 and c > 0 for c, p in zip(costs, prices, strict=True))


def test_brands_are_duplicated_across_categories(products: pa.Table) -> None:
    brands = _column(products, "brand")
    categories = _column(products, "category")
    per_brand: dict[str, set[str | None]] = {}
    for brand, category in zip(brands, categories, strict=True):
        per_brand.setdefault(brand, set()).add(category)
    assert any(len(cats) > 1 for cats in per_brand.values())


# --------------------------------------------------------------------------------------
# Time windows
# --------------------------------------------------------------------------------------


def test_fact_created_at_max_is_the_anchor_date(order_items: pa.Table) -> None:
    values = _column(order_items, "created_at")
    latest = max(values)
    assert latest.astimezone(UTC).date() == ANCHOR_DATE
    assert all(v.astimezone(UTC).date() <= ANCHOR_DATE for v in values)


def test_fact_created_at_spans_two_years(order_items: pa.Table) -> None:
    values = [v.astimezone(UTC) for v in _column(order_items, "created_at")]
    span_days = (max(values) - min(values)).days
    assert 700 <= span_days <= 731


def test_users_created_at_spans_five_years_ending_at_the_anchor(users: pa.Table) -> None:
    values = [v.astimezone(UTC) for v in _column(users, "created_at")]
    assert max(values).date() == ANCHOR_DATE
    span_days = (max(values) - min(values)).days
    assert 1_700 <= span_days <= 1_826


def test_introduced_on_values_are_dates(products: pa.Table) -> None:
    values = _column(products, "introduced_on")
    assert all(isinstance(v, date) and not isinstance(v, datetime) for v in values)
    assert max(values) <= ANCHOR_DATE


# --------------------------------------------------------------------------------------
# Unicode
# --------------------------------------------------------------------------------------


def _has_cjk(value: str) -> bool:
    return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in value)


def _has_emoji(value: str) -> bool:
    return any(ord(ch) >= 0x1F300 for ch in value)


def _has_latin_accent(value: str) -> bool:
    return any(
        0x00C0 <= ord(ch) < 0x0250 and len(unicodedata.normalize("NFD", ch)) > 1 for ch in value
    )


def test_user_names_include_unicode(users: pa.Table) -> None:
    names = _column(users, "name")
    assert any(_has_cjk(n) for n in names), "no CJK user names"
    assert any(_has_emoji(n) for n in names), "no emoji user names"
    assert any(_has_latin_accent(n) for n in names), "no accented user names"
    assert any(not n.isascii() and not _has_cjk(n) for n in names), "no non-CJK non-ASCII names"


def test_product_names_include_unicode(products: pa.Table) -> None:
    names = _column(products, "name")
    assert any(not n.isascii() for n in names)
    assert any(_has_emoji(n) for n in names), "no emoji product names"


def test_notes_contain_newlines_quotes_and_multi_kb_text(order_items: pa.Table) -> None:
    notes = [n for n in _column(order_items, "notes") if n is not None]
    assert notes
    assert any("\n" in n for n in notes), "notes must include embedded newlines"
    assert any('"' in n for n in notes), "notes must include embedded double quotes"
    assert any(len(n.encode("utf-8")) > 2_048 for n in notes), "notes must include multi-KB text"
    assert any(not n.isascii() for n in notes), "notes should include unicode"


# --------------------------------------------------------------------------------------
# known_answers.json
# --------------------------------------------------------------------------------------


def test_known_answers_metadata(known_answers: dict[str, Any]) -> None:
    assert known_answers["anchor_date"] == ANCHOR_DATE.isoformat()
    assert known_answers["seed"] == 42
    assert known_answers["scale"] == 1
    assert isinstance(known_answers["bench_version"], str)
    assert known_answers["row_counts"] == EXPECTED_ROW_COUNTS


@pytest.mark.parametrize("key", REQUIRED_ANSWER_KEYS)
def test_known_answers_contains_required_key(known_answers: dict[str, Any], key: str) -> None:
    entry = known_answers["answers"][key]
    assert isinstance(entry["description"], str)
    assert entry["description"].strip()
    assert entry["columns"], f"{key} has no columns"
    assert entry["rows"], f"{key} has no rows"
    assert entry["row_count"] == len(entry["rows"])
    assert all(len(row) == len(entry["columns"]) for row in entry["rows"])


def test_known_answers_revenue_by_state_includes_null_group(known_answers: dict[str, Any]) -> None:
    entry = known_answers["answers"]["revenue_by_state"]
    states = [row[entry["columns"].index("state")] for row in entry["rows"]]
    assert None in states, "revenue_by_state must keep the NULL state group"
    assert len(states) == len(set(states))


def test_known_answers_state_revenue_sums_to_the_grand_total(known_answers: dict[str, Any]) -> None:
    grand = known_answers["answers"]["grand_totals"]
    by_state = known_answers["answers"]["revenue_by_state"]
    total = Decimal(grand["rows"][0][grand["columns"].index("total_sale_price")])
    state_column = by_state["columns"].index("total_sale_price")
    assert sum((Decimal(row[state_column]) for row in by_state["rows"]), Decimal(0)) == total


def test_known_answers_monthly_window_is_the_trailing_twelve_months(
    known_answers: dict[str, Any],
) -> None:
    entry = known_answers["answers"]["monthly_revenue_trailing_12m"]
    months = [row[entry["columns"].index("month")] for row in entry["rows"]]
    assert len(months) == 12
    assert months == sorted(months)
    assert months[0].startswith("2025-07-01")
    assert months[-1].startswith("2026-06-01")
    assert all(m.endswith("+00:00") for m in months), "months must be UTC-qualified"


def test_known_answers_decimal_values_are_strings(known_answers: dict[str, Any]) -> None:
    """Decimals are serialized as strings so no float rounding creeps into ground truth."""
    grand = known_answers["answers"]["grand_totals"]
    revenue = grand["rows"][0][grand["columns"].index("total_sale_price")]
    assert isinstance(revenue, str)
    assert Decimal(revenue) > 0
    for entry in known_answers["answers"].values():
        for row in entry["rows"]:
            assert not any(isinstance(v, float) for v in row), entry["description"]


def test_known_answers_mixed_aggregation_case(known_answers: dict[str, Any]) -> None:
    entry = known_answers["answers"]["revenue_and_buyers_by_state"]
    assert entry["columns"] == ["state", "total_sale_price", "distinct_buyers"]
    assert any(row[0] is None for row in entry["rows"])
    assert all(isinstance(row[2], int) and row[2] > 0 for row in entry["rows"])


# --------------------------------------------------------------------------------------
# Determinism & freshness
# --------------------------------------------------------------------------------------


def _digests(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_regeneration_is_byte_identical(regenerated: tuple[Path, Path]) -> None:
    first, second = regenerated
    digests = _digests(first)
    assert set(digests) == set(DATA_FILES)
    assert digests == _digests(second)


def test_checked_in_tables_match_a_fresh_generation(
    regenerated: tuple[Path, Path], tables: dict[str, pa.Table]
) -> None:
    fresh_dir, _ = regenerated
    for name, table in tables.items():
        fresh = pq.read_table(fresh_dir / f"{name}.parquet")
        assert fresh.schema.equals(table.schema), f"{name} schema drifted from the generator"
        assert fresh.equals(table), f"{name}.parquet is stale — rerun tools/bench/generate.py"


def test_checked_in_known_answers_match_a_fresh_generation(
    regenerated: tuple[Path, Path], known_answers: dict[str, Any]
) -> None:
    fresh_dir, _ = regenerated
    fresh = json.loads((fresh_dir / "known_answers.json").read_text("utf-8"))
    assert fresh == known_answers, "known_answers.json is stale — rerun tools/bench/generate.py"


def test_scale_only_grows_the_fact_table(
    generator: ModuleType, tables: dict[str, pa.Table]
) -> None:
    """`--scale N` multiplies the fact table; the dimensions must stay byte-for-byte the same."""
    scaled = generator.build_tables(scale=2)
    assert scaled["users"].equals(tables["users"])
    assert scaled["products"].equals(tables["products"])
    assert scaled["order_items"].num_rows == EXPECTED_ROW_COUNTS["order_items"] * 2
    assert scaled["order_items"].schema.equals(tables["order_items"].schema)


def test_generator_never_reads_the_clock() -> None:
    source = GENERATOR_PATH.read_text("utf-8")
    forbidden = re.compile(
        r"\b(?:datetime\.now|date\.today|datetime\.today|datetime\.utcnow|time\.time"
        r"|time\.monotonic|Timestamp\.now)\b"
    )
    assert not forbidden.search(source), "the bench generator must never read the wall clock"
    assert "default_rng(SEED)" in source
    assert "SEED: Final = 42" in source
