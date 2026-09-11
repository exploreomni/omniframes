"""Deterministic generator for the omniframes bench dataset.

Spec: ``internal-docs/BENCH_DATASET.md``. One synthetic ecommerce dataset backs every test lane
(FakeOmniAPI tables, the differential lane's ground truth, the demo notebook) and is meant to be
loadable into a real warehouse behind a real Omni model.

Everything here is a pure function of ``SEED`` and ``ANCHOR_DATE`` — the generator NEVER reads the
wall clock. Arrow/Parquet/CSV writer options are pinned so that regenerating at the same
``BENCH_VERSION`` produces byte-identical files.

Usage::

    uv run python tools/bench/generate.py                 # scale=1 into tests/data/bench/
    uv run python tools/bench/generate.py --scale 25      # bigger fact table (dims unchanged)
    uv run python tools/bench/generate.py --out /tmp/x    # somewhere else
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# --------------------------------------------------------------------------------------
# Constants — any change here bumps BENCH_VERSION and regenerates known_answers.json.
# --------------------------------------------------------------------------------------

BENCH_VERSION: Final = "1"
SEED: Final = 42
ANCHOR_DATE: Final = date(2026, 6, 30)

N_USERS: Final = 500
N_PRODUCTS: Final = 200
N_ORDER_ITEMS: Final = 10_000

USER_HISTORY_DAYS: Final = 5 * 365  # users.created_at spans 5 years ending at the anchor
FACT_HISTORY_DAYS: Final = 730  # order_items.created_at spans 2 years ending at the anchor

ORPHAN_USER_ID_BASE: Final = 900_000  # user_ids with no matching users.id row

# Exact null counts at scale=1 (kept exact so "non-zero nulls" is a guarantee, not a probability).
USERS_NULL_STATE: Final = 30  # 6%
USERS_NULL_AGE: Final = 40  # 8%
USERS_NULL_IS_BUSINESS: Final = 25  # 5%
USERS_EMPTY_SIGNUP_SOURCE: Final = 22  # "" is not NULL
PRODUCTS_NULL_CATEGORY: Final = 8  # 4%
PRODUCTS_NEGATIVE_MARGIN: Final = 14  # price < cost
FACT_NULL_DISCOUNT_FRACTION: Final = 0.70
FACT_NULL_RETURNED_FRACTION: Final = 0.03
FACT_NULL_NOTES_FRACTION: Final = 0.90
FACT_ORPHAN_ORDER_FRACTION: Final = 0.01
FACT_LONG_NOTE_FRACTION: Final = 0.04  # share of the non-null notes that are multi-KB

TABLE_NAMES: Final = ("users", "products", "order_items")

# Pinned writer options — determinism depends on these staying put.
PARQUET_WRITE_OPTIONS: Final[dict[str, Any]] = {
    "version": "2.6",
    "compression": "zstd",
    "compression_level": 3,
    "use_dictionary": False,
    "write_statistics": True,
    "data_page_size": 1 << 20,
    "write_batch_size": 1024,
    "store_schema": True,
    "write_page_index": False,
    "coerce_timestamps": None,
}
CSV_WRITE_OPTIONS: Final[dict[str, Any]] = {
    "include_header": True,
    "batch_size": 1024,
    "delimiter": ",",
    "quoting_style": "all_valid",  # NULL -> bare empty, "" -> quoted empty
}

# --------------------------------------------------------------------------------------
# Value pools
# --------------------------------------------------------------------------------------

# Unicode-heavy names: CJK, kana, hangul, Cyrillic, Greek, Arabic, Hebrew, accents, emoji.
UNICODE_NAMES: Final = (
    "李伟",
    "王芳",
    "张敏",
    "陈静",
    "佐藤 さくら",
    "鈴木 太郎",
    "김지훈",
    "박서연",
    "Дмитрий Иванов",
    "Ольга Смирнова",
    "Γιώργος Παπαδόπουλος",
    "محمد الأحمد",
    "פרימו לוי",
    "José Álvarez",
    "Renée Dubois",
    "Søren Kjærgaard",
    "Zoë Müller",
    "Ángel Peña",
    "Þóra Jónsdóttir",
    "Nguyễn Thị Hương",
    "Ayşe Çelik",
    "Łukasz Wójcik",
    "🚀 Rocket Fields",
    "Mina 🌸 Okada",
    "Élodie 🎉 Laurent",
)
FIRST_NAMES: Final = (
    "Ada",
    "Bram",
    "Cleo",
    "Dax",
    "Elena",
    "Finn",
    "Greta",
    "Hugo",
    "Iris",
    "Jonas",
    "Kira",
    "Liam",
    "Maya",
    "Noor",
    "Otto",
    "Pia",
    "Quinn",
    "Rosa",
    "Silas",
    "Tessa",
    "Ulf",
    "Vera",
    "Wren",
    "Xander",
    "Yara",
    "Zeke",
)
LAST_NAMES: Final = (
    "Abbott",
    "Bergman",
    "Castillo",
    "Devlin",
    "Eriksen",
    "Farrow",
    "Gallagher",
    "Hollis",
    "Ingram",
    "Jarvis",
    "Kowalski",
    "Lindqvist",
    "Marchetti",
    "Nakamura",
    "Oyelaran",
    "Petrova",
    "Quintero",
    "Rasmussen",
    "Sandoval",
    "Thibault",
    "Uddin",
    "Vasquez",
    "Whitfield",
    "Yoshida",
    "Zimmer",
)
EMAIL_DOMAINS: Final = ("example.com", "example.org", "example.net", "test.example")

US_STATES: Final = (
    "Arizona",
    "California",
    "Colorado",
    "Florida",
    "Georgia",
    "Illinois",
    "Indiana",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "New Jersey",
    "New York",
    "North Carolina",
    "Ohio",
    "Oregon",
    "Pennsylvania",
    "Tennessee",
    "Texas",
    "Virginia",
    "Washington",
)
STATE_WEIGHTS: Final = (
    0.04,
    0.14,
    0.03,
    0.08,
    0.04,
    0.06,
    0.03,
    0.04,
    0.04,
    0.03,
    0.04,
    0.10,
    0.04,
    0.05,
    0.03,
    0.05,
    0.03,
    0.09,
    0.02,
    0.02,
)
COUNTRIES: Final = (
    "United States",
    "Canada",
    "United Kingdom",
    "Germany",
    "France",
    "Japan",
    "Brazil",
    "Australia",
)
COUNTRY_WEIGHTS: Final = (0.62, 0.09, 0.08, 0.05, 0.04, 0.05, 0.04, 0.03)

SIGNUP_SOURCES: Final = ("organic", "paid_search", "email", "referral", "social", "partner")
SIGNUP_SOURCE_WEIGHTS: Final = (0.34, 0.22, 0.16, 0.12, 0.10, 0.06)

CATEGORIES: Final = (
    "Accessories",
    "Blankets",
    "Fitness",
    "Home Office",
    "Kitchen",
    "Outdoor",
    "Pet Supplies",
    "Stationery",
    "Tools",
    "Travel",
)
CATEGORY_WEIGHTS: Final = (0.14, 0.06, 0.10, 0.12, 0.13, 0.09, 0.08, 0.11, 0.09, 0.08)
BRANDS: Final = (
    "Alpenglow",
    "Brightwell",
    "Cobalt & Co",
    "Dunemark",
    "Everlark",
    "Fjordly",
    "Grovewright",
    "Halcyon",
    "Ironbark",
    "Junipero",
    "Kestrel",
    "Lumenaut",
)
PRODUCT_NOUNS: Final = (
    "Tote",
    "Mug",
    "Lamp",
    "Throw",
    "Kettle",
    "Notebook",
    "Wrench",
    "Duffel",
    "Leash",
    "Mat",
    "Stand",
    "Bottle",
    "Planner",
    "Cutting Board",
    "Headlamp",
    "Cable Kit",
)
PRODUCT_UNICODE_SUFFIXES: Final = (
    " ✨",
    " 🌿",
    " (限定版)",
    " — Édition Spéciale",
    " 🔧",
    " Ⅱ",
)

STATUSES: Final = ("complete", "shipped", "processing", "cancelled", "returned")
STATUS_WEIGHTS: Final = (0.52, 0.22, 0.12, 0.09, 0.05)

ORDER_SIZES: Final = (1, 2, 3, 4, 5, 6, 7, 8)
ORDER_SIZE_WEIGHTS: Final = (0.36, 0.23, 0.15, 0.10, 0.06, 0.045, 0.035, 0.02)

QUANTITIES: Final = (1, 2, 3, 4, 5, 6, 7, 8)
QUANTITY_WEIGHTS: Final = (0.46, 0.24, 0.13, 0.07, 0.045, 0.03, 0.015, 0.01)

MONTH_SEASONALITY: Final = (
    0.86,  # January
    0.82,
    0.94,
    0.98,
    1.04,
    1.00,
    0.94,
    0.96,
    1.02,
    1.12,
    1.46,
    1.58,  # December
)
WEEKDAY_SEASONALITY: Final = (1.00, 0.97, 0.96, 1.00, 1.12, 1.26, 1.14)  # Mon..Sun

SHORT_NOTES: Final = (
    'Customer wrote: "please gift-wrap this one".\nLeave it at the side door.',
    "Second attempt — first parcel was returned to sender.\nRe-shipped 2 days later.",
    'Support ticket #4821: "arrived scuffed", refund of one unit approved.',
    "Gift order.\nCard reads: “Happy birthday, Zoë!”\nNo invoice in the box, please.",
    'Bulk buyer. Asked about a "net 30" account — routed to sales.',
    "Address had a typo (apt 4B vs 4D).\nCorrected by the courier on the second try.",
    "重要: 客户要求发票开具公司抬头。\nInvoice must be issued to the company.",
    'Fragile. Packer noted: "double-boxed, extra foam".',
    "Return window extended to 60 days as a goodwill gesture.\nApproved by ops.",
    'Customer asked to "hold until Friday" — warehouse honored the request.',
)
LONG_NOTE_PARAGRAPH: Final = (
    "Escalation log entry. The customer contacted support about this line item and the agent "
    'recorded the following verbatim: "the package showed up a day early, which was great, but '
    'the outer carton was crushed on one corner and the box inside had a small tear". '
    "Photographs were attached to the ticket and reviewed by the fulfillment lead.\n"
    "Resolution: a replacement unit was dispatched from the secondary warehouse and the original "
    "unit was written off rather than returned, because the freight cost exceeded the item's "
    "cost basis. The customer was offered a partial credit as an alternative and declined.\n"
    "Follow-up: packaging engineering was asked to review the corner-crush failure mode for this "
    "SKU; the interim mitigation is an extra layer of corrugate on orders shipping to zones 6-8.\n"
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


_F64Array = np.ndarray[Any, np.dtype[np.float64]]
_I64Array = np.ndarray[Any, np.dtype[np.int64]]


def _normalized(weights: Sequence[float] | np.ndarray[Any, np.dtype[Any]]) -> _F64Array:
    arr = np.asarray(weights, dtype=np.float64)
    # np.asarray again: numpy's stubs type `arr / arr.sum()` as Any under Python 3.11.
    return np.asarray(arr / arr.sum(), dtype=np.float64)


def _pick(
    rng: np.random.Generator,
    pool: Sequence[str],
    size: int,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Sample ``size`` values from ``pool`` by index (avoids numpy fixed-width string dtypes)."""
    probabilities = None if weights is None else _normalized(weights)
    idx = rng.choice(len(pool), size=size, p=probabilities)
    return [pool[int(i)] for i in idx]


def _cents(values: Iterable[int]) -> list[Decimal]:
    return [Decimal(int(v)).scaleb(-2) for v in values]


def _to_cents(values: list[Any]) -> _I64Array:
    """Inverse of :func:`_cents`: exact decimal -> integer cents."""
    return np.array([int(Decimal(str(v)).scaleb(2)) for v in values], dtype=np.int64)


def _mask_indices(rng: np.random.Generator, total: int, count: int) -> set[int]:
    """Exactly ``count`` distinct indices in ``[0, total)``."""
    if count <= 0:
        return set()
    chosen = rng.choice(total, size=min(count, total), replace=False)
    return {int(i) for i in chosen}


def _null_out(values: list[Any], indices: set[int]) -> list[Any]:
    return [None if i in indices else v for i, v in enumerate(values)]


# --------------------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------------------


def build_users(rng: np.random.Generator) -> pa.Table:
    """500 rows. Unicode names, NULL state/age/is_business, empty-string signup_source."""
    ids = list(range(1, N_USERS + 1))

    firsts = _pick(rng, FIRST_NAMES, N_USERS)
    lasts = _pick(rng, LAST_NAMES, N_USERS)
    names: list[str] = [f"{f} {ln}" for f, ln in zip(firsts, lasts, strict=True)]
    # Guarantee every unicode class is present, at deterministic positions.
    unicode_slots = sorted(_mask_indices(rng, N_USERS, len(UNICODE_NAMES) * 3))
    for slot, name in zip(unicode_slots, UNICODE_NAMES * 3, strict=True):
        names[slot] = name

    handles = _pick(rng, FIRST_NAMES, N_USERS)
    domains = _pick(rng, EMAIL_DOMAINS, N_USERS)
    emails = [
        f"{handle.lower()}.{uid}@{domain}"
        for handle, uid, domain in zip(handles, ids, domains, strict=True)
    ]

    states: list[Any] = _pick(rng, US_STATES, N_USERS, STATE_WEIGHTS)
    states = _null_out(states, _mask_indices(rng, N_USERS, USERS_NULL_STATE))

    countries = _pick(rng, COUNTRIES, N_USERS, COUNTRY_WEIGHTS)

    day_offsets = rng.integers(0, USER_HISTORY_DAYS, size=N_USERS)
    second_offsets = rng.integers(0, 86_400, size=N_USERS)
    # Guarantee at least one row lands exactly on the anchor date.
    day_offsets[0] = 0
    second_offsets[0] = 0
    user_epoch = datetime(ANCHOR_DATE.year, ANCHOR_DATE.month, ANCHOR_DATE.day, tzinfo=UTC)
    created_at = [
        user_epoch - timedelta(days=int(d), seconds=int(s))
        for d, s in zip(day_offsets, second_offsets, strict=True)
    ]

    ages: list[Any] = [int(a) for a in rng.integers(18, 79, size=N_USERS)]
    ages = _null_out(ages, _mask_indices(rng, N_USERS, USERS_NULL_AGE))

    is_business: list[Any] = [bool(b) for b in rng.random(N_USERS) < 0.18]
    is_business = _null_out(is_business, _mask_indices(rng, N_USERS, USERS_NULL_IS_BUSINESS))

    signup_source: list[str] = _pick(rng, SIGNUP_SOURCES, N_USERS, SIGNUP_SOURCE_WEIGHTS)
    for i in _mask_indices(rng, N_USERS, USERS_EMPTY_SIGNUP_SOURCE):
        signup_source[i] = ""  # empty string, deliberately NOT null

    ltv_cents = np.round(np.exp(rng.normal(7.6, 1.05, size=N_USERS)) * 100).astype(np.int64)
    ltv_cents = np.clip(ltv_cents, 0, 99_999_999_99)

    fields: list[pa.Field[Any]] = [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("email", pa.string(), nullable=False),
        pa.field("state", pa.string()),
        pa.field("country", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("age", pa.int64()),
        pa.field("is_business", pa.bool_()),
        pa.field("signup_source", pa.string(), nullable=False),
        pa.field("lifetime_value", pa.decimal128(12, 2), nullable=False),
    ]
    schema = pa.schema(fields)
    return pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(names, type=pa.string()),
            pa.array(emails, type=pa.string()),
            pa.array(states, type=pa.string()),
            pa.array(countries, type=pa.string()),
            pa.array(created_at, type=pa.timestamp("us", tz="UTC")),
            pa.array(ages, type=pa.int64()),
            pa.array(is_business, type=pa.bool_()),
            pa.array(signup_source, type=pa.string()),
            pa.array(_cents(ltv_cents), type=pa.decimal128(12, 2)),
        ],
        schema=schema,
    )


def build_products(rng: np.random.Generator) -> pa.Table:
    """200 rows. NULL categories, brands duplicated across categories, some negative margins."""
    ids = list(range(1, N_PRODUCTS + 1))

    brands = _pick(rng, BRANDS, N_PRODUCTS)
    nouns = _pick(rng, PRODUCT_NOUNS, N_PRODUCTS)
    names = [f"{b} {n}" for b, n in zip(brands, nouns, strict=True)]
    unicode_slots = sorted(_mask_indices(rng, N_PRODUCTS, len(PRODUCT_UNICODE_SUFFIXES) * 3))
    for slot, suffix in zip(unicode_slots, PRODUCT_UNICODE_SUFFIXES * 3, strict=True):
        names[slot] = names[slot] + suffix

    categories: list[Any] = _pick(rng, CATEGORIES, N_PRODUCTS, CATEGORY_WEIGHTS)
    categories = _null_out(categories, _mask_indices(rng, N_PRODUCTS, PRODUCTS_NULL_CATEGORY))

    cost_cents = np.round(np.exp(rng.normal(3.05, 0.75, size=N_PRODUCTS)) * 100).astype(np.int64)
    cost_cents = np.clip(cost_cents, 100, 99_999_999)
    markup = rng.uniform(1.15, 2.60, size=N_PRODUCTS)
    price_cents = np.round(cost_cents * markup).astype(np.int64)
    # Bake in negative margins (price < cost) for a fixed handful of products.
    for i in _mask_indices(rng, N_PRODUCTS, PRODUCTS_NEGATIVE_MARGIN):
        price_cents[i] = max(100, int(cost_cents[i]) * 82 // 100)
    price_cents = np.clip(price_cents, 100, 99_999_999)

    introduced_offsets = rng.integers(0, 7 * 365, size=N_PRODUCTS)
    introduced_on = [ANCHOR_DATE - timedelta(days=int(d)) for d in introduced_offsets]

    fields: list[pa.Field[Any]] = [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("category", pa.string()),
        pa.field("brand", pa.string(), nullable=False),
        pa.field("cost", pa.decimal128(10, 2), nullable=False),
        pa.field("price", pa.decimal128(10, 2), nullable=False),
        pa.field("introduced_on", pa.date32(), nullable=False),
    ]
    schema = pa.schema(fields)
    return pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(names, type=pa.string()),
            pa.array(categories, type=pa.string()),
            pa.array(brands, type=pa.string()),
            pa.array(_cents(cost_cents), type=pa.decimal128(10, 2)),
            pa.array(_cents(price_cents), type=pa.decimal128(10, 2)),
            pa.array(introduced_on, type=pa.date32()),
        ],
        schema=schema,
    )


def _seasonal_day_weights() -> _F64Array:
    start = ANCHOR_DATE - timedelta(days=FACT_HISTORY_DAYS - 1)
    weights = np.empty(FACT_HISTORY_DAYS, dtype=np.float64)
    for i in range(FACT_HISTORY_DAYS):
        day = start + timedelta(days=i)
        weights[i] = MONTH_SEASONALITY[day.month - 1] * WEEKDAY_SEASONALITY[day.weekday()]
    return _normalized(weights)


def build_order_items(rng: np.random.Generator, products: pa.Table, *, scale: int = 1) -> pa.Table:
    """The fact table. ``scale`` multiplies the row count; dimensions stay fixed."""
    n_rows = N_ORDER_ITEMS * scale

    # --- orders: sizes, then order-level attributes -------------------------------------
    estimated_orders = int(n_rows / 2.4) + 256
    sizes = rng.choice(
        ORDER_SIZES, size=estimated_orders, p=_normalized(ORDER_SIZE_WEIGHTS)
    ).astype(np.int64)
    cumulative = np.cumsum(sizes)
    cutoff = int(np.searchsorted(cumulative, n_rows, side="left")) + 1
    sizes = sizes[:cutoff]
    overshoot = int(sizes.sum()) - n_rows
    sizes[-1] -= overshoot
    if sizes[-1] <= 0:  # pragma: no cover - defensive; the estimate always overshoots
        raise RuntimeError("order-size trimming produced an empty trailing order")
    n_orders = int(sizes.shape[0])

    order_ids_per_order = np.arange(1, n_orders + 1, dtype=np.int64)
    order_id = np.repeat(order_ids_per_order, sizes)

    # Buyer popularity: a mild power law so distinct-buyer counts are interesting.
    buyer_weights = _normalized(1.0 / np.arange(1, N_USERS + 1) ** 0.55)
    user_per_order = rng.choice(N_USERS, size=n_orders, p=buyer_weights).astype(np.int64) + 1
    n_orphan_orders = max(1, round(n_orders * FACT_ORPHAN_ORDER_FRACTION))
    orphan_orders = sorted(_mask_indices(rng, n_orders, n_orphan_orders))
    for offset, order_index in enumerate(orphan_orders):
        user_per_order[order_index] = ORPHAN_USER_ID_BASE + offset + 1

    day_weights = _seasonal_day_weights()
    day_offsets = rng.choice(FACT_HISTORY_DAYS, size=n_orders, p=day_weights)
    second_offsets = rng.integers(0, 86_400, size=n_orders)
    # Guarantee at least one row exactly on the anchor date.
    day_offsets[0] = FACT_HISTORY_DAYS - 1
    second_offsets[0] = 43_200
    fact_start = ANCHOR_DATE - timedelta(days=FACT_HISTORY_DAYS - 1)
    fact_epoch = datetime(fact_start.year, fact_start.month, fact_start.day, tzinfo=UTC)
    created_per_order = [
        fact_epoch + timedelta(days=int(d), seconds=int(s))
        for d, s in zip(day_offsets, second_offsets, strict=True)
    ]

    user_id = np.repeat(user_per_order, sizes)
    created_at = [created_per_order[i] for i in np.repeat(np.arange(n_orders), sizes)]

    # --- item-level attributes -----------------------------------------------------------
    ids = list(range(1, n_rows + 1))
    product_index = rng.choice(N_PRODUCTS, size=n_rows)
    product_id = (product_index + 1).astype(np.int64)
    statuses = _pick(rng, STATUSES, n_rows, STATUS_WEIGHTS)
    quantity = rng.choice(QUANTITIES, size=n_rows, p=_normalized(QUANTITY_WEIGHTS)).astype(np.int64)

    product_price_cents = _to_cents(products.column("price").to_pylist())
    unit_multiplier = rng.uniform(0.88, 1.06, size=n_rows)
    unit_cents = np.round(product_price_cents[product_index] * unit_multiplier).astype(np.int64)
    unit_cents = np.maximum(unit_cents, 1)
    sale_cents = unit_cents * quantity  # per-unit x quantity already applied; always > 0

    discount_cents = np.round(sale_cents * rng.uniform(0.0, 0.45, size=n_rows)).astype(np.int64)
    discount_cents = np.minimum(discount_cents, sale_cents - 1)  # strictly < sale_price
    discount_cents = np.maximum(discount_cents, 0)
    discount: list[Any] = _cents(discount_cents)
    n_null_discount = round(n_rows * FACT_NULL_DISCOUNT_FRACTION)
    discount = _null_out(discount, _mask_indices(rng, n_rows, n_null_discount))

    returned: list[Any] = [bool(b) for b in rng.random(n_rows) < 0.07]
    n_null_returned = round(n_rows * FACT_NULL_RETURNED_FRACTION)
    returned = _null_out(returned, _mask_indices(rng, n_rows, n_null_returned))

    n_null_notes = round(n_rows * FACT_NULL_NOTES_FRACTION)
    note_rows = sorted(set(range(n_rows)) - _mask_indices(rng, n_rows, n_null_notes))
    note_bodies = _pick(rng, SHORT_NOTES, len(note_rows))
    n_long = max(1, round(len(note_rows) * FACT_LONG_NOTE_FRACTION))
    long_slots = _mask_indices(rng, len(note_rows), n_long)
    notes: list[Any] = [None] * n_rows
    for position, row in enumerate(note_rows):
        body = note_bodies[position]
        if position in long_slots:
            body = f"{body}\n{LONG_NOTE_PARAGRAPH * 8}"
        notes[row] = body

    fields: list[pa.Field[Any]] = [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("order_id", pa.int64(), nullable=False),
        pa.field("user_id", pa.int64(), nullable=False),
        pa.field("product_id", pa.int64(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("quantity", pa.int64(), nullable=False),
        pa.field("sale_price", pa.decimal128(12, 2), nullable=False),
        pa.field("discount", pa.decimal128(12, 2)),
        pa.field("returned", pa.bool_()),
        pa.field("notes", pa.large_string()),
    ]
    schema = pa.schema(fields)
    return pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(order_id, type=pa.int64()),
            pa.array(user_id, type=pa.int64()),
            pa.array(product_id, type=pa.int64()),
            pa.array(created_at, type=pa.timestamp("us", tz="UTC")),
            pa.array(statuses, type=pa.string()),
            pa.array(quantity, type=pa.int64()),
            pa.array(_cents(sale_cents), type=pa.decimal128(12, 2)),
            pa.array(discount, type=pa.decimal128(12, 2)),
            pa.array(returned, type=pa.bool_()),
            pa.array(notes, type=pa.large_string()),
        ],
        schema=schema,
    )


def build_tables(*, scale: int = 1) -> dict[str, pa.Table]:
    """All three tables. Dimensions are drawn first so ``--scale`` never perturbs them."""
    rng = np.random.default_rng(SEED)
    users = build_users(rng)
    products = build_products(rng)
    order_items = build_order_items(rng, products, scale=scale)
    return {"users": users, "products": products, "order_items": order_items}


# --------------------------------------------------------------------------------------
# known_answers.json (DuckDB)
# --------------------------------------------------------------------------------------

# The governed topic joins order_items -> users / products from the fact table, so every
# ground-truth query below LEFT JOINs (orphan user_ids therefore land in the NULL state group).
KNOWN_ANSWER_QUERIES: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "grand_totals",
        "Grand totals over the whole fact table: revenue, row count, distinct buyers, quantity. "
        "distinct_buyers mirrors the users.count measure (COUNT DISTINCT users.id, so orphan "
        "user_ids do not count); distinct_user_ids is the raw fact-column cardinality.",
        """
        SELECT SUM(oi.sale_price)              AS total_sale_price,
               COUNT(*)                        AS order_items_count,
               COUNT(DISTINCT u.id)            AS distinct_buyers,
               COUNT(DISTINCT oi.user_id)      AS distinct_user_ids,
               SUM(oi.quantity)                AS total_quantity,
               COUNT(DISTINCT p.id)            AS distinct_products
        FROM order_items oi
        LEFT JOIN users u ON u.id = oi.user_id
        LEFT JOIN products p ON p.id = oi.product_id
        """,
    ),
    (
        "revenue_by_state",
        "Revenue by users.state (LEFT JOIN; the NULL group holds NULL states and orphan users).",
        """
        SELECT u.state                         AS state,
               SUM(oi.sale_price)              AS total_sale_price,
               COUNT(*)                        AS order_items_count
        FROM order_items oi
        LEFT JOIN users u ON u.id = oi.user_id
        GROUP BY 1
        ORDER BY state NULLS FIRST
        """,
    ),
    (
        "monthly_revenue_trailing_12m",
        "Monthly revenue for the 12 months ending on the anchor date (2025-07 .. 2026-06).",
        """
        SELECT CAST(DATE_TRUNC('month', oi.created_at) AS TIMESTAMP) AS month,
               SUM(oi.sale_price)                                    AS total_sale_price,
               COUNT(*)                                              AS order_items_count
        FROM order_items oi
        WHERE oi.created_at >= TIMESTAMPTZ '2025-07-01 00:00:00+00'
          AND oi.created_at <  TIMESTAMPTZ '2026-07-01 00:00:00+00'
        GROUP BY 1
        ORDER BY month
        """,
    ),
    (
        "revenue_by_category_month",
        "Revenue by products.category x month over the trailing 12 months (NULL category kept).",
        """
        SELECT p.category                                            AS category,
               CAST(DATE_TRUNC('month', oi.created_at) AS TIMESTAMP) AS month,
               SUM(oi.sale_price)                                    AS total_sale_price,
               COUNT(*)                                              AS order_items_count
        FROM order_items oi
        LEFT JOIN products p ON p.id = oi.product_id
        WHERE oi.created_at >= TIMESTAMPTZ '2025-07-01 00:00:00+00'
          AND oi.created_at <  TIMESTAMPTZ '2026-07-01 00:00:00+00'
        GROUP BY 1, 2
        ORDER BY category NULLS FIRST, month
        """,
    ),
    (
        "distinct_buyers_by_month",
        "Distinct buyers (COUNT DISTINCT users.id) per month over the full 2-year window.",
        """
        SELECT CAST(DATE_TRUNC('month', oi.created_at) AS TIMESTAMP) AS month,
               COUNT(DISTINCT u.id)                                  AS distinct_buyers,
               COUNT(DISTINCT oi.user_id)                            AS distinct_user_ids
        FROM order_items oi
        LEFT JOIN users u ON u.id = oi.user_id
        GROUP BY 1
        ORDER BY month
        """,
    ),
    (
        "revenue_and_buyers_by_state",
        "Mixed aggregation: governed revenue measure + ad-hoc COUNT(DISTINCT users.id) by state.",
        """
        SELECT u.state                    AS state,
               SUM(oi.sale_price)         AS total_sale_price,
               COUNT(DISTINCT u.id)       AS distinct_buyers
        FROM order_items oi
        LEFT JOIN users u ON u.id = oi.user_id
        GROUP BY 1
        ORDER BY state NULLS FIRST
        """,
    ),
    (
        "count_by_status",
        "Row count and revenue by order_items.status.",
        """
        SELECT oi.status                   AS status,
               COUNT(*)                    AS order_items_count,
               SUM(oi.sale_price)          AS total_sale_price
        FROM order_items oi
        GROUP BY 1
        ORDER BY status
        """,
    ),
    (
        "average_sale_price_by_category",
        "AVG(sale_price) by products.category, cast to a fixed-scale decimal for exactness.",
        """
        SELECT p.category                                       AS category,
               CAST(AVG(oi.sale_price) AS DECIMAL(38, 10))      AS average_sale_price,
               COUNT(*)                                         AS order_items_count
        FROM order_items oi
        LEFT JOIN products p ON p.id = oi.product_id
        GROUP BY 1
        ORDER BY category NULLS FIRST
        """,
    ),
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str | float):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        # DuckDB hands back naive UTC timestamps (session TimeZone is pinned to UTC); the
        # timestamptz -> Python conversion is deliberately avoided, it needs pytz.
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported known-answer value type: {type(value)!r}")  # pragma: no cover


def compute_known_answers(tables: dict[str, pa.Table], *, scale: int) -> dict[str, Any]:
    """Run every ground-truth query through DuckDB against the in-memory Arrow tables."""
    con = duckdb.connect()
    try:
        con.execute("SET threads TO 1")
        con.execute("SET TimeZone='UTC'")
        for name in TABLE_NAMES:
            con.register(name, tables[name])

        answers: dict[str, Any] = {}
        for answer_id, description, sql in KNOWN_ANSWER_QUERIES:
            cleaned = " ".join(sql.split())
            cursor = con.execute(cleaned)
            columns = [d[0] for d in cursor.description or []]
            rows = [[_jsonable(v) for v in row] for row in cursor.fetchall()]
            answers[answer_id] = {
                "description": description,
                "sql": cleaned,
                "columns": columns,
                "row_count": len(rows),
                "rows": rows,
            }
    finally:
        con.close()

    return {
        "bench_version": BENCH_VERSION,
        "seed": SEED,
        "scale": scale,
        "anchor_date": ANCHOR_DATE.isoformat(),
        "row_counts": {name: tables[name].num_rows for name in TABLE_NAMES},
        "answers": answers,
    }


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------


def write_dataset(tables: dict[str, pa.Table], known_answers: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in TABLE_NAMES:
        table = tables[name]
        pq.write_table(table, out / f"{name}.parquet", **PARQUET_WRITE_OPTIONS)
        pacsv.write_csv(table, out / f"{name}.csv", pacsv.WriteOptions(**CSV_WRITE_OPTIONS))
    text = json.dumps(known_answers, indent=2, ensure_ascii=False, sort_keys=False)
    (out / "known_answers.json").write_text(text + "\n", encoding="utf-8")


def generate(out: Path, *, scale: int = 1) -> dict[str, str]:
    """Generate the dataset into ``out``; returns ``{filename: sha256}``."""
    tables = build_tables(scale=scale)
    known_answers = compute_known_answers(tables, scale=scale)
    write_dataset(tables, known_answers, out)
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir())
        if path.is_file()
    }


def default_out_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "data" / "bench"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="multiply the fact-table row count (dimensions stay fixed); default 1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory; default tests/data/bench/",
    )
    args = parser.parse_args(argv)
    if args.scale < 1:
        parser.error("--scale must be >= 1")

    out: Path = args.out if args.out is not None else default_out_dir()
    digests = generate(out, scale=args.scale)
    print(f"bench dataset v{BENCH_VERSION} (seed={SEED}, scale={args.scale}) -> {out}")
    for filename, digest in digests.items():
        print(f"  {digest}  {filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
