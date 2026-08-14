"""Deterministic builder for the wire-lane NDJSON fixtures.

Run it from the repo root::

    uv run python tests/wire/build_fixtures.py

Everything here is fixed data — no clock, no RNG — so re-running reproduces byte-identical
bodies for a given pyarrow version.  The ``result`` payloads are **real** Arrow IPC streams
(base64 of ``ipc.new_stream`` output), and the NDJSON framing follows CONTRACT_NOTES §2.2
exactly: one compact JSON object per line, ``\\n`` after every framed line, nulls omitted, and
the tolerated *unterminated* upstream-failure tail written without a trailing separator.

Only the keys documented in CONTRACT_NOTES are load-bearing; ``cache_metadata`` / ``stats``
payloads are plausible filler because the contract does not pin their shape.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

MODEL_ID = "8f1a0c2e-1111-4a2b-9c3d-000000000001"

JOB_HAPPY = "0f4d3c2b-0000-4000-8000-00000000a001"
JOB_EXOTIC = "0f4d3c2b-0000-4000-8000-00000000a002"
JOB_TOTALS = "0f4d3c2b-0000-4000-8000-00000000a003"
JOB_FAST = "0f4d3c2b-0000-4000-8000-00000000a004"
JOB_SLOW = "0f4d3c2b-0000-4000-8000-00000000a005"
JOB_ERROR = "0f4d3c2b-0000-4000-8000-00000000a006"
JOB_KILLED = "0f4d3c2b-0000-4000-8000-00000000a007"
JOB_REDACTED = "0f4d3c2b-0000-4000-8000-00000000a008"
JOB_UNPLANNED = "0f4d3c2b-0000-4000-8000-00000000a009"
JOB_PLANNED = "0f4d3c2b-0000-4000-8000-00000000a00a"
JOB_MISSING = "0f4d3c2b-0000-4000-8000-00000000a00b"
JOB_TRAILING = "0f4d3c2b-0000-4000-8000-00000000a00c"
JOB_EXECUTING = "0f4d3c2b-0000-4000-8000-00000000a00d"
JOB_UNKNOWN_STATUS = "0f4d3c2b-0000-4000-8000-00000000a00e"
JOB_REQUERY = "0f4d3c2b-0000-4000-8000-00000000a00f"

REDACTED_MESSAGE = (
    "Query failed. Error details are only visible to users with permission to view SQL."
)

# --------------------------------------------------------------------------------------------
# Arrow tables
# --------------------------------------------------------------------------------------------


def happy_table() -> pa.Table:
    """A plain grouped result: one dimension (with a NULL group key) and two measures."""
    return pa.table(
        {
            "users.state": pa.array(["CA", "NY", None], pa.string()),
            "order_items.order_count": pa.array([12, 7, 3], pa.int64()),
            "order_items.total_sale_price": pa.array([1234.5, 890.25, 42.0], pa.float64()),
        }
    )


def exotic_table() -> pa.Table:
    """Arrow types the wire really produces: decimal128, tz-aware timestamps, large_string, nulls."""
    return pa.table(
        {
            "users.state": pa.array(["CA", None, "NY"], pa.large_string()),
            "users.lifetime_value": pa.array(
                [Decimal("1234.56"), None, Decimal("-0.99")], pa.decimal128(12, 2)
            ),
            "users.created_at": pa.array(
                [
                    datetime(2021, 3, 14, 15, 9, 26, 535000, tzinfo=UTC),
                    datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC),
                    None,
                ],
                pa.timestamp("us", tz="UTC"),
            ),
            "users.is_business": pa.array([True, None, False], pa.bool_()),
            "users.age": pa.array([41, None, 7], pa.int64()),
        }
    )


def totals_table() -> pa.Table:
    """A raw-SQL ``column_totals`` result: totals rows, ``__omni_summ`` sidecars, reserved columns.

    Rows 0-2 are data (row 2 carries an *empty* indicator, which must not count as a total);
    rows 3-5 are a subtotal, the grand total, and a row total.  ``TOTAL_SALE_PRICE`` gets its
    totals values from a sidecar of the same type, ``ORDER_COUNT`` from a string sidecar (cast on
    merge), and the ``_1``-suffixed duplicate must lose to the first occurrence.
    """
    return pa.table(
        {
            # Uppercase raw-SQL column names: the sidecar key is the *lowercased* field name.
            "STATE": pa.array(["CA", "NY", None, "CA", None, "CA"], pa.string()),
            "TOTAL_SALE_PRICE": pa.array([10.0, 20.0, 5.0, None, None, 1.5], pa.float64()),
            "total_sale_price__omni_summ": pa.array(
                [None, None, None, 30.0, 35.0, None], pa.float64()
            ),
            # Duplicate sidecar — first occurrence wins, so these values must never appear.
            "total_sale_price__omni_summ_1": pa.array(
                [None, None, None, 111.0, 222.0, 333.0], pa.float64()
            ),
            "ORDER_COUNT": pa.array([2, 3, 1, None, None, 1], pa.int64()),
            "order_count__omni_summ": pa.array([None, None, None, "6", "6", None], pa.string()),
            "PRODUCT__omni_sort": pa.array([0, 1, 2, 3, 4, 5], pa.int64()),
            "$omni_id": pa.array(["r0", "r1", "r2", "r3", "r4", "r5"], pa.string()),
            "$omni_column_total_indicator": pa.array(
                [None, None, "", "column_subtotal::STATE", "::total::", None], pa.string()
            ),
            "$omni_row_total_indicator": pa.array(
                [None, None, None, None, None, "row_total"], pa.string()
            ),
        }
    )


def empty_table() -> pa.Table:
    """A zero-row result with the happy schema (what a failed plan still ships)."""
    return happy_table().slice(0, 0)


def arrow_b64(table: pa.Table) -> str:
    """Base64 of an Arrow IPC **stream** (not a file) — exactly what ``result`` carries."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii")


# --------------------------------------------------------------------------------------------
# summary / query payloads
# --------------------------------------------------------------------------------------------


def wire_field(
    name: str,
    data_type: str,
    *,
    is_dimension: bool = True,
    aggregate_type: str | None = None,
    date_type: str | None = None,
    sql: str = "",
) -> dict[str, Any]:
    view_name, _, field_name = name.partition(".")
    return {
        "field_name": field_name or name,
        "fully_qualified_name": name,
        "view_name": view_name,
        "data_type": data_type,
        "is_dimension": is_dimension,
        "is_calc": False,
        "label": (field_name or name).replace("_", " ").title(),
        "format": None,
        "date_type": date_type,
        "aggregate_type": aggregate_type,
        "filter_only_field": False,
        "hidden": False,
        "sql": sql,
    }


HAPPY_FIELDS: dict[str, Any] = {
    "users.state": wire_field("users.state", "STRING", sql='"users"."state"'),
    "order_items.order_count": wire_field(
        "order_items.order_count",
        "NUMBER",
        is_dimension=False,
        aggregate_type="count",
        sql="COUNT(*)",
    ),
    "order_items.total_sale_price": wire_field(
        "order_items.total_sale_price",
        "NUMBER",
        is_dimension=False,
        aggregate_type="sum",
        sql='SUM("order_items"."sale_price")',
    ),
}

EXOTIC_FIELDS: dict[str, Any] = {
    "users.state": wire_field("users.state", "STRING"),
    "users.lifetime_value": wire_field("users.lifetime_value", "NUMBER", is_dimension=False),
    "users.created_at": wire_field("users.created_at", "TIMESTAMP", date_type="timestamp"),
    "users.is_business": wire_field("users.is_business", "BOOLEAN"),
    "users.age": wire_field("users.age", "NUMBER"),
}

TOTALS_FIELDS: dict[str, Any] = {
    "STATE": wire_field("STATE", "STRING"),
    "TOTAL_SALE_PRICE": wire_field("TOTAL_SALE_PRICE", "NUMBER", is_dimension=False),
    "ORDER_COUNT": wire_field("ORDER_COUNT", "NUMBER", is_dimension=False),
}

HAPPY_SQL = (
    'SELECT "users"."state", COUNT(*), SUM("order_items"."sale_price")\n'
    'FROM "order_items" LEFT JOIN "users" ON "users"."id" = "order_items"."user_id"\n'
    "GROUP BY 1 ORDER BY 3 DESC LIMIT 50000"
)


def summary(
    fields: Mapping[str, Any],
    *,
    display_sql: str = HAPPY_SQL,
    omni_sql: str = "query { state, order_count }",
    cache_type: str = "MISS",
    missing_fields: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "fields": dict(fields),
        "display_sql": display_sql,
        "omni_sql": omni_sql,
        "omni_sql_parse_failed": False,
        "cache_type": cache_type,
        "stage_summaries": [{"succeeded": True, "warnings": []}],
        "stats": {"rows": 3, "duration_ms": 118},
        "plan_stats": {"stages": 1},
        "missing_fields": list(missing_fields),
        "invalid_calculations": {},
        "locale_options": {"timezone": "UTC"},
    }


def query(
    *,
    table: str = "order_items",
    fields: Sequence[str] = ("users.state", "order_items.order_count"),
    user_edited_sql: str = "",
) -> dict[str, Any]:
    return {
        "modelId": MODEL_ID,
        "table": table,
        "join_paths_from_topic_name": "order_items_topic",
        "fields": list(fields),
        "filters": {},
        "sorts": [],
        "limit": 50000,
        "offset": 0,
        "pivots": [],
        "calculations": [],
        "fill_fields": [],
        "column_totals": {},
        "row_totals": {},
        "userEditedSQL": user_edited_sql,
        "default_group_by": True,
        "version": 9,
    }


# --------------------------------------------------------------------------------------------
# NDJSON framing
# --------------------------------------------------------------------------------------------


def header_line(jobs_submitted: Mapping[str, str | None]) -> dict[str, Any]:
    return {"jobs_submitted": dict(jobs_submitted)}


def footer_line(remaining: Sequence[str] = ()) -> dict[str, Any]:
    """``timed_out`` is a *string*, and is "true" iff ``remaining_job_ids`` is non-empty."""
    return {
        "remaining_job_ids": list(remaining),
        "timed_out": "true" if remaining else "false",
    }


def ndjson(
    lines: Sequence[Mapping[str, Any]],
    *,
    unterminated_tail: str | None = None,
) -> bytes:
    body = "".join(f"{json.dumps(line, separators=(',', ':'))}\n" for line in lines)
    if unterminated_tail is not None:
        body += unterminated_tail
    return body.encode("utf-8")


# --------------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------------


def build_fixtures() -> dict[str, bytes]:
    """Every fixture body, keyed by file name."""
    fixtures: dict[str, bytes] = {}

    # 1. Happy path: header, one COMPLETE job with a result, footer with nothing remaining.
    fixtures["happy_single_job.ndjson"] = ndjson(
        [
            header_line({JOB_HAPPY: "cri-happy"}),
            {
                "job_id": JOB_HAPPY,
                "status": "COMPLETE",
                "client_result_id": "cri-happy",
                "summary": summary(HAPPY_FIELDS),
                "cache_metadata": {"cache_type": "MISS", "job_id": JOB_HAPPY},
                "query": query(),
                "result": arrow_b64(happy_table()),
                "stream_stats": {"server_stream": 42},
            },
            footer_line(),
        ]
    )

    # 2. Exotic Arrow types in the result payload.
    fixtures["exotic_types.ndjson"] = ndjson(
        [
            header_line({JOB_EXOTIC: None}),
            {
                "job_id": JOB_EXOTIC,
                "status": "COMPLETE",
                "summary": summary(EXOTIC_FIELDS, display_sql='SELECT * FROM "users" LIMIT 3'),
                "cache_metadata": {"cache_type": "EXACT", "job_id": JOB_EXOTIC},
                "query": query(table="users", fields=list(EXOTIC_FIELDS)),
                "result": arrow_b64(exotic_table()),
                "stream_stats": {"server_stream": 7},
            },
            footer_line(),
        ]
    )

    # 3. Totals rows, __omni_summ sidecars, indicator + other reserved columns.
    fixtures["totals_with_sidecars.ndjson"] = ndjson(
        [
            header_line({JOB_TOTALS: "cri-totals"}),
            {
                "job_id": JOB_TOTALS,
                "status": "COMPLETE",
                "client_result_id": "cri-totals",
                "summary": summary(
                    TOTALS_FIELDS,
                    display_sql='SELECT state, SUM(sale_price) FROM "order_items" GROUP BY 1',
                ),
                "cache_metadata": {"cache_type": "MISS", "job_id": JOB_TOTALS},
                "query": query(
                    fields=list(TOTALS_FIELDS),
                    user_edited_sql="SELECT state, SUM(sale_price) AS total_sale_price FROM order_items GROUP BY 1",
                ),
                "result": arrow_b64(totals_table()),
                "stream_stats": {"server_stream": 63},
            },
            footer_line(),
        ]
    )

    # 4. Wait cycle — the run response completes one job and leaves another running.
    fixtures["wait_cycle_run.ndjson"] = ndjson(
        [
            header_line({JOB_FAST: "cri-fast", JOB_SLOW: None}),
            {
                "job_id": JOB_FAST,
                "status": "COMPLETE",
                "client_result_id": "cri-fast",
                "summary": summary(HAPPY_FIELDS),
                "cache_metadata": {"cache_type": "EXACT", "job_id": JOB_FAST},
                "query": query(),
                "result": arrow_b64(happy_table()),
                "stream_stats": {"server_stream": 11},
            },
            footer_line([JOB_SLOW]),
        ]
    )

    # 4b. A /query/wait slice that expired: footer only, no header, job still remaining.
    fixtures["wait_cycle_wait_timeout.ndjson"] = ndjson([footer_line([JOB_SLOW])])

    # 4c. The wait slice that finishes the job (no header line on /query/wait responses).
    fixtures["wait_cycle_wait.ndjson"] = ndjson(
        [
            {
                "job_id": JOB_SLOW,
                "status": "COMPLETE",
                "summary": summary(HAPPY_FIELDS, cache_type="EXACT_STALE"),
                "cache_metadata": {"cache_type": "EXACT_STALE", "job_id": JOB_SLOW},
                "query": query(),
                "result": arrow_b64(happy_table()),
                "stream_stats": {"server_stream": 29814},
            },
            footer_line(),
        ]
    )

    # 5. Error lines: a plain query error (client_result_id is the literal string "null") and a
    #    killed job carrying kill_reason.
    fixtures["error_line.ndjson"] = ndjson(
        [
            header_line({JOB_ERROR: "cri-error", JOB_KILLED: None}),
            {
                "job_id": JOB_ERROR,
                "status": "ERROR",
                "client_result_id": "null",
                "error_type": "QUERY",
                "error_message": 'Binder Error: Referenced column "totl_sale_price" not found',
                "summary": summary(HAPPY_FIELDS),
                "query": query(),
            },
            {
                "job_id": JOB_KILLED,
                "status": "FAILED",
                "error_type": "KILL",
                "error_message": "Query was killed",
                "kill_reason": "TIMEOUT",
            },
            footer_line(),
        ]
    )

    # 5b. The same failure without VIEW_SQL: message replaced, SQL blanked in the summary.
    redacted_summary = summary(
        {name: {**payload, "sql": ""} for name, payload in HAPPY_FIELDS.items()},
        display_sql="",
        omni_sql="",
    )
    fixtures["error_line_redacted.ndjson"] = ndjson(
        [
            header_line({JOB_REDACTED: "cri-redacted"}),
            {
                "job_id": JOB_REDACTED,
                "status": "ERROR",
                "client_result_id": "cri-redacted",
                "error_type": "QUERY",
                "error_message": REDACTED_MESSAGE,
                "summary": redacted_summary,
            },
            footer_line(),
        ]
    )

    # 6. COMPLETE, but the planner failed — the marker only shows up in summary.display_sql.
    fixtures["complete_failed_to_plan.ndjson"] = ndjson(
        [
            header_line({JOB_UNPLANNED: "cri-unplanned"}),
            {
                "job_id": JOB_UNPLANNED,
                "status": "COMPLETE",
                "client_result_id": "cri-unplanned",
                "summary": summary(
                    HAPPY_FIELDS,
                    display_sql=(
                        "-- Failed to plan query: no join path from users to order_items\n"
                        "-- (topic order_items_topic)"
                    ),
                ),
                "cache_metadata": {"cache_type": "MISS", "job_id": JOB_UNPLANNED},
                "query": query(),
                "result": arrow_b64(empty_table()),
                "stream_stats": {"server_stream": 3},
            },
            footer_line(),
        ]
    )

    # 7. planOnly: PLANNED status, summary.fields present, no result / cache_metadata.
    fixtures["plan_only.ndjson"] = ndjson(
        [
            header_line({JOB_PLANNED: "cri-plan"}),
            {
                "job_id": JOB_PLANNED,
                "status": "PLANNED",
                "client_result_id": "cri-plan",
                "summary": summary(HAPPY_FIELDS),
                "query": query(),
            },
            footer_line(),
        ]
    )

    # 8. A "successful" job that silently dropped requested fields.
    fixtures["missing_fields.ndjson"] = ndjson(
        [
            header_line({JOB_MISSING: "cri-missing"}),
            {
                "job_id": JOB_MISSING,
                "status": "COMPLETE",
                "client_result_id": "cri-missing",
                "summary": summary(
                    HAPPY_FIELDS,
                    missing_fields=["users.stat", "order_items.created_at[fortnight]"],
                ),
                "cache_metadata": {"cache_type": "MISS", "job_id": JOB_MISSING},
                "query": query(),
                "result": arrow_b64(happy_table()),
                "stream_stats": {"server_stream": 55},
            },
            footer_line(),
        ]
    )

    # 9. Mid-stream upstream failure: an unterminated tail instead of a footer.
    fixtures["trailing_error.ndjson"] = ndjson(
        [
            header_line({JOB_TRAILING: "cri-trailing"}),
            {
                "job_id": JOB_TRAILING,
                "status": "COMPLETE",
                "client_result_id": "cri-trailing",
                "summary": summary(HAPPY_FIELDS),
                "cache_metadata": {"cache_type": "MISS", "job_id": JOB_TRAILING},
                "query": query(),
                "result": arrow_b64(happy_table()),
                "stream_stats": {"server_stream": 18},
            },
        ],
        unterminated_tail=json.dumps(
            {"message": "Upstream request failed", "reason": "ECONNRESET"},
            separators=(",", ":"),
        ),
    )

    # 9b. The same, cut off mid-object — still must not blow up the parser.
    fixtures["trailing_error_truncated.ndjson"] = ndjson(
        [header_line({JOB_TRAILING: "cri-trailing"})],
        unterminated_tail='{"message":"Upstream request fail',
    )

    # 10. Open status enum: a known non-terminal status and one we have never seen.
    fixtures["unknown_status.ndjson"] = ndjson(
        [
            header_line({JOB_EXECUTING: "cri-exec", JOB_UNKNOWN_STATUS: None}),
            {"job_id": JOB_EXECUTING, "status": "EXECUTING"},
            {"job_id": JOB_UNKNOWN_STATUS, "status": "WARP_SPEED_ENGAGED"},
            footer_line([JOB_EXECUTING, JOB_UNKNOWN_STATUS]),
        ]
    )

    # 11. Requery line with no result: client-side materialization we refuse to do.
    fixtures["requery_without_result.ndjson"] = ndjson(
        [
            header_line({JOB_REQUERY: "cri-requery"}),
            {
                "job_id": JOB_REQUERY,
                "status": "COMPLETE",
                "client_result_id": "cri-requery",
                "summary": summary(HAPPY_FIELDS, cache_type="CUBE_REQUERY"),
                "query": query(),
                "requery_sql": 'SELECT * FROM "omni_cube_1234"',
                "requery_table_name": "omni_cube_1234",
                "requery_fallback_sql": HAPPY_SQL,
                "column_name_mapping": {"c0": "users.state"},
            },
            footer_line(),
        ]
    )

    return fixtures


def write_fixtures(out_dir: Path = FIXTURES_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, body in build_fixtures().items():
        path = out_dir / name
        path.write_bytes(body)
        written.append(path)
    return written


def main() -> None:
    for path in write_fixtures():
        print(f"wrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()
