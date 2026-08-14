"""Arrow/summary decoding against checked-in wire bodies (CONTRACT_NOTES §2.3/§2.6)."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pytest

from omniframes.errors import QueryError, TransportError
from omniframes.transport.arrow import (
    check_missing_fields,
    decode_result,
    missing_fields,
    schema_from_summary,
)
from omniframes.transport.ndjson import parse_response
from omniframes.transport.normalize import normalize
from omniframes.types import OmniDataType
from tests.wire import read_fixture


def job(name: str, index: int = 0):
    return parse_response(read_fixture(name)).jobs[index]


# --------------------------------------------------------------------------------------------
# decode_result
# --------------------------------------------------------------------------------------------


def test_decode_result_reads_the_ipc_stream():
    line = job("happy_single_job.ndjson")
    assert line.result is not None

    table = decode_result(line.result)

    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert table.column_names == [
        "users.state",
        "order_items.order_count",
        "order_items.total_sale_price",
    ]
    assert table.column("users.state").to_pylist() == ["CA", "NY", None]
    assert table.column("order_items.total_sale_price").to_pylist() == [1234.5, 890.25, 42.0]


def test_decode_result_handles_exotic_types():
    line = job("exotic_types.ndjson")
    assert line.result is not None

    table = decode_result(line.result)
    schema = table.schema

    assert schema.field("users.state").type == pa.large_string()
    assert schema.field("users.lifetime_value").type == pa.decimal128(12, 2)
    assert schema.field("users.created_at").type == pa.timestamp("us", tz="UTC")
    assert schema.field("users.is_business").type == pa.bool_()
    assert schema.field("users.age").type == pa.int64()

    assert table.column("users.lifetime_value").to_pylist() == [
        Decimal("1234.56"),
        None,
        Decimal("-0.99"),
    ]
    assert table.column("users.created_at").to_pylist()[0] == datetime(
        2021, 3, 14, 15, 9, 26, 535000, tzinfo=UTC
    )
    assert table.column("users.created_at").to_pylist()[2] is None
    assert table.column("users.is_business").to_pylist() == [True, None, False]
    assert table.column("users.age").to_pylist() == [41, None, 7]


def test_decode_result_accepts_a_zero_row_stream():
    line = job("complete_failed_to_plan.ndjson")
    assert line.result is not None

    table = decode_result(line.result)

    assert table.num_rows == 0
    assert table.column_names[0] == "users.state"


def test_decode_result_rejects_non_base64():
    with pytest.raises(TransportError, match="valid base64"):
        decode_result("not base64!!!")


def test_decode_result_rejects_empty_payload():
    with pytest.raises(TransportError, match="empty"):
        decode_result("")


def test_decode_result_rejects_non_arrow_bytes():
    payload = base64.b64encode(b"definitely not arrow ipc").decode("ascii")

    with pytest.raises(TransportError, match="Arrow IPC stream"):
        decode_result(payload)


def test_decode_result_rejects_an_ipc_file_payload():
    table = pa.table({"a": pa.array([1, 2, 3], pa.int64())})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    payload = base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii")

    with pytest.raises(TransportError, match="Arrow IPC stream"):
        decode_result(payload)


# --------------------------------------------------------------------------------------------
# schema_from_summary
# --------------------------------------------------------------------------------------------


def test_schema_from_summary_preserves_order_and_metadata():
    line = job("plan_only.ndjson")
    assert line.summary is not None

    schema = schema_from_summary(line.summary["fields"])

    assert schema.names == (
        "users.state",
        "order_items.order_count",
        "order_items.total_sale_price",
    )
    assert len(schema) == 3
    state = schema["users.state"]
    assert state.data_type is OmniDataType.STRING
    assert state.view_name == "users"
    assert state.is_dimension is True
    assert state.is_calc is False
    assert state.raw["fully_qualified_name"] == "users.state"

    measure = schema["order_items.total_sale_price"]
    assert measure.data_type is OmniDataType.NUMBER
    assert measure.is_dimension is False
    assert measure.aggregate_type == "sum"


def test_schema_from_summary_covers_the_exotic_data_types():
    line = job("exotic_types.ndjson")
    assert line.summary is not None

    schema = schema_from_summary(line.summary["fields"])

    assert [f.data_type for f in schema.fields] == [
        OmniDataType.STRING,
        OmniDataType.NUMBER,
        OmniDataType.TIMESTAMP,
        OmniDataType.BOOLEAN,
        OmniDataType.NUMBER,
    ]
    assert schema["users.created_at"].date_type == "timestamp"


def test_schema_from_summary_tolerates_missing_and_odd_payloads():
    assert schema_from_summary(None).names == ()
    assert schema_from_summary({}).names == ()

    schema = schema_from_summary({"weird.field": None, "other.field": {"data_type": "NOPE"}})

    assert schema.names == ("weird.field", "other.field")
    assert schema["weird.field"].data_type is OmniDataType.UNKNOWN
    assert schema["other.field"].data_type is OmniDataType.UNKNOWN


def test_schema_matches_the_decoded_arrow_columns():
    line = job("happy_single_job.ndjson")
    assert line.summary is not None
    assert line.result is not None

    schema = schema_from_summary(line.summary["fields"])
    table = decode_result(line.result)

    assert list(schema.names) == table.column_names


def test_schema_collapses_a_formatted_grain_pair():
    """docs/SQLTIER.md §5: one field, under the plain name, with the ``__raw`` half's type."""
    line = job("grain_pair.ndjson")
    assert line.summary is not None

    schema = schema_from_summary(line.summary["fields"])

    assert schema.names == ("order_items.created_at[month]", "order_items.total_sale_price")
    month = schema["order_items.created_at[month]"]
    assert month.data_type is OmniDataType.TIMESTAMP
    assert month.date_type == "timestamp"
    # The display format belongs to the string half that was dropped.
    assert month.raw["format"] is None


@pytest.mark.parametrize("fixture", ["grain_pair.ndjson", "omnisql_expressions.ndjson"])
def test_schema_and_normalized_result_agree_on_the_collapsed_columns(fixture):
    line = job(fixture)
    assert line.summary is not None
    assert line.result is not None

    schema = schema_from_summary(line.summary["fields"])
    result = normalize(decode_result(line.result), line.summary["fields"])

    assert list(schema.names) == result.data.column_names


def test_schema_leaves_a_lone_raw_entry_alone():
    schema = schema_from_summary(
        {
            "order_items.created_at[month]__raw": {"data_type": "TIMESTAMP"},
            "revenue__raw": {"data_type": "NUMBER"},
            "revenue": {"data_type": "STRING"},
        }
    )

    assert schema.names == (
        "order_items.created_at[month]__raw",
        "revenue__raw",
        "revenue",
    )


# --------------------------------------------------------------------------------------------
# missing_fields
# --------------------------------------------------------------------------------------------


def test_missing_fields_is_empty_on_a_clean_run():
    line = job("happy_single_job.ndjson")

    assert missing_fields(line.summary) == ()
    assert missing_fields(None) == ()
    assert missing_fields({}) == ()
    check_missing_fields(line.summary)  # does not raise


def test_missing_fields_surfaces_silently_dropped_fields():
    line = job("missing_fields.ndjson")

    assert missing_fields(line.summary) == ("users.stat", "order_items.created_at[fortnight]")


def test_check_missing_fields_raises_a_query_error():
    line = job("missing_fields.ndjson")

    with pytest.raises(QueryError, match=r"users\.stat") as excinfo:
        check_missing_fields(line.summary, job_id=line.job_id)

    assert excinfo.value.error_type == "MISSING_FIELDS"
    assert excinfo.value.job_id == line.job_id
    assert "order_items.created_at[fortnight]" in str(excinfo.value)
