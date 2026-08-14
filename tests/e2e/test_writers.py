"""``df.write`` and ``df.to_polars()`` end to end (docs/SQLTIER.md §6).

Round trips, not shapes: each writer is checked by reading the file back with an independent
reader and comparing it to what ``collect()`` returned.  Parquet is lossless, so the comparison
is exact — decimals included, which is the whole reason a frame of money should be written as
Parquet rather than as CSV.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pytest
from pyarrow import csv as pa_csv
from pyarrow import parquet as pa_parquet

from omniframes import OmniSession
from omniframes import functions as F
from omniframes.dataframe import DataFrame
from omniframes.errors import CompileError
from omniframes.io.writers import DataFrameWriter
from omniframes.transport import HttpTransport
from tests.fakes import (
    BENCH_MODEL_NAME,
    BENCH_TOPIC_NAME,
    DEFAULT_TOKEN,
    FakeOmniAPI,
)

BASE_URL = "https://bench.example.omni.co"
PRICE = "order_items.sale_price"
STATE = "users.state"


@pytest.fixture
def handler() -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    yield fake
    fake.close()


@pytest.fixture
def orders(handler: FakeOmniAPI) -> Iterator[DataFrame]:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    with client:
        transport = HttpTransport(
            base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
        )
        session = OmniSession.builder.base_url(BASE_URL).transport(transport).get_or_create()
        yield session.read.topic(BENCH_MODEL_NAME, BENCH_TOPIC_NAME)


@pytest.fixture
def revenue(orders: DataFrame) -> DataFrame:
    """A frame with a decimal, an integer and a nullable string column."""
    return orders.group_by(STATE).agg(
        F.sum(PRICE).alias("revenue"), F.count("order_items.id").alias("n")
    )


# --------------------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------------------


def test_parquet_round_trips_the_frame_exactly(revenue: DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "revenue.parquet"
    expected = revenue.collect()
    revenue.write.parquet(path)

    assert pa_parquet.read_table(path).equals(expected)


def test_parquet_takes_the_compression_it_is_given(revenue: DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "revenue.snappy.parquet"
    revenue.write.parquet(path, compression="snappy")
    metadata = pa_parquet.ParquetFile(path).metadata.row_group(0).column(0)

    assert metadata.compression.lower() == "snappy"
    assert pa_parquet.read_table(path).num_rows == revenue.collect().num_rows


def test_csv_round_trips_the_values_as_text(revenue: DataFrame, tmp_path: Path) -> None:
    """CSV has no types, so the comparison is over the rendered values — which must all be there."""
    path = tmp_path / "revenue.csv"
    expected = revenue.collect()
    revenue.write.csv(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reread = list(csv.reader(handle))

    assert reread[0] == list(expected.column_names)
    assert reread[1:] == _as_text(expected)
    # pyarrow re-reads it as a table too — it just has to guess the types back, which is the
    # part CSV cannot promise (the decimal above comes back as a float).
    assert pa_csv.read_csv(path).num_rows == expected.num_rows


def test_csv_can_omit_the_header(revenue: DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "headerless.csv"
    revenue.write.csv(path, include_header=False)
    first = path.read_text("utf-8").splitlines()[0]

    assert STATE not in first
    assert len(path.read_text("utf-8").splitlines()) == revenue.collect().num_rows


def test_a_writer_writes_what_collect_returns_normalized_and_aliased(
    orders: DataFrame, tmp_path: Path
) -> None:
    """Aliases, reserved-column stripping and the tier-2 column names all apply first."""
    frame = orders.select(F.col(STATE).alias("state"), F.col("users.age").alias("age")).limit(5)
    path = tmp_path / "aliased.parquet"
    frame.write.parquet(path)

    assert pa_parquet.read_table(path).column_names == ["state", "age"]


def test_the_writer_is_reachable_from_the_frame_and_names_itself(orders: DataFrame) -> None:
    writer = orders.write

    assert isinstance(writer, DataFrameWriter)
    assert repr(writer) == "DataFrameWriter(csv, parquet)"


@pytest.mark.parametrize("what", ["csv", "parquet"])
def test_an_empty_path_is_refused(revenue: DataFrame, what: str) -> None:
    with pytest.raises(CompileError, match=f"write.{what}\\(\\) needs a file path"):
        getattr(revenue.write, what)("")


def _as_text(table: pa.Table) -> list[list[str]]:
    return [
        ["" if value is None else str(value) for value in row.values()] for row in table.to_pylist()
    ]


# --------------------------------------------------------------------------------------
# to_polars() — behind the optional extra
# --------------------------------------------------------------------------------------


def test_to_polars_returns_a_polars_frame(revenue: DataFrame) -> None:
    polars = pytest.importorskip("polars", reason="the polars extra is not installed")
    expected = revenue.collect()
    frame: Any = revenue.to_polars()

    assert isinstance(frame, polars.DataFrame)
    assert frame.columns == list(expected.column_names)
    assert frame.height == expected.num_rows


def test_to_polars_says_which_extra_to_install_when_polars_is_absent(
    revenue: DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinned wording: the install command is the only useful part of an ImportError."""
    monkeypatch.setitem(sys.modules, "polars", None)

    with pytest.raises(ImportError) as error:
        revenue.to_polars()
    assert str(error.value) == (
        "to_polars() needs the optional dependency: pip install 'omniframes[polars]'"
    )


def test_the_camel_case_alias_exists(revenue: DataFrame) -> None:
    pytest.importorskip("polars", reason="the polars extra is not installed")

    assert revenue.toPolars().height == revenue.collect().num_rows
