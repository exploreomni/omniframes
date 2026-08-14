"""``df.write`` — the two file formats omniframes writes (docs/SQLTIER.md §6).

Deliberately small: a writer's job is to put the frame somewhere, not to grow a second API.
Both formats go through :meth:`~omniframes.dataframe.DataFrame.collect`, so what lands on disk
is exactly what a user would have seen — normalized, aliased, totals stripped — and Arrow does
the writing, so decimals stay decimals in Parquet rather than becoming floats on the way out.

::

    df.write.parquet("revenue.parquet")
    df.write.csv("revenue.csv", include_header=False)

The frame is fetched once per call: two formats means two queries.  That is the honest cost of a
lazy frame, and calling ``collect()`` yourself and writing the table twice is the way around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

import pyarrow as pa

from omniframes.errors import CompileError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omniframes.dataframe import DataFrame

__all__ = ["Compression", "DataFrameWriter"]

#: Parquet codecs pyarrow accepts. ``zstd`` is the default: the best ratio of the lot, and every
#: reader that matters has supported it for years.
Compression: TypeAlias = Literal["gzip", "bz2", "brotli", "lz4", "zstd", "snappy", "none"]


class DataFrameWriter:
    """``df.write`` — runs the frame and writes the result to a file."""

    __slots__ = ("_frame",)

    def __init__(self, frame: DataFrame) -> None:
        self._frame = frame

    def __repr__(self) -> str:
        return "DataFrameWriter(csv, parquet)"

    def csv(self, path: str | Path, *, include_header: bool = True) -> None:
        """Write the frame as CSV.

        CSV has no types: a decimal comes back as text and a NULL is an empty field.  Use
        :meth:`parquet` whenever the values matter more than the readability.
        """
        from pyarrow import csv as pa_csv

        pa_csv.write_csv(
            self._collect(),
            _path(path, "csv"),
            write_options=pa_csv.WriteOptions(include_header=include_header),
        )

    def parquet(self, path: str | Path, *, compression: Compression = "zstd") -> None:
        """Write the frame as Parquet — a lossless round trip of the Arrow table."""
        from pyarrow import parquet as pa_parquet

        pa_parquet.write_table(self._collect(), _path(path, "parquet"), compression=compression)

    def _collect(self) -> pa.Table:
        return self._frame.collect()


def _path(path: str | Path, what: str) -> str:
    if not isinstance(path, str | Path) or not str(path):
        raise CompileError(f"write.{what}() needs a file path; got {path!r}")
    return str(path)
