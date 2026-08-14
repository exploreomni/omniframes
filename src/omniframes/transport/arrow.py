"""Decoding of the ``result`` payload and of ``summary`` metadata (CONTRACT_NOTES §2.3/§2.6).

``result`` is base64 of an Arrow IPC **stream** (not a file), and ``summary.fields`` is the only
authoritative schema for interpreting it — never the catalog.  ``summary.missing_fields`` is the
trap this module also covers: when it is non-empty the server silently dropped fields the caller
asked for, so the query "succeeded" while returning the wrong shape.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import Any

import pyarrow as pa

from omniframes.errors import QueryError, TransportError
from omniframes.transport.normalize import collapse_grain_names
from omniframes.types import OmniField, OmniSchema

__all__ = [
    "check_missing_fields",
    "decode_result",
    "missing_fields",
    "schema_from_summary",
]


def decode_result(b64: str) -> pa.Table:
    """Decode a job line's base64 ``result`` into a :class:`pyarrow.Table`.

    The payload is an Arrow IPC *stream*, so it is read with ``ipc.open_stream``; an IPC *file*
    reader would reject it.
    """
    try:
        payload = base64.b64decode(b64, validate=False)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise TransportError("job result is not valid base64") from exc
    if not payload:
        raise TransportError("job result is empty; expected an Arrow IPC stream")
    try:
        with pa.ipc.open_stream(pa.BufferReader(payload)) as reader:
            return reader.read_all()
    except (pa.ArrowInvalid, ValueError, OSError) as exc:
        raise TransportError("job result is not a readable Arrow IPC stream") from exc


def schema_from_summary(fields: Mapping[str, Any] | None) -> OmniSchema:
    """Build an :class:`~omniframes.types.OmniSchema` from ``summary.fields``.

    ``fields`` is ``Record<fieldName, Field>``; its JSON order is the server's field order and is
    preserved.  Entries that are not objects are kept as bare names with ``UNKNOWN`` type rather
    than dropped, so a schema never silently loses a column.

    A formatted grain arrives as two entries — the ``__raw`` timestamp and the formatted string —
    and is collapsed exactly as the result columns are (docs/SQLTIER.md §5), so the schema keeps
    describing the frame the user gets: one ``field[grain]`` field, with the ``__raw`` half's
    ``TIMESTAMP`` type and position.
    """
    if not fields:
        return OmniSchema(())
    payloads = {str(name): payload for name, payload in fields.items()}
    selected, names = collapse_grain_names(tuple(payloads))
    parsed: list[OmniField] = []
    for wire, name in zip(selected, names, strict=True):
        payload = payloads[wire]
        attributes = payload if isinstance(payload, Mapping) else {}
        parsed.append(OmniField.from_wire(name, dict(attributes)))
    return OmniSchema(tuple(parsed))


def missing_fields(summary: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Fields the server dropped from the query (``summary.missing_fields``).

    Non-empty means a requested field was bad (unknown name, invalid grain, wrong grain kind) and
    was silently removed — the job still reports success.
    """
    if not summary:
        return ()
    raw = summary.get("missing_fields") or ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(name) for name in raw)


def check_missing_fields(summary: Mapping[str, Any] | None, *, job_id: str | None = None) -> None:
    """Raise :class:`~omniframes.errors.QueryError` when ``summary.missing_fields`` is non-empty.

    This is the "surfacing" half of :func:`missing_fields`: the server treats dropped fields as a
    success, and we refuse to hand back a frame whose columns are not the ones that were asked
    for.
    """
    dropped = missing_fields(summary)
    if not dropped:
        return
    names = ", ".join(dropped)
    raise QueryError(
        f"the query omitted {len(dropped)} requested field(s): {names}. "
        "Check the field name, its view, and any [grain] suffix.",
        error_type="MISSING_FIELDS",
        job_id=job_id,
    )
