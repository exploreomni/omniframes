"""The transport layer: everything that knows how to talk to Omni.

Nothing above this package may assume HTTP — :class:`QueryTransport` is the seam, and
:class:`HttpTransport` is merely the implementation that speaks to a real org.  The rest of the
package is pure data handling: NDJSON framing (:mod:`omniframes.transport.ndjson`), Arrow/summary
decoding (:mod:`omniframes.transport.arrow`) and result normalization
(:mod:`omniframes.transport.normalize`).

The wire contract every module here implements lives in ``docs/CONTRACT_NOTES.md``.
"""

from __future__ import annotations

from omniframes.transport.arrow import (
    check_missing_fields,
    decode_result,
    missing_fields,
    schema_from_summary,
)
from omniframes.transport.base import PlanResult, QueryResult, QueryTransport
from omniframes.transport.http import (
    DEFAULT_RATE_LIMIT_WAIT_SECONDS,
    HttpTransport,
    normalize_base_url,
)
from omniframes.transport.ndjson import (
    Footer,
    Header,
    JobLine,
    JobStatus,
    ParsedResponse,
    StreamAccumulator,
    TrailingError,
    parse_response,
)
from omniframes.transport.normalize import NormalizedResult, is_reserved_column, normalize

__all__ = [
    "DEFAULT_RATE_LIMIT_WAIT_SECONDS",
    "Footer",
    "Header",
    "HttpTransport",
    "JobLine",
    "JobStatus",
    "NormalizedResult",
    "ParsedResponse",
    "PlanResult",
    "QueryResult",
    "QueryTransport",
    "StreamAccumulator",
    "TrailingError",
    "check_missing_fields",
    "decode_result",
    "is_reserved_column",
    "missing_fields",
    "normalize",
    "normalize_base_url",
    "parse_response",
    "schema_from_summary",
]
