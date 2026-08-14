"""Typed parser for the ``/query/run`` + ``/query/wait`` NDJSON protocol.

The wire contract lives in ``docs/CONTRACT_NOTES.md`` §2.2 and is summarized here because every
rule below is load-bearing:

* ``200`` with ``Content-Type: text/ndjson``; one JSON object per ``\\n`` with a trailing
  separator after every line.  A mid-stream upstream failure appends a final **unterminated**
  line ``{"message": ..., "reason": ...}`` — parsing must tolerate it (:class:`TrailingError`).
* Line 1 is the header ``{"jobs_submitted": {job_id: client_result_id | null}}``; job ids come
  from this line only.
* Job lines carry ``status``, an **open enum**: only ``COMPLETE``/``ERROR``/``FAILED`` are known
  terminal states, ``ERROR``/``FAILED`` are failures, and any status we do not know is preserved
  verbatim and treated as non-terminal.
* ``status: "COMPLETE"`` whose ``summary.display_sql`` contains ``"Failed to plan query"`` is a
  real failure (:attr:`JobLine.failed_to_plan`).
* The footer is ``{"remaining_job_ids": [...], "timed_out": "true"|"false"}`` — ``timed_out`` is
  a **string**, and is ``"true"`` iff ``remaining_job_ids`` is non-empty.
* ``/query/wait`` responses are the same framing minus the header line; job lines appear exactly
  once across the whole run+wait cycle, so they are accumulated
  (:class:`StreamAccumulator`) while the footer is replaced by each new response.

This module is pure parsing: it never performs I/O and it raises only on protocol violations
(malformed JSON mid-stream, unrecognizable line shapes).  Mapping job failures onto the
:mod:`omniframes.errors` hierarchy is the HTTP transport's job.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

from omniframes.errors import TransportError

__all__ = [
    "FAILED_TO_PLAN_MARKER",
    "FAILURE_STATUSES",
    "REDACTED_ERROR_MESSAGE",
    "TERMINAL_STATUSES",
    "Footer",
    "Header",
    "JobLine",
    "JobStatus",
    "NdjsonLine",
    "ParsedResponse",
    "StreamAccumulator",
    "TrailingError",
    "iter_json_lines",
    "parse_line",
    "parse_response",
]

#: Substituted for ``error_message`` when the caller lacks the ``VIEW_SQL`` permission
#: (CONTRACT_NOTES §2.2); SQL fields inside ``summary`` are blanked at the same time.
REDACTED_ERROR_MESSAGE = (
    "Query failed. Error details are only visible to users with permission to view SQL."
)

#: A ``COMPLETE`` job whose ``summary.display_sql`` contains this marker actually failed.
FAILED_TO_PLAN_MARKER = "Failed to plan query"


class JobStatus(enum.StrEnum):
    """Job status — an **open** enum.

    Unknown values are preserved verbatim as pseudo-members (``JobStatus("WHATEVER").value ==
    "WHATEVER"``) instead of raising or collapsing to ``UNKNOWN``, because the server is free to
    add states.  Only :data:`TERMINAL_STATUSES` end the wait loop and only
    :data:`FAILURE_STATUSES` mean the job failed; everything else — known or not — is
    non-terminal.
    """

    ADDED = "ADDED"
    PLANNING = "PLANNING"
    PLANNING_COMPLETE = "PLANNING_COMPLETE"
    PLANNED = "PLANNED"
    EXECUTING = "EXECUTING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
    FAILED = "FAILED"

    @classmethod
    def _missing_(cls, value: object) -> JobStatus | None:
        if not isinstance(value, str):
            return None
        pseudo = str.__new__(cls, value)
        pseudo._name_ = value
        pseudo._value_ = value
        # setdefault keeps pseudo-members singletons, so identity comparisons stay meaningful.
        return cast("JobStatus", cls._value2member_map_.setdefault(value, pseudo))

    @classmethod
    def from_wire(cls, value: str | None) -> JobStatus:
        """Parse a wire ``status``.

        A missing status becomes the empty pseudo-member — neither terminal nor a failure, so a
        malformed line keeps the job in the polling loop instead of silently succeeding.
        """
        return cls(value if value is not None else "")

    @property
    def is_known(self) -> bool:
        """Whether this status is one of the values documented in CONTRACT_NOTES §2.2."""
        return self in _KNOWN_STATUSES

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a final state (unknown statuses are never terminal)."""
        return self in TERMINAL_STATUSES

    @property
    def is_failure(self) -> bool:
        """Whether this status by itself means the job failed."""
        return self in FAILURE_STATUSES


#: The only statuses that end a job.  Anything else — including statuses we have never seen —
#: means "still running", so the wait loop keeps polling.
TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.COMPLETE, JobStatus.ERROR, JobStatus.FAILED}
)

#: Terminal statuses that map onto a failure.
FAILURE_STATUSES: frozenset[JobStatus] = frozenset({JobStatus.ERROR, JobStatus.FAILED})

_KNOWN_STATUSES: frozenset[JobStatus] = frozenset(JobStatus)


@dataclass(frozen=True, slots=True)
class Header:
    """The first line of a ``/query/run`` response: ``{"jobs_submitted": {...}}``.

    Job ids are available **only** here — the upstream ``omni_job_ids`` HTTP header is not
    forwarded.  Values are the caller's ``client_result_id`` or ``null``.
    """

    jobs_submitted: Mapping[str, str | None]
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(self.jobs_submitted)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Header:
        submitted = payload.get("jobs_submitted") or {}
        jobs: dict[str, str | None] = {}
        if isinstance(submitted, Mapping):
            for job_id, client_result_id in submitted.items():
                jobs[str(job_id)] = None if client_result_id is None else str(client_result_id)
        return cls(jobs_submitted=jobs, raw=payload)


@dataclass(frozen=True, slots=True)
class JobLine:
    """One job's terminal (or intermediate) state line.

    Nulls are omitted on the wire, so every optional field defaults to ``None``.  ``raw`` keeps
    the complete payload for fields this dataclass does not model (``error``, ``requery_*``,
    ``column_name_mapping``, future additions).
    """

    job_id: str
    status: JobStatus
    client_result_id: str | None = None
    summary: Mapping[str, Any] | None = None
    query: Mapping[str, Any] | None = None
    result: str | None = None
    cache_metadata: Mapping[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None
    kill_reason: str | None = None
    stream_stats: Mapping[str, Any] | None = None
    requery_sql: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JobLine:
        return cls(
            job_id=str(payload.get("job_id", "")),
            status=JobStatus.from_wire(_as_str(payload.get("status"))),
            client_result_id=_as_str(payload.get("client_result_id")),
            summary=_as_mapping(payload.get("summary")),
            query=_as_mapping(payload.get("query")),
            result=_as_str(payload.get("result")),
            cache_metadata=_as_mapping(payload.get("cache_metadata")),
            error_type=_as_str(payload.get("error_type")),
            error_message=_as_str(payload.get("error_message")),
            kill_reason=_as_str(payload.get("kill_reason")),
            stream_stats=_as_mapping(payload.get("stream_stats")),
            requery_sql=_as_str(payload.get("requery_sql")),
            raw=payload,
        )

    @property
    def client_result_id_or_none(self) -> str | None:
        """``client_result_id`` with the literal string ``"null"`` normalized to ``None``.

        Error lines may carry the *string* ``"null"`` rather than a JSON null
        (CONTRACT_NOTES §2.2); :attr:`client_result_id` keeps whatever arrived.
        """
        if self.client_result_id is None or self.client_result_id == "null":
            return None
        return self.client_result_id

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def has_failure_status(self) -> bool:
        """``status`` alone says the job failed (``ERROR`` / ``FAILED``)."""
        return self.status.is_failure

    @property
    def display_sql(self) -> str | None:
        return _as_str((self.summary or {}).get("display_sql"))

    @property
    def failed_to_plan(self) -> bool:
        """The ``COMPLETE``-but-actually-broken case: planner failure hidden in ``display_sql``."""
        display_sql = self.display_sql
        return display_sql is not None and FAILED_TO_PLAN_MARKER in display_sql

    @property
    def needs_client_materialization(self) -> bool:
        """A requery line with ``requery_sql`` but no ``result``.

        Should not happen on the API path; we refuse it rather than silently returning nothing.
        """
        return self.requery_sql is not None and self.result is None

    @property
    def is_redacted(self) -> bool:
        """The error message was replaced because the caller lacks ``VIEW_SQL``."""
        return self.error_message == REDACTED_ERROR_MESSAGE

    @property
    def is_failure(self) -> bool:
        """Any kind of failure: bad status, hidden plan failure, or an unusable requery line."""
        return self.failure_reason is not None

    @property
    def failure_reason(self) -> str | None:
        """A human-readable failure description, or ``None`` when the job did not fail."""
        if self.has_failure_status:
            if self.error_message:
                return self.error_message
            return f"job {self.job_id} finished with status {self.status.value}"
        if self.failed_to_plan:
            return (
                f"job {self.job_id} reported {self.status.value} but the planner failed "
                f"({FAILED_TO_PLAN_MARKER!r} in summary.display_sql)"
            )
        if self.needs_client_materialization:
            return (
                f"job {self.job_id} returned requery_sql without a result; client-side "
                "materialization is not supported"
            )
        return None


@dataclass(frozen=True, slots=True)
class Footer:
    """The last framed line: which jobs are still running, and whether the wait window expired.

    ``timed_out`` arrives as the *string* ``"true"``/``"false"`` on the NDJSON path (only the
    ``resultType`` 408 body uses a real boolean), so both are accepted and normalized here.
    """

    remaining_job_ids: tuple[str, ...] = ()
    timed_out: bool = False
    timed_out_raw: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Footer:
        remaining_raw = payload.get("remaining_job_ids") or ()
        remaining: tuple[str, ...]
        if isinstance(remaining_raw, str):
            remaining = (remaining_raw,)
        else:
            remaining = tuple(str(job_id) for job_id in remaining_raw)
        timed_out = payload.get("timed_out")
        return cls(
            remaining_job_ids=remaining,
            timed_out=_parse_timed_out(timed_out),
            timed_out_raw=None if timed_out is None else str(timed_out),
            raw=payload,
        )


@dataclass(frozen=True, slots=True)
class TrailingError:
    """The tolerated tail: ``{"message": ..., "reason": ...}`` appended on upstream failure.

    The server writes it *without* the trailing separator and may cut it off mid-object, so
    ``raw`` is ``None`` when the text was not decodable JSON; ``text`` always holds the bytes as
    received.
    """

    message: str | None = None
    reason: str | None = None
    text: str = ""
    raw: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

    @property
    def is_json(self) -> bool:
        """Whether the trailing line decoded as JSON (``False`` when the stream was truncated)."""
        return self.raw is not None

    @property
    def detail(self) -> str:
        """Best available description of the upstream failure."""
        parts = [part for part in (self.message, self.reason) if part]
        return ": ".join(parts) if parts else self.text

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], text: str) -> TrailingError:
        return cls(
            message=_as_str(payload.get("message")),
            reason=_as_str(payload.get("reason")),
            text=text,
            raw=payload,
        )


NdjsonLine: TypeAlias = Header | JobLine | Footer | TrailingError


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    """Everything one HTTP response body contained, in arrival order."""

    lines: tuple[NdjsonLine, ...] = ()
    header: Header | None = None
    jobs: tuple[JobLine, ...] = ()
    footer: Footer | None = None
    trailing_error: TrailingError | None = None

    @property
    def remaining_job_ids(self) -> tuple[str, ...]:
        return self.footer.remaining_job_ids if self.footer is not None else ()

    @property
    def timed_out(self) -> bool:
        return self.footer.timed_out if self.footer is not None else False


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, Any]", value)
    return None


def _parse_timed_out(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def iter_json_lines(body: str | bytes) -> Iterator[str]:
    """Yield the non-empty text of each NDJSON line in ``body``.

    Every framed line ends with ``\\n``; the tolerated trailing error line does not.  Blank
    chunks (from the trailing separator) are skipped.
    """
    if isinstance(body, bytes | bytearray):
        try:
            text = bytes(body).decode("utf-8")
        except UnicodeDecodeError as exc:  # pragma: no cover - defensive
            raise TransportError("NDJSON response body is not valid UTF-8") from exc
    else:
        text = body
    for chunk in text.split("\n"):
        stripped = chunk.strip("\r").strip()
        if stripped:
            yield stripped


def _classify(payload: Mapping[str, Any], text: str) -> NdjsonLine:
    if "jobs_submitted" in payload:
        return Header.from_payload(payload)
    if "job_id" in payload:
        return JobLine.from_payload(payload)
    if "remaining_job_ids" in payload or "timed_out" in payload:
        return Footer.from_payload(payload)
    if "message" in payload or "reason" in payload:
        return TrailingError.from_payload(payload, text)
    raise TransportError(f"unrecognized NDJSON line shape: {_snippet(text)}")


def parse_line(text: str) -> NdjsonLine:
    """Parse a single, complete NDJSON line into its typed form.

    Raises :class:`~omniframes.errors.TransportError` for malformed JSON or an unrecognized
    object shape.  Truncated tails are handled by :func:`parse_response`, not here.
    """
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise TransportError(f"malformed JSON in NDJSON line: {_snippet(text)}") from exc
    if not isinstance(payload, dict):
        raise TransportError(f"NDJSON line is not a JSON object: {_snippet(text)}")
    return _classify(cast("Mapping[str, Any]", payload), text)


def parse_response(body: str | bytes) -> ParsedResponse:
    """Parse a full ``/query/run`` or ``/query/wait`` response body.

    Tolerant exactly where the contract says to be: a final line that does not decode as JSON
    becomes a :class:`TrailingError` carrying the raw text.  A malformed line anywhere else is a
    protocol violation and raises :class:`~omniframes.errors.TransportError`.
    """
    chunks = list(iter_json_lines(body))
    lines: list[NdjsonLine] = []
    header: Header | None = None
    jobs: list[JobLine] = []
    footer: Footer | None = None
    trailing_error: TrailingError | None = None

    for index, text in enumerate(chunks):
        is_last = index == len(chunks) - 1
        line: NdjsonLine
        try:
            line = parse_line(text)
        except TransportError:
            if not is_last:
                raise
            # A truncated tail is the documented upstream-failure shape; keep the raw text.
            line = TrailingError(text=text)
        lines.append(line)
        if isinstance(line, Header):
            if header is None:
                header = line
        elif isinstance(line, JobLine):
            jobs.append(line)
        elif isinstance(line, Footer):
            footer = line
        else:
            trailing_error = line

    return ParsedResponse(
        lines=tuple(lines),
        header=header,
        jobs=tuple(jobs),
        footer=footer,
        trailing_error=trailing_error,
    )


def _snippet(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…"


class StreamAccumulator:
    """Merges the run response with each successive ``/query/wait`` response.

    Contract rules this encodes (CONTRACT_NOTES §2.2):

    * the header appears once, on the run response only — the first one wins;
    * **job lines appear exactly once** across the whole cycle, so they accumulate; a repeated
      ``job_id`` keeps the **terminal** line (ties keep the one already held) and is recorded in
      :attr:`duplicate_job_ids`;
    * the footer is *replaced* by every response (a poll that times out returns only a footer);
    * :attr:`remaining_job_ids` from the latest footer drives the polling loop.
    """

    __slots__ = (
        "_duplicate_job_ids",
        "_footer",
        "_header",
        "_jobs",
        "_response_count",
        "_trailing_error",
    )

    def __init__(self) -> None:
        self._header: Header | None = None
        self._jobs: dict[str, JobLine] = {}
        self._duplicate_job_ids: list[str] = []
        self._footer: Footer | None = None
        self._trailing_error: TrailingError | None = None
        self._response_count = 0

    def add_response(self, body: str | bytes) -> ParsedResponse:
        """Parse and merge one response body; returns what that body contained."""
        parsed = parse_response(body)
        self.add_parsed(parsed)
        return parsed

    def add_parsed(self, parsed: ParsedResponse) -> None:
        """Merge an already-parsed response."""
        self._response_count += 1
        if parsed.header is not None and self._header is None:
            self._header = parsed.header
        for job in parsed.jobs:
            previous = self._jobs.get(job.job_id)
            if previous is not None:
                self._duplicate_job_ids.append(job.job_id)
                # A repeat keeps the TERMINAL line, not the first one.  §2.2 says job lines
                # arrive once per job, but the same section documents ``status`` as an open enum
                # whose ADDED/EXECUTING/PLANNING values exist and must be read as non-terminal —
                # so an intermediate line in the run response followed by the real answer in a
                # ``/query/wait`` slice is a sequence the contract allows (and one the repo's own
                # ``unknown_status.ndjson`` fixture contains).  First-wins is the only dedup
                # policy that can discard a successful result, so terminality decides; a tie
                # keeps the line already held.
                if previous.is_terminal or not job.is_terminal:
                    continue
            self._jobs[job.job_id] = job
        if parsed.footer is not None:
            self._footer = parsed.footer
        if parsed.trailing_error is not None:
            self._trailing_error = parsed.trailing_error

    @property
    def response_count(self) -> int:
        return self._response_count

    @property
    def header(self) -> Header | None:
        return self._header

    @property
    def footer(self) -> Footer | None:
        return self._footer

    @property
    def trailing_error(self) -> TrailingError | None:
        return self._trailing_error

    @property
    def jobs(self) -> tuple[JobLine, ...]:
        """Every job line seen, in arrival order."""
        return tuple(self._jobs.values())

    @property
    def jobs_by_id(self) -> Mapping[str, JobLine]:
        return dict(self._jobs)

    @property
    def duplicate_job_ids(self) -> tuple[str, ...]:
        """Job ids that arrived more than once (a contract violation we survive)."""
        return tuple(self._duplicate_job_ids)

    @property
    def submitted_job_ids(self) -> tuple[str, ...]:
        """Job ids from the header, i.e. everything the run call started."""
        return self._header.job_ids if self._header is not None else ()

    @property
    def remaining_job_ids(self) -> tuple[str, ...]:
        """``remaining_job_ids`` exactly as the latest footer reported it."""
        return self._footer.remaining_job_ids if self._footer is not None else ()

    @property
    def pending_job_ids(self) -> tuple[str, ...]:
        """:attr:`remaining_job_ids` minus jobs that already delivered a terminal line.

        Belt and braces for the polling loop: we never wait on a job we have already resolved.
        """
        return tuple(
            job_id
            for job_id in self.remaining_job_ids
            if job_id not in self._jobs or not self._jobs[job_id].is_terminal
        )

    @property
    def timed_out(self) -> bool:
        """The latest footer's ``timed_out``, parsed from its string form."""
        return self._footer.timed_out if self._footer is not None else False

    @property
    def is_done(self) -> bool:
        """No job is still outstanding — the polling loop can stop."""
        return not self.pending_job_ids

    @property
    def failures(self) -> tuple[JobLine, ...]:
        """Job lines that failed, including the ``COMPLETE``-but-failed-to-plan case."""
        return tuple(job for job in self._jobs.values() if job.is_failure)

    def job(self, job_id: str) -> JobLine:
        """Look up one job line; raises :class:`KeyError` when it has not arrived."""
        return self._jobs[job_id]

    def __repr__(self) -> str:
        return (
            f"StreamAccumulator(responses={self._response_count}, jobs={len(self._jobs)}, "
            f"pending={len(self.pending_job_ids)}, timed_out={self.timed_out})"
        )
