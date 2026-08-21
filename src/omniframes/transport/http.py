"""The httpx implementation of :class:`~omniframes.transport.base.QueryTransport`.

Everything here follows ``docs/CONTRACT_NOTES.md``; the section references in the docstrings
point at it.  The load-bearing behaviors:

* **Base URL normalization** — users hand us ``acme.omni.co``, ``https://acme.omni.co`` or a URL
  that already ends in ``/api/v1``.  :func:`normalize_base_url` turns all of them into the same
  origin, because every path this module builds starts with ``/api/v1``.
* **The run → wait loop** (§2.2) — ``POST /api/v1/query/run`` waits ~10 s server-side, then
  reports what is still running in the NDJSON *footer*; ``GET /api/v1/query/wait`` slices are
  ~30 s.  Job lines arrive exactly once and are accumulated by
  :class:`~omniframes.transport.ndjson.StreamAccumulator` while the footer is replaced.  The
  loop is bounded by a client-side deadline, and sleeps between polls whose pending set did not
  shrink.  Timeouts are **in-band** on this path (HTTP 200 + a footer), never a 408.
* **Error-envelope mapping** (§1) — the server speaks three different error shapes and uses 403
  (not 401) for a bad token.  :meth:`HttpTransport._map_error` folds all of them onto the
  ``omniframes.errors`` hierarchy so nothing above the transport ever sees an httpx exception or
  a status code.
* **Retries** — 429 comes from the AWS WAF (60 req/min shared by every client using the same
  ``Authorization`` value). Idempotent GETs honor numeric ``Retry-After`` values within a
  cumulative wait budget; POSTs retain bounded exponential backoff with jitter. Connect/timeout
  failures are retried for idempotent GETs only: a ``/query/run`` POST is never replayed, since
  a retry would submit a second job.
* **Secrecy** — the API key lives in exactly one attribute, is never formatted into a message,
  and any server text that happens to echo it is scrubbed before it reaches an exception.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Mapping
from math import isfinite
from types import TracebackType
from typing import Any, Final
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from omniframes import __version__
from omniframes.errors import (
    AuthError,
    FeatureFlagError,
    ModelPermissionError,
    OmniframesError,
    QueryError,
    QueryTimeoutError,
    TransportError,
)
from omniframes.transport.arrow import check_missing_fields, decode_result
from omniframes.transport.base import PlanResult, QueryResult
from omniframes.transport.ndjson import JobLine, StreamAccumulator

__all__ = [
    "DEFAULT_MAX_DEADLINE_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_RATE_LIMIT_WAIT_SECONDS",
    "DEFAULT_TIMEOUT",
    "REDACTED_API_KEY",
    "USER_AGENT",
    "WAF_ACTION_HEADER",
    "WORKBOOK_URL_HEADER",
    "HttpTransport",
    "normalize_base_url",
]

#: What the API key is replaced by everywhere it could otherwise be seen.
REDACTED_API_KEY: Final = "omni_osk_***"

#: ``User-Agent`` sent by clients this transport constructs itself.
USER_AGENT: Final = f"omniframes/{__version__}"

#: Response header carrying the workbook URL when the envelope asked for ``workbookUrl: true``.
WORKBOOK_URL_HEADER: Final = "X-Omni-Workbook-Url"

#: Set to ``block`` when the AWS WAF rate limiter, not the application, produced the 429.
WAF_ACTION_HEADER: Final = "X-Omni-Waf-Action"

#: Generous read timeout: a ``/query/run`` call blocks server-side for its whole wait window and
#: a ``/query/wait`` slice is ~30 s, so the default has to leave plenty of room above both.
DEFAULT_TIMEOUT: Final = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

#: Ceiling on the client-side wall time of one :meth:`HttpTransport.run` call.
DEFAULT_MAX_DEADLINE_SECONDS: Final = 600.0

#: Pause before re-polling ``/query/wait`` when the pending set did not shrink.
DEFAULT_POLL_INTERVAL_SECONDS: Final = 1.0

#: One full 60-second WAF window, plus enough slack for a request to clear it.
DEFAULT_RATE_LIMIT_WAIT_SECONDS: Final = 65.0

_RUN_PATH: Final = "/api/v1/query/run"
_WAIT_PATH: Final = "/api/v1/query/wait"
_WHOAMI_PATH: Final = "/api/v1/whoami"
_MODELS_PATH: Final = "/api/v1/models"
_GENERATE_QUERY_PATH: Final = "/api/v1/ai/generate-query"

#: Trailing ``/api`` or ``/api/vN`` on a user-supplied base URL — stripped, since every path this
#: module builds already carries the full ``/api/v1`` prefix.
_API_SUFFIX_RE: Final = re.compile(r"/api(?:/v\d+)?/?$", re.IGNORECASE)

#: Messages that mean "the credential itself was rejected" (CONTRACT_NOTES §1).
_AUTH_MARKERS: Final = (
    "invalid bearer token",
    "bad authorization header",
    "required permissions to use this api key",
    "user-scoped api keys can only be used",
    "expired",
)
_FEATURE_FLAG_MARKER: Final = "feature not enabled"
_PERMISSION_DENIED_MARKER: Final = "permission denied"

#: httpx failures that mean "the request never got an answer" and are therefore retryable for
#: idempotent methods.
_NETWORK_ERRORS: Final = (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError)

_BACKOFF_BASE_SECONDS: Final = 0.5
_MAX_BACKOFF_SECONDS: Final = 8.0
_JITTER_FRACTION: Final = 0.5
_MAX_ZERO_DELAY_RETRIES: Final = 8


def _segment(value: str) -> str:
    """Percent-encode one caller-supplied path segment.

    Topic names and document identifiers are strings a user hands in, and they may legitimately
    contain characters the URL grammar reserves.  Interpolated raw, a ``/`` adds a path segment,
    a ``?`` moves the rest of the path into the query string, a ``#`` truncates it and a ``..``
    walks up to a *different endpoint* — ``topic/../../whoami`` normalizes to ``/api/v1/whoami``.
    ``safe=""`` escapes ``/`` too, so a segment stays exactly one segment.
    """
    return quote(value, safe="")


def normalize_base_url(base_url: str) -> str:
    """Normalize a user-supplied Omni host into a bare origin.

    Accepts ``acme.omni.co``, ``https://acme.omni.co/``, ``https://acme.omni.co/api`` and
    ``https://acme.omni.co/api/v1`` — all of which become ``https://acme.omni.co``.  A missing
    scheme defaults to ``https``; any query string or fragment is dropped.

    Raises:
        TransportError: the value is empty, has no host, or uses a non-HTTP scheme.
    """
    raw = base_url.strip()
    if not raw:
        raise TransportError("base_url must not be empty (e.g. 'acme.omni.co')")
    if "://" not in raw:
        raw = f"https://{raw}"

    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise TransportError(f"base_url must be an http(s) URL; got scheme {parsed.scheme!r}")
    if not parsed.netloc:
        raise TransportError(f"base_url has no host: {base_url!r}")

    path = parsed.path
    while True:
        trimmed = _API_SUFFIX_RE.sub("", path)
        if trimmed == path:
            break
        path = trimmed
    return urlunsplit((scheme, parsed.netloc, path.rstrip("/"), "", ""))


class HttpTransport:
    """Talks to a real Omni org over HTTP.

    Args:
        base_url: the org host, in any of the forms :func:`normalize_base_url` accepts.
        api_key: an org API key or personal access token (``omni_osk_…``).  Stored privately and
            never rendered — see :meth:`__repr__`.
        timeout: httpx timeout configuration; a bare float sets every phase.  Defaults to
            :data:`DEFAULT_TIMEOUT`.
        max_deadline_seconds: ceiling on the wall time of one :meth:`run`, including every
            ``/query/wait`` poll.  A per-call ``deadline_seconds`` may lower it, not raise it.
        user_id: **membership** id to impersonate (CONTRACT_NOTES §1).  Sent as the preferred
            ``?userId=`` query parameter on query calls, and suppressed when the caller already
            put ``userId`` in the envelope body (supplying both is a 400).
        branch_id: default top-level ``branchId`` for run envelopes that do not carry one.
            ``branchId`` is top-level only — inside ``query`` it is a hard 400.
        client: an existing ``httpx.Client`` to use instead of constructing one (tests inject
            ``httpx.Client(transport=httpx.MockTransport(handler))``).  Injected clients are
            **not** closed by :meth:`close`.
        sleep: injectable ``time.sleep`` (poll pacing and retry backoff).
        monotonic: injectable ``time.monotonic`` (deadline accounting).
        jitter: injectable ``random.random``; returns a value in ``[0, 1)``.
        max_retries: bound on retries for POST 429s and for network failures on idempotent GETs.
        rate_limit_max_wait_seconds: cumulative waiting budget for 429 recovery on one
            idempotent GET. This bounds waiting, not wall time; a multi-request action may spend
            the budget once per request.
        poll_interval_seconds: pause between ``/query/wait`` polls that made no progress.
    """

    __slots__ = (
        "_api_key",
        "_base_url",
        "_branch_id",
        "_client",
        "_jitter",
        "_max_deadline_seconds",
        "_max_retries",
        "_monotonic",
        "_owns_client",
        "_poll_interval_seconds",
        "_rate_limit_max_wait",
        "_sleep",
        "_timeout",
        "_user_id",
    )

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: httpx.Timeout | float | None = None,
        max_deadline_seconds: float = DEFAULT_MAX_DEADLINE_SECONDS,
        user_id: str | None = None,
        branch_id: str | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
        max_retries: int = 3,
        rate_limit_max_wait_seconds: float = DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        if not api_key or not api_key.strip():
            raise TransportError("api_key must not be empty (set OMNI_API_KEY)")
        if max_deadline_seconds <= 0:
            raise TransportError("max_deadline_seconds must be positive")
        if max_retries < 0:
            raise TransportError("max_retries must not be negative")
        rate_limit_wait: object = rate_limit_max_wait_seconds
        if (
            isinstance(rate_limit_wait, bool)
            or not isinstance(rate_limit_wait, int | float)
            or not isfinite(rate_limit_wait)
            or rate_limit_wait < 0
        ):
            raise TransportError(
                "rate_limit_max_wait_seconds must be a finite, non-negative number"
            )

        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key.strip()
        self._timeout = _coerce_timeout(timeout)
        self._max_deadline_seconds = float(max_deadline_seconds)
        self._user_id = user_id
        self._branch_id = branch_id
        self._sleep = sleep
        self._monotonic = monotonic
        self._jitter = jitter
        self._max_retries = max_retries
        self._rate_limit_max_wait = float(rate_limit_wait)
        self._poll_interval_seconds = float(poll_interval_seconds)

        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.Client(headers=self._auth_headers(), timeout=self._timeout)
        )

    # -- identity ----------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The normalized origin every request is built from."""
        return self._base_url

    @property
    def user_id(self) -> str | None:
        """The impersonated membership id, if any."""
        return self._user_id

    @property
    def branch_id(self) -> str | None:
        """The default ``branchId`` applied to run envelopes that lack one."""
        return self._branch_id

    @property
    def rate_limit_max_wait_seconds(self) -> float:
        """The cumulative 429 wait budget for one idempotent GET request."""
        return self._rate_limit_max_wait

    def __repr__(self) -> str:
        """Never renders the key — that is the whole point of this method existing."""
        return f"HttpTransport(base_url={self._base_url!r}, api_key={REDACTED_API_KEY})"

    def __enter__(self) -> HttpTransport:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying client — but only if this transport created it."""
        if self._owns_client:
            self._client.close()

    # -- query execution ---------------------------------------------------------------

    def run(
        self, envelope: dict[str, Any], *, deadline_seconds: float | None = None
    ) -> QueryResult:
        """Execute a run envelope to a terminal result, driving the whole wait loop.

        Raises:
            QueryError: a job reached a terminal failure — including the two disguised ones:
                ``COMPLETE`` with ``"Failed to plan query"`` in ``summary.display_sql``, and a
                success whose ``summary.missing_fields`` is non-empty.
            QueryTimeoutError: the client-side deadline elapsed with jobs still running.
            TransportError: protocol failure (upstream tail, missing result payload, HTTP error).
        """
        accumulator, workbook_url = self._drive(
            envelope, plan_only=False, deadline_seconds=deadline_seconds
        )
        self._raise_first_failure(accumulator)
        job = self._primary_job(accumulator)
        if job.result is None:
            raise TransportError(
                f"job {job.job_id} finished with status {job.status.value} but carried no "
                "result payload"
            )
        table = decode_result(job.result)
        check_missing_fields(job.summary, job_id=job.job_id)
        return QueryResult(
            job_id=job.job_id,
            table=table,
            summary=dict(job.summary or {}),
            cache_metadata=dict(job.cache_metadata or {}),
            query=dict(job.query or {}),
            workbook_url=workbook_url,
        )

    def plan(self, envelope: dict[str, Any]) -> PlanResult:
        """Run the envelope with ``planOnly: true`` — schema and SQL, no execution (§2.4).

        ``planOnly`` is forced on, and ``resultType``/``formatResults``/``workbookUrl`` are
        dropped because the server 400s each of those combinations.  The ``PLANNED`` line
        normally arrives inside the run window; if the footer still lists the job, the same wait
        loop :meth:`run` uses takes over.  ``summary.missing_fields`` is checked here too — a
        plan that silently dropped fields would otherwise become a wrong ``df.schema``.
        """
        accumulator, _ = self._drive(envelope, plan_only=True, deadline_seconds=None)
        self._raise_first_failure(accumulator)
        job = self._primary_job(accumulator)
        if job.summary is None:
            raise TransportError(
                f"planOnly job {job.job_id} returned status {job.status.value} without a "
                "summary; summary.fields is the only schema authority"
            )
        check_missing_fields(job.summary, job_id=job.job_id)
        return PlanResult(
            job_id=job.job_id,
            summary=dict(job.summary),
            query=dict(job.query or {}),
        )

    # -- catalog -----------------------------------------------------------------------

    def whoami(self, model_ids: tuple[str, ...] = ()) -> dict[str, Any]:
        """``GET /api/v1/whoami`` — works even when the ``query-api`` flag is off (§1)."""
        params = {"modelId": ",".join(model_ids)} if model_ids else None
        return _json_object(self._send("GET", _WHOAMI_PATH, params=params))

    def list_models(self, **params: Any) -> dict[str, Any]:
        """One page of ``GET /api/v1/models``.

        The server's param schema is strict (unknown params 400), so ``None`` values are dropped
        rather than sent; everything else is passed through verbatim, ``cursor`` included.
        """
        query = {key: value for key, value in params.items() if value is not None}
        return _json_object(self._send("GET", _MODELS_PATH, params=query or None))

    def list_topics(self, model_id: str) -> dict[str, Any]:
        """``GET /api/v1/models/{modelId}/topic``."""
        return _json_object(self._send("GET", f"{_MODELS_PATH}/{_segment(model_id)}/topic"))

    def get_topic(self, model_id: str, topic_name: str) -> dict[str, Any]:
        """``GET /api/v1/models/{modelId}/topic/{topicName}`` — the field-metadata endpoint."""
        path = f"{_MODELS_PATH}/{_segment(model_id)}/topic/{_segment(topic_name)}"
        return _json_object(self._send("GET", path))

    def document_queries(self, document_identifier: str) -> dict[str, Any]:
        """``GET /api/v1/documents/{identifier}/queries``."""
        path = f"/api/v1/documents/{_segment(document_identifier)}/queries"
        return _json_object(self._send("GET", path))

    def generate_query(self, body: dict[str, Any]) -> dict[str, Any]:
        """``POST /api/v1/ai/generate-query``.

        ``runQuery`` is forced to ``False``: its server-side default is ``true``, which would run
        the generated query outside our pipeline (and would additionally require the
        ``query-api`` flag).
        """
        payload = dict(body)
        payload["runQuery"] = False
        return _json_object(self._send("POST", _GENERATE_QUERY_PATH, json_body=payload))

    # -- the run/wait loop -------------------------------------------------------------

    def _drive(
        self,
        envelope: Mapping[str, Any],
        *,
        plan_only: bool,
        deadline_seconds: float | None,
    ) -> tuple[StreamAccumulator, str | None]:
        """POST the envelope, then poll ``/query/wait`` until every job is terminal."""
        body = self._prepare_envelope(envelope, plan_only=plan_only)
        permission = _permission_hint(body)
        budget = self._budget(deadline_seconds)
        started = self._monotonic()

        response = self._send(
            "POST",
            _RUN_PATH,
            params=self._impersonation_params(body),
            json_body=body,
            permission=permission,
        )
        workbook_url = response.headers.get(WORKBOOK_URL_HEADER)
        accumulator = StreamAccumulator()
        self._absorb(accumulator, response)

        previous: frozenset[str] | None = None
        while not accumulator.is_done:
            pending = accumulator.pending_job_ids
            self._check_deadline(accumulator, started, budget)
            if previous is not None and not frozenset(pending) < previous:
                # No job finished during the last slice; pace ourselves before asking again.
                self._sleep(self._poll_interval_seconds)
                self._check_deadline(accumulator, started, budget)
            previous = frozenset(pending)
            response = self._send(
                "GET",
                _WAIT_PATH,
                params={"jobIds": ",".join(pending)},
                permission=permission,
            )
            self._absorb(accumulator, response)

        return accumulator, workbook_url

    def _prepare_envelope(self, envelope: Mapping[str, Any], *, plan_only: bool) -> dict[str, Any]:
        body = dict(envelope)
        if self._branch_id is not None and body.get("branchId") is None:
            body["branchId"] = self._branch_id
        if plan_only:
            body["planOnly"] = True
            for incompatible in ("resultType", "formatResults", "workbookUrl"):
                body.pop(incompatible, None)
        return body

    def _impersonation_params(self, body: Mapping[str, Any]) -> dict[str, str] | None:
        """``?userId=`` unless the envelope already carries the legacy body field (both → 400)."""
        if self._user_id is None or body.get("userId") is not None:
            return None
        return {"userId": self._user_id}

    def _budget(self, deadline_seconds: float | None) -> float:
        if deadline_seconds is None:
            return self._max_deadline_seconds
        return min(float(deadline_seconds), self._max_deadline_seconds)

    def _check_deadline(
        self, accumulator: StreamAccumulator, started: float, budget: float
    ) -> None:
        if self._monotonic() - started < budget:
            return
        remaining = accumulator.pending_job_ids
        raise QueryTimeoutError(
            f"the query deadline of {budget:g}s elapsed with {len(remaining)} job(s) still "
            f"running: {', '.join(remaining) or '(none reported)'}. The jobs keep running "
            "server-side; raise deadline_seconds or narrow the query.",
            remaining_job_ids=remaining,
        )

    def _absorb(self, accumulator: StreamAccumulator, response: httpx.Response) -> None:
        accumulator.add_response(response.content)
        trailing = accumulator.trailing_error
        if trailing is not None:
            raise TransportError(
                "the Omni API stream ended with an upstream failure: "
                f"{self._scrub(trailing.detail)}"
            )

    # -- job selection ------------------------------------------------------------------

    def _ordered_jobs(self, accumulator: StreamAccumulator) -> tuple[JobLine, ...]:
        """Job lines in submission order (header order), with any unheralded ones appended."""
        by_id = accumulator.jobs_by_id
        ordered = [by_id[job_id] for job_id in accumulator.submitted_job_ids if job_id in by_id]
        seen = {job.job_id for job in ordered}
        ordered.extend(job for job in accumulator.jobs if job.job_id not in seen)
        return tuple(ordered)

    def _primary_job(self, accumulator: StreamAccumulator) -> JobLine:
        """The job whose outcome this call returns.

        ``/query/run`` submits a single job per call — ``staticQueryReferences`` fold into one
        plan (§3.5) — so "the first job the header announced" is the whole story in practice.
        """
        ordered = self._ordered_jobs(accumulator)
        if not ordered:
            raise TransportError(
                "the Omni API returned no job lines; expected one per submitted job "
                "(CONTRACT_NOTES §2.2)"
            )
        return ordered[0]

    def _raise_first_failure(self, accumulator: StreamAccumulator) -> None:
        for job in self._ordered_jobs(accumulator):
            if job.is_failure:
                reason = job.failure_reason or f"job {job.job_id} failed"
                # The redacted-message case (no VIEW_SQL) flows through here unchanged: the
                # server already replaced error_message, and JobLine.failure_reason returns it.
                raise QueryError(self._scrub(reason), error_type=job.error_type, job_id=job.job_id)

    # -- HTTP --------------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT}

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any | None = None,
        permission: str | None = None,
    ) -> httpx.Response:
        """One request, with bounded retries, mapped onto the omniframes error hierarchy."""
        url = f"{self._base_url}{path}"
        # Only idempotent reads may be replayed: retrying POST /query/run would submit a second
        # job, and we cannot tell a lost response from a lost request.
        retry_network = method == "GET"
        network_attempts = 0
        throttle_attempts = 0
        throttle_waited = 0.0
        last_retry_after: str | None = None
        zero_delay_retries = 0

        while True:
            try:
                response = self._client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=self._auth_headers(),
                    timeout=self._timeout,
                )
            except _NETWORK_ERRORS as exc:
                if not retry_network or network_attempts >= self._max_retries:
                    raise TransportError(
                        f"could not reach the Omni API ({method} {path}) after "
                        f"{network_attempts + 1} attempt(s): {self._scrub(str(exc))}"
                    ) from exc
                network_attempts += 1
                self._sleep(self._backoff_delay(network_attempts))
                continue

            if response.status_code == 429:
                if retry_network:
                    throttle_attempts += 1
                    last_retry_after = response.headers.get("Retry-After")
                    remaining = self._rate_limit_max_wait - throttle_waited
                    retry_after = _retry_after(response)

                    if retry_after is not None:
                        delay = retry_after
                        if delay > remaining:
                            raise self._rate_limit_error(
                                response,
                                throttle_attempts,
                                method=method,
                                path=path,
                                waited=throttle_waited,
                                budget=self._rate_limit_max_wait,
                                last_retry_after=last_retry_after,
                                requested_wait=delay,
                                remaining_budget=max(remaining, 0.0),
                            )
                    else:
                        if remaining <= 0:
                            raise self._rate_limit_error(
                                response,
                                throttle_attempts,
                                method=method,
                                path=path,
                                waited=throttle_waited,
                                budget=self._rate_limit_max_wait,
                                last_retry_after=last_retry_after,
                            )
                        exponent = min(throttle_attempts - 1, 6)
                        delay = min(
                            _BACKOFF_BASE_SECONDS * 2.0**exponent,
                            _MAX_BACKOFF_SECONDS,
                        ) * (1.0 + _JITTER_FRACTION * self._jitter())
                        delay = min(delay, remaining)

                    if delay == 0:
                        if remaining <= 0 or zero_delay_retries >= _MAX_ZERO_DELAY_RETRIES:
                            raise self._rate_limit_error(
                                response,
                                throttle_attempts,
                                method=method,
                                path=path,
                                waited=throttle_waited,
                                budget=self._rate_limit_max_wait,
                                last_retry_after=last_retry_after,
                            )
                        zero_delay_retries += 1
                    else:
                        zero_delay_retries = 0

                    self._sleep(delay)
                    throttle_waited += delay
                    continue

                if throttle_attempts >= self._max_retries:
                    raise self._rate_limit_error(response, throttle_attempts + 1)
                throttle_attempts += 1
                self._sleep(self._backoff_delay(throttle_attempts, _retry_after(response)))
                continue

            if response.status_code >= 400:
                raise self._map_error(response, permission=permission)
            return response

    def _backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff with jitter, capped; an explicit ``Retry-After`` wins."""
        if retry_after is not None:
            return min(retry_after, _MAX_BACKOFF_SECONDS)
        delay = min(_BACKOFF_BASE_SECONDS * 2.0 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
        return delay * (1.0 + _JITTER_FRACTION * self._jitter())

    def _rate_limit_error(
        self,
        response: httpx.Response,
        attempts: int,
        *,
        method: str | None = None,
        path: str | None = None,
        waited: float | None = None,
        budget: float | None = None,
        last_retry_after: str | None = None,
        requested_wait: float | None = None,
        remaining_budget: float | None = None,
    ) -> TransportError:
        """Build a credential-safe 429 diagnostic without ever reading the response body."""
        waf = response.headers.get(WAF_ACTION_HEADER)
        source = f" (WAF action: {waf})" if waf else ""
        if method is None or path is None or waited is None or budget is None:
            return TransportError(
                self._scrub(
                    f"the Omni API rate-limited this request{source} and it still failed after "
                    f"{attempts} attempts. Rate limiting is keyed on the API key itself: every "
                    "client sharing the key shares one 60 requests/minute bucket. Slow down, or "
                    "use a separate key for this workload."
                ),
                status=response.status_code,
            )

        retry_after = last_retry_after if last_retry_after is not None else "absent"
        requested = ""
        if requested_wait is not None and remaining_budget is not None:
            requested = (
                f" The server asked for a {requested_wait:g}s wait, which exceeds the remaining "
                f"rate-limit budget ({remaining_budget:g}s of {budget:g}s); raise it via "
                f"OmniSession.builder.rate_limit_wait(seconds) or retry after {requested_wait:g}s."
            )
        return TransportError(
            self._scrub(
                f"the Omni API rate-limited {method} {path}{source}: {attempts} 429 response(s) "
                f"after waiting {waited:g}s of its {budget:g}s rate-limit budget; last "
                f"Retry-After: {retry_after}. Rate limiting is keyed on the API key itself: every "
                "client sharing the key shares one 60 requests/minute bucket. Slow down, or use "
                f"a separate key for this workload.{requested}"
            ),
            status=response.status_code,
        )

    def _map_error(self, response: httpx.Response, *, permission: str | None) -> OmniframesError:
        """Fold all three server error envelopes (§1) onto the omniframes hierarchy."""
        status = response.status_code
        payload = _json_body(response)
        message = self._scrub(_error_message(payload) or _text_snippet(response))
        where = f"{response.request.method} {response.request.url.path}"
        lowered = message.lower()

        if status == 408:
            # Only `resultType` mode 408s; the NDJSON path reports timeouts in-band.
            return QueryTimeoutError(
                f"the Omni API timed out the query ({status} from {where}): {message}",
                remaining_job_ids=_remaining_job_ids(payload),
            )
        if _FEATURE_FLAG_MARKER in lowered:
            return FeatureFlagError(
                f"the Omni Query API is not enabled for this organization ({status} from "
                f"{where}: {message}). An organization admin has to enable the Query API "
                "feature for the org; the same API key then works unchanged.",
                status=status,
            )
        if _PERMISSION_DENIED_MARKER in lowered:
            return ModelPermissionError(
                f"Omni denied access to this model ({status} from {where}: {message}). "
                f"{_permission_advice(permission)}",
                permission=permission,
                status=status,
            )
        if status in (400, 401, 403) and any(marker in lowered for marker in _AUTH_MARKERS):
            return AuthError(
                f"Omni rejected the API key ({status} from {where}): {message}. Check "
                "OMNI_API_KEY — it must be an org API key or personal access token, sent as "
                "'Authorization: Bearer <token>'.",
                status=status,
            )
        if status == 401:
            return AuthError(
                f"Omni rejected the API key ({status} from {where}): {message}", status=status
            )
        return TransportError(
            f"the Omni API returned {status} for {where}: {message}", status=status
        )

    def _scrub(self, text: str) -> str:
        """Remove the API key from any text that is about to become an exception message."""
        return text.replace(self._api_key, REDACTED_API_KEY)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _coerce_timeout(timeout: httpx.Timeout | float | None) -> httpx.Timeout:
    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(timeout)


def _permission_hint(envelope: Mapping[str, Any]) -> str | None:
    """Infer which model permission a 403 on this envelope would be about (§1).

    Topic queries need ``QUERY_TOPICS``, bare-view queries ``QUERY_FULL_MODEL``, raw SQL
    ``QUERY_SQL``.  Returns ``None`` when the envelope does not say.
    """
    query = envelope.get("query")
    if not isinstance(query, Mapping):
        return None
    if query.get("userEditedSQL"):
        return "QUERY_SQL"
    if query.get("join_paths_from_topic_name"):
        return "QUERY_TOPICS"
    if query.get("table"):
        return "QUERY_FULL_MODEL"
    return None


def _permission_advice(permission: str | None) -> str:
    if permission == "QUERY_TOPICS":
        return "The key's user needs the QUERY_TOPICS permission to query topics on this model."
    if permission == "QUERY_FULL_MODEL":
        return (
            "The key's user needs the QUERY_FULL_MODEL permission to query a bare view "
            "(topic queries only need QUERY_TOPICS)."
        )
    if permission == "QUERY_SQL":
        return "The key's user needs the QUERY_SQL permission to run raw-SQL jobs on this model."
    return (
        "The key's user needs QUERY_TOPICS (topic queries) or QUERY_FULL_MODEL (bare-view "
        "queries) on this model; GET /whoami reports rolesByModel."
    )


def _json_body(response: httpx.Response) -> Mapping[str, Any] | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _error_message(payload: Mapping[str, Any] | None) -> str | None:
    """Pull the human message out of whichever of the three error envelopes arrived (§1)."""
    if payload is None:
        return None
    detail = payload.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    error = payload.get("error")
    if isinstance(error, str) and error:
        return error
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    message = payload.get("message")
    if isinstance(message, str) and message:
        return message
    return None


def _remaining_job_ids(payload: Mapping[str, Any] | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    raw = payload.get("remaining_job_ids") or ()
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list | tuple):
        return tuple(str(job_id) for job_id in raw)
    return ()


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    return seconds if isfinite(seconds) and seconds >= 0 else None


def _text_snippet(response: httpx.Response, limit: int = 200) -> str:
    text = response.text.strip()
    if not text:
        return "(empty response body)"
    return text if len(text) <= limit else f"{text[:limit]}…"


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise TransportError(
            f"expected a JSON body from {response.request.url.path}, got "
            f"{response.headers.get('content-type', 'no content-type')}"
        ) from exc
    if not isinstance(payload, dict):
        raise TransportError(
            f"expected a JSON object from {response.request.url.path}, got {type(payload).__name__}"
        )
    return payload
