"""FakeOmniAPI — an in-process stand-in for a real Omni org.

Use it as an :class:`httpx.MockTransport` handler::

    handler = FakeOmniAPI()
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://bench.example.omni.co",
    )

Everything it serves is byte-faithful to ``docs/CONTRACT_NOTES.md`` — that is the point.  The
NDJSON framing (§2.2) in particular is exact: ``Content-Type: text/ndjson``, one compact JSON
object per line with a trailing ``\\n`` after *every* framed line, a header line carrying the
submitted job ids, one line per job, and a footer whose ``timed_out`` is the **string**
``"true"``/``"false"`` and is ``"true"`` iff ``remaining_job_ids`` is non-empty.

Query results are **real**: the wire query object is compiled to DuckDB SQL over the checked-in
bench parquet files (:mod:`tests.fakes.engine`) and the ``result`` payload is a genuine base64
Arrow IPC stream, so ``tests/data/bench/known_answers.json`` is a valid oracle for what comes
back.

Configurable failure modes (constructor arguments):

===========================  =================================================================
``feature_flag_off``         ``query-api`` disabled: 403 ``Feature not enabled`` on the query
                             endpoints, but **not** on ``/whoami`` (§1).
``permissions``              The model permissions ``/whoami`` reports.  Drop ``QUERY_TOPICS``
                             to get 403 ``Permission denied`` on ``/query/run``.
``redact_sql``               The caller lacks ``VIEW_SQL``: job ``error_message`` is replaced
                             with the exact redaction literal and every SQL field is blanked.
``rate_limited_after``       After N requests every further request is a WAF 429 carrying
                             ``X-Omni-Waf-Action: block``.
``slow_job_polls``           The run response times out (footer ``timed_out: "true"`` plus
                             ``remaining_job_ids``); the job only completes on the Nth
                             ``/query/wait`` call, exercising the client's wait loop.
``documents``                What ``GET /documents/{id}/queries`` serves, keyed by identifier.
                             An identifier mapped to an empty list is a document with **no
                             dashboard** — the §4 404.
``ai_credits_exhausted``     ``POST /ai/generate-query`` answers 402 (the AI-credit shutoff).
===========================  =================================================================

An envelope carrying ``workbookUrl: true`` gets the response header ``X-Omni-Workbook-Url``
alongside the normal NDJSON stream (§2.1).  The URL *shape* is LIVE-VALIDATE #10 — the fake mints
``https://<host>/w/fake/<job-id>`` so a client's plumbing can be tested, not so the format can be
relied on.  The combination with ``staticQueryReferences`` stays a 400, exactly as documented.

Scope note: the fake implements what the current milestone exercises and grows with the
milestones (docs/DESIGN.md §5).  Today's query scope is ``modelId`` + ``table`` /
``join_paths_from_topic_name`` + ``fields`` (dimensions, ``[grain]`` suffixes and the six
governed measures) + filters (dimension-keyed → ``WHERE``, measure-keyed → ``HAVING``) + sorts +
``column_totals`` + limit/offset, plus the two SQL paths: **verbatim SQL jobs**
(``userEditedSQL`` with a no-rewrite marker, ``sqlSortsEnabled``, ``__omni_summ`` totals
sidecars) — :mod:`tests.fakes.sqljobs` —
and **parsed OmniSQL jobs** (``userEditedSQL`` with ``rewriteSql`` absent, ``${…}`` model
references, governed joins) — :mod:`tests.fakes.omnisql`.  Pivots, calculations, ``fill_fields``,
``row_totals`` and ``resultType`` are refused loudly rather than faked.

API keys never reach the request log, the ``repr`` or any error message.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import httpx
import pyarrow as pa

from tests.fakes.bench_model import (
    BENCH_MODEL_ID,
    BENCH_MODEL_NAME,
    BENCH_MODEL_VIEWS,
    BENCH_TOPIC,
    DEFAULT_PERMISSIONS,
    OTHER_MODELS,
    FakeModel,
    FakeTopic,
    FakeView,
)
from tests.fakes.documents import (
    DEFAULT_DOCUMENTS,
    NO_QUERY_GENERATED,
    RUN_QUERY_REFUSAL,
    SavedQuery,
    generate_for_prompt,
)
from tests.fakes.engine import BenchEngine, PlanFailure, PlannedQuery, arrow_data_type
from tests.fakes.omnisql import is_omnisql_job, run_omnisql_job
from tests.fakes.sqljobs import SqlJob, is_raw_sql_job, run_sql_job

__all__ = [
    "DEFAULT_TOKEN",
    "MEMBERSHIP_ID",
    "NDJSON_CONTENT_TYPE",
    "REDACTED_ERROR_MESSAGE",
    "USER_ID",
    "WORKBOOK_URL_HEADER",
    "FakeOmniAPI",
    "RecordedRequest",
]

#: Token shape from CONTRACT_NOTES §1: ``omni_osk_`` + 50 base62 chars + a 6-char CRC.
DEFAULT_TOKEN: Final = "omni_osk_" + "t" * 56

USER_ID: Final = "7a1c0b33-1111-4111-8111-000000000001"
MEMBERSHIP_ID: Final = "7a1c0b33-2222-4222-8222-000000000002"

NDJSON_CONTENT_TYPE: Final = "text/ndjson"

#: Response header carrying the workbook URL when the envelope asked for ``workbookUrl: true``
#: (CONTRACT_NOTES §2.1).  Spelled out here rather than imported from the client, so a drift
#: between the two is a test failure rather than a shared mistake.
WORKBOOK_URL_HEADER: Final = "X-Omni-Workbook-Url"

#: Substituted for ``error_message`` when the caller lacks ``VIEW_SQL`` (§2.2).  Kept as a
#: literal here on purpose: the fake must be able to drift from the parser and be caught.
REDACTED_ERROR_MESSAGE: Final = (
    "Query failed. Error details are only visible to users with permission to view SQL."
)

_CACHE_POLICIES: Final = frozenset(
    {"Standard", "SkipRequery", "SkipCache", "SkipCacheAndRebuildExtracts"}
)
_MAX_PAGE_SIZE: Final = 100
_SERVER_STREAM_MS: Final = 7


@dataclass(frozen=True)
class RecordedRequest:
    """One entry of :attr:`FakeOmniAPI.requests`.

    Headers are deliberately **not** recorded: the bearer token must never reach a log.
    """

    method: str
    path: str
    params: dict[str, list[str]] = field(default_factory=dict)
    body: Any = None

    def param(self, name: str) -> str | None:
        values = self.params.get(name)
        return values[0] if values else None

    @property
    def query(self) -> Mapping[str, Any] | None:
        """``body["query"]`` when the body was a query envelope."""
        if isinstance(self.body, Mapping):
            candidate = self.body.get("query")
            if isinstance(candidate, Mapping):
                return candidate
        return None


@dataclass
class _Job:
    """A submitted job: its finished line plus how many ``/query/wait`` calls it still needs."""

    job_id: str
    line: dict[str, Any]
    waits_remaining: int = 0


class FakeOmniAPI:
    """A minimal, wire-exact Omni org: auth, whoami, catalog, ``/query/run``, ``/query/wait``."""

    def __init__(
        self,
        *,
        tokens: Iterable[str] = (DEFAULT_TOKEN,),
        model_id: str = BENCH_MODEL_ID,
        model_name: str = BENCH_MODEL_NAME,
        topic: FakeTopic = BENCH_TOPIC,
        model_views: Sequence[FakeView] = BENCH_MODEL_VIEWS,
        data_dir: Path | None = None,
        key_scope: str = "organization",
        org_role: str = "ADMIN",
        role_name: str = "Admin",
        base_role: str = "ADMIN",
        permissions: Sequence[str] = DEFAULT_PERMISSIONS,
        feature_flag_off: bool = False,
        redact_sql: bool = False,
        rate_limited_after: int | None = None,
        slow_job_polls: int = 0,
        default_page_size: int = 20,
        extra_models: Sequence[FakeModel] = OTHER_MODELS,
        documents: Mapping[str, Sequence[SavedQuery]] = DEFAULT_DOCUMENTS,
        ai_credits_exhausted: bool = False,
    ) -> None:
        self.tokens = frozenset(tokens)
        self.model_id = model_id
        self.model_name = model_name
        self.topic = topic
        #: Every view of the composed model — a superset of ``topic.views`` (see ``_list_views``).
        self.model_views = tuple(model_views)
        self.key_scope = key_scope
        self.org_role = org_role
        self.role_name = role_name
        self.base_role = base_role
        self.permissions = tuple(permissions)
        self.feature_flag_off = feature_flag_off
        self.redact_sql = redact_sql
        self.rate_limited_after = rate_limited_after
        self.slow_job_polls = slow_job_polls
        self.default_page_size = default_page_size
        self.documents = {key: tuple(value) for key, value in documents.items()}
        self.ai_credits_exhausted = ai_credits_exhausted

        bench_model = FakeModel(id=model_id, name=model_name)
        self.models: tuple[FakeModel, ...] = (bench_model, *extra_models)
        self.engine = BenchEngine(data_dir=data_dir, topic=topic)

        self.requests: list[RecordedRequest] = []
        #: Every workbook URL this fake minted, newest last.  Recorded so a test can assert that
        #: ``df.omni_url()`` returns *exactly* what the server sent without hard-coding the URL
        #: format, which CONTRACT_NOTES §6 #10 lists as unconfirmed and SQLTIER §6 forbids
        #: pinning offline.
        self.workbook_urls: list[str] = []
        self._request_count = 0
        self._job_counter = 0
        self._jobs: dict[str, _Job] = {}

    # -- introspection ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FakeOmniAPI(model_id={self.model_id!r}, tokens={len(self.tokens)}, "
            f"requests={len(self.requests)})"
        )

    @property
    def last_request(self) -> RecordedRequest:
        return self.requests[-1]

    @property
    def paths(self) -> list[str]:
        """``METHOD path`` for every request received, in order."""
        return [f"{r.method} {r.path}" for r in self.requests]

    def close(self) -> None:
        self.engine.close()

    # -- entry point -----------------------------------------------------------------

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body, body_ok = _decode_json(request)
        self.requests.append(
            RecordedRequest(
                method=request.method,
                path=request.url.path,
                params=_multi_params(request.url),
                body=body,
            )
        )
        self._request_count += 1

        if self.rate_limited_after is not None and self._request_count > self.rate_limited_after:
            return _rate_limited()

        auth_failure = self._authenticate(request)
        if auth_failure is not None:
            return auth_failure

        segments = [segment for segment in request.url.path.split("/") if segment]
        if segments[:2] != ["api", "v1"]:
            return _detail(404, "Not found")
        route = segments[2:]
        method = request.method.upper()

        if method == "GET" and route == ["whoami"]:
            return self._whoami(request)
        if method == "GET" and route == ["models"]:
            return self._list_models(request)
        if method == "GET" and len(route) == 3 and route[0] == "models" and route[2] == "topic":
            return self._list_topics(route[1])
        if method == "GET" and len(route) == 4 and route[0] == "models" and route[2] == "topic":
            return self._get_topic(route[1], route[3])
        if method == "GET" and len(route) == 3 and route[0] == "models" and route[2] == "view":
            return self._list_views(route[1])
        if (
            method == "GET"
            and len(route) == 3
            and route[0] == "documents"
            and route[2] == "queries"
        ):
            return self._document_queries(route[1])
        if method == "POST" and route == ["ai", "generate-query"]:
            return self._generate_query(body, body_ok)
        if method == "POST" and route == ["query", "run"]:
            return self._run(request, body, body_ok)
        if method == "GET" and route == ["query", "wait"]:
            return self._wait(request)
        return _detail(404, "Not found")

    # -- auth ------------------------------------------------------------------------

    def _authenticate(self, request: httpx.Request) -> httpx.Response | None:
        header = request.headers.get("Authorization")
        scheme, _, value = (header or "").partition(" ")
        token = value.strip()
        if scheme.lower() != "bearer" or not token:
            return _envelope_error(
                400, "Bad authorization header, must be formatted as Bearer <token>"
            )
        if token not in self.tokens:
            return _envelope_error(403, "Invalid bearer token")
        return None

    def _query_guard(self, *, check_permissions: bool) -> httpx.Response | None:
        """Feature-flag and permission gates that apply to the query endpoints only (§1)."""
        if self.feature_flag_off:
            return _detail(403, "Feature not enabled")
        if check_permissions and "QUERY_TOPICS" not in self.permissions:
            return _detail(403, "Permission denied")
        return None

    # -- whoami ----------------------------------------------------------------------

    def _whoami(self, request: httpx.Request) -> httpx.Response:
        """The connect-time preflight. Deliberately unaffected by the ``query-api`` flag (§1)."""
        wanted = _csv_params(request.url, "modelId")
        roles: dict[str, Any] = {}
        for model in self.models:
            if wanted and model.id not in wanted:
                continue
            roles[model.id] = {
                "roleName": self.role_name,
                "baseRole": self.base_role,
                "connectionId": model.connection_id,
                "permissions": list(self.permissions),
            }
        return httpx.Response(
            200,
            json={
                "user": {"id": USER_ID, "membershipId": MEMBERSHIP_ID},
                "keyScope": self.key_scope,
                "orgRole": self.org_role,
                "rolesByModel": roles,
                "rolesByModelTruncated": False,
            },
        )

    # -- catalog ---------------------------------------------------------------------

    def _list_models(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        raw_page_size = params.get("pageSize")
        page_size = self.default_page_size
        if raw_page_size is not None:
            try:
                page_size = int(raw_page_size)
            except ValueError:
                return _detail(400, "pageSize must be an integer")
            if not 1 <= page_size <= _MAX_PAGE_SIZE:
                return _detail(400, f"pageSize must be between 1 and {_MAX_PAGE_SIZE}")

        cursor = params.get("cursor")
        offset = 0
        if cursor is not None:
            decoded = _decode_cursor(cursor)
            if decoded is None:
                # The one endpoint with its own error envelope (CONTRACT_NOTES §1).
                return httpx.Response(400, json={"error": "Invalid cursor", "success": False})
            offset = decoded

        records = list(self.models)
        model_kind = params.get("modelKind")
        if model_kind is not None:
            records = [m for m in records if m.model_kind == model_kind]
        # Both filters are *exact* server-side: Prisma gets `{name}` / `{id: modelId}`
        # (CONTRACT_NOTES §4), not a contains-clause.
        name = params.get("name")
        if name:
            # JS truthiness, not `is not None`: `name` is `z.string().optional()` so an empty
            # string passes validation, and `...(name && {name})` then drops the filter — the
            # reply is page one of the whole catalog, not an empty page.
            #
            # The comparison is exact but *case-insensitive*: `m.name = ${name}` runs against
            # `OmniModel.name`, a Postgres CITEXT column (`packages/db-models/prisma/
            # schema.prisma:1391`).  Exact and case-sensitive are not the same claim.
            records = [m for m in records if m.name.casefold() == name.casefold()]
        model_id = params.get("modelId")
        if model_id is not None:
            # Declared `z.uuid()` under a `.strict()` schema: a non-UUID is a 400, not no-match.
            if _ZOD_UUID.match(model_id) is None:
                return _detail(400, "modelId must be a valid uuid")
            records = [m for m in records if m.id == model_id]

        page = records[offset : offset + page_size]
        next_offset = offset + page_size
        has_next = next_offset < len(records)
        return httpx.Response(
            200,
            json={
                "pageInfo": {
                    "hasNextPage": has_next,
                    "nextCursor": _encode_cursor(next_offset) if has_next else None,
                    "pageSize": page_size,
                    "totalRecords": len(records),
                },
                "records": [m.to_wire() for m in page],
            },
        )

    def _list_topics(self, model_id: str) -> httpx.Response:
        if not self._model_exists(model_id):
            return _detail(404, f"Model {model_id} not found")
        topics = [self.topic.summary_wire()] if model_id == self.model_id else []
        return httpx.Response(200, json={"success": True, "topics": topics})

    def _get_topic(self, model_id: str, topic_name: str) -> httpx.Response:
        if not self._model_exists(model_id):
            return _detail(404, f"Model {model_id} not found")
        if model_id != self.model_id or topic_name != self.topic.name:
            return _detail(404, f"Topic {topic_name} not found")
        return httpx.Response(
            200,
            json={"success": True, "topic": self.topic.to_wire(redact_sql=self.redact_sql)},
        )

    def _list_views(self, model_id: str) -> httpx.Response:
        """``GET /models/{id}/view`` — flattened & lossy: field names and kinds, no data types.

        Serves every non-ignored view of the *composed* model, so it is a superset of the views
        the topic detail reaches (CONTRACT_NOTES §4).
        """
        if not self._model_exists(model_id):
            return _detail(404, f"Model {model_id} not found")
        views = self.model_views if model_id == self.model_id else ()
        payload = [
            {
                "name": view.name,
                "label": view.label,
                "description": None,
                "hidden": False,
                # `VIEW_FIELD_TYPE` maps to the *lowercase* strings `dimension`/`measure`/
                # `filter` (`packages/bi-app/app/types/api/models/view-field-type.ts`) — the
                # uppercase spelling is the TS constant's name, not its wire value.
                "fields": [
                    *({"name": f.field_name, "type": "dimension"} for f in view.dimensions),
                    *({"name": f.field_name, "type": "measure"} for f in view.measures),
                ],
            }
            for view in views
        ]
        return httpx.Response(200, json={"success": True, "views": payload})

    def _model_exists(self, model_id: str) -> bool:
        return any(model.id == model_id for model in self.models)

    # -- saved queries & AI (CONTRACT_NOTES §4) ---------------------------------------

    def _document_queries(self, identifier: str) -> httpx.Response:
        """``GET /api/v1/documents/{identifier}/queries``.

        Two distinct 404s: an identifier the org has never heard of, and a document that exists
        but carries no dashboard — the second is the one §4 calls out, and a client has to treat
        it as an ordinary answer rather than an outage.  Neither is gated on ``query-api``: the
        endpoint hands out query *objects*, it does not run them.
        """
        saved = self.documents.get(identifier)
        if saved is None:
            return _detail(404, f"Document {identifier} not found")
        if not saved:
            return _detail(404, f"Document {identifier} does not have a dashboard")
        return httpx.Response(200, json={"queries": [query.to_wire() for query in saved]})

    def _generate_query(self, body: Any, body_ok: bool) -> httpx.Response:
        """``POST /api/v1/ai/generate-query`` — a deterministic stub over the bench model.

        ``runQuery`` must be ``false``.  Its *server* default is ``true``, so an absent key means
        "run it", and a fake that quietly accepted that would be pretending to have executed a
        query it never ran.  With ``runQuery: false`` the endpoint needs no ``query-api`` flag
        (§4), so the flag is deliberately not checked here.
        """
        if self.ai_credits_exhausted:
            return _detail(402, "AI credits exhausted")
        if not body_ok:
            return _detail(400, "Request body is not valid JSON")
        if not isinstance(body, Mapping):
            return _detail(400, "Request body must be an object")
        if body.get("runQuery") is not False:
            return _detail(400, RUN_QUERY_REFUSAL)

        model_id = body.get("modelId")
        if not isinstance(model_id, str) or not _is_uuid(model_id):
            return _detail(400, "modelId must be a UUID")
        if model_id != self.model_id:
            return _detail(404, f"Model {model_id} not found")
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _detail(400, "prompt is required")

        generated = generate_for_prompt(prompt)
        if generated is None:
            return _detail(400, NO_QUERY_GENERATED)
        return httpx.Response(200, json=generated.to_wire())

    # -- /query/run ------------------------------------------------------------------

    def _run(self, request: httpx.Request, body: Any, body_ok: bool) -> httpx.Response:
        guard = self._query_guard(check_permissions=True)
        if guard is not None:
            return guard
        if not body_ok:
            return _detail(400, "Request body is not valid JSON")
        if not isinstance(body, Mapping):
            return _detail(400, "Request body must be an object")

        invalid = _validate_envelope(body, request.url)
        if invalid is not None:
            return invalid
        query = body["query"]

        job_id = self._next_job_id()
        plan_only = bool(body.get("planOnly"))
        line = self._build_job_line(job_id, query, plan_only=plan_only)
        workbook_url = (
            f"{request.url.scheme}://{request.url.netloc.decode()}/w/fake/{job_id}"
            if body.get("workbookUrl") is True
            else None
        )
        return self._submit(job_id, line, workbook_url=workbook_url)

    def _build_job_line(
        self, job_id: str, query: Mapping[str, Any], *, plan_only: bool
    ) -> dict[str, Any]:
        if query.get("modelId") != self.model_id:
            return self._error_line(
                job_id,
                f"Model {query.get('modelId')} not found",
                error_type="PLAN",
                query=query,
            )
        try:
            # ``userEditedSQL`` selects one of two SQL paths, decided by the no-rewrite marker:
            # with one the text runs verbatim (§3.4), without one it is parsed as OmniSQL and
            # planned as a governed model job (§3.6).  It is never ignored.
            if is_raw_sql_job(query) or is_omnisql_job(query):
                return self._sql_job_line(job_id, query, plan_only=plan_only)
            if query.get("staticQueryReferences"):
                raise PlanFailure(
                    'query.staticQueryReferences consumers (the type: "query" filter arm and '
                    "the XLOOKUP calc operators) are not modeled by FakeOmniAPI"
                )
            planned = self.engine.plan(query)
            if plan_only:
                schema = self.engine.schema(planned)
                return self._planned_line(job_id, query, planned, schema)
            table = self.engine.execute(planned)
        except PlanFailure as exc:
            return self._error_line(job_id, str(exc), error_type=exc.error_type, query=query)
        return self._complete_line(job_id, query, planned, table)

    def _sql_job_line(
        self, job_id: str, query: Mapping[str, Any], *, plan_only: bool
    ) -> dict[str, Any]:
        """A SQL job — verbatim (§3.4) or parsed OmniSQL (§3.6): the same line shape either way."""
        runner = run_omnisql_job if is_omnisql_job(query) else run_sql_job
        job, table = runner(self.engine, query, plan_only=plan_only)
        if plan_only or table is None:
            return {
                "job_id": job_id,
                "status": "PLANNED",
                "summary": self._sql_summary(job, rows=0),
                "query": dict(query),
            }
        return {
            "job_id": job_id,
            "status": "COMPLETE",
            "summary": self._sql_summary(job, rows=table.num_rows),
            "cache_metadata": {"cache_type": "MISS", "job_id": job_id},
            "query": dict(query),
            "result": _arrow_b64(table),
            "stream_stats": {"server_stream": _SERVER_STREAM_MS},
        }

    def _submit(
        self, job_id: str, line: dict[str, Any], *, workbook_url: str | None = None
    ) -> httpx.Response:
        """Emit the run response, honoring ``slow_job_polls``.

        With ``slow_job_polls = N`` the run response carries no job line at all: just the header
        and a footer that reports the job as still running, exactly like a real wait-window
        expiry.  The job then lands on the Nth ``/query/wait`` call.

        ``workbook_url`` rides the **run** response's headers, whether or not the job finished
        inside the wait window — the URL is a property of the submitted query, not of its result.
        """
        if workbook_url is not None:
            self.workbook_urls.append(workbook_url)
        headers = None if workbook_url is None else {WORKBOOK_URL_HEADER: workbook_url}
        header = {"jobs_submitted": {job_id: None}}
        if self.slow_job_polls > 0:
            self._jobs[job_id] = _Job(job_id, line, waits_remaining=self.slow_job_polls)
            return _ndjson([header, _footer([job_id])], headers=headers)
        self._jobs[job_id] = _Job(job_id, line, waits_remaining=0)
        return _ndjson([header, line, _footer()], headers=headers)

    # -- /query/wait -----------------------------------------------------------------

    def _wait(self, request: httpx.Request) -> httpx.Response:
        # No per-model authz on wait (org-scoped + unguessable job ids), but the feature flag
        # still gates the endpoint.
        guard = self._query_guard(check_permissions=False)
        if guard is not None:
            return guard
        job_ids = _wait_job_ids(request.url)
        if not job_ids:
            return _detail(400, "jobIds is required")

        lines: list[Mapping[str, Any]] = []
        remaining: list[str] = []
        for job_id in job_ids:
            job = self._jobs.get(job_id)
            if job is None:
                lines.append(
                    {"job_id": job_id, "status": "FAILED", "error_message": "Unable to find job"}
                )
                continue
            if job.waits_remaining > 0:
                job.waits_remaining -= 1
            if job.waits_remaining == 0:
                lines.append(job.line)
            else:
                remaining.append(job_id)
        lines.append(_footer(remaining))
        return _ndjson(lines)

    # -- job lines -------------------------------------------------------------------

    def _next_job_id(self) -> str:
        self._job_counter += 1
        return str(uuid.UUID(int=self._job_counter, version=4))

    def _complete_line(
        self,
        job_id: str,
        query: Mapping[str, Any],
        planned: PlannedQuery,
        table: pa.Table,
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "status": "COMPLETE",
            "summary": self._summary(planned, table.schema, rows=table.num_rows),
            "cache_metadata": {"cache_type": "MISS", "job_id": job_id},
            "query": dict(query),
            "result": _arrow_b64(table),
            "stream_stats": {"server_stream": _SERVER_STREAM_MS},
        }

    def _planned_line(
        self,
        job_id: str,
        query: Mapping[str, Any],
        planned: PlannedQuery,
        schema: pa.Schema,
    ) -> dict[str, Any]:
        """``planOnly`` (§2.4): status PLANNED, a summary with fields, no result."""
        return {
            "job_id": job_id,
            "status": "PLANNED",
            "summary": self._summary(planned, schema, rows=0),
            "query": dict(query),
        }

    def _error_line(
        self,
        job_id: str,
        message: str,
        *,
        error_type: str,
        query: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "job_id": job_id,
            "status": "ERROR",
            "client_result_id": "null",
            "error_type": error_type,
            "error_message": REDACTED_ERROR_MESSAGE if self.redact_sql else message,
            "query": dict(query),
        }

    def _summary(self, planned: PlannedQuery, schema: pa.Schema, *, rows: int) -> dict[str, Any]:
        """``summary`` per §2.3 — ``fields`` keyed by requested field name, in request order.

        The key is the name **exactly as it was asked for**, ``[grain]`` bracket included, and
        the reserved totals-indicator column is deliberately absent: ``summary.fields`` describes
        the fields the caller requested, not Omni's internal columns.
        """
        types = {name: schema.field(name).type for name in schema.names}
        fields: dict[str, Any] = {}
        for field_def in planned.fields:
            payload = field_def.to_wire(redact_sql=self.redact_sql)
            dtype = types.get(field_def.name)
            if dtype is not None:
                payload["data_type"] = arrow_data_type(dtype)
            fields[field_def.name] = payload
        return {
            "fields": fields,
            "display_sql": "" if self.redact_sql else planned.sql,
            "omni_sql": "" if self.redact_sql else _omni_sql(planned),
            "omni_sql_parse_failed": False,
            "cache_type": "MISS",
            "stage_summaries": [{"succeeded": True, "warnings": []}],
            "stats": {"rows": rows},
            "plan_stats": {"stages": 1},
            "missing_fields": list(planned.missing_fields),
            "invalid_calculations": {},
            "locale_options": {"timezone": "UTC"},
        }

    def _sql_summary(self, job: SqlJob, *, rows: int) -> dict[str, Any]:
        """``summary`` for a SQL job — see :attr:`SqlJob.result_fields` for where ``fields`` come
        from on each path.

        ``display_sql`` is the SQL that actually ran, so a ``sqlSortsEnabled`` sort wrapper (or
        the OmniSQL resolver's substituted, join-expanded statement) is visible in it.
        ``omni_sql`` is the OmniSQL text on the parsed path and empty on the verbatim one, which
        has no Omni-flavored form.  There are no ``missing_fields``: a SQL job has no field names
        to miss.
        """
        return {
            "fields": dict(job.result_fields),
            "display_sql": "" if self.redact_sql else job.sql,
            "omni_sql": "" if self.redact_sql else job.omni_sql,
            "omni_sql_parse_failed": False,
            "cache_type": "MISS",
            "stage_summaries": [{"succeeded": True, "warnings": []}],
            "stats": {"rows": rows},
            "plan_stats": {"stages": 1},
            "missing_fields": [],
            "invalid_calculations": {},
            "locale_options": {"timezone": "UTC"},
        }


# --------------------------------------------------------------------------------------
# Envelope validation (CONTRACT_NOTES §2.1)
# --------------------------------------------------------------------------------------


def _validate_envelope(body: Mapping[str, Any], url: httpx.URL) -> httpx.Response | None:
    query = body.get("query")
    if not isinstance(query, Mapping):
        return _detail(400, "query is required")

    model_id = query.get("modelId")
    if not isinstance(model_id, str) or not _is_uuid(model_id):
        return _detail(400, "query.modelId must be a UUID")

    fields = query.get("fields")
    if fields is not None and (
        not isinstance(fields, list) or any(not isinstance(f, str) for f in fields)
    ):
        return _detail(400, "query.fields must be an array of strings")

    if "limit" in query:
        limit = query["limit"]
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            return _detail(400, "query.limit must be a positive integer or null")

    cache = body.get("cache")
    if cache is not None and cache not in _CACHE_POLICIES:
        return _detail(400, f"cache must be one of {sorted(_CACHE_POLICIES)}")

    if body.get("planOnly") and body.get("resultType") is not None:
        return _detail(400, "planOnly and resultType cannot both be provided")
    if body.get("planOnly") and body.get("workbookUrl"):
        return _detail(400, "planOnly and workbookUrl cannot both be provided")
    if body.get("workbookUrl") and query.get("staticQueryReferences"):
        # The §2.1 refinement: a workbook URL cannot be minted for a query carrying references.
        return _detail(
            400, "workbookUrl cannot be combined with a non-empty query.staticQueryReferences"
        )
    if body.get("formatResults") is not None and body.get("resultType") is None:
        return _detail(400, "formatResults cannot be provided without resultType")
    if body.get("userId") is not None and url.params.get("userId") is not None:
        return _detail(400, "userId may be provided in the body or the query string, not both")
    if "branchId" in query:
        return _detail(400, "branchId is a top-level field, not a query field")

    unsupported = _unsupported_feature(body, query)
    if unsupported is not None:
        return _detail(400, f"{unsupported} is not implemented by FakeOmniAPI")
    return None


def _unsupported_feature(body: Mapping[str, Any], query: Mapping[str, Any]) -> str | None:
    """Loud refusal beats a plausible lie for envelope features the fake does not model yet.

    ``userEditedSQL`` and ``staticQueryReferences`` are handled at the job line: both SQL paths
    are implemented, and references are ignored on the raw path but refused on the others.
    """
    if body.get("resultType") is not None:
        return "resultType"
    for key in ("pivots", "calculations", "fill_fields", "row_totals"):
        if query.get(key):
            return f"query.{key}"
    return None


# --------------------------------------------------------------------------------------
# Response helpers
# --------------------------------------------------------------------------------------


#: `z.uuid()`'s grammar, ported from the zod the server pins (`zod/v4/core/regexes.js`).  Kept as
#: its own copy rather than imported from `omniframes.catalog`: the fake mirrors the *server*, so
#: a client-side check that drifts from this one has to show up as a test failure.
_ZOD_UUID: Final = re.compile(
    r"^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r"|00000000-0000-0000-0000-000000000000"
    r"|ffffffff-ffff-ffff-ffff-ffffffffffff)$"
)


def _detail(status: int, message: str) -> httpx.Response:
    """The Remix route error envelope: ``{"detail", "status"}``."""
    return httpx.Response(status, json={"detail": message, "status": status})


def _envelope_error(status: int, message: str) -> httpx.Response:
    """The Express auth-middleware envelope: ``{"error": {"code", "message"}}``."""
    return httpx.Response(status, json={"error": {"code": status, "message": message}})


def _rate_limited() -> httpx.Response:
    """AWS WAF block: 429 plus ``X-Omni-Waf-Action: block`` (§1).

    Only the status and the header are pinned by the contract (the body shape is
    LIVE-VALIDATE #7), so clients must key off those two.
    """
    return httpx.Response(
        429,
        json={"error": {"code": 429, "message": "Request blocked"}},
        headers={"X-Omni-Waf-Action": "block"},
    )


def _ndjson(
    lines: Sequence[Mapping[str, Any]], *, headers: Mapping[str, str] | None = None
) -> httpx.Response:
    """§2.2 framing: compact JSON objects, ``\\n`` after every line, ``text/ndjson``."""
    body = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)
    return httpx.Response(
        200,
        content=body.encode("utf-8"),
        headers={"Content-Type": NDJSON_CONTENT_TYPE, **(headers or {})},
    )


def _footer(remaining: Sequence[str] = ()) -> dict[str, Any]:
    """``timed_out`` is a STRING, and is "true" iff ``remaining_job_ids`` is non-empty."""
    return {"remaining_job_ids": list(remaining), "timed_out": "true" if remaining else "false"}


def _arrow_b64(table: pa.Table) -> str:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii")


def _omni_sql(planned: PlannedQuery) -> str:
    return "query { " + ", ".join(planned.field_names) + " }"


# --------------------------------------------------------------------------------------
# Request helpers
# --------------------------------------------------------------------------------------


def _decode_json(request: httpx.Request) -> tuple[Any, bool]:
    raw = request.content
    if not raw:
        return None, True
    try:
        return json.loads(raw), True
    except ValueError:
        return None, False


def _multi_params(url: httpx.URL) -> dict[str, list[str]]:
    params: dict[str, list[str]] = {}
    for key, value in url.params.multi_items():
        params.setdefault(key, []).append(value)
    return params


def _csv_params(url: httpx.URL, name: str) -> list[str]:
    """Collect a repeatable, comma-separated query parameter (``?modelId=a,b&modelId=c``)."""
    values: list[str] = []
    for key, value in url.params.multi_items():
        if key != name:
            continue
        values.extend(part.strip() for part in value.split(",") if part.strip())
    return values


def _wait_job_ids(url: httpx.URL) -> list[str]:
    """``jobIds`` (comma-separated, preferred) or legacy ``job_ids`` (JSON array or csv)."""
    ids = _csv_params(url, "jobIds")
    if ids:
        return ids
    legacy = url.params.get("job_ids")
    if legacy is None:
        return []
    text = legacy.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError:
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    return [part.strip() for part in text.split(",") if part.strip()]


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(f"offset:{offset}".encode()).decode("ascii")


def _decode_cursor(cursor: str) -> int | None:
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    prefix, _, value = decoded.partition(":")
    if prefix != "offset" or not value.isdigit():
        return None
    return int(value)


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True
