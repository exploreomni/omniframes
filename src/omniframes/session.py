"""The session: credentials, catalog, and the entry point to reading data (INTERNALS §5).

::

    from omniframes import OmniSession

    session = (
        OmniSession.builder
        .host("acme.omniapp.co")
        .api_key_from_env()          # OMNI_API_KEY
        .get_or_create()
    )
    df = session.read.topic("bench_ecommerce", "order_items")

**Building a session makes no Omni API calls** (docs/DESIGN.md §3). Missing credentials in
Google Colab can be read from ``google.colab.userdata``, which contacts the notebook frontend.
The ``whoami`` preflight runs lazily, once, before the first call that needs Omni — or eagerly
via :meth:`verify` —
so a typo in the key surfaces as an auth error rather than as a confusing query failure.
``whoami`` is the right preflight because it answers even when the ``query-api`` feature flag
is off (CONTRACT_NOTES §1).

The API key lives in the transport and nowhere else: it never appears in a repr, a log or an
error message.
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from math import isfinite
from types import TracebackType
from typing import Any, Final

from omniframes._user_agent import _detect_hosted_runtime
from omniframes.catalog import Catalog
from omniframes.compile.querymodel import QUERY_VERSION, CachePolicy
from omniframes.compile.semantic import EnvelopeOptions
from omniframes.dataframe import DataFrame
from omniframes.errors import (
    CompileError,
    FeatureFlagError,
    ModelPermissionError,
    OmniframesError,
    TransportError,
)
from omniframes.plan import nodes
from omniframes.transport.base import PlanResult, QueryResult, QueryTransport
from omniframes.transport.http import HttpTransport

__all__ = ["DataFrameReader", "OmniSession", "SessionBuilder"]

#: Environment variables shared with the official ``omni-python-sdk`` (CONTRACT_NOTES §1).
API_KEY_ENV = "OMNI_API_KEY"
BASE_URL_ENV = "OMNI_BASE_URL"

_FEATURE_FLAG_ADVICE = (
    "An Omni organization admin has to enable the Query API (the `query-api` feature flag) for "
    "this organization; the same API key then works unchanged."
)
_PERMISSION_ADVICE = (
    "An Omni organization admin can grant it: QUERY_TOPICS covers session.read.topic(...), "
    "QUERY_FULL_MODEL is additionally required for session.read.view(...). "
    "session.verify() reports the permissions this key actually has, per model."
)

#: Fallback for a transport error that carries no ``status`` of its own: the status also appears
#: in the message :class:`~omniframes.transport.http.HttpTransport` builds.  The attribute is the
#: real answer (see :func:`_status_of`); this keeps an older transport working.
_STATUS_IN_MESSAGE: Final = re.compile(r"returned (\d{3}) for ")

#: The wording CONTRACT_NOTES §4 documents for "the document exists, but has no dashboard".
_NO_DASHBOARD: Final = "does not have a dashboard"


def _configuration_value(variable: str) -> str | None:
    """Read environment configuration, falling back to secrets only inside Colab."""
    value = os.environ.get(variable)
    if value:
        return value
    if _detect_hosted_runtime(os.environ) != "google-colab":
        return None

    # Keep Colab optional and avoid importing its notebook integrations on other runtimes.
    try:
        userdata = importlib.import_module("google.colab.userdata")
    except ImportError:
        raise CompileError(
            f"Google Colab secrets are unavailable; set {variable} in the environment instead."
        ) from None

    try:
        value = userdata.get(variable)
    except Exception as error:
        if isinstance(error, getattr(userdata, "SecretNotFoundError", ())):
            advice = "add it in Colab's Secrets panel"
        elif isinstance(error, getattr(userdata, "NotebookAccessError", ())):
            advice = "enable Notebook access for it in Colab's Secrets panel"
        elif isinstance(error, getattr(userdata, "TimeoutException", ())):
            advice = "run this cell from the Colab UI and try again"
        else:
            advice = "check Colab's Secrets panel and try again"
        # Never include provider exception text or chains: they may contain credentials.
        raise CompileError(
            f"could not read Google Colab secret {variable}: {advice}, "
            f"or set {variable} in the environment."
        ) from None
    if not isinstance(value, str) or not value:
        raise CompileError(
            f"Google Colab secret {variable} must contain a non-empty string; "
            f"update it in Colab's Secrets panel or set {variable} in the environment."
        )
    return value


class SessionBuilder:
    """Collects configuration; :meth:`get_or_create` builds a session without calling Omni.

    Explicit settings take precedence over environment variables, then Google Colab secrets.
    """

    __slots__ = (
        "_api_key",
        "_base_url",
        "_branch",
        "_cache",
        "_decomposition_row_cap",
        "_rate_limit_wait",
        "_timezone",
        "_transport",
        "_user_id",
    )

    def __init__(self) -> None:
        self._base_url: str | None = None
        self._api_key: str | None = None
        self._branch: str | None = None
        self._timezone: str | None = None
        self._cache: CachePolicy | None = None
        self._user_id: str | None = None
        self._transport: QueryTransport | None = None
        self._decomposition_row_cap: int | None = None
        self._rate_limit_wait: float | None = None

    def __repr__(self) -> str:
        return f"SessionBuilder(base_url={self._base_url!r})"

    def base_url(self, base_url: str) -> SessionBuilder:
        """The org URL. ``acme.omniapp.co``, ``https://acme.omniapp.co`` and ``…/api/v1`` all work."""
        self._base_url = base_url
        return self

    def host(self, host: str) -> SessionBuilder:
        """Alias of :meth:`base_url`, for when a bare hostname reads better."""
        return self.base_url(host)

    def base_url_from_env(self, variable: str = BASE_URL_ENV) -> SessionBuilder:
        """Read the org URL from the environment, then Colab secrets (``OMNI_BASE_URL``).

        In Google Colab, an unset or empty variable falls back to the secret of the same name.
        """
        value = _configuration_value(variable)
        if not value:
            raise CompileError(f"{variable} is not set")
        return self.base_url(value)

    def api_key(self, api_key: str) -> SessionBuilder:
        """The org API key or personal access token. Stored only inside the transport."""
        self._api_key = api_key
        return self

    def api_key_from_env(self, variable: str = API_KEY_ENV) -> SessionBuilder:
        """Read the key from the environment, then Colab secrets (``OMNI_API_KEY``).

        In Google Colab, an unset or empty variable falls back to the secret of the same name.
        """
        value = _configuration_value(variable)
        if not value:
            raise CompileError(
                f"{variable} is not set. Create an API key in Omni (Settings → API keys) and "
                f"export it as {variable}."
            )
        return self.api_key(value)

    def branch(self, branch_id: str) -> SessionBuilder:
        """Run every query against a model branch (top-level ``branchId``; must be a UUID)."""
        self._branch = branch_id
        return self

    def timezone(self, timezone: str) -> SessionBuilder:
        """An IANA timezone for query results (needs org + connection support server-side)."""
        self._timezone = timezone
        return self

    def cache(self, cache: str | CachePolicy) -> SessionBuilder:
        """Cache policy: ``Standard``, ``SkipRequery``, ``SkipCache``…

        The values published in the OpenAPI spec (``disabled``/``normal``/…) are rejected by the
        server, so they are rejected here too — with the list of what actually works.
        """
        if isinstance(cache, CachePolicy):
            self._cache = cache
            return self
        try:
            self._cache = CachePolicy(cache)
        except ValueError:
            allowed = ", ".join(policy.value for policy in CachePolicy)
            raise CompileError(
                f"{cache!r} is not an Omni cache policy; use one of: {allowed}"
            ) from None
        return self

    def user_id(self, membership_id: str) -> SessionBuilder:
        """Impersonate a **membership** id (not a user id) — CONTRACT_NOTES §1."""
        self._user_id = membership_id
        return self

    def decomposition_row_cap(self, rows: int | None) -> SessionBuilder:
        """Cap the raw scan a mixed aggregation pulls down (docs/HYBRID.md §2.1).

        SQL-expressible ad-hoc aggregations use a warehouse ``GROUP BY`` and fetch no raw
        rows, so this cap does not apply to them (docs/SQLTIER.md §1). It applies when the
        splitter falls back to local aggregation over raw rows.

        When a mixed ``agg()`` needs this fallback, the ad-hoc half is computed here over raw
        rows, and that scan is **unlimited by default**: a silently
        capped input to a local aggregation is a wrong answer, not a truncated page.  This is
        the safety valve for when unlimited is not affordable — it sends ``limit: rows`` instead
        and warns loudly (:class:`~omniframes.errors.TruncationWarning`) whenever the cap is
        actually hit, so a capped answer can never pass for a complete one.  ``None`` (the
        default) means unlimited.
        """
        if rows is not None and (isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0):
            raise CompileError(
                f"decomposition_row_cap() takes a positive integer or None (unlimited); got {rows!r}"
            )
        self._decomposition_row_cap = rows
        return self

    def rate_limit_wait(self, seconds: float) -> SessionBuilder:
        """Wait up to ``seconds`` for each rate-limited GET, not for the whole action.

        A catalog action can make multiple GET requests, and each receives its own waiting budget.
        """
        wait: object = seconds
        if (
            isinstance(wait, bool)
            or not isinstance(wait, int | float)
            or not isfinite(wait)
            or wait < 0
        ):
            raise CompileError("rate_limit_wait() takes a finite, non-negative number")
        if self._transport is not None:
            raise CompileError("configure rate-limit waiting on the transport you provided")
        self._rate_limit_wait = float(wait)
        return self

    def transport(self, transport: QueryTransport) -> SessionBuilder:
        """Use an existing transport instead of building an :class:`HttpTransport`.

        This is the seam tests use (``HttpTransport`` over ``httpx.MockTransport``) and the seam
        an in-product notebook broker will use.
        """
        if self._rate_limit_wait is not None:
            raise CompileError("configure rate-limit waiting on the transport you provided")
        self._transport = transport
        return self

    def get_or_create(self) -> OmniSession:
        """Build the session without calling Omni; missing settings can use Colab secrets."""
        transport = self._transport
        owns_transport = transport is None
        if transport is None:
            base_url = self._base_url or _configuration_value(BASE_URL_ENV)
            if not base_url:
                raise CompileError(
                    f"no Omni host configured: call .host('acme.omniapp.co') or set {BASE_URL_ENV}"
                )
            api_key = self._api_key or _configuration_value(API_KEY_ENV)
            if not api_key:
                raise CompileError(
                    "no API key configured: call .api_key(...) / .api_key_from_env() or set "
                    f"{API_KEY_ENV}"
                )
            if self._rate_limit_wait is None:
                transport = HttpTransport(
                    base_url=base_url,
                    api_key=api_key,
                    branch_id=self._branch,
                    user_id=self._user_id,
                )
            else:
                transport = HttpTransport(
                    base_url=base_url,
                    api_key=api_key,
                    branch_id=self._branch,
                    user_id=self._user_id,
                    rate_limit_max_wait_seconds=self._rate_limit_wait,
                )
        return OmniSession(
            transport=transport,
            branch=self._branch,
            timezone=self._timezone,
            cache=self._cache,
            user_id=self._user_id,
            owns_transport=owns_transport,
            decomposition_row_cap=self._decomposition_row_cap,
        )

    getOrCreate = get_or_create


class _BuilderAccessor:
    """``OmniSession.builder`` — a fresh builder every time, PySpark-style."""

    def __get__(self, instance: object, owner: type | None = None) -> SessionBuilder:
        return SessionBuilder()


class OmniSession:
    """A configured connection to one Omni organization."""

    __slots__ = (
        "_branch",
        "_cache",
        "_catalog",
        "_decomposition_row_cap",
        "_owns_transport",
        "_read",
        "_timezone",
        "_transport",
        "_user_id",
        "_whoami",
    )

    #: Start here: ``OmniSession.builder.host(...).api_key_from_env().get_or_create()``.
    builder = _BuilderAccessor()

    def __init__(
        self,
        *,
        transport: QueryTransport,
        branch: str | None = None,
        timezone: str | None = None,
        cache: CachePolicy | None = None,
        user_id: str | None = None,
        owns_transport: bool = False,
        decomposition_row_cap: int | None = None,
    ) -> None:
        self._transport = transport
        self._branch = branch
        self._timezone = timezone
        self._cache = cache
        self._user_id = user_id
        self._owns_transport = owns_transport
        self._decomposition_row_cap = decomposition_row_cap
        self._whoami: dict[str, Any] | None = None
        self._catalog = Catalog(transport, preflight=self._preflight)
        self._read = DataFrameReader(self)

    # -- identity ----------------------------------------------------------------------

    def __repr__(self) -> str:
        """Never renders the API key — it is not even stored on this object."""
        return (
            f"OmniSession(transport={self._transport!r}, branch={self._branch!r}, "
            f"timezone={self._timezone!r})"
        )

    def __enter__(self) -> OmniSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the transport, if this session created it."""
        if self._owns_transport:
            self._transport.close()

    @property
    def branch(self) -> str | None:
        """The model branch every query runs against, if any."""
        return self._branch

    @property
    def timezone(self) -> str | None:
        return self._timezone

    @property
    def cache(self) -> CachePolicy | None:
        return self._cache

    @property
    def user_id(self) -> str | None:
        """The impersonated membership id, if any."""
        return self._user_id

    @property
    def decomposition_row_cap(self) -> int | None:
        """The opt-in cap on a mixed aggregation's raw scan; ``None`` means unlimited."""
        return self._decomposition_row_cap

    @property
    def catalog(self) -> Catalog:
        """Discovery: models, topics, views (read-through cached)."""
        return self._catalog

    @property
    def read(self) -> DataFrameReader:
        """``session.read.topic(...)`` / ``session.read.view(...)``."""
        return self._read

    def envelope_options(self) -> EnvelopeOptions:
        """The session-level knobs the compiler puts on the run envelope."""
        return EnvelopeOptions(
            branch_id=self._branch,
            cache=self._cache,
            timezone=self._timezone,
            user_id=self._user_id,
        )

    # -- preflight ---------------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """Run the ``whoami`` preflight now and return its payload.

        Answers three questions in one call: is the key valid, which models can it see, and
        which permissions does it hold on each (``rolesByModel``).
        """
        with _crisp_permissions():
            payload = self._transport.whoami()
        self._whoami = payload
        return payload

    @property
    def whoami(self) -> dict[str, Any]:
        """The cached ``whoami`` payload, running the preflight on first access."""
        if self._whoami is None:
            return self.verify()
        return self._whoami

    def _preflight(self) -> None:
        if self._whoami is None:
            self.verify()

    # -- transport calls ---------------------------------------------------------------

    def run(self, envelope: Mapping[str, Any]) -> QueryResult:
        """Execute a run envelope, after the (cached) preflight."""
        self._preflight()
        with _crisp_permissions():
            return self._transport.run(dict(envelope))

    def plan(self, envelope: Mapping[str, Any]) -> PlanResult:
        """Run an envelope with ``planOnly: true`` — schema and SQL, no data."""
        self._preflight()
        with _crisp_permissions():
            return self._transport.plan(dict(envelope))

    def document_queries(self, document_identifier: str) -> tuple[dict[str, Any], ...]:
        """The queries stored on a document's dashboard (CONTRACT_NOTES §4).

        Two 404s are normal answers rather than outages, and they mean different things: the
        identifier is unknown, or the document exists but carries no dashboard (and therefore
        no saved queries at all).  Both are reported as such.
        """
        if not isinstance(document_identifier, str) or not document_identifier:
            raise CompileError("a document identifier is required")
        self._preflight()
        with _crisp_permissions():
            try:
                payload = self._transport.document_queries(document_identifier)
            except TransportError as exc:
                raise _document_error(document_identifier, exc) from exc
        raw = payload.get("queries")
        if not isinstance(raw, Sequence) or isinstance(raw, str):
            raise OmniframesError(
                f"the saved-query response for {document_identifier!r} carried no 'queries' list"
            )
        return tuple(dict(entry) for entry in raw if isinstance(entry, Mapping))

    def ask(self, prompt: str, *, model: str, topic: str | None = None) -> DataFrame:
        """Ask Omni's AI for a query, and get back a lazy frame that runs it *here*.

        ``runQuery`` is always ``false`` (CONTRACT_NOTES §4): omniframes takes the generated
        query object and executes it through its own pipeline, so the result is normalized,
        limited and explained like every other frame — and the endpoint needs no ``query-api``
        flag.  ``explain()`` shows the prompt that produced the query, and the query itself goes
        on the wire exactly as Omni wrote it.

        Args:
            prompt: what to ask for, in English.
            model: the model (name or id) the question is about.
            topic: optional topic to steer the generator (``currentTopicName``).

        Raises:
            OmniframesError: Omni generated no query for this prompt (400), or the org's AI
                credits are exhausted (402).
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise CompileError("ask() needs a prompt, e.g. session.ask('revenue by state', ...)")
        info = self.catalog.model(model)
        body: dict[str, Any] = {
            "modelId": info.id,
            "prompt": prompt,
            # Explicit, not merely defaulted: the server's own default is `true`, which would
            # run the query outside this pipeline (CONTRACT_NOTES §4).
            "runQuery": False,
        }
        if topic is not None:
            body["currentTopicName"] = topic

        self._preflight()
        with _crisp_permissions():
            try:
                payload = self._transport.generate_query(body)
            except TransportError as exc:
                raise _ask_error(prompt, exc) from exc

        query = payload.get("query")
        if not isinstance(query, Mapping) or not query:
            detail = payload.get("error")
            suffix = f": {detail}" if detail else ""
            raise OmniframesError(
                f"Omni generated no query for {prompt!r}{suffix}. Rephrase the question, or "
                "name the fields you want with select()."
            )
        blob = _normalized_query(dict(query), model_id=info.id)
        return DataFrame(
            self,
            nodes.Scan(
                nodes.SavedQueryScan(
                    document_id=info.id, name=prompt.strip(), query=blob, origin="ask"
                )
            ),
        )


class DataFrameReader:
    """``session.read`` — the two governed ways into a model's data."""

    __slots__ = ("_session",)

    def __init__(self, session: OmniSession) -> None:
        self._session = session

    def __repr__(self) -> str:
        return "DataFrameReader(topic, view, sql, saved_query)"

    def topic(self, model: str, topic: str) -> DataFrame:
        """Read a **topic** — the governed default path (needs ``QUERY_TOPICS``).

        The topic carries the model's join paths, so fields from any joined view can be selected
        without saying how they join.  Both the model and the topic are resolved through the
        catalog here, so a typo fails immediately with the available names.
        """
        catalog = self._session.catalog
        model_info = catalog.model(model)
        summaries = catalog.topics(model_info.name)
        for summary in summaries:
            if summary.name == topic:
                return DataFrame(
                    self._session,
                    nodes.Scan(
                        nodes.TopicScan(
                            model_name=model_info.name,
                            model_id=model_info.id,
                            topic=summary.name,
                            base_view=summary.base_view_name,
                        )
                    ),
                )
        available = ", ".join(sorted(summary.name for summary in summaries)) or "(none)"
        raise CompileError(f"model {model_info.name!r} has no topic {topic!r}. Topics: {available}")

    def view(self, model: str, view: str) -> DataFrame:
        """Read a **bare view**, outside any topic (needs ``QUERY_FULL_MODEL``).

        No topic means no governed join paths: only this view's own fields are selectable.
        Prefer :meth:`topic` unless you specifically want the ungoverned shape.

        The name is validated against every view in the composed model — including views no
        topic reaches, which is the point of a bare-view read (docs/DESIGN.md).  That check is
        one request, not one per topic: it does not need the field metadata it would discard.
        """
        catalog = self._session.catalog
        model_info = catalog.model(model)
        names = catalog.view_names(model_info.name)
        if view in names:
            return DataFrame(
                self._session,
                nodes.Scan(
                    nodes.ViewScan(
                        model_name=model_info.name,
                        model_id=model_info.id,
                        view=view,
                    )
                ),
            )
        available = ", ".join(sorted(names)) or "(none)"
        raise CompileError(f"model {model_info.name!r} has no view {view!r}. Views: {available}")

    def sql(self, model: str, sql: str) -> DataFrame:
        """Run **your** SQL on the model's connection (needs ``QUERY_SQL``).

        The statement is sent as ``userEditedSQL`` with ``rewriteSql: false`` — the marker
        that makes the server run the text verbatim instead of parsing it as OmniSQL
        (CONTRACT_NOTES §3.5).  Omniframes never sends one without the other; that is enforced
        when the step is built, not merely intended.

        The SQL is **opaque**: omniframes does not parse it and cannot push anything into it, so
        every operation written on top runs in the local engine over its result, and
        ``explain()`` says so.  Table names are yours to qualify — the SQL runs against the
        connection's own schema, not against the model's views.

        ``df.schema`` works through a ``planOnly`` round trip; ``df.columns`` needs either that
        or a ``select()``, because until the server plans the statement nobody knows what it
        returns.
        """
        if not isinstance(sql, str) or not sql.strip():
            raise CompileError("read.sql() needs a SQL statement")
        info = self._session.catalog.model(model)
        return DataFrame(
            self._session,
            nodes.Scan(nodes.SqlScan(model_id=info.id, sql=sql.strip(), model_name=info.name)),
        )

    def saved_queries(self, document_identifier: str) -> tuple[tuple[str, int], ...]:
        """``(name, index)`` for every query stored on a document's dashboard (§4).

        The index is what :meth:`saved_query` takes when two saved queries share a name.
        """
        entries = self._session.document_queries(document_identifier)
        return tuple(
            (_saved_query_name(entry, index), index) for index, entry in enumerate(entries)
        )

    def saved_query(
        self,
        document_identifier: str,
        name_or_index: str | int = 0,
        *,
        model: str | None = None,
    ) -> DataFrame:
        """Read a query stored on a document's dashboard and run it **verbatim** (§4).

        The stored blob goes back on the wire exactly as it arrived — omniframes did not write
        it, and re-serializing it would quietly normalize keys it was handed.  The only two
        things touched are the ones the contract says a client must supply: ``modelId`` is
        verified (or injected from ``model=``), and ``version`` is filled in when the blob
        predates it.

        Operations written on top of the frame run locally, for the same reason a raw-SQL scan's
        do: the query is somebody else's, and rewriting it would no longer be *that* query.

        Args:
            document_identifier: the document (workbook/dashboard) identifier.
            name_or_index: the saved query's name, or its position (default: the first).
            model: name or id of the model to run it against, when the blob does not say.

        Raises:
            OmniframesError: the document is unknown, or exists without a dashboard.
            CompileError: no saved query matches ``name_or_index``.
        """
        entries = self._session.document_queries(document_identifier)
        names = [_saved_query_name(entry, index) for index, entry in enumerate(entries)]
        entry, name = _pick_saved_query(entries, names, name_or_index, document_identifier)

        raw = entry.get("query")
        if not isinstance(raw, Mapping):
            raise OmniframesError(
                f"the saved query {name!r} on {document_identifier!r} carried no query object"
            )
        model_id = self._session.catalog.model(model).id if model is not None else None
        blob = _normalized_query(dict(raw), model_id=model_id, label=name)
        return DataFrame(
            self._session,
            nodes.Scan(
                nodes.SavedQueryScan(document_id=document_identifier, name=name, query=blob)
            ),
        )


def _status_of(exc: Exception) -> int | None:
    """The HTTP status a transport error carries.

    :class:`~omniframes.errors.TransportError` records it as an attribute, which is the answer
    for every error this module handles.  The regex stays as a fallback for the errors that
    carry no status of their own (an older transport, a re-raised error built from a message).
    """
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    match = _STATUS_IN_MESSAGE.search(str(exc))
    return int(match.group(1)) if match is not None else None


def _document_error(identifier: str, exc: TransportError) -> Exception:
    """Tell the two documented 404s apart (CONTRACT_NOTES §4); pass anything else through."""
    if _status_of(exc) != 404:
        return exc
    if _NO_DASHBOARD in str(exc):
        return OmniframesError(
            f"the document {identifier!r} exists but has no dashboard, so it has no saved "
            "queries: saved queries live on a dashboard. Open the document in Omni and add "
            "one, or point at a document that already has one."
        )
    return OmniframesError(
        f"no document {identifier!r} is visible to this API key. The identifier is the one in "
        "the document's URL, not its title."
    )


def _ask_error(prompt: str, exc: TransportError) -> Exception:
    """Turn generate-query's two documented refusals into advice (CONTRACT_NOTES §4)."""
    status = _status_of(exc)
    if status == 402:
        return OmniframesError(
            "Omni's AI query generation is unavailable: this organization's AI credits are "
            "exhausted. An Omni organization admin can top them up; every other part of "
            "omniframes keeps working without it."
        )
    if status == 400:
        return OmniframesError(
            f"Omni could not generate a query for {prompt!r}. Rephrase the question (naming the "
            "fields or the topic usually helps), or write the query with select()."
        )
    return exc


def _saved_query_name(entry: Mapping[str, Any], index: int) -> str:
    name = entry.get("name")
    return name if isinstance(name, str) and name else f"(unnamed #{index})"


def _pick_saved_query(
    entries: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    name_or_index: str | int,
    identifier: str,
) -> tuple[Mapping[str, Any], str]:
    """The requested saved query, or a :class:`CompileError` listing what is on the document."""
    available = ", ".join(f"{name!r} [{index}]" for index, name in enumerate(names)) or "(none)"
    if isinstance(name_or_index, bool) or not isinstance(name_or_index, str | int):
        raise CompileError(
            "read.saved_query() takes a saved query's name or its index; got "
            f"{type(name_or_index).__name__}"
        )
    if isinstance(name_or_index, int):
        if not 0 <= name_or_index < len(entries):
            raise CompileError(
                f"the document {identifier!r} has {len(entries)} saved queries, so index "
                f"{name_or_index} is out of range. Available: {available}"
            )
        return entries[name_or_index], names[name_or_index]
    for index, name in enumerate(names):
        if name == name_or_index:
            return entries[index], name
    raise CompileError(
        f"the document {identifier!r} has no saved query called {name_or_index!r}. "
        f"Available: {available}"
    )


def _normalized_query(
    blob: dict[str, Any], *, model_id: str | None, label: str = ""
) -> dict[str, Any]:
    """The stored query, with only the two keys CONTRACT_NOTES §4 says a client must supply.

    Everything else is left exactly as it arrived: this blob is going back on the wire
    unchanged, and "normalizing" somebody else's query is how a client changes the answer.
    """
    if model_id is not None:
        blob["modelId"] = model_id
    if not blob.get("modelId"):
        subject = f"the stored query {label!r}" if label else "the generated query"
        raise CompileError(
            f"{subject} carries no modelId, so Omni cannot tell which model (and therefore "
            "which connection) to run it against. Pass model=... to say which one."
        )
    blob.setdefault("version", QUERY_VERSION)
    return blob


@contextmanager
def _crisp_permissions() -> Iterator[None]:
    """Re-raise the two 403 families with the administrative remedy attached.

    The transport already explains *what* Omni refused; what a user needs next is *who can fix
    it*, which is always an org admin — no amount of retrying or re-keying helps.
    """
    try:
        yield
    except FeatureFlagError as exc:
        raise FeatureFlagError(f"{exc} {_FEATURE_FLAG_ADVICE}", status=exc.status) from exc
    except ModelPermissionError as exc:
        raise ModelPermissionError(
            f"{exc} {_PERMISSION_ADVICE}", permission=exc.permission, status=exc.status
        ) from exc
