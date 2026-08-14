"""M0 live validation: walk the LIVE-VALIDATE register against a real Omni org.

``docs/CONTRACT_NOTES.md`` was written by reading the Omni server source, not by calling it.
Section 6 of that document lists everything derived-but-unconfirmed.  This script is how those
entries get closed: it drives the transport end to end against a live org and prints, per item,

    PASS      the behavior matched what CONTRACT_NOTES says
    FAIL      it did not — CONTRACT_NOTES (or the transport) is wrong and must be updated
    OBSERVED  an open question; the output *is* the answer, to be written back into the doc
    SKIP      not probeable from here (needs load, credits, or a cold warehouse)

Usage::

    export OMNI_BASE_URL=acme.omni.co        # or https://acme.omni.co/api/v1 — both work
    export OMNI_API_KEY=...                  # never printed, not even on failure
    export OMNI_MODEL_ID=...                 # optional; defaults to the first model returned
    export OMNI_TOPIC=...                    # optional; defaults to the first visible topic
    uv run python scripts/live_smoke.py

Exit codes: ``0`` all required steps passed (or credentials were absent), ``1`` a required step
failed, ``2`` the run aborted early.  Every line printed is scrubbed of the API key.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

from omniframes.compile.querymodel import (
    NullFilter,
    NumberFilter,
    NumberFilterKind,
    Query,
    RunRequest,
    Sort,
)
from omniframes.errors import OmniframesError
from omniframes.transport import HttpTransport, normalize, schema_from_summary
from omniframes.transport.http import REDACTED_API_KEY

PASS: Final = "PASS"
FAIL: Final = "FAIL"
OBSERVED: Final = "OBSERVED"
SKIP: Final = "SKIP"

#: Key used for the ``staticQueryReferences`` syntax probe (LIVE-VALIDATE #1).
REFERENCE_KEY: Final = "omniframes_ref_probe"

#: Above 50 000 the server switches into high-limit override mode; the public docs claim a hard
#: cap of 75 000 (LIVE-VALIDATE #3).
HIGH_LIMIT_PROBE: Final = 75_001

MISSING_ENV_MESSAGE: Final = """\
live_smoke needs credentials for a real Omni org and found none, so there is nothing to do.

    export OMNI_BASE_URL=acme.omni.co   # your Omni host (a full https URL works too)
    export OMNI_API_KEY=omni_osk_...    # org API key or personal access token
    export OMNI_MODEL_ID=<uuid>         # optional: which model to probe
    export OMNI_TOPIC=<topic name>      # optional: which topic to probe
    uv run python scripts/live_smoke.py

The key is read from the environment only; it is never written to a file and never printed.\
"""


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One line of the register."""

    ref: str
    status: str
    detail: str
    evidence: tuple[str, ...] = ()


class Report:
    """Prints each item as it happens and keeps them for the closing summary."""

    def __init__(self, secret: str) -> None:
        self._secret = secret
        self.items: list[Item] = []

    def scrub(self, text: str) -> str:
        """Remove the API key from anything on its way to stdout."""
        return text.replace(self._secret, REDACTED_API_KEY) if self._secret else text

    def emit(self, text: str = "") -> None:
        print(self.scrub(text), flush=True)

    def record(self, ref: str, status: str, detail: str, evidence: Sequence[str] = ()) -> Item:
        item = Item(ref=ref, status=status, detail=detail, evidence=tuple(evidence))
        self.items.append(item)
        self.emit(f"[{status:<8}] {ref} — {detail}")
        for line in item.evidence:
            for physical in line.splitlines():
                self.emit(f"             | {physical}")
        return item

    @property
    def failures(self) -> tuple[Item, ...]:
        return tuple(item for item in self.items if item.status == FAIL)

    def summarize(self) -> None:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        self.emit()
        self.emit("=" * 92)
        rendered = "  ".join(f"{status}={counts.get(status, 0)}" for status in _STATUS_ORDER)
        self.emit(f"live_smoke summary: {rendered}")
        for item in self.failures:
            self.emit(f"  FAILED: {item.ref} — {item.detail}")
        self.emit(
            "Write every OBSERVED result back into docs/CONTRACT_NOTES.md §6 and delete the "
            "register entry it closes."
        )
        self.emit("=" * 92)


_STATUS_ORDER: Final = (PASS, FAIL, OBSERVED, SKIP)


def snippet(value: Any, limit: int = 480) -> str:
    """A compact, JSON-ish rendering of raw evidence, truncated."""
    try:
        text = json.dumps(value, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}…"


def describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------------
# Catalog picking
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldRef:
    """One selectable field from the topic-detail payload."""

    name: str
    data_type: str
    view_name: str


def _field_ref(payload: Mapping[str, Any], view_name: str) -> FieldRef | None:
    if payload.get("hidden") or payload.get("filter_only_field"):
        return None
    field_name = str(payload.get("field_name") or "")
    fully_qualified = str(payload.get("fully_qualified_name") or "")
    name = fully_qualified or (f"{view_name}.{field_name}" if field_name else "")
    if not name or "[" in name:
        return None
    return FieldRef(
        name=name,
        data_type=str(payload.get("data_type") or "UNKNOWN"),
        view_name=str(payload.get("view_name") or view_name),
    )


def collect_fields(topic: Mapping[str, Any]) -> tuple[list[FieldRef], list[FieldRef]]:
    """(dimensions, measures) across every view of a topic-detail payload."""
    dimensions: list[FieldRef] = []
    measures: list[FieldRef] = []
    views = topic.get("views")
    if not isinstance(views, list):
        return dimensions, measures
    for view in views:
        if not isinstance(view, Mapping):
            continue
        view_name = str(view.get("name") or "")
        for key, bucket in (("dimensions", dimensions), ("measures", measures)):
            entries = view.get(key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                ref = _field_ref(entry, view_name)
                if ref is not None:
                    bucket.append(ref)
    return dimensions, measures


def pick_dimension(dimensions: Sequence[FieldRef], base_view: str) -> FieldRef | None:
    """Prefer a plain string dimension on the base view — the least surprising group key."""
    for wants_base_view in (True, False):
        for wants_string in (True, False):
            for ref in dimensions:
                if wants_base_view and base_view and ref.view_name != base_view:
                    continue
                if wants_string and ref.data_type != "STRING":
                    continue
                return ref
    return None


# --------------------------------------------------------------------------------------------
# The register
# --------------------------------------------------------------------------------------------


def step_whoami(client: HttpTransport, report: Report) -> Mapping[str, Any] | None:
    """§1 — the connect-time preflight; works even with the ``query-api`` flag off."""
    try:
        payload = client.whoami()
    except OmniframesError as exc:
        report.record("1. GET /whoami", FAIL, describe_error(exc))
        return None
    user = payload.get("user")
    user_map: Mapping[str, Any] = user if isinstance(user, Mapping) else {}
    status = PASS if user_map.get("id") else FAIL
    report.record(
        "1. GET /whoami",
        status,
        f"keyScope={payload.get('keyScope')!r} orgRole={payload.get('orgRole')!r} "
        f"membershipId={user_map.get('membershipId')!r}",
        [snippet({k: v for k, v in payload.items() if k != "rolesByModel"})],
    )
    return payload


def step_models(client: HttpTransport, report: Report, model_id: str | None) -> str | None:
    """§4 — cursor-paginated model list; also fixes which model everything below uses."""
    try:
        page = client.list_models(pageSize=20)
    except OmniframesError as exc:
        report.record("2. GET /models", FAIL, describe_error(exc))
        return None

    records = page.get("records")
    rows: list[Any] = records if isinstance(records, list) else []
    page_info = page.get("pageInfo")
    chosen = model_id
    if chosen is None:
        for row in rows:
            if isinstance(row, Mapping) and row.get("id"):
                chosen = str(row["id"])
                break

    if chosen is None:
        report.record(
            "2. GET /models",
            FAIL,
            f"no models returned ({len(rows)} records); set OMNI_MODEL_ID explicitly",
            [snippet(page)],
        )
        return None

    names = [row.get("name") for row in rows if isinstance(row, Mapping)][:5]
    report.record(
        "2. GET /models",
        PASS,
        f"{len(rows)} record(s); using modelId={chosen}",
        [f"pageInfo={snippet(page_info)}", f"first names={snippet(names)}"],
    )

    try:
        scoped = client.whoami((chosen,))
    except OmniframesError as exc:
        report.record("2b. GET /whoami?modelId=", FAIL, describe_error(exc))
        return chosen

    roles = scoped.get("rolesByModel")
    role = roles.get(chosen) if isinstance(roles, Mapping) else None
    permissions = role.get("permissions") if isinstance(role, Mapping) else None
    report.record(
        "2b. GET /whoami?modelId=",
        PASS if role is not None else FAIL,
        f"permissions={snippet(permissions)}",
        [snippet(role)],
    )
    return chosen


def step_topic(
    client: HttpTransport, report: Report, model_id: str, topic_name: str | None
) -> tuple[str, str, list[FieldRef], list[FieldRef]] | None:
    """§4 — topic list + detail; the detail endpoint is the only full field-metadata source."""
    try:
        listing = client.list_topics(model_id)
    except OmniframesError as exc:
        report.record("3. GET /models/{id}/topic", FAIL, describe_error(exc))
        return None

    topics = listing.get("topics")
    rows: list[Any] = topics if isinstance(topics, list) else []
    chosen = topic_name
    base_view = ""
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or "")
        if chosen is None and name and not row.get("hidden"):
            chosen = name
        if name == chosen:
            base_view = str(row.get("base_view_name") or "")

    if chosen is None:
        report.record(
            "3. GET /models/{id}/topic",
            FAIL,
            "no topics returned; set OMNI_TOPIC explicitly",
            [snippet(listing)],
        )
        return None

    report.record(
        "3. GET /models/{id}/topic",
        PASS,
        f"{len(rows)} topic(s); using topic={chosen!r} base_view={base_view!r}",
        [snippet([row.get("name") for row in rows if isinstance(row, Mapping)][:10])],
    )

    try:
        detail = client.get_topic(model_id, chosen)
    except OmniframesError as exc:
        report.record("3b. GET /models/{id}/topic/{name}", FAIL, describe_error(exc))
        return None

    topic = detail.get("topic")
    topic_map: Mapping[str, Any] = topic if isinstance(topic, Mapping) else {}
    base_view = str(topic_map.get("base_view_name") or base_view)
    dimensions, measures = collect_fields(topic_map)
    status = PASS if dimensions or measures else FAIL
    report.record(
        "3b. GET /models/{id}/topic/{name}",
        status,
        f"{len(dimensions)} dimension(s), {len(measures)} measure(s)",
        [
            f"dimensions={snippet([f'{d.name}:{d.data_type}' for d in dimensions[:6]])}",
            f"measures={snippet([f'{m.name}:{m.data_type}' for m in measures[:6]])}",
        ],
    )
    if status == FAIL:
        return None
    return chosen, base_view, dimensions, measures


def build_probe_query(
    model_id: str,
    topic_name: str,
    base_view: str,
    dimension: FieldRef,
    measure: FieldRef | None,
) -> Query:
    """The tiny semantic query every execution step below reuses: filter + sort + limit 5."""
    fields = [dimension.name] + ([measure.name] if measure is not None else [])
    sort_column = measure.name if measure is not None else dimension.name
    return Query(
        model_id=model_id,
        table=base_view,
        join_paths_from_topic_name=topic_name,
        fields=fields,
        filters={dimension.name: NullFilter(is_negative=True)},
        sorts=[Sort(column_name=sort_column, sort_descending=measure is not None)],
        limit=5,
    )


def step_plan(client: HttpTransport, report: Report, probe: Query) -> None:
    """§2.4 — ``planOnly`` returns ``summary.fields`` (this is how ``df.schema`` works)."""
    request = RunRequest(query=probe, plan_only=True)
    try:
        request.validate()
        result = client.plan(request.to_wire())
    except OmniframesError as exc:
        report.record("4. planOnly schema", FAIL, describe_error(exc))
        return

    schema = schema_from_summary(result.summary.get("fields"))
    status = PASS if len(schema) else FAIL
    report.record(
        "4. planOnly schema",
        status,
        f"{len(schema)} field(s): "
        + ", ".join(f"{f.name}:{f.data_type.value}" for f in schema.fields),
        [
            f"display_sql={snippet(result.summary.get('display_sql'))}",
            f"missing_fields={snippet(result.summary.get('missing_fields'))}",
        ],
    )


def step_run(client: HttpTransport, report: Report, probe: Query) -> None:
    """§2.2/§2.6/§2.7 — the whole run→wait→Arrow→normalize path on real data."""
    request = RunRequest(query=probe)
    try:
        request.validate()
        result = client.run(request.to_wire())
    except OmniframesError as exc:
        report.record("5. semantic query (filter+sort+limit 5)", FAIL, describe_error(exc))
        return

    table = result.table
    normalized = normalize(table, result.summary.get("fields"))
    limit = probe.effective_limit
    status = PASS if limit is None or table.num_rows <= limit else FAIL
    report.record(
        "5. semantic query (filter+sort+limit 5)",
        status,
        f"{table.num_rows} row(s), {table.num_columns} column(s), "
        f"cache_type={result.summary.get('cache_type')!r}",
        [
            f"arrow schema: {str(table.schema).replace(chr(10), ' | ')}",
            f"dropped reserved columns: {snippet(normalized.dropped_columns)}",
            f"rows: {snippet(normalized.data.to_pylist(), limit=900)}",
            f"stats={snippet(result.summary.get('stats'))}",
        ],
    )


def probe_measure_filter(
    client: HttpTransport, report: Report, probe: Query, measure: FieldRef | None
) -> None:
    """LIVE-VALIDATE #4 — do measure-keyed ``filters`` entries become HAVING?"""
    ref = "LV#4 measure-keyed filter → HAVING"
    if measure is None:
        report.record(ref, SKIP, "the chosen topic exposes no measure")
        return

    having = replace(
        probe,
        filters={
            **probe.filters,
            measure.name: NumberFilter(NumberFilterKind.GREATER_THAN, ["0"]),
        },
    )
    request = RunRequest(query=having)
    try:
        request.validate()
        result = client.run(request.to_wire())
    except OmniframesError as exc:
        report.record(ref, OBSERVED, f"rejected — {describe_error(exc)}")
        return

    display_sql = str(result.summary.get("display_sql") or "")
    missing = result.summary.get("missing_fields")
    verdict = "HAVING present" if "HAVING" in display_sql.upper() else "no HAVING in display_sql"
    report.record(
        ref,
        OBSERVED,
        f"accepted; {verdict}; {result.table.num_rows} row(s); missing_fields={snippet(missing)}",
        [f"display_sql={snippet(display_sql, limit=700)}"],
    )


def probe_timezone(client: HttpTransport, report: Report, probe: Query) -> None:
    """LIVE-VALIDATE #2 — is the ``timezone`` request field usable without org settings?"""
    ref = "LV#2 timezone request field"
    request = RunRequest(query=probe, timezone="America/Los_Angeles")
    try:
        request.validate()
        result = client.run(request.to_wire())
    except OmniframesError as exc:
        report.record(ref, OBSERVED, f"rejected — {describe_error(exc)}")
        return
    report.record(
        ref,
        OBSERVED,
        f"accepted; {result.table.num_rows} row(s)",
        [f"locale_options={snippet(result.summary.get('locale_options'))}"],
    )


def probe_high_limit(client: HttpTransport, report: Report, probe: Query) -> None:
    """LIVE-VALIDATE #3 — what happens above 50 000 / the documented 75 000 cap?

    Planned, not executed: this asks whether the server *accepts* the limit and what it echoes
    back. Register #6 (how a high-limit result actually streams) still needs a real fetch.
    """
    ref = f"LV#3 limit={HIGH_LIMIT_PROBE} (planOnly)"
    request = RunRequest(query=replace(probe, limit=HIGH_LIMIT_PROBE), plan_only=True)
    try:
        request.validate()
        result = client.plan(request.to_wire())
    except OmniframesError as exc:
        report.record(ref, OBSERVED, f"rejected — {describe_error(exc)}")
        return
    report.record(
        ref,
        OBSERVED,
        f"accepted; server echoed limit={snippet(result.query.get('limit'))}",
        [f"plan_stats={snippet(result.summary.get('plan_stats'))}"],
    )


def probe_sql_references(
    client: HttpTransport,
    report: Report,
    model_id: str,
    topic_name: str,
    base_view: str,
    dimension: FieldRef,
) -> None:
    """LIVE-VALIDATE #1 — how is a ``staticQueryReferences`` key spelled inside ``userEditedSQL``?

    Tries the plausible spellings in turn; whichever one comes back with a result (or with
    ``used_query_references``) is the answer that goes into CONTRACT_NOTES §3.5.
    """
    reference = Query(
        model_id=model_id,
        table=base_view,
        join_paths_from_topic_name=topic_name,
        fields=[dimension.name],
        limit=5,
    )
    candidates = (
        ("bare identifier", f"SELECT * FROM {REFERENCE_KEY} LIMIT 5"),
        ("quoted identifier", f'SELECT * FROM "{REFERENCE_KEY}" LIMIT 5'),
    )
    for label, sql in candidates:
        ref = f"LV#1 staticQueryReferences in userEditedSQL ({label})"
        query = Query.for_sql(
            model_id,
            sql,
            limit=5,
            static_query_references={REFERENCE_KEY: reference},
        )
        request = RunRequest(query=query)
        try:
            request.validate()
            result = client.run(request.to_wire())
        except OmniframesError as exc:
            report.record(ref, OBSERVED, f"rejected — {describe_error(exc)}", [f"sql={sql}"])
            continue
        used = result.summary.get("used_query_references") or result.query.get(
            "used_query_references"
        )
        report.record(
            ref,
            OBSERVED,
            f"accepted; {result.table.num_rows} row(s); used_query_references={snippet(used)}",
            [f"sql={sql}", f"columns={snippet(result.table.column_names)}"],
        )


def record_unprobeable(report: Report) -> None:
    """Register entries that cannot be closed by a single scripted run."""
    report.record(
        "LV#5 generate-query response variance",
        SKIP,
        "consumes AI credits; run POST /ai/generate-query by hand with runQuery=false",
    )
    report.record(
        "LV#6 high-limit result streaming",
        SKIP,
        "needs a genuinely large fetch against a warm warehouse; not a smoke-test shape",
    )
    report.record(
        "LV#7 WAF 429 shape",
        SKIP,
        "needs >60 requests/minute on one key; deliberately not triggered here",
    )
    report.record(
        "LV#8 wait-loop timing on a cold warehouse",
        SKIP,
        "needs a multi-minute cold job; run a heavy query with OMNI_DEADLINE_SECONDS raised",
    )


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


def run_register(client: HttpTransport, report: Report) -> None:
    """Walk CONTRACT_NOTES §6 top to bottom, stopping only when a prerequisite is missing."""
    if step_whoami(client, report) is None:
        return

    model_id = step_models(client, report, _env("OMNI_MODEL_ID"))
    if model_id is None:
        return

    picked = step_topic(client, report, model_id, _env("OMNI_TOPIC"))
    if picked is None:
        return
    topic_name, base_view, dimensions, measures = picked

    dimension = pick_dimension(dimensions, base_view)
    if dimension is None:
        report.record(
            "4. planOnly schema",
            FAIL,
            f"topic {topic_name!r} exposes no selectable dimension",
        )
        return
    measure = measures[0] if measures else None

    probe = build_probe_query(model_id, topic_name, base_view, dimension, measure)
    report.emit()
    report.emit(f"probe query: fields={list(probe.fields)} limit={probe.effective_limit}")
    report.emit()

    step_plan(client, report, probe)
    step_run(client, report, probe)

    report.emit()
    probe_measure_filter(client, report, probe, measure)
    probe_timezone(client, report, probe)
    probe_high_limit(client, report, probe)
    probe_sql_references(client, report, model_id, topic_name, base_view, dimension)
    record_unprobeable(report)


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _float_env(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # no options: everything comes from the environment
    base_url = _env("OMNI_BASE_URL")
    api_key = _env("OMNI_API_KEY")
    if base_url is None or api_key is None:
        print(MISSING_ENV_MESSAGE)
        return 0

    report = Report(api_key)
    try:
        client = HttpTransport(
            base_url,
            api_key,
            max_deadline_seconds=_float_env("OMNI_DEADLINE_SECONDS", 300.0),
            user_id=_env("OMNI_USER_ID"),
            branch_id=_env("OMNI_BRANCH_ID"),
        )
    except OmniframesError as exc:
        print(f"could not build the transport: {describe_error(exc)}")
        return 2

    report.emit(f"live_smoke against {client!r}")
    report.emit("closing the LIVE-VALIDATE register in docs/CONTRACT_NOTES.md §6")
    report.emit("=" * 92)
    try:
        run_register(client, report)
    except Exception as exc:  # a crash must never print an unscrubbed traceback
        report.record("live_smoke", FAIL, f"aborted — {describe_error(exc)}")
        report.summarize()
        return 2
    finally:
        client.close()

    report.summarize()
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
