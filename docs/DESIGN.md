# Omniframes — design document

The architecture and the settled semantic decisions. Read together with
[CONTRACT_NOTES.md](CONTRACT_NOTES.md) (the wire contract). These decisions came out of design
review — do not relitigate them casually; if one must change, update this doc in the same change.

## 1. Shape of the library

Five layers, one repo, zero server-side dependencies beyond today's public API:

```
User API      OmniSession / DataFrame / GroupedData / Column / functions (F)
Logical plan  plan/nodes.py — immutable relational nodes
Compiler      compile/ — tier selection, semantic compile, SQL gen, splitter, local engine
Transport     transport/ — QueryTransport protocol; HTTP impl; NDJSON/Arrow/normalize
Omni API      existing public endpoints only
```

Everything user-facing is **immutable and lazy**: a `DataFrame` wraps a plan node; every
transformation returns a new `DataFrame`; nothing touches the network until an action
(`collect`, `to_pandas`, `to_arrow`, `show`, `count`, `first`, `schema`, `explain(analyze=True)`).

## 2. The three tiers

At action time the planner compiles the plan to the highest tier that can express it, maximizing
the remote portion:

- **Tier 1 — semantic query.** Scan(topic/view) + projections (dims, grains, model measures) +
  compilable filters + sorts + limit/offset. Fully governed. Dimension+measure selection IS the
  group-by (Omni semantics); `group_by().agg()` is sugar compiled identically.
- **Tier 2 — SQL job** (M5+). SQLGlot-built outer SQL over embedded semantic sub-queries passed
  via `staticQueryReferences` (`userEditedSQL` + `rewriteSql: false` + `sqlSortsEnabled: true`).
  Covers: ad-hoc aggregations over raw columns, post-aggregation filters (HAVING), computed
  columns not expressible as typed filters. Until M5, these shapes fall to tier 3.
- **Tier 3 — local execution.** The splitter pushes maximal remote sub-plans; a small operator
  interpreter (project/filter/join/aggregate/sort/limit/UDF-apply) finishes locally. The local
  engine runs on **pyarrow.compute** (Kleene logic, null-keyed group-by, decimal arithmetic —
  SQL parity for free); pandas appears only at the UDF boundary and as the differential lane's
  independent reference. Full design: docs/HYBRID.md (authoritative for M3).

**The splitter produces a DAG, not a prefix.** A plan may decompose into MULTIPLE remote
sub-plans feeding local operators. Canonical case — mixed aggregation:

```python
df.group_by("users.state").agg(
    F.measure("order_items.total_sale_price"),  # governed measure → must run remotely
    F.count_distinct("users.id"),  # ad-hoc agg → tier 2, else local over raw rows
)
```

decomposes into (a) a tier-1 query for `users.state` + the measure, and (b) a raw
`users.state` + `users.id` scan aggregated locally, joined on the group keys locally.
Rules:
- Governed measures ALWAYS execute remotely. Their definitions are server-side; the local engine
  never emulates them.
- Ad-hoc aggregations execute in tier 2 when available, else locally over a remote raw-row scan.
- Mixed `agg()` = decompose into per-kind sub-plans + local join on group keys.

`explain()` renders the full split (every remote sub-plan with its tier and payload summary, and
every local operator). No silent local fallback, ever.

## 3. Settled API semantics

- **Limit policy.** The library ALWAYS sends an explicit `limit`. Without a user `.limit(n)`:
  send `DEFAULT_FETCH_LIMIT = 50_000`. Every action warns (`TruncationWarning`) when returned
  rows == the applied limit. `df.limit(None)` maps to `null` (unlimited; the server 400s it with
  pivots). `count()` counts the materialized frame (post-limit, PySpark-consistent) and inherits
  the truncation warning.
- **Alias is client-side only.** The wire has no aliasing; columns arrive as
  `order_items.created_at[month]`. `.alias()` is recorded in a plan-level alias map, applied as
  a rename at normalize time. Sorts/filters written against an alias reverse-resolve to the wire
  name at compile time. Collisions (two aliases to one name, alias shadowing a real field) are
  build-time errors.
- **Naming.** Snake_case primary API (`group_by`, `to_pandas`); thin camelCase aliases
  (`groupBy = group_by`, `toPandas`, `withColumn`, `orderBy = sort`) for PySpark muscle memory.
- **Grain-filter rule** (see CONTRACT_NOTES §3.1): timestamp grains filter on the bare field,
  numeric grains on the bracketed name. The tier-1 filter compiler owns this mapping.
- **Filter-after-aggregation on governed measures IS tier 1**: measure-keyed `filters` entries
  compile to a genuine HAVING server-side (source-pinned; CONTRACT_NOTES §3.1). The compiler
  emits one entry per measure (composite for multiple conditions). Post-aggregation filters on
  AD-HOC aggregates still route to tier 2/3, and tier-2 SQL expresses HAVING in the SQL text
  (the sql_job `filters` map silently skips measure filters).
- **No I/O at build time.** `OmniSession.builder...get_or_create()` performs no network calls.
  The whoami preflight runs lazily before the first real call (cached), or explicitly via
  `session.verify()`. Clear errors name the `query-api` flag / `QUERY_TOPICS` / `QUERY_FULL_MODEL`
  when 403s arrive.
- **`Column.__bool__` raises** with a message directing to `&`, `|`, `~` (never `and/or/not`).
- **Schema access** (`df.schema`) runs a cached `planOnly: true` round trip; `summary.fields`
  (with `missing_fields` checked) is the only schema authority. Never the catalog metadata.
- **Ad-hoc aggregation names**: `F.count_distinct("users.id")` yields column name
  `count_distinct(users.id)` unless aliased.
- **Totals** are opt-in (`df.with_totals()`); normalization strips totals rows and reserved
  columns by default (CONTRACT_NOTES §2.7).
- **`read.view()`** (bare view, no topic) requires `QUERY_FULL_MODEL`; its 403 gets a distinct
  message. `read.topic()` is the governed default path.
- **Security.** API keys live only in the transport; `repr(session)` and all errors/logs redact
  them (tested). Docs never show a literal key.

## 4. Transport

`QueryTransport` protocol (transport/base.py): `run_query(envelope) -> RunResult`,
`plan_query(query) -> PlanResult`, plus catalog calls. Implementations:
- `HttpTransport` (httpx): request signing, NDJSON accumulation loop with client-side deadline
  budget, wait-loop per CONTRACT_NOTES §2.2, error-envelope mapping (all three shapes), bounded
  retries on connect errors, backoff on 429 (respect `X-Omni-Waf-Action`), explicit timeouts.
- Tests use the in-process **FakeOmniAPI** via `httpx.MockTransport` — full wire fidelity.
- A future `BrokerTransport` (in-product notebooks) implements the same protocol; nothing above
  the transport may assume HTTP.

Errors: `OmniframesError` → `AuthError` (bad token), `PermissionError`-family (`FeatureFlagError`
for `query-api`, `ModelPermissionError`), `QueryError` (job error lines, incl. redaction case),
`CompileError` (bad plan/alias/grain), `TruncationWarning` (warning, not error).

## 5. Testing model

- **Golden lane** (`tests/golden/`): compiler snapshots — plan → query JSON — as checked-in
  JSON files, diff-reviewed. No snapshot library.
- **Wire lane** (`tests/wire/`): `HttpTransport` against fixture NDJSON bytes covering every
  documented quirk (string `timed_out`, wait cycles, all error envelopes, redaction, totals,
  unterminated tail, 403s, 429, exotic Arrow types: decimal128, tz-aware timestamps,
  large_string, nulls).
- **Differential lane** (`tests/differential/`): every logical operation executed via pushdown
  (against FakeOmniAPI) vs. pure pandas on the same base data must agree. Comparator: local
  engine uses `dropna=False` group semantics; null ordering normalized before comparison;
  documented dtype-promotion table; NULL-heavy seed rows guaranteed by the bench dataset.
- **Live probe** (`scripts/live_smoke.py`, needs `OMNI_BASE_URL`+`OMNI_API_KEY`): a standalone
  script with a PASS/FAIL/OBSERVED/SKIP protocol, not a pytest lane. It is what closes the
  LIVE-VALIDATE register (CONTRACT_NOTES §6), and it is deliberately outside the milestone gate
  because it needs a real org. There is **no** `tests/live/`: the `live` pytest marker is
  registered but applied to nothing, so `pytest -m live` collects zero tests — never read a green
  run of it as live coverage.
- **FakeOmniAPI** (`tests/fakes/`): in-process httpx.MockTransport ASGI-style fake serving
  whoami/catalog/run/wait with exact NDJSON framing over the bench dataset, executing semantic
  queries via DuckDB (dev dependency only). It implements ONLY what the current milestone's
  tests exercise, and grows per milestone.

Milestone gate (all must pass before a milestone is called done):

```bash
uv run ruff format --check && uv run ruff check && uv run mypy && uv run pytest -m "not live"
```

## 6. Out of scope for 0.1 (do not build)

Server-side wishlist items; `df.write.table()` (no CTAS endpoint); durable
`save_as_workbook()`; wire-level pivot pushdown (`df.pivot()` is local, post-collect);
period-over-period; cross-field OR via `controls`; calculations emission; telemetry (none in 0.x).
