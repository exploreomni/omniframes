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
- **Tier 2 — OmniSQL job**. ONE SQLGlot-built statement sent as `userEditedSQL` with the
  `rewriteSql` key **absent**, which is what makes the server parse it as OmniSQL and plan it as
  a governed model job: `${base_view}` in FROM resolves the catalog-provided view against the model, `${view.field}` and
  `${view.measure}` resolve against the model (measures expand to their governed SQL), and
  row-level policies apply. No `staticQueryReferences`, no reference core, no second query
  object — the statement *is* the plan. Covers: ad-hoc aggregations over raw columns,
  post-aggregation filters (HAVING), computed columns and filters no typed filter can express,
  and governed measures mixed with ad-hoc aggregates in one `agg()`. Full design:
  docs/SQLTIER.md (the SQL compiler); wire truth: CONTRACT_NOTES §3.5/§3.6.
- **Tier 3 — local execution.** The splitter pushes maximal remote sub-plans; a small operator
  interpreter (project/filter/join/aggregate/sort/limit/UDF-apply) finishes locally. The local
  engine runs on **pyarrow.compute** (Kleene logic, null-keyed group-by, decimal arithmetic —
  SQL parity for free); pandas appears only at the UDF boundary and as the differential lane's
  independent reference. Full design: docs/HYBRID.md (the splitter and local engine).

**The splitter produces a DAG, not a prefix.** A plan may decompose into MULTIPLE remote
sub-plans feeding local operators. Canonical case — mixed aggregation:

```python
df.group_by("users.state").agg(
    F.measure("order_items.total_sale_price"),  # governed measure → must run remotely
    F.count_distinct("users.id"),  # ad-hoc agg → tier 2, else local over raw rows
)
```

Tier 2 takes this whole node as one statement — `${order_items.total_sale_price}` is a legal
select item beside `COUNT(DISTINCT ${users.id})` — so the common case is a single request. When
tier 2 declines the node (a shape OmniSQL has no rendering for), it decomposes into (a) a tier-1
query for `users.state` + the measure and (b) a raw `users.state` + `users.id` scan aggregated
locally, joined on the group keys locally. Rules:
- Governed measures ALWAYS execute remotely. Their definitions are server-side; the local engine
  never emulates them.
- Ad-hoc aggregations execute in tier 2 when available, else locally over a remote raw-row scan.
- Mixed `agg()` = one tier-2 statement when it compiles; otherwise per-kind sub-plans + a local
  join on the group keys.

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
- **No Omni API calls at build time.** `OmniSession.builder...get_or_create()` resolves each
  setting from an explicit builder value, then its environment variable, then one notebook
  secret provider. `.secrets(provider, scope=..., api_key_name=..., base_url_name=...)` only
  configures lookup; `None` disables it. Automatic selection recognizes loaded Colab or the
  active notebook's `dbutils`, independently of telemetry. Databricks requires a scope for
  secret lookup; Snowflake Workspaces and legacy notebooks require explicit selection and
  their respective secret identifiers. Ambiguous selection fails instead of choosing a
  credential source. The `*_from_env()` helpers are strictly environment-only. Secret reads
  can contact the provider, but complete credentials or an injected transport skip them.
  Provider dependencies are optional and loaded only as needed. The whoami preflight runs
  lazily before the first real Omni call (cached), or explicitly via `session.verify()`.
  Clear errors name the `query-api` flag / `QUERY_TOPICS` / `QUERY_FULL_MODEL` when 403s arrive.
- **`Column.__bool__` raises** with a message directing to `&`, `|`, `~` (never `and/or/not`).
- **Schema access** (`df.schema`) runs a cached `planOnly: true` round trip; `summary.fields`
  (with `missing_fields` checked) is the only schema authority. Never the catalog metadata.
- **Ad-hoc aggregation names**: `F.count_distinct("users.id")` yields column name
  `count_distinct(users.id)` unless aliased.
- **Totals** are opt-in (`df.with_totals()`); normalization strips totals rows and reserved
  columns by default (CONTRACT_NOTES §2.7).
- **`read.view()`** (bare view, no topic) requires `QUERY_FULL_MODEL`; its 403 gets a distinct
  message. `read.topic()` is the governed default path.
- **`read.view()` accepts any view in the composed model**, including views no topic reaches.
  *Settled 2026-08-26.* Validation goes through the flattened `GET /models/{id}/view` list
  (CONTRACT_NOTES §4), which returns every non-ignored view of the composed model. The earlier
  behavior — only views reachable through a topic — was an artifact of validating against
  `Catalog.views()`, which reads views out of topic-detail payloads because that is the only
  place their *fields* have types. Reachability is a topic concept; a bare view is read outside
  any topic, so gating on it contradicted the method's own contract. The change strictly widens
  the accepted set: nothing that used to work stops working. `hidden` views are accepted too —
  the loader filters only `ignored` ones — so a query against a hidden view is the server's call
  to refuse, not a name omniframes rejects locally.
- **The catalog resolves cheaply and hydrates lazily.** *Settled 2026-08-26.* Turning a name or
  id into a model must never enumerate the catalog: `Catalog.model()` uses the exact-match
  `?modelId=` / `?name=` filters (one request), falling back to the cursor walk only to build
  the "available models" error list. Likewise a *name* check never pays for *field metadata* —
  `Catalog.view_names()` (flattened, one request) answers it instead of `Catalog.views()`
  (topic-detail fan-out, `1 + N_topics` requests). Both caches are separate from the full
  listing: a filtered resolution never satisfies `Catalog.models()`, which still walks the whole
  cursor when the user genuinely asks for everything. `Catalog.views()` keeps its typed,
  topic-scoped semantics for callers who actually want the fields — no upstream `include=fields`
  API is needed, because the paths that were paying for that metadata never used it.
- **Security.** API keys live only in the transport; `repr(session)` and all errors/logs redact
  them (tested). Docs never show a literal key. Every HTTP request carries a coarse User-Agent
  with the Omniframes version, Python language and implementation versions, and (when detected)
  an allowlisted Colab or Databricks label. Raw environment values, hostnames, user/workspace/
  cluster identifiers, paths, and compiler/build strings are never included.

## 4. Transport

`QueryTransport` protocol (transport/base.py): `run_query(envelope) -> RunResult`,
`plan_query(query) -> PlanResult`, plus catalog calls. Implementations:
- `HttpTransport` (httpx): request signing, NDJSON accumulation loop with client-side deadline
  budget, wait-loop per CONTRACT_NOTES §2.2, error-envelope mapping (all three shapes), bounded
  retries on connect errors, and WAF-aware 429 recovery: GETs have a cumulative wait budget
  while POSTs remain attempt-bounded. `max_retries` governs GET network failures and POST 429s;
  ambiguous POST network failures are never retried. Explicit timeouts. It sends
  `omniframes/<version> python/<version> <implementation>/<version>` as its User-Agent and adds
  only a static `runtime/google-colab[-enterprise]` or `runtime/databricks` label when a known
  process marker is present.
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
  LIVE-VALIDATE register (CONTRACT_NOTES §6), and it is deliberately outside the offline validation gate
  because it needs a real org.
- **Live integration lane** (`tests/integration/`): pinned WWI results, catalog relationships,
  and permissions for Querier and Restricted Querier PATs. Requires an explicit
  `--live --principal querier|restricted`; without `--live`, these tests skip even when
  credentials are present. Ordinary CI excludes them with `-m "not live"`.
- **FakeOmniAPI** (`tests/fakes/`): in-process httpx.MockTransport ASGI-style fake serving
  whoami/catalog/run/wait with exact NDJSON framing over the bench dataset, executing semantic
  queries via DuckDB (dev dependency only). It implements the wire behaviors exercised by the
  test suite; unsupported behavior is documented in the
  [bench Omni model specification](https://github.com/exploreomni/omniframes/blob/main/internal-docs/bench_omni_model.md).

Validation gate (all checks must pass):

```bash
uv run ruff format --check && uv run ruff check && uv run mypy && uv run pytest -m "not live"
```

## 6. Out of scope for 0.1 (do not build)

Server-side wishlist items; `df.write.table()` (no CTAS endpoint); durable
`save_as_workbook()`; wire-level pivot pushdown (local pivots use pandas through `map_pandas()`);
period-over-period; cross-field OR via `controls`; calculations emission; event telemetry (the
coarse client/runtime User-Agent described above is the only usage metadata sent in 0.x).
