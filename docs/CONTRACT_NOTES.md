# Omni Query API — contract notes

**This document is the source of truth for everything Omniframes puts on the wire.** It was
produced by reading the Omni monorepo server source (checkout `~/omni/omni`, commit
`86569cd50c6197836ed66365ccc2258b213ea2de`, 2026-08-14 — §4's model filters and flattened view
list re-verified at `916a42204fca897542433ede1d6f8bc074cd4fd0`, 2026-08-25), NOT the published
OpenAPI spec — the
public spec is wrong in several load-bearing places (see [Public-docs discrepancies](#5-public-docs-discrepancies)).
File references below are paths within that monorepo.

Anything marked **`LIVE-VALIDATE`** is derived from source but not yet confirmed against a live
org; the register at the bottom lists them all.

---

## 1. Authentication & preflight

- `Authorization: Bearer <token>`. Token = org API key or personal access token (PAT), format
  `omni_osk_` + 50 base62 chars + 6-char CRC. Env convention (shared with the official
  `omni-python-sdk`): `OMNI_API_KEY`, `OMNI_BASE_URL`.
- Malformed header → **400** `"Bad authorization header, must be formatted as Bearer <token>"`.
- Invalid token → **403** (`{"error": {"code": 403, "message": "Invalid bearer token"}}`) — NOT
  401, despite the OpenAPI spec.
- An **org key acts as its creator** (permissions-wise). A PAT additionally re-checks
  `canUserCreateApiKeys` per request — a demoted PAT gets 403
  `"User no longer has the required permissions to use this API key"`.
- **Impersonation:** `?userId=<membershipId>` (query param preferred; body field is legacy;
  providing both → 400). It's a **membership id**, not a user id. Org keys may impersonate any
  membership (unknown → 404 `"User with id <x> does not exist"`); PATs may only pass their own
  (else 403 `"User-scoped API keys can only be used to act on behalf of the authenticated user."`).
- **`GET /api/v1/whoami`** — the connect-time preflight. Works even when the `query-api` feature
  flag is off (it skips that check), so it's the right first call. Optional `?modelId=<uuid|csv>`.
  **Model-filtered calls 404** (`"Model(s) not found: <id>"`) for any requested model without a
  *resolved role row* — deliberately indistinguishable from a nonexistent model
  (`api.v1.whoami/get-handler.server.ts:115-118`). `GET /models` freely lists models (branch
  models, schema models) that trip this, so never treat the model list as whoami-resolvable;
  live-confirmed 2026-08-14 (first listed model of the demo org 404s while its topics and
  queries work fine). Omniframes only ever calls whoami unfiltered.
  Response: `{user: {id, membershipId}, keyScope: "user"|"organization", orgRole,
  rolesByModel: {<modelId>: {roleName, baseRole, connectionId, permissions: [...]}}}` with
  `rolesByModelTruncated: true` when >200 models. Relevant permissions: `QUERY_TOPICS` (topic
  queries), `QUERY_FULL_MODEL` (bare-view queries), `QUERY_SQL`, `VIEW_SQL` (unredacted SQL/errors).
- **`query-api` feature flag off** → 403 `{"detail": "Feature not enabled", "status": 403}` on
  query endpoints. Missing `QUERY_TOPICS` → 403 `{"detail": "Permission denied", "status": 403}`.
- **Rate limiting** is at the AWS WAF: 60 req/min keyed on the exact `Authorization` header value
  (elevated tiers exist per-org). Blocked → **429** with header `X-Omni-Waf-Action: block`.
  All clients sharing a key share one bucket. For idempotent GETs, Omniframes honors a numeric
  `Retry-After` value verbatim within its per-request cumulative wait budget; POST 429s remain
  attempt-bounded.

### Error envelopes (a client must handle all three)

| Shape | Source |
|---|---|
| `{"detail": "<msg>", "status": <code>}` | Remix route errors (validation, forbidden, etc.) |
| `{"error": {"code": <code>, "message": "<msg>"}}` | Express auth middleware (bad/expired token) |
| `{"error": "<msg>", "success": false}` | `/models` cursor-parse failure |

---

## 2. `POST /api/v1/query/run`

### 2.1 Request envelope

```jsonc
{
  "query": { /* §3 — REQUIRED */ },
  "branchId": "<uuid>",          // optional; TOP-LEVEL ONLY (inside query → hard 400)
  "cache": "SkipRequery",        // Standard | SkipRequery | SkipCache | SkipCacheAndRebuildExtracts
                                 // default SkipRequery. The OpenAPI values (disabled|normal|...) 400.
  "resultType": "csv"|"json"|"xlsx",  // switches to single-document mode (§2.5)
  "formatResults": true,         // only valid WITH resultType
  "planOnly": false,             // schema/plan without execution (§2.4)
  "timezone": "America/Los_Angeles", // 400 unless the connection enables user-specific timezones
                                     // ("The timezone parameter requires user-specific timezones
                                     //  to be enabled on the connection.") — live-confirmed 2026-08-14
  "userId": "<membershipId>",    // legacy body location; prefer ?userId= query param
  "workbookUrl": false           // true → response header X-Omni-Workbook-Url
}
```

400-refinements: `planOnly`+`resultType`; `planOnly`+`workbookUrl`;
`workbookUrl`+non-empty `query.staticQueryReferences`; `formatResults` without `resultType`;
`query.limit: null` + non-empty `query.pivots` without `resultType`; `userId` in both places.

Only `query.fields` (array of strings), `query.modelId` (uuid) and `query.limit` (positive or
null) are schema-validated server-side. Everything else passes through to the Kotlin planner
(snake_case Jackson, unknown keys **silently dropped** — misspelled fields don't error).

### 2.2 NDJSON response (default mode)

`200`, `Content-Type: text/ndjson` (not `application/x-ndjson`). One JSON object per `\n`;
trailing separator after every line. Parse tolerantly: a mid-stream upstream failure appends a
final **unterminated** line `{"message": "...", "reason": "..."}`.

Line sequence:

1. **Header**: `{"jobs_submitted": {"<job-uuid>": "<client-result-id or null>"}}`.
   (The upstream `omni_job_ids` header is NOT forwarded — job ids come from this line only.)
2. **Job lines** — exactly one per job, emitted when the job reaches a terminal state within the
   wait window. Fields (snake_case, nulls omitted):
   - Success: `{"job_id", "status": "COMPLETE", "client_result_id", "summary": {…§2.3},
     "cache_metadata": {…}, "query": {…}, "result": "<base64 Arrow IPC stream>",
     "stream_stats": {"server_stream": <ms>}}`. Requery variants may add `requery_sql`,
     `requery_table_name`, `requery_fallback_sql`, `column_name_mapping` — a line with
     `requery_sql` and NO `result` means client-side materialization is expected; we treat that
     as an error (should not happen on the API path, guard anyway).
   - Error: `{"job_id", "status": "ERROR"|"FAILED", "error_type", "error_message",
     "kill_reason"?, "summary"?, "error"?, "query"?}`.
     `client_result_id` may be the **literal string `"null"`**.
     `error_type` ∈ KILL, QUERY, FORBIDDEN, UNKNOWN, INTERNAL, REQUERY, UPLOADED_TABLE_GONE,
     CONTEXTUAL, CALCS_FAILED, PLAN, REDIRECT, OAUTH_AUTHENTICATION_REQUIRED, OFFLINE,
     PLAN_PK_REQUIRED_DUE_TO_JOIN_FANOUT, PLAN_CALC_REQUIRES_SORT, DBT_RENDER_FAILED,
     RATE_LIMITED. Without `VIEW_SQL` permission, `error_message` is replaced with
     `"Query failed. Error details are only visible to users with permission to view SQL."` and
     SQL fields in `summary` are blanked.
   - `/query/wait` may also emit `{"job_id", "status": "FAILED", "error_message": "Unable to find job"}`
     for unknown job ids.
   - **`status` is an OPEN enum** (COMPLETE, ERROR, FAILED observed terminal; ADDED, EXECUTING,
     PLANNING, PLANNING_COMPLETE, PLANNED exist). Treat unknown statuses as non-terminal.
   - **Gotcha:** `status: "COMPLETE"` whose `summary.display_sql` contains
     `"Failed to plan query"` is a REAL FAILURE. Check for it.
3. **Footer**: `{"remaining_job_ids": [...], "timed_out": "true"|"false"}` — `timed_out` is a
   **string**, and is `"true"` iff `remaining_job_ids` is non-empty.

**Polling loop:** run's server-side wait window is ~10 s; `/query/wait` slices are ~30 s. Loop:

```
lines = POST /query/run
while footer.remaining_job_ids:
    check client deadline; sleep briefly if the remaining set didn't shrink
    lines += GET /api/v1/query/wait?jobIds=<comma-joined ids>
```

`/query/wait` params: `jobIds` (comma-separated, preferred) or legacy `job_ids` (JSON array or
csv). No `wait_time_ms` on the public endpoint. Response = same NDJSON minus the header line.
A poll that times out returns only a footer. Job lines appear exactly once — accumulate across
run+wait responses. No per-model authz on wait (org-scoped + unguessable job ids).

### 2.3 `summary`

`summary.fields: Record<fieldName, Field>` is the **authoritative schema** for interpreting the
decoded Arrow — `data_type` per field: `ARRAY | BOOLEAN | INTERVAL | JSON | NUMBER |
OTHER_UNGROUPABLE | STRING | TIMESTAMP | UNKNOWN`. Also per field: `field_name`,
`fully_qualified_name`, `view_name`, `is_dimension`, `is_calc`, `label`, `format`, `date_type`,
`aggregate_type`, `filter_only_field`, `hidden`.

Other summary keys: `display_sql`, `omni_sql`, `omni_sql_parse_failed`, `cache_type`
(`EXACT | EXACT_STALE | PARTIAL | CUBE_REQUERY | EXTRACT_REQUERY | MISS | UNKNOWN`),
`stage_summaries[] {succeeded, warnings[]}`, `stats`, `plan_stats`, `missing_fields: string[]`,
`invalid_calculations: Record<calcName, error>`, `locale_options`.

**`missing_fields` non-empty means the server dropped fields you asked for** (bad name, bad
grain). Surface this as an error client-side — the query "succeeds" otherwise.

### 2.4 `planOnly: true`

Job line has `status: "PLANNED"`, `summary` (incl. `fields` — this is how `df.schema` works
without execution) and `query`, but NO `result` / `cache_metadata`.

### 2.5 `resultType` mode (we use it only for `session.ask` parity; not for frames)

Single document, `text/csv` / `application/json` / `application/vnd.ms-excel`. Timeout is
**HTTP 408** with `{"detail": "Query timed out", "remaining_job_ids": [...], "timed_out": true}` —
boolean here, unlike the NDJSON footer. `json` rows are keyed by **display label** (deduplicated),
not field name.

### 2.6 Arrow decoding

`result` is base64 of an **Arrow IPC stream** (not file):
`pyarrow.ipc.open_stream(io.BytesIO(base64.b64decode(result))).read_all()`.

### 2.7 Result normalization (before any frame reaches the user)

- **Strip reserved columns** matching: `^\$omni_`, `^__omni_`, `__omni_sort(_\d+)?$`,
  `__omni_summ(_\d+)?$`.
- **Totals rows** are flagged by indicator columns `$omni_column_total_indicator` /
  `__omni_column_total_indicator` (+ `…row_total_indicator` variants). Values: `column_total`,
  `row_total`, `::total::` (grand total), `column_subtotal::<fieldName>`. There is **no**
  `row_type` column on `/query/run` — derive it. Totals rows are removed by default and exposed
  via `with_totals()`.
- **`<field>__omni_summ` sidecars** (lowercased field name) carry totals values for raw-SQL
  queries with `column_totals` — on a totals row read the sidecar, falling back to the base column.
  Only the first occurrence wins when `_\d+`-suffixed duplicates exist.
- `column_name_mapping` appears only on requery lines; inert for us.
- **Grain `__raw` sidecars (live-observed 2026-08-14, tier 1 AND OmniSQL alike).** When the
  model formats a grain, selecting `field[grain]` yields TWO columns: `field[grain]__raw`
  (`DATE_TRUNC` value, `TIMESTAMP`, at the item's original select position) and `field[grain]`
  (formatted `STRING`, e.g. `TO_CHAR(..., 'YYYY-MM')`, appended after the other columns).
  `__raw` does NOT match the reserved-column regexes above. Normalization reconciles the pair —
  `__raw` wins, under the plain name, and the formatted column is dropped (docs/SQLTIER.md §5);
  FakeOmniAPI emits the pair for the bench model's formatted month grain on both paths.

---

## 3. The query object

The server applies these defaults when keys are absent (`createQuery`): `limit: 1000`,
`column_limit: 50`, `version: <latest>`, `table: ''`, empty collections for the rest.
**Omniframes always sends explicit values for everything it controls.**

```jsonc
{
  "modelId": "<uuid>",                  // REQUIRED (camelCase in JSON)
  "table": "order_items",              // base view; may be "" when join_paths_from_topic_name set
  "join_paths_from_topic_name": "order_items_topic",  // the TOPIC — governed join paths
  "fields": ["users.state", "order_items.total_sale_price", "order_items.created_at[month]"],
  "filters": { "<fieldName>": { /* §3.1 */ } },       // implicitly AND across fields
  "sorts": [ { "column_name": "order_items.created_at[month]",  // EXACT field name incl. bracket
               "sort_descending": true,
               "null_sort": "OMNI_DEFAULT" } ],       // LAST | FIRST | DIALECT_DEFAULT | OMNI_DEFAULT
  "limit": 50000,                       // ALWAYS explicit. null = unlimited. >50000 = high-limit mode.
  "offset": 0,
  "pivots": [],                         // pivot fields must ALSO be in fields
  "calculations": [],                   // §3.3 — requires client-built sql_expression; avoid
  "fill_fields": [],
  "column_totals": {},                  // Record<field | "::total::", {"type": "aggregation"}>
  "row_totals": {},
  "userEditedSQL": "",                  // §3.4
  "default_group_by": true,
  "version": 9                          // ALWAYS send 9 (QUERY_VERSION_CALC_PUSHDOWN)
}
```

Also available: `staticQueryReferences` (§3.5), `sqlSortsEnabled` (§3.4), `column_limit`,
`custom_summary_types`, `join_via_map`, `controls` (cross-field OR — see §3.1), period-over-period
machinery (out of scope for 0.1).

### 3.1 Filters

Keyed by field name; discriminated on `type`. Common optional keys on every arm:
`is_negative: bool|null`, `cancel_query_filter`, `ignore_if_unjoinable`.

| type | Shape |
|---|---|
| `string` | `{kind: CONTAINS\|ENDS_WITH\|STARTS_WITH\|EQUALS\|IS_EMPTY\|SQL_LIKE, values: string[], case_insensitive?}`. EQUALS + many values → IN; others OR. IS_EMPTY takes `values: []`. |
| `number` | `{kind: LESS_THAN\|GREATER_THAN\|EQUALS\|BETWEEN, values: string[], is_inclusive?}`. **Values are STRINGS** (`"42"`) — the raw path has no number→string transform. `is_inclusive` makes GT/LT into ≥/≤. BETWEEN = `[lower, upper]`, **upper exclusive**. |
| `date` | `{kind, left_side?, right_side?}`. Kinds: BETWEEN, ON_OR_AFTER, BEFORE, TIME_FOR_INTERVAL_DURATION, TIME_FOR_UNIT_DURATION, QUERY_OFFSET, IS_ON_DAY_OF_WEEK, IS_ON_DAY_OF_MONTH, IS_ON_DAY_OF_QUARTER, IS_ON_DAY_OF_YEAR, IS_IN_MONTH_OF_YEAR, IS_IN_QUARTER_OF_YEAR, IS_IN_WEEK_OF_YEAR, IS_AT_HOUR_OF_DAY. Mappings: single unit ("2023-03", "last quarter") → TIME_FOR_UNIT_DURATION via `left_side`; relative range ("last 30 days") → TIME_FOR_INTERVAL_DURATION `left_side: "30 days ago"`, `right_side: "30 days"`; absolute range → BETWEEN (end exclusive); ON_OR_AFTER uses `left_side`; BEFORE uses `right_side` (exclusive). Literal grammar: `"YYYY-MM-DD HH:MM:SS"` (truncatable), `"2012 Q3"`, `"FY2012"`, `"N [complete] <unit> ago"`, `"today"/"yesterday"/"tomorrow"`, `"this/last/next <period>"`. |
| `boolean` | `{is_negative: false}` = is true, `{is_negative: true}` = is false, `is_negative` omitted/null = no-op placeholder. `treat_nulls_as_false?`. |
| `null` | `{is_negative?}` — IS NULL / IS NOT NULL. |
| `composite` | `{conjunction: "AND"\|"OR", filters: [<any non-composite or composite>], is_negative?}` — recursive, **no depth cap on this endpoint** (the cap of 4 applies only to document writes). Composites nest within ONE field's entry. |
| `query` | `{field_name, query_id, is_negative?, disregard_limit?}` — `query_id` is a `staticQueryReferences` key; `field_name` required. |
| `user_attribute` | `{user_attribute_name}` |

Cross-field OR requires the `controls` array (`MULTI_FIELD_FILTER`) — **out of scope for 0.1**
(tier 2/3 handles cross-field OR instead).

**Measure filters → HAVING (source-pinned; live-confirmed 2026-08-14** — `display_sql` came back
with `HAVING COUNT(*) > 0` and correct rows**).** A `filters` entry keyed by a MEASURE field name
compiles to a genuine `HAVING` over the aggregate on the semantic path
(`QueryToRelContext.partitionFilters` branches on the model field type; the filter is translated
against `meas.toAggregateExpression(...)` and applied post-`aggregate` —
`AlgebraFromApi.kt:3181-3199`; proven by cross-dialect SQL snapshots in `TestSqlGenCore.kt`
"filters on measures uber-test"). Details that matter to the compiler:
- A measure filtered but not selected is force-added to the aggregate and projected away — it
  never appears as a result column.
- Exactly ONE filter entry per measure (server keys them in a map) — multiple conditions on the
  same measure must be a `composite` in that one entry (our normalizer already does this).
- Dimension filters stay WHERE-side (pre-aggregation). No version gating on any of this.
- **Raw-SQL jobs: measure filters are silently SKIPPED** (the SQL wrapper view has an empty
  measures map — `OmniJobPlanner.kt:2798-2803`). Tier-2 SQL must express HAVING in the SQL text
  itself, never via the job's `filters` map.
- Nonexistent filter key → hard error `No such field "<name>"` (unless `ignore_if_unjoinable`).

**Grain-filter rule:** timestamp-producing grains (`month`, `week`, `date`, `hour`, …) are
filtered on the **bare field name** (no bracket) with a date filter; number-producing grains
(`hour_of_day`, `day_of_week_num`, `month_num`, …) are filtered on the **bracketed name** with a
number filter. Getting this wrong yields missing-field or wrong semantics.

### 3.2 Time grains

`field[grain]`, case-insensitive, canonical lowercase. Date grains: `year, quarter, month, week,
date, hour, minute, second, millisecond, quarter_of_year, week_of_year, day_of_week_name,
day_of_week_num, month_name, month_num, hour_of_day, day_of_month, day_of_year, day_of_quarter,
fiscal_year, fiscal_quarter, epoch, time_of_day`. Duration dimensions accept `seconds, minutes,
hours, days, weeks, months, quarters, years`. Invalid grain → the field lands in
`summary.missing_fields` (no hard error!).

### 3.3 Calculations — effectively unavailable on this endpoint

`{calc_name, sql_expression}` where `sql_expression` is a serialized parse tree the client must
build (`{type: field|literal|call|…}` with operators like `Omni.OMNI_FX_MULTIPLY`).
`original_formula` alone does NOT work here — formula parsing happens only on document-write
paths, and Kotlin requires `sql_expression`. Calcs are **post-limit row operations**, not in-DB
aggregation. Consequence: **Omniframes does not emit calculations in 0.1**; ad-hoc expressions
and aggregations ride tier 2 (SQL) or tier 3 (local). `calc_name` must also appear in `fields`
if we ever emit one.

### 3.4 Raw SQL jobs

Set `userEditedSQL: "<sql>"` **and** `rewriteSql: false` (else the text is parsed as OmniSQL
and planned as a governed model job — §3.5). `sqlSortsEnabled: true` makes the server apply `sorts` / `calculations` /
`column_totals` on top of the SQL result; when false those are stripped. The connection is
resolved server-side from `modelId`. `resultType`/NDJSON behavior is identical.

### 3.5 `staticQueryReferences` — and how `userEditedSQL` actually composes

`Record<refKey, <full query object> & {"model_id": "<uuid>"}>` (note the extra **snake_case**
`model_id` inside each referenced query). The only way an external client passes query
references; they become `query_references` on the job (single job per call — references fold
into one plan). Consumed by **exactly two** things:
1. `type: "query"` filters (`query_id` = refKey);
2. XLOOKUP-family calc operators (`used_query_references` in a parse response is populated
   *only* from calculation expressions — `QueryParseResponse.kt` `usedQueryRefs =
   calculations.flatMap { collectLookupQueryIds(...) }`; reference translation feeds
   filter-by-query and lookups — `OmniJobPlanner.kt` `translateQueryReferences`).

**A refKey can NOT be referenced as a table in `userEditedSQL`.** Live-refuted 2026-08-14
(omni.demo.exploreomni.dev): bare and double-quoted spellings pass through to the warehouse
verbatim (`relation "<key>" does not exist` from Postgres); the `${refKey}` spelling fails
OmniSQL substitution (`No such view "<key>"`). There is no server mechanism for it on
`/query/run`.

**How `userEditedSQL` is actually processed** (`OmniJobPlanner.kt:502-534`):
- `rewriteSql: false` **or** a line matching `^\s*--\s*DO NOT PARSE\s*$` in the SQL
  (`OmniSqlParser.doNotParseRegex`), with no sorts/calcs on the job → **verbatim raw SQL job**:
  the text goes to the warehouse untouched. `staticQueryReferences`, the `filters` map's measure
  entries, and all semantic constructs are ignored on this path.
- Otherwise the SQL is parsed as **OmniSQL**: `${view}` / `${topic}` in FROM position and
  `${view.field}` in expressions resolve against the model, and the job is planned as a governed
  model job (parse failure falls back to a raw SQL job via `PlannerFailedException`; an
  unresolvable `${…}` ref is a hard error, not a fallback). Live-confirmed 2026-08-14: plain
  SELECTs and ad-hoc aggregates (`COUNT(DISTINCT ${view.field})` + GROUP BY/ORDER BY/LIMIT)
  compile to governed SQL; result columns come back semantically scoped (`view.field`; an
  SQL alias `n` surfaces as `<base_view>.n`). **This — not `staticQueryReferences` — is the
  composition mechanism for tier-2-style outer SQL.**

Constraint: rejected with `workbookUrl: true`.

### 3.6 The OmniSQL path (`userEditedSQL` with `rewriteSql` omitted)

All live-confirmed 2026-08-14 against omni.demo.exploreomni.dev (Postgres connection) with a
~20-case probe battery; server-source pins in §3.5. This is the delivery mechanism for tier 2.

**Envelope**: `userEditedSQL: "<omnisql>"` + `modelId`, with `rewriteSql` **absent** (not
`false`). The server parses the text as OmniSQL and plans a **governed model job**: joins come
from the topic's relationships (pruned to the views actually referenced), measures expand to
their governed SQL, row-level policies apply.

**Reference syntax**:
- `${topic}` in FROM position — brings the topic's join graph. (`${view}` also resolves; the
  view-vs-topic precedence when names differ is unverified — Omniframes always names its topic.)
- `${view.field}` anywhere in an expression: select items, WHERE, HAVING, ORDER BY, function
  args, arithmetic. Bracketed grain refs work: `${order_items.created_at[month]}`.
- `${view.measure}` — a governed measure ref, expanded server-side (e.g.
  `COALESCE(SUM("sale_price"), 0)`), **freely mixable with ad-hoc aggregates in one
  statement**: `SELECT ${users.country}, ${order_items.sale_price_sum}, COUNT(DISTINCT
  ${users.id}) FROM ${order_items} GROUP BY 1` works. Without GROUP BY a measure-only select
  is a grand total.

**Result naming (probed exhaustively 2026-08-14; two regimes)**:
- **Bare refs** (`${view.field}`, `${view.measure}`, bare grain refs): SQL aliases are
  **ignored** and duplicate selections are **deduplicated** — one column per distinct field,
  named exactly `view.field`, regardless of aliasing or how many times it appears. This is
  tier-1 semantics: renames are the client's job.
- **Expression items** (anything that isn't a bare ref — arithmetic, functions, aggregates,
  literals): the SQL alias **is** honored and duplicates survive, surfacing as
  `<scope_view>.<alias>` (unaliased `COUNT(*)` → `count`; literal-only items scope to the FROM
  ref's base view). The `scope_view` prefix is **not reliably predictable** for multi-view
  expressions — first-ref-wins is refuted (`${users.age}+${products.cost}` → `users.…` but
  `COALESCE(${products.brand}, ${users.country})` → `users.…` and
  `${order_items.sale_price_sum}/COUNT(DISTINCT ${users.id})` → `users.…`); clients must
  match expression columns by alias suffix, never by predicting the prefix.

**Verified constructs**: GROUP BY (positional), HAVING over ad-hoc aggregates AND over
`${measure}` refs (expanded inline, probe L4), ORDER BY with `NULLS LAST` — including ORDER BY
a non-selected field (the server wraps the statement in a subquery with `omni_sort_expr_n`
sidecars that do NOT leak into the result, probe L5) — `LIMIT`/`OFFSET`, functions (`UPPER`,
`CAST`, `COALESCE`, `NULLIF`), arithmetic including measure arithmetic
(`${m1}/NULLIF(${m2},0)`, `${m}/COUNT(DISTINCT ${f})`, probe L3), `LIKE … ESCAPE '!'` (parsed
and re-rendered with the warehouse's escape char, probe L2), `''`-doubled string literals,
CTEs (accepted and **flattened** — a CTE + outer WHERE on an aggregate is rewritten into
HAVING on one statement). `planOnly` works; `summary.fields` carries the scoped names/types.

**Hard constraints — silent rewrites observed**:
- The **query-object `limit` is IGNORED** on this path, and a statement without `LIMIT` runs
  unlimited. `LIMIT`/`OFFSET` must always be in the SQL text.
- `SELECT DISTINCT` is **silently stripped**. Never emit it (dedup stays local).
- WHERE predicates mixing a bare field ref AND a grain ref of the same field are **merged with
  predicate loss** (observed: the tighter bound dropped). Plain same-column compound ranges are
  SAFE — `(${f} >= x AND ${f} < y)` survives with both bounds intact (probe L1). Rule: never
  reference a field's grain variant in WHERE alongside (or instead of) the bare field;
  bare-ref-only conjuncts compose fine.
- String literals: backslash is not an escape (passes verbatim); the server re-renders the
  statement per warehouse dialect, so client-side dialect knowledge is unnecessary here (the
  tier-2 `sql_dialect`/backslash-refusal machinery is obsolete on this path; it remains relevant
  only to verbatim `read.sql`).
- Grain select items split into `__raw` + formatted pairs exactly as tier 1 does (§2.7).

**Errors**: unknown FROM ref → `Could not substitute Omni SQL … No such view "<name>"`;
unknown field → `… Field "<name>" not found … No such field "<name>"`. Both arrive as normal
job-error lines.

`LIVE-VALIDATE` residue for this path is tracked in §6 item 11.

---

## 4. Catalog endpoints

- **`GET /api/v1/models`** — cursor-paginated: `?pageSize=` (1..100, default 20), `?cursor=`
  (opaque — echo back exactly), `?modelKind=SHARED|...`, `?name=`, `?modelId=`,
  `?sortField/sortDirection`.
  `?name=` and `?modelId=` are **exact** matches, not substring — but `?name=` is
  **case-insensitive**: it compares against `OmniModel.name`, a Postgres `CITEXT` column
  (`packages/db-models/prisma/schema.prisma:1391`). `buildWhereClause` hands Prisma
  `...(name && {name})` / `...(modelId && {id: modelId})`
  (`packages/bi-app/app/routes/api.unstable.models/get-handler.server.ts:56,58`, re-exported at
  `api.v1.models.ts`). `modelId` is declared `z.uuid()` on a `.strict()` schema
  (`packages/bi-app/app/types/api/models/schema.ts:2260`), so a **non-UUID `modelId` is a 400,
  not an empty page** — only send it for a UUID-shaped input. Both filters make model resolution
  a single request instead of a full cursor walk.
  **A filter is only applied when the server considers it truthy.** `name` is
  `z.string().optional()` with no min-length, so `?name=` passes validation and `...(name &&
  {name})` then drops it — the reply is page one of the *whole catalog*, not an empty page. A
  filtered reply must therefore be checked against the filter that was sent; treating
  `records[0]` as the answer silently resolves an arbitrary model. (`modelId` differs: `z.uuid()`
  rejects the empty string outright, so it 400s rather than being dropped.)
  Response `{pageInfo: {hasNextPage, nextCursor, pageSize, totalRecords}, records: [{id, name,
  modelKind, connectionId, baseModelId, createdAt, updatedAt, deletedAt}]}`. The **strict**
  param schema 400s on unknown query params.
- **`GET /api/v1/models/{modelId}/topic`** — list: `{success, topics: [{name, base_view_name,
  label, description, group_label, hidden}]}`. Optional `?branch_id=`.
- **`GET /api/v1/models/{modelId}/topic/{topicName}`** — detail; THE field-metadata endpoint:
  `{success, topic: {..., views: [{name, label, dimensions: Field[], measures: Field[],
  filter_only_fields: Field[]}], relationships: [{join_type, sql, left_view_name,
  right_view_name, ...}]}}`. 404 `"Topic <name> not found"`.
- **`GET /api/v1/models/{modelId}/view`** — flattened & lossy: `{success, views: [{name, label,
  description, hidden, fields: [{name, type: "dimension"|"measure"|"filter"}]}]}`. The `type`
  values are **lowercase** — `VIEW_FIELD_TYPE` is `{DIMENSION: 'dimension', FILTER: 'filter',
  MEASURE: 'measure'}` (`packages/bi-app/app/types/api/models/view-field-type.ts`), so the
  uppercase spelling is the TS constant's name, not what goes on the wire. Field **names and
  kinds only, no data types** — the loader calls `getComposedModel` with
  `excludeFieldProps: ['expr']` and maps `dimensions`/`measures`/`filter_only_fields` down to
  `{name, type}` (`packages/bi-app/app/routes/api.unstable.model.$modelId.view.ts`, re-exported
  at `api.v1.models.$modelId.view.ts`). Scope: **every view of the composed model** with
  `ignored` filtered out — `hidden` views are included, and views no topic reaches are too, so
  this is a *superset* of what the topic-detail walk returns. Accepts `?branch_id=`.
  There is **no per-view detail GET** — `api.v1.models.$modelId.view.$viewName.ts` exports only
  an `action`, no `loader`; full field metadata comes from the topic-detail endpoint only.
- **`GET /api/v1/documents/{identifier}/queries`** — `{queries: [{id, name, query, url}]}`;
  the `query` blob is a stored query (verify/inject `modelId` before running). 404 when the
  document has no dashboard.
- **`POST /api/v1/ai/generate-query`** — `{modelId, prompt, currentTopicName?, runQuery?: bool
  (default true!), userId?}` → `{query, topic, baseView, error}`. Omniframes always sends
  `runQuery: false` and executes the returned query through its own pipeline. 400 when no query
  could be generated; 402 on AI-credit shutoff. Requires the `query-api` flag only when
  `runQuery` isn't false.

Field object (topic detail & `summary.fields`): `field_name`, `fully_qualified_name`,
`view_name`, `data_type` (§2.3 enum), `is_dimension`, `label`, `aggregate_type`, `date_type`,
`format`, `sql` (redacted without VIEW_SQL), `hidden`, `filter_only_field`.

---

## 5. Public-docs discrepancies

| Topic | Public docs say | Server actually does |
|---|---|---|
| Response format | single JSON object | `text/ndjson` stream |
| `cache` enum | `disabled/normal/refresh/refresh_all` | `Standard/SkipRequery/SkipCache/SkipCacheAndRebuildExtracts` |
| Default limit | 500 (schema description) / 1000 (openapi) | 1000 (`DEFAULT_ROW_LIMIT`) |
| Max limit | 75000 | >50000 = high-limit override mode; no 75000 cap — live 2026-08-14: `planOnly` with `limit: 75001` accepted and echoed `limit: null` |
| Invalid token | 401 | 403 |
| Timeout | 408 | NDJSON path: in-band footer, always 200 (408 only in `resultType` mode) |
| `/query/wait` params | `job_ids` JSON array | `jobIds` comma-separated preferred (legacy accepts both) |
| `branchId`, `timezone`, `workbookUrl` | absent | supported |

---

## 6. LIVE-VALIDATE register

Run `scripts/live_smoke.py` against a real org (needs `OMNI_BASE_URL`, `OMNI_API_KEY`) to close.

**Closed 2026-08-14** (omni.demo.exploreomni.dev, Postgres connection; facts folded into the
body sections above):

1. `staticQueryReferences`-as-table in `userEditedSQL` — **REFUTED**; the real composition
   mechanism is parsed OmniSQL `${…}` refs (§3.5). Consequence: the tier-2 envelope Omniframes
   0.1 emits (`rewriteSql: false` + `FROM ref_1`) **fails on a live org** with
   `relation "ref_1" does not exist`, and FakeOmniAPI's temp-view materialization of refKeys
   models a mechanism the server does not have. Tier 2 (and the splitter shapes that prefer it,
   e.g. mixed aggregation) needs redesign onto OmniSQL before live use.
2. `timezone` — param exists; 400 unless the connection enables user-specific timezones (§2).
3. Max limit — no 75000 cap; `limit: 75001` accepted, echoed as `null` (§5 table).
4. Measure-keyed `filters` → HAVING — confirmed in live `display_sql` (§3.1).
10. `workbookUrl` header — `X-Omni-Workbook-Url` spelling correct; live URL format is a share
    link `https://<host>/e/<short-id>/<n>` (the fake's `/w/fake/<job-id>` pins plumbing only).
    Emission on the in-band-timeout path is still unobserved.

**Still open:**

5. `generate-query` response variance (topic vs baseView, error shapes).
6. High-limit mode (`limit > 50000`) result streaming behavior.
7. WAF 429 shape under the shared 60/min bucket: whether it reliably sends `Retry-After` (and
   whether that value is numeric seconds or an HTTP date), and whether it reliably accompanies
   the response with `X-Omni-Waf-Action: block`.
8. Wait-loop timing under a genuinely cold warehouse (multi-minute jobs).
9. **Dialect portability of `read.sql`.** A verbatim raw-SQL job (`rewriteSql: false`) reaches
   the warehouse untouched, so its dialect is the user's own problem and omniframes has nothing
   to confirm. Tier 2's half of this item closed 2026-08-14: an OmniSQL statement is parsed and
   **re-rendered per warehouse** by the server (§3.6), quoting, `LIMIT`/`OFFSET`, `NULLS LAST`
   and the `LIKE … ESCAPE` character included, so the dialect omniframes emits in is not
   observable downstream. The `sql_dialect` builder knob and the backslash refusal that existed
   for it were removed with the v1 mechanism (docs/SQLTIER.md §6).
10. `workbookUrl` on the in-band-timeout path (footer with `remaining_job_ids`) — the header's
    normal-path behavior closed 2026-08-14 (see above); whether it is emitted when the run times
    out in-band is the one residue.
11. **OmniSQL path residue** (§3.6, probed on one Postgres org only): exact semantics of the
    same-column predicate merge (which side wins, when); whether `GROUP BY` over all columns is
    a safe dedup substitute for the stripped `DISTINCT`; the scoping rule for multi-view
    expressions beyond first-ref-wins (only one shape tested); topic-vs-view precedence for
    `FROM ${name}` when a topic and an unrelated view share a name; which permission gates the
    path (`QUERY_SQL` vs `QUERY_TOPICS`) and behavior on topic-locked orgs; re-verification on
    a non-Postgres warehouse.
12. **Grain `__raw` collapse, live re-verification** (§2.7): the client now keeps the `__raw`
    values under the plain name and drops the formatted column, and FakeOmniAPI emits the pair
    (docs/SQLTIER.md §5). What is still open is the check against a real org: that the pair
    arrives in the shape assumed here for a grain the model formats, on both the semantic and
    the OmniSQL path, and that `summary.fields` collapses the same way.
