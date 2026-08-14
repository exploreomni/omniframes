# Omni Query API — contract notes

**This document is the source of truth for everything Omniframes puts on the wire.** It was
produced by reading the Omni monorepo server source (checkout `~/omni/omni`, commit
`86569cd50c6197836ed66365ccc2258b213ea2de`, 2026-08-14), NOT the published OpenAPI spec — the
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
  Response: `{user: {id, membershipId}, keyScope: "user"|"organization", orgRole,
  rolesByModel: {<modelId>: {roleName, baseRole, connectionId, permissions: [...]}}}` with
  `rolesByModelTruncated: true` when >200 models. Relevant permissions: `QUERY_TOPICS` (topic
  queries), `QUERY_FULL_MODEL` (bare-view queries), `QUERY_SQL`, `VIEW_SQL` (unredacted SQL/errors).
- **`query-api` feature flag off** → 403 `{"detail": "Feature not enabled", "status": 403}` on
  query endpoints. Missing `QUERY_TOPICS` → 403 `{"detail": "Permission denied", "status": 403}`.
- **Rate limiting** is at the AWS WAF: 60 req/min keyed on the exact `Authorization` header value
  (elevated tiers exist per-org). Blocked → **429** with header `X-Omni-Waf-Action: block`.
  All clients sharing a key share one bucket — back off on 429, bounded retries.

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
  "timezone": "America/Los_Angeles", // LIVE-VALIDATE: requires org+connection settings, else 400
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

**Measure filters → HAVING (source-pinned).** A `filters` entry keyed by a MEASURE field name
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

Set `userEditedSQL: "<sql>"` **and** `rewriteSql: false` (else the SQL is IGNORED and the query
runs as a model job). `sqlSortsEnabled: true` makes the server apply `sorts` / `calculations` /
`column_totals` on top of the SQL result; when false those are stripped. The connection is
resolved server-side from `modelId`. `resultType`/NDJSON behavior is identical.

### 3.5 `staticQueryReferences`

`Record<refKey, <full query object> & {"model_id": "<uuid>"}>` (note the extra **snake_case**
`model_id` inside each referenced query). The only way an external client passes query
references; they become `query_references` on the job (single job per call — references fold
into one plan). Consumed by:
1. `type: "query"` filters (`query_id` = refKey);
2. XLOOKUP-family calc operators (first operand = refKey string literal) — not used in 0.1;
3. **Raw SQL** — the server SQL parser recognizes references used in the SQL text and emits
   `used_query_references`. `LIVE-VALIDATE`: the exact syntax for referencing a refKey as a
   table in `userEditedSQL` (expected: use the refKey as an identifier).

Constraint: rejected with `workbookUrl: true`.

---

## 4. Catalog endpoints

- **`GET /api/v1/models`** — cursor-paginated: `?pageSize=` (1..100, default 20), `?cursor=`
  (opaque — echo back exactly), `?modelKind=SHARED|...`, `?name=`, `?sortField/sortDirection`.
  Response `{pageInfo: {hasNextPage, nextCursor, pageSize, totalRecords}, records: [{id, name,
  modelKind, connectionId, baseModelId, createdAt, updatedAt, deletedAt}]}`. The **strict**
  param schema 400s on unknown query params.
- **`GET /api/v1/models/{modelId}/topic`** — list: `{success, topics: [{name, base_view_name,
  label, description, group_label, hidden}]}`. Optional `?branch_id=`.
- **`GET /api/v1/models/{modelId}/topic/{topicName}`** — detail; THE field-metadata endpoint:
  `{success, topic: {..., views: [{name, label, dimensions: Field[], measures: Field[],
  filter_only_fields: Field[]}], relationships: [{join_type, sql, left_view_name,
  right_view_name, ...}]}}`. 404 `"Topic <name> not found"`.
- **`GET /api/v1/models/{modelId}/view`** — flattened & lossy (`{views: [{name, label, fields:
  [{name, type: DIMENSION|MEASURE|FILTER}]}]}`). There is **no per-view detail GET** — full field
  metadata comes from the topic-detail endpoint only.
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
| Max limit | 75000 | >50000 = high-limit override mode; no hard cap seen at 75000 in source (`LIVE-VALIDATE`) |
| Invalid token | 401 | 403 |
| Timeout | 408 | NDJSON path: in-band footer, always 200 (408 only in `resultType` mode) |
| `/query/wait` params | `job_ids` JSON array | `jobIds` comma-separated preferred (legacy accepts both) |
| `branchId`, `timezone`, `workbookUrl` | absent | supported |

---

## 6. LIVE-VALIDATE register

Run `scripts/live_smoke.py` against a real org (needs `OMNI_BASE_URL`, `OMNI_API_KEY`) to close:

1. `staticQueryReferences` referenced from `userEditedSQL` — exact table-identifier syntax and
   single-job ergonomics. **Offline half closed** (M5): tier 2 emits the refKey as a bare table
   identifier (`FROM ref_1`) with the reference's dotted field names quoted as columns, and
   FakeOmniAPI materializes each reference as a temp view named exactly its key — so client and
   fake now share one assumption instead of the fake holding it alone. The live probe in
   `scripts/live_smoke.py` still has to confirm it against a real warehouse.
2. `timezone` request field — availability without org/connection settings.
3. Effective max-limit behavior above 50000 / the documented 75000 cap.
4. Measure-keyed entries in `filters` → HAVING: **source-pinned as real HAVING** (see §3.1);
   live run is confirmation-only (check `display_sql` contains HAVING with the aggregate).
5. `generate-query` response variance (topic vs baseView, error shapes).
6. High-limit mode (`limit > 50000`) result streaming behavior.
7. WAF 429 shape under the shared 60/min bucket.
8. Wait-loop timing under a genuinely cold warehouse (multi-minute jobs).
9. **Tier-2 SQL across real warehouse dialects.** Omniframes generates the outer statement with
   sqlglot's default (ANSI-ish) dialect: double-quoted identifiers carrying dots and brackets
   (`"order_items.created_at[month]"`), `LIMIT`/`OFFSET`, `ORDER BY … NULLS LAST`,
   `LIKE … ESCAPE '!'`, `LOWER()` on both sides for case-insensitive matching, and
   `CAST('…' AS DATE/TIMESTAMP)` literals. Each of those is standard and each is somewhere
   non-portable in practice (`LIMIT` on SQL Server, `NULLS LAST` on MySQL, quoting on BigQuery).
   Confirm per connection type; `SessionBuilder.sql_dialect(name)` is the escape hatch when a
   warehouse disagrees. Two things are **not** left to the live probe, because they are safety
   rather than syntax: the LIKE escape character is `!`, never `\` (`ESCAPE '\'` does not
   tokenize on Snowflake/BigQuery/Redshift/Spark/MySQL, where a backslash escapes inside a string
   literal), and a filter value containing a backslash is **refused** at tier 2 unless
   `sql_dialect` names the warehouse — the default dialect escapes only `'`, so its rendering of
   such a value is a different string (or different SQL) on half the warehouses in that list.
10. **`workbookUrl` response-header shape.** With `workbookUrl: true` the server is expected to
    return the URL in the `X-Omni-Workbook-Url` response header alongside the normal NDJSON
    stream. Unconfirmed: the exact header spelling/casing, the URL format, and whether it is
    emitted when the run times out in-band (footer with `remaining_job_ids`) rather than
    completing. `df.omni_url()` reads the header and raises when it is absent; the fake mints
    `https://<host>/w/fake/<job-id>` so only the plumbing is pinned offline, never the format.
