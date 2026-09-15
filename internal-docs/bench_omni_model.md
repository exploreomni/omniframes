# The bench Omni model (`bench_ecommerce`)

The governed model that sits over the bench dataset. **This is the contract between the offline
suite and the live probe (`scripts/live_smoke.py`):** `tests/fakes/` serves exactly the model described here, so an expectation
written against the FakeOmniAPI must hold verbatim against the live org. If the live model has to
deviate, change `tests/fakes/bench_model.py` and this document in the same commit.

Companion docs: [BENCH_DATASET.md](BENCH_DATASET.md) (the data), [CONTRACT_NOTES.md](../docs/CONTRACT_NOTES.md)
(the wire), [DESIGN.md](../docs/DESIGN.md) §5 (the test lanes).

---

## 1. Warehouse load

Load the three checked-in tables into one schema on the connection the model will use. The
parquet files are the source of truth; the CSVs are a fallback for warehouses without parquet
ingest.

```
tests/data/bench/users.parquet         500 rows
tests/data/bench/products.parquet      200 rows
tests/data/bench/order_items.parquet 10 000 rows
```

Suggested target: schema `OMNIFRAMES_BENCH`, table names `users`, `products`, `order_items`
(unchanged — the fake's view names are the table names). Larger loads:
`uv run python tools/bench/generate.py --scale 25 --out /tmp/bench25` scales the fact table only,
so join selectivity — and therefore every ratio a test asserts — stays put.

**The fake's warehouse schema.** Offline the three tables are registered in DuckDB under their
**bare** names — `users`, `products`, `order_items` — with *no* schema qualifier, so a raw-SQL
job (§6) writes `FROM order_items oi LEFT JOIN users u ON u.id = oi.user_id`. Live, the same SQL
has to name the connection's schema (`OMNIFRAMES_BENCH.order_items`). Semantic queries are
unaffected — they never mention a table — so this is the one place a tier-2 SQL string cannot be
byte-identical across the two lanes, and the caller, not the fake, owns the qualifier.

Column types must survive the load:

| column | type | why it matters |
|---|---|---|
| `users.created_at`, `order_items.created_at` | `timestamp` (UTC) | grain + date-filter tests |
| `products.introduced_on` | `date` (not timestamp) | date-typed filter literals |
| `users.lifetime_value`, `products.cost`/`price`, `order_items.sale_price`/`discount` | `decimal(p,2)` | decimal128 on the Arrow wire |
| `order_items.notes` | large text | multi-KB values, embedded newlines/quotes |
| `users.state`, `users.age`, `users.is_business`, `products.category`, `order_items.discount`/`returned`/`notes` | nullable | NULL group keys and three-valued logic |
| `users.signup_source` | non-null text containing `''` | empty string must stay distinct from NULL |

`order_items.user_id` contains ~1 % orphans (ids ≥ 900 000 with no `users` row) **by design** —
they are what makes the LEFT joins observable. Do not add a foreign key that would reject them.

---

## 2. Model

```yaml
# model.yaml
name: bench_ecommerce
connection: <the connection holding OMNIFRAMES_BENCH>
schema: OMNIFRAMES_BENCH

relationships:
  - join_from_view: order_items
    join_to_view: users
    join_type: always_left
    relationship_type: many_to_one
    on_sql: ${order_items.user_id} = ${users.id}

  - join_from_view: order_items
    join_to_view: products
    join_type: always_left
    relationship_type: many_to_one
    on_sql: ${order_items.product_id} = ${products.id}
```

Both joins fan **in from the fact table** and are LEFT joins. That is load-bearing: the NULL
`users.state` group in `known_answers.json` (`revenue_by_state`, 590 order items) is the union of
genuinely NULL states and the orphan `user_id`s. An INNER join silently drops those rows and every
row-count expectation shifts.

---

## 3. Topic

One topic, `order_items`, base view `order_items`, both joins reachable.

```yaml
# topics/order_items.topic
base_view: order_items
label: Order Items
group_label: Ecommerce
description: Order items joined to the buying user and the purchased product.
hidden: false
joins:
  users: {}
  products: {}
```

The API surfaces this at:

- `GET /api/v1/models/{modelId}/topic` → `{success, topics: [{name: "order_items", base_view_name:
  "order_items", label, description, group_label, hidden}]}`
- `GET /api/v1/models/{modelId}/topic/order_items` → `{success, topic: {…, views: [order_items,
  users, products], relationships: [→users, →products]}}` — the only source of full field metadata
  (CONTRACT_NOTES §4).

---

## 4. Views

Every base column is a dimension; timestamp dimensions carry the standard grain set. `data_type`
below is the value the API reports in `summary.fields[*].data_type` and in the topic payload.

### `views/order_items.view`

```yaml
schema: OMNIFRAMES_BENCH
table_name: order_items
dimensions:
  id:          { sql: ${TABLE}.id,          type: number, primary_key: true }
  order_id:    { sql: ${TABLE}.order_id,    type: number }
  user_id:     { sql: ${TABLE}.user_id,     type: number }
  product_id:  { sql: ${TABLE}.product_id,  type: number }
  created_at:  { sql: ${TABLE}.created_at,  type: timestamp }
  status:      { sql: ${TABLE}.status,      type: string }
  quantity:    { sql: ${TABLE}.quantity,    type: number }
  sale_price:  { sql: ${TABLE}.sale_price,  type: number }
  discount:    { sql: ${TABLE}.discount,    type: number }
  returned:    { sql: ${TABLE}.returned,    type: boolean }
  notes:       { sql: ${TABLE}.notes,       type: string }
measures:
  total_sale_price:   { sql: ${sale_price}, aggregate_type: sum }
  count:              { aggregate_type: count }
  total_quantity:     { sql: ${quantity},   aggregate_type: sum }
  average_sale_price: { sql: ${sale_price}, aggregate_type: average }
```

| field | `data_type` | notes |
|---|---|---|
| `order_items.id` / `order_id` / `user_id` / `product_id` / `quantity` | `NUMBER` | integers |
| `order_items.sale_price` / `discount` | `NUMBER` | decimal(12,2) → decimal128 on the wire |
| `order_items.created_at` | `TIMESTAMP` | `date_type: timestamp`; filtered with `type: date` |
| `order_items.status` / `notes` | `STRING` | `notes` is nullable, multi-KB, newline-bearing |
| `order_items.returned` | `BOOLEAN` | nullable → three-valued logic |

### `views/users.view`

```yaml
schema: OMNIFRAMES_BENCH
table_name: users
dimensions:
  id:             { sql: ${TABLE}.id,             type: number, primary_key: true }
  name:           { sql: ${TABLE}.name,           type: string }
  email:          { sql: ${TABLE}.email,          type: string }
  state:          { sql: ${TABLE}.state,          type: string }
  country:        { sql: ${TABLE}.country,        type: string }
  created_at:     { sql: ${TABLE}.created_at,     type: timestamp }
  age:            { sql: ${TABLE}.age,            type: number }
  is_business:    { sql: ${TABLE}.is_business,    type: boolean }
  signup_source:  { sql: ${TABLE}.signup_source,  type: string }
  lifetime_value: { sql: ${TABLE}.lifetime_value, type: number }
measures:
  count: { sql: ${id}, aggregate_type: count_distinct }
```

`users.count` is `COUNT(DISTINCT users.id)` — **not** `COUNT(*)`. Through the topic's LEFT join
that makes orphan `user_id`s contribute zero distinct buyers, which is exactly what the
`distinct_buyers` column of `known_answers.json` encodes (497, against 535 distinct raw
`order_items.user_id` values).

### `views/products.view`

```yaml
schema: OMNIFRAMES_BENCH
table_name: products
dimensions:
  id:            { sql: ${TABLE}.id,            type: number, primary_key: true }
  name:          { sql: ${TABLE}.name,          type: string }
  category:      { sql: ${TABLE}.category,      type: string }
  brand:         { sql: ${TABLE}.brand,         type: string }
  cost:          { sql: ${TABLE}.cost,          type: number }
  price:         { sql: ${TABLE}.price,         type: number }
  introduced_on: { sql: ${TABLE}.introduced_on, type: date }
measures:
  count: { sql: ${id}, aggregate_type: count_distinct }
```

`products.introduced_on` reports `data_type: TIMESTAMP` with `date_type: date` — the Omni type
enum has no separate DATE member (CONTRACT_NOTES §2.3). Filter literals for it are plain
`YYYY-MM-DD`.

### 4.1 The grain set on the date dimensions

`users.created_at`, `order_items.created_at` (`date_type: timestamp`) and
`products.introduced_on` (`date_type: date`) accept `field[grain]` suffixes (CONTRACT_NOTES
§3.2). Grain names are matched case-insensitively; the `summary.fields` key and the result column
name are the field name **exactly as requested**, bracket and all (`order_items.created_at[MONTH]`
comes back as `order_items.created_at[MONTH]`) — with the one exception §4.2 describes.

| grain | `data_type` | on a `date` column | rendered as |
|---|---|---|---|
| `year`, `quarter`, `month`, `week`, `date` | `TIMESTAMP` | yes — result stays date-shaped | `date_trunc` |
| `hour`, `minute`, `second` | `TIMESTAMP` | no → `missing_fields` | `date_trunc` |
| `day_of_week_num`, `day_of_month`, `day_of_year`, `month_num`, `quarter_of_year`, `week_of_year` | `NUMBER` | yes | `EXTRACT` |
| `hour_of_day` | `NUMBER` | no → `missing_fields` | `EXTRACT` |
| `month_name`, `day_of_week_name` | `STRING` | yes | `monthname` / `dayname` |

Truncating grains keep the base `date_type`; numeric and name grains report `date_type: null`.
An unknown grain, or one that does not fit the field (`products.introduced_on[hour_of_day]` — a
date has no time of day), lands in `summary.missing_fields` and the field is dropped: **not** a
hard error (§3.2). The remaining grains of the §3.2 list (`millisecond`, `day_of_quarter`,
`fiscal_year`, `fiscal_quarter`, `epoch`, `time_of_day`, and the duration grains) are not
implemented offline and therefore read as missing fields — see the backlog below.

### 4.2 `month` is FORMATTED — the `__raw` pair (CONTRACT_NOTES §2.7)

The bench model puts a display format on exactly one grain:

```yaml
# views/order_items.view, views/users.view — on every timestamp/date dimension
created_at:
  timeframes:
    month: { format: 'YYYY-MM' }
```

A **formatted** grain does not come back as one column. Live — on the semantic path and the
OmniSQL path alike — selecting `order_items.created_at[month]` returns a PAIR:

| result column | value | position |
|---|---|---|
| `order_items.created_at[month]__raw` | the `DATE_TRUNC` timestamp, `data_type: TIMESTAMP` | the item's own select position |
| `order_items.created_at[month]` | the formatted string (`"2026-01"`), `data_type: STRING`, `format: YYYY-MM` | appended **after every other column** |

`__raw` deliberately does not match the reserved-column regexes of §2.7, so a client has to
reconcile the pair rather than strip it — omniframes keeps the `__raw` VALUES under the plain
name and drops the formatted column (docs/SQLTIER.md §5, "`__raw` wins"), because a month grain
is a timestamp whether or not the model formats it.

Consequences inside the fake, all of them mirroring the live shape:

- a `sorts[]` entry naming the grain orders by the `__raw` half — sorting `'2026-01'` strings is
  only accidentally chronological, and stops being so the moment a format changes;
- both halves are dimensions, so both join the GROUP BY; the formatted one is a function of the
  raw one, so the group set is unchanged;
- `summary.fields` carries both entries, in result order.

**`month` is the only formatted grain on purpose.** Every other grain stays raw-only, so the
one-column and the two-column shapes are both exercised offline. If the live model formats a
different set, `GRAIN_FORMATS` in `tests/fakes/bench_model.py` and this subsection move together.

**Grain-filter rule** (CONTRACT_NOTES §3.1), which the fake enforces by construction: a filter
keyed on the **bare** field name compiles against the underlying column even when the projection
is grained (so a `date BETWEEN` narrows the window under a `created_at[month]` group-by), while a
filter keyed on the **bracketed** numeric-grain name compiles against that grain's `EXTRACT`.

---

## 5. The measures the lanes assert on

Straight from [BENCH_DATASET.md](BENCH_DATASET.md) § "The governed model over it":

| field | definition |
|---|---|
| `order_items.total_sale_price` | `SUM(sale_price)` |
| `order_items.count` | `COUNT(*)` |
| `order_items.total_quantity` | `SUM(quantity)` |
| `order_items.average_sale_price` | `AVG(sale_price)` |
| `users.count` | `COUNT(DISTINCT users.id)` |
| `products.count` | `COUNT(DISTINCT products.id)` |

Ground truth for each lives in `tests/data/bench/known_answers.json`; tests assert against that
file, never against a recomputation inside the test.

**Selecting dimensions and measures together IS the group-by** (Omni semantics, DESIGN.md §2):
every dimension field in `fields` becomes a group key, and measures alone produce the single
aggregate row. Measures are sortable like any other field (`sorts[].column_name` is the exact
field name), and a measure-keyed entry in `filters` compiles to a genuine `HAVING` (§5.2).

### 5.1 Column totals

`query.column_totals` (CONTRACT_NOTES §2.7 / §3) is executed: `{"::total::": {"type":
"aggregation"}}` totals every measure in the query, and a measure-keyed entry totals just that
column. The result gains the reserved indicator column `$omni_column_total_indicator` — `null` on
data rows, `::total::` on the grand-total row, `column_total` when the totals were requested per
measure column. Dimension columns on a totals row are NULL, and each measure is **re-aggregated**
at the total grain rather than summed from the group values (so `users.count` totals to 497
distinct buyers, not the sum of the per-state counts).

`__omni_summ` sidecars are deliberately **not** emitted here: they are a raw-SQL
(`userEditedSQL`) artifact, and a semantic job that invented them would train the client on a
shape the live org never sends for this query kind. The sidecar shape lives in §6.5, where it
belongs.

### 5.2 Measure filters compile to HAVING

A `filters` entry keyed by a **governed measure** is translated against that measure's *aggregate*
expression and lands **post-`GROUP BY`, in `HAVING`** — source-pinned server behavior
(CONTRACT_NOTES §3.1, "Measure filters → HAVING"; LIVE-VALIDATE #4 is now confirmation-only). A
dimension-keyed entry stays WHERE-side, pre-aggregation, so a query carrying both narrows the rows
first and then the groups those rows produced:

```sql
SELECT "users"."state" AS "users.state",
       SUM("order_items"."sale_price") AS "order_items.total_sale_price"
FROM "order_items"
LEFT JOIN "users" ON "users"."id" = "order_items"."user_id"
WHERE "order_items"."status" = 'complete'
GROUP BY 1
HAVING SUM("order_items"."sale_price") > 50000
```

Three consequences the fake reproduces:

- **A filtered measure that is not selected is projected away.** It is still force-added to the
  aggregate so the `HAVING` can see it, but it never becomes a result column and never appears in
  `summary.fields`.
- **A measure filter forces the group-by.** `fields` holding only dimensions plus a measure filter
  is still an aggregate query: `GROUP BY` those dimensions, then `HAVING`. (Without the filter the
  same `fields` would return raw rows.)
- **One entry per measure.** `filters` is keyed by field name, so several conditions on the same
  measure arrive as a `composite` inside that single entry.

Implemented arms on the HAVING side: `type: "number"` (`LESS_THAN`, `GREATER_THAN`, `EQUALS`,
`BETWEEN`, with `is_inclusive` and `is_negative`; values are strings) and `type: "composite"`
(`AND`/`OR`, recursive, `is_negative`). Every other arm — a `string`, `date`, `boolean`, `null`,
`query` or `user_attribute` filter aimed at a measure, or a number `kind` outside that list — is a
`PLAN` job error naming what it was handed, rather than a predicate the fake cannot vouch for.

`column_totals` **combined with** a measure filter is likewise a `PLAN` error: what a total over a
`HAVING`-restricted group set aggregates is not pinned by the contract, and guessing would hand
back a plausible wrong number.

### 5.3 Offline-only rendering choices

Two places where the fake pins something the wire leaves open. Both are assumptions to confirm
when `scripts/live_smoke.py` runs (LIVE-VALIDATE-adjacent — they are not in the CONTRACT_NOTES §6 register
because they are the fake's choices, not claims about the server):

- **`order_items.average_sale_price` is materialized as `DECIMAL(38,10)`.** DuckDB's `AVG` over a
  `decimal(12,2)` is a `DOUBLE`, which cannot be compared exactly; `known_answers.json` casts the
  average the same way. A live warehouse may hand back a float or a different scale, so an
  average assertion is the first thing to check if the live twin of a test drifts.
- **Column totals aggregate the post-filter, PRE-limit row set.** Totals are computed by a second
  query sharing the data query's `FROM`/`JOIN`/`WHERE` with no `GROUP BY`, `ORDER BY` or
  `LIMIT`/`OFFSET`, so a `limit: 3` query still totals every matching row. This is the reading
  that makes a totals row meaningful; it is not spelled out in the server source.

`query.default_group_by` is accepted and ignored — omniframes always sends `true`, and the fake
always groups by every dimension field. A `false` value has no validated meaning offline.

---

## 6. The two SQL paths: verbatim SQL and parsed OmniSQL

`query.userEditedSQL` selects one of **two** jobs, and the "do not rewrite" marker beside it is
the whole difference (CONTRACT_NOTES §3.4 vs §3.6):

- **with** a marker — a **verbatim SQL job**: the text goes to the warehouse untouched. This is
  `read.sql`, the user's own SQL. §6.1–§6.7 below.
- **without** one — a **parsed-OmniSQL job**: the server parses the text, binds every `${…}`
  reference against the model, and plans a governed model job. This is tier 2. §6.8.

Either way the fake executes against DuckDB over the same bench tables the semantic path
queries, so `known_answers.json` is the oracle for a SQL job exactly as it is for a model job,
and the wire framing is unchanged: same NDJSON lines, same base64 Arrow `result`, same `summary`
keys.

### 6.1 Which path a statement takes

| `query` keys | what runs |
|---|---|
| `userEditedSQL` + `rewriteSql: false` | the SQL, verbatim |
| `userEditedSQL` + `parsed: false` | the SQL, verbatim |
| `userEditedSQL` + `dbtMode: true` | the SQL, verbatim |
| `userEditedSQL` **alone** | the text as **OmniSQL** — parsed, bound, planned as a model job (§6.8) |

The last row is the one that matters. The SQL is neither run as written nor ignored: `${…}`
references resolve against the model, the joins come from the topic, and what reaches the
warehouse is governed SQL the client never wrote. The fake reproduces the split literally
(`tests.fakes.sqljobs.is_raw_sql_job` / `tests.fakes.omnisql.is_omnisql_job`), so a statement
that forgets — or wrongly adds — the marker fails offline the way it would fail live.

### 6.2 `summary.fields` on a SQL job

Synthesized from the Arrow result schema, keyed by the result column name exactly as the SQL
spelled it, and **every column is a `is_dimension: true` field with `aggregate_type: null`** —
the view Omni wraps around `userEditedSQL` has an empty measures map (CONTRACT_NOTES §3.1).
`data_type` comes from the Arrow type via the §2.3 enum, `missing_fields` is always `[]` (a SQL
job has no field names to miss), and `display_sql` is the SQL that actually ran — which means a
`sqlSortsEnabled` sort wrapper is visible in it. `omni_sql` is empty: a hand-written SQL job has
no Omni-flavored form. (A parsed-OmniSQL job differs on both counts — §6.8.)

### 6.3 `sqlSortsEnabled` gates sorts **and** column totals

`sqlSortsEnabled: true` applies the envelope's `sorts` on top of the SQL result (the fake wraps
the statement in `SELECT * FROM (…) ORDER BY …`) and executes `column_totals`. Falsy — including
absent — **strips both, silently**: the server forces them to `[]` (CONTRACT_NOTES §3.4), so a
totals request that arrives without the flag produces a result with no sidecars and no indicator
column, and no complaint. A sort naming a column the SQL does not produce *is* refused (`PLAN`),
because on a SQL job `sorts[].column_name` is a result column, not a model field.

### 6.4 Filters on a SQL job

| filter key | what happens |
|---|---|
| a governed **measure** (`order_items.total_sale_price`, …) | **silently skipped** — contract-pinned (§3.1: the SQL wrapper view has an empty measures map) |
| a **dimension** (`users.state`, …) | `PLAN` error |
| an unknown name | `PLAN` error, `No such field "<name>"` |

The measure row is why tier-2 SQL must carry its own `HAVING`: a `filters` entry aimed at a
measure is a no-op that narrows nothing. The dimension row is a deliberate offline gap — live,
Omni splices dimension filters into the SQL through mustache templating, and a fake that
approximated that would hand back rows that look filtered and are not.

### 6.5 `column_totals` → `__omni_summ` sidecars

`column_totals` keyed by **result column** (with `sqlSortsEnabled: true`) produces the §2.7
sidecar shape — the same framing `tests/wire/fixtures/totals_with_sidecars.ndjson` models, which
is what the client's normalizer was written against:

- each totaled column gains a `<column-lowercased>__omni_summ` sidecar, immediately after it,
  in the base column's own Arrow type;
- the sidecar is NULL on every data row;
- **one** row is appended, with every base column NULL — the value lives in the sidecar, which
  carries `SUM(<column>)` over the whole SQL result;
- `$omni_column_total_indicator` is NULL on data rows and `column_total` on the appended row.

Only numeric columns can be totaled; anything else is a `PLAN` error. `"::total::"` is a `PLAN`
error too: it means "total every measure", and a SQL job has none.

This is the mirror image of §5.1 — a *semantic* `column_totals` job puts the values in the base
columns and emits no sidecars at all. Both shapes exist on the wire; which one you get is
decided by the job kind, not by the request.

### 6.6 `staticQueryReferences` (CONTRACT_NOTES §3.5)

Raw SQL ignores `staticQueryReferences`. Reference keys do not become tables: bare and
quoted keys in `userEditedSQL` reach the warehouse unchanged and fail if no such table exists.
This matches the live-confirmed behavior in CONTRACT_NOTES §3.5; tier 2 composes through parsed
OmniSQL (§6.8).

- `workbookUrl: true` alongside a non-empty `staticQueryReferences` is a **400** at the envelope
  (CONTRACT_NOTES §2.1), not a job error.
- References on semantic and parsed-OmniSQL jobs are refused (`PLAN`). Their real consumers —
  `type: "query"` filter arms and XLOOKUP calc operators — remain outside the fake's scope.

### 6.7 Offline-only choices on the SQL path

Alongside §5.3, two things the wire leaves open that the fake pins:

- **`display_sql` is the executed SQL**, sort wrapper included, rather than `userEditedSQL`
  verbatim. That makes `sqlSortsEnabled` observable; a live org may render it differently.
- **The envelope `limit`/`offset` are inert on a SQL job.** The SQL owns its own row count, and
  the contract does not say the server re-limits it. Sidecar totals therefore aggregate the
  entire SQL result.

### 6.8 The parsed-OmniSQL path (CONTRACT_NOTES §3.6) — tier 2

`userEditedSQL` with **`rewriteSql` absent** (not `false`). The statement IS the plan: there is
no reference core, no second query object, and the query object's `fields`/`filters`/`sorts`
have nothing to say. `tests/fakes/omnisql.py` is the resolver; docs/SQLTIER.md §7 is its spec.

```sql
SELECT ${users.state}, ${order_items.total_sale_price},
       COUNT(DISTINCT ${users.id}) AS of_expr_1
FROM ${order_items}
GROUP BY 1
LIMIT 50000
```

**Substitution.** `${…}` occurrences outside string literals are bound against the bench model:

| reference | resolves to |
|---|---|
| `${order_items}` in FROM position | the topic's base view, plus the topic's LEFT JOINs **pruned to the views the statement actually references** (a statement inside `order_items` joins nothing) |
| `${view.field}` | the qualified warehouse column |
| `${view.field[grain]}` | the grain's `date_trunc`/`EXTRACT` expression, and the §4.2 `__raw` pair when the grain is formatted |
| `${view.measure}` | the measure's aggregate expression, expanded inline — freely mixable with ad-hoc aggregates, and legal in `HAVING` and in arithmetic |

Quoted text is data: a value that happens to contain `${users.state}` stays that string.

**Result naming — two regimes, exactly as probed live.** This is the part a client cannot guess,
so the fake reproduces both:

- a **bare ref** select item (`${view.field}`, a bare grain ref, `${view.measure}`) surfaces
  under its canonical `view.field` name. **The SQL alias is ignored** — renames are the client's
  job, same as tier 1;
- **every other select item** is an *expression item*: its SQL alias **is** honored and
  duplicates survive, published as `<scope_view>.<alias>`. The live `scope_view` is not
  predictable (first-ref-wins is refuted), so the fake picks the lexicographically greatest
  referenced view — deterministic enough to assert on, **arbitrary on purpose**: a client that
  learned the rule would be learning a lie, and only matching by alias *suffix* survives it.

`summary.fields` binds bare-ref columns back to the model's own metadata (label, `date_type`,
`aggregate_type`) and synthesizes expression columns as dimensions unless they aggregate;
`display_sql` is the bound DuckDB statement, and `omni_sql` is the OmniSQL text as sent.

**The query-object `limit` is inert here.** The statement owns its row count: no `LIMIT` clause
means unlimited, whatever the envelope says (CONTRACT_NOTES §3.6). `LIMIT`/`OFFSET` live in the
text.

**Substitution errors carry the server's own wording**, so the client's tier-2 error mapping
(docs/SQLTIER.md §8) is exercised offline:

| shape | message |
|---|---|
| unknown FROM ref | `Could not substitute Omni SQL: No such view "<name>"` |
| unknown field ref | `Could not substitute Omni SQL: Field "<name>" not found: No such field "<name>"` |

A name outside `view.field[grain]`'s charset never reaches the statement — it is a missing
field, not a splice point.

**Stricter than the server, on purpose.** The server *silently rewrites* the shapes below, which
offline would mean a plausible wrong answer that only breaks against a real org. Each is instead
a loud `PLAN` error whose message starts with `FakeOmniAPI rejects this OmniSQL statement`, so an
emission bug is a red test:

| shape | why it must not be emitted |
|---|---|
| `SELECT DISTINCT` | silently STRIPPED — a pushed-down dedup comes back with duplicates; dedup stays local |
| the same bare ref selected twice | silently DEDUPLICATED, which shifts every later positional `GROUP BY`/`ORDER BY`; dedup at emission |
| a `WHERE` mixing a bare ref and a grain ref of ONE field | predicates MERGED WITH LOSS (the tighter bound disappears); keep grain refs out of `WHERE` |
| `sorts`, `filters`, `column_totals`, `sqlSortsEnabled` or `staticQueryReferences` on the job | the statement is the whole plan; the server has nowhere to put them |

Three more are refusals rather than server-behavior mirrors, and mark the edge of what the fake
models: a CTE (the server accepts and *flattens* them — emit the flat statement), an explicit
`JOIN` in the text (joins come from the topic), and a `FROM ${view}` naming something other than
the topic's root (topic-vs-view precedence is CONTRACT_NOTES §6 item 11, unverified).

---

## 7. Saved queries and generated queries (CONTRACT_NOTES §4)

Two endpoints hand a client a query object it did not write. Both are served over this same
bench model, and every blob they return is executable through `/query/run` unchanged.

**`GET /api/v1/documents/{identifier}/queries`** → `{queries: [{id, name, query, url}]}`. The
fake serves three canned queries (revenue by state, monthly revenue, average sale price by
category) on the document `bench_dashboard`, configurable through
`FakeOmniAPI(documents={...})`. Two distinct 404s, both in the `{"detail", "status"}` envelope:

| identifier | answer |
|---|---|
| `bench_dashboard` | 200, the three saved queries |
| `bench_workbook` (configured with an empty list) | 404 `Document bench_workbook does not have a dashboard` — §4's documented "no dashboard" case |
| anything else | 404 `Document <id> not found` |

**`POST /api/v1/ai/generate-query`** → `{query, topic, baseView, error}`. Deterministic offline:
the prompt is lowercased and matched by substring, first match winning.

| prompt contains | query |
|---|---|
| `monthly revenue` | `order_items.created_at[month]` + `order_items.total_sale_price`, sorted by month |
| `revenue by state` | `users.state` + `order_items.total_sale_price`, sorted by revenue |
| `top products` | `products.name` + `order_items.total_sale_price`, sorted by revenue, `limit: 10` |
| anything else | **400** `No query could be generated for that prompt` (§4) |

`runQuery` must be **`false`**, and absent counts as wrong — the server's default is `true`, so
an omitted key means "execute this outside the client's pipeline". Any other value is a 400
naming that. With `runQuery: false` the endpoint needs no `query-api` flag (§4), so the fake
deliberately does not gate it; `FakeOmniAPI(ai_credits_exhausted=True)` serves the 402.

---

## 8. Permissions and org settings for the live key

`scripts/live_smoke.py` needs an API key whose membership can exercise both the happy path and the
documented 403s (CONTRACT_NOTES §1):

- `query-api` feature flag **enabled** on the org (otherwise every query endpoint 403s
  `Feature not enabled`; `/whoami` keeps working and is the right preflight).
- Model role on `bench_ecommerce` granting `QUERY_TOPICS` (topic queries), `QUERY_FULL_MODEL`
  (bare-view queries via `read.view()`) and `VIEW_SQL` (unredacted `display_sql` and error
  messages).
- Optionally a second, restricted key without `VIEW_SQL` to close the redaction path live — the
  fake models it as `FakeOmniAPI(redact_sql=True)`.

`GET /api/v1/whoami?modelId=<bench model id>` should answer with
`rolesByModel[<modelId>].permissions` containing at least `QUERY_TOPICS`, `QUERY_FULL_MODEL`,
`VIEW_SQL`, and `keyScope` of `organization` (or `user` for a PAT).

---

## 9. Offline ↔ live parity checklist

| what | offline (FakeOmniAPI) | live |
|---|---|---|
| model name | `bench_ecommerce` | must match |
| model id | `3f2b1a0c-9d8e-4c7b-a6f5-000000000001` (`tests.fakes.BENCH_MODEL_ID`) | real UUID; live tests read it from the env / `/models` by name |
| topic | `order_items` | must match, incl. `base_view_name` |
| views | `order_items`, `users`, `products` | must match |
| joins | LEFT, many-to-one, from `order_items` | must match |
| field names | `view.column` for every base column | must match |
| measures | the six above | must match |
| formatted grains | `month` only, `YYYY-MM` (§4.2) | must match — the `__raw` pair follows from it |
| row counts | 500 / 200 / 10 000 | must match at scale 1 |

A live test that fails while its offline twin passes means one of these rows drifted — check this
table before suspecting the client.

### Known gaps in the offline twin

The fake covers the wire behaviors exercised by the test suite (DESIGN.md §5). Unsupported
query shapes are **refused loudly** rather than approximated; some request options are accepted
but have no effect. The table below records both kinds of gap.

**Implemented today**

- dimension `fields`, and `[grain]` suffixes on the date dimensions (§4.1), including the
  `__raw` + formatted PAIR a formatted grain projects on every path (§4.2);
- the six governed measures, with dimension+measure selection as the group-by (§5);
- `filters`: string / number / boolean / null / date (`BETWEEN`, `ON_OR_AFTER`, `BEFORE`) /
  composite AND-OR, including the grain-filter rule in both directions;
- measure-keyed `filters` → `HAVING` (number kinds and composites of them), including the
  force-added-then-projected-away measure and the forced group-by (§5.2);
- `sorts` (dimensions, grains and measures), `limit`, `offset`, `planOnly`;
- `column_totals` with the `$omni_column_total_indicator` row framing (§5.1);
- `missing_fields` for unknown names and unusable grains;
- **verbatim SQL jobs** (§6.1–§6.7): the three no-rewrite markers, `sqlSortsEnabled` over
  `sorts` and `column_totals`, measure-filter skipping, synthesized `summary.fields`, warehouse
  errors as `error_type: "QUERY"`, `planOnly`;
- **parsed-OmniSQL jobs** (§6.8): `${topic}` FROM refs with the topic's join graph pruned to the
  referenced views, `${view.field}` / `${view.field[grain]}` / `${view.measure}` substitution,
  both result-naming regimes, the server's substitution-error texts, and loud rejections of the
  four shapes the server silently rewrites;
- **`__omni_summ` sidecars** on a raw-SQL `column_totals` job (§6.5), in the shape the client's
  normalizer round-trips;
- **`staticQueryReferences`** ignored on raw SQL, with the envelope rejection for
  `workbookUrl: true` preserved (§6.6);
- `GET /documents/{identifier}/queries` and `POST /ai/generate-query` (§7).
- `workbookUrl: true` returns an `X-Omni-Workbook-Url` response header; invalid combinations
  with `planOnly` or query references are rejected.

**Known limitations**

| gap | behavior in the fake | status / verification needed |
|---|---|---|
| dimension-keyed `filters` on a raw-SQL job (Omni's mustache templating) | `PLAN` job error | pinning the templating syntax |
| `"::total::"` and non-numeric columns in a raw-SQL `column_totals` | `PLAN` job error | — (a SQL job has no measures to grand-total) |
| on the OmniSQL path: CTEs, an explicit `JOIN`, a `FROM ${view}` that is not the topic's root, a non-`SELECT` statement | `PLAN` job error, `FakeOmniAPI rejects…` | flattening/precedence pinned (CONTRACT_NOTES §6 item 11) |
| `staticQueryReferences` on semantic or parsed-OmniSQL jobs (`type: "query"` filter arms, XLOOKUP calc operators) | `PLAN` job error | not implemented |
| `type: "user_attribute"` filter arms | `PLAN` job error | user-attribute support |
| `calculations` (including on the `sqlSortsEnabled` path) | 400 | out of scope for 0.1 (DESIGN.md §6) |
| `fill_fields` | 400 | not implemented |
| `pivots` | 400 | wire pivots are out of scope; use pandas through `map_pandas()` for local pivots |
| `row_totals` | 400 | not implemented (`column_totals` is supported) |
| `resultType` / `formatResults` single-document mode | 400 | `session.ask` parity work |
| relative date literals (`"30 days ago"`, `"last quarter"`), and the date kinds beyond `BETWEEN` / `ON_OR_AFTER` / `BEFORE` | `PLAN` job error | not implemented |
| non-number arms of a measure-keyed filter (`string`, `date`, `boolean`, `null` on a measure) | `PLAN` job error naming the arm | not implemented; requires verified measure-filter semantics |
| `column_totals` next to a measure-keyed filter | `PLAN` job error | pinning what a total over a HAVING-restricted group set means |
| grains outside §4.1 (`millisecond`, `day_of_quarter`, `fiscal_*`, `epoch`, `time_of_day`, durations) | `summary.missing_fields` | not implemented |
| cross-field OR via `controls`, period-over-period, `join_via_map`, `column_limit`, `custom_summary_types` | ignored / unmodeled | out of scope for 0.1 |
| `branchId`, `timezone`, `cache` (the enum is validated, the answer is always `cache_type: MISS`) | accepted and inert | branch, timezone, and cache effects are not modeled |
| envelope `limit`/`offset` **on either SQL path** | accepted and inert; the SQL statement controls its row count | matches the SQL execution contract (§6) |
