# SQLTIER — tier 2, the SQL-job tier (M5)

Authoritative design for M5. Read with CONTRACT_NOTES §3.4/§3.5 (the wire), HYBRID.md (the
splitter/local engine this slots into), and docs/bench_omni_model.md §6 (what the fake serves).
Every DECISION is marked with its rationale; implement them as written.

Tier 2 exists to kill tier 3's biggest cost: the **unlimited raw scan** feeding a local
aggregate. A `GROUP BY` that runs in the warehouse moves five orders of magnitude less data.
Everything else it picks up (SQL WHERE for cross-field OR, computed SELECT columns, HAVING on
ad-hoc aggregates, ORDER BY/LIMIT on top) rides the same envelope.

## 1. What upgrades, what doesn't

The splitter tries tiers in order **at every node** (HYBRID's invariant, unchanged):
tier 1 (`try_semantic`) → tier 2 (`try_sql`, new) → local.

Shapes `try_sql` accepts — the outer chain over a **reference core** (§2):

| Shape (logical plan) | SQL rendering |
|---|---|
| `Aggregate(keys=dims/grains, aggs=ad-hoc only)` | `SELECT keys, AGG(...) FROM ref_1 GROUP BY keys` |
| Non-tier-1 `Filter` below the aggregate (cross-field OR, arithmetic, string-pred mixes) | outer `WHERE` (references bare columns of `ref_1`) |
| `Filter` above the aggregate whose predicate references ad-hoc agg **output columns** | `HAVING` with the full aggregate expression substituted (ANSI HAVING cannot see SELECT aliases) |
| `Project`/`WithColumn` of `Arithmetic` over fields (computed columns) | SELECT expressions `AS "name"` |
| `Sort` / `Limit` / `offset` above any of these | `ORDER BY` / `LIMIT` / `OFFSET` in the SQL text |

**Mixed aggregation** (governed measures + ad-hoc in one `agg()`): the governed half stays a
tier-1 remote step exactly as in M3; the **ad-hoc half** (today: raw scan + `LocalAggregate`)
is upgraded to a single tier-2 step when expressible; the **`AlignJoin` stays local**.

> DECISION — AlignJoin does not move into SQL. A NULL-safe join needs
> `IS NOT DISTINCT FROM`, which is not portable across warehouse dialects; the local
> AlignJoin is proven, and each side is already aggregated (small). Only the scan cost
> mattered, and the tier-2 GROUP BY removes it.

Stays tier 3 (or refused) — unchanged from M3/M4:
- UDFs / `map_pandas` and everything above them.
- User-facing `Join`/`Union` (M4 local ops) and everything above them.
- Anything above a `SqlScan` / `SavedQueryScan` (opaque scans — never re-derive SQL over them).
- Relative date literals — including next to `==`/`!=`/`IN`, not only the ordering operators.
- A **string** literal compared against a field carrying a timestamp grain
  (`created_at.grain("month") == "2026-03"`). The grain pins the type, so tier 1 compiles that to
  a *date* filter (CONTRACT_NOTES §3.1); SQL has no way to say it, and rendering it as text would
  both fail on a strict dialect and — for `between()` — silently widen tier 1's half-open upper
  bound. A bare (ungrained) field compared to a string stays tier 2, matching tier 1's own
  string-filter treatment.
- A `HAVING` predicate that also names a column the aggregate consumed
  (`(n > 10) | (users.age == 30)`): the aggregate in one operand does not make the rest of the
  predicate legal after the `GROUP BY`.
- An aggregate whose input is not reference-expressible (§2) — e.g. above a user `.limit()`
  (the limit pins the frontier; a page-then-aggregate is the user's stated question) or above
  local-only work. Falls back to M3 behavior untouched.
- `with_totals()` on any non-tier-1 plan: still `CompileError` (unchanged).

## 2. The reference core and the envelope

A tier-2 job is **one** run-envelope whose `query` is a SQL job plus semantic references:

```jsonc
{
  "query": {
    "modelId": "<model>",
    "userEditedSQL": "SELECT ... FROM ref_1 ...",
    "rewriteSql": false,            // ALWAYS — without it the SQL is silently IGNORED
    "sqlSortsEnabled": false,       // DECISION below
    "staticQueryReferences": { "ref_1": { ...tier-1 query..., "model_id": "<model>" } },
    "fields": [], "table": "", "limit": <explicit>, "version": 9, ...
  }, ...
}
```

- **Reference core** = the subtree the reference query fetches: `Scan → [tier-1-compilable
  Filters] → Project(bare fields)`, **unlimited** (`limit: None` on the wire). Built with the
  EXISTING tier-1 machinery: construct the plan nodes and call `compile_semantic` — never
  hand-assemble a `Query`. Compilable dimension filters are pushed *into* the reference
  (governed, pre-aggregation); non-compilable ones move to the outer `WHERE`, and the reference
  then projects whatever bare fields the outer SQL mentions (reuse the splitter's widening
  helpers `_columns_for`/`split_grain`).
- Fields inside a reference keep their **dotted wire names** (`users.state`,
  `order_items.created_at[month]`); the outer SQL references them as **quoted identifiers**.
- **refKey naming**: `ref_1`, `ref_2`, … in first-use order (deterministic; matches the fake's
  bare-identifier requirement, and golden snapshots stay stable).
- Verify `querymodel.Query.to_wire()` emits the snake_case `model_id` inside each
  `staticQueryReferences` value (the fake 400s without it). If it does not, fix `to_wire()` —
  that is a querymodel bug, in-scope for M5.
- `Query.sql_job(...)` exists and already forces `rewrite_sql=False`; call it with
  `sql_sorts_enabled=False`.

> DECISION — `sqlSortsEnabled: false`, ORDER BY/LIMIT live in the SQL text. One code path
> renders the whole statement; envelope `sorts` stay `[]` so the flag gates nothing we use
> (`column_totals` never rides tier 2 — `with_totals()` × decomposition is already an error).

> DECISION — the wire `limit` is ALSO set explicitly (same value as the SQL `LIMIT`).
> The sql_job shape's limit handling is not contract-pinned; sending both identical values is
> harmless in either interpretation and keeps the "limit is always explicit" invariant.
> DEFAULT_FETCH_LIMIT applies to the final SELECT exactly as in tier 1; `user_limit` semantics
> for TruncationWarning carry over unchanged.

> DECISION — the SQL selects output columns `AS` the **final user-facing names** (aliases
> applied in the SQL, quoted), so the tier-2 step's `aliases` map is empty and `columns` are
> the outputs. Normalize still runs (strips any reserved columns) but has nothing to rename.
> Rationale: one naming step instead of two; `summary.fields` keys match what users see.

## 3. compile/sqlgen.py

```python
def try_sql(plan: nodes.PlanNode, *, options: SplitOptions) -> SemanticCompilation | None
```

Returns a `SemanticCompilation` with `tier=2` — **reuse the existing carrier**; do not invent a
parallel class. Field mapping for tier 2: `query` = the SQL-job `Query` (references included),
`scan` = the underlying `ScanSource` (explain's `topic:` line), `aliases` = `{}`, `columns` =
output names, `sorts` = outer sort summary, `group_keys`/`measures` = the SQL GROUP BY keys and
`()` (governed measures never appear in a tier-2 step), `measure_filters` = `()`.
`RemoteStep(label="sql", tier=2)` then works everywhere unchanged (executor, dataframe,
truncation warnings) because `RemoteStep` only reads these attributes.

Internally: match the plan bottom-up exactly like `semantic._match` (reuse its style), build a
`_SqlShape` (core, where, group keys, aggs, having, computed, sorts, limit/offset), then render.

### Expr → sqlglot (build the AST programmatically; NEVER string-format user values)

Use `sqlglot.exp` constructors; literals via `exp.Literal.string()` / `.number()`,
`exp.Boolean`, `exp.Null()`, dates as `exp.Cast(this=exp.Literal.string(iso), to=DATE/TIMESTAMP)`.
Identifiers always `exp.column(name, quoted=True)` / `exp.to_identifier(name, quoted=True)`.

| Expr | sqlglot |
|---|---|
| `FieldRef(name, grain)` | quoted column `"name[grain]"` (the reference already computed the grain) |
| `Literal` | typed literal as above |
| `Comparison(EQ/NE/LT/LE/GT/GE)` | `exp.EQ/NEQ/LT/LTE/GT/GTE` |
| `Arithmetic(ADD/SUB/MUL/DIV)` | `exp.Add/Sub/Mul/Div` |
| `BooleanOp(AND/OR)` | `exp.and_/exp.or_` (n-ary folds) |
| `Not(x)` | `exp.Not` |
| `IsNull(x)` | `exp.Is(this=x, expression=exp.Null())` (+ `Not` for negation) |
| `IsIn(x, vs)` | `exp.In` |
| `StringPredicate(CONTAINS/STARTS_WITH/ENDS_WITH/LIKE)` | `LIKE` with `%`-wrapping; escape `%`/`_`/`!` in user values (`ESCAPE '!'` — **never** `'\'`, which does not tokenize on Snowflake/BigQuery/Redshift/Spark/MySQL) — LIKE only for the LIKE kind, the other three build the pattern from the escaped value |
| `…case_insensitive=True` | wrap both sides in `LOWER()` — DECISION: portable, unlike `ILIKE` |
| `Between` numbers | `col >= low AND col <= high` (inclusive — matches M2 client semantics) |
| `Between` dates | `col >= low AND col < high` (half-open — matches M2) |
| `AdHocAgg(fn, operand, distinct)` | `exp.Sum/Count/Avg/Min/Max`; `count_distinct` → `exp.Count(this=exp.Distinct(...))` |

NULL semantics are native three-valued logic in SQL — identical to the local engine's Kleene
results by construction; the tier-2-vs-tier-3 differential (§7) is the proof, not a hand-waved
claim.

> DECISION — dialect: generate with sqlglot's default (ANSI-ish) dialect via `.sql(dialect=None)`.
> The SQL we emit is deliberately boring (quoted identifiers, standard aggregates, LIKE/LOWER).
> Escape hatch: `SessionBuilder.sql_dialect(name)` → `EnvelopeOptions.sql_dialect` →
> `SplitOptions` → passed to `.sql(dialect=...)`. Add CONTRACT_NOTES LIVE-VALIDATE #9:
> "tier-2 SQL across real warehouse dialects (quoting, LIMIT/OFFSET, LOWER/LIKE)".
>
> The default dialect is a *syntax* choice, and it must never become a safety one. It escapes
> only the single quote, which is correct where a backslash is an ordinary character (ANSI,
> Postgres, DuckDB) and wrong on every backslash-escaping warehouse (Snowflake, BigQuery,
> Redshift, Databricks/Spark, MySQL), where the default rendering of `x\'` ends the literal one
> character early and the rest of the value is parsed as SQL. Omniframes is never told the
> connection's dialect, so **a string literal containing a backslash is refused when no
> `sql_dialect` is set** and the predicate runs one tier down; with an explicit dialect sqlglot
> escapes for it and the value rides through untouched. "A value is a value, never a fragment of
> statement" is unconditional, so it may not depend on a dialect nobody told us.

Determinism: fixed key order everywhere (fields in first-use order, refs in first-use order,
`.sql(pretty=True)` for stable multi-line output) — golden `.sql` files must be byte-stable.

## 4. Splitter integration

- `SplitOptions` grows: `disable_sql: bool = False`, `sql_dialect: str | None = None`.
  (`EnvelopeOptions` untouched except the builder plumb-through for `sql_dialect` — it is a
  compile knob, not an envelope field, so it belongs on `SplitOptions`; the session passes it
  when building `SplitOptions`.)
- `_Splitter.split()`: after `try_semantic(widened)` returns None and before `_local`:
  `if not options.disable_sql: c = try_sql(widened or node, options=...); if c: return self.remote(c, label="sql")`.
- `_Splitter._aggregate()`: before building the raw scan + `LocalAggregate` pair for the ad-hoc
  half, attempt `try_sql(nodes.Aggregate(node.child, keys, adhoc), ...)`; on success that
  remote step replaces `computed` and feeds the same `AlignJoin`. On None, M3 behavior verbatim.
- `explain()` rendering for a tier-2 step:

```
Remote [tier 2 · sql → POST /api/v1/query/run]
  topic: order_items   model: bench_ecommerce
  sql:
    SELECT "users.state", COUNT(DISTINCT "users.id") AS "buyers"
    FROM ref_1
    GROUP BY "users.state"
    … (+3 more lines)
  references:
    ref_1 [semantic]: fields [users.state, users.id]  filters: order_items.returned = false  (unlimited)
```

  Truncate the SQL at 8 lines with `… (+K more lines)`. Each reference renders one line reusing
  the tier-1 summary helpers (fields/filters/limit note). Update `explain.py` accordingly;
  single-remote tier-1 rendering stays byte-identical.

## 5. Fallback discipline

`try_sql` returns `None` for anything it cannot express (internally it may raise
`CannotCompile` and catch at the boundary). **A tier-2 construction failure is never a user
error when tier 3 can express the plan** — the splitter just proceeds to `_local`. The chosen
tier is always visible in `explain()` (`tier 2 · sql` vs local ops). No `df.hint()`;
`disable_sql` (reachable for tests via `SplitOptions`; do not add a public builder knob for it)
is the only override.

## 6. Writers & extras (also M5)

- `df.write` → `DataFrameWriter` (new `io/writers.py`): `csv(path)` via `pyarrow.csv.write_csv`,
  `parquet(path)` via `pyarrow.parquet.write_table`, both over `collect()`'s normalized table;
  minimal options (`csv(path, include_header=True)`, `parquet(path, compression="zstd")`).
- `to_polars()` (+`toPolars`): `polars.from_arrow(self.collect())` behind
  `try: import polars` with `ImportError("to_polars() needs the optional dependency: pip
  install 'omniframes[polars]'")`. Dev-test it — polars is NOT in the dev group; the unit test
  uses `pytest.importorskip` and a monkeypatched-ImportError test pins the message.
- `df.omni_url()`: run the envelope with `workbookUrl: true`; return
  `QueryResult.workbook_url`. Only legal for a **single-remote tier-1** plan (the wire 400s
  `workbookUrl` × `staticQueryReferences`, and a DAG has no one query to open) — `CompileError`
  otherwise, message saying which constraint bit. Executes the query (accepted cost; document).

> DECISION — the fake grows a minimal `workbookUrl` echo as part of M5 (the ONE permitted
> tests/fakes change): when the envelope carries `workbookUrl: true` (and no
> staticQueryReferences — else 400 per the existing refinement), respond with header
> `X-Omni-Workbook-Url: https://<host>/w/fake/<job-id>` alongside the normal stream. Shape is
> LIVE-VALIDATE (#10): header presence/format on a real org. e2e asserts the client plumbing,
> not the URL shape.

## 7. Test plan

- **Tier-2-vs-tier-3 differential** (strongest check; new `tests/differential/test_tiers.py`):
  same logical plan executed with `disable_sql` on and off must agree via the existing
  comparator. Cases: pure ad-hoc agg (each fn incl. `count_distinct`, NULL group keys), mixed
  agg (governed + ad-hoc), HAVING on an ad-hoc output, cross-field OR filter, computed column,
  sort+limit above an aggregate. Comparator note: tier-2 `AVG` may arrive as decimal where the
  local engine produced float64 — the promotion table already normalizes both to float compare
  with `rel_tol`; extend it if a case surfaces rather than special-casing a test.
- **Golden SQL snapshots**: checked-in `tests/golden/snapshots/sql_*.sql` (the rendered
  statement, `pretty=True`) + envelope snapshots `semantic_sql_*.json` (references included) via
  the existing snapshot pattern. Cover every row of the §1 table.
- **e2e vs known answers** through the fake: `distinct_buyers_by_month` (tier 2 now — assert
  via `handler.requests` that the envelope carries `userEditedSQL` + `rewriteSql: false` +
  `staticQueryReferences`), `revenue_and_buyers_by_state` mixed (assert exactly TWO remote
  requests: one semantic, one sql), HAVING case, cross-field OR case; `explain()` snapshot for
  the tier-2 step and for the mixed DAG.
- **Writers/extras**: csv/parquet round-trip (write → read back → equals collect()), polars
  skip-or-run + ImportError message, `omni_url()` e2e (header plumbed; 400-combination raises
  CompileError client-side before any request), fake's workbookUrl echo test in
  tests/fakes/test_fake_omni.py.
- **Unit**: sqlgen translation table (each Expr → expected SQL fragment), escaping (`%`/`_`/quotes
  in values; a value containing `'; DROP TABLE` renders as a literal, pinned), refKey ordering,
  reference-core construction (filters pushed in vs left out).

## 8. Doc follow-ups the M5 agent must make

- CONTRACT_NOTES §6: add LIVE-VALIDATE #9 (dialect) and #10 (workbookUrl header shape);
  tick #1's offline half (the fake's refKey-as-identifier assumption now has a client
  counterpart — the live probe in scripts/live_smoke.py already exists).
- DESIGN.md §2 tier-2 bullet: point to this doc.
- docs/bench_omni_model.md: only if the fake's workbookUrl echo lands (§6 DECISION).
