# SQLTIER v2 — tier 2 on OmniSQL

Authoritative design for the tier-2 redesign (supersedes v1 — historical note at the bottom).
Read with CONTRACT_NOTES §3.5/§3.6 (the wire truth, live-pinned 2026-08-14), §2.7 (grain
`__raw` sidecars), §6 items 11–12 (open residue), and HYBRID.md (the splitter/local engine
this slots into). Every DECISION is marked with its rationale; implement as written.

Tier 2 still exists to kill tier 3's biggest cost: the **unlimited raw scan** feeding a local
aggregate. What changed is the delivery mechanism: a tier-2 job is now **one OmniSQL
statement** — `userEditedSQL` with `rewriteSql` ABSENT — that the server parses, binds against
the model (joins from the topic's relationships, measures expanded to their governed SQL,
row-level policies applied), and plans as a governed model job. There is no reference core, no
`staticQueryReferences`, no second query object. The statement IS the plan.

```sql
SELECT ${users.state}, ${order_items.sale_price_sum},
    COUNT(DISTINCT ${users.id}) AS of_expr_1
FROM ${order_items}
GROUP BY 1
HAVING COUNT(DISTINCT ${users.id}) > 10
ORDER BY 3 DESC
LIMIT 50000
```

## 1. What upgrades, what doesn't

Splitter invariant unchanged: tier 1 (`try_semantic`) → tier 2 (`try_sql`) → local, at every
node. Shapes `try_sql` accepts:

| Shape (logical plan) | OmniSQL rendering |
|---|---|
| `Aggregate(keys=dims/grains, aggs=ad-hoc and/or MEASURES)` | `SELECT ${key}…, ${measure}…, AGG(${field}) AS a_n FROM ${topic} GROUP BY 1…k` |
| Tier-1-incompatible `Filter` below the aggregate (cross-field OR, arithmetic, string-pred mixes) — AND every tier-1-compilable filter that used to ride the reference core | `WHERE`, one conjunct per underlying column (§3.3) |
| `Filter` above the aggregate over ad-hoc agg or MEASURE outputs | `HAVING` with the ad-hoc aggregate substituted inline (verified P5b) or the `${measure}` ref expanded server-side (CONTRACT_NOTES §3.6) |
| `Project`/`WithColumn` computed columns | select-list expressions `AS <alias>` |
| `Sort`/`Limit`/`offset` on top | `ORDER BY <position>` / `LIMIT`/`OFFSET` in the text (verified P1/P10) |

**NEW — mixed aggregation collapses.** Governed measures are legal select items
(`${view.measure}` expands server-side, mixes with ad-hoc aggregates — CONTRACT_NOTES §3.6,
P2/P3). A mixed `agg()` becomes ONE tier-2 statement; the M3 decomposition
(tier-1 measures + raw scan + `LocalAggregate` + `AlignJoin`) survives only as the fallback
when `try_sql` declines. See §4.

Refuses tier 2 (falls to tier 3, or to the M3 decomposition for aggregates) — the v1 list plus
the deltas marked ⊖ (capability lost with the reference core) and hazard rules from §3.3:

- UDFs / `map_pandas`; user-facing `Join`/`Union`; anything above `SqlScan`/`SavedQueryScan`;
  `with_totals()` on non-tier-1 plans (all unchanged).
- Relative date literals — ⊖ previously rode the reference core as typed filters; OmniSQL text
  has no rendering for Omni's date grammar. Tier 3 still answers them (its raw scan carries
  them as tier-1 filters).
- A string literal compared against a timestamp-grain field (unchanged rule, same reason).
- Grain refs in `WHERE` — the §3.3 hazard: mixing a bare ref and a grain ref of the same
  field is where the live predicate loss occurs (CONTRACT_NOTES §3.6). Grain refs stay legal
  as SELECT items and GROUP BY keys.
- Predicates over columns the aggregate consumed, double limits, sorts below aggregates —
  unchanged `_match` rules.
- `drop_duplicates` pushdown: **never** — `SELECT DISTINCT` is silently stripped
  (CONTRACT_NOTES §3.6). Dedup stays local.
- A field, topic, or view whose name fails the sentinel charset (§3.1) — refuse, tier 3.

## 2. Envelope and guards

```jsonc
{
  "query": {
    "modelId": "<model>",
    "userEditedSQL": "SELECT ${users.state} … FROM ${order_items} … LIMIT 50000",
    //  NO rewriteSql key — absent selects the parsed-OmniSQL path (§3.5/§3.6)
    //  NO staticQueryReferences, NO sqlSortsEnabled
    "fields": [], "table": "", "limit": 50000,   // limit: client bookkeeping only, see below
    "version": 9, ...
  }, ...
}
```

> DECISION — `Query.for_omnisql(model_id, sql, *, limit, offset)`: sets
> `user_edited_sql=sql`, `rewrite_sql=None` (key absent on the wire), and a **non-wire flag**
> `omnisql=True` (excluded from `to_wire()`). `Query.validate()` becomes: a SQL job requires
> `rewrite_sql is False` (verbatim, `for_sql`) XOR (`omnisql` flag and `rewrite_sql is None`).
> `RemoteStep.__post_init__` mirrors it off the envelope: `userEditedSQL` present requires
> either `rewriteSql: false` in the bytes, or the compilation's `omnisql` flag with the key
> absent. The structural guarantee survives both ways: user-provided SQL (`read.sql`) is built
> only by `for_sql` and can never ride the parsed path; compiler-generated OmniSQL is built
> only by `for_omnisql` from plan nodes and can never ride the verbatim path.

> DECISION — the envelope `limit` mirrors the SQL `LIMIT` (or `null` for `limit(None)`) but is
> **client bookkeeping only**: the server IGNORES the query-object limit on this path and a
> statement without `LIMIT` runs unlimited (§3.6). The always-explicit-limit invariant
> therefore lives in the TEXT: `LIMIT DEFAULT_FETCH_LIMIT` when the user never called
> `.limit(n)`, `LIMIT n` when they did, no LIMIT clause only for `limit(None)`.
> `user_limit`/TruncationWarning plumbing reads the mirrored envelope value unchanged.

## 3. compile/sqlgen.py

`try_sql`/`compile_sql` keep their signatures and the `SemanticCompilation(tier=2)` carrier.
The plan-matching machinery survives verbatim: `_match`, `_Shape`, `_core_selects` (minus the
measure refusal), `_substitute`, `_partition`, `_conjuncts`, `_has_aggregate`,
`_references_keys`. What is deleted: `_reference`, `_split_pushable` (no reference to push
into — §3.3 replaces it), `_refuse_dialect_sensitive_literals`, `REFERENCE_PREFIX`,
`reference_summary`. What changes: rendering (§3.1), naming (§3.2), WHERE discipline (§3.3).

### 3.1 Rendering: sentinel substitution

sqlglot cannot carry `${users.state}` as an identifier (the `.` splits into table.column, the
`$`/`{` invite quoting). The AST is still built programmatically — values are still typed
literal nodes, injection stance unchanged — but every model reference renders through an
opaque sentinel:

> DECISION — **sentinel substitution.** `FieldRef`/`MeasureRef`/the FROM target render as
> `exp.column("__OF_REF_<n>__", quoted=False)` sentinels; after `.sql(pretty=True)` the
> sentinels are string-replaced with `${wire_name}` tokens. Sentinels are `[A-Z0-9_]`-only so
> no dialect quotes or rewrites them, the mapping is positional and deterministic, and no user
> VALUE ever travels through the replacement (values are literals in the AST; only model
> IDENTIFIERS become sentinels). Charset gate: a substituted name must match
> `^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)?(\[[A-Za-z0-9_]+\])?$` (topic/view: the first group alone)
> — anything else refuses to tier 3, so a hostile name can never smuggle text into the
> statement.

Emission dialect: sqlglot default (ANSI-ish), no dialect knob — the server re-renders the
parsed statement per warehouse (§3.6), which is what makes §6 possible.

Expr → sqlglot table: unchanged for literals, comparisons, arithmetic, boolean ops, IS NULL,
IN, BETWEEN, LIKE (`!` escape, `LOWER()` for case-insensitive) — except:

| Expr | v2 rendering |
|---|---|
| `FieldRef(name, grain)` | sentinel → `${view.field}` / `${view.field[grain]}` |
| `MeasureRef(name)` | sentinel → `${view.measure}` — legal as a select item (bare or inside select-item arithmetic, e.g. `${m}/NULLIF(${m2},0)` and `${m}/COUNT(DISTINCT ${f})`) and in HAVING, all expanded server-side (CONTRACT_NOTES §3.6); still refused in WHERE |
| `AdHocAgg` | unchanged (`COUNT(DISTINCT <sentinel>)` …) |

GROUP BY and ORDER BY are **positional** (`GROUP BY 1, 2`, `ORDER BY 3 DESC`) — the only
verified form, and positional ORDER BY over a grain item sorts by its `__raw` value
(chronological, P4). Non-aggregated sorts over an unselected column render the expression
inline — verified: the server wraps the statement in a subquery whose `omni_sort_expr_<n>`
sidecars never reach the result columns (CONTRACT_NOTES §3.6).
`exp.Ordered(nulls_first=False)` stays — `NULLS LAST` verified (P1).

### 3.2 Naming — two regimes, predict only what is guaranteed

The server names result columns itself; live probing (CONTRACT_NOTES §3.6) splits select
items into exactly two regimes:

- **Bare refs** — `${view.field}`, `${view.field[grain]}`, `${view.measure}`: the wire name
  is exactly the canonical `view.field` name. SQL aliases are IGNORED, and duplicate bare
  refs are DEDUPLICATED to one column. This is tier-1 semantics to the letter.
- **Expression items** — everything else: the SQL alias is honored, prefixed with a scope
  view the client CANNOT predict (first-ref-wins is REFUTED: `COALESCE(${products.brand},
  ${users.country})` scopes to `users` while `${users.age} + ${products.cost}` also scopes
  to `users` — no consistent positional rule exists). Duplicate expression items survive
  with their aliases.

> DECISION — lean on what each regime guarantees, predict nothing more:
> - Bare refs: emit WITHOUT aliases, and DEDUP the select list at emission (the server would
>   collapse duplicates and shift positions otherwise); the tier-1 client-side alias map,
>   keyed by exact wire name, handles renames — including two user aliases over one field,
>   resolved client-side after the single wire column arrives.
> - Expression items: emit with globally unique generated aliases `of_expr_<n>` (n in select
>   order). Normalize matches the result column whose name ENDS WITH `.of_expr_<n>` — unique
>   in the column set by construction — and renames it to its user-facing name; the scope
>   prefix is never guessed. `SemanticCompilation.aliases` carries both kinds of entry
>   (exact wire names, and `of_expr_<n>` suffix keys marked as such); `columns` carries the
>   user-facing output order.

The tier-2-vs-tier-3 differential (§9) remains the regression net for naming drift.

### 3.3 WHERE discipline

The live predicate loss (probe M4) is specific to mixing a BARE ref and a GRAIN ref of the
same field in one WHERE. Plain same-column conjunct composition and parenthesized compound
ranges — `(${a} >= x AND ${a} < y)`, the BETWEEN shapes — survive intact
(CONTRACT_NOTES §3.6). Rules:

- Emit WHERE conjuncts naturally, BETWEEN shapes included; no per-column grouping needed.
- Grain refs never appear in WHERE — the one live-observed loss shape; a grain predicate
  tier 1 could express never reaches tier 2 anyway (tier 1 wins first), and one it could not
  refuses.
- ⊖ Tier-1-compilable filters below the aggregate — v1 pushed them into the reference core as
  governed typed filters — now render as WHERE conjuncts; the ones with no SQL rendering
  (relative dates, grained-date strings) refuse tier 2 entirely.

## 4. Splitter integration

- `SplitOptions`: `disable_sql` stays (differential lane). `sql_dialect` is DELETED (§6).
- `split()`: unchanged — `try_semantic` → `try_sql` → `_local`.
- `_aggregate()` (mixed path): FIRST attempt the whole node — keys + measures + ad-hoc — as
  one tier-2 statement via `try_sql`. On success: single `RemoteStep(label="sql", tier=2)`,
  no `AlignJoin`, no decomposition, and `decomposition_row_cap` does not apply (nothing
  decomposed). On decline: M3 decomposition verbatim (governed tier-1 step + ad-hoc half
  tier-2-else-raw-scan-plus-`LocalAggregate` + local `AlignJoin`).

> DECISION — tier 1 still wins measure-only aggregates (native governed path, measure-filter
> map, `omni_url()` eligibility, no parse layer); tier 2 takes what tier 1 declines. The
> tier ORDER is untouched; only tier 2's reach grew.

`explain()` for a tier-2 step: keep `Remote [tier 2 · sql → POST /api/v1/query/run]`, the
`topic:`/`model:` line, and the `sql:` block (`sql_lines`, 8-line budget) — the `${…}` tokens
make it self-documenting. The `references:` block is deleted with the mechanism.

## 5. Grain `__raw` normalization (tier 1 AND tier 2 — fixes LV#12)

Live, BOTH tiers return a formatted grain as a pair: `X__raw` (`DATE_TRUNC` timestamp, at the
item's select position) + `X` (formatted STRING, appended last) — CONTRACT_NOTES §2.7.

> DECISION — **`__raw` wins.** When `X` and `X__raw` are both present and `X__raw` was not
> itself requested, normalize keeps the `__raw` VALUES under the name `X` and drops both the
> formatted column and the sidecar name. Rationale: type-stable schema (a month grain is a
> TIMESTAMP whether or not the model formats it), correct local sort/join/group semantics
> downstream (formatted strings do not order chronologically in general), dataframe-native
> values. The formatted string is display formatting, not data; users who want it can format
> locally. Divergence from the Omni UI's rendering is documented in the user guide.
> `df.schema` applies the same collapse to `summary.fields` (report `X` with `X__raw`'s type,
> drop the sidecar entry). FakeOmniAPI emits the pair for month grains on BOTH the semantic
> path and the OmniSQL path (bench model formats month as `YYYY-MM`), so the collapse is
> exercised offline; other grains stay raw-only, matching an unformatted model.

Grain refs are regime-(a) bare refs (§3.2): `view.field[grain]` and its `__raw` twin are
exact, alias-proof names, so the collapse works on predicted names and needs no suffix
matching.

## 6. Obsolete machinery — removal

> DECISION — REMOVE (not deprecate; 0.1.0.dev has no users): `SessionBuilder.sql_dialect()`
> and its `EnvelopeOptions`/`SplitOptions` plumbing; `_refuse_dialect_sensitive_literals` and
> the backslash refusal (the server owns per-warehouse rendering on the parsed path — a
> backslash literal rides through as a value, P7b); CONTRACT_NOTES §6 LV#9's tier-2 half
> (dialect portability is now the server's problem; the note shrinks to the verbatim
> `read.sql` caveat, which was always the user's own SQL). The `!` LIKE escape STAYS — the
> parse re-renders `ESCAPE` with the warehouse's own escape character, so escaping is
> server-owned end to end (CONTRACT_NOTES §3.6). CHANGELOG records the removal.

## 7. FakeOmniAPI: the OmniSQL resolver

New `tests/fakes/omnisql.py`, dispatched from the sql_job handler when `rewrite_sql` is
absent (verbatim path with `rewrite_sql: false` keeps its current handler):

- Substitute `${topic}` → the bench base table with LEFT JOINs from the bench relationships,
  pruned to the views the statement references (mirrors P1/P7 join pruning).
  `${view.field}` → qualified DuckDB column; `${view.field[grain]}` → `DATE_TRUNC` expression
  (+ the month-format pair per §5); `${view.measure}` → the measure's SQL from the bench
  model definitions. Unknown topic/field → the server's error text verbatim
  (`Could not substitute Omni SQL … No such view/field "<name>"` — §3.6) so the client's
  error mapping is exercised offline.
- Rename result columns per the §3.2 regimes: bare refs surface under their canonical names
  with duplicates collapsed to one column and aliases ignored (matching the server);
  expression items keep their `of_expr_<n>` aliases under a scope prefix the fake chooses
  **arbitrarily** — deliberately not a rule the client could learn, so only suffix matching
  can work against it. The bare-ref rule being shared between fake and client is
  self-fulfilling by construction — the live re-verification step (§10) is the external
  check, and both regimes are probe-pinned in CONTRACT_NOTES §3.6.
- **Stricter than the server on the hazard shapes**: the fake REJECTS (400-style error, loud)
  `SELECT DISTINCT`, a duplicate bare select item (the server dedups silently; the client
  must dedup at emission), a bare-plus-grain WHERE mix on one field, a `rewriteSql`-absent
  job that carries sorts/calculations, and any `staticQueryReferences` on an OmniSQL job.
  The server silently rewrites these; a loud fake failure turns an emission bug into a red
  test instead of a silent live divergence.

Grow-only-what-tests-need still applies: the resolver implements exactly the constructs §1
emits, nothing speculative.

## 8. Error mapping

A job-error line matching `Could not substitute Omni SQL` on a tier-2 step means omniframes
emitted a statement the model no longer binds (schema drift between planOnly and run, or an
emission bug). The executor wraps it: `QueryError` with the offending statement attached and
a message saying the tier-2 statement was rejected by the server, naming the missing
view/field from the server text, and pointing at `explain()`. Verbatim `read.sql` errors are
untouched (the SQL is the user's own).

## 9. Test migration

- **Golden**: regenerate `sql_*.sql` (OmniSQL text) and `semantic_sql_*.json` (no
  `staticQueryReferences`, no `rewriteSql`, no `sqlSortsEnabled`; `omnisql` flag is non-wire
  so envelopes stay clean). New cases: mixed-agg single statement, measure grand total.
- **Unit (sqlgen)**: the translation table rows (`${}` rendering, sentinel charset gate,
  measure select items incl. measure arithmetic and measure HAVING), §3.2 both regimes
  (bare-ref dedup at emission, `of_expr_<n>` alias generation and ordering), §3.3 refusals
  (grain-in-WHERE; BETWEEN shapes now EMIT and are golden-pinned), injection pins (a value
  containing `'; DROP TABLE` and a value containing `${users.state}` both render as
  literals — the sentinel pass must never touch a literal).
- **Unit (querymodel/semantic)**: `for_omnisql` wire shape (no `rewriteSql` key), the XOR
  guard both ways, `RemoteStep` envelope mirror.
- **Differential**: `test_tiers.py` re-run with the new emission (`disable_sql` on/off must
  agree) + a mixed-agg case now asserting ONE remote request vs the decomposed tier-3 result.
- **Wire/normalize**: `__raw` pair fixtures (tier-1 semantic AND OmniSQL results), the §5
  collapse incl. schema; a user-requested `X__raw`-style name passing through untouched;
  suffix-match renaming of `of_expr_<n>` columns under arbitrary scope prefixes.
- **e2e**: envelope assertions flip (`userEditedSQL` with `${…}`, no `rewriteSql`, no refs;
  mixed agg = one request); explain snapshots; fake resolver tests incl. the loud hazard
  rejections; error-mapping test off the fake's substitution error.

## 10. Work packages

**The L probe battery ran 2026-08-14** (compound ranges, LIKE ESCAPE, measure arithmetic,
measure HAVING, unselected ORDER BY, expression scoping, duplicate select items); every
outcome is folded into the DECISIONs above and recorded in CONTRACT_NOTES §3.6. **No probes
remain before implementation.**

**Work packages (disjoint files, parallelizable after WP1+WP2 land):**

| WP | Files | Delivers | Acceptance |
|---|---|---|---|
| WP1 | `compile/querymodel.py`, `compile/semantic.py` (RemoteStep + measure-refusal seams), their unit tests | `for_omnisql`, `omnisql` flag, XOR guards | guard tests both ways; wire shape pinned |
| WP2 | `compile/sqlgen.py`, `compile/explain.py`, `tests/unit/test_sqlgen*.py`, golden snapshots | §3 emission, naming, WHERE discipline, explain | golden + unit lanes green; injection pins |
| WP3 | `compile/splitter.py`, `tests/unit/test_splitter.py`, `tests/differential/test_tiers.py` | §4 mixed collapse + fallback | differential green both `disable_sql` states |
| WP4 | `tests/fakes/omnisql.py` (new), `tests/fakes/sqljobs.py`, `tests/fakes/engine.py`, `tests/fakes/bench_model.py`, fake tests, `docs/bench_omni_model.md` | §7 resolver + grain pairs + loud hazards | fake tests incl. error texts; e2e unblocked |
| WP5 | `transport/normalize.py`, `transport/arrow.py`, `dataframe.py` (schema), wire/unit tests | §5 `__raw` collapse everywhere + §3.2 `of_expr_<n>` suffix-match renaming | wire fixtures + schema tests green |
| WP6 | `session.py`, `compile/executor.py`, `tests/e2e/*`, `docs/DESIGN.md`, `docs/INTERNALS.md`, `CHANGELOG.md` | §6 removal, §8 error mapping, e2e migration, docs | milestone gate green |

Order: WP1 → WP2 → {WP3, WP4, WP5} in parallel → WP6. WP4 may start alongside WP2 (it
consumes §3.2's rules, not the code).

**Final gate**: `uv run ruff format --check && uv run ruff check && uv run mypy &&
uv run pytest -m "not live"` all green, `mkdocs build --strict`, THEN live re-verification:
the frame probe (tier-2 ad-hoc agg, mixed agg, HAVING case must PASS live) and
`scripts/live_smoke.py` (0 FAIL), against the org in `.env` while it lasts.

---

*Historical note — v1 (refuted 2026-08-14).* v1 delivered tier 2 as verbatim SQL
(`rewriteSql: false`) over a governed tier-1 query embedded in `staticQueryReferences` and
named `FROM ref_1`. The server has no refKey-as-table mechanism — the envelope fails live
with `relation "ref_1" does not exist`, and FakeOmniAPI's temp-view materialization modeled
something that does not exist (CONTRACT_NOTES §3.5, §6 closure #1). v1's plan-shape analysis,
HAVING substitution, typed-literal injection stance, and fallback discipline all survive in
v2; the delivery mechanism, naming model, and dialect machinery do not.
