# Internals — expression tree, plan nodes, compile pipeline

The precise internal shapes. DESIGN.md says *what* and *why*; this doc pins *exactly how* so the
layers compose. Implementations must match these signatures (rename only with a doc update).

## 1. Expressions (`src/omniframes/column.py`)

An immutable tree of frozen dataclasses, all subclasses of `Expr`:

```
FieldRef(name: str, grain: str | None = None)      # "users.state"; grain → wire "name[grain]"
MeasureRef(name: str)                              # governed model measure
Literal(value: object)                             # str | int | float | bool | date | datetime | None
AdHocAgg(fn: AggFn, operand: FieldRef, distinct: bool = False)
    # AggFn enum: SUM, COUNT, AVG, MIN, MAX  (count_distinct = COUNT + distinct=True)
Comparison(op: CmpOp, left: Expr, right: Expr)     # EQ NE LT LE GT GE
Arithmetic(op: ArithOp, left: Expr, right: Expr)   # ADD SUB MUL DIV — never tier 1
BooleanOp(op: BoolOp, operands: tuple[Expr, ...])  # AND OR (n-ary, flattened on construction)
Not(operand: Expr)
IsNull(operand: Expr)
IsIn(operand: Expr, values: tuple[object, ...])
StringPredicate(kind: StrPredKind, operand: Expr, value: str, case_insensitive: bool = False)
    # StrPredKind: CONTAINS, STARTS_WITH, ENDS_WITH, LIKE
Between(operand: Expr, low: object, high: object)
    # NUMBERS: inclusive both ends (PySpark-consistent) — compiles to composite
    #   AND(GREATER_THAN is_inclusive, LESS_THAN is_inclusive), NOT the wire BETWEEN kind
    #   (whose upper bound is exclusive).
    # DATES/DATETIMES: half-open [low, high) — the wire offers only ON_OR_AFTER (>=) and
    #   BEFORE (<) for dates, so an inclusive upper bound is not exactly expressible without
    #   date arithmetic we refuse to invent. Documented loudly in Column.between.
SortKey(expr: Expr, descending: bool = False)
```

`Column` is the user-facing wrapper: `Column(expr: Expr, alias: str | None = None)`.
Methods (each returns a new Column): `alias(name)` (also `.name(...)` no — only `alias`),
`grain(g)` (valid only over a bare FieldRef; validates against `querymodel` grain list, raises
`CompileError` otherwise), `desc()` / `asc()` (wrap into SortKey at use site), comparison and
arithmetic dunders building the nodes above, `__invert__` → Not, `__and__`/`__or__` → BooleanOp,
`is_null()`/`is_not_null()`, `isin(*values)`, `contains/starts_with/ends_with/like`,
`between(low, high)`. `__bool__` raises `TypeError` with the `&`/`|`/`~` guidance.
`__eq__` builds a Comparison (so `__hash__ = None` issues: set `__hash__ = object.__hash__`
explicitly to keep Columns usable in identity sets; document it).

`functions.py` (`F`): `col(name)`, `lit(v)`, `measure(name)`, `sum/avg/min/max/count(col_or_name)`,
`count_distinct(col_or_name)`, and `udf`. `DataFrame.map_pandas()` applies a function to a whole
frame. String arguments are accepted anywhere a Column is (auto-wrapped via `F.col`).

## 2. Plan nodes (`src/omniframes/plan/nodes.py`)

Frozen dataclasses, subclasses of `PlanNode`; children are explicit fields. Every node exposes
`children: tuple[PlanNode, ...]` (property) for generic walking.

```
Scan(source: ScanSource)                     # leaf
  ScanSource is a union of frozen dataclasses:
    TopicScan(model_name: str, model_id: str, topic: str, base_view: str)
    ViewScan(model_name: str, model_id: str, view: str)
    SqlScan(model_id: str, sql: str)
    SavedQueryScan(document_id: str, name: str, query: dict)  # hydrated at read time
Project(child, columns: tuple[Column, ...])  # select(); may contain dims, measures, ad-hoc aggs
Filter(child, predicate: Expr)
Aggregate(child, keys: tuple[Column, ...], aggs: tuple[Column, ...])   # group_by().agg()
Sort(child, keys: tuple[SortKey, ...])
Limit(child, n: int | None, offset: int = 0)   # n=None means user asked for unlimited
Join(left, right, on: tuple[str, ...] | Expr, how: JoinHow)
Union(left, right)
WithColumn(child, name: str, expr: Expr)
MapPandas(child, fn: Callable, schema_hint: OmniSchema | None)
```

`plan/visitor.py`: a small generic `transform(node, fn)` / `walk(node)` utility pair, plus
`transform_expr(expr, fn)` / `walk_expr(expr)` for expression trees. Every `Expr` that
reports `children` can be rebuilt around new ones — `AdHocAgg` is the single deliberate
exception, and it says so by name. The compiler has its own inline rewriters
(`splitter._rebind`, `semantic.alias_map` + `resolve`); these are the reusable form.

**Alias map:** aliases are collected per-plan at compile time by walking Project/Aggregate
columns (`{alias: wire_name}`). Building a plan with two aliases mapping the same wire name, a
duplicate alias, or an alias shadowing a projected wire name raises `CompileError` at
construction of the DataFrame op (fail fast, not at action time). Sorts/filters referencing an
alias resolve through the map before compilation. Selecting one field **both bare and aliased**
(`select("users.state", F.col("users.state").alias("s"))`) is not a collision but a *copy*: one
`fields` entry returns one column, so tier 1 declines it (`CannotCompile`) and tier 2 writes
`x, x AS y`. Asking for the identical column twice (same output name) collapses to one.

### Semantic-query implementation notes

- `Limit.n: int | Unset | None` — three states (user n / UNSET sentinel = library default 50 000 /
  None = unlimited), matching the wire trichotomy. `offset()` without `limit()` uses UNSET.
- `predicate_to_filters` RAISES `CannotCompile(reason)` (reason is load-bearing for error
  messages); `try_semantic()` returns None on non-match.
- `compile_plan(plan, *, options: EnvelopeOptions | None)` — scans carry their own model/topic
  identity; options carry session branch/cache/timezone/userId. `ExecutionPlan`/`RemoteStep`
  live in `compile/semantic.py`, re-exported from `compile/__init__`.
- The literal's Python type picks the filter arm (no schema at compile time): `== "2026-03"` is
  a STRING filter; use `date`/`datetime` literals or `.grain()` for date filters. Date
  comparisons support only `>=` (ON_OR_AFTER) and `<` (BEFORE); `>`/`<=` on dates raise
  CannotCompile.
- No implicit select-*: acting on a frame without `select()` is a CompileError. Ops above a
  `Limit` node (select/filter/sort after `.limit()`) are CannotCompile in tier 1.
- Topic scans send BOTH `table=<base_view>` and `join_paths_from_topic_name=<topic>`; view
  scans send only `table` (this also drives the transport's permission hint in 403 mapping).
- `show(n)`/`first()` suppress TruncationWarning (self-imposed limits); collect/to_pandas/
  to_arrow/count warn.
- The whoami preflight also guards the first catalog call; `catalog.views()` derives from
  topic-detail payloads (no per-view endpoint exists).

### SQL and stored-query implementation notes

- `SqlScan(model_id, sql, model_name="")` — the model **name** rides along purely so `explain()`
  can print `model: <name>` next to `sql: <raw SQL job>`; the wire only ever sees `model_id`.
- `SavedQueryScan(document_id, name, query, origin="saved query")` — `origin` tells the two
  endpoints of CONTRACT_NOTES §4 apart: `"saved query"` (`GET /documents/{id}/queries`) and
  `"ask"` (`POST /ai/generate-query`), where `document_id` instead carries the model the prompt
  was answered against. It is what `explain()` renders as `saved query: <name> (<doc>)` vs.
  `ask("<prompt>")   model: <id>`.
- `SemanticCompilation` grew four derived attributes and one override, all in
  `compile/semantic.py`:
  - `envelope_query: WireDict | None` — a stored query object to put on the wire **verbatim**
    instead of `query.to_wire()`. Set for saved queries and `session.ask()`: omniframes did not
    write those blobs and re-serializing them would quietly normalize keys it was handed (§4).
    `query` then only *describes* the payload (fields, limit) for `explain()`.
  - `opaque` — the *server* decides this step's payload (`SqlScan` / `SavedQueryScan`). An opaque
    step is a wall in both directions: nothing pushes into it, nothing above it compiles.
  - `is_sql` — omniframes *wrote* this step's SQL (a tier-2 OmniSQL job). The exact complement
    of `opaque`: both put `userEditedSQL` on the wire, but only this one carries `${…}` model
    refs and compiler-chosen output columns, which is why `explain()` renders them differently.
    The two are also told apart *on the wire* by the `rewriteSql` key: `false` for the verbatim
    payload, **absent** for the parsed OmniSQL one (CONTRACT_NOTES §3.5/§3.6).
  - `role` — what the step is, for `explain()` and the truncation warning: `"semantic"`,
    `"sql"`, `"raw SQL job"`, `"saved query"` or `"generated query"`.
- `RemoteStep.applied_limit` — the limit this step actually sends (`None` = unlimited), read off
  the **envelope** when `envelope_query` is set, because there the bytes, not the typed `Query`
  that merely describes them, are what the server applies. `_warn_if_truncated` and the
  executor's per-step warning both read it. The envelope's `limit` is a *trichotomy*, and
  `semantic.envelope_limit` decodes all three arms: an int is itself, `null` is unlimited, and a
  **key that is absent entirely** — which only a blob omniframes was handed can be — is
  `DEFAULT_SERVER_LIMIT = 1000`, the value `createQuery` fills in (CONTRACT_NOTES §3). Collapsing
  the absent key into `None` would print `limit: unlimited (null)` over a 1000-row truncation and
  suppress the warning; `explain()` renders it as `1000 (server default — the stored query
  carries no limit)` so the two cases never read alike.
- `RemoteStep.__post_init__` refuses, by construction, to carry `userEditedSQL` under the wrong
  reading of `rewriteSql` — the key the server picks the path from, and the one failure it does
  not report (CONTRACT_NOTES §3.4/§3.6). It mirrors the compilation's non-wire `omnisql` flag:
  a compiled OmniSQL statement must have the key **absent**, anything else must carry
  `rewriteSql: false`. Every remote step passes through it, compiled or verbatim, so neither
  path can reach the transport under the other's marker.
- `QueryError.statement` — the OmniSQL text the server refused, set only when the executor
  re-reads a `Could not substitute Omni SQL` job error as omniframes' own emission bug or as
  model drift (docs/SQLTIER.md §8). `compile/executor.remote_errors(step)` is the context
  manager that does it, and it wraps every request a compiled step makes: the DAG walk inside
  `execute()`, and `dataframe`'s single-step `collect`/`schema` shortcut. A raw-SQL job's errors
  pass through untouched — that SQL is the user's own.
- `TransportError.status: int | None` — the HTTP status the failure came from, `None` for a
  request that never got an answer. It is the **only** HTTP detail that crosses the transport
  seam, and it does so as an attribute rather than as text: `session.py` branches on it to tell
  the two documented `/documents` 404s apart and to give `generate-query`'s 400/402 their own
  advice (§4), and recovering it by re-reading the message would make the wording load-bearing.

## 3. Wire names

`wire_name(expr) -> str` (in `compile/semantic.py`):
- `FieldRef("v.c")` → `"v.c"`; with grain → `"v.c[month]"`
- `MeasureRef("v.m")` → `"v.m"`
- `AdHocAgg(SUM, v.c)` → display name `"sum(v.c)"` — has NO wire name (never tier 1).
Result columns arrive under wire names; `normalize(aliases=...)` renames to aliases last.

## 4. Compile pipeline (`compile/`)

```
split(plan: PlanNode, *, options: SplitOptions | None = None) -> ExecutionPlan
```

- DataFrame actions call `splitter.split`. At each node it tries `try_semantic`, then
  `try_sql`, and finally local execution or aggregate decomposition. A successful remote
  compilation holds a `querymodel.Query`, alias map, and projected column order.
- `semantic.py` implements `try_semantic(plan) -> SemanticCompilation | None`; unsupported
  shapes return `None`, while invalid plans raise `CompileError`. Its separate `compile_plan`
  helper builds a single remote plan and raises on unsupported shapes instead of invoking
  the splitter. It is not the DataFrame's three-tier entry point.
- `predicate_to_filters(expr) -> dict[str, Filter]` normalizes tier-1 predicates: top-level
  AND splits per field; per-field OR becomes a composite; the grain-filter rule (DESIGN §3)
  applies. Governed measure filters compile to HAVING in tier 1. Cross-field OR, arithmetic,
  and filters over ad-hoc aggregates require SQL or local execution.
- Limit policy: user Limit(n) → n; absent → `DEFAULT_FETCH_LIMIT = 50_000`
  (defined in `compile/querymodel.py`); Limit(None) → wire null. Local aggregation inputs
  use the separate unlimited-by-default decomposition policy (HYBRID §2.1).
- `ExecutionPlan` (defined in `compile/semantic.py`, re-exported by `compile/__init__.py`)
  contains remote steps and an optional DAG root of `RemoteStep`/`LocalStep` nodes. A single
  remote query has `root=None`. `explain_text` in `compile/explain.py` renders the execution:

```
== Physical plan ==
Remote [tier 1 · semantic → POST /api/v1/query/run]
  topic: order_items   model: bench_ecommerce
  fields: [...]
  filters: users.state = 'California' AND NOT order_items.returned
  sort: ...   limit: ...   version: 9
Local [pandas]
  (none — fully pushed down)
```

## 5. Session/catalog

- `OmniSession.builder` → `SessionBuilder`: `.host(str)` / `.base_url(str)`, `.api_key(str)`,
  `.api_key_from_env()` (OMNI_API_KEY), `.branch(str)`, `.timezone(str)`, `.cache(str)`,
  `.user_id(str)`, `.rate_limit_wait(float)` (per-GET 429 waiting budget),
  `.transport(QueryTransport)` (injection for tests), `.get_or_create()`.
  No network I/O. `session.verify()` runs whoami eagerly; otherwise the first action triggers a
  cached whoami preflight for crisp errors.
- `session.catalog`: `models()` (paginate all), `model(name_or_id)` (one exact-match filtered
  request — `?modelId=` for a UUID, else `?name=`; the cursor walk is only the fallback that
  builds the error's model list), `topics(model)`, `topic(model, name)` (full metadata → typed
  `TopicInfo` with views/fields/relationships), `views(model)` (typed, topic-scoped, `1+N_topics`
  requests), `view_names(model)` (flattened `/view`, one request, composed-model scope). All
  read-through-cache; `refresh()` clears. Filtered resolutions cache separately from `models()`,
  so a cheap lookup never masquerades as the complete catalog.
- `session.read.topic(model, topic)` / `read.view(model, view)` → DataFrame(Scan). Resolution of
  model name and topic existence happens at read time (one catalog call; clear error naming the
  model/topic). Both are three requests on a cold session — whoami, the filtered model lookup,
  and one list call — independent of catalog and topic count. `df.schema` → `session._transport.plan(...)` cached on the DataFrame instance.
- Actions: `collect() -> pa.Table` (normalized), `to_pandas()`, `to_arrow()`, `show(n=20)`,
  `count()`, `first()`, `explain(analyze=False)`, `omni_url()`. `with_totals()` marks a frame
  for server-side totals and remains lazy.
  Truncation warning per DESIGN §3.

## 6. Testing interfaces

- Fake wiring: `HttpTransport(base_url="https://bench.omniapp.co", api_key=BENCH_KEY,
  client=httpx.Client(transport=httpx.MockTransport(FakeOmniAPI()), base_url=...))`.
- Golden tests build DataFrames against a **stub catalog** (no I/O): construct scans directly or
  stub the transport's catalog responses with the fake's payloads.
- Differential lane: run the same logical ops via (a) the full pipeline against the fake
  and (b) pure pandas over `tests/data/bench/*.parquet`, compare with the comparator rules
  (DESIGN §5).
