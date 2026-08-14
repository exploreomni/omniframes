# The hybrid engine (M3) — authoritative design

How a plan that tier 1 cannot fully express gets executed: split into **remote steps** (governed
queries, exactly as today) feeding **local operators**, with the split always visible in
`explain()`. This doc fits the as-built code (`compile/semantic.py`, `plan/nodes.py`,
`column.py`, `dataframe.py`) and assumes M2's end state (Aggregate compiles tier-1 when all
aggs are governed measures; measure filters compile to HAVING; `with_totals()` exists).
Rules here are binding; deviations require editing this doc in the same change.

Every DECISION below is a judgment call with its one-line rationale; everything else follows
from the contract or the existing code.

## 0. Module map

```
compile/splitter.py   split(plan, *, options) -> ExecutionPlan          (new)
compile/local.py      LocalOp types, eval_expr, run_local_op            (new)
compile/executor.py   execute(execution, run_remote) -> pa.Table        (new)
compile/semantic.py   ExecutionPlan gains `root` and `output_columns`   (extended, back-compat)
compile/explain.py    renders multi-step DAGs                           (extended)
column.py             + Udf expr; functions.py + F.udf                  (extended)
dataframe.py          with_column / map_pandas / mapInPandas; _collect routes through executor
```

## 1. ExecutionPlan generalization

```python
Step: TypeAlias = "RemoteStep | LocalStep"


@dataclass(frozen=True)
class LocalStep:
    op: LocalOp  # see §3
    inputs: tuple[Step, ...]  # ≥1; dataflow edges


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[RemoteStep, ...]  # ALL remote steps, DFS order (explain, tests)
    root: Step | None = None  # None ⇒ M1/M2 degenerate: exactly steps[0]
    output_columns: tuple[str, ...] | None = None  # final user-facing columns when root is local
```

- Constructor compatibility: every existing `ExecutionPlan((remote_step,))` call keeps working;
  `root=None` means "single remote step, no local work".
- `.remote` keeps its exact current behavior (raises when `root` is a LocalStep or
  `len(steps) != 1`) so M1/M2 fast paths and tests are untouched.
- `.columns` returns `output_columns` when set, else the single remote step's columns.
- `.tier` when root is local: `3` (report the *lowest* tier involved; explain shows detail).

`DataFrame._collect` becomes: if `execution.root is None` → current fast path (unchanged);
else `executor.execute(execution, run_remote=self._session_runner)` where `_session_runner`
runs one RemoteStep (`session.run(step.envelope)` → `normalize(..., aliases=step.alias_map)`)
and returns the normalized Arrow table. `execute` is pure given `run_remote` — fully testable
without a session.

## 2. The splitter

`split(plan, *, options) -> ExecutionPlan`. Top-down recursion; the invariant is
**maximality**: at every node, first attempt `try_semantic(node)` on the whole subtree — if it
compiles, that subtree is one RemoteStep and recursion stops. Only on `CannotCompile` does the
node itself become a local op over the split of its child(ren). Because the attempt happens at
every level, the remote frontier is automatically maximal; no separate analysis pass.

Per-node rules when the subtree does NOT compile whole:

| Node | Local op | Notes |
|---|---|---|
| `Filter` (uncompilable predicate: cross-field OR, arithmetic, string-vs-measure, Udf) | `LocalFilter(predicate)` | **Column widening** (§2.2): the remote child must also fetch every field the predicate references. |
| `Project` (uncompilable column: arithmetic, Udf) | `LocalProject(columns)` | Compilable columns still push down via widening; the local project computes the rest and sets final order. |
| `Sort` above a local op | `LocalSort(keys)` | A Sort whose subtree compiles rides remote as today. |
| `Limit` above a local op | `LocalLimit(n, offset)` | See §2.3 — a Limit *below* local ops pins the frontier instead. |
| `Aggregate` with any `AdHocAgg` | decomposition — §2.1 | Pure-measure aggregates compile tier-1 (M2), never reach here. |
| `WithColumn` | `LocalWithColumn(name, expr)` | Always local in M3 (tier 2 takes some in M5). Widening applies to referenced fields. |
| `MapPandas` | `LocalMapPandas(fn, schema_hint)` | Always local, by definition. |
| `Join` / `Union` | M4 — splitter handles children independently, op is local | Nodes exist; DataFrame exposes them in M4. |

The recursion carries a `required: frozenset[str]` of extra field names needed by local ops
above (see §2.2) and, at the end, wraps the root in a `LocalProject` that drops widened columns
and fixes output order/aliases.

### 2.1 Mixed-aggregation decomposition

`Aggregate(child, keys=K, aggs=M ∪ A)` where `M` are `MeasureRef`s and `A` are `AdHocAgg`s
(A non-empty; if M is empty it is the same shape minus step 1):

1. **RemoteStep "measures"**: `compile_semantic(Aggregate(child, K, M))` — the governed side,
   filters and scan included, `DEFAULT_FETCH_LIMIT` + TruncationWarning as usual. Omitted when
   `M` is empty.
2. **RemoteStep "raw scan"**: `compile_semantic(Project(child, K ∪ operands(A)))` — raw rows of
   the group keys plus every distinct `AdHocAgg.operand`. **`limit: None` (unlimited)** — see
   the DECISION below.
3. `LocalAggregate(keys=K_names, aggs=A)` over step 2's output.
4. `AlignJoin(on=K_names)` joining step 1 ⨝ step 3 (skipped when `M` is empty). Output column
   order: keys, then aggs in the user's original `agg()` order (measures and ad-hoc interleaved
   as written).
5. Everything ABOVE a decomposed Aggregate (Sort/Limit/Filter/…) runs **locally**.
   DECISION: uniform and honest beats clever re-pushdown; a sort on a measure column *could*
   ride step 1, but the join's row alignment makes local sorting equally correct and the rule
   trivially predictable.

`K_names` are the user-facing key names (aliases applied) — both remote steps carry the same
alias map, so the join keys line up by name.

DECISION — **the raw scan is unlimited** (`limit: None`): a silently 50k-capped input to a local
aggregation is a wrong-answer bug, not a truncation inconvenience. The wire allows `null` limit
(no pivots involved). Cost is visible: `explain()` prints `raw scan (unlimited)` and the docs
say ad-hoc aggregation pulls raw rows until tier 2 (M5) pushes it into SQL. A safety valve
`OmniSession.builder.decomposition_row_cap(n)` (default `None`) turns the unlimited scan into
`limit n` + a *mandatory* `TruncationWarning` when hit — opt-in cap, never a silent default.
When the rows under the aggregate need local work of their own (a UDF, a `read.sql` scan, a
join) the scan cannot carry the cap on the wire; it then becomes a `LocalLimit(n,
decomposition_cap=True)` instead, which `explain()` renders as `limit: n (decomposition cap)`
and which the executor warns about on exactly the same rule. "Mandatory" is unconditional: a
capped aggregate is a wrong answer, so there is no arrangement of the plan in which the cap
bites quietly.

DECISION — **`with_totals()` on a decomposed aggregate is a `CompileError`**: totals are a
tier-1 server feature; emulating totals over the local side would break "the local engine never
computes governed measures".

### 2.2 Column widening

When a local op references fields the remote projection below it does not carry, the splitter
**widens** the remote projection with those `FieldRef`s (bare dimensions/grains only —
referencing a measure from a local *filter* is only legal above the aggregate that produced it,
where it resolves to an output column, §3.1). Widened columns are tracked and dropped by the
final `LocalProject`. Widening a raw scan created by decomposition is the same mechanism.
Referencing a field that cannot be widened (e.g. under a `MapPandas` whose output schema is
unknown) is a `CompileError` naming the column.

### 2.3 Limit pins the frontier

`…local ops… → Limit → …remote-compilable…` : the Limit rides remote (today's semantics), and
the local ops run on the limited result — semantically honest, matches M1's refusal reason.
`Limit → …local ops…` (limit *above* local work): `LocalLimit`. Both directions tested.

### 2.4 TruncationWarning policy

The executor warns per RemoteStep whose returned rows == its applied limit (never for
`limit: None`), naming the step: `"remote step 2 (raw scan) returned exactly its 50000-row
limit; the local aggregate may be wrong — set decomposition_row_cap(None) or use .limit()"`.
`show()`/`first()` keep suppressing warnings only for the final user-facing limit, never for
intermediate scans.

## 3. The local engine (`compile/local.py`)

DECISION — **the local engine operates on `pyarrow` Tables via `pyarrow.compute`, not pandas**.
pandas enters only at the `MapPandas`/`Udf` boundary and in `to_pandas()`. Rationale: Arrow
compute gives SQL parity for free — Kleene three-valued `and_kleene`/`or_kleene`/`invert`,
null-propagating comparisons, hash `group_by` that keeps a NULL-key group (SQL
`GROUP BY`-with-`dropna=False` semantics), `count_distinct` excluding nulls, decimal128
arithmetic, `sort_indices(null_placement=)` — while pandas needs per-dtype workarounds for every
one of those. DESIGN.md's "pandas operator interpreter" wording is superseded by this doc.

Ops (all pure functions `(inputs: tuple[pa.Table, ...]) -> pa.Table`):

```python
LocalFilter(predicate: Expr)        # mask = eval_expr(predicate); pc.filter — NULL mask drops the row
LocalProject(names: tuple[str, ...], renames: Mapping[str, str])
LocalWithColumn(name: str, expr: Expr)
LocalAggregate(keys: tuple[str, ...], aggs: tuple[LocalAgg, ...])   # LocalAgg: (out_name, AggFn, operand_col, distinct)
LocalSort(keys: tuple[tuple[str, bool], ...])                        # (name, descending)
LocalLimit(n: int | None, offset: int)
LocalMapPandas(fn, schema_hint)     # table.to_pandas(types_mapper=pd.ArrowDtype) → fn → pa.Table.from_pandas
AlignJoin(on: tuple[str, ...])      # §3.2
LocalJoin(on: tuple[str, ...], how: JoinHow = INNER)   # M4 — the USER join; §3.2
LocalUnion()                                            # M4 — UNION ALL by position; §3.3
```

### As-built (M4) — the two user-facing set operators

- `LocalJoin` is the operator `df.join(...)` compiles to, and it is deliberately **not**
  `AlignJoin`. Its one rule: a NULL key never matches anything, not even another NULL (§3.2).
  pandas' `merge` matches NA keys to each other — right for aligning a decomposed aggregate,
  wrong for a join — so null-keyed rows are held out of the merge entirely and re-attached
  afterwards as unmatched rows: dropped by `inner`, kept and NULL-padded by the side an outer
  join preserves. Overlapping non-key column names are refused by the splitter before the plan
  runs, so the output is always keys, then the left frame's other columns, then the right's.
- `LocalUnion` stacks two results by position (`UNION ALL` — nothing de-duplicates), but insists
  the column **names** agree as well as their count: omniframes' columns are named wire outputs,
  and borrowing the left side's names for a differently named right side would relabel data
  rather than stack it. Types widen per §3.3; a pair with no common type is a `CompileError`.
- Both take two inputs and are always local — the query API takes one query, so each side is its
  own remote sub-plan and the tiers of the two sides need not match (a governed aggregate joined
  to a raw-SQL job is the canonical case).

### 3.1 `eval_expr(expr: Expr, table: pa.Table) -> pa.Array | pa.Scalar`

- `FieldRef` resolves against the table's column names in order: exact name → alias → wire name
  → display name (tables reaching local ops are already normalized+aliased). `MeasureRef`
  resolves the same way — it is legal only as a *column reference* to an already-computed remote
  result; if the name is absent, `CompileError` ("governed measures only exist remotely").
- `Comparison` → `pc.equal/not_equal/less/…` (null-propagating: `NULL > 5` → NULL).
- `BooleanOp`/`Not` → `and_kleene`/`or_kleene`/`invert` (`NOT NULL` → NULL → row dropped by
  LocalFilter; test proves `~(col == x)` keeps neither the match nor the NULLs).
- `IsNull` → `is_null`; `IsIn` → `pc.is_in` (NULL never matches); `Between(low, high)` →
  numbers: `low <= x AND x <= high` (inclusive, matching M2's compiled semantics); dates:
  `low <= x AND x < high` (half-open, matching the wire). String date-grammar literals
  ("30 days ago") are refused locally: `CannotCompile("relative date literals are evaluated by
  Omni")` — they only ever execute remotely.
- `StringPredicate` → `match_substring`/`starts_with`/`ends_with`/`match_like`, with
  `ignore_case=case_insensitive`; default case-SENSITIVE, matching the wire.
- `Arithmetic` → `pc.add/subtract/multiply/divide` (decimal-aware; division of ints → float64).
- `Udf(fn, operands)` → evaluate operands, convert to pandas Series (ArrowDtype), apply, convert
  back. `Literal` → `pa.scalar`.

### 3.2 NULL group keys and the AlignJoin

- `LocalAggregate` uses `pa.TableGroupBy(table, keys, use_threads=False)` — NULL keys form a
  real group (SQL semantics; `use_threads=False` keeps output order deterministic).
  Aggregation null semantics (tests for each): `sum`/`min`/`max`/`mean` skip nulls and return
  NULL for an all-null or empty group; `count(col)` counts non-null; `count_distinct(col)`
  excludes NULL (`count_distinct` with mode "only_valid").
- `AlignJoin` merges two tables that aggregate THE SAME underlying rows, keyed by group: it must
  treat NULL == NULL (both sides have one NULL-state row that must become one output row) —
  the opposite of SQL join semantics. DECISION: implement via pandas
  `merge(how="outer", on=keys)` over ArrowDtype frames — pandas factorization matches NA keys,
  giving exactly the alignment we need with zero sentinel hacks; convert back to Arrow after.
  This is an internal op: the user-facing `Join` (M4) will implement SQL semantics
  (NULL keys never match) and must NOT reuse AlignJoin.

### 3.3 Dtype promotion (local aggs; the differential comparator uses the same table)

| operand | sum | avg | min/max | count / count_distinct |
|---|---|---|---|---|
| int64 | int64 | float64 | int64 | int64 |
| float64 | float64 | float64 | float64 | int64 |
| decimal128(p, s) | decimal128(38, s) — the operand is widened before the aggregate rather than trusted to Arrow, which only widens a grouped decimal sum from pyarrow 21 while the declared floor is 15 | float64 — DECISION: Arrow `mean` does not keep decimal; cast once, document, comparator uses tolerance for avg only | decimal128(p, s) | int64 |
| timestamp / date | error | error | same type | int64 |
| string / bool | error | error | string: min/max ok; bool: error | int64 |

Errors are `CompileError` at split time (not runtime) when the operand's type is known from the
plan; otherwise runtime `CompileError` from the op.

### 3.4 Local sort order

`sort_indices(null_placement="at_end")` for every direction — nulls last, deterministic
(stable). The wire's `OMNI_DEFAULT` null order is dialect-dependent and unpinned; the
differential comparator therefore never compares row order across engines unless the test
sorts explicitly with nulls normalized (§4).

## 4. The differential comparator (`tests/differential/`)

`assert_frames_agree(pushdown: pa.Table, reference: pa.Table, *, sort: bool = True)`:

1. Column names and order must match exactly.
2. Normalize dtypes per §3.3's table (both sides): decimals compared as exact strings
   (`Decimal` → `str`), floats with `math.isclose(rel_tol=1e-9)` — avg columns are float by
   §3.3, everything decimal stays exact; timestamps normalized to UTC-aware `datetime`.
3. When `sort=True` (the default; use it unless the test asserts ordering), both tables are
   sorted by ALL columns using a canonical key (`(value is NULL, stringified value)`) so null
   position differences can never fail a test.
4. Compare as row-lists; on mismatch print the first differing row from each side.

Reference computations in the differential lane are **independent pandas** (plain
`read_parquet` over `tests/data/bench/`, `groupby(dropna=False)`, explicit `Decimal` handling) —
NOT the local engine — so the lane cross-checks three implementations: DuckDB (fake), Arrow
compute (local engine), pandas (reference).

## 5. UDF boundary

```python
# column.py
@dataclass(frozen=True)
class Udf(Expr):
    fn: Callable[..., Any]
    operands: tuple[Expr, ...]
    name: str                      # display name, e.g. "my_fn(users.state)"

# functions.py
def udf(fn: Callable[..., Any]) -> Callable[..., Column]   # F.udf(f)(col, ...) → Column(Udf(...))

# dataframe.py
def with_column(self, name: str, col: Column | str) -> DataFrame        # + withColumn alias
def map_pandas(self, fn: Callable[[pd.DataFrame], pd.DataFrame],
               schema_hint: OmniSchema | None = None) -> DataFrame      # + mapInPandas alias
```

- A `Udf` anywhere in a projection/filter/with_column pins the frontier below it (it is just an
  uncompilable expression — the generic rules apply). `MapPandas` pins it structurally.
- `schema_hint` contract: after `map_pandas`, `df.schema` returns the hint if given, else raises
  `CompileError` explaining that a Python function's output schema cannot be planned; `columns`
  behaves the same. Ops above a `MapPandas` compile against the hint when present; without it,
  only ops that need no schema (limit, map_pandas again) are allowed — anything referencing a
  column raises the §2.2 widening error.
- `explain()` prints the function's `__name__` for both.

## 6. Still CannotCompile after M3 (and who picks them up)

- `SqlScan` / `SavedQueryScan` sources — M4.
- User-facing `Join` / `Union` DataFrame methods — M4 (nodes + splitter handling exist).
- Ad-hoc aggregation pushdown, HAVING on ad-hoc aggs in SQL, computed-column pushdown — tier 2,
  M5 (M3 executes them locally; M5 re-routes when expressible).
- `with_totals()` over a decomposed aggregate — permanent `CompileError` (§2.1).
- Relative date literals in LOCAL filter evaluation — permanent (`§3.1`); they work tier-1.
- Local emulation of governed measures or grains — permanent by design.
- Cross-field OR **as a wire filter** — permanent for 0.1 (executes locally instead; tier 2 can
  take it in M5).

## 7. explain() format

Single-remote plans keep today's format byte-for-byte (existing tests). Multi-step:

```
== Physical plan ==
Remote step 1 [tier 1 · semantic → POST /api/v1/query/run]
  topic: order_items   model: bench_ecommerce
  fields: [users.state, order_items.total_sale_price]
  filters: users.state != NULL
  limit: 50000   version: 9
Remote step 2 [tier 1 · raw scan (unlimited) → POST /api/v1/query/run]
  topic: order_items   model: bench_ecommerce
  fields: [users.state, users.id]
  limit: null   version: 9
Local [arrow compute]
  aggregate over step 2: keys=[users.state], aggs=[count_distinct(users.id) AS buyers]
  align-join: step 1 ⨝ aggregate on [users.state]
  sort: buyers desc
  project: [users.state, revenue, buyers]
```

Every local line names its inputs; every remote step is numbered in `steps` order. `analyze=True`
appends per-remote-step `display_sql` exactly as today, per step.

Since M5 that decomposition is the **fallback**, not the default: tier 2 takes the whole mixed
aggregate as one OmniSQL statement (docs/SQLTIER.md §4), and this shape appears only when it
declines. The rules below are unchanged — they are what runs when it does.

### As-built (M4/M5) — the opaque step's rendering

A step whose payload omniframes did not write — `SqlScan` (`read.sql`) or `SavedQueryScan`
(`read.saved_query` / `session.ask`), i.e. `SemanticCompilation.opaque` — is rendered off the
**envelope**, not off the typed `Query`, because for those steps the bytes are the truth:

```
Remote step 1 [tier 2 · raw SQL job → POST /api/v1/query/run]
  sql: <raw SQL job>   model: bench_ecommerce
  userEditedSQL:
    SELECT u.state AS state, SUM(oi.sale_price) AS revenue
    FROM order_items oi LEFT JOIN users u ON u.id = oi.user_id
    GROUP BY 1
  rewriteSql: false   sqlSortsEnabled: true
  limit: 50000   version: 9
Local [arrow compute]
  project over step 1: [state, revenue]
  filter: state = 'California'
  project: [state, revenue]
```

`rewriteSql` is printed rather than assumed: it is the key the server picks the path from, and
taking the wrong one fails *silently* (CONTRACT_NOTES §3.4/§3.6), so it belongs where a reader
can see it. `false` here means "run this text verbatim"; a tier-2 OmniSQL step leaves the key
absent and prints its statement under `sql:` instead. A stored query prints
its field list, its filter **keys** with `(stored — sent verbatim, not recompiled)`, and its
own sorts instead of a SQL body — the point being that none of it was recompiled:

```
Remote [tier 1 · saved query → POST /api/v1/query/run]
  saved query: Revenue by state (bench_dashboard)
  fields: [users.state, order_items.total_sale_price]
  sort: order_items.total_sale_price DESC
  limit: 1000   version: 9
```

This is distinct from the tier-2 `sql` rendering of §4 in SQLTIER.md, which shows the OmniSQL
statement omniframes *wrote*, `${…}` refs and all. `is_sql` and `opaque` are exact complements;
`explain.py` branches on them in that order.

## 8. Test obligations (each rule above names its test)

Unit (`tests/unit/test_splitter.py`, `test_local.py`): maximality (compilable prefix stays one
RemoteStep), each per-node rule, widening + final drop, limit-pins-frontier both directions,
decomposition shapes (M∪A, A-only), NULL group in LocalAggregate, AlignJoin NULL-key merge,
Kleene filter table (TRUE/FALSE/NULL × and/or/not), every promotion-table cell, case
sensitivity, unlimited raw-scan envelope (`limit: None` on the wire), decomposition_row_cap
warning, with_totals×decomposition error, Udf/MapPandas frontier pinning, schema_hint contract.
Golden: explain() snapshots for the §7 example and a UDF plan. E2E (`tests/e2e/`): mixed agg
`revenue_and_buyers_by_state` vs known_answers (497-buyer alignment incl. the NULL-state
group), UDF filter fallback, map_pandas roundtrip. Differential: every local op vs pandas
reference over the bench data, NULL-heavy columns included.
