# The mental model

Omniframes borrows PySpark's *shape*, not its semantics. This page is the set of ideas that make
everything else predictable. Read it once and the API stops surprising you.

!!! info "About the examples"
    Every Python block below has been executed, verbatim, against a wire-faithful fake of the
    Omni query API — the outputs are what it printed. Model `ecommerce`, topic `order_items`,
    joined to `users` and `products` ([the bench model](offline-testing.md)).

```python
import omniframes as of
from omniframes import functions as F

session = of.OmniSession.builder.host("acme.omni.co").api_key_from_env().get_or_create()
orders = session.read.topic("ecommerce", "order_items")
```

---

## 1. Measure-first: the selection *is* the group-by

In SQL you write `GROUP BY` and then decide what to aggregate. In Omni's semantic layer the
model already knows what every measure means, so **asking for a dimension and a measure together
is the group-by**. There is no separate grouping step to get wrong.

```python
by_state = orders.select("users.state", F.measure("order_items.count"))
```

That returns one row per state. `group_by().agg()` is sugar for the same thing — literally the
same query object on the wire:

```python
same = orders.group_by("users.state").agg(F.measure("order_items.count"))

assert by_state.explain() == same.explain()
```

Use whichever reads better. `group_by()` with no keys aggregates the whole frame to one row.

The practical consequences:

- **A frame with no measure returns raw rows.** `orders.select("users.state")` is 10 000 rows,
  not 21.
- **Adding a measure changes the row count**, because it changes the query from a row scan into
  an aggregate. That is the semantic layer working, not a bug.
- **You cannot "un-group".** To get raw rows back, drop the measure.

---

## 2. Governed measures vs. ad-hoc aggregations

Two things look similar and behave completely differently. Getting this distinction is most of
the value of the library.

| | `F.measure("order_items.total_sale_price")` | `F.sum("order_items.sale_price")` |
|---|---|---|
| What it is | a **governed model measure** | an **ad-hoc aggregation** over a raw column |
| Who defines it | the Omni model, server-side | you, right here |
| Where it runs | **always** remotely, tier 1 | tier 2 SQL when expressible, else tier 3 |
| Governance | join paths, row-level security, the org's one definition of the metric | none beyond the reference query's own filters |
| Default column name | `order_items.total_sale_price` | `sum(order_items.sale_price)` |

```python
governed = orders.group_by("users.state").agg(F.measure("order_items.total_sale_price"))
ad_hoc = orders.group_by("users.state").agg(F.sum("order_items.sale_price").alias("revenue"))
```

Both answer "revenue by state", and against this model they agree to the cent. They are still
not the same thing: the governed measure is whatever the model says revenue is *today* — if the
model starts excluding returns tomorrow, the governed frame follows and the ad-hoc one does not.

**Omniframes never emulates a governed measure locally.** Its definition lives in the model; a
local re-implementation would be a different number wearing the same name. `F.measure(...)`
therefore pins at least one remote step into every plan that mentions it.

Ad-hoc aggregations are `F.sum`, `F.avg`, `F.min`, `F.max`, `F.count` and `F.count_distinct`.
They take a raw column, never a measure — wrapping a measure raises immediately:

```python
try:
    F.sum(F.measure("order_items.total_sale_price"))
except of.CompileError as error:
    print(error)
```

```text
'order_items.total_sale_price' is a governed measure and is already aggregated; select it with F.measure(...) instead of wrapping it in an ad-hoc aggregation
```

### Filtering after aggregation is a real `HAVING`

A predicate on a **governed measure** compiles to a measure-keyed wire filter, which the server
applies *after* the group-by — a genuine `HAVING`, not a client-side pass. It works whether the
measure is selected or not, and whether you write the filter before or after the `group_by()`:

```python
big_states = (
    orders.group_by("users.state")
    .agg(F.measure("order_items.count").alias("orders"))
    .filter(F.measure("order_items.total_sale_price") > 50_000)
)
```

`order_items.total_sale_price` never becomes a column here; the server force-adds it to the
aggregate so the `HAVING` can see it, then projects it away. A predicate on an **ad-hoc**
aggregate's output column is also a `HAVING` — written into tier-2 SQL instead (§3).

---

## 3. What runs where

At action time the planner tries, **at every node of the plan**, the highest tier that can
express the subtree: tier 1 (governed semantic query) → tier 2 (SQL job) → tier 3 (local Arrow
compute). The result is a DAG, not a prefix — a single frame can use all three at once.

`explain()` prints the answer every time. There is no silent local fallback, ever.

### Tier 1 — the governed semantic query

Scan + projections (dimensions, grains, measures) + compilable filters + sorts + limit/offset.
Fully governed.

```python
tier_one = (
    orders.select("order_items.id", "users.state", "order_items.status")
    .filter(F.col("users.state") == "California")
    .sort(F.col("order_items.id").desc())
    .limit(5)
)
print(tier_one.explain())
```

```text
== Physical plan ==
Remote [tier 1 · semantic → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  fields: [order_items.id, users.state, order_items.status]
  filters: users.state = 'California'
  sort: order_items.id DESC   limit: 5   version: 9
Local [pandas]
  (none — fully pushed down)
```

### Tier 2 — one governed OmniSQL statement

When tier 1 cannot express something — an ad-hoc aggregation, a cross-field `OR`, a computed
column, a `HAVING` on an ad-hoc aggregate — omniframes writes **OmniSQL**: ordinary SQL in which
`${order_items}` is the topic and `${users.state}` is a model field. Omni parses it against the
model, so the topic's joins, the measures' definitions and the row-level policies all apply, and
the statement runs as one governed job. The rows never leave the warehouse.

```python
tier_two = (
    orders.group_by("users.state")
    .agg(F.count_distinct("users.id").alias("buyers"))
    .filter(F.col("buyers") > 25)
)
print(tier_two.explain())
```

```text
== Physical plan ==
Remote [tier 2 · sql → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  sql:
    SELECT
      ${users.state},
      COUNT(DISTINCT ${users.id}) AS of_expr_1
    FROM ${order_items}
    GROUP BY
      1
    HAVING
      COUNT(DISTINCT ${users.id}) > 25
    … (+1 more lines)
Local [pandas]
  (none — fully pushed down)
```

Two details worth reading off that statement. `of_expr_1` is a generated alias, not your name:
Omni names result columns itself and prefixes an expression's alias with a view it picks, so
omniframes matches the column back by suffix and renames it to `buyers` before you see it. And
the `LIMIT` lives in the text — on this path the query object's own `limit` is ignored, so the
statement always carries one (the `… (+1 more lines)` above is it).

Only the 21 result rows come back: the aggregation happened in the warehouse.

### Tier 3 — local, over the largest remote prefix

UDFs, `map_pandas`, cross-frame joins and unions, and anything above them run in this process on
Arrow compute. The query underneath is still pushed down as far as it goes:

```python
def is_coastal(state):
    return state in {"California", "Oregon", "Washington"}


tier_three = (
    orders.select("users.state", "order_items.status")
    .filter(F.col("order_items.status") == "complete")  # compilable → rides remote
    .filter(F.udf(is_coastal)("users.state"))  # Python → runs here
)
print(tier_three.explain())
```

```text
== Physical plan ==
Remote step 1 [tier 1 · semantic → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  fields: [users.state, order_items.status]
  filters: order_items.status = 'complete'
  sort: (none)   limit: 50000   version: 9
Local [arrow compute]
  filter over step 1: is_coastal(users.state)
  project: [users.state, order_items.status]
```

Note what the plan is telling you: the status filter went to the server, the Python predicate did
not, and **50 000 rows** will be fetched to evaluate it. That is the honest cost of a UDF, and it
is on the screen before you pay it.

### Mixing a governed measure with an ad-hoc one

`${order_items.total_sale_price}` is a legal select item beside `COUNT(DISTINCT …)` — the server
expands the measure to its governed SQL inside the same statement — so an `agg()` that mixes the
two kinds is still one request.

```python
mixed = orders.group_by("users.state").agg(
    F.measure("order_items.total_sale_price").alias("revenue"),
    F.count_distinct("users.id").alias("buyers"),
)
print(mixed.explain())
```

```text
== Physical plan ==
Remote [tier 2 · sql → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  sql:
    SELECT
      ${users.state},
      ${order_items.total_sale_price},
      COUNT(DISTINCT ${users.id}) AS of_expr_1
    FROM ${order_items}
    GROUP BY
      1
    LIMIT 50000
Local [pandas]
  (none — fully pushed down)
```

When tier 2 *cannot* write the statement — the shapes §3 lists as out of reach — the same frame
falls back to a decomposition: a tier-1 query for the measures, a raw scan for the ad-hoc half,
and a local **align-join** on the group keys. `explain()` names all three. That align-join is
**not** a SQL join: it pairs the two halves' NULL-state groups with each other, because they are
the same group. A user-written `df.join(...)` has SQL semantics instead (§7).

### Things that pin the frontier

- A **user `.limit(n)`** rides remote, and operations written above it run on the limited result.
  `page.filter(...)` after `.limit(20)` filters those 20 rows — that is what you asked for.
- A **UDF or `map_pandas`** makes everything above it local.
- A **raw-SQL scan** (`session.read.sql`), a **saved query** or a `session.ask(...)` frame is
  opaque: omniframes did not write that query and will never re-derive it, so operations on top
  run locally over its result.

---

## 4. Limits and truncation

**The library always sends an explicit limit.** There is no such thing as an unbounded query by
accident.

| you write | wire `limit` |
|---|---|
| nothing | `50000` (`DEFAULT_FETCH_LIMIT`) |
| `.limit(200)` | `200` |
| `.limit(None)` | `null` — genuinely unlimited, at your own risk |

Limits compose to the tightest: `df.limit(50).limit(10)` is 10 rows. `offset(n)` travels with
the limit as SQL's `LIMIT n OFFSET k`.

When a result comes back with **exactly** the applied limit, the rows are probably not all of
them, and omniframes says so:

```text
TruncationWarning: the result has exactly 3 rows, which is the applied limit — rows are probably
missing. Raise .limit(n), use .limit(None) for everything, or narrow the query.
```

Raise it to an error while you develop:

```python
import warnings
from omniframes import TruncationWarning

warnings.simplefilter("error", TruncationWarning)
```

Two deliberate exceptions: `show(n)` and `first()` never warn about the limit they imposed
themselves — but they *do* still warn about an intermediate scan nobody asked for. And
`count()` counts the **materialized** frame (post-limit, PySpark-consistent), inheriting the same
warning; for a governed row count select `F.measure("order_items.count")` instead.

---

## 5. Aliases are client-side

The Omni query API has no aliasing. `.alias()` is recorded in the plan, applied as a **rename
after the result comes back**, and reverse-resolved to the wire name when a filter or sort
mentions it.

```python
month = F.col("order_items.created_at").grain("month").alias("month")

aliased = (
    orders.group_by(month)
    .agg(F.measure("order_items.total_sale_price").alias("revenue"))
    .sort(F.col("month").desc())  # written against the alias
    .limit(3)
)
```

On the wire that query asks for `order_items.created_at[month]` and sorts by
`order_items.created_at[month]`; the columns you get back are `month` and `revenue`. `explain()`
shows the map on its own `aliases:` line.

**A grain column is a timestamp, not a label.** When the model formats a grain, Omni returns it
twice — the truncated timestamp and a formatted string (`"2026-03"`) — and omniframes keeps the
timestamp. So a month grain has the same dtype whether or not somebody formatted it in the
model, it sorts and joins chronologically, and it groups like a date. The consequence is that the
column can read differently here than in the Omni UI, which shows the formatted string; format it
yourself (`strftime`, `dt.to_period`) if you want the label.

Two collisions fail **at build time**, where you wrote them, rather than at action time:

```python
try:
    orders.select(F.col("users.state").alias("x"), F.col("users.age").alias("x"))
except of.CompileError as error:
    print(error)

try:
    orders.select(F.col("users.state").alias("users.age"), F.col("users.age"))
except of.CompileError as error:
    print(error)
```

```text
alias 'x' is used twice in one select(); aliases must be unique
alias 'users.age' shadows the selected field 'users.age'; pick a different name
```

Aliases are also how you make a join or a union line up, since both match on **output** column
names.

---

## 6. `between()`: numbers include both ends, dates do not

This is the one asymmetry in the API, and it is deliberate.

```python
numbers = orders.select("order_items.quantity").filter(
    F.col("order_items.quantity").between(3, 5)
)  # 3 <= quantity <= 5 — both ends included, PySpark-consistent
```

Numbers compile to a composite of `>=` and `<=`, **not** to the wire's `BETWEEN` kind, whose
upper bound is exclusive.

```python
from datetime import date

dates = orders.select("order_items.created_at").filter(
    F.col("order_items.created_at").between(date(2026, 6, 1), date(2026, 7, 1))
)  # 2026-06-01 <= created_at < 2026-07-01 — half-open
```

```text
  filters: '2026-06-01' <= order_items.created_at < '2026-07-01'
```

Dates are **half-open**. The wire's date filter offers only `ON_OR_AFTER` (`>=`) and `BEFORE`
(`<`), so an inclusive upper bound is not exactly expressible — and omniframes refuses to
synthesize one by adding "the smallest unit" to a literal that may be relative or truncated.
Half-open is also what a month or quarter window actually wants: `[2026-06-01, 2026-07-01)` is
June, exactly, with no last-microsecond question to get wrong.

`<=` and `>` on a date are not expressible either. Use the next boundary.

Which arm applies is decided by the **literal's Python type**, because the compiler has no field
types: `== "2026-03"` is a *string* filter; pass a `date`/`datetime`, or use `.grain()`, for a
date filter.

---

## 7. NULLs: SQL's three-valued logic, everywhere

Omniframes commits to SQL semantics — including in the local engine, which is why it runs on
Arrow compute (Kleene `and`/`or`/`not`, null-propagating comparisons, a real NULL group key)
rather than on pandas. The consequence is that **remote and local execution of the same
predicate agree**, which is exactly what a hybrid engine has to promise.

Concretely, over 10 000 order items of which 300 have a NULL `returned` flag:

```python
returned = F.col("order_items.returned")

print(orders.select("order_items.id").filter(returned.is_null()).limit(None).count())
print(orders.select("order_items.id").filter(~(returned == True)).limit(None).count())  # noqa: E712
```

```text
300
9051
```

`~(returned == True)` keeps **9 051** rows, not 9 351. The 300 NULL rows are not kept: `NULL =
TRUE` is NULL, `NOT NULL` is NULL, and a filter drops a row whose predicate is NULL. This is SQL,
not a bug — but it is the thing that catches people, so:

- `IS NULL` / `IS NOT NULL` are the only predicates that *ask about* NULL — use
  `.is_null()` / `.is_not_null()`.
- A negated predicate never resurrects NULLs. `~col.isin("California")` excludes the NULL-state
  rows too (8 150 of 10 000, not 8 740).
- Aggregations skip NULLs; `count(col)` counts non-null values and `count_distinct(col)`
  excludes NULL entirely.

**NULL is a real group key.** `group_by("users.state")` returns a row whose state is `None`,
holding every order item whose buyer has no state — including the ones whose `user_id` matches no
user at all, because the topic's joins are LEFT joins. Dropping that group silently is how
totals stop adding up, so omniframes keeps it.

**In a user join, NULL keys never match.** `df.join(other, "users.state")` follows SQL: a
NULL-keyed row takes no part in matching and reappears only as an unmatched row — dropped by an
inner join, kept with the other side NULL by a join type that preserves its side. An `outer`
join of two frames that each have a NULL-state group therefore produces **two** NULL rows, not
one. (The align-join inside a decomposed aggregate does the opposite, on purpose — §3.)

---

## 8. Everything is lazy and immutable

Every transformation returns a **new** DataFrame wrapping a new plan; nothing mutates and
nothing runs. The network is touched only by an action: `collect`, `to_arrow`, `to_pandas`,
`to_polars`, `show`, `count`, `first`, `write.*`, `omni_url`, the `schema` property, and
`explain(analyze=True)`.

`explain()` on its own performs no I/O — it is free to call, always, and it is the answer to
"what is this going to do?".

## See also

- [API reference](api.md) for every method and its exact contract.
- [Architecture](DESIGN.md) and [the hybrid engine](HYBRID.md) / [the SQL tier](SQLTIER.md) for
  the design behind the tiers.
- [Wire contract](CONTRACT_NOTES.md) for what actually goes on the wire, and why.
