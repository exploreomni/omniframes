# Quickstart

Install, authenticate, run a query, read an `explain()`. Ten minutes.

!!! info "About the examples"
    Every Python block on this page has been executed, verbatim, against a wire-faithful fake of
    the Omni query API — the outputs below are what it printed. The model is called `ecommerce`,
    with one topic, `order_items`, joined to `users` and `products`
    ([the bench model](offline-testing.md)). Substitute your own model, topic and field names;
    everything else is exactly as written.

## 1. Install

Omniframes is **not on PyPI yet** (0.1.0 is pending a name claim). Until it lands, install from a
checkout:

=== "uv"

    ```bash
    git clone https://github.com/exploreomni/omniframes
    cd omniframes
    uv sync --all-extras     # dev environment, all optional extras
    uv run python            # a REPL with omniframes importable
    ```

=== "pip"

    ```bash
    git clone https://github.com/exploreomni/omniframes
    cd omniframes
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[polars]"
    ```

Python 3.11+ is required. `pandas`, `pyarrow`, `httpx` and `sqlglot` come along; `polars` is an
optional extra (`omniframes[polars]`) used only by `to_polars()`.

Once the package is published, this becomes `uv add omniframes` / `pip install omniframes`.

## 2. Authenticate

Omniframes reads the same two environment variables as the official `omni-python-sdk`:

```bash
export OMNI_BASE_URL="https://acme.omni.co"
export OMNI_API_KEY="…"   # Settings → API keys, in Omni
```

!!! danger "Never hard-code the key"
    Put it in the environment or a secret manager — never in a notebook cell, a committed file,
    or a docstring. Omniframes holds the key inside the transport and nowhere else: it never
    appears in a `repr`, a log line, or an error message. Keep it that way on your side too.

### Two org-side prerequisites

1.  **The `query-api` feature flag must be enabled for your organization.** Without it every
    query endpoint answers `403 Feature not enabled` — and `whoami` keeps working, which is why
    omniframes preflights with it. The error names the remedy and who can apply it:

    ```text
    the Omni Query API is not enabled for this organization (403 from POST /api/v1/query/run:
    Feature not enabled). An organization admin has to enable the Query API feature for the org;
    the same API key then works unchanged. An Omni organization admin has to enable the Query API
    (the `query-api` feature flag) for this organization; the same API key then works unchanged.
    ```

2.  **The key's user needs model permissions.** `QUERY_TOPICS` covers `session.read.topic(...)`;
    `QUERY_FULL_MODEL` is additionally required for `session.read.view(...)`; `QUERY_SQL` for
    `session.read.sql(...)`; `VIEW_SQL` to see un-redacted SQL in `explain(analyze=True)` and in
    error messages. A missing one is equally explicit:

    ```text
    Omni denied access to this model (403 from POST /api/v1/query/run: Permission denied). The
    key's user needs the QUERY_TOPICS permission to query topics on this model. An Omni
    organization admin can grant it: …
    ```

## 3. Build a session

```python
import omniframes as of
from omniframes import functions as F

session = (
    of.OmniSession.builder.host("acme.omni.co")
    .api_key_from_env()  # OMNI_API_KEY; .api_key("…") also exists but prefer the env
    .get_or_create()
)
```

**Building a session performs no network I/O at all.** No handshake, no catalog fetch, nothing —
so constructing one in a module-level cell or a fixture is free. The `whoami` preflight runs
lazily, once, before the first call that needs the network.

Run it eagerly when you want to check credentials up front:

```python
identity = session.verify()

print(identity["keyScope"])  # 'organization' for an org key, 'user' for a PAT
print(sorted(identity))  # ['keyScope', 'orgRole', 'rolesByModel', ...]
```

`verify()` answers three questions in one call: is the key valid, which models can it see, and
which permissions does it hold on each (`rolesByModel[<modelId>]["permissions"]`). It is the
first thing to run when a query 403s.

Other builder knobs — all optional, all covered in the [API reference](api.md):
`.base_url(...)` / `.base_url_from_env()`, `.branch(uuid)` to query a model branch,
`.timezone("America/Los_Angeles")`, `.cache("SkipCache")`, `.user_id(membership_id)` to
impersonate, `.sql_dialect("snowflake")`, `.decomposition_row_cap(n)`, and `.transport(...)` for
injecting a fake in tests.

`.sql_dialect(...)` is worth one extra sentence: the tier-2 SQL omniframes writes is deliberately
boring and needs no dialect, but nothing in the API tells a client which warehouse it is talking
to. Without a dialect, a filter value containing a **backslash** cannot be rendered safely (the
escaping rules differ between Postgres/DuckDB and Snowflake/BigQuery/Redshift/Spark/MySQL), so
such a predicate is evaluated locally instead of pushed into SQL. Naming the warehouse pushes it
down again.

## 4. Look around

The catalog is read-through cached and does the minimum number of calls:

```python
print([model.name for model in session.catalog.models()])
print([topic.name for topic in session.catalog.topics("ecommerce")])

topic = session.catalog.topic("ecommerce", "order_items")
print([view.name for view in topic.views])
print(topic.field("users.state").data_type)
```

```text
['ecommerce', 'bench_marketing', 'bench_finance', 'bench_support', 'bench_ecommerce_branch']
['order_items']
['order_items', 'users', 'products']
OmniDataType.STRING
```

## 5. Your first query

A **topic** is the governed way in: it carries the model's join paths, so fields from any joined
view are selectable without saying how they join.

```python
orders = session.read.topic("ecommerce", "order_items")

(
    orders.select("order_items.id", "users.state", "order_items.status")
    .filter(F.col("users.state") == "California")
    .sort(F.col("order_items.id").desc())
    .limit(5)
    .show()
)
```

```text
+----------------+-------------+--------------------+
| order_items.id | users.state | order_items.status |
+----------------+-------------+--------------------+
| 10000          | California  | cancelled          |
| 9999           | California  | complete           |
| 9998           | California  | complete           |
| 9997           | California  | shipped            |
| 9996           | California  | processing         |
+----------------+-------------+--------------------+
```

Field names are fully qualified (`view.column`). Nothing ran until `.show()` — everything before
it just built a plan.

!!! tip "Use `&`, `|`, `~` — never `and`, `or`, `not`"
    Python cannot overload the keywords, so `(a == 1) and (b == 2)` would silently evaluate to
    `b == 2`. A `Column` raises a `TypeError` naming the operators instead. Parenthesize every
    operand: `(F.col("a") == 1) & (F.col("b") == 2)`.

Actions: `collect()` / `to_arrow()` (Arrow table), `to_pandas()`, `to_polars()`, `show(n)`,
`count()`, `first()`, `write.parquet(path)` / `write.csv(path)`, and `schema` (a `planOnly`
round trip — it plans, it never executes).

## 6. Your first aggregate

Ask for a **governed measure** and Omni computes it, under its own server-side definition:

```python
(
    orders.group_by("users.state")
    .agg(
        F.measure("order_items.total_sale_price").alias("revenue"),
        F.measure("order_items.count").alias("orders"),
    )
    .sort(F.col("revenue").desc())
    .limit(5)
    .show()
)
```

```text
+--------------+-----------+--------+
| users.state  | revenue   | orders |
+--------------+-----------+--------+
| California   | 128597.44 | 1260   |
| New York     | 99863.96  | 955    |
| Florida      | 79630.35  | 695    |
| Texas        | 77864.44  | 723    |
| Pennsylvania | 68905.90  | 663    |
+--------------+-----------+--------+
```

Add a time grain to bucket a timestamp:

```python
month = F.col("order_items.created_at").grain("month").alias("month")

(
    orders.group_by(month)
    .agg(F.measure("order_items.total_sale_price").alias("revenue"))
    .sort(F.col("month").desc())
    .limit(3)
    .show()
)
```

```text
+---------------------------+----------+
| month                     | revenue  |
+---------------------------+----------+
| 2026-06-01 00:00:00+00:00 | 48764.83 |
| 2026-05-01 00:00:00+00:00 | 52536.16 |
| 2026-04-01 00:00:00+00:00 | 41157.42 |
+---------------------------+----------+
```

`group_by().agg()` is sugar. `orders.select("users.state", F.measure("order_items.count"))`
compiles to the byte-identical query — in Omni, selecting dimensions alongside measures **is**
the group-by. The [mental model](mental-model.md) starts there.

## 7. Read the plan

`explain()` is the whole safety story. It compiles the plan (no I/O) and prints every query that
will be sent, at which tier, plus every operator that will run in this process.

```python
plan = (
    orders.group_by("users.state")
    .agg(F.measure("order_items.total_sale_price").alias("revenue"))
    .sort(F.col("revenue").desc())
    .limit(5)
)

print(plan.explain())
```

```text
== Physical plan ==
Remote [tier 1 · semantic → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  fields: [users.state, order_items.total_sale_price]
  group by: [users.state]
  measures: [order_items.total_sale_price]
  sort: order_items.total_sale_price DESC   limit: 5   version: 9
  aliases: order_items.total_sale_price -> revenue
Local [pandas]
  (none — fully pushed down)
```

`(none — fully pushed down)` is the line to look for: the entire frame is one governed query.
When it says something else, that something is exactly what will run on this machine.

`explain(analyze=True)` additionally sends each step to Omni with `planOnly: true` and appends
the server's own SQL — blanked out for callers without `VIEW_SQL`, which the output says rather
than pretending there is none.

## Next

- [Mental model](mental-model.md) — the ideas that make the rest predictable.
- [API reference](api.md) — every method, with its semantics.
- `examples/demo.ipynb` in the repository — the same ground, end to end, runnable offline.
