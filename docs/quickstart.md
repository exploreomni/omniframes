# Quickstart

Install, authenticate, run a query, read an `explain()`. Ten minutes.

!!! info "About the examples"
    The query examples on this page have been executed, verbatim, against a wire-faithful fake of
    the Omni query API — the outputs below are what it printed. The model is called `ecommerce`,
    with one topic, `order_items`, joined to `users` and `products`
    ([the bench model](offline-testing.md)). Substitute your own model, topic and field names;
    everything else is exactly as written.

## 1. Install

Omniframes is available on PyPI. It is in **beta** — expect breaking changes as the API evolves.

=== "uv"

    ```bash
    uv add omniframes
    ```

=== "pip"

    ```bash
    pip install omniframes
    ```

Python 3.11+ is required. `pandas`, `pyarrow`, `httpx` and `sqlglot` come along; `polars` is an
optional extra (`omniframes[polars]`) used only by `to_polars()`.

## 2. Authenticate

Omniframes reads the same two environment variables as the official `omni-python-sdk`:

```bash
export OMNI_BASE_URL="https://acme.omniapp.co"
export OMNI_API_KEY="…"   # Settings → API keys, in Omni
```

Use your organization’s `<example-slug>.omniapp.co` hostname; replace `acme` with your own slug.

### Notebook secrets

For each credential, `get_or_create()` checks an explicit builder value, then `OMNI_BASE_URL`
or `OMNI_API_KEY` in the environment, then **one** notebook secret provider. It reads only
missing credentials. `.secrets(...)` configures that fallback without accessing any secrets;
passing `.transport(...)` skips credential resolution entirely.

The default `.secrets("auto")` recognizes a loaded Colab runtime or the current notebook's
`dbutils` object. If both are present, select a provider explicitly. Snowflake requires explicit
selection because we have not identified a supported notebook detector. Package
installation alone never selects a provider, and secret selection is independent of the
User-Agent runtime label.

Use `.secrets(None)` to disable notebook lookup. The `.base_url_from_env()` and
`.api_key_from_env()` helpers read **only** environment variables and fail immediately when
the requested variable is missing. For custom secret names, use `.secrets(api_key_name="my_key",
base_url_name="my_url")`; these names do not change the environment variable names.

#### Google Colab

In Colab, open **Secrets** (the key icon in the sidebar), add `OMNI_BASE_URL` with your Omni
organization URL and `OMNI_API_KEY` with your API key, and enable **Notebook access** for both.
The builder detects Colab and reads those secrets automatically:

```python
from omniframes import OmniSession

session = OmniSession.builder.get_or_create()
```

You can also supply `.host("acme.omniapp.co")` and store only `OMNI_API_KEY` in Secrets, or
select `.secrets("colab", api_key_name="my_omni_key")` explicitly. Automatic selection follows
the `"google.colab" in sys.modules` pattern in a
[Google-published notebook](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/use-cases/media-generation/consistent_imagery_generation.ipynb).
This is a runtime hint, not a guarantee that Secrets is available; Colab Enterprise is excluded
from automatic selection.

Colab's published
[`userdata.get()`](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py)
contacts the notebook frontend, so secret lookup needs a connected Colab UI. If a secret is
missing, add it; if access is denied, enable **Notebook access**. When Secrets is unavailable,
configure the environment variables instead.

#### Databricks

Create a secret scope containing `OMNI_API_KEY`, grant the notebook's principal access, and
specify the scope:

```python
session = (
    OmniSession.builder.host("acme.omniapp.co").secrets("databricks", scope="omni").get_or_create()
)
```

Omniframes uses the notebook's existing `dbutils` and the published
[`dbutils.secrets.get(scope, key)`](https://docs.databricks.com/aws/en/dev-tools/databricks-utils#secrets-utility-dbutilssecrets)
API. It does not create an SDK client or try remote authentication. `.secrets(scope="omni")`
also works with automatic selection. The scope is required only when a missing credential
needs a secret lookup. Omit `.host(...)` to read `OMNI_BASE_URL` from the scope too.

The active IPython namespace lookup follows a
[Databricks-published pattern](https://docs.databricks.com/aws/en/dev-tools/databricks-connect-legacy#access-databricks-utilities),
also used by the [Databricks SDK](https://github.com/databricks/databricks-sdk-py/blob/main/databricks/sdk/runtime/__init__.py).
Finding `dbutils` is a capability hint; it does not guarantee secret access or a hosted runtime.

#### Snowflake

For **Notebooks in Workspaces**, attach a `GENERIC_STRING` secret and an external access
integration (EAI) to the notebook service. Configure the EAI's network rule to allow your Omni
hostname. Set the normalized `database/schema/name` path as documented in
[Snowflake's secrets guide](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-in-workspaces/notebooks-in-workspaces-using-secrets):

```python
session = (
    OmniSession.builder.host("acme.omniapp.co")
    .secrets("snowflake", api_key_name="analytics/notebooks/omni_api_key")
    .get_or_create()
)
```

This provider calls `snowflake.snowpark.secrets.get_generic_secret_string()` from the notebook
runtime. For a URL stored as a secret, omit `.host(...)` and also set `base_url_name` to its
normalized path.

For **legacy Snowflake Notebooks**, associate a `GENERIC_STRING` secret with both the EAI and
the notebook under an alias, then select the legacy provider:

```python
session = (
    OmniSession.builder.host("acme.omniapp.co")
    .secrets("snowflake-legacy", api_key_name="omni_api_key")
    .get_or_create()
)
```

The legacy provider reads `streamlit.secrets[alias]`, following
[Snowflake's legacy notebook guide](https://docs.snowflake.com/en/user-guide/ui-snowsight/notebooks-external-access).
Here `api_key_name` is the notebook alias, rather than a Workspaces path.

Errors identify missing secrets, denied access, or unavailable runtime services where the
provider exposes that distinction. Some platforms combine missing and inaccessible secrets.
Provider exception text and secret values are omitted. A failed selected provider never causes
a lookup on another platform.

The [manual verification notebook](https://github.com/exploreomni/omniframes/blob/main/examples/notebook_secrets.ipynb)
checks each platform without printing credentials. Automated coverage uses simulated runtimes;
it does not establish that a live notebook's permissions, frontend, or network are configured.

!!! danger "Never hard-code the key"
    Put it in the environment or a secret manager — never in a notebook cell, a committed file,
    or a docstring. Omniframes holds the key inside the transport and nowhere else: it never
    appears in a `repr`, a log line, or an error message. Keep it that way on your side too.

    Requests do carry a User-Agent with the Omniframes version, Python language and implementation
    versions, and a coarse Colab or Databricks label when detected. It never includes raw
    environment values, hostnames, usernames, workspace/cluster identifiers, or paths.

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

session = of.OmniSession.builder.host("acme.omniapp.co").get_or_create()
```

**Building a session makes no Omni API calls.** Authentication and catalog requests remain
lazy. Resolving a missing credential from a notebook provider can contact its secret service
or frontend; explicit values or environment variables skip that lookup. The `whoami` preflight
runs lazily, once, before the first call that needs Omni.

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
`.base_url(...)` / `.base_url_from_env()`, `.secrets(...)` for notebook credentials,
`.branch(uuid)` to query a model branch,
`.timezone("America/Los_Angeles")`, `.cache("SkipCache")`, `.user_id(membership_id)` to
impersonate, `.decomposition_row_cap(n)`, `.rate_limit_wait(seconds)` for each rate-limited GET,
and `.transport(...)` for injecting a fake in tests.

There is deliberately no warehouse-dialect knob. The tier-2 statement omniframes writes is
OmniSQL, which Omni parses against the model and re-renders in the warehouse's own dialect —
quoting, `LIMIT`, `NULLS LAST` and the `LIKE … ESCAPE` character are all decided server-side, so
there is nothing about the connection a client needs to know.

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

If you chain `select(...).group_by(...).agg(F.measure(...))` on a topic, the group keys replace
the previously selected dimensions: omitted dimensions drop out, and Omni evaluates the
measure at the new grain without summing an intermediate result locally. The grouping fields
and governed measures must be available in the earlier `select()`; otherwise compilation
raises `CompileError`. See [regrouping a selected topic DataFrame](mental-model.md#regrouping-a-selected-topic-dataframe)
for an example and the scope of this behavior.

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
