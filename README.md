# Omniframes

A PySpark-style Python DataFrame library for [Omni](https://omni.co). You write dataframe code; it
compiles into governed semantic queries and pushes as much compute as possible into Omni's SQL
execution layer.

> **Status: pre-release (0.1.0.dev).** APIs may change. **Not yet published to PyPI** — install
> from a checkout (see [Development](#development)); `pip install omniframes` starts working once
> the name is claimed and the first tag is released.

```python
import omniframes as of
from omniframes import functions as F

session = (
    of.OmniSession.builder.host("acme.omni.co")
    .api_key_from_env()  # reads OMNI_API_KEY
    .get_or_create()
)

orders = session.read.topic("ecommerce", "order_items")

monthly = (
    orders.filter((F.col("users.state") == "California") & ~F.col("order_items.returned"))
    .group_by(F.col("order_items.created_at").grain("month").alias("month"))
    .agg(F.measure("order_items.total_sale_price").alias("revenue"))
    .sort(F.col("month").desc())
    .limit(12)
)

print(monthly.explain())  # shows exactly what runs remotely vs. locally
df = monthly.to_pandas()
```

That frame compiles to one governed query — `explain()` says so:

```text
== Physical plan ==
Remote [tier 1 · semantic → POST /api/v1/query/run]
  topic: order_items   model: ecommerce
  fields: [order_items.created_at[month], order_items.total_sale_price]
  group by: [order_items.created_at[month]]
  measures: [order_items.total_sale_price]
  filters: users.state = 'California' AND NOT order_items.returned
  sort: order_items.created_at[month] DESC   limit: 12   version: 9
  aliases: order_items.created_at[month] -> month, order_items.total_sale_price -> revenue
Local [pandas]
  (none — fully pushed down)
```

## Design

- **Lazy and immutable.** Every DataFrame is a logical plan; nothing executes until an action
  (`collect`, `to_pandas`, `show`, `count`).
- **Three-tier compilation.** The planner pushes each plan down as far as it can:
  1. **Semantic query** — fully governed: model measures, topic join paths, row-level security.
  2. **SQL job** — warehouse-executed SQL over embedded semantic sub-queries.
  3. **Local pandas** — anything Python can do, with the remote prefix still pushed down.
- **Always explicit.** `explain()` shows the pushdown split for every query. Never a silent
  laptop-melter.
- **Omni-native.** Selecting dimensions plus a measure *is* the group-by; `group_by().agg()` is
  familiar sugar over the same semantics. This is not a PySpark drop-in.

## Documentation

The docs site is built with [MkDocs](https://www.mkdocs.org/) + Material:

```bash
uv run mkdocs serve          # live preview on http://127.0.0.1:8000
uv run mkdocs build --strict # what CI runs
```

Start with [`docs/index.md`](docs/index.md) and
[`docs/quickstart.md`](docs/quickstart.md); [`docs/mental-model.md`](docs/mental-model.md) is the
page that makes the rest of the API predictable. `examples/demo.ipynb` walks the whole feature
surface and runs offline, with no credentials.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras         # install environment
uv run pytest                # run tests (live tests auto-skip without credentials)
uv run ruff format && uv run ruff check --fix
uv run mypy
uv run mkdocs build --strict # docs site
uv run marimo edit examples/demo.py  # the demo notebook (needs OMNI_BASE_URL/OMNI_API_KEY live)
```

The full gate, which CI enforces and every milestone must pass:

```bash
uv run ruff format --check && uv run ruff check && uv run mypy && uv run pytest -m "not live"
```

Tests run in-process against a wire-faithful fake of the Omni query API over a deterministic
bench dataset — no credentials, no network. See
[`docs/offline-testing.md`](docs/offline-testing.md) for the fake, the dataset and the five test
lanes. Checks against a live Omni org are a standalone script, `scripts/live_smoke.py`, run with
`OMNI_BASE_URL` and `OMNI_API_KEY` set — not a pytest lane.

## Releasing

Prepare a version/changelog PR with `uv run python scripts/prepare_release.py prepare 0.1.0`.
Review and merge it, then explicitly push the matching `v0.1.0` tag on the merged commit to
publish to PyPI and GitHub Releases. Merging the PR does not publish; a manual Release workflow
run only validates and builds. See the [release runbook](docs/releasing.md) for Trusted Publishing
setup (no API keys), prereleases, and recovery instructions.

## Related projects

- [`omni-python-sdk`](https://github.com/exploreomni/omni-python-sdk) — the official low-level
  Python wrapper for the Omni API. Omniframes is a higher-level DataFrame front end; use the SDK
  when you want direct endpoint access.

## License

Apache-2.0
