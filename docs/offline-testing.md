# Offline testing

The compiler, transport, three tiers, NDJSON framing, and Arrow decoding are exercised
in-process against a **fake of the Omni query API**, backed by a **deterministic synthetic
dataset** with pre-computed answers. The fake implements the documented behaviors exercised by
the tests; live probes check the assumptions it cannot establish on its own.

This page is for contributors. Users need nothing here.

## The two pieces

### `FakeOmniAPI` — the org

`tests/fakes/` holds an `httpx.MockTransport` handler that serves `whoami`, the catalog
endpoints, `POST /api/v1/query/run`, `GET /api/v1/query/wait`, the saved-query endpoint and
`ai/generate-query`. It is faithful where it counts:

- **NDJSON framing is exact** — `Content-Type: text/ndjson`, a header line of submitted job ids,
  one line per job, a footer whose `timed_out` is the *string* `"true"`/`"false"`, and a
  trailing separator after every line.
- **Results are real.** The wire query object is compiled to DuckDB SQL over the checked-in
  parquet files and the `result` payload is a genuine base64 Arrow IPC stream. Decimal columns
  arrive as `decimal128`, timestamps as tz-aware, and the client's normalizer has real bytes to
  normalize.
- **Failure modes are configurable**: `feature_flag_off`, `permissions`, `redact_sql`,
  `rate_limited_after`, `slow_job_polls`, `ai_credits_exhausted`.
- **Silent server behaviors are reproduced literally.** The `rewriteSql` key alone picks between
  the two `userEditedSQL` paths, exactly as it does live: `false` runs the text verbatim on the
  warehouse, an **absent** key parses it as OmniSQL and resolves `${topic}` / `${view.field}`
  against the bench model. A client that puts the wrong marker on a statement fails offline
  instead of lying live.
- **Unsupported query shapes are refused loudly**, never approximated. The fake implements
  the behaviors exercised by the tests. Its [known gaps](bench_omni_model.md#known-gaps-in-the-offline-twin)
  distinguish rejected features from options that are accepted but have no effect.

Wiring it up is three lines — the real `HttpTransport` runs on top of it, so nothing about the
client is stubbed out:

```python
import httpx

from omniframes import OmniSession
from omniframes.transport import HttpTransport
from tests.fakes import DEFAULT_TOKEN, FakeOmniAPI

BASE_URL = "https://bench.example.omni.co"

handler = FakeOmniAPI()
client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
transport = HttpTransport(base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client)
session = OmniSession.builder.base_url(BASE_URL).transport(transport).get_or_create()

orders = session.read.topic("bench_ecommerce", "order_items")
```

`handler.requests` records every request (headers deliberately excluded — the bearer token must
never reach a log), which is how the test suite asserts wire invariants such as "every envelope
carries an explicit limit" and "a generated OmniSQL statement carries no `rewriteSql` key at
all, and its `LIMIT` is in the text".

### The bench dataset — the data

`tests/data/bench/` holds 500 users, 200 products and 10 000 order items as parquet and CSV,
generated deterministically by `tools/bench/generate.py` (seeded; never reads the clock). Every
edge case a query engine trips over is baked in on purpose: NULL group keys, ~1 % orphan foreign
keys, decimals, tz-aware timestamps, an empty string that must stay distinct from NULL, a
nullable boolean for three-valued logic, and multi-KB text with embedded newlines.

`known_answers.json` carries pre-computed results — revenue by state, monthly revenue, distinct
buyers, the mixed-aggregation case. **Tests assert against that file, never against a
recomputation inside the test**, so the same assertion holds verbatim against a live org.

Regenerate with `uv run python tools/bench/generate.py` (`--scale N` scales the fact table only,
so join selectivity stays put).

Full specifications:

- [Bench dataset](BENCH_DATASET.md) — schema, edge cases, invariants tests may rely on.
- [Bench Omni model](bench_omni_model.md) — the governed model over it, view by view, plus the
  offline ↔ live parity checklist and the list of what the fake still refuses.

## The test lanes

| Lane | What it proves |
|---|---|
| `tests/unit/` | Every component in isolation — plan nodes, compiler, splitter, local engine. |
| `tests/golden/` | Compiler snapshots (query JSON, generated SQL, `explain()` text) as checked-in files, diff-reviewed. No snapshot library. |
| `tests/wire/` | `HttpTransport` against fixture NDJSON **bytes** covering every documented quirk: string `timed_out`, wait cycles, all three error envelopes, redaction, totals, an unterminated tail, exotic Arrow types. |
| `tests/differential/` | The same logical operation via pushdown vs. an independent pandas reference over the same parquet — and tier 2 vs. tier 3 for the same plan. Three implementations cross-checking each other: DuckDB, Arrow compute, pandas. |
| `tests/e2e/` | Whole user-facing flows through the real client stack against the fake, asserted against `known_answers.json`. |
| `tests/integration/` | Live WWI catalog, pinned query results, and permissions for two PAT roles. Requires explicit `--live --principal querier|restricted`; see the suite's README. |
| `scripts/live_smoke.py` | A separate live probe for the LIVE-VALIDATE register of CONTRACT_NOTES §6. Needs `OMNI_BASE_URL` + `OMNI_API_KEY`. |

## Running things

```bash
uv sync --all-extras                 # install / refresh the environment
uv run pytest                        # offline tests; live integration tests skip by default
uv run pytest -m "not live" -q       # the offline suite, explicitly
uv run pytest tests/e2e -q           # one lane
uv run pytest tests/unit/test_splitter.py::test_name
uv run python scripts/live_smoke.py  # the live probe (needs credentials); not a pytest lane
```

The validation gate enforced by CI:

```bash
uv run ruff format --check && uv run ruff check && uv run mypy && uv run pytest -m "not live"
```

Plus the docs site:

```bash
uv run mkdocs build --strict
uv run mkdocs serve                  # live preview on http://127.0.0.1:8000
```

## The demo notebook

`examples/demo.ipynb` walks the whole feature surface — read, filter, aggregate, `HAVING`, mixed
aggregation, UDF, raw SQL, cross-frame join, totals, writers — with an `explain()` after every
step so the tier story is visible.

It runs **either** against the in-process fake (the default: no credentials, no network) **or**
against a live org when `OMNI_BASE_URL` and `OMNI_API_KEY` are set. One notebook, two backends,
identical cells — which is the same offline↔live parity contract the test lanes rely on.

`tests/e2e/test_demo_notebook.py` executes it headlessly against the fake on every CI run, so a
notebook that has silently rotted fails the build. Commit it with **outputs cleared**.

## Adding to the fake

Extend the fake alongside the features and tests that need new wire behavior. Two rules:

1. **New wire behavior must cite a source location** in [CONTRACT_NOTES.md](CONTRACT_NOTES.md) —
   read out of the Omni monorepo, not out of the public OpenAPI spec, which is wrong in several
   load-bearing places. Behavior derived from source but not yet confirmed against a live org
   gets a `LIVE-VALIDATE` entry there, not a guess.
2. **Refuse what you cannot vouch for.** A fake that approximates an unpinned behavior teaches
   the client a shape the real server never sends. Every offline-only choice the fake *does*
   make is written down in [bench_omni_model.md](bench_omni_model.md) §5.3 and §6.7.
