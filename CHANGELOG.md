# Changelog

## 0.1.0 (2026-09-15)

Initial release.

- Release preparation generates a version/changelog PR. A maintainer's explicit version-tag
  push runs the release gates, uploads validated distributions through PyPI Trusted Publishing,
  and creates a GitHub Release with the reviewed notes and matching artifacts.

- The package version now has one source of truth shared by `omniframes.__version__`, wheel and
  source-distribution metadata, and PyPI. Release builds reject a `v*` tag that does not exactly
  match it. HTTP requests identify the Omniframes and Python runtime versions and report a
  best-effort, coarse `google-colab`, `google-colab-enterprise`, or `databricks` runtime label;
  no raw environment values or workspace identifiers are sent.

- Idempotent GETs now recover from WAF 429s within a configurable cumulative wait budget;
  `SessionBuilder.rate_limit_wait(...)` configures it without changing POST retry behavior.

- Catalog bootstrap now resolves cheaply and hydrates lazily. `read.topic(...)` and
  `read.view(...)` each cost three requests on a cold session, independent of how many models the
  key can see and how many topics the model has — previously `ceil(N_models/100)` for model
  resolution plus `1 + N_topics` for a bare-view read, all against a shared 60 req/min bucket.
  Model resolution uses the exact-match `?modelId=`/`?name=` filters; view-name validation uses
  the flattened `GET /models/{id}/view` list instead of walking every topic's field metadata and
  discarding it. `catalog.models()` and `catalog.views()` are unchanged for callers who want the
  full listing or the typed fields, and a new `catalog.view_names(model)` exposes the cheap path.

  Model resolution verifies a filtered reply against the filter it sent rather than trusting the
  first record: the server drops a filter it considers empty (`...(name && {name})` is JS
  truthiness), so an unverified `records[0]` would silently resolve an arbitrary model. A miss
  against an already-cached catalog is answered from the cache instead of re-asking the server.

- **Breaking, custom transports only:** `QueryTransport` gains a required `list_views(model_id)`
  member. `Protocol` is not runtime-checked, so a transport injected through
  `SessionBuilder.transport(...)` that predates this release still constructs and only fails at
  the first `read.view(...)`, with a bare `AttributeError`. `mypy` catches it; nothing else will.
  There is no fallback to the old topic-detail path — that path is the `1 + N_topics` fan-out
  this release exists to remove.

- **Behavior change:** `read.view(...)` now accepts any view in the composed model, including
  views no topic reaches, and `hidden` ones (the server filters neither out of the flattened
  list). Bare views are read outside any topic, so the old topic-reachability gate contradicted
  the method's contract; the accepted set strictly widens.

- Lazy, immutable PySpark-style `DataFrame` API over Omni's semantic layer
  (`OmniSession`, `read.topic` / `read.view` / `read.sql` / `read.saved_query`, `session.ask`).
- Three-tier compile chain with a DAG splitter: governed semantic queries (tier 1), one
  generated **OmniSQL** statement planned as a governed model job (tier 2), local Arrow
  execution (tier 3) — with `explain()` showing exactly what runs where. A tier-2 statement
  refers to the model directly (`FROM ${topic}`, `${view.field}`, `${view.measure}`), so an
  `agg()` mixing governed measures with ad-hoc aggregations is a single request.
- Removed before the first release, with the tier-2 mechanism they belonged to: the
  `SessionBuilder.sql_dialect(...)` knob and the refusal to push a filter value containing a
  backslash. Omni parses the statement and re-renders it in the warehouse's own dialect, so
  neither had anything left to do (docs/SQLTIER.md §6).
- A formatted time grain arrives as a timestamp, not as the model's display string: Omni returns
  both, and omniframes keeps the raw value under the plain column name (docs/SQLTIER.md §5).
- Governed measures, time grains, measure filters (HAVING), column totals, cross-frame joins
  and unions with SQL NULL semantics, ad-hoc aggregations, UDFs via `map_pandas`/`F.udf`.
- Offline-first test bed: a wire-faithful fake of the Omni query API over a deterministic
  bench dataset, plus golden, differential and e2e test lanes, and `scripts/live_smoke.py` for
  checks that need a real org.
- Documentation site (MkDocs + Material + mkdocstrings): landing page, quickstart, a
  measure-first mental-model page (tiers, limits, aliases, `between()` and NULL semantics), a
  generated API reference, an offline-testing guide for contributors, and the design notes.
  `uv run mkdocs build --strict` is a CI job; every code sample in the README and the
  user-facing docs is executed against the fake.
- `examples/demo.ipynb` — a guided tour of the whole feature surface with `explain()` after
  every step, running either against the in-process fake (default, no credentials) or against a
  live org (`OMNI_BASE_URL` / `OMNI_API_KEY`, model and topic from `OMNI_MODEL` / `OMNI_TOPIC`).
  Executed headlessly in CI by `tests/e2e/test_demo_notebook.py`.
- Packaging validated end to end: `uv build`, `twine check`, and a clean-venv wheel install that
  runs a fake-backed query (`py.typed` ships in the wheel).
- Nested arithmetic keeps its grouping in tier-2 SQL: `(a - b) / c` no longer flattened to
  `a - b / c`. SQLGlot prints the tree it is handed and never re-derives precedence, so the
  parens are now nodes; without them the warehouse answered by its own precedence and tiers 2
  and 3 disagreed on the same frame.

<!-- Release notes generated using configuration in .github/release.yml at 42b810f9f52d4947075c37466030544cbb88006c -->

### What's Changed
#### Features
* feat(release): automate release preparation and tag publishing by @dspangen in https://github.com/exploreomni/omniframes/pull/11
#### Documentation
* docs: refresh beta status and omni styling by @dspangen in https://github.com/exploreomni/omniframes/pull/22
#### Other changes
* Recover gracefully from transient WAF 429s during catalog resolution by @dspangen in https://github.com/exploreomni/omniframes/pull/2
* Parenthesize nested arithmetic in tier-2 SQL by @dspangen in https://github.com/exploreomni/omniframes/pull/3
* docs(examples): add the marimo guided-tour demo notebook by @dspangen in https://github.com/exploreomni/omniframes/pull/7
* perf(catalog): resolve models and view names without enumerating by @dspangen in https://github.com/exploreomni/omniframes/pull/6
* docs: add coding-agent contribution guidelines by @dspangen in https://github.com/exploreomni/omniframes/pull/10
* feat: add versioned user agent and release safeguards by @dspangen in https://github.com/exploreomni/omniframes/pull/8
* test: add WWI dual-role integration suite by @dspangen in https://github.com/exploreomni/omniframes/pull/9
* docs: clarify query grouping and replace milestone references by @dspangen in https://github.com/exploreomni/omniframes/pull/12
* fix: use omniapp.co for example organization hostnames by @dspangen in https://github.com/exploreomni/omniframes/pull/19
* docs: move bench specifications into internal repository docs by @dspangen in https://github.com/exploreomni/omniframes/pull/18
* refactor(catalog): remove unsupported relationship wire fallback by @dspangen in https://github.com/exploreomni/omniframes/pull/15
* refactor(compile)!: remove redundant traversal and unused options by @dspangen in https://github.com/exploreomni/omniframes/pull/14
* fix(compile): bind topic sql to its base view by @dspangen in https://github.com/exploreomni/omniframes/pull/20
* ci: run integration tests on pushes to main by @dspangen in https://github.com/exploreomni/omniframes/pull/21
* docs(testing): describe the explicit live integration opt-in by @dspangen in https://github.com/exploreomni/omniframes/pull/16
* docs(sql): clarify unresolved topic-specific sql semantics by @dspangen in https://github.com/exploreomni/omniframes/pull/17
* fix(testing): remove unsupported query-reference tables from fake api by @dspangen in https://github.com/exploreomni/omniframes/pull/13

### New Contributors
* @dspangen made their first contribution in https://github.com/exploreomni/omniframes/pull/2

**Full Changelog**: https://github.com/exploreomni/omniframes/commits/v0.1.0

