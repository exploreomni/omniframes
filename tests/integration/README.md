# WWI live integration tests

This suite is a read-only contract against the versioned Wide World Importers model in the
`omni-connectors-testing` organization. It covers the six modeled topics, pinned aggregate
answers, all 29 modeled relationships, and the authorization boundary between a normal Querier
and a Restricted Querier.

Both credentials must be user personal access tokens (PATs), not organization API keys. Put the
following in the gitignored `.env.integration` file:

```dotenv
OMNI_QUERIER_PAT=...
OMNI_RESTRICTED_QUERIER_PAT=...
```

Run each identity in its own pytest process:

```bash
uv run --env-file .env.integration pytest --live --principal querier tests/integration -q
uv run --env-file .env.integration pytest --live --principal restricted tests/integration -q
```

The principal-specific variables take precedence locally. A CI job can instead expose its one
PAT as `OMNI_API_KEY`. Optional overrides are `OMNI_BASE_URL`, `OMNI_WWI_MODEL_ID`, and the
principal-specific `OMNI_QUERIER_MEMBERSHIP_ID` / `OMNI_RESTRICTED_MEMBERSHIP_ID`. Supplying a
membership ID makes the identity check exact in addition to checking user scope, member status,
role, and effective permissions.

GitHub Actions uses the protected `omni-integration` Environment and two secrets:

- `OMNI_QUERIER_PAT`
- `OMNI_RESTRICTED_QUERIER_PAT`

Each workflow job receives only its own secret. Add the two optional membership IDs as
GitHub Environment variables named `OMNI_QUERIER_MEMBERSHIP_ID` and
`OMNI_RESTRICTED_MEMBERSHIP_ID` if exact account pinning is desired. `OMNI_BASE_URL` and
`OMNI_WWI_MODEL_ID` can also be overridden there; otherwise the pinned contract values are used.
The ordinary CI workflow continues to run `pytest -m "not live"` and never receives live
credentials.

The committed contract records the fixture/model version and independently computed raw-SQL
answers. Do not update those values merely to make a drift failure pass: validate the underlying
fixture and model change first, then review the contract update like a schema migration.

The normal-Querier run currently reports one expected failure: tier-2 aggregation from a topic
emits the topic name as an OmniSQL view, and this model has no view named `wwi_sales`. The test
only xfails for that exact binding defect; it will pass automatically when the compiler is fixed
and will still fail on any unrelated error. The Restricted Querier side remains a hard assertion
that the same tier-2/manual-SQL path is denied.
