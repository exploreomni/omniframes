# Releasing Omniframes

A release is prepared in a PR. **Pushing its version tag is the explicit publishing action.**
Merging a PR and manually dispatching the Release workflow never upload to PyPI.

## One-time setup

1. Configure a Trusted Publisher on the existing `omniframes` PyPI project with owner
   `exploreomni`, repository `omniframes`, workflow filename `release.yml`, and environment
   `pypi`. No API key is required.
2. Create the GitHub environment `pypi` and restrict deployments to tags matching `v*`.
   Do not add required reviewers for the normal flow: the maintainer's tag push is the approval.
   The environment and publisher must be configured explicitly before the first release;
   GitHub's automatically created environments have no protection rules.
3. Add an active tag ruleset for `v*` restricting creation, updates, and deletion, with bypass
   limited to release maintainers. Protect `main` with required PR review and CI checks.
   The workflow additionally verifies that the tagged commit is reachable from `origin/main`.
4. Ensure Actions can label PRs and create GitHub Releases. Actions are SHA-pinned; only the
   PyPI job has `id-token: write`, and only the GitHub Release job has `contents: write`.

PyPI setup and repository rules are administrator actions; adding the workflow does not configure
those services. A tag push before publisher setup will fail at upload. Rehearse with manual
workflow dispatch before the first real tag; it needs no PyPI credentials.

## Prepare and review

Use a clean checkout at the current `origin/main`, with `git`, authenticated `gh`, and `uv` available:

```bash
git switch main
git pull --ff-only
uv run python scripts/prepare_release.py prepare 0.1.0
```

The command fetches tags, checks the version, generates GitHub release notes, creates
`codex/release-0.1.0`, updates `_version.py` and `CHANGELOG.md`, commits, pushes the branch,
and opens a PR. It never creates tags, publishes a GitHub Release, or uploads to PyPI.
Review the changes in that PR before merging. Run `uv sync --reinstall-package omniframes`
after a version change so local editable metadata matches the source.

PR titles should follow Conventional Commits. The Release labels workflow assigns categories
from titles. A `!` before the colon adds `release:breaking`; maintainers can also add that label
explicitly. `release:skip` excludes a PR (release-preparation PRs receive it automatically).
Breaking and skip overrides are retained when titles change. Add migration instructions to the
release PR: generated titles cannot supply that context. Uncategorized and older PRs appear
under Other changes, so missing labels do not silently omit them. Direct commits may appear only
in GitHub's full comparison link; review that link for user-facing changes that need prose.

Notes normally use the preceding stable tag as their baseline, making prerelease notes cumulative.
Before the first stable tag, GitHub chooses its default baseline; review first-release notes in
particular. The existing curated `0.1.0 (unreleased)` overview is preserved for the initial release
and copied into initial prereleases. The finalized, dated changelog section is the authority for
GitHub Release text, and can be edited in the preparation PR.

Use `X.Y.Z` for stable releases and `X.Y.ZaN`, `X.Y.ZbN`, or `X.Y.ZrcN` for prereleases.
Development, post, local, leading-zero, and shorthand versions are rejected. Choose patch versions
for compatible fixes, minor versions for features; while below 1.0, put incompatible API changes
in a minor release and document them prominently. The command never chooses a version for you.

If preparation stops after creating the branch, continue that branch/PR manually instead of
rerunning from scratch. Check `git status` and `gh pr list --head codex/release-VERSION` first;
commit/push missing changes and open the PR if needed. Existing release branches and tags are
rejected rather than overwritten. The generated commit is attributed to the invoking Git user;
AI agents must pass their identity, for example `--coauthor "Codex <noreply@openai.com>"`,
which adds the required commit trailer and PR attribution before pushing.

## Publish

After the PR merges and CI passes, fetch main and identify the actual merged release commit
(the squash commit if using squash merge). Do not tag the preparation branch's pre-merge commit.

```bash
git fetch origin main --tags
git tag -a v0.1.0 <merged-release-commit> -m "Release 0.1.0"
git push origin v0.1.0
```

The Release workflow runs the supported Python matrix, dependency-floor tests, and docs build.
It requires the tag, source version, and a nonempty finalized changelog section to agree. It builds
one wheel and one source distribution, validates metadata, and installs each outside the checkout
in a fresh environment to check runtime version, installed metadata, and `py.typed`.

The publish job uploads those saved artifacts using PyPI Trusted Publishing. After success, a
separate job creates a draft GitHub Release, attaches the same files, and publishes it using the
reviewed changelog text. A final job dispatches **Publish docs** on `main` with the release tag.
That docs run is asynchronous; verify it and the live version page separately. Prereleases are marked accordingly. Per-tag concurrency prevents
simultaneous publication runs. No live Omni credentials are required for release checks; run the
separate Omni Integration workflow beforehand when the changes warrant live verification.

## Failed releases and retries

- **Validation or build failure:** nothing was uploaded. Fix the problem in a new PR and use a
  new version/tag. Never move a release tag.
- **Publisher setup or transient failure before any upload:** fix configuration, then rerun the
  failed publish job on the original tag run. Reuse its saved artifacts rather than rebuilding.
- **Partial upload or uncertain outcome:** inspect the PyPI file list and compare SHA-256 hashes
  with the original run's artifacts. Do not enable blanket `skip-existing`: a duplicate can hide
  a different artifact. If recovery requires uploading a missing file, an administrator must use
  that exact saved file through an authorized publisher. Otherwise choose a new version and,
  where appropriate, yank the incomplete/bad release. PyPI files cannot be overwritten.
- **PyPI succeeded, GitHub Release failed:** rerun only the failed GitHub Release job. It resumes
  an existing draft, replaces its draft attachments, and leaves an already published release
  intact. Do not rerun the successful upload job. If artifacts have expired, retrieve and verify
  the published PyPI distributions before manually completing the GitHub Release.
- **Docs dispatch or deployment failure:** the package is already published. Retry only the
  failed dispatch job, or run **Publish docs** on `main` with the release tag. Do not rerun the
  PyPI upload. A successful dispatch only means the docs run was scheduled.
- **Bad published code:** ship a new version and consider yanking the bad release; do not delete
  or recreate its Git tag.

Manual dispatch always validates/builds only, even when targeting an existing release tag. It is
not a way to bypass the explicit tag-push publishing trigger.

## Publishing documentation

The [docs site](https://exploreomni.github.io/omniframes/) uses mike to retain multiple versions,
with a version selector in the header:

- `stable` points to the newest published stable version and is the default landing page.
- `dev` follows `main`. Before any stable docs are published, the site defaults to `dev`
  (or the first published prerelease if `dev` is not yet available).
- Release tags such as `v0.1.0` publish to `0.1.0/`. Prereleases get their own version as well.
- Publishing one version preserves the others on the generated `gh-pages` branch.
- Prereleases never become `stable`; backfilling an older release cannot move `stable` backwards.

The **Publish docs** workflow (`.github/workflows/docs.yml`) builds strictly, saves the versioned
site to `gh-pages`, and deploys it through GitHub Pages. Pushes to `main` publish `dev`; after
PyPI and GitHub Release publication succeed, the Release workflow dispatches **Publish docs**
on `main` with `release_tag`. Manual Release rehearsals never dispatch docs. PRs only build docs
in CI. Concurrent docs updates are serialized. If a queued run is superseded, manually dispatch
the workflow on `main` with that release tag to publish the missing version.

Direct tag-triggered docs deployments are intentionally disabled. They can report success while
serving an earlier artifact for the same commit (see
[deploy-pages issue #383](https://github.com/actions/deploy-pages/issues/383)). Dispatching from
`main` uses the recovery path verified for `0.1.1`, while still building the exact tagged source.
Check the separate docs run and the live version URL before reporting docs publication complete.

To retry or backfill docs for an existing release, run **Publish docs** on `main` and enter its
`release_tag`, for example `v0.1.0`. Leave the input blank to rebuild `dev`. Backfills use the
current workflow's locked documentation tools with the tagged source, docs, and theme. A small
inherited config enables the selector for releases predating versioning. Historical docs must
still pass the strict build with those tools. Rebuilding a version replaces only that version.

Repository setup: select **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The `github-pages` environment only needs to allow branch `main`. The Release workflow’s docs
dispatch job needs `actions: write`. The build job needs
`contents: write` to save generated versions; the deployment job uses `pages: write` and
`id-token: write`. No personal token or PyPI credentials are needed. Do not hand-edit `gh-pages`.

PyPI displays `README.md` and the `Documentation` link from `pyproject.toml` as package metadata.
Those update with the next package release; deploying the docs does not upload a new package or
change metadata for an existing PyPI release. Keep `mkdocs.yml`'s `site_url`, the package's
`Documentation` URL, and README links aligned if the site moves.
