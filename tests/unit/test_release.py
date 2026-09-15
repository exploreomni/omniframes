"""Release preparation invariants; no credentials, Git writes, or uploads."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from scripts import prepare_release as release


@pytest.mark.parametrize("version", ["0.1.0", "1.2.3", "1.0.0a1", "1.0.0b2", "1.0.0rc1"])
def test_accepts_canonical_release_versions(version):
    release.version_key(version)


@pytest.mark.parametrize(
    "version",
    ["v1.0.0", "1.0", "01.0.0", "1.0.0rc01", "1.0.0.dev0", "1.0.0+local", "1.0.0.post1", "1.0.0\n"],
)
def test_rejects_non_release_or_noncanonical_versions(version):
    with pytest.raises(ValueError, match="Invalid release version"):
        release.version_key(version)


def test_baseline_uses_semantic_order_and_keeps_prerelease_changes():
    assert (
        release.previous_tag(["v0.9.0", "v0.10.0", "v0.11.0rc1", "unrelated"], "0.11.0")
        == "v0.10.0"
    )
    assert release.previous_tag(["v0.10.0", "v0.11.0rc1"], "0.11.0rc2") == "v0.10.0"
    assert release.previous_tag([], "0.1.0") is None


@pytest.mark.parametrize("tag", ["v1.0.0", "v1.1.0", "v1.0.1rc1"])
def test_existing_or_newer_tag_blocks_release(tag):
    with pytest.raises(ValueError, match="newer"):
        release.previous_tag([tag], "1.0.0")


def test_initial_release_preserves_curated_notes_and_generated_headings():
    original = "# Changelog\n\n## 0.1.0 (unreleased)\n\nInitial overview.\n"
    generated = "## What's Changed\n* New API (#1)\n\n## New Contributors\n* @someone\n"
    result = release.update_changelog(original, "0.1.0", generated, "2026-09-05")
    assert "unreleased" not in result
    assert release.release_notes(result, "0.1.0") == (
        "Initial overview.\n\n### What's Changed\n* New API (#1)\n\n### New Contributors\n* @someone\n"
    )


def test_initial_prerelease_preserves_overview_for_final_release():
    original = "# Changelog\n\n## 0.1.0 (unreleased)\n\nInitial overview.\n"
    rc = release.update_changelog(original, "0.1.0rc1", "## Changes\n* RC", "2026-09-05")
    assert "## 0.1.0 (unreleased)" in rc
    final = release.update_changelog(rc, "0.1.0", "## Changes\n* Final", "2026-09-06")
    assert release.release_notes(final, "0.1.0") == "Initial overview.\n\n### Changes\n* Final\n"
    assert "* RC" in release.release_notes(final, "0.1.0rc1")


def test_new_release_does_not_include_previous_release_notes():
    original = "# Changelog\n\n## 0.1.0 (2026-09-05)\n\nOld changes.\n"
    result = release.update_changelog(
        original, "0.2.0", "## Features\n* New changes.", "2026-09-06"
    )
    assert release.release_notes(result, "0.2.0") == "### Features\n* New changes.\n"
    assert release.release_notes(result, "0.1.0") == "Old changes.\n"
    with pytest.raises(ValueError, match="already contains"):
        release.update_changelog(result, "0.2.0", "Duplicate", "2026-09-06")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("## 0.1.0 (unreleased)\n\nNotes", "exactly one"),
        ("## 0.1.0 (2026-09-05)\n\n", "empty"),
        ("## 0.1.0 (2026-99-99)\n\nNotes", "month must"),
        ("## 0.1.0 (2026-09-05)\nA\n## 0.1.0 (2026-09-06)\nB", "exactly one"),
    ],
)
def test_missing_empty_invalid_or_duplicate_notes_are_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        release.release_notes(text, "0.1.0")


def release_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release, "ROOT", tmp_path)
    path = tmp_path / release.VERSION_FILE
    path.parent.mkdir(parents=True)
    path.write_text('__version__ = "0.1.0"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 0.1.0 (2026-09-05)\n\nNotes.\n")


def test_validation_checks_main_and_tag_before_writing_notes(tmp_path, monkeypatch):
    release_checkout(tmp_path, monkeypatch)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return "commit" if args[1] == "rev-parse" else ""

    monkeypatch.setattr(release, "run", fake_run)
    output = tmp_path / "notes.md"
    release.validate("v0.1.0", output)
    assert output.read_text() == "Notes.\n"
    assert ("git", "merge-base", "--is-ancestor", "HEAD", "origin/main") in calls
    assert ("git", "rev-parse", "refs/tags/v0.1.0^{commit}") in calls
    with pytest.raises(ValueError, match="does not match package"):
        release.validate("v0.2.0", None)


def test_validation_rejects_non_main_commit(tmp_path, monkeypatch):
    release_checkout(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(release, "run", fail)
    output = tmp_path / "notes.md"
    with pytest.raises(subprocess.CalledProcessError):
        release.validate("v0.1.0", output)
    assert not output.exists()


def test_preparation_stops_on_dirty_checkout_without_network_or_mutation(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        return " M README.md"

    monkeypatch.setattr(release, "run", fake_run)
    with pytest.raises(ValueError, match="clean working tree"):
        release.prepare("0.1.0")
    assert calls == [("git", "status", "--porcelain")]


def test_existing_branch_is_not_overwritten(tmp_path, monkeypatch):
    release_checkout(tmp_path, monkeypatch)
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args)
        if args[1] == "rev-parse":
            return "main-sha"
        if args[1] == "branch":
            return "codex/release-0.1.0"
        return ""

    monkeypatch.setattr(release, "run", fake_run)
    with pytest.raises(ValueError, match="already exists"):
        release.prepare("0.1.0")
    assert not any(args[1] in {"switch", "commit", "push"} for args in calls)


@pytest.mark.parametrize(
    "coauthor",
    ["Codex <noreply@openai.com>", "Claude Sonnet <noreply@anthropic.com>"],
)
def test_preparation_creates_reviewable_pr_but_never_tags(tmp_path, monkeypatch, coauthor):
    release_checkout(tmp_path, monkeypatch)
    (tmp_path / release.VERSION_FILE).write_text('__version__ = "0.1.0.dev0"\n')
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 0.1.0 (unreleased)\n\nOverview.\n")
    calls = []
    messages = []

    def fake_run(*args, input_text=None):
        calls.append(args)
        if args[1] == "rev-parse":
            return "main-sha"
        if args[:3] == ("gh", "repo", "view"):
            return "exploreomni/omniframes"
        if args[:2] == ("gh", "api"):
            return '{"body": "## Features\\n* New API"}'
        if args[1] == "commit":
            messages.append(input_text)
        if args[:3] == ("gh", "pr", "create"):
            body = Path(args[args.index("--body-file") + 1]).read_text()
            assert coauthor in body
        return ""

    monkeypatch.setattr(release, "run", fake_run)
    release.prepare("0.1.0", coauthor)
    assert release.read_version((tmp_path / release.VERSION_FILE).read_text()) == "0.1.0"
    assert "New API" in release.release_notes((tmp_path / "CHANGELOG.md").read_text(), "0.1.0")
    assert messages == [f"chore(release): prepare 0.1.0\n\nCo-Authored-By: {coauthor}\n"]
    assert any(args[:3] == ("gh", "pr", "create") for args in calls)
    assert all(args[2].startswith("--") for args in calls if args[:2] == ("git", "tag"))
    assert ("git", "push", "--set-upstream", "origin", "codex/release-0.1.0") in calls
