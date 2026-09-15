"""Prepare a release PR, or validate a tagged checkout (stdlib only; requires git and gh)."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = Path("src/omniframes/_version.py")
VERSION_RE = re.compile(
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:(a|b|rc)(0|[1-9]\d*))?", re.ASCII
)


def version_key(version: str) -> tuple[int, int, int, int, int]:
    """Accept canonical release versions, excluding dev/local/post versions."""
    match = VERSION_RE.fullmatch(version)
    if match is None:
        raise ValueError(f"Invalid release version: {version!r}; use X.Y.Z or X.Y.Z[a|b|rc]N")
    major, minor, patch, stage, number = match.groups()
    return (
        int(major),
        int(minor),
        int(patch),
        {"a": 0, "b": 1, "rc": 2, None: 3}[stage],
        int(number or 0),
    )


def run(*args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        args, cwd=ROOT, input=input_text, text=True, check=True, stdout=subprocess.PIPE
    ).stdout.strip()


def read_version(source: str) -> str:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError("No literal __version__ found")


def previous_tag(tags: list[str], version: str) -> str | None:
    """Use a stable baseline so final releases include changes from their prereleases."""
    target = version_key(version)
    candidates = []
    for tag in tags:
        if not tag.startswith("v") or not VERSION_RE.fullmatch(tag[1:]):
            continue
        key = version_key(tag[1:])
        if key >= target:
            raise ValueError(f"Release must be newer than existing tag {tag}")
        if key[3] == 3:
            candidates.append((key, tag))
    return max(candidates)[1] if candidates else None


def release_notes(changelog: str, version: str) -> str:
    version_key(version)
    pattern = rf"^## {re.escape(version)} \((\d{{4}}-\d{{2}}-\d{{2}})\)\n(.*?)(?=^## |\Z)"
    sections = list(re.finditer(pattern, changelog, flags=re.MULTILINE | re.DOTALL))
    if len(sections) != 1:
        raise ValueError(f"Expected exactly one finalized changelog section for {version}")
    dt.date.fromisoformat(sections[0][1])
    notes = sections[0][2].strip()
    if not notes:
        raise ValueError("Release notes must not be empty")
    return notes + "\n"


def update_changelog(changelog: str, version: str, generated: str, date: str) -> str:
    if not changelog.startswith("# Changelog\n"):
        raise ValueError("Expected # Changelog heading")
    if re.search(rf"^## {re.escape(version)} \(\d", changelog, re.MULTILINE):
        raise ValueError(f"Changelog already contains release {version}")
    # Preserve the curated first-release overview, including when preparing an RC first.
    initial = re.search(
        r"^## 0\.1\.0 \(unreleased\)\n(.*?)(?=^## |\Z)", changelog, re.MULTILINE | re.DOTALL
    )
    overview = ""
    if initial and version_key(version)[:3] == (0, 1, 0):
        overview = initial[1].strip() + "\n\n"
        if version == "0.1.0":
            changelog = changelog[: initial.start()] + changelog[initial.end() :]
    # GitHub emits level-two headings; nest them below our version heading so
    # extraction cannot mistake "What's Changed" for another release.
    generated = re.sub(
        r"^(#{1,5}) ",
        lambda match: "#" * max(3, len(match[1]) + 1) + " ",
        generated.strip(),
        flags=re.MULTILINE,
    )
    section = f"## {version} ({date})\n\n{overview}{generated}\n\n"
    return "# Changelog\n\n" + section + changelog.removeprefix("# Changelog\n").lstrip()


def validate(tag: str, notes_output: Path | None) -> None:
    version = read_version((ROOT / VERSION_FILE).read_text())
    version_key(version)
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match package version v{version}")
    run("git", "merge-base", "--is-ancestor", "HEAD", "origin/main")
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse", f"refs/tags/{tag}^{{commit}}"):
        raise ValueError("Checkout does not match release tag")
    notes = release_notes((ROOT / "CHANGELOG.md").read_text(), version)
    if notes_output:
        notes_output.write_text(notes)
    print(f"Validated {tag}")


def prepare(version: str, coauthor: str | None = None) -> None:
    version_key(version)
    if run("git", "status", "--porcelain"):
        raise ValueError("Start with a clean working tree")
    run("gh", "auth", "status")
    run("git", "fetch", "origin", "main", "--tags")
    if run("git", "rev-parse", "HEAD") != run("git", "rev-parse", "origin/main"):
        raise ValueError("Start from the current origin/main commit")
    baseline = previous_tag(run("git", "tag", "--merged", "HEAD").splitlines(), version)
    branch = f"codex/release-{version}"
    if run("git", "branch", "--list", branch) or run(
        "git", "ls-remote", "--heads", "origin", branch
    ):
        raise ValueError(f"Branch {branch} already exists; continue its existing release PR")
    if run("git", "tag", "--list", f"v{version}"):
        raise ValueError(f"Tag v{version} already exists")
    repo = run("gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")
    command = [
        "gh",
        "api",
        f"repos/{repo}/releases/generate-notes",
        "-f",
        f"tag_name=v{version}",
        "-f",
        f"target_commitish={run('git', 'rev-parse', 'HEAD')}",
        "-f",
        "configuration_file_path=.github/release.yml",
    ]
    if baseline:
        command += ["-f", f"previous_tag_name={baseline}"]
    generated = json.loads(run(*command))["body"]
    changelog = update_changelog(
        (ROOT / "CHANGELOG.md").read_text(),
        version,
        generated,
        dt.datetime.now(dt.UTC).date().isoformat(),
    )
    source = (ROOT / VERSION_FILE).read_text()
    old = read_version(source)
    # The initial development version is the only non-release value accepted here.
    current = (0, 1, 0, -1, 0) if old == "0.1.0.dev0" else version_key(old)
    if current >= version_key(version):
        raise ValueError(f"Version must be newer than current package version {old}")
    updated, count = re.subn(
        r'^__version__ = "[^"]+"$', f'__version__ = "{version}"', source, flags=re.MULTILINE
    )
    if count != 1:
        raise ValueError("Expected one __version__ assignment")
    run("git", "switch", "-c", branch)
    (ROOT / VERSION_FILE).write_text(updated)
    (ROOT / "CHANGELOG.md").write_text(changelog)
    run("git", "add", str(VERSION_FILE), "CHANGELOG.md")
    message = f"chore(release): prepare {version}\n"
    if coauthor:
        message += f"\nCo-Authored-By: {coauthor}\n"
    run("git", "commit", "-F", "-", input_text=message)
    run("git", "push", "--set-upstream", "origin", branch)
    with tempfile.TemporaryDirectory() as directory:
        body = Path(directory) / "body.md"
        body.write_text(
            f"Prepare {version}: update the package version and changelog.\n\n"
            "Review the generated notes, add migration guidance where needed, and merge after CI passes. "
            f"Then explicitly tag the merged commit as `v{version}` and push that tag to publish. "
            "Merging this PR does not publish.\n\n"
            "Generated by the Omniframes release preparation command.\n"
            + (f"\nPrepared with {coauthor}.\n" if coauthor else "")
        )
        print(
            run(
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                f"chore(release): prepare {version}",
                "--body-file",
                str(body),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser(
        "prepare", help="Commit, push, and open a release PR; never tag"
    )
    preparation.add_argument("version")
    preparation.add_argument(
        "--coauthor", help="Agent identity for commit/PR attribution, when applicable"
    )
    check = commands.add_parser("validate", help="Validate a tagged checkout without publishing")
    check.add_argument("tag")
    check.add_argument("--notes-output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.version, args.coauthor)
        else:
            validate(args.tag, args.notes_output)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Release preparation/validation failed: {error}\n")


if __name__ == "__main__":
    main()
