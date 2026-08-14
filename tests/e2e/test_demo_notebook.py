"""``examples/demo.ipynb`` is executed headlessly on every run — a notebook that rots, fails.

The notebook is the definition-of-done artifact for 0.1: it walks the whole feature surface
against the in-process fake, printing ``explain()`` after every step.  It is also the live
integration demo (set ``OMNI_BASE_URL`` / ``OMNI_API_KEY`` and the same cells run against a real
org), which is exactly why it has to keep working offline: the offline run is the only one CI
can make.

The notebook is committed with its **outputs cleared**; this test supplies them, in memory, and
throws them away.  It executes with ``examples/`` as the working directory — where a kernel
opening the file would start — and the live-branch environment variables are unset for the
duration, because a credentialed developer's machine must not change what CI checks.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import nbformat
import pytest
from nbclient import NotebookClient

from tests.fakes import DEFAULT_TOKEN

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "demo.ipynb"

#: The environment that decides which backend the setup cell picks (see the notebook's first
#: markdown cell).  Cleared so this test always exercises the offline path.
LIVE_ENV = ("OMNI_BASE_URL", "OMNI_API_KEY", "OMNI_MODEL", "OMNI_TOPIC", "OMNI_BENCH_SCHEMA")

#: Fragments that must appear in the captured output.  Each pins a claim the notebook makes about
#: *where* work happens — the tier story is the whole point of the artifact, and a notebook that
#: still ran while silently falling back to local execution is the regression this catches.
EXPECTED_FRAGMENTS = (
    "in-process FakeOmniAPI",  # the setup cell chose the offline backend
    "tier 1 · semantic",  # the governed query
    "tier 2 · sql",  # the mixed aggregation, collapsed to one OmniSQL statement
    "tier 2 · raw SQL job",  # read.sql
    "Local [arrow compute]",  # the UDF fallback, named
    # No "align-join": docs/SQLTIER.md §4 collapses a mixed aggregation into a single tier-2
    # statement, so the M3 decomposition no longer appears in the notebook.  It stays covered by
    # tests/unit/test_splitter.py and tests/differential/test_tiers.py.
    "(none — fully pushed down)",  # at least one frame compiles to one query
    "rewriteSql: false",  # CONTRACT_NOTES §3.4's silent failure, avoided
    "having:",  # the measure filter reached the server
)


@pytest.fixture(scope="module")
def executed() -> Iterator[nbformat.NotebookNode]:
    """The notebook, run start to finish against the fake."""
    # nbformat ships no annotations, so the return type is Any and mypy wants the call flagged.
    notebook: nbformat.NotebookNode = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]

    previous = {name: os.environ.pop(name, None) for name in LIVE_ENV}
    try:
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            # Collect every failure below rather than raising on the first one: the report is
            # far more useful when it shows which cell broke and what the rest did.
            allow_errors=True,
            # Where a kernel opening examples/demo.ipynb would start.
            resources={"metadata": {"path": str(NOTEBOOK.parent)}},
        ).execute()
        yield notebook
    finally:
        for name, value in previous.items():
            if value is not None:
                os.environ[name] = value


def outputs_of(notebook: nbformat.NotebookNode) -> list[dict[str, Any]]:
    return [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", ())
    ]


def captured_text(notebook: nbformat.NotebookNode) -> str:
    chunks: list[str] = []
    for output in outputs_of(notebook):
        if output.get("output_type") == "stream":
            chunks.append(output.get("text", ""))
        elif output.get("output_type") in {"execute_result", "display_data"}:
            chunks.append(str(output.get("data", {}).get("text/plain", "")))
    return "\n".join(chunks)


def test_the_notebook_is_committed_with_its_outputs_cleared() -> None:
    """A committed output is a stale answer waiting to be believed, and a diff nobody reads."""
    notebook = json.loads(NOTEBOOK.read_text("utf-8"))
    dirty = [
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]

    assert dirty == [], (
        f"cells {dirty} carry outputs; clear them before committing "
        "(jupyter nbconvert --clear-output --inplace examples/demo.ipynb)"
    )


def test_the_notebook_never_shows_an_api_key() -> None:
    """The demo is read over shoulders and pasted into tickets; a key must not be in it."""
    assert "omni_osk_" not in NOTEBOOK.read_text("utf-8")


def test_the_notebook_runs_start_to_finish_without_an_error(
    executed: nbformat.NotebookNode,
) -> None:
    errors = [output for output in outputs_of(executed) if output.get("output_type") == "error"]

    assert errors == [], "\n\n".join(
        f"{error.get('ename')}: {error.get('evalue')}\n" + "\n".join(error.get("traceback", ()))
        for error in errors
    )


@pytest.mark.parametrize("fragment", EXPECTED_FRAGMENTS)
def test_the_tier_story_is_visible_in_the_output(
    executed: nbformat.NotebookNode, fragment: str
) -> None:
    assert fragment in captured_text(executed)


def test_every_code_cell_produced_output(executed: nbformat.NotebookNode) -> None:
    """A silent cell in a demo is dead weight — every one of them is there to show something."""
    silent = [
        index
        for index, cell in enumerate(executed.cells)
        if cell.cell_type == "code" and not cell.get("outputs")
    ]

    assert silent == []


def test_no_api_key_reaches_the_rendered_output(executed: nbformat.NotebookNode) -> None:
    """The fake's token has a real key's shape; nothing may render one."""
    text = captured_text(executed)

    assert DEFAULT_TOKEN not in text
    assert "omni_osk_" not in text
