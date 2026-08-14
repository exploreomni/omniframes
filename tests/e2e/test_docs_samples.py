"""Every ``python`` code block in the user-facing docs is executed, against the fake.

A sample that no longer runs is worse than no sample: it is a confident, wrong answer to "how do
I use this?".  So the README, the docs site's landing page, the quickstart, the mental-model page
and the offline-testing guide are all treated as one executable script per file — the blocks run
in document order, sharing a namespace, exactly as a reader working down the page would type
them.

The samples are written for a real org (``host("acme.omni.co")``, ``api_key_from_env()``, the
model ``ecommerce``), so this module supplies the org: ``SessionBuilder.get_or_create`` is
patched to hand back a session wired to an in-process :class:`FakeOmniAPI` serving the bench
model **under the name the docs use**.  A sample that builds its own transport — the
offline-testing guide's, which is *about* the fake — is left alone and runs verbatim.

To exclude a block, put ``<!-- no-run -->`` on the line before its opening fence.  Nothing does
today; the escape hatch exists so an intentionally non-runnable snippet can say so out loud
rather than quietly rotting.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from omniframes import OmniSession
from omniframes.session import API_KEY_ENV, SessionBuilder
from omniframes.transport import HttpTransport
from tests.fakes import DEFAULT_TOKEN, FakeOmniAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://bench.example.omni.co"

#: The model name the user-facing samples use.  The fake serves the bench model under it, so the
#: docs can read the way a real org does instead of naming a test fixture.
DOCS_MODEL_NAME = "ecommerce"

#: Every file whose ```python blocks are executed, in the order a reader meets them.
DOC_FILES = (
    "README.md",
    "docs/index.md",
    "docs/quickstart.md",
    "docs/mental-model.md",
    "docs/offline-testing.md",
)

_FENCE = re.compile(r"^```python\s*$")
_CLOSE = re.compile(r"^```\s*$")
_SKIP = "<!-- no-run -->"


@dataclass(frozen=True)
class Sample:
    """One fenced block, with enough identity to name it in a failure."""

    path: str
    line: int
    source: str

    def __str__(self) -> str:  # pragma: no cover - pytest id only
        return f"{self.path}:{self.line}"


def extract(path: Path) -> list[Sample]:
    """The ``python`` blocks of one document, in order, minus any marked ``<!-- no-run -->``."""
    lines = path.read_text("utf-8").splitlines()
    name = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else path.name
    samples: list[Sample] = []
    index = 0
    while index < len(lines):
        if not _FENCE.match(lines[index]):
            index += 1
            continue
        opened_at = index
        index += 1
        body: list[str] = []
        while index < len(lines) and not _CLOSE.match(lines[index]):
            body.append(lines[index])
            index += 1
        index += 1
        if opened_at > 0 and _SKIP in lines[opened_at - 1]:
            continue
        samples.append(Sample(path=name, line=opened_at + 1, source="\n".join(body) + "\n"))
    return samples


def fake_session(handler: FakeOmniAPI) -> OmniSession:
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)
    transport = HttpTransport(
        base_url=BASE_URL, api_key=DEFAULT_TOKEN, client=client, sleep=lambda _: None
    )
    return OmniSession.builder.base_url(BASE_URL).transport(transport).get_or_create()


@pytest.fixture
def docs_org(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOmniAPI]:
    """Make ``builder...get_or_create()`` return a session onto the fake, under the docs' names."""
    handler = FakeOmniAPI(model_name=DOCS_MODEL_NAME)
    original = SessionBuilder.get_or_create

    def patched(self: SessionBuilder) -> OmniSession:
        # A sample that injected its own transport is testing the transport seam itself — the
        # offline-testing guide's is exactly that — so it must run untouched.
        if self._transport is not None:
            return original(self)
        return fake_session(handler)

    monkeypatch.setattr(SessionBuilder, "get_or_create", patched)
    monkeypatch.setenv(API_KEY_ENV, DEFAULT_TOKEN)
    try:
        yield handler
    finally:
        handler.close()


@pytest.mark.parametrize("document", DOC_FILES)
def test_every_python_sample_in_the_docs_runs(document: str, docs_org: FakeOmniAPI) -> None:
    """One namespace per file: the blocks of a page build on each other, as a reader does."""
    del docs_org
    path = REPO_ROOT / document
    samples = extract(path)

    assert samples, f"{document} has no ```python blocks — has it moved?"

    namespace: dict[str, object] = {"__name__": "__docs__"}
    for sample in samples:
        # Scoped: docs/mental-model.md turns TruncationWarning into an error to show the knob,
        # and that must not escape into the rest of the suite.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                exec(compile(sample.source, str(sample), "exec"), namespace)
            except Exception as error:  # pragma: no cover - the message is the point
                pytest.fail(
                    f"the code sample at {sample} raised "
                    f"{type(error).__name__}: {error}\n\n{sample.source}"
                )


def test_the_harness_honors_the_opt_out_marker(tmp_path: Path) -> None:
    """The escape hatch has to work before anyone needs it in anger."""
    document = tmp_path / "sample.md"
    document.write_text(
        "text\n\n```python\nx = 1\n```\n\ntext\n\n<!-- no-run -->\n```python\nboom\n```\n",
        "utf-8",
    )

    assert [sample.source for sample in extract(document)] == ["x = 1\n"]
