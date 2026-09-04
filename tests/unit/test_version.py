"""Release-version invariants."""

from importlib.metadata import version

import omniframes


def test_public_version_matches_installed_distribution_metadata() -> None:
    assert omniframes.__version__ == version("omniframes")
