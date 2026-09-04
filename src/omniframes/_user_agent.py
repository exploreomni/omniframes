"""Build the coarse, privacy-conscious identity sent with Omni HTTP requests.

Only compatibility information is included: the Omniframes release, Python language version,
Python implementation/version, and an allowlisted hosted-runtime label when one can be detected.
Raw environment values are deliberately never copied into the header: they can contain private
workspace identifiers, high-cardinality release details, or even invalid header characters.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Mapping
from typing import Final

from omniframes._version import __version__

__all__ = ["USER_AGENT", "build_user_agent"]

_TOKEN_UNSAFE_RE: Final = re.compile(r"[^!#$%&'*+.^_`|~0-9A-Za-z-]+")
_MAX_TOKEN_LENGTH: Final = 64

_DATABRICKS_VALUE_MARKERS: Final = (
    "DATABRICKS_RUNTIME_VERSION",
    "DATABRICKS_ENV_VERSION",
)
_DATABRICKS_APP_MARKERS: Final = (
    "DATABRICKS_APP_NAME",
    "DATABRICKS_APP_PORT",
    "DATABRICKS_WORKSPACE_ID",
)
_DATABRICKS_BOOLEAN_MARKERS: Final = (
    "IS_IN_DB_MODEL_SERVING_ENV",
    "IS_IN_DATABRICKS_MODEL_SERVING_ENV",
)
_COLAB_VALUE_MARKERS: Final = (
    "COLAB_RELEASE_TAG",
    "COLAB_BACKEND_VERSION",
    "COLAB_JUPYTER_TRANSPORT",
)
_TRUE_VALUES: Final = frozenset({"1", "on", "true", "yes"})
_DATABRICKS_VERSION_PATH: Final = "/databricks/DBR_VERSION"
_COLAB_HOSTNAME_PATH: Final = "/var/colab/hostname"


def _format_version(major: int, minor: int, micro: int, releaselevel: str, serial: int) -> str:
    """Preserve prerelease identity while keeping an RFC-compatible version token."""
    base = f"{major}.{minor}.{micro}"
    if releaselevel == "final":
        return base
    suffix = {"alpha": "a", "beta": "b", "candidate": "rc"}.get(releaselevel, f"-{releaselevel}-")
    return f"{base}{suffix}{serial}"


_PYTHON_VERSION: Final = _format_version(
    sys.version_info.major,
    sys.version_info.minor,
    sys.version_info.micro,
    sys.version_info.releaselevel,
    sys.version_info.serial,
)
_IMPLEMENTATION_VERSION: Final = _format_version(
    sys.implementation.version.major,
    sys.implementation.version.minor,
    sys.implementation.version.micro,
    sys.implementation.version.releaselevel,
    sys.implementation.version.serial,
)


def _token(value: str) -> str:
    """Return one RFC-compatible User-Agent product token."""
    normalized = _TOKEN_UNSAFE_RE.sub("-", value.strip()).strip("-")
    return normalized[:_MAX_TOKEN_LENGTH].rstrip("-") or "unknown"


def _is_true(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _TRUE_VALUES


def _has_value(environ: Mapping[str, str], names: tuple[str, ...]) -> bool:
    return any(bool(environ.get(name, "").strip()) for name in names)


def _path_exists(path: str, exists: Callable[[str], bool]) -> bool:
    try:
        return exists(path)
    except OSError:
        return False


def _detect_hosted_runtime(
    environ: Mapping[str, str], *, path_exists: Callable[[str], bool] = os.path.exists
) -> str | None:
    """Return an allowlisted runtime label from non-sensitive process markers.

    The result is observational telemetry, not a security boundary: callers can set environment
    variables themselves. In particular, generic Databricks connection variables such as
    ``DATABRICKS_HOST`` do not count because they are routinely set on developer machines.
    """
    databricks_app = all(bool(environ.get(name, "").strip()) for name in _DATABRICKS_APP_MARKERS)
    if (
        _has_value(environ, _DATABRICKS_VALUE_MARKERS)
        or any(_is_true(environ.get(name)) for name in _DATABRICKS_BOOLEAN_MARKERS)
        or databricks_app
        or _path_exists(_DATABRICKS_VERSION_PATH, path_exists)
    ):
        return "databricks"

    if environ.get("VERTEX_PRODUCT", "").strip().upper() == "COLAB_ENTERPRISE":
        return "google-colab-enterprise"
    if _has_value(environ, _COLAB_VALUE_MARKERS) or _path_exists(_COLAB_HOSTNAME_PATH, path_exists):
        return "google-colab"
    return None


def build_user_agent(
    *,
    library_version: str = __version__,
    python_version: str = _PYTHON_VERSION,
    implementation: str = sys.implementation.name,
    implementation_version: str = _IMPLEMENTATION_VERSION,
    environ: Mapping[str, str] | None = None,
    path_exists: Callable[[str], bool] = os.path.exists,
) -> str:
    """Build the User-Agent value for the current or an injected Python runtime."""
    runtime_environ = os.environ if environ is None else environ
    parts = [
        f"omniframes/{_token(library_version)}",
        f"python/{_token(python_version)}",
        f"{_token(implementation.lower())}/{_token(implementation_version)}",
    ]
    hosted_runtime = _detect_hosted_runtime(runtime_environ, path_exists=path_exists)
    if hosted_runtime is not None:
        parts.append(f"runtime/{hosted_runtime}")
    return " ".join(parts)


# Runtime markers are established before user code imports packages, so calculating once keeps
# every request in a process stable and avoids repeatedly inspecting the environment.
USER_AGENT: Final = build_user_agent()
