"""Resolve session credentials through one optional notebook secret provider.

Provider selection is independent of User-Agent telemetry. Only existing notebook context
is inspected automatically; importing an installed SDK does not establish a notebook runtime.
See docs/quickstart.md for the published provider APIs and runtime-selection precedents.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from omniframes.errors import CompileError

# Environment names shared with the official Omni SDK (CONTRACT_NOTES.md §1).
API_KEY_ENV = "OMNI_API_KEY"
BASE_URL_ENV = "OMNI_BASE_URL"

_PROVIDERS = frozenset({"auto", "colab", "databricks", "snowflake", "snowflake-legacy"})
_LABELS = {
    "colab": "Google Colab",
    "databricks": "Databricks",
    "snowflake": "Snowflake",
    "snowflake-legacy": "legacy Snowflake notebooks",
}
_SETUP = {
    "colab": "check Colab's Secrets panel and enable Notebook access",
    "databricks": "check the configured scope, key, and secret READ permission",
    "snowflake": "check the attached secret identifier, permissions, and external access integration",
    "snowflake-legacy": "check the notebook's secret alias, permissions, and external access integration",
}


@dataclass(frozen=True, repr=False)
class SecretConfig:
    """Names and provider options, never secret values. Construction performs no I/O."""

    provider: str | None = "auto"
    scope: str | None = None
    api_key_name: str = API_KEY_ENV
    base_url_name: str = BASE_URL_ENV

    def __post_init__(self) -> None:
        if self.provider is not None and (
            not isinstance(self.provider, str) or self.provider not in _PROVIDERS
        ):
            raise CompileError(
                "secrets() provider must be 'auto', 'colab', 'databricks', 'snowflake', "
                "'snowflake-legacy', or None (disabled)"
            )
        for option, value in (
            ("api_key_name", self.api_key_name),
            ("base_url_name", self.base_url_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CompileError(f"secrets() {option} must be a non-empty string")
        if self.scope is not None:
            if not isinstance(self.scope, str) or not self.scope.strip():
                raise CompileError("secrets() scope must be a non-empty string")
            if self.provider not in {"auto", "databricks"}:
                raise CompileError("secrets() scope is only supported for Databricks")


def _notebook_dbutils() -> Any:
    """Use the notebook's existing utilities without constructing an SDK/Spark client.

    Databricks publishes the user_ns lookup in its Connect documentation; its SDK also uses
    ns_table['user_global']. Merely installing/importing the Databricks SDK is insufficient.
    """
    ipython = sys.modules.get("IPython")
    if ipython is None:
        return None
    try:
        shell = ipython.get_ipython()
        if shell is None:
            return None
        namespaces = getattr(shell, "ns_table", {})
        global_namespace = (
            namespaces.get("user_global", {}) if isinstance(namespaces, Mapping) else {}
        )
        for namespace in (getattr(shell, "user_ns", {}), global_namespace):
            if isinstance(namespace, Mapping):
                dbutils = namespace.get("dbutils")
                if callable(getattr(getattr(dbutils, "secrets", None), "get", None)):
                    return dbutils
    except Exception:
        # Context discovery is optional and must not expose notebook exception text.
        return None
    return None


def _failure_reason(provider: str, error: Exception) -> str:
    """Classify known provider errors without rendering their potentially sensitive text."""
    name = type(error).__name__
    if isinstance(error, TimeoutError) or name in {"TimeoutException", "DeadlineExceeded"}:
        if provider == "colab":
            return "secret lookup timed out; run from a connected Colab UI"
        return "secret lookup timed out; check that the notebook runtime is connected"
    if isinstance(error, PermissionError) or name in {"NotebookAccessError", "PermissionDenied"}:
        return "secret access was denied"
    if provider == "colab" and name == "SecretNotFoundError":
        return "secret was not found"
    if provider == "databricks":
        # The SDK exposes error_code; classic JVM wrappers have no equally stable contract.
        try:
            code = getattr(error, "error_code", None)
        except Exception:
            code = None
        if name in {"ResourceDoesNotExist", "NotFound"} or (
            isinstance(code, str) and code == "RESOURCE_DOES_NOT_EXIST"
        ):
            return "secret or scope was not found"
        if isinstance(code, str) and code == "PERMISSION_DENIED":
            return "secret access was denied"
        if name == "Unauthenticated" or (isinstance(code, str) and code == "UNAUTHENTICATED"):
            return "notebook authentication is unavailable"
    if provider == "snowflake" and isinstance(error, ValueError):
        # Snowpark deliberately does not distinguish absent secrets from unauthorized ones.
        return "secret was not found or is not accessible"
    if provider == "snowflake-legacy" and isinstance(error, KeyError):
        return "secret alias was not found"
    if isinstance(error, ImportError | NotImplementedError | FileNotFoundError):
        return "secret service or notebook secret configuration is unavailable"
    return "secret lookup failed"


@dataclass(frozen=True, repr=False)
class _SecretReader:
    provider: str
    read: Callable[[str], object]

    def get(self, name: str) -> str:
        label = _LABELS[self.provider]
        try:
            value = self.read(name)
        except Exception as error:
            reason = _failure_reason(self.provider, error)
            raise CompileError(
                f"{label} secret {name}: {reason}; {_SETUP[self.provider]}, "
                "or configure the Omni host/API key explicitly or through environment variables."
            ) from None
        if not isinstance(value, str) or not value.strip():
            raise CompileError(
                f"{label} secret {name} must contain a non-empty string; {_SETUP[self.provider]}."
            )
        return value


def _select_reader(config: SecretConfig) -> _SecretReader | None:
    provider = config.provider
    if provider is None:
        return None
    dbutils = None
    if provider in {"auto", "databricks"}:
        dbutils = _notebook_dbutils()
    if provider == "auto":
        # Google publishes this loaded-module pattern in its notebook examples. Enterprise
        # is excluded because its secret capability is not the consumer Colab Secrets UI.
        colab = (
            sys.modules.get("google.colab") is not None
            and os.environ.get("VERTEX_PRODUCT", "").strip().upper() != "COLAB_ENTERPRISE"
        )
        if colab and dbutils is not None:
            raise CompileError(
                "multiple notebook secret providers are available; select one with .secrets(...)"
            )
        if colab:
            provider = "colab"
        elif dbutils is not None:
            provider = "databricks"
        else:
            return None
    if config.scope is not None and provider != "databricks":
        raise CompileError("secrets() scope is only supported for Databricks; select it explicitly")
    if provider == "databricks":
        if not config.scope:
            raise CompileError(
                "Databricks secret lookup requires a scope: call .secrets('databricks', scope='...')"
            )
        if dbutils is None:
            raise CompileError(
                "Databricks notebook dbutils is unavailable; run in a Databricks notebook "
                "or configure the Omni host/API key through environment variables."
            )
        # Resolve attributes inside the reader so runtime failures are sanitized consistently.
        return _SecretReader(
            provider, lambda name: dbutils.secrets.get(scope=config.scope, key=name)
        )

    # Only the selected provider is imported, and only when credentials are actually missing.
    try:
        if provider == "colab":
            userdata = importlib.import_module("google.colab.userdata")
            return _SecretReader(provider, lambda name: userdata.get(name))
        if provider == "snowflake":
            snowpark = importlib.import_module("snowflake.snowpark.secrets")
            return _SecretReader(provider, lambda name: snowpark.get_generic_secret_string(name))
        streamlit = importlib.import_module("streamlit")
        return _SecretReader(provider, lambda name: streamlit.secrets[name])
    except Exception:
        raise CompileError(
            f"{_LABELS[provider]} secrets are unavailable in this Python environment; "
            "check the selected provider and its runtime packages, or use environment variables."
        ) from None


def resolve_credentials(
    base_url: str | None, api_key: str | None, config: SecretConfig
) -> tuple[str | None, str | None]:
    """Resolve each missing field, selecting at most one notebook provider per build."""
    base_url = base_url or os.environ.get(BASE_URL_ENV)
    api_key = api_key or os.environ.get(API_KEY_ENV)
    if base_url and api_key:
        return base_url, api_key
    reader = _select_reader(config)
    if reader is not None:
        base_url = base_url or reader.get(config.base_url_name)
        api_key = api_key or reader.get(config.api_key_name)
    return base_url, api_key
