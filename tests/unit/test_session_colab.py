"""Colab credential resolution without a Colab installation or a live notebook."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from collections.abc import Iterator
from functools import partial
from types import ModuleType
from typing import Any
from unittest.mock import Mock, call

import httpx
import pytest
from tests.fakes import DEFAULT_TOKEN, FakeOmniAPI

from omniframes import OmniSession
from omniframes import session as session_module
from omniframes._user_agent import _detect_hosted_runtime
from omniframes.errors import CompileError
from omniframes.session import API_KEY_ENV, BASE_URL_ENV
from omniframes.transport import HttpTransport

BASE_URL = "https://bench.omniapp.co"


class SecretNotFoundError(Exception):
    pass


class NotebookAccessError(Exception):
    pass


class TimeoutException(Exception):
    pass


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(
        session_module,
        "_detect_hosted_runtime",
        partial(_detect_hosted_runtime, path_exists=lambda _path: False),
    )


@pytest.fixture
def colab(monkeypatch: pytest.MonkeyPatch) -> Mock:
    monkeypatch.setenv("COLAB_RELEASE_TAG", "test-runtime")
    get = Mock(side_effect={BASE_URL_ENV: BASE_URL, API_KEY_ENV: DEFAULT_TOKEN}.__getitem__)
    userdata = ModuleType("google.colab.userdata")
    userdata.__dict__.update(
        get=get,
        SecretNotFoundError=SecretNotFoundError,
        NotebookAccessError=NotebookAccessError,
        TimeoutException=TimeoutException,
    )
    monkeypatch.setitem(sys.modules, "google.colab.userdata", userdata)
    return get


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeOmniAPI]:
    fake = FakeOmniAPI()
    with httpx.Client(transport=httpx.MockTransport(fake)) as client:

        def transport(**options: Any) -> HttpTransport:
            return HttpTransport(**options, client=client)

        monkeypatch.setattr(session_module, "HttpTransport", transport)
        yield fake
    fake.close()


def test_colab_secrets_authenticate_without_eager_omni_calls(colab: Mock, api: FakeOmniAPI) -> None:
    builder = OmniSession.builder
    colab.assert_not_called()

    with builder.get_or_create() as session:
        assert api.requests == []
        assert BASE_URL in repr(session)
        assert DEFAULT_TOKEN not in repr(session)
        assert DEFAULT_TOKEN not in repr(builder)
        assert API_KEY_ENV not in os.environ
        assert BASE_URL_ENV not in os.environ
        session.verify()

    assert api.paths == ["GET /api/v1/whoami"]
    assert colab.call_args_list == [call(BASE_URL_ENV), call(API_KEY_ENV)]


@pytest.mark.parametrize("source", ["explicit", "environment", "helpers"])
def test_configured_credentials_skip_colab_imports(
    source: str, colab: Mock, api: FakeOmniAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(BASE_URL_ENV, BASE_URL)
    monkeypatch.setenv(API_KEY_ENV, DEFAULT_TOKEN)
    builder = OmniSession.builder
    if source == "explicit":
        builder.host(BASE_URL).api_key(DEFAULT_TOKEN)
        monkeypatch.setenv(BASE_URL_ENV, "https://wrong.omniapp.co")
        monkeypatch.setenv(API_KEY_ENV, "wrong-test-token")
    elif source == "helpers":
        builder.base_url_from_env().api_key_from_env()
    importer = Mock(side_effect=AssertionError("must not import Colab"))
    monkeypatch.setattr(importlib, "import_module", importer)

    with builder.get_or_create() as session:
        assert BASE_URL in repr(session)
        session.verify()

    importer.assert_not_called()
    colab.assert_not_called()


@pytest.mark.parametrize("source", ["explicit", "environment"])
@pytest.mark.parametrize("configured", [BASE_URL_ENV, API_KEY_ENV])
def test_only_missing_credentials_use_colab(
    source: str,
    configured: str,
    colab: Mock,
    api: FakeOmniAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = OmniSession.builder
    value = BASE_URL if configured == BASE_URL_ENV else DEFAULT_TOKEN
    if source == "environment":
        monkeypatch.setenv(configured, value)
    elif configured == BASE_URL_ENV:
        builder.host(value)
    else:
        builder.api_key(value)

    with builder.get_or_create() as session:
        session.verify()

    colab.assert_called_once_with(API_KEY_ENV if configured == BASE_URL_ENV else BASE_URL_ENV)


@pytest.mark.parametrize("custom_names", [False, True])
def test_from_env_helpers_fall_back_to_same_named_secrets(
    custom_names: bool, colab: Mock, api: FakeOmniAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_name = "MY_OMNI_HOST" if custom_names else BASE_URL_ENV
    key_name = "MY_OMNI_TOKEN" if custom_names else API_KEY_ENV
    colab.side_effect = {host_name: BASE_URL, key_name: DEFAULT_TOKEN}.__getitem__
    monkeypatch.setenv(host_name, "")
    monkeypatch.setenv(key_name, "")

    builder = OmniSession.builder.base_url_from_env(host_name).api_key_from_env(key_name)
    with builder.get_or_create() as session:
        session.verify()

    assert colab.call_args_list == [call(host_name), call(key_name)]
    assert DEFAULT_TOKEN not in repr(builder)


def test_injected_transport_skips_colab(colab: Mock, api: FakeOmniAPI) -> None:
    with httpx.Client(transport=httpx.MockTransport(api)) as client:
        transport = HttpTransport(BASE_URL, DEFAULT_TOKEN, client=client)
        with OmniSession.builder.transport(transport).get_or_create() as session:
            assert api.requests == []
            session.verify()

    colab.assert_not_called()


@pytest.mark.parametrize("runtime", ["local", "databricks", "colab-enterprise"])
def test_other_runtimes_do_not_import_colab(
    runtime: str, colab: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COLAB_RELEASE_TAG")
    if runtime == "databricks":
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "test-runtime")
    elif runtime == "colab-enterprise":
        monkeypatch.setenv("VERTEX_PRODUCT", "COLAB_ENTERPRISE")
    importer = Mock(side_effect=AssertionError("must not import Colab"))
    monkeypatch.setattr(importlib, "import_module", importer)

    with pytest.raises(CompileError, match="no Omni host configured"):
        OmniSession.builder.get_or_create()
    with pytest.raises(CompileError, match="no API key configured"):
        OmniSession.builder.host(BASE_URL).get_or_create()

    importer.assert_not_called()
    colab.assert_not_called()


@pytest.mark.parametrize(
    ("error", "remedy"),
    [
        (SecretNotFoundError, "add it"),
        (NotebookAccessError, "enable Notebook access"),
        (TimeoutException, "Colab UI"),
        (RuntimeError, "check Colab's Secrets panel"),
    ],
)
@pytest.mark.parametrize("helper", [False, True])
def test_colab_errors_are_actionable_and_do_not_leak_credentials(
    error: type[Exception], remedy: str, helper: bool, colab: Mock
) -> None:
    colab.side_effect = error(DEFAULT_TOKEN)
    builder = OmniSession.builder.host(BASE_URL)
    resolve = builder.api_key_from_env if helper else builder.get_or_create
    with pytest.raises(CompileError, match=remedy) as caught:
        resolve()

    assert API_KEY_ENV in str(caught.value)
    assert "environment" in str(caught.value)
    assert DEFAULT_TOKEN not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("value", ["", None, 123])
def test_empty_or_invalid_colab_secrets_are_rejected(value: object, colab: Mock) -> None:
    colab.side_effect = None
    colab.return_value = value

    with pytest.raises(CompileError, match="OMNI_API_KEY must contain a non-empty string"):
        OmniSession.builder.host(BASE_URL).get_or_create()


def test_unavailable_userdata_has_an_environment_remedy(
    colab: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "google.colab.userdata", None)

    with pytest.raises(CompileError, match="Colab secrets are unavailable; set OMNI_API_KEY"):
        OmniSession.builder.host(BASE_URL).get_or_create()

    colab.assert_not_called()


def test_userdata_without_exception_classes_still_sanitizes_errors(
    colab: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    userdata = sys.modules["google.colab.userdata"]
    for name in ("SecretNotFoundError", "NotebookAccessError", "TimeoutException"):
        monkeypatch.delattr(userdata, name)
    colab.side_effect = RuntimeError(DEFAULT_TOKEN)

    with pytest.raises(CompileError, match="check Colab's Secrets panel") as caught:
        OmniSession.builder.host(BASE_URL).get_or_create()

    assert DEFAULT_TOKEN not in "".join(traceback.format_exception(caught.value))
