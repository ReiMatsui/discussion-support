"""Pytest 共通設定。"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """既定では実 OpenAI / Tavily キーをダミー値で覆い、誤って外部 API が呼ばれないようにする。

    ``@pytest.mark.integration`` 付きのテストでは元の環境変数を維持して
    実 API を呼べるようにする。
    """

    is_integration = request.node.get_closest_marker("integration") is not None
    if not is_integration:
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    from das import settings as settings_module

    settings_module.reset_settings()
    yield
    settings_module.reset_settings()


@pytest.fixture
def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
