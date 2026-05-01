"""Pytest 共通設定。"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """テストでは実際の OpenAI / Tavily キーが漏れないよう既定値を上書きする。"""

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    # Settings シングルトンを破棄
    from das import settings as settings_module

    settings_module.reset_settings()
    yield
    settings_module.reset_settings()


@pytest.fixture
def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
