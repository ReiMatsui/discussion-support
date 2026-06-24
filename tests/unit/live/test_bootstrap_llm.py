"""論点抽出・脱線判定のLLM呼び出しパラメータ構築（Fix 9）のテスト."""
from __future__ import annotations

from das.asr.live._bootstrap import _build_chat_params


def test_gpt5_uses_minimal_reasoning_and_completion_tokens():
    p = _build_chat_params("gpt-5-mini", "x", max_out=800, temperature=0.0)
    assert p["reasoning_effort"] == "minimal"
    assert p["max_completion_tokens"] == 800
    assert "temperature" not in p        # 推論モデルはtemperature不可
    assert "max_tokens" not in p


def test_o_series_uses_low_reasoning():
    p = _build_chat_params("o3-mini", "x", max_out=500, temperature=0.0)
    assert p["reasoning_effort"] == "low"   # o系は minimal 非対応
    assert p["max_completion_tokens"] == 500
    assert "temperature" not in p


def test_classic_model_uses_temperature_and_max_tokens():
    p = _build_chat_params("gpt-4o-mini", "x", max_out=512, temperature=0.3)
    assert p["temperature"] == 0.3
    assert p["max_tokens"] == 512
    assert "reasoning_effort" not in p
    assert "max_completion_tokens" not in p


def test_message_content_is_prompt():
    p = _build_chat_params("gpt-5-mini", "プロンプト本文", max_out=100, temperature=0.0)
    assert p["messages"][0]["content"] == "プロンプト本文"
    assert p["model"] == "gpt-5-mini"
