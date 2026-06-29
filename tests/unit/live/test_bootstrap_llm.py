"""論点抽出・脱線判定のLLM呼び出しパラメータ構築（Fix 9）のテスト."""
from __future__ import annotations

from das.asr.live import _bootstrap as bootstrap
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


def test_structured_output_schema_is_attached():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }

    p = _build_chat_params(
        "gpt-5-mini", "x", max_out=100, temperature=0.0,
        schema_name="sample_result", schema=schema)

    assert p["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "sample_result",
            "strict": True,
            "schema": schema,
        },
    }


def test_live_llm_helpers_use_structured_outputs(monkeypatch):
    seen: list[str] = []

    def fake_post(params, *_args, **_kwargs):
        fmt = params["response_format"]["json_schema"]
        assert fmt["strict"] is True
        seen.append(fmt["name"])
        if fmt["name"] == "topics_result":
            return {"topics": [{"topic": "論点A", "speaker": "A"}]}
        if fmt["name"] == "drift_result":
            return {"drift": False, "reason": ""}
        if fmt["name"] == "agenda_result":
            return {"agenda": "AI導入"}
        if fmt["name"] == "participation_result":
            return {"invite": True, "speaker": "B", "reason": "静か"}
        if fmt["name"] == "fact_correction_result":
            return {
                "should_correct": True,
                "confidence": "high",
                "claim": "対象の値は100",
                "correction": "対象の値は200です。",
                "reason": "値が違う",
            }
        raise AssertionError(fmt["name"])

    monkeypatch.setattr(bootstrap, "_post_chat_json", fake_post)

    assert bootstrap.extract_topics([{"speaker": "A", "text": "x"}], [], "key", "m") == [
        {"topic": "論点A", "speaker": "A"}
    ]
    assert bootstrap.check_drift(
        [{"speaker": "A", "text": "x"}], [{"topic": "T", "speaker": "議題"}],
        "key", "m") == {"drift": False, "reason": ""}
    assert bootstrap.detect_agenda([{"speaker": "A", "text": "x"}], "key", "m") == "AI導入"
    assert bootstrap.check_participation(
        [{"speaker": "B", "time_share": 0.1, "turns": 1, "silent_sec": 20}],
        [{"speaker": "A", "text": "x"}], "key", "m") == {
            "invite": True, "speaker": "B", "reason": "静か",
        }
    assert bootstrap.check_fact_correction(
        [{"speaker": "A", "text": "対象の値は100です"}], "key", "m"
    )["correction"] == "対象の値は200です。"
    assert seen == [
        "topics_result",
        "drift_result",
        "agenda_result",
        "participation_result",
        "fact_correction_result",
    ]
