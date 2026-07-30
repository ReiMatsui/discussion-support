"""論点抽出・脱線判定のLLM呼び出しパラメータ構築（Fix 9）のテスト."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from contextlib import contextmanager

from das.asr.live import _bootstrap as bootstrap
from das.asr.live._bootstrap import _build_chat_params


@contextmanager
def _noop(x):
    yield x


def _resp(content: str):
    """urlopen の戻り（with 文で使えて .read() が本文を返すもの）を作る."""
    body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    return _noop(io.BytesIO(body))


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


# -- 400 の原因を出す（2026-07-29） -----------------------------------


def test_http_error_prints_the_response_body(monkeypatch, capsys):
    """400 の本文を出す.

    OpenAI は「未知のモデル名」「そのモデルでは使えないパラメータ」といった
    理由を本文に書いてくる。捨てると `400 Bad Request` だけが延々と流れ、
    LLM機能が全滅していても何が悪いのか特定できない（実会話で発生した）。
    """
    import urllib.error

    def _boom(*a, **k):
        raise urllib.error.HTTPError(
            "http://x", 400, "Bad Request", {},
            io.BytesIO(b'{"error":{"message":"unknown model gpt-5.4-mini"}}'))

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    got = bootstrap._post_chat_json({"model": "gpt-5.4-mini"}, "k",
                                    timeout=1, label="topic")
    out = capsys.readouterr().out
    assert got is None
    assert "unknown model gpt-5.4-mini" in out, "本文が出ていない"
    assert "gpt-5.4-mini" in out                # どのモデルで起きたか
    assert "400" in out


def test_http_error_without_a_body_still_reports(monkeypatch, capsys):
    """本文が読めないときも理由（reason）だけは出す."""
    def _boom(*a, **k):
        raise urllib.error.HTTPError("http://x", 429, "Too Many Requests",
                                     {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert bootstrap._post_chat_json({"model": "m"}, "k", timeout=1,
                                     label="drift") is None
    assert "429" in capsys.readouterr().out


def test_unsupported_reasoning_effort_falls_back_and_is_remembered(
        monkeypatch, capsys):
    """reasoning_effort が拒否されたら次の値で再試行し、モデルに対して覚える.

    どの値が使えるかはモデルの世代で変わる。gpt-5.4-mini は 'minimal' を拒否
    した（使えるのは none/low/…）。世代ごとの対応表を持つと次の更新で同じ
    ことが起きるので、弾かれたら弱いほうから順に試す。
    """
    bootstrap._EFFORT.clear()
    seen: list[str | None] = []

    def _fake(*a, **k):
        body = json.loads(a[0].data)
        seen.append(body.get("reasoning_effort"))
        if body.get("reasoning_effort") == "minimal":
            raise urllib.error.HTTPError(
                "http://x", 400, "Bad Request", {},
                io.BytesIO(b'{"error":{"message":"Unsupported value:'
                           b' \'reasoning_effort\' does not support'
                           b' \'minimal\' with this model.",'
                           b'"param":"reasoning_effort"}}'))
        return _resp('{"topics": []}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    params = _build_chat_params("gpt-5.4-mini", "x", max_out=100,
                                temperature=0.0)
    assert params["reasoning_effort"] == "minimal"
    got = bootstrap._post_chat_json(params, "k", timeout=1, label="topic")

    assert got == {"topics": []}, "再試行が成功していない"
    assert seen == ["minimal", "none"], f"試した順が違う: {seen}"
    assert "'none' で再試行" in capsys.readouterr().out
    # 次からは最初から通る値で組む（余分な呼び出しは1回だけ）
    assert _build_chat_params("gpt-5.4-mini", "x", max_out=100,
                              temperature=0.0)["reasoning_effort"] == "none"
    bootstrap._EFFORT.clear()


def test_other_400_is_not_retried(monkeypatch, capsys):
    """reasoning_effort と無関係な 400 は再試行しない（無駄打ちを増やさない）."""
    bootstrap._EFFORT.clear()
    calls: list[int] = []

    def _fake(*a, **k):
        calls.append(1)
        raise urllib.error.HTTPError(
            "http://x", 400, "Bad Request", {},
            io.BytesIO(b'{"error":{"message":"unknown model","param":"model"}}'))

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    params = _build_chat_params("gpt-5.4-mini", "x", max_out=100,
                               temperature=0.0)
    assert bootstrap._post_chat_json(params, "k", timeout=1, label="topic") is None
    assert len(calls) == 1
    assert "unknown model" in capsys.readouterr().out
