"""SessionState の脱線検出シード（Fix 8）のユニットテスト."""
from __future__ import annotations

import datetime

from das.asr.live._session_state import SessionState


def _make_state() -> SessionState:
    return SessionState(
        args=object(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
    )


def test_seed_topic_adds_when_empty():
    s = _make_state()
    s.seed_topic("AIツール導入の是非")
    assert [t["topic"] for t in s.topics] == ["AIツール導入の是非"]
    assert s.topics[0]["speaker"] == "議題"


def test_delivery_event_includes_timing(tmp_path):
    """delivery イベントに speak_start_latency_ms などの timing を残せる（Phase4観測）."""
    import json
    s = _make_state()
    s.interventions_path = str(tmp_path / "o.interventions.jsonl")
    s.add_facilitator_delivery_event("本題に戻しましょう",
                                     timing={"speak_start_latency_ms": 420.0})
    with open(s.interventions_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    assert lines[-1]["type"] == "delivery"
    assert lines[-1]["timing"]["speak_start_latency_ms"] == 420.0


def test_delivery_event_omits_timing_when_absent(tmp_path):
    """timing 未指定なら delivery イベントに timing キーを付けない（既存互換）."""
    import json
    s = _make_state()
    s.interventions_path = str(tmp_path / "o.interventions.jsonl")
    s.add_facilitator_delivery_event("本題に戻しましょう")
    with open(s.interventions_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    assert "timing" not in lines[-1]


def test_seed_topic_noop_for_empty_input():
    s = _make_state()
    s.seed_topic("")
    s.seed_topic(None)
    assert s.topics == []


def test_seed_topic_does_not_override_existing():
    s = _make_state()
    s.topics.append({"topic": "既存論点", "speaker": "話者1"})
    s.seed_topic("議題テーマ")
    assert [t["topic"] for t in s.topics] == ["既存論点"]


# --- M6: partial 受信で沈黙タイマーを更新 ---------------------------------

def test_show_partial_updates_silence_timer_on_new_text():
    """非空の新しい partial を受けたら沈黙タイマーを更新する（発話中を沈黙と誤認しない）."""
    s = _make_state()
    s._last_utt_time[0] = 0.0
    s.show_partial("#1", "この論点はもう少し")
    assert s._last_utt_time[0] > 0.0


def test_show_partial_does_not_update_on_repeated_text():
    """同一 partial の再送では更新しない（沈黙が永久に 0 に張り付くのを防ぐ）."""
    s = _make_state()
    s.show_partial("#1", "同じ文字列")
    s._last_utt_time[0] = 0.0            # 変化検出のため一旦戻す
    s.show_partial("#1", "同じ文字列")   # 同一 partial の再送
    assert s._last_utt_time[0] == 0.0


def test_show_partial_empty_does_not_update():
    """空（strip 後空）の partial では更新しない."""
    s = _make_state()
    s._last_utt_time[0] = 0.0
    s.show_partial("#1", "   ")
    assert s._last_utt_time[0] == 0.0


def test_show_partial_records_change_time():
    """F3: partial が変化したら変化時刻(_last_partial_change)を記録する."""
    s = _make_state()
    s._last_partial_change = 0.0
    s.show_partial("#1", "喋っている途中")
    assert s._last_partial_change > 0.0


# --- F3: アクティブな partial をフロア占有として扱う ----------------------

def test_effective_silence_is_zero_while_active_partial():
    """partial 非空かつ直近更新中は「フロア占有」= 沈黙 0 を返す."""
    import time as _t

    from das.asr.live._workers import _effective_silence
    s = _make_state()
    last = [0.0]                # 実際には大きな沈黙が経過している状態
    now = _t.monotonic()
    s.partial_text = "まだ喋っている途中で"
    s._last_partial_change = now
    assert _effective_silence(s, now, last) == 0.0


def test_effective_silence_normal_when_no_partial():
    """partial が空なら従来どおり now - last_utt_time を返す."""
    import time as _t

    from das.asr.live._workers import _effective_silence
    s = _make_state()
    last = [0.0]
    now = _t.monotonic()
    s.partial_text = ""
    assert _effective_silence(s, now, last) == now - last[0]


def test_effective_silence_ignores_stale_partial():
    """partial が10秒以上変化していなければ stale として無視し、通常の沈黙を返す."""
    import time as _t

    from das.asr.live._workers import _effective_silence
    s = _make_state()
    last = [0.0]
    now = _t.monotonic()
    s.partial_text = "クリアされずに固着した partial"
    s._last_partial_change = now - 11.0  # 10秒超前
    assert _effective_silence(s, now, last) == now - last[0]
