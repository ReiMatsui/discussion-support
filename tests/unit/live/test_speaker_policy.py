"""リアルタイム介入に渡す話者名の安全化テスト."""
from __future__ import annotations

import datetime

from das.asr.live._session_state import SessionState
from das.asr.live._speaker_policy import (
    intervention_records,
    intervention_speaker_name,
    is_intervention_signal,
    is_reliable_human_speaker,
    is_triage_signal,
    triage_records,
)


def _state() -> SessionState:
    return SessionState(
        args=object(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )


def test_stt_fallback_is_not_treated_as_personal_speaker() -> None:
    s = _state()
    rec = {"speaker": "@diar:1", "speaker_source": "stt_fallback", "text": "発言"}

    assert is_reliable_human_speaker(rec) is False
    assert intervention_speaker_name(s, rec) == "発話者"


def test_named_or_auto_registered_person_is_reliable() -> None:
    s = _state()
    rec = {"speaker": "人物1", "speaker_source": "voiceprint", "text": "発言"}

    assert is_reliable_human_speaker(rec) is True
    assert intervention_speaker_name(s, rec) == "参加者A"


def test_backchannel_is_not_intervention_signal() -> None:
    rec = {"speaker": "?", "text": "はい", "bc": True}

    assert is_intervention_signal(rec) is False
    assert intervention_records([rec]) == []


def test_displayed_unsure_speaker_is_not_intervention_signal() -> None:
    rec = {"speaker": "未確定", "text": "外部音声らしい発話です"}

    assert is_intervention_signal(rec) is False
    assert is_reliable_human_speaker(rec) is False
    assert intervention_records([rec]) == []


def test_uncertain_long_utterance_is_kept_as_room_context() -> None:
    s = _state()
    rec = {
        "speaker": "@diar:1",
        "speaker_source": "stt_fallback",
        "text": "この論点はもう少し整理したほうが良いと思います",
    }

    assert is_intervention_signal(rec) is True
    assert is_reliable_human_speaker(rec) is False
    assert intervention_speaker_name(s, rec) == "発話者"
    assert intervention_records([rec]) == [rec]


def test_unsure_speaker_content_is_triage_signal_but_not_intervention() -> None:
    """未確定話者の中身のある発話は triage（呼びかけ検出）対象に含める（修正5）.

    呼びかけは話者同一性に依存しないため triage_records には入るが、fact/drift 等
    の材料になる intervention_records からは従来どおり除外される。"""
    for spk in ("?", "未確定"):
        rec = {"speaker": spk, "text": "AIさん、ここまで整理して"}
        assert is_triage_signal(rec) is True
        assert is_intervention_signal(rec) is False
        assert triage_records([rec]) == [rec]
        assert intervention_records([rec]) == []


def test_backchannel_is_not_triage_signal_even_when_unsure() -> None:
    """相槌は未確定話者でも triage 対象から除外する（bc / 相槌regex 両方）."""
    assert is_triage_signal({"speaker": "?", "text": "はい", "bc": True}) is False
    assert is_triage_signal({"speaker": "?", "text": "なるほど"}) is False
    assert triage_records([{"speaker": "?", "text": "うん"}]) == []


def test_empty_text_is_not_triage_signal() -> None:
    assert is_triage_signal({"speaker": "?", "text": "   "}) is False
