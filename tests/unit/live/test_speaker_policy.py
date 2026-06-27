"""リアルタイム介入に渡す話者名の安全化テスト."""
from __future__ import annotations

import datetime

from das.asr.live._session_state import SessionState
from das.asr.live._speaker_policy import (
    intervention_records,
    intervention_speaker_name,
    is_intervention_signal,
    is_reliable_human_speaker,
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
    assert intervention_speaker_name(s, rec) == "人物1"


def test_backchannel_is_not_intervention_signal() -> None:
    rec = {"speaker": "?", "text": "はい", "bc": True}

    assert is_intervention_signal(rec) is False
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
