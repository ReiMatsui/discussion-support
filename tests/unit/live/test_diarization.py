"""話者分離の統合・評価ロジックのテスト."""
from __future__ import annotations

from typing import Any, cast

import click

from das.asr.live._diarization import (
    DiarizationEvent,
    SpeakerResolver,
    TimeSegment,
    score_diarization,
)
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState


def test_resolver_prefers_high_confidence_voiceprint() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 3000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(900, 3100, "SPEAKER_02", "pyannote"),
        ],
        voiceprint_speaker="田中",
        voiceprint_confidence=0.92,
    )

    assert got.speaker == "田中"
    assert got.source == "voiceprint"
    assert got.reason == "voiceprint_high_confidence"


def test_resolver_uses_diarization_when_overlap_is_large() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 3000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(900, 2400, "SPEAKER_02", "pyannote"),
            DiarizationEvent(2400, 2800, "SPEAKER_03", "pyannote"),
        ],
    )

    assert got.speaker == "SPEAKER_02"
    assert got.source == "pyannote"
    assert got.confidence == 0.75
    assert got.reason == "diarization_overlap_0.75"


def test_resolver_falls_back_to_stt_when_all_signals_are_weak() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 3000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(1000, 1800, "SPEAKER_02", "pyannote"),
        ],
        voiceprint_speaker="田中",
        voiceprint_confidence=0.40,
    )

    assert got.speaker == "#1"
    assert got.source == "stt"
    assert got.reason == "fallback_stt_label"


def test_resolver_accepts_short_boundary_shifted_diarization_overlap() -> None:
    resolver = SpeakerResolver()

    got = resolver.resolve(
        utterance=TimeSegment(1000, 2000),
        stt_speaker="#1",
        diarization_events=[
            DiarizationEvent(750, 1300, "SPEAKER_02", "pyannote"),
        ],
    )

    assert got.speaker == "SPEAKER_02"
    assert got.source == "pyannote"
    assert got.reason == "diarization_overlap_0.55"


def test_score_diarization_maps_provider_labels_by_overlap() -> None:
    reference = [
        DiarizationEvent(0, 1000, "田中", "gold"),
        DiarizationEvent(1000, 2000, "佐藤", "gold"),
    ]
    hypothesis = [
        DiarizationEvent(0, 900, "SPEAKER_01", "pyannote"),
        DiarizationEvent(900, 2000, "SPEAKER_02", "pyannote"),
    ]

    score = score_diarization(reference, hypothesis)

    assert score.total_ms == 2000
    assert score.correct_ms == 1900
    assert score.confusion_ms == 100
    assert score.missed_ms == 0
    assert score.false_alarm_ms == 0
    assert score.accuracy == 0.95


def test_liveargs_and_cli_have_diarization_option() -> None:
    from das.asr.live import main
    from das.asr.live._bootstrap import LiveArgs

    assert LiveArgs().model == "stt-rt-v5"
    assert LiveArgs().soniox_endpoint is True
    assert LiveArgs().diarization == "none"
    assert LiveArgs(diarization="pyannote").diarization == "pyannote"
    assert LiveArgs(
        diarization="assemblyai",
        diarization_max_speakers=3,
    ).diarization_max_speakers == 3
    for param in main.params:
        if param.name == "diarization":
            choice_type = cast(click.Choice, param.type)
            assert set(choice_type.choices) == {"none", "pyannote", "assemblyai"}
            break
    else:
        raise AssertionError("diarization option not found")
    assert any(param.name == "soniox_endpoint" for param in main.params)


def test_recv_loop_uses_diarization_when_voiceprint_is_unavailable() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.diarization_events = [
        DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote"),
    ]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == [{
        "ms": 1000,
        "end_ms": 3000,
        "speaker": "@diar:1",
        "text": "これはテストです",
        "diarization_raw_speaker": "SPEAKER_00",
        "speaker_source": "pyannote",
        "speaker_confidence": 1.0,
        "speaker_reason": "diarization_overlap_1.00",
    }]
    assert state.disp_name(state.records[0]["speaker"]) == "参加者A"


def test_recv_loop_prefers_internal_voiceprint_over_external_diarization() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class Tracker:
        def __init__(self) -> None:
            self.last = {
                "kind": "合流",
                "label": "1",
                "name": "人物1",
                "rename": None,
            }

        def classify(self, *args: object, **kwargs: object) -> str:
            return "人物1"

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        tracker=Tracker(),  # type: ignore[arg-type]
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 3)
    state.diarization_events = [
        DiarizationEvent(900, 3100, "SPEAKER_00", "pyannote"),
    ]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == [{
        "ms": 1000,
        "end_ms": 3000,
        "speaker": "人物1",
        "text": "これはテストです",
        "speaker_source": "voiceprint",
        "speaker_confidence": 1.0,
        "speaker_reason": "voiceprint_high_confidence",
    }]


def test_recv_loop_normalizes_stt_label_when_diarization_is_enabled_but_unresolved() -> None:
    import datetime

    class Args:
        lang = "ja"
        vp_debug = False

    class Backend:
        def parse_message(self, raw: dict[str, Any], lang: str) -> dict[str, Any]:
            return raw

    class Provider:
        name = "fake"

        def start(self) -> None:
            pass

        def send_audio(self, pcm16k: bytes) -> None:
            pass

        def drain_events(self) -> list[DiarizationEvent]:
            return []

        def active_events(self) -> list[DiarizationEvent]:
            return []

        def close(self) -> None:
            pass

    state = SessionState(  # type: ignore[no-untyped-call]
        args=Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        serve=False,
        diarization_provider=Provider(),
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    loop = RecvLoop(state, Args(), Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これはテストです"
    loop.cur_ms = 1000
    loop.cur_end = 3000

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == [{
        "ms": 1000,
        "end_ms": 3000,
        "speaker": "@diar:1",
        "text": "これはテストです",
        "stt_raw_speaker": "#1",
        "speaker_source": "stt_fallback",
        "speaker_confidence": 0.0,
        "speaker_reason": "diarization_no_confident_overlap_stt_fallback",
    }]
    assert state.disp_name(state.records[0]["speaker"]) == "参加者A"
