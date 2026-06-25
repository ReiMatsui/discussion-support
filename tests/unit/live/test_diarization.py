"""話者分離の統合・評価ロジックのテスト."""
from __future__ import annotations

from das.asr.live._diarization import (
    DiarizationEvent,
    SpeakerResolver,
    TimeSegment,
    score_diarization,
)


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
    assert got.confidence == 0.7
    assert got.reason == "diarization_overlap_0.70"


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
