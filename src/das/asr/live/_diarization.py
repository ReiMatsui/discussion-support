"""話者分離イベントの統合・評価ロジック.

STT は「何を言ったか」、diarization は「誰がいつ話したか」を返す。
このモジュールは両者を疎結合に保ち、Soniox の高精度テキストを残したまま
pyannote/Speechmatics/Deepgram などの話者分離結果を差し替えられるようにする。
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TimeSegment:
    """ミリ秒単位の半開区間 [start_ms, end_ms)."""

    start_ms: int
    end_ms: int

    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def overlap_ms(self, other: TimeSegment) -> int:
        return max(0, min(self.end_ms, other.end_ms) - max(self.start_ms, other.start_ms))


@dataclass(frozen=True)
class DiarizationEvent:
    """外部/内部 diarization provider が返す話者区間."""

    start_ms: int
    end_ms: int | None
    speaker: str
    source: str
    confidence: float | None = None

    def closed(self, fallback_end_ms: int | None = None) -> TimeSegment | None:
        end_ms = self.end_ms if self.end_ms is not None else fallback_end_ms
        if end_ms is None or end_ms <= self.start_ms:
            return None
        return TimeSegment(self.start_ms, end_ms)


class DiarizationProvider(Protocol):
    """リアルタイム話者分離 provider の最小インターフェース."""

    @property
    def name(self) -> str: ...

    def start(self) -> None: ...

    def send_audio(self, pcm16k: bytes) -> None: ...

    def drain_events(self) -> list[DiarizationEvent]: ...

    def active_events(self) -> list[DiarizationEvent]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ResolvedSpeaker:
    """1発話に対する最終話者判定."""

    speaker: str
    confidence: float
    source: str
    reason: str


class SpeakerResolver:
    """STT発話区間・外部diarization・声紋候補から最終話者を決める.

    判定の優先順位:
      1. 声紋が高信頼なら声紋を採用
      2. 外部diarizationが発話区間の過半を覆うならそれを採用
      3. どちらも弱い場合はSTTラベルにフォールバック
    """

    def __init__(
        self,
        *,
        diarization_min_overlap: float = 0.55,
        diarization_min_overlap_ms: int = 250,
        boundary_tolerance_ms: int = 250,
        voiceprint_high_confidence: float = 0.70,
    ) -> None:
        self.diarization_min_overlap = diarization_min_overlap
        self.diarization_min_overlap_ms = diarization_min_overlap_ms
        self.boundary_tolerance_ms = boundary_tolerance_ms
        self.voiceprint_high_confidence = voiceprint_high_confidence

    def resolve(
        self,
        *,
        utterance: TimeSegment,
        stt_speaker: str,
        diarization_events: Iterable[DiarizationEvent] = (),
        voiceprint_speaker: str | None = None,
        voiceprint_confidence: float | None = None,
    ) -> ResolvedSpeaker:
        if (
            voiceprint_speaker
            and voiceprint_confidence is not None
            and voiceprint_confidence >= self.voiceprint_high_confidence
        ):
            return ResolvedSpeaker(
                speaker=voiceprint_speaker,
                confidence=voiceprint_confidence,
                source="voiceprint",
                reason="voiceprint_high_confidence",
            )

        duration = max(utterance.duration_ms(), 1)
        padded = TimeSegment(
            max(0, utterance.start_ms - self.boundary_tolerance_ms),
            utterance.end_ms + self.boundary_tolerance_ms,
        )
        actual_overlaps: defaultdict[str, int] = defaultdict(int)
        padded_overlaps: defaultdict[str, int] = defaultdict(int)
        sources: dict[str, str] = {}
        for event in diarization_events:
            seg = event.closed(fallback_end_ms=utterance.end_ms)
            if seg is None:
                continue
            actual_ov = utterance.overlap_ms(seg)
            padded_ov = padded.overlap_ms(seg)
            if max(actual_ov, padded_ov) <= 0:
                continue
            actual_overlaps[event.speaker] += min(actual_ov, duration)
            padded_overlaps[event.speaker] += min(padded_ov, duration)
            sources.setdefault(event.speaker, event.source)

        if padded_overlaps:
            ranked = sorted(padded_overlaps.items(), key=lambda item: item[1], reverse=True)
            speaker, overlap = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0
            actual_overlap = actual_overlaps.get(speaker, 0)
            actual_ratio = actual_overlap / duration
            padded_ratio = overlap / duration
            enough_overlap = (
                actual_ratio >= self.diarization_min_overlap
                or (
                    duration <= 1500
                    and actual_overlap >= self.diarization_min_overlap_ms
                    and actual_ratio >= 0.25
                    and overlap >= second * 1.5
                )
            )
            if enough_overlap:
                return ResolvedSpeaker(
                    speaker=speaker,
                    confidence=min(1.0, max(actual_ratio, padded_ratio)),
                    source=sources.get(speaker, "diarization"),
                    reason=f"diarization_overlap_{max(actual_ratio, padded_ratio):.2f}",
                )

        return ResolvedSpeaker(
            speaker=stt_speaker,
            confidence=0.0,
            source="stt",
            reason="fallback_stt_label",
        )


def has_overlapping_speakers(
    events: Iterable[DiarizationEvent],
    start_ms: int,
    end_ms: int,
    *,
    min_ratio: float = 0.2,
) -> bool:
    """発話区間 [start_ms, end_ms) を、複数の話者クラスタが同時に占めているか判定する.

    pyannote+声紋照合のハイブリッド構成（docs/design/pyannote_live1_trial_2026-07-09.md
    §8.4/§9）で「重複発話区間は安全側で未確定にする」ために使う。混ざった声で
    クラスタ音声バッファを汚さないよう、蓄積前にこの判定でスキップする。
    ``min_ratio`` 以上を占める話者が2人以上いれば重複発話とみなす（相槌程度の
    薄い重なりは対象外）。
    """
    duration = max(end_ms - start_ms, 1)
    if duration <= 0:
        return False
    totals: defaultdict[str, int] = defaultdict(int)
    for event in events:
        seg = event.closed(fallback_end_ms=end_ms)
        if seg is None:
            continue
        ov = max(0, min(seg.end_ms, end_ms) - max(seg.start_ms, start_ms))
        if ov <= 0:
            continue
        totals[event.speaker] += ov
    substantial = [sp for sp, ov in totals.items() if ov / duration >= min_ratio]
    return len(substantial) >= 2
