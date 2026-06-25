"""話者分離イベントの統合・評価ロジック.

STT は「何を言ったか」、diarization は「誰がいつ話したか」を返す。
このモジュールは両者を疎結合に保ち、Soniox の高精度テキストを残したまま
pyannote/Speechmatics/Deepgram などの話者分離結果を差し替えられるようにする。
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
        voiceprint_high_confidence: float = 0.70,
    ) -> None:
        self.diarization_min_overlap = diarization_min_overlap
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
        overlaps: defaultdict[str, int] = defaultdict(int)
        sources: dict[str, str] = {}
        for event in diarization_events:
            seg = event.closed(fallback_end_ms=utterance.end_ms)
            if seg is None:
                continue
            ov = utterance.overlap_ms(seg)
            if ov <= 0:
                continue
            overlaps[event.speaker] += ov
            sources.setdefault(event.speaker, event.source)

        if overlaps:
            speaker, overlap = max(overlaps.items(), key=lambda item: item[1])
            ratio = overlap / duration
            if ratio >= self.diarization_min_overlap:
                return ResolvedSpeaker(
                    speaker=speaker,
                    confidence=min(1.0, ratio),
                    source=sources.get(speaker, "diarization"),
                    reason=f"diarization_overlap_{ratio:.2f}",
                )

        return ResolvedSpeaker(
            speaker=stt_speaker,
            confidence=0.0,
            source="stt",
            reason="fallback_stt_label",
        )


@dataclass(frozen=True)
class DiarizationScore:
    """話者分離評価の集計結果."""

    total_ms: int
    correct_ms: int
    confusion_ms: int
    missed_ms: int
    false_alarm_ms: int

    @property
    def accuracy(self) -> float:
        return self.correct_ms / self.total_ms if self.total_ms else 0.0

    @property
    def confusion_rate(self) -> float:
        return self.confusion_ms / self.total_ms if self.total_ms else 0.0

    @property
    def missed_rate(self) -> float:
        return self.missed_ms / self.total_ms if self.total_ms else 0.0

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarm_ms / self.total_ms if self.total_ms else 0.0


def _best_speaker_mapping(
    reference: list[tuple[DiarizationEvent, TimeSegment]],
    hypothesis: list[tuple[DiarizationEvent, TimeSegment]],
) -> dict[str, str]:
    """hypothesis speaker -> reference speaker の貪欲対応表を作る.

    providerごとに SPEAKER_00 / A / Guest-1 のようなラベル体系が違うため、
    評価時だけ最大重なりで正解ラベルへ写像する。
    """
    pair_overlap: Counter[tuple[str, str]] = Counter()
    for hyp, hseg in hypothesis:
        for ref, rseg in reference:
            ov = hseg.overlap_ms(rseg)
            if ov > 0:
                pair_overlap[(hyp.speaker, ref.speaker)] += ov

    mapping: dict[str, str] = {}
    used_ref: set[str] = set()
    for (hyp_sp, ref_sp), _ov in pair_overlap.most_common():
        if hyp_sp in mapping or ref_sp in used_ref:
            continue
        mapping[hyp_sp] = ref_sp
        used_ref.add(ref_sp)
    return mapping


def score_diarization(
    reference: Iterable[DiarizationEvent],
    hypothesis: Iterable[DiarizationEvent],
) -> DiarizationScore:
    """時間重なりベースで話者分離を評価する.

    1ms単位で厳密なDERを計算するのではなく、発話単位の意思決定に十分な
    「どれだけ正しい話者時間を覆えたか」を測る軽量スコア。
    """
    refs: list[tuple[DiarizationEvent, TimeSegment]] = []
    for ref in reference:
        seg = ref.closed()
        if seg is not None:
            refs.append((ref, seg))
    hyps: list[tuple[DiarizationEvent, TimeSegment]] = []
    for hyp in hypothesis:
        seg = hyp.closed()
        if seg is not None:
            hyps.append((hyp, seg))
    total_ms = sum(seg.duration_ms() for _ref, seg in refs)
    mapping = _best_speaker_mapping(refs, hyps)

    correct_ms = 0
    confusion_ms = 0
    covered_ref_ms: Counter[int] = Counter()
    used_hyp_ms: Counter[int] = Counter()

    for hi, (hyp, hseg) in enumerate(hyps):
        mapped = mapping.get(hyp.speaker, hyp.speaker)
        for ri, (ref, rseg) in enumerate(refs):
            ov = hseg.overlap_ms(rseg)
            if ov <= 0:
                continue
            covered_ref_ms[ri] += ov
            used_hyp_ms[hi] += ov
            if mapped == ref.speaker:
                correct_ms += ov
            else:
                confusion_ms += ov

    missed_ms = 0
    for ri, (_ref, rseg) in enumerate(refs):
        missed_ms += max(0, rseg.duration_ms() - covered_ref_ms[ri])

    false_alarm_ms = 0
    for hi, (_hyp, hseg) in enumerate(hyps):
        false_alarm_ms += max(0, hseg.duration_ms() - used_hyp_ms[hi])

    return DiarizationScore(
        total_ms=total_ms,
        correct_ms=correct_ms,
        confusion_ms=confusion_ms,
        missed_ms=missed_ms,
        false_alarm_ms=false_alarm_ms,
    )
