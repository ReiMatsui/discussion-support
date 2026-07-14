"""ハイブリッド構成での前話者追従の抑制（flush配線）のテスト.

実測（transcripts/2026-07-14_1729, GT81発話）で声紋一致92%(n=13)に対し
相槌追従28%(n=32)・低信頼追従0%(n=2)と、3人の掛け合いでは追従が害だったため、
cluster_namer 有効時は「相槌追従/低信頼追従」を帰属根拠として信用せず、
「声紋一致 > pyannoteクラスタ(名寄せ済み) > 未確定」の優先度に倒す。
cluster_namer 無し（Soniox単独・pyannote単独）の追従挙動は不変であること
（1対1会話で有効な設計を壊さない）も回帰として固定する。
"""
from __future__ import annotations

import datetime
from types import SimpleNamespace

from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState


class _Args:
    lang = "ja"
    vp_debug = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _Tracker:
    """kind と返すキーを固定できる最小フェイク声紋トラッカー."""

    def __init__(self, kind: str, key: str) -> None:
        self.last = {"kind": kind, "label": "1"}
        self._key = key

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return self._key


class _SttResolver:
    """diarizationの重なりが無く、常にSTTフォールバックになる SpeakerResolver."""

    def resolve(self, **kwargs):
        return SimpleNamespace(source="stt", speaker=kwargs["stt_speaker"],
                               confidence=0.0, reason="fallback_stt_label")


class _PyannoteResolver:
    """常にpyannoteクラスタを返すフェイク SpeakerResolver."""

    def __init__(self, speaker: str) -> None:
        self._speaker = speaker

    def resolve(self, **kwargs):
        return SimpleNamespace(source="pyannote", speaker=self._speaker,
                               confidence=0.9, reason="diarization_overlap")


class _Namer:
    """名寄せ・声紋名前付けは常に不成立のフェイク ClusterVoiceNamer."""

    def __init__(self) -> None:
        self.tracker = SimpleNamespace(dedupe=0.5)
        self.last_match = None

    def observe(self, raw_cluster, wav, *, overlapped=False):
        return None

    def canonical_cluster(self, raw_cluster):
        return raw_cluster

    def nearest_cluster(self, raw_cluster, exclude=None):
        return None


def _make_state(tmp_path, *, tracker, namer, resolver):
    state = SessionState(  # type: ignore[no-untyped-call]
        args=SimpleNamespace(diarization_max_speakers=None),
        started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp_path / "o.md"),
        html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"),
        turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"),
        tracker=tracker,
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 5)
    state.cluster_namer = namer  # type: ignore[assignment]
    state.speaker_resolver = resolver  # type: ignore[assignment]
    return state


def _flush(state, *, text="これは検証用の発言です", ms=1000, end=3000):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = text
    loop.cur_ms = ms
    loop.cur_end = end
    loop.flush()  # type: ignore[no-untyped-call]
    return state


def test_hybrid_backchannel_follow_not_adopted(tmp_path):
    """ハイブリッド時、相槌追従の結果は採用されず未確定に倒す."""
    state = _make_state(tmp_path, tracker=_Tracker("相槌追従", "松井"),
                        namer=_Namer(), resolver=_SttResolver())

    _flush(state)

    rec = state.records[-1]
    assert rec["speaker"] == "?"                    # 追従の「松井」を信用しない
    assert rec["speaker_source"] == "hybrid_follow_suppressed"
    assert rec["speaker_reason"] == "untrusted_previous_speaker_follow"


def test_hybrid_low_confidence_follow_not_adopted(tmp_path):
    """ハイブリッド時、低信頼追従（匿名prevへの弱い継続）も未確定に倒す."""
    state = _make_state(tmp_path, tracker=_Tracker("低信頼追従", "人物1"),
                        namer=_Namer(), resolver=_SttResolver())

    _flush(state)

    assert state.records[-1]["speaker"] == "?"


def test_hybrid_follow_defers_to_pyannote_cluster(tmp_path):
    """ハイブリッド時、追従発話にpyannoteクラスタが重なればクラスタ帰属が勝つ."""
    state = _make_state(tmp_path, tracker=_Tracker("相槌追従", "松井"),
                        namer=_Namer(), resolver=_PyannoteResolver("SPEAKER_00"))

    _flush(state)

    rec = state.records[-1]
    assert rec["speaker"] == "@diar:1"              # クラスタ由来キーに帰属
    assert rec["speaker_source"] == "pyannote"      # 抑制の痕跡は上書きされる


def test_hybrid_accumulating_kind_not_suppressed(tmp_path):
    """蓄積中（声紋判定前の長い発話）は抑制対象外＝従来どおり（対象は追従2種のみ）."""
    state = _make_state(tmp_path, tracker=_Tracker("蓄積中", "#1"),
                        namer=_Namer(), resolver=_SttResolver())

    _flush(state)

    assert state.records[-1]["speaker"] == "#1"


def test_non_hybrid_backchannel_follow_still_adopted(tmp_path):
    """cluster_namer無し（非ハイブリッド）では従来どおり追従が効く（回帰）."""
    state = _make_state(tmp_path, tracker=_Tracker("相槌追従", "松井"),
                        namer=None, resolver=_SttResolver())

    _flush(state)

    rec = state.records[-1]
    assert rec["speaker"] == "松井"                 # 1対1会話等で有効な既存設計は不変
    assert "speaker_source" not in rec
