"""前話者追従の全モード廃止後の flush 配線テスト.

実測（transcripts/2026-07-14_1729, GT81発話）で声紋一致92%(n=13)に対し
相槌追従28%(n=32)・低信頼追従0%(n=2)と、3人の掛け合いでは追従がランダム未満で
害だった＋相槌は聞き手が打つ＝直前話者とは別人が多い、というユーザー判断により、
前話者追従は全モードで帰属根拠から外した（2026-07-14）。抑制は
VoiceProfiles._classify に一本化され、かつて RecvLoop.flush にあった
ハイブリッド限定の _HYBRID_UNTRUSTED_FOLLOW_KINDS による二重の抑制は撤去した。
（注: その後の再設計 b8897ef で、_classify は照合不成立時に「ラベル継続」＝
同一STTラベルの声紋確定済み対応先を維持する。これは会話の直前話者への追従とは
別物。相槌の最終的な未確定化は flush 側の constrain 入力規則が担う。
docs/design/attribution_logic_review_2026-07.md D4 で本 docstring を実装に同期。）ここでは flush 側の残る責務を固定する:
- tracker が返す未確定はどのモードでもそのまま records に載る（追従復活なし）
- 未確定発話に pyannote クラスタが重なればクラスタ帰属が勝つ（ハイブリッドの
  優先度「声紋一致 > pyannoteクラスタ > 未確定」は tracker 側の廃止だけで成立）
- 未確定は stt_fallback として参加者化されない
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

    def nearest_cluster(self, raw_cluster):
        return None

    def rename_confirmed(self, old, new):
        return None   # rekey からの伝搬（review P2）。フェイクでは何もしない


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


def test_tracker_unsure_passes_through_hybrid(tmp_path):
    """ハイブリッド時、trackerの相槌未確定はそのまま未確定として記録される.

    追従の抑制は VoiceProfiles 側で完結しており、flush 側で追従を復活させる
    経路が無いこと（stt_fallback で参加者化されないこと）を固定する。
    """
    state = _make_state(tmp_path, tracker=_Tracker("相槌未確定", "?"),
                        namer=_Namer(), resolver=_SttResolver())

    _flush(state)

    rec = state.records[-1]
    assert rec["speaker"] == "?"
    assert "speaker_source" not in rec       # stt_fallback として参加者化されない


def test_tracker_unsure_passes_through_non_hybrid(tmp_path):
    """非ハイブリッド（cluster_namer無し）でも同じく未確定のまま記録される.

    旧仕様（追従がハイブリッド限定で抑制され、非ハイブリッドでは「松井」に
    追従）はユーザー判断で全モード廃止（実測正解率28%・3人会話）。
    """
    state = _make_state(tmp_path, tracker=_Tracker("相槌未確定", "?"),
                        namer=None, resolver=_SttResolver())

    _flush(state)

    rec = state.records[-1]
    assert rec["speaker"] == "?"
    assert "speaker_source" not in rec


def test_unsure_defers_to_pyannote_cluster(tmp_path):
    """未確定に倒した発話にpyannoteクラスタが重なれば、クラスタ帰属が勝つ."""
    state = _make_state(tmp_path, tracker=_Tracker("相槌未確定", "?"),
                        namer=_Namer(), resolver=_PyannoteResolver("SPEAKER_00"))

    _flush(state)

    rec = state.records[-1]
    assert rec["speaker"] == "@diar:1"              # クラスタ由来キーに帰属
    assert rec["speaker_source"] == "pyannote"


def test_accumulating_kind_keeps_label_placeholder(tmp_path):
    """蓄積中（声紋判定前の長い発話）の #ラベルは従来どおり記録される（回帰）.

    #ラベルへの継続はSTTラベルベースの機構（遡及リネームの土台）であって
    「追従」ではないため、廃止の対象外。
    """
    state = _make_state(tmp_path, tracker=_Tracker("蓄積中", "#1"),
                        namer=_Namer(), resolver=_SttResolver())

    _flush(state)

    assert state.records[-1]["speaker"] == "#1"
