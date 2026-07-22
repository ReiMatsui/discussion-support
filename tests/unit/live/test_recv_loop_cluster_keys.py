"""flush配線の匿名クラスタキー解決の統合テスト（ハイブリッド構成）.

未確定クラスタの @diar:N 発行・max-speakers 超過時の constrain・diag への
final_key / cluster_naming イベント記録を、フェイクの namer/resolver で検証する。
（かつてはクラスタ間名寄せ・最近傍統合の検証（test_recv_loop_cluster_merge.py）
だったが、機構の削除（handoff §18.9）に伴い、残る配線のテストとして再編。）
"""
from __future__ import annotations

import datetime
import json
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
    """flushの声紋前段を素通りさせる最小フェイク（kindは非高信頼の「蓄積中」）."""

    def __init__(self) -> None:
        self.last = {"kind": "蓄積中", "label": "1"}

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return "#1"


class _Resolver:
    """常に外部diarization由来の話者を返すフェイク SpeakerResolver."""

    def __init__(self, speaker: str, source: str = "pyannote") -> None:
        self._speaker = speaker
        self._source = source

    def resolve(self, **kwargs):
        return SimpleNamespace(source=self._source, speaker=self._speaker,
                               confidence=0.9, reason="diarization_overlap")


class _Namer:
    """声紋名前付けが常に不成立のフェイク ClusterVoiceNamer."""

    def __init__(self) -> None:
        self.last_match = None

    def observe(self, raw_cluster, wav, *, overlapped=False):
        return None

    def rename_confirmed(self, old, new):
        return None


def _make_state(tmp_path, *, namer, speaker, max_speakers=None):
    state = SessionState(  # type: ignore[no-untyped-call]
        args=SimpleNamespace(diarization_max_speakers=max_speakers),
        started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp_path / "o.md"),
        html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"),
        turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"),
        tracker=_Tracker(),
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 5)
    state.cluster_namer = namer  # type: ignore[assignment]
    state.speaker_resolver = _Resolver(speaker)  # type: ignore[assignment]
    return state


def _flush(state, *, text="これは検証用の発言です", ms=1000, end=3000):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = text
    loop.cur_ms = ms
    loop.cur_end = end
    loop.flush()  # type: ignore[no-untyped-call]
    return state


def test_unconfirmed_cluster_issues_new_key(tmp_path):
    """未確定クラスタは従来どおり @diar:N の新規キーが発行される."""
    state = _make_state(tmp_path, namer=_Namer(), speaker="SPEAKER_01")

    _flush(state)

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_01": "@diar:1"}
    assert state.records[-1]["speaker"] == "@diar:1"


def test_cluster_over_max_speakers_falls_to_unsure(tmp_path):
    """max-speakers到達後の新規クラスタは constrain で未確定に落ちる."""
    state = _make_state(tmp_path, namer=_Namer(), speaker="SPEAKER_05",
                        max_speakers=1)
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")   # @diar:1
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")   # 匿名ラベルを確定させ、スロット1を占有

    _flush(state)

    assert state.records[-1]["speaker"] == "?"   # UNSURE_SPEAKER（上限超過の既存挙動）


def test_diag_records_final_key_after_constrain(tmp_path):
    """diag に constrain 後の最終キー(final_key)が追記される（既存フィールドは不変）.

    従来は constrain 前の key しか記録されず、「resolver は正しいキーを選んだのに
    constrain で未確定に落ちた」事象（2026-07-14 実セッション）の切り分けが
    diag からできなかった。final_key は追加のみで、diag 消費側の互換性を保つ。
    """
    state = _make_state(tmp_path, namer=_Namer(), speaker="SPEAKER_05",
                        max_speakers=1)
    state.key_for_diarization_speaker("pyannote", "SPEAKER_00")   # @diar:1
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")   # スロット1を占有 → 新キーは constrain で未確定へ

    _flush(state)

    with open(state.diag_path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if '"final_key"' in line]
    assert events, "final_key を含む diag 行が無い"
    ev = events[-1]
    assert ev["key"] == "@diar:2"     # constrain 前（resolver の出力）は従来どおり
    assert ev["final_key"] == "?"     # constrain で未確定へ落ちたことが diag から読める
    assert state.records[-1]["speaker"] == ev["final_key"]   # records と一致


def test_cluster_namer_last_match_written_to_diag_once(tmp_path):
    """クラスタ照合イベントが diag に1行書かれ、消費されて重複出力しない（F6）."""
    namer = _Namer()
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    namer.last_match = {"kind": "確定見送り(低確信)", "cluster": "pyannote:SPEAKER_01",
                        "name": "人物1", "sim": 0.54, "need": 0.65}

    _flush(state)
    _flush(state)   # 2回目は last_match が消費済みなので書かれない

    with open(state.diag_path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f
                  if '"cluster_naming"' in line]
    assert len(events) == 1
    assert events[0]["type"] == "cluster_naming"
    assert events[0]["kind"] == "確定見送り(低確信)"
    assert namer.last_match is None
