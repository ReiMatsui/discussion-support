"""flush配線のクラスタ間名寄せ（登録者ゼロ対策）の統合テスト.

設計: docs/design/handoff_2026-07-14_unregistered_speakers.md §3 参照。
cluster_namer 有効時のみ働く経路（名寄せ成立時の遡及統合、canonicalへの
キー集約、max-speakers超過時の最近傍統合）を、フェイクの namer/resolver で検証する。
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
    """ClusterVoiceNamer の名寄せ結果だけを模したフェイク."""

    def __init__(self, aliases=None, nearest=None) -> None:
        self._aliases = dict(aliases or {})
        self._nearest = nearest

    def observe(self, raw_cluster, wav, *, overlapped=False):
        return None   # 声紋名前付けは常に未確定（名寄せ経路の検証に集中する）

    def canonical_cluster(self, raw_cluster):
        return self._aliases.get(raw_cluster, raw_cluster)

    def nearest_cluster(self, raw_cluster, exclude=None):
        return self._nearest


class _FakePyannoteProvider:
    name = "pyannote"

    def drain_events(self):
        return []


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


def test_merged_cluster_rekeys_absorbed_key_into_canonical(tmp_path):
    """名寄せ成立時、吸収側の @diar:N が rekey で canonical のキーへ遡及統合される."""
    namer = _Namer(aliases={"pyannote:SPEAKER_01": "pyannote:SPEAKER_00"})
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    state.diarization_speaker_keys = {"pyannote:SPEAKER_00": "@diar:1",
                                      "pyannote:SPEAKER_01": "@diar:2"}
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:2", "text": "昔の発言"}]

    _flush(state)

    assert state.records[0]["speaker"] == "@diar:1"    # 過去レコードの遡及統合
    assert state.records[-1]["speaker"] == "@diar:1"   # 新しい発話も canonical のキー
    assert "pyannote:SPEAKER_01" not in state.diarization_speaker_keys


def test_merged_cluster_without_canonical_key_reuses_absorbed_key(tmp_path):
    """canonical が未キーで吸収側にキーがある場合、そのキーを canonical へ付け替える."""
    namer = _Namer(aliases={"pyannote:SPEAKER_01": "pyannote:SPEAKER_00"})
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    state.diarization_speaker_keys = {"pyannote:SPEAKER_01": "@diar:1"}

    _flush(state)

    assert state.records[-1]["speaker"] == "@diar:1"
    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "@diar:1"}


def test_merged_cluster_issues_key_under_canonical_when_both_unkeyed(tmp_path):
    """両方未キーなら canonical の source/speaker でキーを発行する（pending集約）."""
    namer = _Namer(aliases={"pyannote:SPEAKER_01": "pyannote:SPEAKER_00"})
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")
    # provider未設定＝ヒステリシスなし → 即時発行（canonical側で発行されることを確認）

    _flush(state)

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_00": "@diar:1"}
    assert state.records[-1]["speaker"] == "@diar:1"


def test_unmerged_cluster_over_max_speakers_joins_nearest(tmp_path):
    """max-speakers到達後の未キークラスタは、最近傍クラスタの既存キーへ統合される."""
    namer = _Namer(nearest="pyannote:SPEAKER_00")
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_05", max_speakers=1)
    state.diarization_speaker_keys = {"pyannote:SPEAKER_00": "@diar:1"}
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")   # 匿名ラベル（参加者A）を確定させ、スロット1を占有

    _flush(state)

    assert state.records[-1]["speaker"] == "@diar:1"   # 新規参加者を作らず統合
    assert "pyannote:SPEAKER_05" not in state.diarization_speaker_keys


def test_unmerged_cluster_over_max_speakers_without_nearest_falls_back(tmp_path):
    """最近傍が取れなければ従来どおり（constrainで未確定へ落ちる＝既存挙動）."""
    namer = _Namer(nearest=None)
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_05", max_speakers=1)
    state.diarization_speaker_keys = {"pyannote:SPEAKER_00": "@diar:1"}
    state.records = [{"ms": 0, "end_ms": 500, "speaker": "@diar:1", "text": "既存参加者"}]
    state.disp_name("@diar:1")

    _flush(state)

    assert state.records[-1]["speaker"] == "?"   # UNSURE_SPEAKER（上限超過の既存挙動）


def test_unmerged_cluster_under_limit_issues_new_key(tmp_path):
    """名寄せ不成立でも上限未達なら従来どおり新規キーを発行する."""
    namer = _Namer()
    state = _make_state(tmp_path, namer=namer, speaker="SPEAKER_01")

    _flush(state)

    assert state.diarization_speaker_keys == {"pyannote:SPEAKER_01": "@diar:1"}
    assert state.records[-1]["speaker"] == "@diar:1"
