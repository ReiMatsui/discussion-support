"""相槌へのクラスタ根拠つき帰属（handoff §15.4）の flush 配線テスト.

背景: 相槌は聞き手が打つ＝直前話者と別人が多く、Soniox は3人の相槌を同一
STTラベルに混ぜる（Chiba 2026-07-16_1551 実測: 未確定21件中18件が1秒未満の
相槌で、S1/S2/S3 の「うん」全部に同じラベル対応が付いた）ため、STTラベル・
声紋継続由来の帰属は従来どおり未確定に落とす。ただし pyannote クラスタは
声で束ねるため相槌でも分離でき（trial §8 未確定回収は pyannote 優位）、
クラスタ由来の根拠（確定名 / クラスタ匿名キー）がある場合だけ帰属を通す。
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
    def __init__(self, kind: str, key: str) -> None:
        self.last = {"kind": kind, "label": "1"}
        self._key = key

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return self._key


class _SttResolver:
    def resolve(self, **kwargs):
        return SimpleNamespace(source="stt", speaker=kwargs["stt_speaker"],
                               confidence=0.0, reason="fallback_stt_label")


class _PyannoteResolver:
    def __init__(self, speaker: str) -> None:
        self._speaker = speaker

    def resolve(self, **kwargs):
        return SimpleNamespace(source="pyannote", speaker=self._speaker,
                               confidence=0.9, reason="diarization_overlap")


class _Namer:
    """observe が固定値を返すフェイク ClusterVoiceNamer."""

    def __init__(self, name=None) -> None:
        self.tracker = SimpleNamespace(dedupe=0.5)
        self.merge_sim = 0.5
        self.last_match = None
        self._name = name

    def observe(self, raw_cluster, wav, *, overlapped=False):
        return self._name

    def canonical_cluster(self, raw_cluster):
        return raw_cluster

    def nearest_cluster(self, raw_cluster):
        return None

    def rename_confirmed(self, old, new):
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


def _flush_bc(state, *, text="うん。", ms=1000, end=1400):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = text
    loop.cur_ms = ms
    loop.cur_end = end
    loop.flush()  # type: ignore[no-untyped-call]
    return state.records[-1]


def test_backchannel_attributed_via_confirmed_cluster(tmp_path):
    """相槌でも、クラスタ確定名（声紋で裏付け済み）があれば帰属される."""
    state = _make_state(tmp_path, tracker=_Tracker("ラベル継続", "人物9"),
                        namer=_Namer(name="田中"),
                        resolver=_PyannoteResolver("SPEAKER_00"))

    rec = _flush_bc(state)

    assert rec["speaker"] == "田中"
    assert rec["speaker_source"] == "cluster_voiceprint"
    assert rec.get("bc") is True                     # UI折りたたみ印は維持


def test_backchannel_attributed_via_cluster_key(tmp_path):
    """相槌でも、クラスタ匿名キー(@diar:N)の根拠があれば帰属される."""
    state = _make_state(tmp_path, tracker=_Tracker("ラベル継続", "人物9"),
                        namer=_Namer(name=None),
                        resolver=_PyannoteResolver("SPEAKER_00"))

    rec = _flush_bc(state)

    assert rec["speaker"] == "@diar:1"
    assert rec["speaker_source"] == "pyannote"
    assert rec.get("bc") is True


def test_backchannel_without_cluster_stays_unsure(tmp_path):
    """クラスタ根拠が無い相槌は従来どおり未確定（STTラベル・声紋継続は不信）."""
    state = _make_state(tmp_path, tracker=_Tracker("ラベル継続", "人物9"),
                        namer=_Namer(name=None), resolver=_SttResolver())

    rec = _flush_bc(state)

    assert rec["speaker"] == "?"
    assert rec.get("bc") is True


def test_non_backchannel_unaffected(tmp_path):
    """相槌でない発話の経路は本変更の影響を受けない（既存挙動の固定）."""
    state = _make_state(tmp_path, tracker=_Tracker("ラベル継続", "人物9"),
                        namer=_Namer(name=None), resolver=_SttResolver())

    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = "これは検証用の通常発言です"
    loop.cur_ms = 1000
    loop.cur_end = 3000
    loop.flush()  # type: ignore[no-untyped-call]
    rec = state.records[-1]

    assert rec["speaker"] == "人物9"
    assert "bc" not in rec
