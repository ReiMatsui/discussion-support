"""不純ラベル門番（_attribution ステップ3d, handoff §18.8）の統合テスト.

声紋層が「ラベル不純」で棄権した発話のクラスタ回収は、
「その発話自身の声紋1位候補が回収先と一致し、類似が
CLUSTER_IMPURE_RECOVERY_ENDORSE_MIN_SIM 以上」のときだけ許可される。
Chiba 12会話の実測校正（回収の通算正解45%に対し、裏付けあり回収は
開発5会話で正解37/誤り6）。台帳・蓄積の副作用（キー発行・ヒステリシス
pending）は門番の遮断時も従来どおり進むこと（オフライン反実仮想との
意味論一致）、および cluster_namer 無し構成（pyannote単独）は挙動不変で
あることも固定する。
"""
from __future__ import annotations

import datetime
from types import SimpleNamespace

from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState


class _Args:
    vp_debug = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _ImpureTracker:
    """「ラベル不純」で棄権する声紋層のフェイク（1位候補と類似は指定可能）."""

    def __init__(self, cand=None, sim=None) -> None:
        info = {"label": "1", "hist": ["人物1", "人物2"]}
        if cand is not None:
            info["name"] = cand
        if sim is not None:
            info["sim"] = sim
        self.last = {"kind": "ラベル不純", **info}

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return "?"


class _ConfirmedNamer:
    """クラスタ確定名を常に返す ClusterVoiceNamer のフェイク."""

    def __init__(self, name="人物1") -> None:
        self._name = name
        self.merge_sim = None
        self.last_match = None

    def observe(self, raw_cluster, wav, *, overlapped=False):
        return self._name

    def canonical_cluster(self, raw_cluster):
        return raw_cluster

    def rename_confirmed(self, old, new):
        return None

    def nearest_cluster(self, raw_cluster):
        return None


class _AnonNamer(_ConfirmedNamer):
    """確定名なし（匿名キー経路 3c に落ちる）フェイク."""

    def observe(self, raw_cluster, wav, *, overlapped=False):
        return None


class _Resolver:
    def resolve(self, **kwargs):
        return SimpleNamespace(source="pyannote", speaker="SPEAKER_00",
                               confidence=0.9, reason="diarization_overlap")


class _FakePyannoteProvider:
    name = "pyannote"

    def drain_events(self):
        return []


def _make_state(tmp_path, *, tracker, namer):
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
    state.speaker_resolver = _Resolver()  # type: ignore[assignment]
    state.diarization_provider = _FakePyannoteProvider()  # type: ignore[assignment]
    return state


def _flush(state, *, text="これは検証用の発言です", ms=1000, end=6000):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = text
    loop.cur_ms = ms
    loop.cur_end = end
    loop.flush()  # type: ignore[no-untyped-call]
    return state


def test_impure_with_endorsement_is_recovered(tmp_path):
    """裏付けあり（1位候補=回収先・sim>=床）は従来どおり確定名へ回収される."""
    state = _make_state(tmp_path, tracker=_ImpureTracker(cand="人物1", sim=0.44),
                        namer=_ConfirmedNamer("人物1"))
    _flush(state)
    assert state.records[-1]["speaker"] == "人物1"
    assert state.records[-1]["speaker_source"] == "cluster_voiceprint"


def test_impure_candidate_mismatch_is_blocked(tmp_path):
    """1位候補が回収先と別人なら未確定に落ちる（理由を記録）."""
    state = _make_state(tmp_path, tracker=_ImpureTracker(cand="人物2", sim=0.60),
                        namer=_ConfirmedNamer("人物1"))
    _flush(state)
    rec = state.records[-1]
    assert rec["speaker"] == "?"
    assert rec["speaker_source"] == "cluster_impure_label"
    assert rec["speaker_reason"] == "impure_stt_label_without_voiceprint_endorsement"


def test_impure_low_sim_is_blocked(tmp_path):
    """候補一致でも類似が床未満（偽承認帯）なら未確定に落ちる."""
    state = _make_state(tmp_path, tracker=_ImpureTracker(cand="人物1", sim=0.10),
                        namer=_ConfirmedNamer("人物1"))
    _flush(state)
    assert state.records[-1]["speaker"] == "?"


def test_impure_without_ranking_is_blocked(tmp_path):
    """声紋候補が無い（短発話等で埋め込み未計算）不純発話は回収しない."""
    state = _make_state(tmp_path, tracker=_ImpureTracker(),
                        namer=_ConfirmedNamer("人物1"))
    _flush(state)
    assert state.records[-1]["speaker"] == "?"


def test_impure_anonymous_key_blocked_but_ledger_side_effects_preserved(tmp_path):
    """匿名キー経路（3c）も遮断されるが、キー発行・pending の副作用は従来どおり進む.

    オフライン反実仮想（最終ラベルのみ差し替え）と実装の意味論を一致させる
    ための固定: ヒステリシス消化・@diar 発行は起き、記録だけが未確定になる。
    """
    state = _make_state(tmp_path, tracker=_ImpureTracker(cand="人物2", sim=0.5),
                        namer=_AnonNamer())
    _flush(state, ms=1000, end=6000)   # 5秒 > ヒステリシス3秒 → 発行される
    assert state.records[-1]["speaker"] == "?"
    assert state.diarization_speaker_keys.get("pyannote:SPEAKER_00") == "@diar:1"


def test_pure_kinds_are_unchanged(tmp_path):
    """不純以外（蓄積中等）の発話は門番の対象外（確定名へ従来どおり）."""
    tracker = _ImpureTracker(cand="人物2", sim=0.5)
    tracker.last = {"kind": "蓄積中", "label": "1"}
    state = _make_state(tmp_path, tracker=tracker, namer=_ConfirmedNamer("人物1"))
    _flush(state)
    assert state.records[-1]["speaker"] == "人物1"


def test_no_cluster_namer_mode_is_unchanged(tmp_path):
    """cluster_namer 無し（pyannote単独）は門番が掛からず挙動不変."""
    state = _make_state(tmp_path, tracker=_ImpureTracker(cand="人物2", sim=0.5),
                        namer=None)
    _flush(state, ms=1000, end=6000)
    assert state.records[-1]["speaker"] == "@diar:1"
