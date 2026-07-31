"""起動時の組み立て（run_session から切り出した各段）のテスト.

**なぜ要るのか**: ここはユニットテストが1行も通っていなかった（2026-07-28 に
カバレッジを取って判明）。起動処理は「動かしてみるまで分からない」場所で、
配線を間違えても気づけるのはユーザーがアプリを立ち上げた瞬間になる。

守るのは配線だけ——モデルを実際に読むわけでも、スレッドを実際に回すわけでも
ない。「どの条件でどれを作るか／作らないか」を固定する。
"""
from __future__ import annotations

import threading
from typing import ClassVar

import pytest

from das.asr.live import _bootstrap, _seat_audio


class _Args:
    """run_session が読む属性だけを持つ最小の引数."""

    no_vp = True
    vp_model = "redimnet"
    voices = "voices.json"
    diarization = "none"
    diarization_max_speakers = 3
    vp_cluster_naming = False
    docs = None
    af = False

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------- 声紋


def test_no_vp_skips_the_voiceprint_model_entirely() -> None:
    """--no-vp なら読み込みを試みない（起動を待たせない）."""
    assert _bootstrap._build_tracker(_Args(no_vp=True)) is None


def test_tracker_loads_the_configured_model_once(monkeypatch, capsys) -> None:
    """指定モデルだけを読み込む（旧代替モデルへのフォールバックは削除済み）.

    別モデルのしきい値で黙って動くと帰属の性質が変わるため、読めない場合は
    声紋なしへ落とす（次のテスト）。成功時は話者数上限が声紋層へ渡ること。
    """
    tried: list[str] = []

    class _VP:
        profiles: ClassVar[dict] = {}

        def __init__(self, **kw):
            tried.append(kw["model"])

        def set_max_human_speakers(self, n):
            self.max_speakers = n

    monkeypatch.setattr(_bootstrap, "VoiceProfiles", _VP)
    tracker = _bootstrap._build_tracker(_Args(no_vp=False))

    assert tried == ["redimnet"], "指定モデル以外を読み込んでいる"
    assert tracker is not None
    assert tracker.max_speakers == 3, "話者数の上限が声紋層へ渡っていない"


def test_tracker_is_none_and_says_how_to_fix_when_the_model_fails(
        monkeypatch, capsys) -> None:
    """読めなければ None（代替へは落とさない）。復旧手順を必ず出す."""
    class _VP:
        def __init__(self, **kw):
            raise RuntimeError("依存が無い")

    monkeypatch.setattr(_bootstrap, "VoiceProfiles", _VP)
    assert _bootstrap._build_tracker(_Args(no_vp=False)) is None
    out = capsys.readouterr().out
    assert "声紋照合がOFF" in out
    assert "uv add" in out, "復旧手順が出ていない"


# ------------------------------------------------------------ 話者分離


def test_no_diarization_provider_by_default() -> None:
    assert _bootstrap._build_diarizer(_Args(diarization="none")) is None


def test_pyannote_needs_its_api_key(monkeypatch) -> None:
    monkeypatch.delenv("PYANNOTEAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        _bootstrap._build_diarizer(_Args(diarization="pyannote"))


def test_pyannote_provider_gets_the_speaker_cap(monkeypatch) -> None:
    seen: dict = {}

    def _fake(key, *, max_speakers):
        seen.update(key=key, max_speakers=max_speakers)
        return "provider"

    monkeypatch.setenv("PYANNOTEAI_API_KEY", "k")
    monkeypatch.setattr(_bootstrap, "PyannoteStreamingDiarizationProvider", _fake)
    got = _bootstrap._build_diarizer(_Args(diarization="pyannote",
                                           diarization_max_speakers=4))
    assert got == "provider"
    assert seen == {"key": "k", "max_speakers": 4}


# -------------------------------------------------- ハイブリッド構成


def test_cluster_layer_is_off_without_the_flag() -> None:
    namer, seat = _bootstrap._build_cluster_layer(
        _Args(diarization="pyannote", vp_cluster_naming=False), object())
    assert (namer, seat) == (None, None)


def test_cluster_layer_is_off_without_a_tracker(capsys) -> None:
    """声紋が無ければ照合しようがない。黙って無視せず理由を出す."""
    namer, seat = _bootstrap._build_cluster_layer(
        _Args(diarization="pyannote", vp_cluster_naming=True), None)
    assert (namer, seat) == (None, None)
    assert "無効なため無視" in capsys.readouterr().out


def test_cluster_layer_builds_both_and_turns_on_hybrid(monkeypatch) -> None:
    """席の音声とクラスタ命名は同時に作られ、声紋層もハイブリッドに切り替わる."""
    class _Tracker:
        hybrid = False

        def set_hybrid(self, on):
            self.hybrid = on

    monkeypatch.setattr(_bootstrap, "ClusterVoiceNamer", lambda t: ("namer", t))
    monkeypatch.setattr(_bootstrap, "SeatAudio",
                        lambda t, **kw: ("seat", t))
    tracker = _Tracker()
    namer, seat = _bootstrap._build_cluster_layer(
        _Args(diarization="pyannote", vp_cluster_naming=True), tracker)
    assert namer == ("namer", tracker)
    assert seat == ("seat", tracker)
    assert tracker.hybrid is True


# ------------------------------------------------------ ワーカーの起動


class _FakeThread:
    started: ClassVar[list] = []

    def __init__(self, *, target, args=(), kwargs=None, daemon=False):
        self.target = target

    def start(self):
        _FakeThread.started.append(getattr(self.target, "__name__", str(self.target)))


@pytest.fixture
def started(monkeypatch):
    _FakeThread.started = []
    monkeypatch.setattr(threading, "Thread", _FakeThread)
    return _FakeThread.started


class _State:
    agent = None
    af_runtime = None


def test_no_llm_workers_without_an_api_key(started, capsys) -> None:
    _bootstrap._start_llm_workers(_State(), _Args(), oai_key="",
                                  oai_model="m", out_path="/tmp/o.md",
                                  explicit_agenda=False)
    assert started == []
    assert "無効" in capsys.readouterr().out


def test_only_topic_extraction_without_an_agent(started) -> None:
    """介入先が居ないなら、判断材料を作るワーカーは動かさない（API代の無駄）."""
    _bootstrap._start_llm_workers(_State(), _Args(), oai_key="k",
                                  oai_model="m", out_path="/tmp/o.md",
                                  explicit_agenda=False)
    assert started == ["_run_topic_worker"]


def test_full_worker_set_with_an_agent(started) -> None:
    s = _State()
    s.agent = object()
    _bootstrap._start_llm_workers(s, _Args(), oai_key="k", oai_model="m",
                                  out_path="/tmp/o.md", explicit_agenda=False)
    assert started == [
        "_run_topic_worker", "_run_drift_checker", "_run_triage_worker",
        "_run_fact_checker", "_run_participation_checker",
        "_run_structuring_checker", "_run_agenda_detector",
    ]


def test_agenda_detector_is_skipped_when_the_topic_was_given(started) -> None:
    s = _State()
    s.agent = object()
    _bootstrap._start_llm_workers(s, _Args(), oai_key="k", oai_model="m",
                                  out_path="/tmp/o.md", explicit_agenda=True)
    assert "_run_agenda_detector" not in started


def test_af_runtime_starts_only_when_asked(started, monkeypatch) -> None:
    monkeypatch.delenv("DAS_AF_RUNTIME", raising=False)
    s = _State()
    s.agent = object()
    _bootstrap._start_llm_workers(s, _Args(af=True), oai_key="k", oai_model="m",
                                  out_path="/tmp/o.md", explicit_agenda=True)
    assert "run_af_runtime" in started
    assert "_run_af_checker" in started


# ---------------------------------------------------------- 受信ループ


class _Recv:
    """台本どおりの状態を返す RecvLoop の代役."""

    script: ClassVar[list] = []
    made = 0

    def __init__(self, state, args, backend):
        _Recv.made += 1

    def run(self, ws):
        return _Recv.script.pop(0) if _Recv.script else "finished"


class _LoopState:
    def __init__(self, script):
        _Recv.script = list(script)
        _Recv.made = 0
        self.stop = threading.Event()
        self.reset_requested = threading.Event()
        self.start_requested = threading.Event()
        self.stt_ws = None
        self.diarization_provider = None
        self.waiting_to_start = False
        self.resetting = False
        self.rev = 0
        self.reset_calls = 0

    def reset_for_new_meeting(self):
        self.reset_calls += 1


def test_receive_loop_returns_when_stt_finishes(monkeypatch) -> None:
    monkeypatch.setattr(_bootstrap, "RecvLoop", _Recv)
    s = _LoopState(["finished"])
    _bootstrap._receive_until_stopped(s, _Args(), object(), lambda: "ws")
    assert _Recv.made == 1, "終了なのに作り直している"


def test_receive_loop_reconnects_after_a_disconnect(monkeypatch) -> None:
    """切断されたら繋ぎ直し、受信ループも作り直す（前の断片を持ち越さない）."""
    monkeypatch.setattr(_bootstrap, "RecvLoop", _Recv)
    monkeypatch.setattr(_bootstrap.time, "sleep", lambda *_: None)
    s = _LoopState(["disconnected", "finished"])
    calls = []
    _bootstrap._receive_until_stopped(
        s, _Args(), object(), lambda: calls.append("connect") or "ws")
    assert calls == ["connect"]
    assert s.stt_ws == "ws"
    assert _Recv.made == 2


def test_receive_loop_rebuilds_the_session_on_reset(monkeypatch) -> None:
    """リセット要求では会議状態も作り直す（前の会議の発話を持ち込まない）."""
    monkeypatch.setattr(_bootstrap, "RecvLoop", _Recv)
    s = _LoopState(["ok", "finished"])

    def _run(ws):
        # 1回目の run で UI からリセットが押された状況を作る
        if len(_Recv.script) == 2:
            s.reset_requested.set()
        return _Recv.script.pop(0) if _Recv.script else "finished"

    monkeypatch.setattr(_Recv, "run", lambda self, ws: _run(ws))
    _bootstrap._receive_until_stopped(s, _Args(), object(), lambda: "ws2")
    assert s.reset_calls == 1, "会議状態が作り直されていない"
    assert s.stt_ws == "ws2"
    assert not s.reset_requested.is_set()
    assert _Recv.made == 2


def test_receive_loop_stops_when_asked(monkeypatch) -> None:
    monkeypatch.setattr(_bootstrap, "RecvLoop", _Recv)
    s = _LoopState([])

    def _run(ws):
        s.stop.set()
        return "ok"

    monkeypatch.setattr(_Recv, "run", lambda self, ws: _run(ws))
    _bootstrap._receive_until_stopped(s, _Args(), object(), lambda: "ws")
    assert _Recv.made == 1


# -- 席の割当てだけ別の声紋モデルを使う（handoff §38） -----------------


def test_seat_embedder_is_skipped_for_non_redimnet() -> None:
    """声紋層が redimnet でなければ、席も同じモデルのまま（注入しない）."""
    class _T:
        model = "other-model"
    assert _seat_audio.seat_embedder(_T()) is None


def test_seat_embedder_falls_back_when_the_model_cannot_be_read(monkeypatch) -> None:
    """読み込みに失敗しても止めない.

    席の割当ては可逆な補助なので、声紋層と同じモデルに落ちれば従来どおり動く。
    ここで例外を投げるとセッションごと起動しなくなる。
    """
    class _T:
        model = "redimnet"

    def _boom(*a, **k):
        raise RuntimeError("ネットワークが無い")

    monkeypatch.setattr(_seat_audio, "make_embedder", _boom)
    assert _seat_audio.seat_embedder(_T()) is None


def test_seat_embedder_is_injected_when_available(monkeypatch) -> None:
    """読めたら SeatAudio にその埋め込み器が渡る."""
    class _T:
        model = "redimnet"
    monkeypatch.setattr(_seat_audio, "make_embedder", lambda m, s: (m, s))
    assert _seat_audio.seat_embedder(_T()) == ("redimnet", "b5")


def test_reset_survives_a_transient_stt_connect_failure(monkeypatch) -> None:
    """リセット時のSTT接続失敗はセッションを落とさず、再試行して復帰する.

    通常の切断は再試行するのに、リセット経路だけ connect が素通しで、瞬断と
    重なるとセッション全体が落ちていた（レビュー 2026-07-30）。
    """
    monkeypatch.setattr(_bootstrap, "RecvLoop", _Recv)
    monkeypatch.setattr(_bootstrap.time, "sleep", lambda *_: None)
    s = _LoopState(["ok", "finished"])

    def _run(ws):
        if len(_Recv.script) == 2:
            s.reset_requested.set()
        return _Recv.script.pop(0) if _Recv.script else "finished"

    monkeypatch.setattr(_Recv, "run", lambda self, ws: _run(ws))
    attempts = []

    def _connect():
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("瞬断")
        return "ws3"

    _bootstrap._receive_until_stopped(s, _Args(), object(), _connect)
    assert len(attempts) == 3, "接続を再試行していない"
    assert s.stt_ws == "ws3"
    assert not s.reset_requested.is_set()
