"""SortformerLocalDiarizationProvider（サブプロセス方式）のユニットテスト.

NeMo は使わず、ワーカーのプロトコル（stdin PCM / stdout JSON Lines）だけを
模したフェイクワーカーで、イベント変換・不活性フォールバック・再start を
検証する。実モデルの結合は scripts/sortformer_worker.py 側の責務
（クラウドでの実音声 E2E 検証済み: 142016 で 39.4% ≒ バッチ 42.3%）。
"""
from __future__ import annotations

import sys
import textwrap
import time

from das.asr.live._bootstrap import vp_cluster_naming_disabled_warning
from das.asr.live._sortformer_diarization import (
    SortformerLocalDiarizationProvider,
    resolve_worker_python,
)

# stdin を読み切ってから、届いたバイト数に応じたイベントを吐くフェイク。
# ログ混入耐性の検証のため、素の print 行も混ぜる。
_FAKE_WORKER = textwrap.dedent("""
    import json, sys
    data = sys.stdin.buffer.read()
    print('{"e": "ready"}')
    print("[NeMo I] fake log line that must be ignored")
    if len(data) >= 3200:
        print(json.dumps({"e": "start", "ms": 0, "spk": "SPEAKER_00"}))
        print(json.dumps({"e": "end", "ms": 800, "spk": "SPEAKER_00"}))
        print(json.dumps({"e": "start", "ms": 900, "spk": "SPEAKER_01"}))
    sys.stdout.flush()
""")


def _make_provider(tmp_path, body: str = _FAKE_WORKER):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(body)
    return SortformerLocalDiarizationProvider(
        python_path=sys.executable, worker_path=worker)


def _wait(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_events_parsed_and_log_lines_ignored(tmp_path):
    """start/end がイベントに変換され、非JSON行は無視される."""
    p = _make_provider(tmp_path)
    p.start()
    p.send_audio(b"\0" * 3200)
    p.close()   # stdin EOF → フェイクが出力して終了
    assert _wait(lambda: len(p._events.queue) >= 1)
    events = p.drain_events()
    assert [(e.start_ms, e.end_ms, e.speaker, e.source) for e in events] == [
        (0, 800, "SPEAKER_00", "sortformer")]
    # 終了していない SPEAKER_01 は active_events に出る
    actives = p.active_events()
    assert [(e.start_ms, e.speaker) for e in actives] == [(900, "SPEAKER_01")]
    assert all(e.end_ms is None for e in actives)


def test_missing_python_degrades_to_inert(tmp_path):
    """ワーカー起動失敗（python が無い）は例外にせず不活性に落ちる."""
    p = SortformerLocalDiarizationProvider(
        python_path=str(tmp_path / "no-such-python"),
        worker_path=tmp_path / "w.py")
    p.start()   # 例外を投げない
    p.send_audio(b"\0" * 3200)   # 送信も no-op
    assert p.drain_events() == []
    assert p.active_events() == []
    p.close()


def test_worker_crash_degrades_to_inert(tmp_path):
    """途中死したワーカーは検知され、以後 send_audio が no-op になる."""
    p = _make_provider(tmp_path, "import sys; sys.exit(3)")
    p.start()
    assert _wait(lambda: p._dead)
    p.send_audio(b"\0" * 3200)   # BrokenPipe を握って落ちない
    assert p.drain_events() == []
    p.close()


def test_restart_resets_state_and_keeps_timeline_monotonic(tmp_path):
    """close→start（STT再接続対）で状態は捨てるが、時刻基点とラベル世代は引き継ぐ.

    pyannote provider の F3 と同じ対策: ワーカーは再起動ごとに時刻0・
    SPEAKER_00 から数え直すため、(1) 供給済み音声の累計msを基点に足さないと
    イベントが過去時刻にずれて resolver の照合が全滅し、(2) ラベルを
    R{epoch}: で区別しないと新旧の SPEAKER_00（別人になり得る）が同一キーに
    合流して誤帰属する（2026-07-25 自己監査で発見・修正）。
    """
    p = _make_provider(tmp_path)
    p.start()
    p.send_audio(b"\0" * 3200)   # 100ms ぶん供給
    p.close()
    assert _wait(lambda: len(p._events.queue) >= 1)

    p.start()   # 再起動: 旧イベント・active は消える
    assert p.drain_events() == []
    assert p.active_events() == []
    p.send_audio(b"\0" * 3200)
    p.close()
    assert _wait(lambda: len(p._events.queue) >= 1)
    events = p.drain_events()
    # ラベルは世代前置で旧世代と区別され、時刻は供給済み100msを基点に進む
    assert [e.speaker for e in events] == ["R1:SPEAKER_00"]
    assert events[0].start_ms == 100 + 0
    assert events[0].end_ms == 100 + 800


def test_provider_name_is_sortformer(tmp_path):
    assert _make_provider(tmp_path).name == "sortformer"


def test_resolve_worker_python_priority(monkeypatch):
    """python の解決順: 明示引数 > SORTFORMER_PYTHON > 既定パス."""
    monkeypatch.setenv("SORTFORMER_PYTHON", "/env/python")
    assert resolve_worker_python("/arg/python") == "/arg/python"
    assert resolve_worker_python(None) == "/env/python"
    monkeypatch.delenv("SORTFORMER_PYTHON")
    assert resolve_worker_python(None).endswith("/.venvs/sortformer/bin/python")


def test_sortformer_provider_gets_participant_hysteresis(tmp_path):
    """sortformer 使用時も新規ラベルの参加者化に3秒ヒステリシスが掛かる.

    pyannote と同族のクラスタ型匿名ラベルなので、短い誤活性が即
    @diar:N（偽参加者）にならないことを固定する（2026-07-22 追加）。
    """
    from tests.unit.live.test_session_state import _make_state
    s = _make_state()
    s.diarization_provider = _make_provider(tmp_path)
    assert s.key_for_diarization_speaker("sortformer", "SPEAKER_00",
                                         duration_ms=1500) == "?"
    key = s.key_for_diarization_speaker("sortformer", "SPEAKER_00",
                                        duration_ms=1600)
    assert key.startswith("@diar:")


def test_vp_cluster_naming_allowed_for_sortformer():
    """--vp-cluster-naming は sortformer でも有効（警告が出ない）."""
    assert vp_cluster_naming_disabled_warning("sortformer", True) is None
    assert vp_cluster_naming_disabled_warning("pyannote", True) is None
    assert vp_cluster_naming_disabled_warning("none", True) is not None
    assert vp_cluster_naming_disabled_warning("assemblyai", True) is not None
