"""F2: エコー判定を声紋トラッカーの副作用より前に評価する回帰テスト.

漏れ込んだAI音声が「新規話者の蓄積・自動登録」に化けるのを、
(1) classify より前のテキスト安全網、(2) エコー窓中の count=False で二重に防ぐ。
"""
from __future__ import annotations

import datetime
import json

from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState


class _Args:
    lang = "ja"
    vp_debug = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _RecordingTracker:
    """classify 呼び出しの count 引数を記録するフェイク声紋トラッカー."""

    def __init__(self, ret: str = "人物1", kind: str = "声紋一致") -> None:
        self.last: dict | None = {"kind": kind, "label": "1", "name": ret, "sim": 0.8}
        self.calls: list[dict] = []
        self._ret = ret

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        self.calls.append({"count": count, "chars": chars, "enroll": enroll})
        return self._ret


class _EchoAgent:
    """安全網・count 抑止の判定に必要な最小フェイク agent."""

    def __init__(self, *, in_echo: bool = True, ai_speaking: bool = True,
                 sim: float = 0.9) -> None:
        self.in_echo_window = in_echo
        self.ai_speaking = ai_speaking
        self._sim = sim

    def _best_similarity(self, text: str) -> float:
        return self._sim


def _make_state(tmp_path, *, tracker=None):
    state = SessionState(  # type: ignore[no-untyped-call]
        args=_Args(),
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
    state.asr_pcm_buf = bytearray(b"\0" * 16000 * 2 * 3)
    return state


def _loop_with(state, *, text="まず、今日の目的を確認しましょう", ms=1000, end=3000):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"  # type: ignore[assignment]
    loop.cur_text = text
    loop.cur_ms = ms
    loop.cur_end = end
    return loop


def test_echo_in_window_is_dropped_before_classify(tmp_path):
    """エコー窓中に配信済みAIテキストと類似の発話が来たら classify を呼ばずに破棄する
    （匿名話者の蓄積・自動登録が起きない, D2）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = _EchoAgent(in_echo=True, sim=0.9)  # type: ignore[assignment]
    loop = _loop_with(state)

    loop.flush()  # type: ignore[no-untyped-call]

    assert tracker.calls == []      # classify が呼ばれない = 蓄積・登録が起きない
    assert state.records == []      # 記録もされない


def test_echo_drop_is_logged_to_diag(tmp_path):
    """破棄したエコーは echo_drop として diag に1行残る（記録なしに登録通知だけ、を防ぐ）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = _EchoAgent(in_echo=True, sim=0.9)  # type: ignore[assignment]
    loop = _loop_with(state)

    loop.flush()  # type: ignore[no-untyped-call]

    with open(state.diag_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    drops = [x for x in lines if x.get("type") == "echo_drop"]
    assert len(drops) == 1
    assert drops[0]["src"] == "agent"
    assert drops[0]["sim"] >= 0.35
    assert "まず" in drops[0]["text"]


def test_voiceprint_echo_drop_is_logged_to_diag(tmp_path):
    """声紋がAIと判定して捨てた発話も diag に残る.

    テキスト安全網だけが記録を残し、声紋経路は黙って捨てていた。捨てられた
    発話は records にも turns にも残らないので、記録から挙動を再生する
    （handoff §23）ときにこの経路だけ穴になる。
    """
    tracker = _RecordingTracker(ret="__AI__", kind="声紋一致")
    state = _make_state(tmp_path, tracker=tracker)
    loop = _loop_with(state, text="では次の議題に移ります")

    loop.flush()  # type: ignore[no-untyped-call]

    assert state.records == []      # 捨てられている
    with open(state.diag_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    drops = [x for x in lines if x.get("type") == "echo_drop"]
    assert len(drops) == 1, "声紋経路のエコー除去が記録されていない"
    assert drops[0]["src"] == "voiceprint"
    assert drops[0]["key"] == "__AI__"
    assert "では次の議題" in drops[0]["text"]


def test_ai_active_suppresses_registration_enroll(tmp_path):
    """安全網に引っかからない(sim低)漏れ込みでも、AI発話中は enroll=False で蓄積・
    自動登録を抑止する（照合・話者判定自体は count=True で行う, D2/P2-2）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    # in_echo_window=False で安全網はスキップ、ai_speaking=True で enroll 抑止
    state.agent = _EchoAgent(in_echo=False, ai_speaking=True, sim=0.0)  # type: ignore[assignment]
    loop = _loop_with(state, text="室内に漏れ込んだ声です")

    loop.flush()  # type: ignore[no-untyped-call]

    assert len(tracker.calls) == 1
    assert tracker.calls[0]["count"] is True     # 照合・補正は行う（P2-2）
    assert tracker.calls[0]["enroll"] is False   # 蓄積・登録だけ止める


def test_normal_utterance_outside_echo_window_counts_normally(tmp_path):
    """エコー窓外の通常発話は従来どおり count=True で classify され、記録される."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = _EchoAgent(in_echo=False, ai_speaking=False, sim=0.0)  # type: ignore[assignment]
    loop = _loop_with(state, text="これは普通の発言です")

    loop.flush()  # type: ignore[no-untyped-call]

    assert len(tracker.calls) == 1
    assert tracker.calls[0]["count"] is True
    assert tracker.calls[0]["enroll"] is True
    assert len(state.records) == 1
    assert state.records[0]["text"] == "これは普通の発言です"


def test_retired_echo_texts_drop_after_partner_detach(tmp_path):
    """partner 切断後も、TTL内の退役テキストと類似の発話はエコー破棄する（P2-4）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = None
    state.partner = None
    state.add_retired_echo_texts(["まず、今日の目的を確認しましょう"])
    loop = _loop_with(state, text="まず、今日の目的を確認しましょう")

    loop.flush()  # type: ignore[no-untyped-call]

    assert tracker.calls == []      # classify を呼ばず破棄（蓄積・登録が起きない）
    assert state.records == []
    with open(state.diag_path, encoding="utf-8") as f:
        drops = [json.loads(x) for x in f.read().splitlines()
                 if x.strip() and json.loads(x).get("type") == "echo_drop"]
    assert drops and drops[0]["src"] == "retired"


def test_retired_echo_texts_do_not_drop_dissimilar(tmp_path):
    """退役テキストと似ていない通常発話は、切断後も従来どおり処理される（P2-4）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = None
    state.partner = None
    state.add_retired_echo_texts(["まず、今日の目的を確認しましょう"])
    loop = _loop_with(state, text="週末は釣りに行ってきました")

    loop.flush()  # type: ignore[no-untyped-call]

    assert len(tracker.calls) == 1
    assert len(state.records) == 1


def _record_agent_interval(state, start_ms, end_ms):
    """AI再生区間を [start_ms, end_ms] で記録する（マイクmsタイムライン）."""
    state.asr_pcm_total_bytes = start_ms * 32
    state.note_ai_speech_start("agent")
    state.asr_pcm_total_bytes = end_ms * 32
    state.note_ai_speech_end("agent")


def test_late_echo_overlapping_interval_is_dropped(tmp_path):
    """壁時計のエコー窓を過ぎても、発話区間がAI再生区間と重なればエコー破棄する（P2-1）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    # 窓は過ぎている（in_echo=False, ai_speaking=False）が、AIが鳴っていた区間と重なる。
    state.agent = _EchoAgent(in_echo=False, ai_speaking=False, sim=0.9)  # type: ignore[assignment]
    _record_agent_interval(state, 1000, 3000)
    loop = _loop_with(state, ms=1100, end=2900)

    loop.flush()  # type: ignore[no-untyped-call]

    assert tracker.calls == []   # 区間重なりで安全網が発火 → classify を呼ばない
    assert state.records == []


def test_late_overlap_suppresses_count_even_when_window_passed(tmp_path):
    """安全網に掛からない(sim低)重なり発話でも、区間重なりなら count=False にする（P2-1）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = _EchoAgent(in_echo=False, ai_speaking=False, sim=0.0)  # type: ignore[assignment]
    _record_agent_interval(state, 1000, 3000)
    loop = _loop_with(state, text="室内に漏れ込んだ声です", ms=1100, end=2900)

    loop.flush()  # type: ignore[no-untyped-call]

    assert len(tracker.calls) == 1
    assert tracker.calls[0]["count"] is True     # 照合は行う
    assert tracker.calls[0]["enroll"] is False   # 蓄積・登録は止める


def test_non_overlapping_utterance_counts_despite_recorded_intervals(tmp_path):
    """AI再生区間が記録済みでも、重ならない発話は従来どおり count=True で記録する（P2-1）."""
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = _EchoAgent(in_echo=False, ai_speaking=False, sim=0.9)  # type: ignore[assignment]
    _record_agent_interval(state, 1000, 3000)
    loop = _loop_with(state, text="全く別の時間の発言です", ms=10000, end=12000)

    loop.flush()  # type: ignore[no-untyped-call]

    assert len(tracker.calls) == 1
    assert tracker.calls[0]["count"] is True
    assert tracker.calls[0]["enroll"] is True
    assert len(state.records) == 1
