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


def test_diag_does_not_carry_the_previous_verdict(tmp_path):
    """STTが話者ラベルを返さない発話に、前の発話の判定を書かないこと.

    その経路は classify を呼ばない（判定の根拠が無い）。にもかかわらず
    tracker.last を直に書いていたため、直前の発話の kind/sim が
    そのまま残った。採点も分析も diag の kind を信じて動くので、
    ここがずれると静かに全部が狂う。
    """
    tracker = _RecordingTracker(ret="人物1", kind="声紋一致")
    state = _make_state(tmp_path, tracker=tracker)

    loop = _loop_with(state, text="こちらは普通の発言です")
    loop.flush()  # type: ignore[no-untyped-call]

    loop.cur_speaker = ""            # STT が話者を返さなかった
    loop.cur_text = "話者不明の発言"
    loop.cur_ms, loop.cur_end = 4000, 6000
    loop.flush()  # type: ignore[no-untyped-call]

    with open(state.diag_path, encoding="utf-8") as f:
        rows = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    unknown = [r for r in rows if r.get("ms") == 4000]
    assert unknown, "話者不明の発話が diag に残っていない"
    assert unknown[0].get("kind") is None, "前の発話の判定が書かれている"
    assert unknown[0]["final_key"] == "?"


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


def test_interval_uses_capture_position_under_send_backlog(tmp_path):
    """送信が遅延していても、AI再生区間はエコーが実際に並ぶ位置に記録される.

    2026-07-30 の講義(2時間3分)で、送信バックログ約60秒の状態でのAI介入の
    エコーが「参加者B」の発話として議事録に混入した。原因は区間の記録が
    送信済み位置(current_asr_ms)だったこと: エコーはマイク取り込み順で
    ストリームに並ぶため、実位置より約60秒手前に区間が記録され、±300msの
    重なり判定から外れて類似度の照合に進めなかった。区間は
    取り込み位置(送信済み+送信待ち)で記録する。
    """
    tracker = _RecordingTracker()
    state = _make_state(tmp_path, tracker=tracker)
    state.agent = _EchoAgent(in_echo=False, ai_speaking=False, sim=0.9)  # type: ignore[assignment]

    # 送信済み 1000ms・送信待ち 60000ms（=送信バックログ60秒）の状態でAIが8秒鳴る。
    state.asr_pcm_total_bytes = 1000 * 32
    state.audio_q.put(b"\0" * (60000 * 32))
    state.note_ai_speech_start("agent")
    state.audio_q.put(b"\0" * (8000 * 32))   # 再生中に取り込まれた8秒（エコー含む）
    state.note_ai_speech_end("agent")

    # エコーはずっと後に確定して届く。cur_ms は取り込み位置ベース（61000ms〜）。
    loop = _loop_with(state, ms=61100, end=66000)
    loop.flush()  # type: ignore[no-untyped-call]

    assert tracker.calls == []   # 区間が実位置にあるので安全網が発火する
    assert state.records == []


def test_open_interval_now_uses_capture_position(tmp_path):
    """再生中（開区間）の判定も取り込み位置で行う.

    送信済み位置を「現在」に使うと、遅延中は now が開始位置より手前になり
    開区間 [start, now] が空集合に化けて、再生中のエコーすら素通りする。
    """
    state = _make_state(tmp_path)
    state.asr_pcm_total_bytes = 1000 * 32
    state.audio_q.put(b"\0" * (60000 * 32))
    state.note_ai_speech_start("agent")   # 開始 61000ms、開いたまま

    assert state.overlaps_ai_speech(61200, 61800, source="agent")


def test_note_send_backlog_updates_and_logs_to_diag(tmp_path):
    """送信バックログの更新と diag 記録（5秒以上で30秒おき）.

    2026-07-30 の講義で送信遅延が約170秒まで蓄積したが、計測がなく
    セッション中に気づけなかった。バックログを毎チャンク更新し、
    5秒以上なら send_backlog として diag に残す（30秒に1回まで）。
    """
    state = _make_state(tmp_path)

    import os

    def _backlog_lines():
        if not os.path.exists(state.diag_path):
            return []
        with open(state.diag_path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if "send_backlog" in x]

    # 5秒未満: 更新はされるが diag には書かない
    state.audio_q.put(b"\0" * (3000 * 32))
    state.note_send_backlog()
    assert state.send_backlog_ms == 3000
    assert _backlog_lines() == []

    # 5秒以上: diag に1行。直後の再呼び出しでは重複しない（30秒レート制限）
    state.audio_q.put(b"\0" * (60000 * 32))
    state.note_send_backlog()
    state.note_send_backlog()
    lines = _backlog_lines()
    assert len(lines) == 1
    assert lines[0]["backlog_ms"] == 63000

    # バックログは api_snapshot で UI に出る
    assert state.api_snapshot()["backlog_ms"] == 63000


class _ImpureTracker:
    """ラベル不純を返すフェイク声紋トラッカー（§47 門番テスト用）."""

    def __init__(self, sim: float) -> None:
        self.last = {"kind": "ラベル不純", "label": "2", "name": "人物2",
                     "sim": sim}

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return "人物2"


def _flush_impure(tmp_path, *, sim: float, text: str) -> list:
    tracker = _ImpureTracker(sim)
    state = _make_state(tmp_path, tracker=tracker)
    state.cluster_namer = object()   # ハイブリッド構成の印（門番の適用条件）
    loop = _loop_with(state, text=text, ms=1000, end=11000)
    loop.flush()  # type: ignore[no-untyped-call]
    return state.records


def test_impure_long_lowsim_goes_unsure(tmp_path):
    """未登録話者の門番（§47）: ラベル不純×30字以上×sim<0.5 は未確定に倒す.

    2026-07-30 の講義で、プロファイルも席も無い未登録の話者6発話（915字）が
    Soniox ラベルの写像で参加者Bへ吸われた（#210 等: 60〜264字, sim 0.37〜
    0.49）。長い発話なのにどの登録済みの声とも似ていないなら、既存の誰かへ
    寄せず未確定にする。校正済み実会話9本では該当0件（impure_lowsim_guard.py）。
    """
    from das.asr.live._constants import UNSURE_SPEAKER
    records = _flush_impure(
        tmp_path, sim=0.45,
        text="と、話が戻っちゃうんだけど、経歴、話して、でも、思っている数、じゃあ先生の話を聞かせてください")
    assert len(records) == 1
    assert records[0]["speaker"] == UNSURE_SPEAKER
    assert records[0]["speaker_source"] == "impure_lowsim_guard"


def test_impure_long_highsim_keeps_attribution(tmp_path):
    """sim>=0.5 なら門番は発火しない（本物の話者の長発話は寄せたまま）."""
    records = _flush_impure(
        tmp_path, sim=0.55,
        text="で、僕は、もう、その前からアニメオタクやったんですが、その頃はまだ言えなかったんです")
    assert len(records) == 1
    assert records[0].get("speaker_source") != "impure_lowsim_guard"


def test_impure_short_lowsim_keeps_attribution(tmp_path):
    """30字未満なら発火しない（短い発話は正解でも sim が低く出る）."""
    records = _flush_impure(tmp_path, sim=0.2, text="京都の方です。")
    assert len(records) == 1
    assert records[0].get("speaker_source") != "impure_lowsim_guard"


def test_retro_does_not_revive_impure_guarded_record(tmp_path):
    """遡及訂正は門番の未確定を席の参照で復活させない（§47）.

    未登録の声には席の参照が無く、貼り直しは必ず既存の誰か（＝誤帰属）に
    なる。測定も「遡及の後に門番」の意味論で行った。
    """
    from das.asr.live._constants import UNSURE_SPEAKER
    state = _make_state(tmp_path)
    state.records.append({"ms": 1000, "end_ms": 2000, "text": "x",
                          "speaker": UNSURE_SPEAKER,
                          "speaker_source": "impure_lowsim_guard"})
    state.records.append({"ms": 3000, "end_ms": 4000, "text": "y",
                          "speaker": UNSURE_SPEAKER,
                          "speaker_source": "seat_assign"})
    applied = state.apply_retro_attribution({1000: "人物2", 3000: "人物2"})
    assert state.records[0]["speaker"] == UNSURE_SPEAKER   # 門番は不可侵
    assert state.records[1]["speaker"] == "人物2"           # 通常の遡及は従来どおり
    assert 1000 not in applied
