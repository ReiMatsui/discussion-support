"""名乗りのLLM判定（§49.14, 二段構えの2段目）のユニットテスト.

守るべき性質:

  - flush: 正規表現で確定できない名乗り候補（is_llm_candidate）だけが
    キューに積まれる（普通の発話・正規表現で確定した発話は積まれない）
  - 判定の適用（_process_nanori_candidate, LLMはフェイク）:
      * 名乗りなら登録して過去の発話を書き換える（source=nanori_llm）
      * 氏名の門（空・8文字超・代名詞）は適用しない
      * 確定級の声紋一致（0.65）は名乗りに勝つ
      * 満席で席が立たないなら書き換えない
  - --no-llm ではキュー自体が張られない（既存の一括停止に含まれる）
"""
from __future__ import annotations

import datetime
import queue

import numpy as np

from das.asr.live._bootstrap import _process_nanori_candidate
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState


class _Args:
    lang = "ja"
    vp_debug = False
    diarization_max_speakers = None


class _Backend:
    def parse_message(self, raw, lang):
        return raw


class _Tracker:
    def __init__(self, *, sim=0.45, kind="声紋一致", profiles=()) -> None:
        self.last = {"kind": kind, "label": "2", "name": "人物2", "sim": sim}
        self.enrolled: list[str] = []
        self._profiles = list(profiles)

    def classify(self, wav, speaker, *, overlapped, count, chars, enroll=True):
        return "人物2"

    def active_profile_names(self):
        return list(self._profiles) + self.enrolled

    def enroll_from_audio(self, name, wav):
        self.enrolled.append(name)
        return True


def _make_state(tmp_path, tracker):
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
    state.asr_pcm_buf = bytearray(np.full(16000 * 10, 12000, dtype="<i2").tobytes())
    state.cluster_namer = object()
    return state


def _flush(state, text):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "2"
    loop.cur_text = text
    loop.cur_ms, loop.cur_end = 1000, 5000
    loop.flush()  # type: ignore[no-untyped-call]


# --- flush 側: 候補の積み込み -----------------------------------------

def test_llm_candidate_is_queued(tmp_path):
    """正規表現で確定できない名乗り候補（「機構の…と言います」）はキューへ."""
    state = _make_state(tmp_path, _Tracker())
    state.nanori_llm_queue = queue.Queue()
    _flush(state, "神戸医療産業都市推進機構の久保と言います。よろしくお願いします")
    item = state.nanori_llm_queue.get_nowait()
    assert item["ms"] == 1000
    assert "久保" in item["text"]
    assert item["sim"] == 0.45


def test_regex_resolved_nanori_is_not_queued(tmp_path):
    """正規表現で確定した名乗りは同期処理され、LLMには回さない."""
    state = _make_state(tmp_path, _Tracker())
    state.nanori_llm_queue = queue.Queue()
    _flush(state, "東京新聞の奥野です。2点お願いします。")
    assert state.nanori_llm_queue.empty()
    assert state.records[-1]["speaker"] == "奥野"


def test_plain_utterance_is_not_queued(tmp_path):
    """名乗りらしさの無い普通の発話は積まれない（コスト0.5%の根拠）."""
    state = _make_state(tmp_path, _Tracker())
    state.nanori_llm_queue = queue.Queue()
    _flush(state, "この議題は前回の続きなので、まず資料を確認しましょう")
    assert state.nanori_llm_queue.empty()


# --- 適用側 -----------------------------------------------------------

def _item(**over):
    base = {"ms": 1000, "text": "神戸医療産業都市推進機構の久保と言います",
            "wav": np.ones(16000, dtype=np.float32),
            "sp_id": "人物2", "kind": "声紋一致", "sim": 0.45}
    base.update(over)
    return base


def test_positive_judgement_is_applied(tmp_path):
    tracker = _Tracker()
    state = _make_state(tmp_path, tracker)
    state.records.append({"ms": 1000, "end_ms": 5000, "speaker": "人物2",
                          "text": "…久保と言います"})
    ok = _process_nanori_candidate(
        state, _item(), post=lambda c: {"nanori": True, "name": "久保"},
        announced=set())
    assert ok
    assert tracker.enrolled == ["久保"]
    assert state.records[0]["speaker"] == "久保"
    assert state.records[0]["speaker_source"] == "nanori_llm"
    assert any("久保" in str(r.get("sys")) for r in state.records if "sys" in r)


def test_negative_judgement_changes_nothing(tmp_path):
    tracker = _Tracker()
    state = _make_state(tmp_path, tracker)
    state.records.append({"ms": 1000, "speaker": "人物2", "text": "x"})
    ok = _process_nanori_candidate(
        state, _item(), post=lambda c: {"nanori": False, "name": None},
        announced=set())
    assert not ok and tracker.enrolled == []
    assert state.records[0]["speaker"] == "人物2"


def test_implausible_names_are_gated(tmp_path):
    """空・8文字超・代名詞は適用しない（評価1-2周目の実測に基づく門）."""
    tracker = _Tracker()
    state = _make_state(tmp_path, tracker)
    state.records.append({"ms": 1000, "speaker": "人物2", "text": "x"})
    for bad in ("", "しんのじはうまへん", "私", "自分"):
        ok = _process_nanori_candidate(
            state, _item(), post=lambda c, b=bad: {"nanori": True, "name": b},
            announced=set())
        assert not ok, bad
    assert tracker.enrolled == []


def test_strong_voiceprint_vetoes_llm_nanori(tmp_path):
    """確定級（>=0.65）の声紋一致は名乗りに勝つ（同期側と同じ校正線）."""
    tracker = _Tracker(sim=0.80)
    state = _make_state(tmp_path, tracker)
    state.records.append({"ms": 1000, "speaker": "人物2", "text": "x"})
    ok = _process_nanori_candidate(
        state, _item(sim=0.80), post=lambda c: {"nanori": True, "name": "久保"},
        announced=set())
    assert not ok and tracker.enrolled == []


def test_full_house_blocks_application(tmp_path):
    """満席で席が立たないなら過去の発話を書き換えない."""
    class _FullArgs(_Args):
        diarization_max_speakers = 1

    tracker = _Tracker()
    state = _make_state(tmp_path, tracker)
    state.args = _FullArgs()
    state.disp_name("人物2")   # 1席を占有
    state.records.append({"ms": 500, "end_ms": 900, "speaker": "人物2",
                          "text": "既存の発話"})
    state.records.append({"ms": 1000, "speaker": "人物2", "text": "x"})
    ok = _process_nanori_candidate(
        state, _item(), post=lambda c: {"nanori": True, "name": "久保"},
        announced=set())
    assert not ok
    assert all(str(r.get("speaker")) != "久保" for r in state.records
               if "speaker" in r)
