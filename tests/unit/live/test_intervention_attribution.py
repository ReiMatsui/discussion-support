"""介入のための帰属情報の受け渡し（§49.17 案B/案D）.

§49.15 の測定: 介入時点の帰属は85-86%で、弱さは「未確定」への正直な棄権に
集約される（誤帰属は序盤含め3-8%）。ゆえに介入側に必要なのは、
(B) 未確定を名指ししないこと、(D) 参加度の集計が未確定に食われないこと。

守るべき性質:

  - ON_UTTERANCE は (speaker, text, meta) で呼ばれ、meta に未確定フラグ・
    確信度・決定経路が入る。旧2引数のコールバックにも従来どおり届く
  - 宛先が「未確定」の介入は全体宛に落ちる（しきい値なし）
  - 分離時間の集計: 窓・クラスタ写像・未対応クラスタの除外
  - 置き換えは records 側の全話者を分離が捕捉しているときだけ
    （物差しの混在を避ける。Soniox単独=分離なしは挙動不変）
  - 全部未確定になった話者も、分離が検出していれば参加者として現れる
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

from das.asr.live._participation import (
    apply_diarization_time,
    diarization_time_stats,
    participation_stats,
)
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._session_state import SessionState
from das.cli._listen import _addressee_head


class _Args:
    lang = "ja"
    vp_debug = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


def _make_state(tmp_path):
    state = SessionState(  # type: ignore[no-untyped-call]
        args=_Args(),
        started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp_path / "o.md"),
        html_path=str(tmp_path / "o.html"),
        diag_path=str(tmp_path / "o.diag"),
        turns_path=str(tmp_path / "o.turns"),
        wav_path=str(tmp_path / "o.wav"),
        tracker=None,
        serve=False,
    )
    state.save = lambda *a, **k: None  # type: ignore[method-assign]
    return state


def _flush(tmp_path, text="今日の議題を確認しましょう、資料の3ページからです"):
    state = _make_state(tmp_path)
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_speaker = "1"
    loop.cur_text = text
    loop.cur_ms, loop.cur_end = 1000, 3000
    loop.flush()  # type: ignore[no-untyped-call]
    return state


# --- B: meta の受け渡し -------------------------------------------------

def test_on_utterance_receives_meta(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr("das.asr.live.ON_UTTERANCE",
                        lambda sp, tx, meta=None: got.append((sp, tx, meta)))
    _flush(tmp_path)
    assert len(got) == 1
    sp, _tx, meta = got[0]
    assert isinstance(meta, dict)
    assert set(meta) == {"unsure", "confidence", "source"}
    assert meta["unsure"] is (sp == "未確定")


def test_two_arg_callback_still_works(tmp_path, monkeypatch):
    """旧2引数のコールバック（外部連携）にも従来どおり届く."""
    got = []

    def old_style(sp, tx):
        got.append((sp, tx))

    monkeypatch.setattr("das.asr.live.ON_UTTERANCE", old_style)
    _flush(tmp_path)
    assert len(got) == 1


# --- B: 未確定への名指しゲート -----------------------------------------

def test_unsure_addressee_falls_back_to_everyone():
    assert _addressee_head("未確定") == "💡介入(全体)"
    assert _addressee_head(None) == "💡介入(全体)"
    assert _addressee_head("田中") == "💡介入(田中さん宛)"
    assert _addressee_head("発言者") == "💡介入(発言者さん宛)"


# --- D: 分離時間ベースの参加度 -----------------------------------------

@dataclass
class _Ev:
    source: str
    speaker: str
    start_ms: int
    end_ms: int | None


def _resolver(mapping):
    return lambda e: mapping.get(f"{e.source}:{e.speaker}")


def test_diarization_time_stats_windows_and_maps():
    events = [
        _Ev("pyannote", "SPEAKER_00", 0, 10_000),          # 窓の外
        _Ev("pyannote", "SPEAKER_00", 400_000, 420_000),   # 20秒
        _Ev("pyannote", "SPEAKER_01", 410_000, 415_000),   # 5秒
        _Ev("pyannote", "SPEAKER_02", 412_000, 414_000),   # 写像なし→除外
        _Ev("pyannote", "SPEAKER_00", 419_000, None),      # 開区間→除外
    ]
    got = diarization_time_stats(
        events, _resolver({"pyannote:SPEAKER_00": "人物1",
                           "pyannote:SPEAKER_01": "人物2"}),
        window_ms=300_000)
    assert got == {"人物1": 20_000.0, "人物2": 5_000.0}


def test_apply_replaces_time_axis_and_reveals_unsure_speaker():
    """未確定に食われて records に現れない話者が、分離計測で参加者になる."""
    records = [{"speaker": "人物1", "text": "あ" * 30, "ms": 0, "end_ms": 30_000}]
    stats = participation_stats(records)
    diar = {"人物1": 30_000.0, "人物3": 25_000.0}   # 人物3は全発話が未確定だった
    merged = apply_diarization_time(stats, diar)
    assert set(merged) == {"人物1", "人物3"}
    assert abs(merged["人物3"]["time_share"] - 25 / 55) < 1e-9
    assert merged["人物3"]["turns"] == 0


def test_apply_is_noop_when_coverage_is_partial():
    """records 側の話者を分離が捕捉していなければ何も変えない（物差し混在の禁止）."""
    records = [
        {"speaker": "人物1", "text": "あ" * 30, "ms": 0, "end_ms": 30_000},
        {"speaker": "#2", "text": "い" * 10, "ms": 5_000, "end_ms": 8_000},
    ]
    stats = participation_stats(records)
    merged = apply_diarization_time(stats, {"人物1": 30_000.0})
    assert merged == stats


def test_apply_is_noop_without_diarization():
    """Soniox単独（分離なし）は挙動不変."""
    records = [{"speaker": "人物1", "text": "あ" * 30, "ms": 0, "end_ms": 30_000}]
    stats = participation_stats(records)
    assert apply_diarization_time(stats, {}) == stats
