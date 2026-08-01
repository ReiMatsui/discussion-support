"""席上限の安全弁（handoff §49）: 人物キーは既存席へ寄せない.

rehacq検証 2026-08-01: 冒頭の雑音声（音声調整の声）が席を先取りすると、
3人目の実在話者——声紋層は「人物3」と正しく同定し 43発話・2,587字を
一致させ続けた——が seat_audio の閉集合割当てで丸ごと参加者Bへ押し込まれた。
寄せ直しの前提「席上限で落ちたキーは席持ちの分裂クラスタ」（§27）は
@diar:N / #N にしか成り立たない。

守るべき性質:

  - 席上限で落ちた**人物キー**（人物N / 実名）は寄せず未確定にする
    （声紋と本人キーを控え、後で本人が席を得たときだけ遡及で貼り直せる）
  - 席上限で落ちた**暫定キー**（@diar:N 等）は従来どおり席の実音声で寄せる
    （千葉13本の実測ではこちらが全件。挙動不変）
  - 遡及訂正は安全弁レコードを**本人へだけ**貼り直せる（他人への貼り直しは
    安全弁が止めた押し込みの復活なので拒否）
"""
from __future__ import annotations

import datetime

import numpy as np

from das.asr.live._constants import AGENT_SPEAKER, UNSURE_SPEAKER
from das.asr.live._recv_loop import RecvLoop
from das.asr.live._seat_audio import RetroAttributor, SeatAudio
from das.asr.live._session_state import SessionState
from das.asr.live._speaker_keys import is_person_key


class _Args:
    lang = "ja"
    vp_debug = False


class _Backend:
    def parse_message(self, raw, lang):
        return raw


def _unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class _Tracker:
    """音声の先頭サンプルで埋め込みを決める最小フェイク."""

    def embed_audio(self, wav):
        if wav is None or wav.size == 0:
            return None
        return _unit(1, 0, 0) if float(wav[0]) > 0 else _unit(0, 1, 0)


def _audio(tag: float, sec: float) -> np.ndarray:
    return np.full(int(16000 * sec), tag, dtype=np.float32)


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
    # A/B の2席が育った状態（test_seat_audio と同じ作り方）
    state.seat_audio = SeatAudio(_Tracker(), ref_sec=30.0, min_ref_sec=3.0)
    state.seat_audio.observe("人物1", _audio(1.0, 4.0))
    state.seat_audio.observe("人物2", _audio(-1.0, 4.0))
    state.retro = RetroAttributor(state.seat_audio)
    return state


def _assign(state, *, sp_id, kind="声紋一致", wav_tag=1.0):
    loop = RecvLoop(state, _Args(), _Backend())  # type: ignore[arg-type]
    loop.cur_ms, loop.cur_end = 1000, 3000
    rec_extra: dict = {}
    diag_extra: dict = {}
    got = loop._assign_seat(
        UNSURE_SPEAKER, sp_id=sp_id, d={"kind": kind, "name": sp_id, "sim": 0.7},
        wav=_audio(wav_tag, 1.0), rec_extra=rec_extra, diag_extra=diag_extra)
    return got, rec_extra, diag_extra


def test_person_key_is_not_absorbed_into_existing_seats(tmp_path):
    """席上限で落ちた人物キーは、席の実音声で寄せずに未確定へ（§49の本体）."""
    state = _make_state(tmp_path)
    got, rec_extra, diag_extra = _assign(state, sp_id="人物9")
    assert got == UNSURE_SPEAKER
    assert rec_extra["speaker_source"] == "unseated_person_guard"
    assert rec_extra["vp_person"] == "人物9"
    assert diag_extra["src"] == "unseated_person_guard"


def test_real_named_person_is_guarded_too(tmp_path):
    """実名キー（過去セッションの登録者など）も同じく寄せない."""
    state = _make_state(tmp_path)
    got, rec_extra, _ = _assign(state, sp_id="田中")
    assert got == UNSURE_SPEAKER
    assert rec_extra["vp_person"] == "田中"


def test_guarded_voice_is_remembered_for_retro(tmp_path):
    """安全弁で未確定にした発話の声紋は控える（本人が席を得たら貼り直すため）."""
    state = _make_state(tmp_path)
    _assign(state, sp_id="人物9")
    assert 1000 in state.retro._embeddings


def test_provisional_key_is_still_absorbed(tmp_path):
    """@diar:N（席持ちの分裂）は従来どおり席の実音声で寄せる——挙動不変."""
    state = _make_state(tmp_path)
    got, rec_extra, _ = _assign(state, sp_id="@diar:9", kind="照合なし",
                                wav_tag=1.0)
    assert got == "人物1"
    assert rec_extra["speaker_reason"] == "seat_full_nearest_seat_audio"


def test_retro_only_restores_to_the_identified_person(tmp_path):
    """遡及訂正は安全弁レコードを本人（vp_person）へだけ貼り直せる."""
    state = _make_state(tmp_path)
    state.records.append({"ms": 100, "end_ms": 900, "speaker": UNSURE_SPEAKER,
                          "text": "そうそう、だから個人のYouTubeが",
                          "speaker_source": "unseated_person_guard",
                          "vp_person": "人物9"})
    # 他人への貼り直しは拒否（押し込みの復活になる）
    assert state.apply_retro_attribution({100: "人物1"}) == {}
    assert state.records[-1]["speaker"] == UNSURE_SPEAKER
    # 本人へは貼り直せる（席の回収・上限の引き上げで本人の席が生まれた後）
    assert state.apply_retro_attribution({100: "人物9"}) == {100: "人物9"}
    assert state.records[-1]["speaker"] == "人物9"


def test_is_person_key_vocabulary():
    """述語の語彙: 人物N・実名だけが人物キー."""
    assert is_person_key("人物3")
    assert is_person_key("田中")
    assert not is_person_key("@diar:1")
    assert not is_person_key("#2")
    assert not is_person_key("__AI__")
    assert not is_person_key(UNSURE_SPEAKER)
    assert not is_person_key(AGENT_SPEAKER)
