"""席の回収（handoff §49）: 雑音声が先取りした席を実在の話者に明け渡す.

rehacq検証 2026-08-01: 冒頭の音声調整の声が参加者Aの席を取り（1発話7字・
16秒以降無発話）、本編の3人目（人物3）が満席で座れず43発話が行き場を失った。
席は一度埋まると回収されなかった。

守るべき性質:

  - 人物キーが SEAT_RECLAIM_MIN_DROPS 回落ち続け、かつ「発話がごく少なく
    長く黙っている匿名席」があるときだけ回収する
  - 回収した幽霊席の過去発話は未確定へ戻る（誤回収でも損害は有界）
  - 幽霊は席の参照・クラスタ台帳からも消える（復活しない）
  - 暫定キー（@diar:N）は何回落ちても席を奪えない
  - 全席が普通に発話している会話（千葉13本の形）では決して発火しない
  - 実名を付けた席は発話が少なくても奪わない
"""
from __future__ import annotations

import datetime
import json

import numpy as np

from das.asr.live._constants import UNSURE_SPEAKER
from das.asr.live._seat_audio import SeatAudio
from das.asr.live._session_state import SessionState


class _Args:
    lang = "ja"
    vp_debug = False
    diarization_max_speakers = 3


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


def _seed_rehacq_shape(state, *, ghost_key="@diar:1", ghost_ms=15_000,
                       ghost_text="ほんとですよ。"):
    """rehacq ランの形: 幽霊1席 + よく喋る2席で満席の状態を作る."""
    state.disp_name(ghost_key)      # 参加者A
    state.disp_name("人物1")        # 参加者B
    state.disp_name("人物2")        # 参加者C
    state.records.append({"ms": ghost_ms, "end_ms": ghost_ms + 800,
                          "speaker": ghost_key, "text": ghost_text})
    for i in range(6):
        state.records.append({"ms": 250_000 + i * 8_000,
                              "end_ms": 250_000 + i * 8_000 + 4_000,
                              "speaker": "人物1" if i % 2 else "人物2",
                              "text": "本編の議論がずっと続いています" * 3})


def _drop_times(state, key, n):
    got = None
    for _ in range(n):
        got = state.constrain_human_speaker_key(key)
    return got


def test_persistent_person_reclaims_a_ghost_seat(tmp_path):
    """落ち続ける人物キーは、3回目で幽霊席を回収して座る（§49の本体）."""
    state = _make_state(tmp_path)
    _seed_rehacq_shape(state)
    assert _drop_times(state, "人物3", 1) == UNSURE_SPEAKER   # 1回目は落ちる
    assert _drop_times(state, "人物3", 1) == UNSURE_SPEAKER   # 2回目も落ちる
    assert _drop_times(state, "人物3", 1) == "人物3"          # 3回目で回収
    # 幽霊の過去発話は未確定へ
    ghost_recs = [r for r in state.records if r.get("ms") == 15_000]
    assert ghost_recs[0]["speaker"] == UNSURE_SPEAKER
    assert ghost_recs[0]["speaker_source"] == "seat_reclaimed"
    # 幽霊は台帳から消え、本人が座っている
    assert "@diar:1" not in state.anonymous_labels
    with open(state.diag_path, encoding="utf-8") as f:
        events = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    rec = [e for e in events if e.get("type") == "seat_reclaimed"]
    assert len(rec) == 1
    assert rec[0]["ghost"] == "@diar:1" and rec[0]["new"] == "人物3"


def test_reclaimed_ghost_leaves_seat_audio_and_cluster_ledger(tmp_path):
    """回収された幽霊は席の参照とクラスタ台帳からも消える（復活の穴を塞ぐ）."""
    state = _make_state(tmp_path)
    _seed_rehacq_shape(state)
    state.seat_audio = SeatAudio(
        tracker=None, ref_sec=30.0, min_ref_sec=1.0,
        embedder=lambda wav: np.array([1.0, 0.0]))
    state.seat_audio.observe("@diar:1", np.ones(16000 * 2, dtype=np.float32))
    state.diarization_speaker_keys["pyannote:SPEAKER_00"] = "@diar:1"
    _drop_times(state, "人物3", 3)
    assert "@diar:1" not in state.seat_audio._embeddings
    assert "pyannote:SPEAKER_00" not in state.diarization_speaker_keys


def test_provisional_key_cannot_take_a_seat(tmp_path):
    """@diar:N は何回落ちても席を奪えない（分裂クラスタに席を渡さない）."""
    state = _make_state(tmp_path)
    _seed_rehacq_shape(state)
    assert _drop_times(state, "@diar:9", 5) == UNSURE_SPEAKER
    assert "@diar:1" in state.anonymous_labels


def test_no_reclaim_when_every_seat_is_talkative(tmp_path):
    """全席が普通に発話している（千葉13本の形）なら決して回収しない."""
    state = _make_state(tmp_path)
    for key in ("@diar:1", "人物1", "人物2"):
        state.disp_name(key)
        for i in range(4):
            state.records.append({"ms": 200_000 + i * 10_000,
                                  "end_ms": 200_000 + i * 10_000 + 3_000,
                                  "speaker": key,
                                  "text": "全員がそれなりに話している会話です"})
    assert _drop_times(state, "人物3", 5) == UNSURE_SPEAKER


def test_no_reclaim_when_the_quiet_seat_spoke_recently(tmp_path):
    """発話が少なくても、直近に喋った席は奪わない（静かな人 ≠ 幽霊）."""
    state = _make_state(tmp_path)
    _seed_rehacq_shape(state, ghost_ms=299_000)   # 直近1秒前に発話
    assert _drop_times(state, "人物3", 5) == UNSURE_SPEAKER


def test_named_seat_is_never_reclaimed(tmp_path):
    """実名を付けた席は、発話が少なくても機械判断で奪わない."""
    state = _make_state(tmp_path)
    state.names["田中"] = "田中"
    state.records.append({"ms": 15_000, "end_ms": 15_800,
                          "speaker": "田中", "text": "うん。"})
    state.disp_name("人物1")
    state.disp_name("人物2")
    for i in range(4):
        state.records.append({"ms": 250_000 + i * 8_000,
                              "end_ms": 250_000 + i * 8_000 + 3_000,
                              "speaker": "人物1" if i % 2 else "人物2",
                              "text": "本編の議論がずっと続いています" * 3})
    assert _drop_times(state, "人物3", 5) == UNSURE_SPEAKER
    assert state.records[0]["speaker"] == "田中"
