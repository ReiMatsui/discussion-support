"""席落ち発話の割当て（handoff §27）のユニットテスト.

守るべき性質:

  - 席上限で落ちた発話だけが対象。上流が未確定なら何もしない
  - 寄せ先は「席を持つ人の実音声」との比較で決める（人物プロファイルではない）
  - 確定は書かない＝次の発話は独立に判定される（可逆であることが §15.12 の
    「不可逆な操作は高確信を要求」との整合の根拠）
  - 数秒しか聞いていない席には寄せない
  - リネームに追従する（追従しないと消えた人格が復活する）
  - ハイブリッド構成以外（seat_audio が無い）は完全に不変
"""
from __future__ import annotations

import numpy as np

from das.asr.live._seat_audio import SeatAudio


def _unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class _Tracker:
    """音声の先頭サンプルで人物を決める最小フェイク（埋め込みの代用）."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_audio(self, wav):
        self.calls += 1
        if wav is None or wav.size == 0:
            return None
        tag = float(wav[0])
        if tag > 0.5:
            return _unit(1, 0, 0)
        if tag < -0.5:
            return _unit(0, 1, 0)
        return _unit(1, 1, 0)      # どちらつかず（1位はAだが僅差）


def _audio(tag: float, sec: float) -> np.ndarray:
    a = np.full(int(16000 * sec), tag, dtype=np.float32)
    return a


def _ready(min_ref_sec: float = 3.0) -> SeatAudio:
    """A/B の2席が参照として育った状態を作る."""
    sa = SeatAudio(_Tracker(), ref_sec=30.0, min_ref_sec=min_ref_sec)
    sa.observe("人物1", _audio(1.0, 4.0))
    sa.observe("人物2", _audio(-1.0, 4.0))
    return sa


def test_picks_the_seat_whose_voice_matches():
    """席を持つ人のうち、その音声に最も似ている1人を返す."""
    sa = _ready()
    picked = sa.nearest(_audio(1.0, 1.0))
    assert picked is not None
    assert picked[0] == "人物1"
    assert picked[1] > 0.9          # sim
    assert picked[2] > 0.0          # 2位との差


def test_no_similarity_floor_is_applied():
    """類似度の下限は課さない（閉集合の割当てなので1位を必ず選ぶ）.

    §27.7: 別人でも 0.89 に達するため絶対しきい値は分離子として働かない。
    安全性を担うのは適用範囲（席上限で落ちた発話に限る）であって閾値ではない。
    下限を課しても成績がほぼ変わらないことも実測済み。
    """
    sa = _ready()
    picked = sa.nearest(_audio(0.0, 1.0))   # どちらつかずの声
    assert picked is not None
    assert picked[0] in ("人物1", "人物2")
    assert picked[2] < 0.2                  # 2位と僅差でも選ぶ


def test_does_not_pick_when_only_one_seat_is_mature():
    """比較にならない（席が1つ）ときは選ばない."""
    sa = SeatAudio(_Tracker(), min_ref_sec=3.0)
    sa.observe("人物1", _audio(1.0, 4.0))
    assert sa.nearest(_audio(1.0, 1.0)) is None


def test_seats_heard_only_briefly_are_not_candidates():
    """数秒しか聞いていない席は候補にしない（序盤の参照は当てにならない）."""
    sa = SeatAudio(_Tracker(), min_ref_sec=3.0)
    sa.observe("人物1", _audio(1.0, 4.0))
    sa.observe("人物2", _audio(-1.0, 0.5))   # 0.5秒しか聞いていない
    # 人物2 は候補から外れ、候補が1つになるので割当ては起きない
    assert sa.nearest(_audio(-1.0, 1.0)) is None
    sa.observe("人物2", _audio(-1.0, 3.0))   # 育った
    picked = sa.nearest(_audio(-1.0, 1.0))
    assert picked is not None and picked[0] == "人物2"


def test_reference_is_frozen_once_grown():
    """参照は ref_sec まで貯めたら凍結し、以後は再計算しない.

    席には誤帰属も混ざる（実測16%）ので貯め続けると参照が汚れる。埋め込みの
    再計算がライブの遅延に効くのも理由。
    """
    sa = SeatAudio(_Tracker(), ref_sec=5.0, min_ref_sec=1.0)
    sa.observe("人物1", _audio(1.0, 6.0))    # 一発で ref_sec 超え → 凍結
    calls = sa.tracker.calls
    sa.observe("人物1", _audio(-1.0, 6.0))   # 別人の音声が来ても取り込まない
    assert sa.tracker.calls == calls
    sa.observe("人物2", _audio(-1.0, 6.0))
    picked = sa.nearest(_audio(1.0, 1.0))
    assert picked is not None and picked[0] == "人物1"   # 汚染されていない


def test_rename_follows_so_the_old_key_is_not_resurrected():
    """リネームに追従する（追従しないと旧キーへ寄せて人格が復活する）."""
    sa = _ready()
    sa.rename("人物1", "田中")
    picked = sa.nearest(_audio(1.0, 1.0))
    assert picked is not None and picked[0] == "田中"


def test_merge_keeps_the_longer_reference():
    """合流（両方に席がある付け替え）で、参照が短くならないこと.

    rekey は改名だけでなく「分裂したクラスタを1人に束ねる」合流でも呼ばれる。
    素直に移すと、統合先が貯めた長い参照が統合元の短い参照で消える。
    """
    sa = SeatAudio(_Tracker(), ref_sec=30.0, min_ref_sec=1.0)
    sa.observe("人物1", _audio(1.0, 12.0))    # 統合先: 12秒
    sa.observe("@diar:2", _audio(1.0, 2.0))   # 統合元: 2秒（同じ人）
    sa.observe("人物2", _audio(-1.0, 5.0))

    sa.rename("@diar:2", "人物1")

    assert sa._seconds["人物1"] == 12.0, "短いほうの参照で上書きされた"
    assert "@diar:2" not in sa._seconds, "旧キーが残ると旧キーへ寄せてしまう"


def test_merge_does_not_freeze_a_short_reference():
    """凍結の印だけが残って、短い参照のまま育たなくなる事故を防ぐ."""
    sa = SeatAudio(_Tracker(), ref_sec=5.0, min_ref_sec=1.0)
    sa.observe("@diar:2", _audio(1.0, 6.0))   # 統合元は凍結済み（6秒）
    sa.observe("人物1", _audio(1.0, 2.0))     # 統合先は育ち途中（2秒）

    sa.rename("@diar:2", "人物1")

    assert sa._seconds["人物1"] == 6.0        # 長いほう（凍結済み）が残る
    assert "人物1" in sa._frozen

    # 逆向き: 統合元が短いときは、凍結の印を持ち込ませない
    sa2 = SeatAudio(_Tracker(), ref_sec=5.0, min_ref_sec=1.0)
    sa2.observe("人物1", _audio(1.0, 6.0))    # 統合先が凍結済み（6秒）
    sa2.observe("@diar:2", _audio(1.0, 1.0))  # 統合元は1秒
    sa2.rename("@diar:2", "人物1")
    assert sa2._seconds["人物1"] == 6.0
    assert "人物1" in sa2._frozen


def test_rename_of_an_unknown_key_leaves_the_target_alone():
    """席を持たないキーの付け替えで、統合先の参照を消さないこと."""
    sa = _ready()
    sa.rename("@diar:9", "人物1")
    picked = sa.nearest(_audio(1.0, 1.0))
    assert picked is not None and picked[0] == "人物1"


def test_reset_clears_everything():
    sa = _ready()
    sa.reset()
    assert sa.nearest(_audio(1.0, 1.0)) is None


def test_ai_and_unsure_keys_are_never_seats():
    """AI声紋・未確定は席ではないので参照に入れない."""
    from das.asr.live._constants import UNSURE_SPEAKER
    sa = SeatAudio(_Tracker(), min_ref_sec=1.0)
    sa.observe("__AI__", _audio(1.0, 4.0))
    sa.observe(UNSURE_SPEAKER, _audio(-1.0, 4.0))
    sa.observe("人物1", _audio(1.0, 4.0))
    assert sa.nearest(_audio(1.0, 1.0)) is None   # 候補は人物1だけ＝比較不能


# ---------------------------------------------------------------------------
# 遡及訂正（handoff §28）
# ---------------------------------------------------------------------------

def test_retro_revises_early_calls_with_the_grown_reference():
    """序盤の判定を、参照が育った後の基準で決め直す.

    誤りはセッション序盤に偏る（実測: 開始0-1分は正解29%、5-10分は90%）。
    参照が育った時点で決め直すと 79.2%→89.5%（5分時点）。
    """
    from das.asr.live._seat_audio import RetroAttributor
    sa = SeatAudio(_Tracker(), ref_sec=30.0, min_ref_sec=1.0)
    retro = RetroAttributor(sa, schedule=(120.0,), interval=300.0)

    # 序盤: 席が1つしか育っておらず判定できない
    sa.observe("人物1", _audio(1.0, 2.0))
    early = sa.embed(_audio(-1.0, 1.0))      # 実際は人物2の声
    assert sa.nearest_from(early) is None
    retro.remember(1000, early)

    # 参照が育つ
    sa.observe("人物2", _audio(-1.0, 2.0))

    assert retro.due(130.0) is True
    assert retro.revise() == {1000: "人物2"}   # 後から正しく決まる


def test_retro_fires_on_schedule_then_at_intervals():
    """予定時刻に達したときだけ発火し、以後は一定間隔で繰り返す."""
    from das.asr.live._seat_audio import RetroAttributor
    retro = RetroAttributor(SeatAudio(_Tracker()), schedule=(120.0, 300.0),
                            interval=300.0)
    assert retro.due(119.0) is False
    assert retro.due(120.0) is True
    assert retro.due(299.0) is False
    assert retro.due(300.0) is True
    assert retro.due(400.0) is False      # 前回から interval 経っていない
    assert retro.due(600.0) is True


def test_retro_reset_clears_remembered_voices():
    from das.asr.live._seat_audio import RetroAttributor
    sa = _ready(min_ref_sec=1.0)
    retro = RetroAttributor(sa, schedule=(120.0,), interval=300.0)
    retro.remember(1000, sa.embed(_audio(1.0, 1.0)))
    retro.reset()
    assert retro.revise() == {}
    assert retro.due(119.0) is False       # 予定も先頭に戻る


# -- 短くて僅差なら黙る（handoff §36） --------------------------------


def test_short_and_close_call_declines():
    """1秒未満で1位と2位が僅差なら、寄せずに黙る.

    残る誤帰属の81%は1秒未満の発話で、その大半がこの割当てを通る。
    実測では、棄権に回った分の63%が誤帰属だった（正解を削る分より多い）。
    """
    from das.asr.live._seat_audio import declines_short
    sa = _ready(min_ref_sec=1.0)
    picked = sa.nearest(_audio(0.0, 0.5))      # どちらつかず＝僅差
    assert picked is not None                  # 選ぶこと自体はできる
    assert declines_short(picked, 500) is True


def test_long_utterance_keeps_the_close_call():
    """同じ僅差でも、1秒以上の発話なら寄せる.

    長い発話に同じ棄権則を掛けると、棄権の7割が正解の取りこぼしになる。
    """
    from das.asr.live._seat_audio import declines_short
    sa = _ready(min_ref_sec=1.0)
    picked = sa.nearest(_audio(0.0, 2.0))
    assert declines_short(picked, 2000) is False


def test_short_but_clear_call_is_kept():
    """短くても差がはっきりしていれば寄せる（短さ自体は理由にならない）."""
    from das.asr.live._seat_audio import declines_short
    sa = _ready(min_ref_sec=1.0)
    picked = sa.nearest(_audio(1.0, 0.3))
    assert declines_short(picked, 300) is False


def test_retro_pulls_a_close_short_call_back_to_unsure():
    """貼り直しでも棄権則が効く（効かないと引き戻せず半分しか働かない）."""
    from das.asr.live._constants import UNSURE_SPEAKER
    from das.asr.live._seat_audio import RetroAttributor
    sa = _ready(min_ref_sec=1.0)
    retro = RetroAttributor(sa, schedule=(120.0,), interval=300.0)
    retro.remember(1000, sa.embed(_audio(0.0, 0.4)), 400)   # 短くて僅差
    retro.remember(2000, sa.embed(_audio(-1.0, 0.4)), 400)  # 短いが明確
    assert retro.revise() == {1000: UNSURE_SPEAKER, 2000: "人物2"}


def test_retro_without_a_length_behaves_as_before():
    """長さを控えていない発話は、これまでどおり寄せる（後方互換）."""
    from das.asr.live._constants import UNSURE_SPEAKER
    from das.asr.live._seat_audio import RetroAttributor
    sa = _ready(min_ref_sec=1.0)
    retro = RetroAttributor(sa, schedule=(120.0,), interval=300.0)
    retro.remember(1000, sa.embed(_audio(0.0, 0.4)))   # 僅差だが長さは不明
    assert retro.revise()[1000] != UNSURE_SPEAKER
