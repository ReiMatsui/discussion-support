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
