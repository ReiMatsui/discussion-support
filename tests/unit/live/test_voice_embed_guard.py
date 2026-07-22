"""_embed のゼロノルム埋め込みガードのテスト（F5）.

無音等でモデルがゼロベクトルを返すと、正規化で NaN に化けて以後の内積比較を
全て壊す。_embed は None に落として NaN を上流に流さない。
（かつてはクラスタ間名寄せの代表埋め込み保存経路の検証と抱き合わせだったが、
名寄せ機構の削除（handoff §18.9）に伴いガード単体のテストとして残す。）
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._voice_profiles import VoiceProfiles


def test_zero_norm_embedding_is_guarded():
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.embed_ms = []
    vp._embed_raw = lambda wav: np.zeros(8)   # 無音等でゼロベクトルを返す想定

    assert vp._embed(np.ones(16000, dtype=np.float32)) is None
