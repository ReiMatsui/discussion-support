"""人物別しきい値（_person_th）の分布ベース再設計のテスト.

旧仕様（受理simのみ記録×中央値-0.12）は、記録条件に person_th 自身が入る
自己参照＝選択バイアスで、しきい値が単調に肥大するラチェットだった
（CallHome 0856 実測: 0.42→0.73 まで上昇し、正しい 0.5-0.65 帯の一致を
全遮断して帰属29%）。新仕様は
  1. 記録側: person_th と独立な固定基準（基準しきい値＋margin通過の生sim）
     で記録し自己参照を断つ
  2. 統計側: 中央値でなく下位35パーセンタイル-0.12
の二本立て。巻き取り防止（本人が安定して高simを出す環境で、別人の
中途半端な類似を弾く）は維持する。

VoiceProfiles.__init__ はMLモデルを読むため、__new__ で必要フィールドだけ
用意し、_embed をスタブしてメインパス（中尺発話の即時判定）を検証する。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._voice_profiles import SR, VoiceProfiles


def _emb_for_sim(sim: float) -> np.ndarray:
    """プロファイルA=(1,0,0)との類似度が sim、B=(0,1,0)とはほぼ0の単位ベクトル."""
    return np.array([sim, 0.0, np.sqrt(1.0 - sim * sim)], dtype=np.float64)


def _unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _tracker() -> VoiceProfiles:
    """既知2人(A,B)を持つトラッカー（redimnet相当の既定しきい値0.42）."""
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.sp_map = {}
    vp.profiles = {"A": _unit(1, 0, 0), "B": _unit(0, 1, 0)}
    vp._active_keys = {"A", "B"}
    vp.pool = []
    vp.label_embs = {}
    vp.own_sims = {}
    vp.own_embs = {}
    vp._own_updates = {}
    vp.counts = {}
    vp.last = None
    vp.n_anon = 0
    vp.same_sims, vp.diff_sims = [], []
    vp.min_sec = 1.0
    vp.short_floor = 0.45
    vp.short_bonus = 0.08
    vp.strict_sec = 3.0
    vp.margin = 0.05
    vp.thresh = 0.42
    vp.consist = 0.34
    vp.model = "redimnet"
    vp.auto = True
    vp.max_human_speakers = None
    return vp


# strict_sec(3.0s) 以上の中尺発話 → メインパスの基準しきい値そのままで照合
_MID = np.ones(int(SR * 4.0), dtype=np.float32)


def _classify_with_sim(vp: VoiceProfiles, sim: float, label: str = "1") -> str:
    vp._embed = lambda wav: _emb_for_sim(sim)  # type: ignore[method-assign]
    # chars=0: 自動登録の蓄積を切り、照合経路だけを検証する
    return vp.classify(_MID, label, count=True, chars=0)


def test_person_th_default_without_history():
    """履歴3件未満は基準値のまま."""
    vp = _tracker()
    assert vp._person_th("A", 0.42) == 0.42
    vp.own_sims["A"] = [0.7, 0.7]
    assert vp._person_th("A", 0.42) == 0.42


def test_person_th_uses_lower_quantile_not_median():
    """しきい値は下位35パーセンタイル-0.12（中央値-0.12だと正しい一致を遮断）.

    CallHome 0856 の実分布（本人の一致が 0.5-0.73 に広く散る 8kHz電話）を
    模した履歴で、0.5 の正しい一致が通ることを確認する。旧仕様の
    中央値(0.625)-0.12=0.505 は 0.5 を弾く。
    """
    vp = _tracker()
    h = [0.5, 0.55, 0.6, 0.65, 0.7, 0.73]
    vp.own_sims["A"] = list(h)
    th = vp._person_th("A", 0.42)
    assert th == max(0.42, float(np.percentile(h, 35)) - 0.12)
    assert th < 0.5   # 本人の下端 0.5 帯の一致を遮断しない
    assert _classify_with_sim(vp, 0.5) == "A"
    assert vp.last["kind"] == "声紋一致"


def test_person_th_still_blocks_absorption():
    """巻き取り防止は維持: 本人が安定して高sim(0.7台)なら 0.5 の別人を弾く.

    YouTube録音ハーネスで機能している本来の意図（同一再生チェーン等で
    別人が 0.5 前後の中途半端な類似を出しても本人の典型に届かなければ
    巻き取らない。吸収帯 0.45-0.59 / 本人帯 0.67-0.82 の実測）を保つ。
    """
    vp = _tracker()
    vp.own_sims["A"] = [0.70, 0.72, 0.74, 0.76, 0.78]
    assert vp._person_th("A", 0.42) >= 0.58
    assert _classify_with_sim(vp, 0.5) != "A"   # 遮断（未確定側へ落ちる）
    assert vp.last["kind"] == "未確定"


def test_raw_sim_recorded_even_when_person_th_rejects():
    """記録は person_th 判定の手前（基準しきい値＋margin通過の生sim）で行う.

    受理された一致だけを記録する旧仕様は、person_th が上がるほど低めの
    一致が履歴から消える自己参照＝選択バイアスで、ラチェットの根本原因
    だった。person_th に弾かれた一致の生simも履歴に入ることを確認する。
    """
    vp = _tracker()
    vp.own_sims["A"] = [0.70, 0.72, 0.74, 0.76, 0.78]
    assert _classify_with_sim(vp, 0.5) != "A"   # person_th には弾かれるが…
    assert 0.5 in vp.own_sims["A"]              # 生simは記録される


def test_below_base_threshold_is_not_recorded():
    """基準しきい値＋marginに届かないsimは記録しない（履歴の汚染防止）.

    記録基準は「person_th が無かった頃に受理されたはずの一致」＝固定の
    基準条件。それ未満の弱い類似（大半は別人・雑音）まで貯めると、
    しきい値が根拠なく下がり巻き取り防止が崩れる。
    """
    vp = _tracker()
    assert _classify_with_sim(vp, 0.3) != "A"   # 基準0.42未満
    assert vp.own_sims.get("A", []) == []


def test_no_ratchet_under_wide_own_distribution():
    """本人simが広く散る環境（8kHz電話相当）でしきい値がラチェットしない.

    高めの一致が続いた後でも、person_th に弾かれた本人の低め一致が履歴に
    入って分位を押し下げるため、しきい値は分布に追従して復元する。
    旧仕様（受理のみ×中央値）では 0.5 帯が二度と受理されなくなっていた。
    """
    vp = _tracker()
    for s in [0.73, 0.72, 0.70, 0.71, 0.73, 0.72]:   # まず高めの一致が続く
        assert _classify_with_sim(vp, s) == "A"
    assert vp._person_th("A", vp.thresh) >= 0.55      # 巻き取り防止は効いている
    # 以後、本人の声が 0.5 帯に落ちる（電話・マイク距離等）。最初は弾かれても
    # 生simが記録され続け、しきい値が分布に追従して 0.5 帯を再受理できる。
    accepted = False
    for _ in range(12):
        _classify_with_sim(vp, 0.5)
        if vp.last["kind"] == "声紋一致":   # ラベル継続でなく照合で受理された
            accepted = True
            break
    assert accepted, "person_th がラチェットして 0.5 帯を遮断し続けた"
    assert vp._person_th("A", vp.thresh) <= 0.5


def test_person_th_never_below_base():
    """どれだけ低いsimが並んでも基準しきい値は下回らない（max(base, ...)）."""
    vp = _tracker()
    vp.own_sims["A"] = [0.43, 0.44, 0.45, 0.46]
    assert vp._person_th("A", 0.42) == 0.42
