"""pyannote Live-1 クラスタ単位の声紋照合による名前付け（ハイブリッド構成）.

設計: docs/design/pyannote_live1_trial_2026-07-09.md §8.4/§9 参照。

3役分業:
  - Soniox            = 文字起こし（現行のまま、挙動は変えない）
  - pyannote Live-1    = 話者クラスタリング（生ラベル SPEAKER_XX の区間確定）
  - VoiceProfiles      = クラスタ単位の名前付け（本モジュールが仲介）

根拠（docs/design 8.4節の盲検裁定の一貫性分析）: pyannoteの生クラスタは同一人物の
束ねとして78%一貫していたのに対し、現行の発話断片単位の声紋名前付けは44%しか
一貫していなかった。断片ごとに照合するより、クラスタ単位でまとめて照合した方が
安定する、という前提でこのモジュールを設計している。

動作:
  pyannote の生クラスタ(SPEAKER_XX。再接続後は provider が前置する
  ``R{epoch}:SPEAKER_XX``)ごとに音声(PCM float32)を蓄積し、累積が
  ``min_sec``（既定 ``PYANNOTE_CLUSTER_NAMING_MIN_SEC`` 秒）に達するたびに
  ``VoiceProfiles.match_profile()`` で声紋照合を試みる。confidenceが十分なら
  そのクラスタ→名前を確定し、以後そのクラスタの発話は確定名にそのまま帰属する
  （再照合しない）。confidence不足ならバッファを保持したまま蓄積を続け、次に
  閾値へ達した時点で再照合する（音声が増えるほど確度が上がる想定）。

不整合対策:
  - クラスタ分裂: 分裂後の各クラスタIDは独立に管理される（本クラスは辞書キーが
    増えるだけ）。分裂しても両方が同じ人物へ照合されるだけなので実害は無い。
  - 重複発話: 呼び出し側（_recv_loop.py）が
    ``_diarization.has_overlapping_speakers`` で重複区間を検出し、
    ``overlapped=True`` として渡す。本クラスは重複区間の音声を蓄積しない
    （声が混ざって声紋がデタラメになるため）。
  - pyannote切断・再接続: 再接続後は provider がラベルに ``R{epoch}:`` を前置
    するため、本クラスから見ると「新しいクラスタキー」として自然に扱われる
    （旧クラスタの蓄積・確定は失われるが、新クラスタの蓄積が最初からやり直しに
    なるだけで、誤った名前を出し続けるよりは安全側）。
"""
from __future__ import annotations

import threading

import numpy as np

from ._constants import (
    PYANNOTE_CLUSTER_CONFIRM_MIN_SIM,
    PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC,
    PYANNOTE_CLUSTER_NAMING_MIN_SEC,
    SR,
)
from ._voice_profiles import VoiceProfiles


class ClusterVoiceNamer:
    """pyannote生クラスタラベル -> 声紋照合による確定名 のマッピングを管理する.

    tracker(VoiceProfiles)には状態を書き込まない（match_profileは副作用なし）。
    確定・蓄積の状態はすべてこのインスタンス内に閉じる。
    """

    def __init__(
        self,
        tracker: VoiceProfiles,
        *,
        min_sec: float = PYANNOTE_CLUSTER_NAMING_MIN_SEC,
        max_buffer_sec: float = PYANNOTE_CLUSTER_NAMING_MAX_BUFFER_SEC,
        confirm_min_sim: float = PYANNOTE_CLUSTER_CONFIRM_MIN_SIM,
    ) -> None:
        self.tracker = tracker
        self.min_sec = min_sec
        self.max_buffer_sec = max_buffer_sec
        # クラスタ→人物の確定に要求する類似度の下限（_constants.py の校正根拠を
        # 参照）。「確定後は再照合しない」設計のため、確定は一致の中でも特に
        # 高確信のものに限る。下回った照合成功は確定せず蓄積を続ける。
        self.confirm_min_sim = float(confirm_min_sim)
        # クラスタ間名寄せ・最近傍統合の機構は 2026-07-21 に削除した（§18.9）。
        # 経緯: §15.12 でクラスタ埋め込み同士の比較に同一/別人を分離できる閾値が
        # 存在しないことが実測で確定し既定無効化 → その後の全測定（Chiba 12会話・
        # Sakura）でも有効化の見込みが出ず、本番到達経路も無いままだったため、
        # opt-in 機構ごと撤去（復元は git 履歴から。当時の実装・テストは
        # cc1006f 前後を参照）。クラスタ分裂は「新しい匿名参加者になる」
        # 安全側の既知限界のまま（§15.12 の判断を変えるものではない）。
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._confirmed: dict[str, str] = {}   # raw_cluster -> 確定名
        # 診断用（直近の照合試行の結果。UI/デバッグログでの可視化用途）。
        self.last_match: dict[str, object] | None = None
        # 軽量ロック（2026-07-15 レビュー F7）: observe 等は recvスレッドから、
        # reset は UI起点の reset_for_new_meeting（入力スレッド）から呼ばれ、
        # _buffers/_confirmed の複合更新が無防備だった。呼び出し頻度は発話単位
        # なので性能影響は無視できる。再入可能な RLock を使う。
        self._lock = threading.RLock()

    def confirmed_name(self, raw_cluster: str) -> str | None:
        """既にこのクラスタに確定済みの名前があれば返す（無ければNone）."""
        with self._lock:
            return self._confirmed.get(raw_cluster)

    def rename_confirmed(self, old: str, new: str) -> None:
        """確定名 old を new に付け替える（SessionState.rekey からの伝搬用）.

        UI /rename・stdin fix 等で表示キーが変わっても _confirmed が旧名の
        ままだと、observe() の確定短絡が旧名を返し続け、リネームした人物が
        別人格として復活する（docs/design/attribution_logic_review_2026-07.md
        C3）。rekey を状態一貫性の単一入口とし、ここへ伝搬する（P2）。
        """
        if old == new:
            return
        # UI/stdinスレッドの rekey から呼ばれるため、recvスレッドの observe と
        # 競合しないようロックで保護（main側 f41155a のロック方針に合わせる）。
        with self._lock:
            for cluster, name in list(self._confirmed.items()):
                if name == old:
                    self._confirmed[cluster] = new

    def reset(self) -> None:
        """会議リセット時に蓄積・確定状態をクリアする."""
        with self._lock:
            self._buffers.clear()
            self._confirmed.clear()
            self.last_match = None

    def _trim_buffer(self, buf: list[np.ndarray]) -> int:
        """バッファを max_buffer_sec に収まるよう古い方から捨て、総サンプル数を返す."""
        total = sum(a.size for a in buf)
        max_samples = int(self.max_buffer_sec * SR)
        while total > max_samples and len(buf) > 1:
            total -= buf.pop(0).size
        return total

    def observe(
        self, raw_cluster: str, wav: np.ndarray, *, overlapped: bool = False
    ) -> str | None:
        """クラスタの発話区間音声を1件観測する.

        既に確定済みなら確定名を返す。まだなら音声を蓄積し、累積が閾値に
        達していれば声紋照合を試みる。照合confidenceが十分なら確定して名前を
        返す。確定に至らない場合は None（未確定のまま蓄積を継続）。

        ``overlapped=True``（重複発話区間）の音声は蓄積しない（安全側）。
        既に確定済みのクラスタであれば overlapped でも確定名をそのまま返す
        （重複区間の全体としての帰属を未確定にするかどうかは呼び出し側の責務）。
        """
        with self._lock:
            return self._observe(raw_cluster, wav, overlapped=overlapped)

    def _observe(
        self, raw_cluster: str, wav: np.ndarray, *, overlapped: bool = False
    ) -> str | None:
        confirmed = self._confirmed.get(raw_cluster)
        if confirmed is not None:
            return confirmed
        if overlapped or wav.size == 0:
            return None
        buf = self._buffers.setdefault(raw_cluster, [])
        buf.append(np.asarray(wav, dtype=np.float32))
        total_samples = self._trim_buffer(buf)
        if total_samples < int(self.min_sec * SR):
            return None
        concat = np.concatenate(buf) if len(buf) > 1 else buf[0]
        match = self.tracker.match_profile(concat)
        self.last_match = {
            "cluster": raw_cluster,
            "total_sec": round(total_samples / SR, 1),
            "match": match,
        }
        if match is None:
            # 登録プロファイルとの照合は不成立。confidence不足として蓄積を継続し、
            # 次の観測でさらに音声が増えてから再照合する（クラスタ間名寄せは
            # 2026-07-21 に機構ごと削除。__init__ の経緯コメント参照）。
            return None
        name, confidence = match
        if confidence < self.confirm_min_sim:
            # 照合は成立したが、確定には確信不足。確定は取り消せない（再照合
            # しない）設計のため、低確信の誤確定1回が以後の全発話を汚染する
            # （Chiba 0532 実測: sim0.54 の誤確定→誤帰属37件。handoff §15.9）。
            # 棄却して蓄積を続け、音声が増えて確信が上がってから確定する。
            self.last_match = {"kind": "確定見送り(低確信)", "cluster": raw_cluster,
                               "name": name, "sim": round(float(confidence), 3),
                               "need": self.confirm_min_sim}
            return None
        self._confirmed[raw_cluster] = name
        self._buffers.pop(raw_cluster, None)
        return name
