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

import numpy as np

from ._constants import (
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
    ) -> None:
        self.tracker = tracker
        self.min_sec = min_sec
        self.max_buffer_sec = max_buffer_sec
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._confirmed: dict[str, str] = {}   # raw_cluster -> 確定名
        # 診断用（直近の照合試行の結果。UI/デバッグログでの可視化用途）。
        self.last_match: dict[str, object] | None = None

    def confirmed_name(self, raw_cluster: str) -> str | None:
        """既にこのクラスタに確定済みの名前があれば返す（無ければNone）."""
        return self._confirmed.get(raw_cluster)

    def reset(self) -> None:
        """会議リセット時に蓄積・確定状態をクリアする."""
        self._buffers.clear()
        self._confirmed.clear()
        self.last_match = None

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
        confirmed = self._confirmed.get(raw_cluster)
        if confirmed is not None:
            return confirmed
        if overlapped or wav.size == 0:
            return None
        buf = self._buffers.setdefault(raw_cluster, [])
        buf.append(np.asarray(wav, dtype=np.float32))
        total_samples = sum(a.size for a in buf)
        max_samples = int(self.max_buffer_sec * SR)
        while total_samples > max_samples and len(buf) > 1:
            total_samples -= buf.pop(0).size
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
            # confidence不足。バッファは維持し、次の観測でさらに蓄積してから再照合する。
            return None
        name, _confidence = match
        self._confirmed[raw_cluster] = name
        self._buffers.pop(raw_cluster, None)
        return name
