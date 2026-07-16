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
        merge_sim: float | None = None,
    ) -> None:
        self.tracker = tracker
        self.min_sec = min_sec
        self.max_buffer_sec = max_buffer_sec
        # クラスタ間名寄せ・最近傍統合の類似度下限。従来は tracker.dedupe
        # （3発話プロファイル同士の比較で校正された値）をそのまま流用していたが、
        # ここで比較するのは 5〜20秒の連結音声の埋め込み同士であり文脈が異なる
        # （docs/design/attribution_logic_review_2026-07.md C6/P4）。独立ノブに
        # 分離して校正可能にする。既定値は従来と同じ tracker.dedupe（モデル別:
        # redimnet 0.50 / ecapa 0.40 / resemblyzer 0.72）＝挙動不変。
        # 校正方法: GT付きライブ再実験の diag（type: cluster_naming）に出る
        # nearest_sim の分布（同一人物/別人）を見て決める。
        self.merge_sim = float(merge_sim) if merge_sim is not None \
            else float(tracker.dedupe)
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._confirmed: dict[str, str] = {}   # raw_cluster -> 確定名
        # クラスタ間名寄せ（docs/design/handoff_2026-07-14_unregistered_speakers.md §3）:
        # 登録者ゼロでもラベルの一貫性を保つため、未照合クラスタ同士を声紋埋め込みで
        # 比較し、似ていれば新規参加者を作らず既存クラスタ(canonical)へ統合する。
        self._embeddings: dict[str, np.ndarray] = {}   # canonical -> 代表埋め込み(L2正規化)
        self._aliases: dict[str, str] = {}             # 吸収された raw_cluster -> canonical
        # 診断用（直近の照合試行の結果。UI/デバッグログでの可視化用途）。
        self.last_match: dict[str, object] | None = None

    def confirmed_name(self, raw_cluster: str) -> str | None:
        """既にこのクラスタに確定済みの名前があれば返す（無ければNone）."""
        return self._confirmed.get(self.canonical_cluster(raw_cluster))

    def rename_confirmed(self, old: str, new: str) -> None:
        """確定名 old を new に付け替える（SessionState.rekey からの伝搬用）.

        UI /rename・stdin fix 等で表示キーが変わっても _confirmed が旧名の
        ままだと、observe() の確定短絡が旧名を返し続け、リネームした人物が
        別人格として復活する（docs/design/attribution_logic_review_2026-07.md
        C3）。rekey を状態一貫性の単一入口とし、ここへ伝搬する（P2）。
        """
        if old == new:
            return
        for cluster, name in list(self._confirmed.items()):
            if name == old:
                self._confirmed[cluster] = new

    def canonical_cluster(self, raw_cluster: str) -> str:
        """名寄せ（エイリアス）を解決した正規のクラスタキーを返す（無ければそのまま）.

        呼び出し側（_recv_loop.py）が匿名キーの発行・統合をこのキーで行うことで、
        吸収されたクラスタの発話が canonical 側の参加者に帰属する
        （docs/design/handoff_2026-07-14_unregistered_speakers.md §3 参照）。
        """
        seen: set[str] = set()
        while raw_cluster in self._aliases and raw_cluster not in seen:
            seen.add(raw_cluster)
            raw_cluster = self._aliases[raw_cluster]
        return raw_cluster

    def nearest_cluster(self, raw_cluster: str) -> tuple[str, float] | None:
        """自クラスタの代表埋め込みに最も近い他クラスタ(canonical)と類似度を返す.

        ``--diarization-max-speakers`` の上限到達後、新規参加者を増やさず
        最も近い既存参加者へ統合するための探索用
        （docs/design/handoff_2026-07-14_unregistered_speakers.md §3 の2）。
        統合してよいかの判断（類似度の下限）は呼び出し側が行うため、
        ここでは閾値をかけず ``(canonical, 類似度)`` を返す。
        自クラスタの埋め込みが未計算、または比較対象が無ければ None。
        （未使用だった exclude 引数は削除。
        docs/design/attribution_logic_review_2026-07.md D2）
        """
        key = self.canonical_cluster(raw_cluster)
        emb = self._embeddings.get(key)
        if emb is None:
            return None
        return self._nearest_embedding(emb, self_key=key)

    def _nearest_embedding(self, emb: np.ndarray, *, self_key: str,
                           ) -> tuple[str, float] | None:
        """埋め込み emb に最も近い他クラスタ(canonical)と類似度を返す共通実装.

        observe() の名寄せ探索と nearest_cluster() の重複実装を一本化した
        （片方だけ修正される事故の防止。
        docs/design/attribution_logic_review_2026-07.md D5）。
        """
        best, best_sim = None, -1.0
        for other, other_emb in self._embeddings.items():
            if other == self_key:
                continue
            sim = float(np.dot(emb, other_emb))
            if sim > best_sim:
                best, best_sim = other, sim
        if best is None:
            return None
        return best, best_sim

    def reset(self) -> None:
        """会議リセット時に蓄積・確定状態をクリアする."""
        self._buffers.clear()
        self._confirmed.clear()
        self._embeddings.clear()
        self._aliases.clear()
        self.last_match = None

    def _trim_buffer(self, buf: list[np.ndarray]) -> int:
        """バッファを max_buffer_sec に収まるよう古い方から捨て、総サンプル数を返す."""
        total = sum(a.size for a in buf)
        max_samples = int(self.max_buffer_sec * SR)
        while total > max_samples and len(buf) > 1:
            total -= buf.pop(0).size
        return total

    def _merge_cluster(self, absorbed: str, canonical: str) -> None:
        """absorbed を canonical へ名寄せする（バッファ統合＋エイリアス登録）.

        以後 absorbed 宛の音声は canonical のバッファに蓄積される
        （docs/design/handoff_2026-07-14_unregistered_speakers.md §3 参照）。
        """
        self._aliases[absorbed] = canonical
        moved = self._buffers.pop(absorbed, None)
        if moved:
            buf = self._buffers.setdefault(canonical, [])
            buf.extend(moved)
            self._trim_buffer(buf)
        self._embeddings.pop(absorbed, None)

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
        # 名寄せ済みクラスタの音声は canonical のバッファに蓄積する（シンプルさ優先。
        # docs/design/handoff_2026-07-14_unregistered_speakers.md §3 参照）。
        raw_cluster = self.canonical_cluster(raw_cluster)
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
            # 登録プロファイルとの照合は不成立。ここでクラスタ間名寄せを試みる
            # （docs/design/handoff_2026-07-14_unregistered_speakers.md §3）:
            # 登録者ゼロでも、既存クラスタの代表埋め込みと tracker.dedupe 以上に
            # 似ていれば「同一人物のクラスタ分裂」とみなし、新規参加者を作らず
            # 既存クラスタ(canonical)へ統合する。
            emb = self.tracker.embed(concat)
            if emb is not None:
                nearest = self._nearest_embedding(emb, self_key=raw_cluster)
                best, best_sim = nearest if nearest is not None else (None, -1.0)
                if best is not None and best_sim >= self.merge_sim:
                    self._merge_cluster(raw_cluster, best)
                    self.last_match = {"kind": "クラスタ名寄せ", "raw": raw_cluster,
                                       "canonical": best, "sim": round(best_sim, 3)}
                    # canonical が確定済みならその名前で帰属できる。未確定なら None を
                    # 返し、呼び出し側が canonical_cluster() で匿名キーを解決する。
                    return self._confirmed.get(best)
                # 名寄せも不成立: 次回の名寄せ先候補として代表埋め込みを最新の
                # concat で更新しておく（音声が増えるほど代表性が上がる想定）。
                # 最近傍とその類似度は diag に残し、merge_sim の校正材料にする
                # （不成立側の分布が見えないと閾値を調整できない。review P4）。
                if best is not None:
                    self.last_match["nearest"] = best
                    self.last_match["nearest_sim"] = round(best_sim, 3)
                self._embeddings[raw_cluster] = emb
            # confidence不足。バッファは維持し、次の観測でさらに蓄積してから再照合する。
            return None
        name, _confidence = match
        self._confirmed[raw_cluster] = name
        # 確定クラスタは以後の未照合クラスタの名寄せ先として引き続き有用なので、
        # 確定時にも代表埋め込みを保存する（§3 参照）。match_profile は副作用なし
        # APIで内部の埋め込みを取り出せないため、ここで embed を1回だけ呼ぶ
        # （クラスタの確定は一度きりなので追加コストは限定的）。
        emb = self.tracker.embed(concat)
        if emb is not None:
            self._embeddings[raw_cluster] = emb
        self._buffers.pop(raw_cluster, None)
        return name
