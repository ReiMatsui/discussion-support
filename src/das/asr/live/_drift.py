"""脱線の状態機械（純ロジック）。本番のチェッカーと replay の両方が使う."""
from __future__ import annotations

from ._constants import _DRIFT_REPEAT_SEC


class DriftRun:
    """「議題と無関係な話が続いている時間」を測る状態機械（純ロジック、LLM は呼ばない）.

    脱線の本質は一言が外れたことではなく、離れた状態がどれだけ続いているか。
    判定器が各発話に付けた on / aside / off を受け取り、末尾から連続する off の
    区間（aside は区間を切らないが数えもしない）を追う。区間の長さが閾値を超えたら
    戻す候補を出し、戻した後も離れたままなら _DRIFT_REPEAT_SEC あけてもう一度。
    誰かが議題に戻れば（on）区間は消える——自力で戻る会話には何も言わない。

    論点の軸は「議題」と「離れる前に出ていた論点」に固定する。離れている間に
    論点抽出が拾った話題を軸に入れると、脱線先が論点になった瞬間に
    「関連話題内」に化けて戻せなくなる（2026-09-13 のシミュレーションで再現）。
    """

    def __init__(self) -> None:
        self.run_start_ms: int | None = None     # off 区間の最初の発話の開始（会議ms）
        self.run_end_ms: int | None = None       # off 区間の最後の発話の終了（会議ms）
        self.reason = ""
        self.anchor_topics: list[str] = []       # 離れる前に出ていた論点（軸）
        self.last_fired_ms: int | None = None    # この区間で最後に候補を出した位置
        self.fires_in_run = 0

    @property
    def active(self) -> bool:
        return self.run_start_ms is not None

    def run_sec(self) -> float:
        if self.run_start_ms is None or self.run_end_ms is None:
            return 0.0
        return max(0.0, (self.run_end_ms - self.run_start_ms) / 1000.0)

    def update_anchors(self, topics: list[str]) -> None:
        """離れていない間だけ軸を更新する."""
        if not self.active:
            self.anchor_topics = list(topics)

    def observe(self, window: list[dict], labels: list[str], reason: str) -> None:
        """判定結果を取り込む。window の各要素は {"ms", "end_ms"} を持つ（None 可）."""
        if not window or len(window) != len(labels):
            return
        # 末尾から: aside は飛ばし、off は区間に含め、on で切る
        off_idx: list[int] = []
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] == "aside":
                continue
            if labels[i] == "off":
                off_idx.append(i)
                continue
            break
        if not off_idx:
            self._clear()
            return
        first, last = off_idx[-1], off_idx[0]
        start_ms = window[first].get("ms")
        end_ms = window[last].get("end_ms") or window[last].get("ms")
        if start_ms is None or end_ms is None:
            return
        if not self.active:
            self.run_start_ms = int(start_ms)
        else:
            self.run_start_ms = min(self.run_start_ms, int(start_ms))   # 窓の中でより古い開始
        self.run_end_ms = max(int(end_ms), self.run_end_ms or 0)
        if reason:
            self.reason = reason

    def _clear(self) -> None:
        self.run_start_ms = None
        self.run_end_ms = None
        self.reason = ""
        self.last_fired_ms = None
        self.fires_in_run = 0

    def should_fire(self, *, threshold_sec: float) -> str | None:
        """候補を出すべきなら理由を返す（出したことも記録する）."""
        if not self.active or self.run_sec() < threshold_sec:
            return None
        if self.last_fired_ms is not None:
            since_ms = (self.run_end_ms or 0) - self.last_fired_ms
            if since_ms < _DRIFT_REPEAT_SEC * 1000:
                return None
        self.last_fired_ms = self.run_end_ms
        self.fires_in_run += 1
        reason = self.reason or "議題と無関係な話"
        if self.fires_in_run > 1:
            reason = f"{reason}（まだ続いています）"
        return reason
