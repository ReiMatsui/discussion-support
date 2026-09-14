"""再生用のジッタバッファ（純ロジック。sounddevice には触らない）.

GPT-Live の音声は実時間で流れてくるが、到着は揺れる（ネットワーク、サーバの
生成の間、こちらの受信スレッドの遅れ）。届いた順に鳴らすと揺れがそのまま
音の途切れになる。ここでは一定量を溜めてから一定速度で取り出す。

- push(): 受信した PCM を積む。
- pull(n): 再生デバイスが n バイト要求したときに呼ぶ。溜めている最中や足りない
  ときは無音を返し、「溜め直し」に入る。
- 溜める量（target_ms）は固定ではなく、観測した揺れに合わせて広げる。尽きるたび
  に一段広げ、発話が滑らかに終わるたびに少し戻す。0.3〜1.5 秒。
- 溜まらないまま待たせすぎないよう、最後の push から wait_max_sec 以上たったら
  溜め切っていなくても鳴らす（発話の終わりの短い残りをいつまでも溜めない）。
"""
from __future__ import annotations

import threading
import time

_BYTES_PER_MS = 24000 * 2 // 1000   # 24kHz PCM16


class JitterBuffer:
    def __init__(self, *, target_ms: int = 400, min_ms: int = 300, max_ms: int = 1500,
                 step_ms: int = 200, wait_max_sec: float = 0.8):
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._buffering = True
        self._last_push_at: float | None = None
        self._ran_dry_at: float | None = None   # 尽きた時刻。次の push が続けば「途中で尽きた」
        self.target_ms = target_ms
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.step_ms = step_ms
        self.wait_max_sec = wait_max_sec
        # 観測（発話ごとに読んで記録する）
        self.underruns = 0            # 尽きた回数（通算）
        self.turn_underruns = 0       # この発話で尽きた回数
        self.max_gap_ms = 0           # この発話の到着の空白の最大（push 間隔）
        self.stall_ms = 0             # この発話で 150ms を超えた空白の合計

    # ---------------------------------------------------------------- 状態
    @property
    def buffered_ms(self) -> int:
        with self._lock:
            return len(self._buf) // _BYTES_PER_MS

    @property
    def buffering(self) -> bool:
        with self._lock:
            return self._buffering

    def push(self, pcm: bytes, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._last_push_at is not None and self._buf:
                gap = int((now - self._last_push_at) * 1000)
                if gap > self.max_gap_ms:
                    self.max_gap_ms = gap
                if gap > 150:
                    self.stall_ms += gap
            if self._ran_dry_at is not None:
                # 尽きた後にまだ音声が続いた＝発話の途中で尽きた（発話の終わりの
                # 鳴り終わりは数えない）。目標を一段広げる
                if now - self._ran_dry_at < 2.0:
                    self.underruns += 1
                    self.turn_underruns += 1
                    self.target_ms = min(self.max_ms, self.target_ms + self.step_ms)
                self._ran_dry_at = None
            self._last_push_at = now
            self._buf.extend(pcm)
            if self._buffering and len(self._buf) >= self.target_ms * _BYTES_PER_MS:
                self._buffering = False

    def pull(self, nbytes: int, *, now: float | None = None) -> bytes:
        """デバイスへ渡す nbytes を返す（足りなければ無音）."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._buffering:
                # 溜まらないまま長く待たない: 最後の push から wait_max_sec たったら
                # あるだけで鳴らす（発話の終わりの残りが目標に届かないとき）
                if (self._buf and self._last_push_at is not None
                        and now - self._last_push_at >= self.wait_max_sec):
                    self._buffering = False
                else:
                    return bytes(nbytes)
            if len(self._buf) >= nbytes:
                out = bytes(self._buf[:nbytes])
                del self._buf[:nbytes]
                return out
            # 尽きた。残りを捨てずに溜め直しへ。途中で尽きたかは次の push で分かる
            self._buffering = True
            if self._ran_dry_at is None:
                self._ran_dry_at = now
            return bytes(nbytes)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._buffering = True

    def begin_turn(self) -> None:
        """新しい発話の頭。観測をリセットし、溜めてから鳴らす."""
        with self._lock:
            self._buffering = True
            self.turn_underruns = 0
            self.max_gap_ms = 0
            self.stall_ms = 0
            self._last_push_at = None
            self._ran_dry_at = None

    def end_turn(self) -> dict:
        """発話の終わり。観測値を返し、滑らかに終われたなら目標を少し戻す."""
        with self._lock:
            stats = {"target_ms": self.target_ms, "underruns": self.turn_underruns,
                     "max_gap_ms": self.max_gap_ms, "stall_ms": self.stall_ms}
            if self.turn_underruns == 0:
                self.target_ms = max(self.min_ms, self.target_ms - self.step_ms // 4)
            return stats
