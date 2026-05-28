"""coordinator_loop の論理だけを合成データで検証するスモークテスト.

カメラもマイクも Whisper も使わず、speaker_verified (= visual AND audio) が
立っている区間だけが ASR キューに到達することを確認する。

実行:
    python scripts/test_visual_asr_logic.py
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from visual_asr_poc import (
    AUDIO_BLOCK,
    SAMPLE_RATE,
    SharedState,
    Utterance,
    coordinator_loop,
)


def _silent_block() -> np.ndarray:
    return np.zeros(AUDIO_BLOCK, dtype=np.float32)


def _loud_block(rms: float = 0.05) -> np.ndarray:
    # ホワイトノイズ的な信号を擬似的に作る
    rng = np.random.default_rng()
    x = rng.standard_normal(AUDIO_BLOCK).astype(np.float32)
    x *= rms / max(float(np.sqrt(np.mean(x * x))), 1e-9)
    return x


def _drive(
    state: SharedState,
    audio_q: queue.Queue,
    pattern: list[tuple[bool, bool, int]],
    block_dt: float,
) -> None:
    """pattern = [(visually_speaking, audio_loud, n_blocks), ...] を順に投入."""
    for visually_speaking, audio_loud, n in pattern:
        with state.lock:
            state.visually_speaking = visually_speaking
        for _ in range(n):
            audio_q.put(_loud_block() if audio_loud else _silent_block())
            # コーディネータが消化する時間を作る
            time.sleep(block_dt * 0.5)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("test")

    state = SharedState()
    audio_q: queue.Queue = queue.Queue()
    asr_q: queue.Queue = queue.Queue()

    block_dt = AUDIO_BLOCK / SAMPLE_RATE

    coord = threading.Thread(
        target=coordinator_loop,
        kwargs=dict(
            state=state,
            audio_q=audio_q,
            asr_queue=asr_q,
            audio_rms_threshold=0.01,
            min_utterance_s=0.2,
            end_silence_s=0.3,
            log=log,
        ),
        daemon=True,
    )
    coord.start()

    # シナリオ:
    #   1) 視覚 false / 音声 大 (隣の人がしゃべってる)              → 拾わない
    #   2) 視覚 true  / 音声 小 (口は動いてるが声は無音、声楽の真似)  → 拾わない
    #   3) 視覚 true  / 音声 大 (本人がしゃべってる)                 → 拾う
    #   4) 視覚 false / 音声 小 (静か)                              → 終端
    #   5) 視覚 false / 音声 大 (再び隣)                            → 拾わない
    #   6) 視覚 true  / 音声 大 (本人がもう一度)                    → 拾う
    #   7) 視覚 false / 音声 小                                     → 終端
    pattern = [
        (False, True, 30),    # ~0.9s
        (True, False, 20),    # ~0.6s
        (True, True, 40),     # ~1.2s   ← utterance #1
        (False, False, 20),   # ~0.6s   (>= end_silence_s で flush)
        (False, True, 30),    # ~0.9s
        (True, True, 30),     # ~0.9s   ← utterance #2
        (False, False, 20),
    ]
    _drive(state, audio_q, pattern, block_dt)

    # コーディネータに残った仕事を消化させる
    time.sleep(1.0)
    state.stop_event.set()
    coord.join(timeout=2)

    utterances: list[Utterance] = []
    while not asr_q.empty():
        utterances.append(asr_q.get_nowait())

    print(f"\n=== 受け取った utterance: {len(utterances)} 件 ===")
    for i, u in enumerate(utterances, 1):
        print(f"  #{i}: {u.duration_s:.2f}s")

    expected = 2
    if len(utterances) != expected:
        print(f"FAIL: expected {expected} utterances, got {len(utterances)}")
        return 1
    # それぞれが妥当な長さ (1.0〜1.5s 程度) か確認
    for i, u in enumerate(utterances, 1):
        if not (0.5 < u.duration_s < 2.5):
            print(f"FAIL: utterance #{i} duration suspicious: {u.duration_s:.2f}s")
            return 1
    print("OK: AND ゲート (視覚 AND 音声) が期待通り動いている")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
