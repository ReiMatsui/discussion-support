"""Visual speaker verification + ASR PoC.

Mac 上で動く検証スクリプト。カメラ映像から「対象人物が口を動かしている (=
visually speaking)」を判定し、その間にマイクで拾えた音声 (=audio active) と
時間的に一致したセグメントだけを Whisper に渡して日本語認識する。

目的:
- 隣の人の声がマイクに漏れても、カメラの正面に映っている人物の口が動いていな
  ければ「対象話者の発話ではない」と判定して捨てる。
- discussion-support 本体に組み込む前段として、視覚 VAD + 音声 VAD の AND
  ロジックが実用に耐えるか確認する。

依存:
    uv pip install opencv-python mediapipe sounddevice numpy faster-whisper
    # mlx-whisper を使う場合 (Apple Silicon, 高速):
    uv pip install mlx-whisper

iPhone を webcam として使う場合:
    macOS Ventura 以降 + iOS 16 以降の iPhone を同じ Apple ID でサインインし、
    Bluetooth/Wi-Fi を ON にしたうえで、Mac 側で本スクリプトを起動する直前に
    iPhone を Mac の近くに置く。OpenCV の cv2.VideoCapture(index) は
    Continuity Camera を通常の Web カメラとして列挙するので、--camera-index で
    切り替える。利用可能なデバイスは --list-devices で確認できる。

使い方:
    # 既定 (カメラ 0 + 既定マイク + faster-whisper small)
    python scripts/visual_asr_poc.py

    # デバイス一覧を確認
    python scripts/visual_asr_poc.py --list-devices

    # iPhone を Continuity Camera で接続した場合 (index は実機で確認)
    python scripts/visual_asr_poc.py --camera-index 1 --audio-device "iPhone Microphone"

    # mlx-whisper を使う
    python scripts/visual_asr_poc.py --asr-backend mlx --asr-model mlx-community/whisper-large-v3-turbo

    # しきい値を調整
    python scripts/visual_asr_poc.py --mar-var-threshold 0.0008 --audio-rms-threshold 0.012
"""

from __future__ import annotations

import argparse
import collections
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np

# 重い依存 (cv2 / mediapipe / sounddevice) は実際に使う関数の中で遅延 import
# する。これにより coordinator_loop / SharedState などの純粋ロジック部分を
# 軽量にテストできる (scripts/test_visual_asr_logic.py を参照)。


# --------------------------------------------------------------------------- #
# 設定
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 16_000  # Whisper の入力に合わせる
AUDIO_BLOCK_MS = 30  # 1 audio frame = 30 ms (480 samples @ 16 kHz)
AUDIO_BLOCK = SAMPLE_RATE * AUDIO_BLOCK_MS // 1000

# MediaPipe FaceMesh のランドマーク index。
# 0..467 のうち、口周りで MAR を計算するのに使う点を選定。
# 参考: https://github.com/google-ai-edge/mediapipe (canonical face model)
MOUTH_LM_HORIZONTAL = (61, 291)  # 左右の口角 (外側)
MOUTH_LM_VERTICAL = (
    (13, 14),    # 中央上唇内側 / 中央下唇内側
    (81, 178),   # 左寄りの上下
    (311, 402),  # 右寄りの上下
)


# --------------------------------------------------------------------------- #
# データ構造
# --------------------------------------------------------------------------- #


@dataclass
class SharedState:
    """音声スレッド / 映像スレッド / ASR ワーカが共有する状態。

    音声ブロックの受け渡しは ``audio_q`` (queue.Queue) で行い、
    映像側からの判定結果と RMS は ``lock`` 配下の単純フィールドで共有する。
    """

    # --- 映像スレッドが書く / コーディネータが読む ---
    visually_speaking: bool = False
    last_mar: float = 0.0
    last_mar_var: float = 0.0
    face_detected: bool = False

    # --- 音声コールバックが書く / コーディネータが読む ---
    last_rms: float = 0.0  # スムージング後 (overlay 表示にも使う)

    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class Utterance:
    pcm: np.ndarray  # float32, mono, 16kHz
    started_at: float
    duration_s: float


# --------------------------------------------------------------------------- #
# 映像 + 口パク検出
# --------------------------------------------------------------------------- #


def compute_mar(landmarks, image_w: int, image_h: int) -> float:
    """MediaPipe の landmarks (NormalizedLandmarkList) から MAR を計算。"""
    px = [(lm.x * image_w, lm.y * image_h) for lm in landmarks]

    def dist(a: int, b: int) -> float:
        ax, ay = px[a]
        bx, by = px[b]
        return float(np.hypot(ax - bx, ay - by))

    horizontal = dist(*MOUTH_LM_HORIZONTAL)
    if horizontal < 1e-6:
        return 0.0
    verticals = [dist(a, b) for a, b in MOUTH_LM_VERTICAL]
    return float(np.mean(verticals) / horizontal)


def video_loop(
    state: SharedState,
    camera_index: int,
    mar_window_frames: int,
    mar_var_threshold: float,
    show_preview: bool,
    log: logging.Logger,
) -> None:
    """カメラから連続的にフレームを取り、口の動きを判定するループ。"""

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.solutions import face_mesh as mp_face_mesh
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "opencv-python と mediapipe が必要です: uv add opencv-python mediapipe"
        ) from exc

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        log.error("camera_index=%s を開けませんでした", camera_index)
        state.stop_event.set()
        return

    # MediaPipe FaceMesh を使う。Face Landmarker でも良いが、追加モデル DL 不要な
    # 旧 API を採用 (PoC なので)。refine_landmarks=False で軽量化。
    # mp.solutions は lazy 属性なので、上で明示 import 済みの mp_face_mesh を使う。
    _ = mp  # 参照を保持 (linter 対策)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    mar_history: collections.deque[float] = collections.deque(maxlen=mar_window_frames)

    try:
        while not state.stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            # 鏡像のほうが直感的なのでフリップ
            frame_bgr = cv2.flip(frame_bgr, 1)
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(frame_rgb)
            face_detected = bool(results.multi_face_landmarks)

            mar = 0.0
            mar_var = 0.0
            visually_speaking = False
            if face_detected:
                landmarks = results.multi_face_landmarks[0].landmark
                mar = compute_mar(landmarks, w, h)
                mar_history.append(mar)
                if len(mar_history) >= max(5, mar_window_frames // 2):
                    mar_var = float(np.var(mar_history))
                    visually_speaking = mar_var > mar_var_threshold
            else:
                mar_history.clear()

            with state.lock:
                state.visually_speaking = visually_speaking
                state.last_mar = mar
                state.last_mar_var = mar_var
                state.face_detected = face_detected

            if show_preview:
                _draw_overlay(cv2, frame_bgr, mar, mar_var, visually_speaking, face_detected, state)
                cv2.imshow("visual-asr-poc (q to quit)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    state.stop_event.set()
                    break
    finally:
        cap.release()
        face_mesh.close()
        if show_preview:
            cv2.destroyAllWindows()


def _draw_overlay(
    cv2,  # noqa: ANN001  (lazy-imported module)
    frame: np.ndarray,
    mar: float,
    mar_var: float,
    visually_speaking: bool,
    face_detected: bool,
    state: SharedState,
) -> None:
    color = (0, 200, 0) if visually_speaking else (0, 0, 200)
    if not face_detected:
        color = (128, 128, 128)
    lines = [
        f"face: {'OK' if face_detected else 'NO'}",
        f"MAR: {mar:.3f}",
        f"MAR var: {mar_var:.5f}",
        f"visual speak: {visually_speaking}",
        f"audio RMS: {state.last_rms:.4f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 24 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )


# --------------------------------------------------------------------------- #
# 音声収集
# --------------------------------------------------------------------------- #


def make_audio_callback(state: SharedState, audio_q: queue.Queue, rms_alpha: float = 0.3):
    """sounddevice.InputStream に渡すコールバックを生成。

    指定された ``audio_q`` に PCM ブロック (np.ndarray, float32, mono) を push し、
    最新の RMS は ``state.last_rms`` に書き込む。
    """

    smoothed_rms = 0.0

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        nonlocal smoothed_rms
        if status:
            # オーバーフロー等は警告するが、止めない
            sys.stderr.write(f"[audio] {status}\n")

        # indata は float32, shape=(frames, channels)。モノラル想定。
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        rms = float(np.sqrt(np.mean(chunk * chunk)))
        smoothed_rms = (1 - rms_alpha) * smoothed_rms + rms_alpha * rms

        with state.lock:
            state.last_rms = smoothed_rms

        # 無制限に積むと暴走するので、過大なバックログは捨てる (~30s 相当)
        if audio_q.qsize() < 1000:
            audio_q.put(chunk)

    return callback


# --------------------------------------------------------------------------- #
# 統合ループ (発話切り出し)
# --------------------------------------------------------------------------- #


def coordinator_loop(
    state: SharedState,
    audio_q: queue.Queue,
    asr_queue: queue.Queue,
    audio_rms_threshold: float,
    min_utterance_s: float,
    end_silence_s: float,
    log: logging.Logger,
) -> None:
    """音声と映像の判定を組み合わせ、対象話者の発話を切り出す。

    音声コールバックから ``audio_q`` に流れてくる PCM ブロック (30ms) を消化しな
    がら、映像側の最新判定 (``state.visually_speaking``) と論理 AND を取り、
    speaker_verified == True が続く区間を蓄積する。
    silence が ``end_silence_s`` 以上続いたら ASR キューに投入する。
    """

    block_dt = AUDIO_BLOCK / SAMPLE_RATE

    utterance_buffer: list[np.ndarray] = []
    utterance_started_at = 0.0
    silence_seconds = 0.0
    # 余韻 (発話後の無音) を含めるためのプリロール上限
    max_trailing = int(end_silence_s / block_dt) + 1

    while not state.stop_event.is_set():
        try:
            chunk = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue

        # ブロック単位の RMS (audio active 判定はここで直接計算したほうが
        # スムージングの遅延の影響を受けないので妥当)
        block_rms = float(np.sqrt(np.mean(chunk * chunk)))
        audio_active = block_rms > audio_rms_threshold

        with state.lock:
            visually_speaking = state.visually_speaking

        speaker_verified = visually_speaking and audio_active

        if speaker_verified:
            if not utterance_buffer:
                utterance_started_at = time.time()
            utterance_buffer.append(chunk)
            silence_seconds = 0.0
        else:
            if utterance_buffer:
                # 余韻として末尾に少しだけ無音を含める (Whisper の認識精度のため)
                if silence_seconds < end_silence_s and len(utterance_buffer) < max_trailing * 10:
                    utterance_buffer.append(chunk)
                silence_seconds += block_dt
                if silence_seconds >= end_silence_s:
                    pcm = np.concatenate(utterance_buffer)
                    duration_s = pcm.size / SAMPLE_RATE
                    if duration_s >= min_utterance_s:
                        log.info(
                            "utterance flush: %.2fs (started %.1fs ago)",
                            duration_s,
                            time.time() - utterance_started_at,
                        )
                        asr_queue.put(
                            Utterance(pcm=pcm, started_at=utterance_started_at, duration_s=duration_s)
                        )
                    else:
                        log.debug("utterance too short (%.2fs), dropped", duration_s)
                    utterance_buffer = []
                    utterance_started_at = 0.0
                    silence_seconds = 0.0


# --------------------------------------------------------------------------- #
# ASR ワーカ
# --------------------------------------------------------------------------- #


class FasterWhisperASR:
    def __init__(self, model_name: str, language: str = "ja") -> None:
        from faster_whisper import WhisperModel

        # Apple Silicon でも CPU で十分動くサイズを推奨 (small / medium)。
        # GPU が無くても float16 → int8 で十分速い。
        self.model = WhisperModel(model_name, device="auto", compute_type="auto")
        self.language = language

    def transcribe(self, pcm: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            pcm,
            language=self.language,
            vad_filter=False,  # 入口で既に切り出してあるので不要
            beam_size=1,  # PoC なので速度優先
        )
        return "".join(seg.text for seg in segments).strip()


class MLXWhisperASR:
    def __init__(self, model_repo: str, language: str = "ja") -> None:
        import mlx_whisper  # noqa: F401  (動作確認)

        self.model_repo = model_repo
        self.language = language

    def transcribe(self, pcm: np.ndarray) -> str:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            pcm,
            path_or_hf_repo=self.model_repo,
            language=self.language,
            condition_on_previous_text=False,
        )
        return str(result.get("text", "")).strip()


def asr_worker(state: SharedState, asr_queue: queue.Queue, asr, log: logging.Logger) -> None:  # noqa: ANN001
    while not state.stop_event.is_set():
        try:
            utt: Utterance = asr_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            t0 = time.time()
            text = asr.transcribe(utt.pcm)
            dt = time.time() - t0
            if text:
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"({utt.duration_s:.1f}s音声 → {dt:.2f}s認識) {text}",
                    flush=True,
                )
            else:
                log.info("ASR returned empty text for %.2fs audio", utt.duration_s)
        except Exception as exc:  # noqa: BLE001
            log.exception("ASR failure: %s", exc)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def list_devices() -> None:
    try:
        import cv2
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "opencv-python と sounddevice が必要です: uv add opencv-python sounddevice"
        ) from exc
    print("=== Video (cv2.VideoCapture indices) ===")
    for i in range(6):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        if ok:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [{i}] OK  ({w}x{h})")
        cap.release()
    print()
    print("=== Audio (sounddevice) ===")
    print(sd.query_devices())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-devices", action="store_true", help="カメラ / マイクを列挙して終了")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--audio-device", default=None, help="sounddevice のデバイス名 or index")

    p.add_argument("--mar-window-ms", type=int, default=500, help="MAR 分散を取る時間窓 (ms)")
    p.add_argument("--mar-var-threshold", type=float, default=0.0006, help="MAR 分散の閾値 (環境依存。--show-preview で値を見ながら調整)")
    p.add_argument("--audio-rms-threshold", type=float, default=0.01, help="無音判定の RMS 閾値")

    p.add_argument("--min-utterance-s", type=float, default=0.4)
    p.add_argument("--end-silence-s", type=float, default=0.5, help="この秒数だけ speaker_verified=False が続いたら発話終了とみなす")

    p.add_argument("--asr-backend", choices=["faster-whisper", "mlx", "none"], default="faster-whisper")
    p.add_argument("--asr-model", default="small", help="faster-whisper の場合はモデル名 (tiny/base/small/medium/large-v3 など)、mlx の場合は HuggingFace repo")
    p.add_argument("--language", default="ja")

    p.add_argument("--show-preview", action="store_true", default=True)
    p.add_argument("--no-preview", dest="show_preview", action="store_false")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("visual-asr-poc")

    if args.list_devices:
        list_devices()
        return 0

    state = SharedState()

    # ASR バックエンド
    asr = None
    if args.asr_backend == "faster-whisper":
        log.info("loading faster-whisper model: %s", args.asr_model)
        asr = FasterWhisperASR(args.asr_model, language=args.language)
    elif args.asr_backend == "mlx":
        log.info("loading mlx-whisper model: %s", args.asr_model)
        asr = MLXWhisperASR(args.asr_model, language=args.language)
    else:
        log.warning("ASR を無効化しています (発話切り出しのテストのみ)")

    # Ctrl-C で全スレッド止める
    def handle_sigint(*_: object) -> None:
        log.info("SIGINT received, shutting down...")
        state.stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    # ASR キュー & ワーカ
    asr_queue: queue.Queue = queue.Queue()
    asr_thread = None
    if asr is not None:
        asr_thread = threading.Thread(target=asr_worker, args=(state, asr_queue, asr, log), daemon=True)
        asr_thread.start()

    # 映像スレッド
    mar_window_frames = max(5, int(args.mar_window_ms / 1000 * 30))  # 30 fps 仮定
    video_thread = threading.Thread(
        target=video_loop,
        args=(
            state,
            args.camera_index,
            mar_window_frames,
            args.mar_var_threshold,
            args.show_preview,
            log,
        ),
        daemon=True,
    )
    video_thread.start()

    # 音声ストリーム
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("sounddevice が必要です: uv add sounddevice") from exc
    audio_q: queue.Queue = queue.Queue()
    audio_cb = make_audio_callback(state, audio_q)
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=AUDIO_BLOCK,
            channels=1,
            dtype="float32",
            device=args.audio_device,
            callback=audio_cb,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("マイクを開けませんでした: %s", exc)
        state.stop_event.set()
        return 1

    # コーディネータ (発話切り出し) — メインスレッドで回す
    with stream:
        log.info("started. Ctrl-C to stop. preview window q キーでも終了します。")
        try:
            coordinator_loop(
                state,
                audio_q,
                asr_queue,
                audio_rms_threshold=args.audio_rms_threshold,
                min_utterance_s=args.min_utterance_s,
                end_silence_s=args.end_silence_s,
                log=log,
            )
        finally:
            state.stop_event.set()

    video_thread.join(timeout=2)
    if asr_thread is not None:
        asr_thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
