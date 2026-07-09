#!/usr/bin/env python3
"""pyannoteAI Live-1 (ストリーミングWS) の実測スクリプト.

目的:
  scripts/benchmark_pyannote.py はバッチ diarization (``POST /v1/diarize``) の
  オフライン比較だったが、本スクリプトは実際に ``PyannoteStreamingDiarizationProvider``
  (``src/das/asr/live/_pyannote_diarization.py``, ``POST /v1/live`` WebSocket) を使い、
  既存の録音wavを実時間でストリーミングして Live-1 の話者分離を実測する。

  Live-1 はサーバ側が「実時間 + 最大5秒バッファ」までしか先行受信を許容しない
  （_pyannote_diarization.py の docstring 参照）。そのため録音を一気に送るのではなく、
  100ms チャンクを wall-clock で 100ms 間隔にペーシングして送信する。
  26分のセッションを丸ごと待つのは検証コストが高いため、既定では先頭5分だけを
  流す（``--head-minutes`` で変更可）。

前提:
  - 対象wavは 16kHz mono PCM16 を想定。wave モジュールでヘッダを検証し、
    一致しなければ明確なエラーで停止する（無音のズレたデータを送って
    課金だけ発生させる事故を防ぐ）。
  - APIキーは scripts/benchmark_pyannote.py の ``_load_dotenv_fallback`` /
    ``resolve_api_key`` をそのまま再利用し、.env の PYANNOTEAI_API_KEY
    (フォールバック PYANNOTE_API_KEY) から読む。無ければ明確なエラーで停止する。

使い方:
  # transcripts/2026-06-25_1554.wav + .turns.jsonl を解決し、先頭5分だけ流す
  uv run python scripts/test_pyannote_live.py --session 2026-06-25_1554

  # 先頭2分だけ（動作確認を素早く回したい時）
  uv run python scripts/test_pyannote_live.py --session 2026-06-25_1554 --head-minutes 2

  # wavを直接指定
  uv run python scripts/test_pyannote_live.py --wav transcripts/2026-06-25_1554.wav

  Ctrl-C で安全に停止（end_of_stream を送ってからソケットを閉じる）。

出力:
  - 標準出力に受信イベントを逐次 ``[mm:ss] SPEAKER_XX start/end`` で表示
  - 終了時に確定した話者区間一覧を transcripts/<session>.pyannote_live.json に保存
  - 流した範囲の turns.jsonl と突き合わせたサマリ
    （scripts/benchmark_pyannote.py の compare_session 等を再利用）
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import wave
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from benchmark_pyannote import (  # noqa: E402
    DEFAULT_UNKNOWN_LABEL,
    PyannoteSegment,
    Turn,
    TRANSCRIPTS_DIR,
    _load_dotenv_fallback,
    compare_session,
    load_turns,
    print_summary,
    resolve_sessions,
)

from das.asr.live._pyannote_diarization import (  # noqa: E402
    PyannoteStreamingDiarizationProvider,
)
from das.asr.live._constants import SR  # noqa: E402
from das.asr.live._diarization import DiarizationEvent  # noqa: E402

_CHUNK_MS = 100
_CHUNK_SAMPLES = SR * _CHUNK_MS // 1000
_CHUNK_BYTES = _CHUNK_SAMPLES * 2  # PCM16 = 2 bytes/sample


def fmt_mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# wav 検証
# ---------------------------------------------------------------------------


def validate_wav_16k_mono_pcm16(wav_path: Path) -> int:
    """16kHz mono PCM16 であることを検証し、フレーム数を返す。違えばエラー終了。"""
    try:
        with wave.open(str(wav_path), "rb") as w:
            sr = w.getframerate()
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            nframes = w.getnframes()
    except (OSError, wave.Error) as exc:
        print(f"エラー: {wav_path} を wav として読めませんでした: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    problems = []
    if sr != 16000:
        problems.append(f"サンプルレート={sr}Hz (16000Hz想定)")
    if channels != 1:
        problems.append(f"チャンネル数={channels} (mono=1想定)")
    if sampwidth != 2:
        problems.append(f"サンプル幅={sampwidth}バイト (PCM16=2バイト想定)")
    if problems:
        print(
            f"エラー: {wav_path} は想定フォーマットと一致しません: {', '.join(problems)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return nframes


def read_pcm16_bytes(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as w:
        return w.readframes(w.getnframes())


# ---------------------------------------------------------------------------
# ストリーミング実行
# ---------------------------------------------------------------------------


class _Interrupted(Exception):
    pass


def stream_live(
    pcm16: bytes,
    api_key: str,
    *,
    head_minutes: float,
) -> list[DiarizationEvent]:
    """PCM16音声を100ms刻み・実時間ペースでLive-1に流し、確定イベントを集める。

    Ctrl-C (SIGINT) を受けたら、ループを安全に抜けて close() で
    end_of_stream を送ってから戻る（生ソケットを黙って閉じない）。
    """
    max_bytes = int(head_minutes * 60 * SR * 2)
    data = pcm16[:max_bytes] if max_bytes > 0 else pcm16
    total_chunks = (len(data) + _CHUNK_BYTES - 1) // _CHUNK_BYTES

    provider = PyannoteStreamingDiarizationProvider(api_key)
    print(f"Live-1 セッション作成中... (対象: 約{len(data) / (SR * 2):.1f}秒 / {total_chunks}チャンク)")
    provider.start()
    print(f"接続完了 (stream_id={provider.stream_id})。ストリーミング開始（100ms/チャンク、実時間ペース）")

    collected: list[DiarizationEvent] = []
    interrupted = False

    def _on_sigint(signum: object, frame: object) -> None:
        raise _Interrupted()

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    start_wall = time.monotonic()
    try:
        offset = 0
        chunk_idx = 0
        while offset < len(data):
            chunk_start_wall = time.monotonic()
            chunk = data[offset : offset + _CHUNK_BYTES]
            offset += _CHUNK_BYTES
            chunk_idx += 1
            provider.send_audio(chunk)

            for ev in provider.drain_events():
                collected.append(ev)
                _print_event(ev)

            # 実時間ペーシング: このチャンクの再生時間ぶん経過するまで待つ
            target_wall = start_wall + chunk_idx * (_CHUNK_MS / 1000.0)
            sleep_for = target_wall - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

            if chunk_idx % 100 == 0:
                elapsed = time.monotonic() - start_wall
                print(
                    f"  ...{chunk_idx}/{total_chunks}チャンク送信済み "
                    f"(経過{elapsed:.0f}秒 / 音声内位置{fmt_mmss(chunk_idx * _CHUNK_MS)})",
                    file=sys.stderr,
                )
    except _Interrupted:
        interrupted = True
        print("\n中断シグナル received。end_of_stream を送って安全に終了します...", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    print("ストリーミング終了。残りの確定イベントを待機中(close)...")
    provider.close()
    for ev in provider.drain_events():
        collected.append(ev)
        _print_event(ev)

    # サーバがend_of_stream受理後もクローズ待ちの間に来たactive(未確定)分は
    # 終端側では確定しない仕様なので、参考として残す（closedイベントのみ返す）。
    if interrupted:
        print("(中断により途中までの結果です)", file=sys.stderr)
    return collected


def _print_event(ev: DiarizationEvent) -> None:
    start_s = fmt_mmss(ev.start_ms)
    if ev.end_ms is None:
        print(f"[{start_s}] {ev.speaker} start")
    else:
        end_s = fmt_mmss(ev.end_ms)
        print(f"[{start_s}-{end_s}] {ev.speaker} (start={start_s} end={end_s})")


# ---------------------------------------------------------------------------
# 保存・比較
# ---------------------------------------------------------------------------


def events_to_segments(events: list[DiarizationEvent]) -> list[PyannoteSegment]:
    segments = []
    for ev in events:
        if ev.end_ms is None or ev.end_ms <= ev.start_ms:
            continue
        segments.append(PyannoteSegment(speaker=ev.speaker, start_ms=ev.start_ms, end_ms=ev.end_ms))
    segments.sort(key=lambda s: s.start_ms)
    return segments


def save_live_result(
    out_path: Path,
    *,
    session_id: str,
    wav_path: Path,
    head_minutes: float,
    segments: list[PyannoteSegment],
) -> None:
    payload = {
        "session": session_id,
        "wav_path": str(wav_path),
        "head_minutes": head_minutes,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segments": [asdict(s) for s in segments],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clip_turns_to_range(turns: list[Turn], end_ms: int) -> list[Turn]:
    """流した範囲(0..end_ms)に収まるターンだけを比較対象にする.

    end_ms を跨ぐターンは比較対象から除く（範囲外の音声と重なるため
    見かけ上の不一致になりうる）。
    """
    clipped: list[Turn] = []
    for t in turns:
        if t.ms is None:
            continue
        if t.ms >= end_ms:
            continue
        if t.end_ms is not None and t.end_ms > end_ms:
            continue
        clipped.append(t)
    return clipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--session",
        action="append",
        default=[],
        metavar="2026-06-25_1554",
        help="transcripts/<session>.wav と .turns.jsonl のペアを解決（1件のみ想定。"
        "複数指定時は先頭のみ使う）",
    )
    ap.add_argument(
        "--wav",
        action="append",
        default=[],
        metavar="PATH",
        help="wavパスを直接指定（同名の .turns.jsonl を自動解決）",
    )
    ap.add_argument(
        "--head-minutes",
        type=float,
        default=5.0,
        help="先頭何分だけを実時間ストリーミングするか（既定: 5分。0以下で全編）",
    )
    ap.add_argument("--api-key", default=None, help="pyannoteAI APIキー（省略時は.env/環境変数）")
    ap.add_argument(
        "--unknown-label",
        default=DEFAULT_UNKNOWN_LABEL,
        help=f"現行システムの『不明/未確定』話者ラベル（既定: {DEFAULT_UNKNOWN_LABEL}）",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=TRANSCRIPTS_DIR,
        help="結果JSONの出力先ディレクトリ（既定: transcripts/）",
    )
    return ap


def resolve_api_key(cli_value: str | None) -> str | None:
    if cli_value:
        return cli_value
    _load_dotenv_fallback()
    return os.environ.get("PYANNOTEAI_API_KEY") or os.environ.get("PYANNOTE_API_KEY")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.session and not args.wav:
        print("エラー: --session か --wav を1つ指定してください", file=sys.stderr)
        return 2

    targets = resolve_sessions(args.session, args.wav)
    session_id, wav_path, turns_path = targets[0]
    if len(targets) > 1:
        print(f"警告: 複数指定されましたが先頭の {session_id} のみ処理します", file=sys.stderr)

    if not wav_path.exists():
        print(f"エラー: {wav_path} が見つかりません", file=sys.stderr)
        return 2

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print(
            "エラー: PYANNOTEAI_API_KEY (または PYANNOTE_API_KEY) が未設定です。"
            "プロジェクトルートの .env に設定するか --api-key で渡してください。",
            file=sys.stderr,
        )
        return 2

    validate_wav_16k_mono_pcm16(wav_path)
    pcm16 = read_pcm16_bytes(wav_path)

    events = stream_live(pcm16, api_key, head_minutes=args.head_minutes)
    segments = events_to_segments(events)

    out_path = args.out_dir / f"{session_id}.pyannote_live.json"
    save_live_result(
        out_path,
        session_id=session_id,
        wav_path=wav_path,
        head_minutes=args.head_minutes,
        segments=segments,
    )
    print(f"\n話者区間 {len(segments)}件 を {out_path} に保存しました。")

    if not turns_path.exists():
        print(f"警告: {turns_path} が見つからないため、現行turnsとの比較はスキップします", file=sys.stderr)
        return 0

    turns = load_turns(turns_path)
    streamed_ms = (
        int(args.head_minutes * 60_000)
        if args.head_minutes > 0
        else int(len(pcm16) / (SR * 2) * 1000)
    )
    clipped_turns = clip_turns_to_range(turns, streamed_ms)
    if not clipped_turns:
        print("比較対象ターンが範囲内にありません（--head-minutesを広げてください）", file=sys.stderr)
        return 0

    params = {"mode": "live1_realtime", "head_minutes": args.head_minutes}
    result = compare_session(
        session_id, wav_path, turns_path, clipped_turns, segments, params, args.unknown_label
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
