#!/usr/bin/env python3
"""本体アプリと独立に動く予備の録音機（実会議の保険）.

本体（`python -m das.asr.live`）が残す録音 wav は「STTへ送れた音声」だけで、
接続断の間の音声は入らない。しかもヘッダは正常終了時に確定するので、途中で
落ちると標準の読み込みでは 0 秒の wav に見える。実会議ではこの2つが致命傷に
なるため、別プロセスで同じマイクから生の音声を録っておく。

- 16kHz mono PCM（本体と同じ形式）。`--wav` でそのまま本体に流し直せる
- 数秒ごとにヘッダを書き直すので、電源断や kill でも直前までは読める
- 30秒以上ほぼ無音なら警告（マイク選択ミスをその場で気づく）

使い方（uv）:
    uv run python scripts/record_backup.py                  # Ctrl-C で終了
    uv run python scripts/record_backup.py --device "USB"   # 入力デバイスを名前の一部で指定
    uv run python scripts/record_backup.py --list           # 入力デバイス一覧
    uv run python scripts/record_backup.py --repair transcripts/2026-09-12_1400.wav
                                                            # 落ちた本体の wav のヘッダを直す
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import struct
import sys
import time

import numpy as np

SR = 16000
HEADER_LEN = 44


def _header(data_size: int, rate: int = SR) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) +
            b"data" + struct.pack("<I", data_size))


class WavWriter:
    """追記しながら定期的にヘッダを確定する PCM16 mono の wav 書き込み."""

    def __init__(self, path: str, rate: int = SR, flush_sec: float = 5.0):
        self.path, self.rate = path, rate
        self.flush_bytes = int(rate * 2 * flush_sec)
        self.f = open(path, "wb")  # noqa: SIM115
        self.f.write(_header(0, rate))
        self.f.flush()
        self.data_size = 0
        self._since_flush = 0

    def write(self, pcm: bytes) -> None:
        self.f.write(pcm)
        self.data_size += len(pcm)
        self._since_flush += len(pcm)
        if self._since_flush >= self.flush_bytes:
            self.sync()

    def sync(self) -> None:
        self._since_flush = 0
        self.f.seek(0)
        self.f.write(_header(self.data_size, self.rate))
        self.f.seek(0, os.SEEK_END)
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self) -> None:
        self.sync()
        self.f.close()


def repair(path: str, rate: int = SR) -> float:
    """ファイル長からヘッダを書き直す（本体の wav が途中で落ちたとき用）."""
    size = os.path.getsize(path)
    if size < HEADER_LEN:
        raise SystemExit(f"{path}: ヘッダすら書かれていません（{size} bytes）")
    with open(path, "rb") as f:
        head = f.read(HEADER_LEN)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise SystemExit(f"{path}: RIFF/WAVE ではありません")
    if head[36:40] != b"data":
        raise SystemExit(f"{path}: 44バイトの標準ヘッダではありません（手作業で確認）")
    ch = struct.unpack("<H", head[22:24])[0]
    fmt_rate = struct.unpack("<I", head[24:28])[0]
    data_size = size - HEADER_LEN
    # 本体もこのスクリプトも 44 バイトの標準ヘッダなので、2つの長さ欄だけ直す
    with open(path, "r+b") as f:
        f.seek(4)
        f.write(struct.pack("<I", 36 + data_size))
        f.seek(40)
        f.write(struct.pack("<I", data_size))
    return data_size / (fmt_rate * 2 * ch)


def _pick_device(name: str | None):
    import sounddevice as sd
    if name is None:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    raise SystemExit(f"入力デバイスが見つかりません: {name}（--list で確認）")


def _list_devices() -> None:
    import sounddevice as sd
    default = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "*" if i == default else " "
            print(f"{mark} [{i}] {d['name']}  ({int(d['default_samplerate'])}Hz)")


def record(out: str, device: str | None, flush_sec: float) -> None:
    import sounddevice as sd
    dev = _pick_device(device)
    info = sd.query_devices(dev if dev is not None else sd.default.device[0])
    print(f"# 入力: {info['name']}")
    print(f"# 保存: {out}（16kHz mono, {flush_sec:.0f}秒ごとにヘッダ確定）")
    print("# Ctrl-C で終了", flush=True)

    w = WavWriter(out, SR, flush_sec)
    level: list[float] = []
    quiet_since: list[float | None] = [None]

    def cb(indata, frames, t, status):
        if status:
            print(f"# 入力の警告: {status}", flush=True)
        x = np.asarray(indata, dtype="float32").reshape(-1)
        w.write((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
        level.append(float(np.sqrt((x ** 2).mean() + 1e-12)))

    t0 = time.time()
    last_report = t0
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            device=dev, callback=cb, blocksize=int(SR * 0.1)):
            while True:
                time.sleep(1.0)
                now = time.time()
                if now - last_report < 10:
                    continue
                last_report = now
                rms = float(np.mean(level)) if level else 0.0
                level.clear()
                db = 20 * np.log10(rms + 1e-9)
                mins = (now - t0) / 60
                print(f"# {mins:5.1f}分  レベル {db:6.1f} dBFS", flush=True)
                if db < -50:
                    if quiet_since[0] is None:
                        quiet_since[0] = now
                    elif now - quiet_since[0] >= 30:
                        print("# 警告: 30秒以上ほぼ無音です。マイクの選択とミュートを確認", flush=True)
                else:
                    quiet_since[0] = None
    except KeyboardInterrupt:
        pass
    finally:
        w.close()
        print(f"# 保存しました: {out}（{w.data_size / (SR * 2) / 60:.1f}分）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None, help="入力デバイス名の一部（省略時は既定）")
    ap.add_argument("--out", default=None, help="保存先（省略時 transcripts/backup_<日時>.wav）")
    ap.add_argument("--flush-sec", type=float, default=5.0, help="ヘッダを確定する間隔（秒）")
    ap.add_argument("--list", action="store_true", help="入力デバイス一覧を出して終わる")
    ap.add_argument("--repair", metavar="WAV", default=None,
                    help="途中で落ちた wav のヘッダをファイル長から書き直す")
    args = ap.parse_args()

    if args.list:
        _list_devices()
        return
    if args.repair:
        sec = repair(args.repair)
        print(f"# 修復しました: {args.repair}（{sec / 60:.1f}分）")
        return
    out = args.out or os.path.join(
        "transcripts", "backup_" + dt.datetime.now().strftime("%Y-%m-%d_%H%M") + ".wav")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        sys.exit(f"既にあります: {out}")
    record(out, args.device, args.flush_sec)


if __name__ == "__main__":
    main()
