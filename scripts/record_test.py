#!/usr/bin/env python3
"""短いテスト録音を自分で取って、その場で音声品質を診断・切り分けする.

手で一時録音する手間を省く。内蔵マイクから録り、16kHz(アプリと同じ取り込み)と
48kHz(ネイティブ)を比べて、高域(話者の個人差)が「取り込み経路で落ちているのか」
「マイク/距離の問題か」を一発で切り分ける。

使い方（uv）:
    uv run python scripts/record_test.py             # 16kHzと48kHzを各15秒録って比較
    uv run python scripts/record_test.py --seconds 20
    uv run python scripts/record_test.py --rate 16000   # 16kHzだけ
    uv run python scripts/record_test.py --keep         # 録音wavを残す
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import wave

import numpy as np


def _record(seconds: int, rate: int) -> np.ndarray:
    import sounddevice as sd
    print(f"  ● {rate}Hz で {seconds}秒 録音します。マイクに普通の距離で喋ってください。")
    for i in (3, 2, 1):
        print(f"    {i}…", end="", flush=True)
        time.sleep(1)
    print(" ▶ 録音中…")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    print("    ✓ 完了")
    return np.asarray(audio).reshape(-1)


def _save_wav(path: str, audio: np.ndarray, rate: int) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=15, help="各録音の秒数")
    ap.add_argument("--rate", type=int, default=None,
                    help="指定すればその1レートだけ録る（16000 など）")
    ap.add_argument("--keep", action="store_true", help="録音wavを消さずに残す")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from check_audio import analyze

    rates = [args.rate] if args.rate else [16000, 48000]
    outdir = tempfile.mkdtemp(prefix="rectest_")
    paths = []
    try:
        for rate in rates:
            audio = _record(args.seconds, rate)
            p = os.path.join(outdir, f"test_{rate}.wav")
            _save_wav(p, audio, rate)
            paths.append((rate, p))
    except Exception as e:
        sys.exit(f"録音に失敗しました（マイク権限/デバイスを確認）: {type(e).__name__}: {e}")

    for rate, p in paths:
        tag = "アプリと同じ取り込み" if rate == 16000 else "ネイティブ・参考"
        print(f"\n========== {rate}Hz ({tag}) ==========")
        analyze(p)

    if len(paths) == 2:
        print("\n=== 切り分け（48kHz vs 16kHz の高域を比べる） ===")
        print("  48kHzに4kHz以上の高域があるのに16kHzで乏しい")
        print("    → 16kHz直接取り込みが原因。アプリを『48kHzで録って16kHzに落とす』に変えれば回収可能。")
        print("  どちらも高域が乏しい")
        print("    → マイク/距離の問題。近接マイク・卓上USBマイク・距離短縮が要る。")
        print("  どちらも高域が十分")
        print("    → 近接ならOK。崩れた本番は『遠かった』のが原因＝位置取りで改善。")

    if args.keep:
        print(f"\n録音を残しました: {outdir}")
    else:
        for _, p in paths:
            os.remove(p)
        os.rmdir(outdir)


if __name__ == "__main__":
    main()
