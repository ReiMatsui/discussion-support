#!/usr/bin/env python3
"""録音音声の品質（話者分離に効く帯域・レベル・SNR）を診断する.

話者識別は高域（2〜8kHz の子音・歯擦音などの個人差）に強く依存する。遠距離の
ラップトップ内蔵マイクや過剰なノイズ抑制で音がこもると、その情報が録音に入らず、
どんなモデルでも話者を分離できなくなる。本スクリプトはそれを数値で確認する。

使い方（uv）:
    uv run python scripts/check_audio.py                      # 最新の長いセッション
    uv run python scripts/check_audio.py transcripts/xxx.wav  # 任意のwav
    uv run python scripts/check_audio.py clips/closemic.wav   # マイク変更後の比較に

マイクを変える前後で走らせ、2-8kHz の割合・SNR・レベルが改善するか見るとよい。
"""
from __future__ import annotations

import glob
import os
import sys
import wave

import numpy as np

TR = "transcripts"


def _find_default() -> str | None:
    wavs = glob.glob(os.path.join(TR, "*.wav"))
    if not wavs:
        return None
    wavs.sort(key=os.path.getsize, reverse=True)
    return wavs[0]


def analyze(path: str):
    with wave.open(path, "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2).mean(1)
    print(f"# {path}  {sr}Hz {ch}ch {n / sr / 60:.1f}分")

    peak, rms = np.abs(x).max(), np.sqrt((x ** 2).mean())
    clip = np.mean(np.abs(x) > 0.98) * 100
    print(f"# レベル: ピーク {peak:.3f}  RMS {20 * np.log10(rms + 1e-9):.1f}dBFS"
          f"  クリップ率 {clip:.2f}%")

    fr = 2048
    frames = x[: len(x) // fr * fr].reshape(-1, fr)
    fe = np.sqrt((frames ** 2).mean(1) + 1e-12)
    hi, lo = np.percentile(fe, 90), np.percentile(fe, 10)
    snr = 20 * np.log10(hi / lo)
    print(f"# 推定SNR(発話90%ile vs 床10%ile): {snr:.1f}dB")

    win = np.hanning(fr)
    acc = np.zeros(fr // 2 + 1)
    cnt = 0
    for i in range(0, len(x) - fr, fr * 7):
        seg = x[i:i + fr]
        if np.sqrt((seg ** 2).mean()) < hi * 0.5:
            continue   # 発話フレームだけ
        acc += np.abs(np.fft.rfft(seg * win)) ** 2
        cnt += 1
    acc /= max(cnt, 1)
    freqs = np.fft.rfftfreq(fr, 1 / sr)
    total = acc.sum() or 1.0
    print(f"# 発話フレーム {cnt}個でスペクトル平均")
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000),
             (4000, 6000), (6000, 8000)]
    hi_energy = 0.0
    for a, b in bands:
        m = (freqs >= a) & (freqs < b)
        pct = acc[m].sum() / total * 100
        if a >= 2000:
            hi_energy += pct
        print(f"#   {a:5d}-{b:5d}Hz: {pct:5.1f}%  " + "#" * int(pct / 2))
    cum = np.cumsum(acc) / total
    f95 = freqs[min(np.searchsorted(cum, 0.95), len(freqs) - 1)]
    print(f"# エネルギー95%到達 {f95:.0f}Hz（{sr // 2}Hzが上限）"
          f"  / 2kHz以上の割合 {hi_energy:.1f}%")

    # 所見
    print("\n=== 所見 ===")
    issues = []
    if f95 < 2500 or hi_energy < 8:
        issues.append("こもり（高域が乏しく話者の個人差が入っていない）")
    if snr < 18:
        issues.append("SNRが低い（雑音/反響が多い）")
    if 20 * np.log10(rms + 1e-9) < -28:
        issues.append("録音レベルが低い")
    if issues:
        print("  問題: " + " / ".join(issues))
        print("  → 話者分離が崩れる主因は音声入力の可能性が高い。"
              "近接マイク/1人1チャンネル、macOSの音声処理(ノイズ抑制)無効化、"
              "マイク距離短縮・入力レベル調整を検討。")
    else:
        print("  音声品質は良好。分離が崩れるなら入力以外（重なり/短さ/ロジック）が主因。")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _find_default()
    if not path or not os.path.exists(path):
        sys.exit(f"wavが見つかりません: {path}")
    analyze(path)


if __name__ == "__main__":
    main()
