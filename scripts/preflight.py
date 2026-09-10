#!/usr/bin/env python3
"""実会議の前に5分で回す点検（落ちる原因は前日と環境が変わった所に集中する）.

見るもの: APIキー、各APIへの到達、マイクの入力レベル、声紋モデルのキャッシュ、
ディスク残量、保存先の書き込み、電源。1つでも ✗ があれば開始しない。

使い方（uv）:
    uv run python scripts/preflight.py                 # 全部
    uv run python scripts/preflight.py --no-agent      # 介入なし運用（OpenAI を見ない）
    uv run python scripts/preflight.py --device "USB"  # マイクを名前の一部で指定
    uv run python scripts/preflight.py --seconds 5     # レベル測定の長さ
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

OK, NG, WARN = "✓", "✗", "△"
ENDPOINTS = {
    "SONIOX_API_KEY": ("stt-rt.soniox.com", "Soniox（音声認識）"),
    "PYANNOTEAI_API_KEY": ("api.pyannote.ai", "pyannote（話者分離）"),
    "OPENAI_API_KEY": ("api.openai.com", "OpenAI（介入・LLM）"),
}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        if not os.path.exists(".env"):
            return
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def check_keys(need: list[str]) -> list[tuple[str, str]]:
    out = []
    for k in need:
        v = os.environ.get(k, "")
        if not v or v.endswith("..."):
            out.append((NG, f"{k} が未設定（.env を確認）"))
        else:
            out.append((OK, f"{k} あり"))
    return out


def check_network(need: list[str]) -> list[tuple[str, str]]:
    out = []
    for k in need:
        host, label = ENDPOINTS[k]
        t0 = time.time()
        try:
            with socket.create_connection((host, 443), timeout=4):
                pass
            out.append((OK, f"{label} に到達（{(time.time() - t0) * 1000:.0f}ms）"))
        except OSError as e:
            out.append((NG, f"{label} に届かない: {type(e).__name__}（Wi-Fi/プロキシを確認）"))
    return out


def check_mic(device: str | None, seconds: float) -> list[tuple[str, str]]:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        return [(NG, f"音声ライブラリが読み込めない: {e}")]
    dev = None
    if device:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and device.lower() in d["name"].lower():
                dev = i
                break
        else:
            return [(NG, f"入力デバイスが見つからない: {device}")]
    try:
        info = sd.query_devices(dev if dev is not None else sd.default.device[0])
        print(f"   入力デバイス: {info['name']}  {seconds:.0f}秒測ります。普通の声量で何か話してください")
        x = sd.rec(int(seconds * 16000), samplerate=16000, channels=1,
                   dtype="float32", device=dev)
        sd.wait()
    except Exception as e:
        return [(NG, f"マイクから録れない: {type(e).__name__}: {e}（権限・デバイス）")]
    x = np.asarray(x).reshape(-1)
    rms = float(np.sqrt((x ** 2).mean() + 1e-12))
    db = 20 * np.log10(rms + 1e-9)
    peak = float(np.abs(x).max())
    out = [(OK, f"マイク: {info['name']}")]
    if db < -50:
        out.append((NG, f"入力がほぼ無音（{db:.1f} dBFS）。ミュート・デバイス選択・距離を確認"))
    elif db < -40:
        out.append((WARN, f"入力が小さい（{db:.1f} dBFS）。マイクを話者に近づける"))
    else:
        out.append((OK, f"入力レベル {db:.1f} dBFS"))
    if peak > 0.98:
        out.append((WARN, "クリップしている（入力ゲインを下げる）"))
    return out


def check_model() -> list[tuple[str, str]]:
    hub = os.environ.get("TORCH_HOME")
    hub = os.path.join(hub, "hub") if hub else os.path.expanduser("~/.cache/torch/hub")
    cand = [p for p in (os.listdir(hub) if os.path.isdir(hub) else []) if "ReDimNet" in p]
    ckpt = os.path.join(hub, "checkpoints")
    weights = [p for p in (os.listdir(ckpt) if os.path.isdir(ckpt) else []) if "redimnet" in p.lower()]
    if cand and weights:
        return [(OK, "声紋モデル ReDimNet はキャッシュ済み（初回ダウンロードは起きない）")]
    return [(WARN, "ReDimNet のキャッシュが見当たらない。起動時に GitHub から取得する"
                   "（ネットが無いと落ちる）。先に一度 `uv run python -m das.asr.live --no-agent"
                   " --no-open` を起動して止めておく")]


def check_disk(path: str = "transcripts", need_gb: float = 2.0) -> list[tuple[str, str]]:
    os.makedirs(path, exist_ok=True)
    free = shutil.disk_usage(path).free / 1e9
    out = [(OK if free >= need_gb else NG, f"ディスク残量 {free:.1f} GB（1時間の録音2本で約0.25GB）")]
    probe = os.path.join(path, ".preflight_write_test")
    try:
        with open(probe, "wb") as f:
            f.write(b"x")
        os.remove(probe)
        out.append((OK, f"{path}/ に書き込める"))
    except OSError as e:
        out.append((NG, f"{path}/ に書けない: {e}"))
    return out


def check_power() -> list[tuple[str, str]]:
    if sys.platform != "darwin":
        return []
    try:
        s = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    if "AC Power" in s:
        return [(OK, "電源に接続")]
    return [(WARN, "バッテリー駆動。1時間超えるなら電源をつなぐ")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-agent", action="store_true", help="OpenAI を点検しない")
    ap.add_argument("--device", default=None, help="入力デバイス名の一部")
    ap.add_argument("--seconds", type=float, default=3.0, help="レベル測定の秒数")
    ap.add_argument("--skip-mic", action="store_true")
    args = ap.parse_args()

    _load_env()
    need = ["SONIOX_API_KEY", "PYANNOTEAI_API_KEY"] + ([] if args.no_agent else ["OPENAI_API_KEY"])
    results: list[tuple[str, str]] = []
    results += check_keys(need)
    results += check_network(need)
    results += check_model()
    results += check_disk()
    results += check_power()
    if not args.skip_mic:
        results += check_mic(args.device, args.seconds)

    print()
    for mark, msg in results:
        print(f" {mark} {msg}")
    ng = sum(1 for m, _ in results if m == NG)
    warn = sum(1 for m, _ in results if m == WARN)
    print()
    if ng:
        print(f"✗ {ng}件。直してからもう一度")
        sys.exit(1)
    print("✓ 開始できます" + (f"（注意 {warn}件）" if warn else ""))


if __name__ == "__main__":
    main()
