#!/usr/bin/env python3
"""同じ音声を「登録なし」「登録あり」の 2 条件で同時に流す（ゼミ録音・YouTube など任意の音声）.

    uv run python eval/run_pair.py --wav transcripts/2026-06-25_1509.wav --name zemi1509 \\
        --voices data/voices_zemi.json --max-speakers 5
    uv run python eval/run_pair.py --wav transcripts/2026-06-25_1520.wav --name zemi1520 \\
        --voices data/voices_zemi.json --max-speakers 5 --minutes 10

やること:
1. --minutes があれば先頭だけを data/pairs/<name>_m<分>.wav に切り出す（元は触らない）
2. das listen-soniox --no-intervention --wav … を 2 本同時に起動する
     登録なし: transcripts/<name>_none.*
     登録あり: transcripts/<name>_enroll.*（--voices の全員を --activate all で有効化）
   ログは data/pairs/logs/<name>_<条件>.log
3. 終わったら、正解付け（eval/annotate.py）と採点（eval/enroll_breakdown.py）のコマンドを出す

千葉コーパスは eval/run_chiba_batch.py（正解が付属）、これは正解を後から
annotate.py で付ける音声用。実時間で流すので所要時間 ≒ 音声の長さ。
--voices を省くと登録なしの 1 本だけ流す。
"""
from __future__ import annotations

import argparse
import math
import shlex
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
PAIRS = ROOT / "data" / "pairs"


def das_command() -> list[str]:
    exe = shutil.which("das") or str(Path(sys.executable).parent / "das")
    if Path(exe).exists():
        return [exe]
    return ["uv", "run", "das"]


def head_wav(src: Path, minutes: float, name: str) -> Path:
    """16kHz mono の wav の先頭 minutes 分を data/pairs/ に書く（既にあれば使い回す）."""
    PAIRS.mkdir(parents=True, exist_ok=True)
    dst = PAIRS / f"{name}_m{minutes:g}.wav"
    with wave.open(str(src), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(f"{src}: 16kHz mono 16bit の wav にしてください（ffmpeg -i in -ar 16000 -ac 1 out.wav）")
        sr = w.getframerate()
        n = min(w.getnframes(), int(minutes * 60 * sr))
        frames = w.readframes(n)
    with wave.open(str(dst), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(sr)
        o.writeframes(frames)
    return dst


def build_commands(wav: Path, name: str, voices: Path | None, max_speakers: int,
                   extra: str = "") -> list[tuple[str, list[str]]]:
    """(セッション名, コマンド) の列。登録なし → 登録あり の順."""
    conds = [("none", "")]
    if voices:
        conds.append(("enroll", f"--voices {shlex.quote(str(voices))} --activate all"))
    out = []
    for cond, vargs in conds:
        session = f"{name}_{cond}"
        soniox = " ".join(x for x in (extra, vargs,
                                      f"--out {shlex.quote(str(TRANSCRIPTS / (session + '.md')))}") if x)
        cmd = [*das_command(), "listen-soniox", "--max-speakers", str(max_speakers),
               "--no-intervention", "--wav", str(wav), "--soniox-args", soniox]
        out.append((session, cmd))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--wav", required=True, help="流す音声（16kHz mono wav）")
    p.add_argument("--name", required=True, help="セッション名の頭（transcripts/<name>_none 等）")
    p.add_argument("--voices", default=None, help="登録あり条件で使う voices.json（省くと登録なしのみ）")
    p.add_argument("--max-speakers", type=int, default=3)
    p.add_argument("--minutes", type=float, default=None, help="先頭何分だけ流すか")
    p.add_argument("--soniox-args", default="", help="文字起こし側へ渡す追加引数（両条件に共通）")
    p.add_argument("--dry-run", action="store_true", help="コマンドを表示して終わる")
    a = p.parse_args(argv)

    src = Path(a.wav)
    if not src.exists():
        sys.exit(f"{src} がありません")
    voices = Path(a.voices) if a.voices else None
    if voices and not voices.exists():
        sys.exit(f"{voices} がありません（scripts/enroll_voices.py で作る）")
    for cond in ("none", "enroll"):
        if (TRANSCRIPTS / f"{a.name}_{cond}.turns.jsonl").exists():
            sys.exit(f"transcripts/{a.name}_{cond}.* が既にあります。--name を変えてください")

    wav = head_wav(src, a.minutes, a.name) if a.minutes else src
    jobs = build_commands(wav, a.name, voices, a.max_speakers, a.soniox_args)
    for session, cmd in jobs:
        print(f"# {session}: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    if a.dry_run:
        return

    logdir = PAIRS / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    minutes = a.minutes
    if minutes is None:
        with wave.open(str(wav), "rb") as w:
            minutes = w.getnframes() / w.getframerate() / 60
    print(f"# {len(jobs)} 本を同時に流します（約 {minutes:.0f} 分）", flush=True)
    procs = []
    for session, cmd in jobs:
        fh = open(logdir / f"{session}.log", "w", encoding="utf-8")
        procs.append((session, subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT), fh))
        time.sleep(3)   # モデル読み込みと接続の同時多発を避ける
    failed = []
    for session, proc, fh in procs:
        proc.wait()
        fh.close()
        ok = proc.returncode == 0 and (TRANSCRIPTS / f"{session}.turns.jsonl").exists()
        print(f"# {session}: {'完了' if ok else f'失敗 (exit {proc.returncode})'}"
              f"  ログ data/pairs/logs/{session}.log", flush=True)
        if not ok:
            failed.append(session)
    if failed:
        sys.exit(f"# 失敗: {', '.join(failed)}")

    print("\n# 次の手順")
    print(f"#  1) 正解付け（登録なしのランに付ける。ブラウザで 1〜9 キー）:")
    print(f"#     uv run python eval/annotate.py {a.name}_none")
    if voices:
        print(f"#  2) 採点（正解は時間で突き合わせるので登録ありにもそのまま使える）:")
        print(f"#     uv run python eval/enroll_breakdown.py --gt eval/gt_{a.name}_none.json"
              f" --none {a.name}_none --enroll {a.name}_enroll --minutes {math.ceil(minutes)}")


if __name__ == "__main__":
    main()
