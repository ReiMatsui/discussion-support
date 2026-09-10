#!/usr/bin/env python3
"""AMI テストセット（16会議）を本システムで流して標準指標で採点する一括実行.

§48.1 の ES2004a 1本を、話者分離研究で慣例のテストセット16会議に広げる
（ES2004a-d, ES2014a-d, IS1009a-d, TS3003a-d。BUT/pyannote の AMI setup と同じ）。
再生は本番と同じ `python -m das.asr.live --wav`（実時間なので会議と同じ時間が
かかる。16本で約9時間。夜に回す）。採点は `eval/ami_metrics.py`。

  uv run python eval/ami_batch.py --fetch          # 音声と公式注釈を取得（初回）
  uv run python eval/ami_batch.py --run            # 未実行の会議を順に流す（中断・再開可）
  uv run python eval/ami_batch.py --score          # 全会議を採点して表にする
  uv run python eval/ami_batch.py --run --mic Array1-01   # 遠距離1ch（SDM）条件で

進捗は data/ami/batch_<mic>.json に残す（会議 → ラン名）。同じ会議は二度
流さない。ランを流し直したいときはその項目を消す。

条件は §48.1 と同じ: 英語・人数未指定・介入なし。`--max-speakers 4` で
人数を与える条件も流せる（別の進捗ファイルになる）。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ami"
TR = ROOT / "transcripts"

MEETINGS = [f"{s}{c}" for s in ("ES2004", "ES2014", "IS1009", "TS3003") for c in "abcd"]
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
MANUAL_ZIP = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"


def wav_path(meeting: str, mic: str) -> Path:
    return DATA / f"{meeting}.{mic}.wav"


def progress_path(mic: str, max_speakers: int | None) -> Path:
    tag = mic + (f"_n{max_speakers}" if max_speakers else "")
    return DATA / f"batch_{tag}.json"


def _download(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"# 取得: {url}", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def fetch(mic: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for m in MEETINGS:
        _download(f"{MIRROR}/{m}/audio/{m}.{mic}.wav", wav_path(m, mic))
    words = DATA / "manual" / "words"
    if not any(words.glob("*.words.xml")):
        z = DATA / "ami_public_manual_1.6.2.zip"
        _download(MANUAL_ZIP, z)
        words.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if "/words/" in name and name.endswith(".words.xml"):
                    (words / Path(name).name).write_bytes(zf.read(name))
        print(f"# 注釈を展開: {words}")
    print("# 取得完了")


def _duration_sec(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / w.getframerate()


def _newest_turns(after: float) -> str | None:
    cands = [p for p in TR.glob("*.turns.jsonl") if p.stat().st_mtime >= after]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime).name[: -len(".turns.jsonl")]


def run_one(meeting: str, mic: str, max_speakers: int | None) -> str | None:
    wav = wav_path(meeting, mic)
    if not wav.exists():
        print(f"# {meeting}: 音声がありません（--fetch）")
        return None
    dur = _duration_sec(wav)
    cmd = [sys.executable, "-m", "das.asr.live", "--wav", str(wav), "--lang", "en",
           "--no-agent", "--no-llm", "--no-setup", "--no-open", "--port", "0"]
    if max_speakers:
        cmd += ["--diarization-max-speakers", str(max_speakers)]
    print(f"# {meeting}: {dur / 60:.1f}分を再生中（{time.strftime('%H:%M')} 開始）", flush=True)
    started = time.time()
    log = DATA / "logs"
    log.mkdir(exist_ok=True)
    with open(log / f"{meeting}.{mic}.log", "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            proc.wait(timeout=dur + 180)
        except subprocess.TimeoutExpired:
            # 再生は終わったのに UI の終了待ちで残っている場合。Ctrl-C 相当で閉じる
            proc.send_signal(signal.SIGINT)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=60)
            if proc.poll() is None:
                proc.kill()
    run = _newest_turns(started)
    print(f"# {meeting}: ラン {run}（終了コード {proc.returncode}）", flush=True)
    return run


def run_all(mic: str, max_speakers: int | None) -> None:
    pp = progress_path(mic, max_speakers)
    done = json.loads(pp.read_text()) if pp.exists() else {}
    for m in MEETINGS:
        if done.get(m):
            continue
        run = run_one(m, mic, max_speakers)
        if run:
            done[m] = run
            pp.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    print(f"# 完了 {len(done)}/{len(MEETINGS)}（{pp}）")


_NUM = re.compile(r"([\d.]+)%")


def score_all(mic: str, max_speakers: int | None) -> None:
    pp = progress_path(mic, max_speakers)
    if not pp.exists():
        print("# 進捗ファイルがありません（--run）")
        return
    done = json.loads(pp.read_text())
    out_dir = DATA / "results"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for m, run in done.items():
        res = subprocess.run([sys.executable, str(ROOT / "eval" / "ami_metrics.py"), run,
                              "--meeting", m], capture_output=True, text=True, cwd=ROOT)
        text = res.stdout + res.stderr
        (out_dir / f"{m}.{mic}.txt").write_text(text, encoding="utf-8")
        vals = {}
        for line in text.splitlines():
            for key, pat in (("der25", "collar±0.25s"), ("conf", "未確定を外した confusion"),
                             ("der_noovl", "重なり除く"), ("der0f", "collar 0, ＋検出区間"),
                             ("cpwer", "cpWER:"), ("wer", "話者無視のWER")):
                if pat in line and key not in vals:
                    mm = _NUM.search(line)
                    if mm:
                        vals[key] = float(mm.group(1))
        rows.append((m, run, vals))
    cols = (("der25", "DER c.25"), ("conf", "conf(未確定除)"), ("der_noovl", "DER 重なり除"),
            ("der0f", "DER c0+検出"), ("cpwer", "cpWER"), ("wer", "WER"))
    print(f"{'会議':<9}" + "".join(f"{lab:>14}" for _k, lab in cols))
    for m, _run, v in rows:
        print(f"{m:<9}" + "".join(f"{v.get(k, float('nan')):14.1f}" for k, _lab in cols))
    for key, label in cols:
        xs = [v[key] for _m, _r, v in rows if key in v]
        if xs:
            print(f"# {label}: 平均 {sum(xs) / len(xs):.1f}%（{len(xs)}会議, "
                  f"最小 {min(xs):.1f} / 最大 {max(xs):.1f}）")
    print(f"# 各会議の詳細: {out_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--mic", default="Mix-Headset", help="Mix-Headset（既定）か Array1-01（SDM）")
    ap.add_argument("--max-speakers", type=int, default=None, help="人数を与える条件（既定は未指定）")
    args = ap.parse_args()
    if not (args.fetch or args.run or args.score):
        ap.error("--fetch / --run / --score のどれかを指定")
    if args.fetch:
        fetch(args.mic)
    if args.run:
        run_all(args.mic, args.max_speakers)
    if args.score:
        score_all(args.mic, args.max_speakers)


if __name__ == "__main__":
    main()
