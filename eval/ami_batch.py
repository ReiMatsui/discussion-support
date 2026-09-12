#!/usr/bin/env python3
"""AMI テストセット（16会議）を本システムで流して標準指標で採点する一括実行.

§48.1 の ES2004a 1本を、話者分離研究で慣例のテストセット16会議に広げる
（ES2004a-d, ES2014a-d, IS1009a-d, TS3003a-d。BUT/pyannote の AMI setup と同じ）。
再生は本番と同じ `python -m das.asr.live --wav`（実時間なので会議と同じ時間が
かかる。16本で約9時間。夜に回す）。採点は `eval/ami_metrics.py`。

  uv run python eval/ami_batch.py --fetch          # 音声と公式注釈を取得（初回）
  uv run python eval/ami_batch.py --run            # 未実行の会議を順に流す（中断・再開可）
  uv run python eval/ami_batch.py --run --parallel 2   # 2会議ずつ同時に（約4.5時間）
  uv run python eval/ami_batch.py --score          # 全会議を採点して表にする
  uv run python eval/ami_batch.py --run --mic Array1-01   # 遠距離1ch（SDM）条件で

進捗は data/ami/batch_<mic>.json に残す（会議 → ラン名）。同じ会議は二度
流さない。ランを流し直したいときはその項目を消す。ラン名は
`ami_<会議>_<mic>` で固定（`--out` 指定）なので、並列でも取り違えない。
Ctrl-C で中断すると走っていた会議は記録されず、次回その会議から流し直す。
失敗した会議（起動時の接続失敗・認識の停止・話者分離の再接続）も記録せず、
次回再試行する。並列数は CPU で決まる: 本体1つが声紋計算で1コア以上を使うので、
MacBook では 2 までが目安（4 では pyannote の keepalive が落ちて再接続が頻発した。
2026-09-11 の実測）。

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
import threading
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


def run_name(meeting: str, mic: str, max_speakers: int | None) -> str:
    return f"ami_{meeting}_{mic}" + (f"_n{max_speakers}" if max_speakers else "")


def _clear_run(run: str) -> None:
    """同じ名前の古い記録を消す（diag は追記なので残すと混ざる）."""
    for p in TR.glob(f"{run}.*"):
        p.unlink()


_ANSI = re.compile(r"\x1b\[[0-9;]*[mK]")
# アプリ自身のメッセージだけを見る。文字起こし本文（"too many buttons" 等）を
# 誤って拾わないよう、語の一致ではなく行の種類で判定する
_FATAL = re.compile(r"^(Traceback|\w*Error:|.*handshake response|.*policy violation)")
_RECONNECT = re.compile(r"切断を検知|keepalive ping timeout|Audio received too fast")


def _log_lines(log_path: Path) -> list[str]:
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    return [ln.strip() for ln in _ANSI.sub("", raw).replace("\r", "\n").splitlines()]


def _check(run: str, log_path: Path, dur_sec: float,
           allow_reconnect: bool) -> str | None:
    """ランが成立したかを見る。問題があれば理由を返す.

    基準: 文字起こしが会議の長さに見合って出ている（1分あたり3発話以上）こと、
    起動時の接続に失敗していないこと、話者分離（pyannote）の接続が途中で
    切れていないこと。切断があると分離に穴が空き、標準指標の比較に使えない
    （`--allow-reconnect` で許容にできるが、公表用の数字には使わない）。
    """
    turns = TR / f"{run}.turns.jsonl"
    if not turns.exists() or turns.stat().st_size == 0:
        return "turns が空（起動時の接続失敗）"
    with open(turns, encoding="utf-8") as f:
        n = sum(1 for _ in f)
    per_min = n / max(dur_sec / 60, 1)
    if per_min < 3:
        return f"発話が {n} 件（{per_min:.1f}件/分）しかない。認識が途中で止まった"
    lines = _log_lines(log_path)
    reconnects = sum(1 for ln in lines if _RECONNECT.search(ln))
    fatal = [ln for ln in lines if _FATAL.match(ln)]
    if reconnects and not allow_reconnect:
        return f"話者分離の接続が {reconnects} 回切れた（負荷。--parallel を減らす）"
    if fatal and not reconnects:
        return f"ログに {fatal[0][:60]!r}"
    return None


def run_one(meeting: str, mic: str, max_speakers: int | None,
            stop: threading.Event, allow_reconnect: bool = False) -> tuple[str, str | None]:
    """1会議を流す。(ラン名, 失敗理由 or None) を返す."""
    wav = wav_path(meeting, mic)
    run = run_name(meeting, mic, max_speakers)
    if not wav.exists():
        return run, "音声がありません（--fetch）"
    dur = _duration_sec(wav)
    _clear_run(run)
    cmd = [sys.executable, "-m", "das.asr.live", "--wav", str(wav), "--lang", "en",
           "--out", str(TR / f"{run}.md"),
           "--no-agent", "--no-llm", "--no-setup", "--no-open", "--port", "0"]
    if max_speakers:
        cmd += ["--diarization-max-speakers", str(max_speakers)]
    print(f"# {meeting}: {dur / 60:.1f}分を再生中（{time.strftime('%H:%M')} 開始）", flush=True)
    log_dir = DATA / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{run}.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        deadline = time.time() + dur + 180
        while proc.poll() is None:
            if stop.is_set():
                proc.send_signal(signal.SIGINT)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=60)
                if proc.poll() is None:
                    proc.kill()
                return run, "中断"
            if time.time() > deadline:
                # 再生は終わったのに終了待ちで残っている場合。Ctrl-C 相当で閉じる
                proc.send_signal(signal.SIGINT)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=60)
                if proc.poll() is None:
                    proc.kill()
                break
            time.sleep(2)
    if stop.is_set():
        return run, "中断"
    why = _check(run, log_path, dur, allow_reconnect)
    print(f"# {meeting}: {'完了' if why is None else '失敗 ' + why}（{time.strftime('%H:%M')}）",
          flush=True)
    return run, why


def run_all(mic: str, max_speakers: int | None, parallel: int, stagger: float,
            allow_reconnect: bool = False, reset: bool = False) -> None:
    pp = progress_path(mic, max_speakers)
    if reset and pp.exists():
        pp.unlink()
        print(f"# 進捗を消しました: {pp}")
    done: dict = json.loads(pp.read_text()) if pp.exists() else {}
    todo = [m for m in MEETINGS if not done.get(m)]
    if not todo:
        print(f"# 全 {len(MEETINGS)} 会議が完了済み（{pp}）")
        return
    total_min = sum(_duration_sec(wav_path(m, mic)) for m in todo if wav_path(m, mic).exists()) / 60
    print(f"# 残り {len(todo)} 会議・音声 {total_min:.0f} 分・並列 {parallel} → "
          f"目安 {total_min / parallel / 60:.1f} 時間。Ctrl-C で中断（同じコマンドで再開）")
    stop = threading.Event()
    lock = threading.Lock()
    failed: dict[str, str] = {}

    def work(i: int, m: str):
        time.sleep((i % parallel) * stagger)   # 同時接続をずらす
        if stop.is_set():
            return
        run, why = run_one(m, mic, max_speakers, stop, allow_reconnect)
        with lock:
            if why is None:
                done[m] = run
                pp.write_text(json.dumps(done, ensure_ascii=False, indent=1))
            elif why != "中断":
                failed[m] = why

    from concurrent.futures import ThreadPoolExecutor

    def _on_sigint(_sig, _frm):
        # 端末の Ctrl-C は子プロセスにも届く（子は保存して終了する）。先に stop を
        # 立てておかないと、終了した子を「完了」と記録してしまう
        stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = [ex.submit(work, i, m) for i, m in enumerate(todo)]
            for f in futs:
                f.result()
    except KeyboardInterrupt:
        stop.set()
        print("\n# 中断。走っていた会議は記録せず、次回同じコマンドで最初から流し直します",
              flush=True)
    print(f"# 完了 {len(done)}/{len(MEETINGS)}（{pp}）")
    if failed:
        print("# 失敗（次回の --run で再試行される）:")
        for m, why in failed.items():
            print(f"#   {m}: {why}")
        print("# 「接続が切れた」「認識が止まった」が並ぶなら CPU が足りていない。"
              "--parallel を減らす（1〜2）")


_NUM = re.compile(r"([\d.]+)%")


def score_all(mic: str, max_speakers: int | None) -> None:
    pp = progress_path(mic, max_speakers)
    if not pp.exists():
        print("# 進捗ファイルがありません（--run）")
        return
    done = json.loads(pp.read_text())
    try:
        import meeteval  # noqa: F401
        import pyannote.metrics  # noqa: F401
    except ImportError as e:
        print(f"# 採点に必要なパッケージがありません（{e.name}）。"
              "`uv add --dev meeteval pyannote.metrics` を実行してから --score")
        return
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
    ap.add_argument("--parallel", type=int, default=1,
                    help="同時に流す会議数（推奨 3〜4。pyannote は10並列まで、Soniox は未公表）")
    ap.add_argument("--stagger", type=float, default=20.0, help="並列時に接続をずらす秒数")
    ap.add_argument("--allow-reconnect", action="store_true",
                    help="話者分離の再接続があったランも完了扱いにする（公表用には使わない）")
    ap.add_argument("--reset", action="store_true", help="進捗を消して全会議を流し直す")
    args = ap.parse_args()
    if args.parallel > 4:
        print("# 注意: --parallel 5 以上は API 側の制限に当たりやすい。失敗が出たら減らす")
    if not (args.fetch or args.run or args.score):
        ap.error("--fetch / --run / --score のどれかを指定")
    if args.fetch:
        fetch(args.mic)
    if args.run:
        run_all(args.mic, args.max_speakers, max(1, args.parallel), args.stagger,
                args.allow_reconnect, args.reset)
    if args.score:
        score_all(args.mic, args.max_speakers)


if __name__ == "__main__":
    main()
