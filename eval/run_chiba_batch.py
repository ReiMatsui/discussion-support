#!/usr/bin/env python3
"""千葉コーパスの再生評価を、複数の会話×条件で並列に回す.

    uv run python eval/run_chiba_batch.py --minutes 4 --jobs 4
    uv run python eval/run_chiba_batch.py --minutes 4 --jobs 4 --convs chiba0332,chiba0432

各ジョブは eval/run_chiba.py を 1 回呼ぶ（条件: 登録なし / 登録あり）。
セッション名は <会話>_m<分>_<条件>_<日時> で付けるので、同時に走っても
transcripts/ の名前が衝突しない。ジョブごとの出力は data/chiba/logs/ に残す。
結果は run_chiba.py が data/chiba/results.csv に追記する（並列でも 1 行ずつ）。

並列数の目安: Soniox と pyannote は 1 ジョブ 1 接続。API の同時接続上限に
当たると接続エラーで落ちるので、落ちたジョブはログを見て --convs で再実行する。
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALL_CONVS = [f"chiba{i:02d}32" for i in range(1, 13)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--convs", default=",".join(ALL_CONVS),
                   help="会話名（カンマ区切り。既定 13 本中の 12 本 chiba0132..chiba1232）")
    p.add_argument("--minutes", type=float, default=4.0, help="先頭何分を流すか")
    p.add_argument("--jobs", type=int, default=4, help="同時に走らせる本数")
    p.add_argument("--enroll-seconds", type=float, default=60.0)
    p.add_argument("--conditions", default="none,enroll",
                   help="none=登録なし, enroll=登録あり（カンマ区切り）")
    a = p.parse_args()

    convs = [c.strip() for c in a.convs.split(",") if c.strip()]
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = ROOT / "data" / "chiba" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, list[str]]] = []
    for conv in convs:
        for cond in conds:
            name = f"{conv}_m{a.minutes:g}_{cond}_{stamp}"
            cmd = [sys.executable, str(ROOT / "eval" / "run_chiba.py"),
                   "--conv", conv, "--minutes", str(a.minutes), "--session-name", name]
            if cond == "enroll":
                cmd += ["--enroll-seconds", str(a.enroll_seconds)]
            elif cond != "none":
                sys.exit(f"未知の条件: {cond}")
            jobs.append((name, cmd))

    # 前処理（GT・ミックス音声）は会話ごとに 1 回、先に直列で済ませる。
    # 同じ会話の 2 条件が同時に同じファイルを書かないようにするため。
    for conv in convs:
        subprocess.run([sys.executable, str(ROOT / "eval" / "prep_chiba.py"),
                        "--conv", conv, "--minutes", str(a.minutes)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print(f"# 前処理完了: {len(convs)} 会話（先頭 {a.minutes:g} 分）", flush=True)

    print(f"# {len(jobs)} ジョブを {a.jobs} 並列で実行（各 {a.minutes:g} 分）", flush=True)
    running: list[tuple[str, subprocess.Popen, object]] = []
    failed: list[str] = []
    pending = list(jobs)

    def reap() -> None:
        for item in list(running):
            name, proc, fh = item
            if proc.poll() is not None:
                fh.close()
                running.remove(item)
                status = "完了" if proc.returncode == 0 else f"失敗 (exit {proc.returncode})"
                print(f"# {name}: {status}", flush=True)
                if proc.returncode != 0:
                    failed.append(name)

    import time
    while pending or running:
        reap()
        while pending and len(running) < a.jobs:
            name, cmd = pending.pop(0)
            fh = open(logdir / f"{name}.log", "w", encoding="utf-8")
            print(f"# 開始 {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            running.append((name, proc, fh))
            time.sleep(3)   # 起動をずらす（モデル読み込みと接続の同時多発を避ける）
        time.sleep(5)

    print(f"# 終了。失敗 {len(failed)} 件" + (": " + ", ".join(failed) if failed else ""), flush=True)
    print(f"# 結果: data/chiba/results.csv、ログ: {logdir.relative_to(ROOT)}/", flush=True)


if __name__ == "__main__":
    main()
