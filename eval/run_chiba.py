#!/usr/bin/env python3
"""千葉大コーパスの帰属測定を1コマンドで実行する（前処理→実走→採点）.

やること:
1. eval/prep_chiba.py 相当の前処理（GT・ミックス音声が無ければ自動生成）
2. das listen-soniox --hybrid を --wav で実走（Soniox APIを使用、要APIキー）
3. 新しく生成されたセッションを自動検出（セッション名の手入力不要）
4. eval/eval_speaker_gt.py のタイムライン突合で採点
5. 結果1行を data/chiba/results.csv に追記（会話をまたいだ比較用）

使い方（実機で）:
    uv run python eval/run_chiba.py                       # chiba0132 全体(9.5分)
    uv run python eval/run_chiba.py --conv chiba0232      # 別の会話
    uv run python eval/run_chiba.py --minutes 5           # 先頭5分だけ（注意:
        chiba0132 は話者Bが序盤ほぼ無言のため全体版を推奨。handoff §15.2）
    uv run python eval/run_chiba.py --skip-run --session <既存セッション名>
        # 実走を飛ばして既存ランを再採点だけしたいとき

設計: docs/design/handoff_2026-07-14_unregistered_speakers.md §14（CallHome
と同一の測定経路）・§15.2（Chiba3Party の位置づけ）。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"


def _load_module(name: str):
    """eval/ はパッケージではないのでファイルパスから直接importする."""
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ensure_prepared(conv: str, minutes: float | None, gap: float,
                    force: bool) -> tuple[Path, Path]:
    """GT・ミックス音声が無ければ prep_chiba のロジックで生成する."""
    session = conv + (f"m{minutes:g}" if minutes else "")
    gt_path = ROOT / "eval" / f"gt_{session}.json"
    mix_name = f"{conv}_mix" + (f"_{minutes:g}min" if minutes else "") + ".wav"
    mix_path = ROOT / "data" / "chiba" / mix_name
    turns_path = TRANSCRIPTS / f"{session}.turns.jsonl"
    if force or not (gt_path.exists() and mix_path.exists() and turns_path.exists()):
        args = ["--conv", conv, "--gap", str(gap)]
        if minutes:
            args += ["--minutes", str(minutes)]
        prep = _load_module("prep_chiba")
        old_argv = sys.argv
        sys.argv = ["prep_chiba.py", *args]
        try:
            prep.main()
        finally:
            sys.argv = old_argv
    return gt_path, mix_path


def das_command() -> list[str]:
    """das 実行コマンドを解決する（uv run 下なら venv の das が見つかる）."""
    exe = shutil.which("das") or str(Path(sys.executable).parent / "das")
    if Path(exe).exists():
        return [exe]
    return ["uv", "run", "das"]   # フォールバック（uv 経由で起動された場合は不要）


def run_live(mix_path: Path, max_speakers: int, extra_soniox: str) -> str:
    """listen-soniox を実走し、新規セッション名を自動検出して返す."""
    before = {p.name for p in TRANSCRIPTS.glob("*.turns.jsonl")}
    started = datetime.datetime.now()
    cmd = das_command() + [
        "listen-soniox", "--hybrid", "--max-speakers", str(max_speakers),
        "--wav", str(mix_path), "--soniox-args", extra_soniox,
    ]
    print(f"# 実行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    new = [p for p in TRANSCRIPTS.glob("*.turns.jsonl")
           if p.name not in before]
    if not new:
        sys.exit("実走後に新しい transcripts/*.turns.jsonl が見つかりません"
                 f"（{started:%H:%M:%S} 以降の生成物なし）。実走ログを確認してください")
    newest = max(new, key=lambda p: p.stat().st_mtime)
    return newest.name.removesuffix(".turns.jsonl")


def append_result(conv: str, session: str, minutes: float | None,
                  stdout_text: str) -> None:
    """採点出力から主要数値を拾って results.csv に1行追記する."""
    import re
    m_acc = re.search(r"帰属精度: (\d+)%", stdout_text)
    path = ROOT / "data" / "chiba" / "results.csv"
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "conv", "minutes", "session", "accuracy_pct"])
        w.writerow([datetime.date.today().isoformat(), conv,
                    minutes or "full", session,
                    m_acc.group(1) if m_acc else ""])
    print(f"# 結果を {path.relative_to(ROOT)} に追記"
          + (f"（精度 {m_acc.group(1)}%）" if m_acc else ""))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--conv", default="chiba0132")
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--gap", type=float, default=0.5)
    p.add_argument("--max-speakers", type=int, default=3)
    p.add_argument("--soniox-args", default="--no-agent",
                   help="listen-soniox へ渡す追加引数（既定: エージェント停止）")
    p.add_argument("--force-prep", action="store_true", help="GT・音声を作り直す")
    p.add_argument("--skip-run", action="store_true",
                   help="実走を飛ばして --session の再採点のみ")
    p.add_argument("--session", default=None,
                   help="--skip-run 時に採点する既存セッション名")
    a = p.parse_args()

    gt_path, mix_path = ensure_prepared(a.conv, a.minutes, a.gap, a.force_prep)

    if a.skip_run:
        if not a.session:
            sys.exit("--skip-run には --session <セッション名> が必要です")
        session = a.session
    else:
        session = run_live(mix_path, a.max_speakers, a.soniox_args)
        print(f"# 新セッション: {session}")

    # 採点（eval_speaker_gt の main をそのまま使い、出力を拾って要約も残す）
    import contextlib
    import io
    esg = _load_module("eval_speaker_gt")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        esg.main(str(gt_path), None if session == gt_path.stem.removeprefix("gt_")
                 else session)
    text = buf.getvalue()
    print(text)
    append_result(a.conv, session, a.minutes, text)


if __name__ == "__main__":
    main()
