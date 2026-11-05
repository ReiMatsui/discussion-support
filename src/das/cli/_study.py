"""研究用モードの分かりやすい起動入口。従来のグラフ構築レーンは起動しない。"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import typer

from das.cli import app


class StudyCondition(StrEnum):
    system = "system"
    none = "none"


@app.command(name="study")
def study(
    config: Path = typer.Option(
        Path("study/facility.json"), "--config", exists=True, dir_okay=False,
        help="課題設定JSON",
    ),
    condition: StudyCondition = typer.Option(
        StudyCondition.system, "--condition", help="system=支援あり、none=無支援",
    ),
    wav: Path | None = typer.Option(
        None, "--wav", exists=True, dir_okay=False, help="マイクの代わりにWAVを入力",
    ),
    device: str | None = typer.Option(None, "--device", help="マイクデバイス名"),
    no_open: bool = typer.Option(False, "--no-open", help="ブラウザを自動で開かない"),
) -> None:
    """研究用の採否判断支援を起動する（マイク・外部APIを使用）。

    無支援: das study --condition none
    接続確認スクリプトの累計3回・費用予算は、この本体には適用されません。
    """
    from das.asr.live import main

    argv = ["--study-config", str(config), "--study-condition", condition.value]
    if wav is not None:
        argv += ["--wav", str(wav)]
    if device is not None:
        argv += ["--device", device]
    if no_open:
        argv.append("--no-open")
    main(argv, standalone_mode=False)
