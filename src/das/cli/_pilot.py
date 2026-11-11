"""小規模試行の準備・録画・本体起動をまとめた入口。"""
from __future__ import annotations

import datetime
from pathlib import Path

import typer

from das.cli import app


@app.command(name="pilot")
def pilot(
    output: Path | None = typer.Option(None, "--output", help="記録の保存先"),
    port: int = typer.Option(8271, "--port", min=0, max=65535),
    no_open: bool = typer.Option(False, "--no-open", help="ブラウザを自動で開かない"),
) -> None:
    """3人での試行GUIを開く。録画と外部APIは画面の操作後に開始する。"""
    from das.pilot.server import serve
    from das.pilot.session import PilotSession

    root = Path(__file__).resolve().parents[3]
    output = output or root / "transcripts/pilot" / datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    serve(PilotSession(root, output), port, open_browser=not no_open)
