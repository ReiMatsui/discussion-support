"""Typer 製 CLI エントリポイント。

サブコマンドは M1.7 以降で本実装する。今は雛形のみ。
"""

from __future__ import annotations

import typer

from das import __version__
from das.logging import configure_logging

app = typer.Typer(
    name="das",
    help="Discussion Argumentation Support — 議論グラフ統合型 議論支援",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _root(
    log_level: str = typer.Option("INFO", "--log-level", help="ログレベル"),
) -> None:
    configure_logging(level=log_level)


@app.command()
def version() -> None:
    """バージョンを表示する。"""

    typer.echo(__version__)


@app.command(name="run-session")
def run_session(transcript: str) -> None:
    """テキスト議論ログを流して統合 AF を構築する (M1.7 で実装)。"""

    typer.echo(f"[stub] would run session for: {transcript}")


@app.command(name="ingest-docs")
def ingest_docs(directory: str) -> None:
    """ドキュメントを論証ノードに分解して永続化する (M1.5 で実装)。"""

    typer.echo(f"[stub] would ingest docs from: {directory}")


if __name__ == "__main__":
    app()
