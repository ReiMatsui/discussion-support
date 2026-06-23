"""Typer 製 CLI エントリポイント。

主要サブコマンド:
  - ``das version``          : バージョン表示
  - ``das ingest-docs``      : ``data/docs/`` を AF 化してスナップショット保存
  - ``das run-session``      : 発話 JSONL を流して統合 AF を構築
  - ``das listen``           : マイクから音声を取り込みリアルタイムに統合 AF を構築 (asr extras)
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import typer

from das import __version__
from das.llm import OpenAIClient
from das.logging import configure_logging
from das.runtime import Orchestrator
from das.settings import get_settings

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


@app.command(name="ingest-docs")
def ingest_docs(
    directory: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="ドキュメントを置いたディレクトリ"
    ),
    output: Path = typer.Option(
        Path("data/runs/docs_snapshot.json"),
        "--output",
        "-o",
        help="AF スナップショットの保存先 JSON",
    ),
) -> None:
    """ディレクトリ内の文書を AF 化してスナップショットを保存する。"""

    asyncio.run(_run_ingest_docs(directory, output))


@app.command(name="ui")
def ui(
    port: int = typer.Option(8501, "--port", help="Streamlit のポート"),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="ブラウザを自動で開かない",
    ),
) -> None:
    """Streamlit ベースの議論グラフ ビューアを起動する (ui extras 必要)。"""

    try:
        from das.ui import streamlit_app
    except ImportError as exc:  # pragma: no cover
        typer.echo(f"UI 依存が未インストールです: {exc}")
        typer.echo("`uv sync --extra ui` (もしくは `uv sync --all-extras`) を実行してください。")
        raise typer.Exit(1) from exc

    app_path = Path(streamlit_app.__file__)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    if headless:
        cmd += ["--server.headless", "true"]
    raise typer.Exit(subprocess.call(cmd))


@app.command(name="visualize")
def visualize(
    snapshot: Path = typer.Argument(..., exists=True, dir_okay=False, help="snapshot.json"),
    output: Path = typer.Option(
        Path("graph.html"),
        "--output",
        "-o",
        help="出力 HTML パス",
    ),
) -> None:
    """``snapshot.json`` を pyvis HTML として可視化する。"""

    from das.viz import load_snapshot as _load
    from das.viz import render_html

    store = _load(snapshot)
    out = render_html(store, output)
    typer.echo(f"[visualize] wrote {out}")


# --- 実体 --------------------------------------------------------------


async def _run_ingest_docs(directory: Path, output: Path) -> None:
    from das.graph.store import NetworkXGraphStore
    from das.viz import dump_snapshot

    settings = get_settings()
    typer.echo(f"[ingest] directory={directory}")

    llm = OpenAIClient()
    store = NetworkXGraphStore()
    orch = Orchestrator.assemble(llm=llm, store=store)
    nodes = await orch.ingest_documents(directory)

    snapshot_path = dump_snapshot(store, output)
    typer.echo(
        f"[ingest] {len(nodes)} nodes from {directory} -> {snapshot_path}\n"
        f"         data_dir={settings.data_dir}"
    )


# サブモジュールを import して @app.command デコレータを登録する。
# app が定義された後に import しないと循環参照になる。
from das.cli import _eval as _eval  # noqa: E402, F401
from das.cli import _listen as _listen  # noqa: E402, F401
from das.cli import _session as _session  # noqa: E402, F401
