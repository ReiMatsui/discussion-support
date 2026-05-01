"""Typer 製 CLI エントリポイント。

主要サブコマンド:
  - ``das version``          : バージョン表示
  - ``das ingest-docs``      : ``data/docs/`` を AF 化してスナップショット保存
  - ``das run-session``      : 発話 JSONL を流して統合 AF を構築
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from das import __version__
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.logging import configure_logging
from das.runtime import Orchestrator
from das.settings import get_settings
from das.types import Utterance

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

    import subprocess
    import sys

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


@app.command(name="run-session")
def run_session(
    transcript: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="発話 1 件 = 1 行の JSONL ファイル",
    ),
    docs: Path | None = typer.Option(
        None,
        "--docs",
        help="議論前に取り込む文書ディレクトリ (省略時は data/docs)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="出力先サブディレクトリ名 (省略時は ISO タイムスタンプ)",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="リンク採用の信頼度閾値 (省略時は設定値)",
    ),
    top_k: int = typer.Option(5, "--top-k", help="リンク候補の embedding top-k"),
    skip_docs: bool = typer.Option(False, "--skip-docs", help="ドキュメントの事前 AF 化をスキップ"),
) -> None:
    """テキスト議論ログを流して統合 AF を構築する。"""

    asyncio.run(
        _run_session_async(
            transcript=transcript,
            docs=docs,
            run_id=run_id,
            threshold=threshold,
            top_k=top_k,
            skip_docs=skip_docs,
        )
    )


# --- 実体 --------------------------------------------------------------


async def _run_ingest_docs(directory: Path, output: Path) -> None:
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


def _load_transcript(path: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            utterances.append(Utterance.model_validate(payload))
    return utterances


async def _run_session_async(
    *,
    transcript: Path,
    docs: Path | None,
    run_id: str | None,
    threshold: float | None,
    top_k: int,
    skip_docs: bool,
) -> None:
    settings = get_settings()
    docs_dir = docs if docs is not None else settings.docs_dir
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[run-session] transcript={transcript}")
    typer.echo(f"[run-session] docs_dir={docs_dir} (skip={skip_docs})")
    typer.echo(f"[run-session] run_dir={run_dir}")

    llm = OpenAIClient()
    store = NetworkXGraphStore(db_path=run_dir / "graph.sqlite")
    orch = Orchestrator.assemble(llm=llm, store=store, threshold=threshold, top_k=top_k)

    if not skip_docs and docs_dir.exists():
        typer.echo("[run-session] ingesting documents...")
        await orch.ingest_documents(docs_dir)

    utterances = _load_transcript(transcript)
    typer.echo(f"[run-session] running {len(utterances)} utterances...")
    await orch.run_session(utterances)

    from das.viz import dump_snapshot

    # snapshot は最優先で保存 (HTML より先に書き出して結果を守る)
    snapshot_path = dump_snapshot(store, run_dir / "snapshot.json")
    n_nodes = len(list(store.nodes()))
    n_edges = len(list(store.edges()))

    html_path: Path | None = None
    try:
        from das.viz import render_html

        html_path = render_html(store, run_dir / "graph.html")
    except ImportError as exc:
        typer.echo(
            f"[run-session] HTML 生成をスキップ (viz extras 未インストール: {exc}).\n"
            f"             `uv sync --extra viz` で有効化、または "
            f"`das visualize {snapshot_path}` で後から生成できます。"
        )

    summary = f"[run-session] done. nodes={n_nodes} edges={n_edges}\n  snapshot -> {snapshot_path}"
    if html_path is not None:
        summary += f"\n  html     -> {html_path}"
    typer.echo(summary)


if __name__ == "__main__":
    app()
