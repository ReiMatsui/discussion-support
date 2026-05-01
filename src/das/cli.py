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
    settings = get_settings()
    typer.echo(f"[ingest] directory={directory}")

    llm = OpenAIClient()
    store = NetworkXGraphStore()
    orch = Orchestrator.assemble(llm=llm, store=store)
    nodes = await orch.ingest_documents(directory)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(store.snapshot(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    typer.echo(
        f"[ingest] {len(nodes)} nodes from {directory} -> {output}\n"
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

    snapshot_path = run_dir / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(store.snapshot(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    n_nodes = len(list(store.nodes()))
    n_edges = len(list(store.edges()))
    typer.echo(f"[run-session] done. nodes={n_nodes} edges={n_edges} -> {snapshot_path}")


if __name__ == "__main__":
    app()
