"""run-session サブコマンド。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from das.cli import app
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.runtime import Orchestrator
from das.settings import get_settings
from das.types import Utterance


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
    top_k: int = typer.Option(5, "--top-k", help="リンク候補の embedding top-k (混合)"),
    top_k_per_source: int = typer.Option(
        0,
        "--top-k-per-source",
        help="source 別 top-k モード。0 (= 既定) なら混合 top-k を使う。"
        "正の値で utterance / document / web の各バケットから top-N を取る (Fix A)",
    ),
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
            top_k_per_source=top_k_per_source if top_k_per_source > 0 else None,
            skip_docs=skip_docs,
        )
    )


# --- 実体 --------------------------------------------------------------


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
    top_k_per_source: int | None,
    skip_docs: bool,
) -> None:
    settings = get_settings()
    docs_dir = docs if docs is not None else settings.docs_dir
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[run-session] transcript={transcript}")
    typer.echo(f"[run-session] docs_dir={docs_dir} (skip={skip_docs})")
    typer.echo(f"[run-session] run_dir={run_dir}")

    llm = OpenAIClient()
    store = NetworkXGraphStore(db_path=run_dir / "graph.sqlite")
    orch = Orchestrator.assemble(
        llm=llm,
        store=store,
        threshold=threshold,
        top_k=top_k,
        top_k_per_source=top_k_per_source,
    )

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
