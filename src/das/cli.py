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


@app.command(name="eval")
def eval_cmd(
    preset: str = typer.Argument(
        "cafeteria",
        help="トピックプリセット: cafeteria / policy_ai",
    ),
    n_runs: int = typer.Option(2, "--n-runs", "-n", help="各条件を何回回すか"),
    max_turns: int = typer.Option(
        20,
        "--max-turns",
        "-t",
        help="各セッションのターン上限 (--until-consensus 利用時は安全上限として機能)",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        "-j",
        help="並列実行する (condition, run) 数。API レート制限に注意",
    ),
    temperature: float = typer.Option(0.7, "--temperature", help="persona の生成 temperature"),
    conditions: str = typer.Option(
        "none,flat_rag,full_proposal",
        "--conditions",
        help="走らせる条件 (カンマ区切り)",
    ),
    no_judge: bool = typer.Option(False, "--no-judge", help="LLM-as-judge をスキップ"),
    until_consensus: bool = typer.Option(
        False,
        "--until-consensus",
        help="合意検出されたら max_turns 未満でも早期終了する (合意形成までの時間を計測)",
    ),
    agreement_window: int = typer.Option(
        3, "--agreement-window", help="合意キーワード判定の直近ターン数"
    ),
    agreement_threshold: float = typer.Option(
        0.67,
        "--agreement-threshold",
        help="合意キーワード割合のしきい値 (0..1)。"
        "逆接「確かに〜が、」は事前に除外される",
    ),
    min_turns_before_consensus: int = typer.Option(
        6,
        "--min-turns-before-consensus",
        help="合意判定を始める最小ターン数 (序盤の誤検出を避ける)",
    ),
    eval_id: str | None = typer.Option(None, "--eval-id", help="出力先 eval_id (省略時は自動)"),
    docs: Path | None = typer.Option(None, "--docs", help="ドキュメントディレクトリ"),
    eval_dir: Path | None = typer.Option(
        None,
        "--eval-dir",
        help="出力ベースディレクトリ (省略時は data/eval)",
    ),
    emit_events: bool = typer.Option(
        False,
        "--emit-events",
        help="UI 連携用: 各イベント (utterance/intervention/run_start/run_end) "
        "を `__DAS_EVT__<json>` 行として stdout に流す",
    ),
    llm_consensus: bool = typer.Option(
        True,
        "--llm-consensus/--no-llm-consensus",
        help="LLM-judge による合意検出を有効化 (Sirota et al. SIGDIAL 2025)。"
        "構造シグナルが立ったときだけ呼ぶので追加コストは小さい",
    ),
) -> None:
    """シミュレーション評価を一括実行する (3 条件比較 + LLM-as-judge)。"""

    asyncio.run(
        _run_eval_cli(
            preset=preset,
            n_runs=n_runs,
            max_turns=max_turns,
            temperature=temperature,
            conditions=conditions,
            no_judge=no_judge,
            eval_id=eval_id,
            docs=docs,
            eval_dir=eval_dir,
            until_consensus=until_consensus,
            agreement_window=agreement_window,
            agreement_threshold=agreement_threshold,
            min_turns_before_consensus=min_turns_before_consensus,
            concurrency=concurrency,
            emit_events=emit_events,
            llm_consensus=llm_consensus,
        )
    )


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


async def _run_eval_cli(
    *,
    preset: str,
    n_runs: int,
    max_turns: int,
    temperature: float,
    conditions: str,
    no_judge: bool,
    eval_id: str | None,
    docs: Path | None,
    eval_dir: Path | None,
    until_consensus: bool = False,
    agreement_window: int = 3,
    agreement_threshold: float = 0.6,
    min_turns_before_consensus: int = 4,
    concurrency: int = 1,
    emit_events: bool = False,
    llm_consensus: bool = True,
) -> None:
    from das.eval import (
        ConditionFlatRAG,
        ConditionFullProposal,
        ConditionNone,
        JudgeAgent,
        cafeteria_personas,
        policy_ai_lecture_personas,
        run_eval,
    )

    presets = {
        "cafeteria": (
            cafeteria_personas,
            "大学のカフェテリアでプラスチック容器を廃止すべきか",
        ),
        "policy_ai": (
            policy_ai_lecture_personas,
            "生成 AI を大学の講義・レポート作成で許容すべきか",
        ),
    }
    if preset not in presets:
        typer.echo(f"未知の preset: {preset}. 利用可能: {list(presets.keys())}")
        raise typer.Exit(1)

    persona_factory, topic = presets[preset]
    personas = persona_factory()

    settings = get_settings()
    docs_dir = docs if docs is not None else settings.docs_dir
    target_eval_dir = eval_dir if eval_dir is not None else settings.data_dir / "eval"

    llm = OpenAIClient()
    factories: dict = {}
    for name in (c.strip() for c in conditions.split(",") if c.strip()):
        if name == "none":
            factories[name] = ConditionNone
        elif name == "flat_rag":
            factories[name] = lambda llm=llm: ConditionFlatRAG(llm=llm)
        elif name == "full_proposal":
            factories[name] = lambda llm=llm: ConditionFullProposal(llm=llm)
        else:
            typer.echo(f"未知の condition: {name}")
            raise typer.Exit(1)

    judge = None if no_judge else JudgeAgent(llm=llm)

    # LLM-judge ベースの合意検出 (Sirota et al. SIGDIAL 2025)
    consensus_agent: object | None = None
    if llm_consensus:
        from das.agents.consensus_agent import ConsensusAgent

        consensus_agent = ConsensusAgent(llm=llm)

    typer.echo(
        f"[eval] preset={preset} topic='{topic}' "
        f"conditions={list(factories.keys())} n_runs={n_runs} "
        f"max_turns={max_turns} concurrency={concurrency} "
        f"until_consensus={'on' if until_consensus else 'off'} "
        f"judge={'on' if judge else 'off'}"
    )

    def _progress(cond: str, done: int, total: int) -> None:
        typer.echo(f"  [{done}/{total}] condition={cond}")

    consensus_kwargs = {
        "agreement_window": agreement_window,
        "agreement_threshold": agreement_threshold,
        "min_turns_before_consensus": min_turns_before_consensus,
    }

    # UI 連携: --emit-events でイベントを stdout に流す。
    # ログ行と区別できるよう先頭に sentinel ``__DAS_EVT__`` を付ける。
    event_emitter = None
    if emit_events:
        import sys as _sys

        def _emit(payload: dict) -> None:
            line = "__DAS_EVT__" + json.dumps(payload, ensure_ascii=False, default=str)
            print(line, flush=True, file=_sys.stdout)

        event_emitter = _emit

    result = await run_eval(
        topic=topic,
        personas=personas,
        condition_factories=factories,
        n_runs=n_runs,
        max_turns=max_turns,
        temperature=temperature,
        docs_dir=docs_dir if docs_dir.exists() else None,
        llm=llm,
        judge=judge,
        eval_dir=target_eval_dir,
        eval_id=eval_id,
        progress=_progress,
        until_consensus=until_consensus,
        consensus_kwargs=consensus_kwargs,
        concurrency=concurrency,
        event_emitter=event_emitter,
        consensus_agent=consensus_agent,
    )

    typer.echo("")
    typer.echo(f"[eval] done. eval_id={result.eval_id}")
    if result.eval_dir is not None:
        typer.echo(f"[eval] saved to {result.eval_dir}")

    # 収束統計 (until_consensus 関係なく常に表示: 後追い判定で意味がある)
    typer.echo("")
    typer.echo("[eval] convergence:")
    grouped = result.by_condition()
    for cond, runs in grouped.items():
        n = len(runs)
        if n == 0:
            continue
        n_conv = sum(1 for r in runs if r.consensus and r.consensus.consensus_reached)
        mean_turns = sum(r.n_turns for r in runs) / n
        conv_turns = [
            r.consensus.detected_at_turn
            for r in runs
            if r.consensus and r.consensus.consensus_reached and r.consensus.detected_at_turn
        ]
        ttc = sum(conv_turns) / len(conv_turns) if conv_turns else None
        ttc_str = f"{ttc:.1f}" if ttc is not None else "-"
        typer.echo(
            f"  {cond}: 収束 {n_conv}/{n} ({n_conv / n:.0%}), "
            f"平均ターン {mean_turns:.1f}, 平均到達ターン {ttc_str}"
        )

    if judge is not None:
        typer.echo("")
        typer.echo("[eval] aggregated scores:")
        for cond, agg in result.aggregate().items():
            typer.echo(
                f"  {cond}: 満足度 {agg.overall_satisfaction_mean:.2f}±{agg.overall_satisfaction_std:.2f}, "
                f"反対理解 {agg.opposition_understanding_mean:.2f}±{agg.opposition_understanding_std:.2f}, "
                f"透明性 {agg.intervention_transparency_mean:.2f}±{agg.intervention_transparency_std:.2f}"
            )


if __name__ == "__main__":
    app()
