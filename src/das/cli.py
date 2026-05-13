"""Typer 製 CLI エントリポイント。

主要サブコマンド:
  - ``das version``          : バージョン表示
  - ``das ingest-docs``      : ``data/docs/`` を AF 化してスナップショット保存
  - ``das run-session``      : 発話 JSONL を流して統合 AF を構築
  - ``das listen``           : マイクから音声を取り込みリアルタイムに統合 AF を構築 (asr extras)
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
    web_search: bool = typer.Option(
        False,
        "--web-search/--no-web-search",
        help="full_proposal 条件で Web 検索エージェントを有効化 "
        "(TAVILY_API_KEY 必須)。事前資料に無い論点をリアルタイム検索",
    ),
    max_web_searches: int = typer.Option(
        5,
        "--max-web-searches",
        help="セッションあたりの Web 検索回数の上限",
    ),
    stance_polling: bool = typer.Option(
        False,
        "--stance-polling/--no-stance-polling",
        help="DEBATE benchmark 流の Pre/Post × Public/Private 立場測定を有効化。"
        "見せかけ合意 (public-private gap) を定量化",
    ),
    budget: float | None = typer.Option(
        None,
        "--budget",
        help="OpenAI 累積コスト上限 (USD)。超過すると BudgetExceeded で停止し、"
        "それまでの部分結果を保存する。例: --budget 1.0",
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
            web_search=web_search,
            max_web_searches=max_web_searches,
            stance_polling=stance_polling,
            budget=budget,
        )
    )


@app.command(name="aqua-rescore")
def aqua_rescore(
    eval_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="既存の eval ディレクトリ (例: data/eval/tier12-smoke)",
    ),
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        "-j",
        help="1 議論内で並列採点する発話数",
    ),
    n_context: int = typer.Option(
        3,
        "--n-context",
        help="採点時に渡す直前文脈の発話数",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="採点に使う LLM モデル (省略時はクライアント既定)",
    ),
    budget: float | None = typer.Option(
        None,
        "--budget",
        help="OpenAI 累積コスト上限 (USD)。超過したら部分結果を保存して停止。例: --budget 0.5",
    ),
) -> None:
    """既存 eval ディレクトリ配下の全 transcript を AQuA で再採点する。

    Behrendt et al. (DELITE @ LREC-COLING 2024) の 20 deliberation indicator を
    LLM-judge で再現する (公開ドイツ語アダプタは使わない)。各 run ディレクトリに
    ``aqua_report.json`` を追記し、``aqua_summary.json`` を eval_dir 直下に書く。
    """

    asyncio.run(_run_aqua_rescore(eval_dir=eval_dir, concurrency=concurrency,
                                  n_context=n_context, model=model, budget=budget))


@app.command(name="listen")
def listen(
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
    skip_docs: bool = typer.Option(
        False, "--skip-docs", help="ドキュメントの事前 AF 化をスキップ"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="ASR モデル名 (省略時は DAS_ASR_MODEL = large-v3)",
    ),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="ASR バックエンド (省略時は DAS_ASR_BACKEND = mlx-whisper)",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="認識言語 (省略時は DAS_ASR_LANGUAGE = ja)",
    ),
) -> None:
    """マイクからの音声をリアルタイム文字起こしして AF を構築する (asr extras 必須)。"""

    asyncio.run(
        _run_listen_async(
            docs=docs,
            run_id=run_id,
            threshold=threshold,
            top_k=top_k,
            skip_docs=skip_docs,
            model=model,
            backend=backend,
            language=language,
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


async def _run_aqua_rescore(
    *,
    eval_dir: Path,
    concurrency: int,
    n_context: int,
    model: str | None,
    budget: float | None = None,
) -> None:
    """既存 eval ディレクトリの全 transcript を AQuA で再採点する。"""

    from collections import defaultdict

    from das.eval.aqua import AQuAAgent, aggregate_aqua_reports
    from das.llm import BudgetExceeded, CostTracker

    typer.echo(f"[aqua] scanning {eval_dir}")

    # eval_dir/<condition>/run_*/transcript.jsonl を全部拾う
    run_paths: list[tuple[str, str, Path]] = []  # (condition, run_id, transcript_path)
    for cond_dir in sorted(eval_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        # condition ディレクトリは "none" / "flat_rag" / "full_proposal" 等
        for run_dir in sorted(cond_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            tpath = run_dir / "transcript.jsonl"
            if tpath.exists():
                run_paths.append((cond_dir.name, run_dir.name, tpath))

    if not run_paths:
        typer.echo("[aqua] no transcripts found", err=True)
        raise typer.Exit(1)

    typer.echo(f"[aqua] found {len(run_paths)} transcripts")

    tracker = CostTracker(budget_usd=budget) if budget is not None else CostTracker()
    if budget is not None:
        typer.echo(f"[aqua] budget cap: ${budget:.4f}")
    agent = AQuAAgent(llm=OpenAIClient(cost_tracker=tracker))

    by_condition: dict[str, list] = defaultdict(list)
    halted = False
    for i, (condition, run_id, tpath) in enumerate(run_paths, start=1):
        typer.echo(f"[aqua] ({i}/{len(run_paths)}) {condition}/{run_id}")
        transcript = _load_transcript(tpath)
        try:
            report = await agent.score_discussion(
                transcript,
                n_context=n_context,
                concurrency=concurrency,
                model=model,
            )
        except BudgetExceeded as exc:
            typer.echo(f"[aqua] BUDGET EXCEEDED: {exc}", err=True)
            typer.echo("[aqua] stopping. Partial results saved so far.", err=True)
            halted = True
            break  # 末尾の summary 保存に進むためここで停止
        out_path = tpath.parent / "aqua_report.json"
        out_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        by_condition[condition].append(report)
        typer.echo(
            f"[aqua]   mean={report.mean:.3f}  std={report.std:.3f}  "
            f"n_utt={report.n_utterances}  -> {out_path.name}  "
            f"[cost: {tracker.format_status()}]"
        )

    # 集約サマリ
    summary = {
        "eval_dir": str(eval_dir),
        "n_runs_scored": sum(len(v) for v in by_condition.values()),
        "n_runs_total": len(run_paths),
        "n_context": n_context,
        "halted_by_budget": halted,
        "cost": tracker.snapshot(),
        "by_condition": {
            cond: aggregate_aqua_reports(reps) for cond, reps in by_condition.items()
        },
    }
    summary_path = eval_dir / "aqua_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"[aqua] summary -> {summary_path}")
    typer.echo(f"[aqua] total cost: {tracker.format_status()}")

    if halted:
        import sys as _sys

        typer.echo("[aqua] halted by budget — exiting with code 2", err=True)
        _sys.exit(2)

    # コンソール表示
    typer.echo("")
    typer.echo("--- AQuA summary (mean ± std, 0-5 scale) ---")
    for cond, agg in summary["by_condition"].items():
        typer.echo(f"  {cond}: {agg['mean']:.3f} ± {agg['std']:.3f}  (n={agg['n_runs']} runs)")

    # 2 条件以上あるときは「指標別の条件差トップ」も出す (full_proposal vs none を優先)
    cond_names = list(summary["by_condition"].keys())
    if len(cond_names) >= 2:
        from das.eval.aqua import INDICATORS

        # 基準: full_proposal を 1 番手に、なければ最後を 1 番手に
        primary = "full_proposal" if "full_proposal" in cond_names else cond_names[-1]
        baseline = "none" if "none" in cond_names else cond_names[0]
        if primary == baseline:
            return

        ind_by_name = {ind.name: ind for ind in INDICATORS}
        primary_means = summary["by_condition"][primary]["per_indicator_mean"]
        baseline_means = summary["by_condition"][baseline]["per_indicator_mean"]

        # |差| × |重み| でソート (集計に効くもの順)
        rows: list[tuple[str, float, float, float, float]] = []
        for name, p_val in primary_means.items():
            b_val = baseline_means.get(name, 0.0)
            diff = p_val - b_val
            weight = ind_by_name[name].weight
            contribution = diff * weight
            rows.append((name, p_val, b_val, diff, contribution))
        rows.sort(key=lambda r: abs(r[4]), reverse=True)

        typer.echo("")
        typer.echo(
            f"--- 指標別の条件差トップ 7 ({primary} - {baseline}, |diff×weight| 降順) ---"
        )
        typer.echo(
            f"  {'indicator':<24} {primary:>14} {baseline:>10} {'diff':>8} {'weight':>8} {'集計寄与':>10}"
        )
        for name, p_val, b_val, diff, contribution in rows[:7]:
            sign = "+" if contribution >= 0 else "-"
            typer.echo(
                f"  {name:<24} {p_val:>14.3f} {b_val:>10.3f} {diff:+8.3f} "
                f"{ind_by_name[name].weight:+8.3f} {sign}{abs(contribution):>9.4f}"
            )


async def _run_listen_async(
    *,
    docs: Path | None,
    run_id: str | None,
    threshold: float | None,
    top_k: int,
    skip_docs: bool,
    model: str | None,
    backend: str | None,
    language: str | None,
) -> None:
    """マイク → ASR → Orchestrator.run_live → snapshot 保存。

    ``run-session`` のライブ版。設計のキモは:
      - ASR 関連の重い import は **この関数の中まで遅延** させて、[asr] extras
        が無い環境でも CLI 自体は import できるようにしておく
      - マイク取り込みと utterance 消費を別タスクで走らせ、Ctrl-C 時には
        EOS (空フレーム) を ASR に送って残りの確定行をフラッシュさせる
    """

    import contextlib
    import signal
    import sys
    from collections.abc import AsyncIterator

    settings = get_settings()
    docs_dir = docs if docs is not None else settings.docs_dir
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[listen] run_dir={run_dir}")
    typer.echo(f"[listen] docs_dir={docs_dir} (skip={skip_docs})")

    # ASR モジュールは [asr] extras 経由でしか入らない。遅延 import で
    # 「extras を入れていない人にもまともなエラーメッセージを出す」を実現。
    try:
        from das.asr import LiveAsrSession, build_engine
        from das.asr.mic import iter_mic_chunks
    except ImportError as exc:
        typer.echo(f"[listen] ASR 依存が未インストールです: {exc}")
        typer.echo("`uv sync --extra asr` を実行してください。")
        raise typer.Exit(1) from exc

    llm = OpenAIClient()
    store = NetworkXGraphStore(db_path=run_dir / "graph.sqlite")
    orch = Orchestrator.assemble(llm=llm, store=store, threshold=threshold, top_k=top_k)

    if not skip_docs and docs_dir.exists():
        typer.echo("[listen] ingesting documents...")
        await orch.ingest_documents(docs_dir)

    typer.echo(
        "[listen] loading ASR engine "
        "(初回はモデル DL で数十秒〜数分かかることがあります)..."
    )
    engine = build_engine(
        model=model,
        backend=backend,
        language=language,
        diarization=False,
    )

    stop_event = asyncio.Event()

    def _on_partial(text: str) -> None:
        # 同じ行を上書きで表示 (確定行は \n 付きで別行に出る)
        sys.stdout.write(f"\r... {text[:120]}")
        sys.stdout.flush()

    session = LiveAsrSession(engine=engine, on_partial=_on_partial)
    await session.start()

    # Ctrl-C で停止フラグを立てる。再度押されたら強制中断 (default 挙動に戻す)。
    loop = asyncio.get_running_loop()

    def _on_sigint() -> None:
        if not stop_event.is_set():
            typer.echo("\n[listen] 停止要求 (Ctrl-C)。残りをフラッシュ中...")
            stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except NotImplementedError:  # pragma: no cover - Windows
        pass

    async def _pump_mic() -> None:
        async for chunk in iter_mic_chunks(stop_event=stop_event):
            await session.push_audio(chunk)
        # 入力が止まったら EOS を投げて ASR ジェネレータを終わりに向かわせる
        await session.stop()

    async def _utt_stream() -> AsyncIterator[Utterance]:
        async for utt in session.iter_utterances():
            sys.stdout.write(f"\r[t{utt.turn_id}] {utt.speaker}: {utt.text}\n")
            sys.stdout.flush()
            yield utt

    typer.echo("[listen] 録音開始。Ctrl-C で停止。")
    mic_task = asyncio.create_task(_pump_mic())
    try:
        await orch.run_live(_utt_stream())
    finally:
        # 異常終了でもマイクを確実に止める
        stop_event.set()
        try:
            await asyncio.wait_for(mic_task, timeout=10.0)
        except TimeoutError:
            mic_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mic_task
        await session.cleanup()

    from das.viz import dump_snapshot

    snapshot_path = dump_snapshot(store, run_dir / "snapshot.json")
    n_nodes = len(list(store.nodes()))
    n_edges = len(list(store.edges()))

    html_path: Path | None = None
    try:
        from das.viz import render_html

        html_path = render_html(store, run_dir / "graph.html")
    except ImportError as exc:
        typer.echo(
            f"[listen] HTML 生成をスキップ (viz extras 未インストール: {exc}). "
            f"`das visualize {snapshot_path}` で後から生成できます。"
        )

    summary = (
        f"\n[listen] done. nodes={n_nodes} edges={n_edges}\n"
        f"  snapshot -> {snapshot_path}"
    )
    if html_path is not None:
        summary += f"\n  html     -> {html_path}"
    typer.echo(summary)


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
    web_search: bool = False,
    max_web_searches: int = 5,
    stance_polling: bool = False,
    budget: float | None = None,
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
    from das.llm import BudgetExceeded, CostTracker

    # (persona_factory, topic, default_docs_subdir)。
    # preset ごとに docs サブディレクトリを分けることで、トピックを切り替えても
    # 関係ない文書ノードが混じらないようにする。
    presets: dict[str, tuple] = {
        "cafeteria": (
            cafeteria_personas,
            "大学のカフェテリアでプラスチック容器を廃止すべきか",
            "docs",
        ),
        "policy_ai": (
            policy_ai_lecture_personas,
            "生成 AI を大学の講義・レポート作成で許容すべきか",
            "docs_policy",
        ),
    }
    if preset not in presets:
        typer.echo(f"未知の preset: {preset}. 利用可能: {list(presets.keys())}")
        raise typer.Exit(1)

    persona_factory, topic, default_docs_subdir = presets[preset]
    personas = persona_factory()

    settings = get_settings()
    preset_docs_dir = settings.data_dir / default_docs_subdir
    docs_dir = docs if docs is not None else preset_docs_dir
    target_eval_dir = eval_dir if eval_dir is not None else settings.data_dir / "eval"

    tracker = CostTracker(budget_usd=budget) if budget is not None else CostTracker()
    if budget is not None:
        typer.echo(f"[eval] budget cap: ${budget:.4f} (BudgetExceeded で停止し部分結果は保存される)")
    llm = OpenAIClient(cost_tracker=tracker)
    factories: dict = {}
    for name in (c.strip() for c in conditions.split(",") if c.strip()):
        if name == "none":
            factories[name] = ConditionNone
        elif name == "flat_rag":
            factories[name] = lambda llm=llm: ConditionFlatRAG(llm=llm)
        elif name == "full_proposal":
            factories[name] = lambda llm=llm: ConditionFullProposal(
                llm=llm,
                enable_web_search=web_search,
                max_web_searches=max_web_searches,
            )
        else:
            typer.echo(f"未知の condition: {name}")
            raise typer.Exit(1)

    judge = None if no_judge else JudgeAgent(llm=llm)

    # LLM-judge ベースの合意検出 (Sirota et al. SIGDIAL 2025)
    consensus_agent: object | None = None
    if llm_consensus:
        from das.agents.consensus_agent import ConsensusAgent

        consensus_agent = ConsensusAgent(llm=llm)

    # Stance polling agent (DEBATE benchmark 流)
    stance_agent_obj = None
    if stance_polling:
        from das.agents.stance_agent import StanceAgent

        stance_agent_obj = StanceAgent(llm=llm)

    typer.echo(
        f"[eval] preset={preset} topic='{topic}' "
        f"conditions={list(factories.keys())} n_runs={n_runs} "
        f"max_turns={max_turns} concurrency={concurrency} "
        f"until_consensus={'on' if until_consensus else 'off'} "
        f"judge={'on' if judge else 'off'}"
    )

    def _progress(cond: str, done: int, total: int) -> None:
        typer.echo(
            f"  [{done}/{total}] condition={cond}  [cost: {tracker.format_status()}]"
        )

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

    try:
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
            stance_agent=stance_agent_obj,
        )
    except BudgetExceeded as exc:
        import sys as _sys

        typer.echo("")
        typer.echo(f"[eval] BUDGET EXCEEDED: {exc}", err=True)
        typer.echo("[eval] 既に保存された run_* は eval_dir 配下に残っています。", err=True)
        # cost snapshot を eval_dir 直下に保存しておく (部分結果のサマリ用)
        if eval_id is not None or target_eval_dir is not None:
            actual_eval_id = eval_id or "eval-aborted"
            cost_dir = target_eval_dir / actual_eval_id
            cost_dir.mkdir(parents=True, exist_ok=True)
            (cost_dir / "cost_snapshot.json").write_text(
                json.dumps(tracker.snapshot(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            typer.echo(f"[eval] cost snapshot -> {cost_dir / 'cost_snapshot.json'}")
        typer.echo(f"[eval] final cost: {tracker.format_status()}")
        # typer.Exit が asyncio.run を通すと exit code 0 になることがあるので
        # sys.exit を直接使う (SystemExit は asyncio.run を確実に貫通する)
        _sys.exit(2)

    typer.echo("")
    typer.echo(f"[eval] done. eval_id={result.eval_id}")
    if result.eval_dir is not None:
        typer.echo(f"[eval] saved to {result.eval_dir}")
        # 通常終了時も cost_snapshot を保存して再現性を確保
        (result.eval_dir / "cost_snapshot.json").write_text(
            json.dumps(tracker.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    typer.echo(f"[eval] final cost: {tracker.format_status()}")

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
