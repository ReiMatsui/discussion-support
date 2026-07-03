"""eval / aqua-rescore サブコマンド。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from das.cli import app
from das.llm import OpenAIClient
from das.settings import get_settings
from das.types import Utterance


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
        help="OpenAI 累積コストの **ソフト上限** (USD)。超過後、"
        "新規 run の開始は gate されるが、進行中の run は最後まで完走する。例: --budget 1.5",
    ),
    hard_budget: float | None = typer.Option(
        None,
        "--hard-budget",
        help="OpenAI 累積コストの **ハード上限** (USD)。超過すると **進行中の "
        "API 呼び出しごと**即停止 (BudgetExceededError)。--budget の 1.5〜2x を推奨。"
        "省略すると hard cap なし (in-flight は際限なく完走)",
    ),
    cond_concurrency: str = typer.Option(
        "",
        "--cond-concurrency",
        help="condition 別の並列度上限 (カンマ区切り)。例: "
        "'full_proposal=1,flat_rag=2'。指定なしは global concurrency と同じ。"
        "予算制約下で重い condition を sequential に走らせて部分結果を守る用途",
    ),
    linking_top_k: int = typer.Option(
        5,
        "--linking-top-k",
        help="LinkingAgent の candidate 数 (混合 top-k モード)。"
        "``--linking-top-k-per-source`` を指定するとそちらが優先される。"
        "下げると per-node の LLM 呼び出しが減ってコスト減 (5→3 で 40% 削減)",
    ),
    linking_top_k_per_source: int = typer.Option(
        0,
        "--linking-top-k-per-source",
        help="source 別 top-k モード。0 (= 既定) なら混合 top-k を使う。"
        "正の値を指定すると utterance / document / web の各バケットから top-N を取る。"
        "発話どうしの類似が高い議論で文書/Web 枠が押し出される問題を防ぐ (Fix A)",
    ),
    linking_model: str = typer.Option(
        "",
        "--linking-model",
        help="LinkingAgent の judgment で使うモデルを上書き。例: 'gpt-5-nano'。"
        "Linking が全コストの 80-90% を占めるので、ここを cheap モデルに替えると "
        "full_proposal の per-run コストが 70-80% 削減される",
    ),
) -> None:
    """シミュレーション評価を一括実行する (3 条件比較 + LLM-as-judge)。"""

    # cond_concurrency をパース
    parsed_cond_conc: dict[str, int] = {}
    for token in (s.strip() for s in cond_concurrency.split(",") if s.strip()):
        if "=" not in token:
            typer.echo(f"--cond-concurrency の書式エラー: '{token}'。'name=N' 形式")
            raise typer.Exit(1)
        name, n_str = token.split("=", 1)
        try:
            parsed_cond_conc[name.strip()] = int(n_str.strip())
        except ValueError as exc:
            typer.echo(f"--cond-concurrency の値が int でない: '{token}'")
            raise typer.Exit(1) from exc

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
            hard_budget=hard_budget,
            condition_concurrency=parsed_cond_conc,
            linking_top_k=linking_top_k,
            linking_top_k_per_source=(
                linking_top_k_per_source if linking_top_k_per_source > 0 else None
            ),
            linking_model=linking_model.strip() or None,
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
        help="累積コストの **ソフト上限** (USD)。超過後の新規 transcript の "
        "scoring は skip、進行中は完走。例: --budget 0.5",
    ),
    hard_budget: float | None = typer.Option(
        None,
        "--hard-budget",
        help="累積コストの **ハード上限** (USD)。進行中の API 呼び出しごと即停止。",
    ),
) -> None:
    """既存 eval ディレクトリ配下の全 transcript を AQuA で再採点する。

    Behrendt et al. (DELITE @ LREC-COLING 2024) の 20 deliberation indicator を
    LLM-judge で再現する (公開ドイツ語アダプタは使わない)。各 run ディレクトリに
    ``aqua_report.json`` を追記し、``aqua_summary.json`` を eval_dir 直下に書く。
    """

    asyncio.run(_run_aqua_rescore(eval_dir=eval_dir, concurrency=concurrency,
                                  n_context=n_context, model=model,
                                  budget=budget, hard_budget=hard_budget))


@app.command(name="eval-rescore")
def eval_rescore(
    eval_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="既存の eval ディレクトリ (das eval が生成したもの)",
    ),
    no_judge: bool = typer.Option(
        False,
        "--no-judge",
        help="judge (LLM 主観採点) を再実行しない。構造指標・citation のみ再計算。",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="judge に使う LLM モデル (省略時はクライアント既定の smart_model)",
    ),
    budget: float | None = typer.Option(
        None, "--budget", help="累積コストの **ソフト上限** (USD)。"
    ),
    hard_budget: float | None = typer.Option(
        None, "--hard-budget", help="累積コストの **ハード上限** (USD)。"
    ),
) -> None:
    """既存 eval を **会話再生成なしで** 再採点する (実行と採点の分離)。

    保存済みの transcript / 介入ログ / AF snapshot / 生 stance から、judge・構造指標・
    citation・consensus を再計算し、judge_reports.json / run_meta.json / summary.json を
    作り直す。judge プロンプトや citation 閾値を直したあとの再採点に使う。

    旧形式 (personas / topic / transcript を保存していない) の eval は明確なエラーで
    停止する (黙って誤計算しない)。
    """

    asyncio.run(
        _run_eval_rescore(
            eval_dir=eval_dir,
            no_judge=no_judge,
            model=model,
            budget=budget,
            hard_budget=hard_budget,
        )
    )


async def _run_eval_rescore(
    *,
    eval_dir: Path,
    no_judge: bool,
    model: str | None,
    budget: float | None,
    hard_budget: float | None,
) -> None:
    from das.eval.judge import JudgeAgent
    from das.eval.rescore import RescoreError, rescore_eval_dir
    from das.llm import CostTracker

    tracker = (
        CostTracker(budget_usd=budget, hard_budget_usd=hard_budget)
        if (budget is not None or hard_budget is not None)
        else CostTracker()
    )
    llm = OpenAIClient(cost_tracker=tracker)
    judge = None if no_judge else JudgeAgent(llm=llm, model=model)

    typer.echo(f"[rescore] scanning {eval_dir} (judge={'off' if no_judge else 'on'})")
    try:
        result = await rescore_eval_dir(eval_dir, llm=llm, judge=judge)
    except RescoreError as exc:
        typer.echo(f"[rescore] ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(
        f"[rescore] done: {len(result.runs)} runs re-scored, "
        f"summary.json 更新済み  [cost: {tracker.format_status()}]"
    )
    for cond, agg in result.aggregate().items():
        typer.echo(
            f"  {cond}: n_runs={agg.n} "
            f"満足度 {agg.overall_satisfaction_mean:.2f}±{agg.overall_satisfaction_std:.2f}"
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


async def _run_aqua_rescore(
    *,
    eval_dir: Path,
    concurrency: int,
    n_context: int,
    model: str | None,
    budget: float | None = None,
    hard_budget: float | None = None,
) -> None:
    """既存 eval ディレクトリの全 transcript を AQuA で再採点する。"""

    from collections import defaultdict

    from das.eval.aqua import AQuAAgent, aggregate_aqua_reports
    from das.llm import BudgetExceededError, CostTracker

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

    tracker = (
        CostTracker(budget_usd=budget, hard_budget_usd=hard_budget)
        if (budget is not None or hard_budget is not None)
        else CostTracker()
    )
    if budget is not None:
        typer.echo(f"[aqua] soft budget: ${budget:.4f} (in-flight transcript は完走)")
    if hard_budget is not None:
        typer.echo(f"[aqua] hard budget: ${hard_budget:.4f}")
    agent = AQuAAgent(llm=OpenAIClient(cost_tracker=tracker))

    by_condition: dict[str, list] = defaultdict(list)
    halted = False
    n_skipped = 0
    for i, (condition, run_id, tpath) in enumerate(run_paths, start=1):
        # soft budget gate: 新規 transcript の scoring 開始前にチェック
        if tracker.should_skip_new_run():
            n_skipped += 1
            typer.echo(
                f"[aqua] ({i}/{len(run_paths)}) skipped {condition}/{run_id} "
                f"(soft budget reached: {tracker.format_status()})"
            )
            halted = True
            continue
        typer.echo(f"[aqua] ({i}/{len(run_paths)}) {condition}/{run_id}")
        transcript = _load_transcript(tpath)
        try:
            report = await agent.score_discussion(
                transcript,
                n_context=n_context,
                concurrency=concurrency,
                model=model,
            )
        except BudgetExceededError as exc:
            typer.echo(f"[aqua] HARD BUDGET EXCEEDED: {exc}", err=True)
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
        "n_runs_skipped_budget": n_skipped,
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
    hard_budget: float | None = None,
    condition_concurrency: dict[str, int] | None = None,
    linking_top_k: int = 5,
    linking_top_k_per_source: int | None = None,
    linking_model: str | None = None,
) -> None:
    from das.eval import (
        ConditionFlatRAG,
        ConditionFullProposal,
        ConditionFullProposalUnlabeled,
        ConditionNone,
        JudgeAgent,
        cafeteria_personas,
        policy_ai_lecture_personas,
        run_eval,
    )
    from das.llm import BudgetExceededError, CostTracker

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

    tracker = (
        CostTracker(budget_usd=budget, hard_budget_usd=hard_budget)
        if (budget is not None or hard_budget is not None)
        else CostTracker()
    )
    if budget is not None:
        typer.echo(
            f"[eval] soft budget: ${budget:.4f} "
            f"(超過後の新規 run は skip、in-flight は完走)"
        )
    if hard_budget is not None:
        typer.echo(
            f"[eval] hard budget: ${hard_budget:.4f} "
            f"(超過したら API 呼び出しごと即停止)"
        )
    llm = OpenAIClient(cost_tracker=tracker)
    factories: dict = {}
    for name in (c.strip() for c in conditions.split(",") if c.strip()):
        if name == "none":
            factories[name] = ConditionNone
        elif name == "flat_rag":
            factories[name] = lambda llm=llm: ConditionFlatRAG(llm=llm)
        elif name == "full_proposal":
            factories[name] = lambda llm=llm, top_k=linking_top_k, top_k_ps=linking_top_k_per_source, lmodel=linking_model: (
                ConditionFullProposal(
                    llm=llm,
                    enable_web_search=web_search,
                    max_web_searches=max_web_searches,
                    top_k=top_k,
                    top_k_per_source=top_k_ps,
                    linking_model=lmodel,
                )
            )
        elif name == "full_proposal_unlabeled":
            # dose 統制 ablation: full_proposal と同一設定で関係ラベルのみ除去 (E3)
            factories[name] = lambda llm=llm, top_k=linking_top_k, top_k_ps=linking_top_k_per_source, lmodel=linking_model: (
                ConditionFullProposalUnlabeled(
                    llm=llm,
                    enable_web_search=web_search,
                    max_web_searches=max_web_searches,
                    top_k=top_k,
                    top_k_per_source=top_k_ps,
                    linking_model=lmodel,
                )
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
        f"judge={'on' if judge else 'off'} "
        f"linking_top_k={linking_top_k}"
        + (
            f" linking_top_k_per_source={linking_top_k_per_source}"
            if linking_top_k_per_source is not None
            else ""
        )
        + (f" linking_model={linking_model}" if linking_model else "")
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
            condition_concurrency=condition_concurrency,
        )
    except BudgetExceededError as exc:
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
