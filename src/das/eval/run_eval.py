"""多数回ラン executor (研究計画書 §5.2 段階B 対応)。

同じ topic + personas に対して、複数の ``Condition`` を ``n_runs`` 回ずつ
走らせ、transcript・AF・judge reports を集計する。

ファイルレイアウト (eval_dir 指定時)::

    <eval_dir>/<eval_id>/
    ├── meta.json                  # 実行設定とサマリ
    ├── summary.json               # 条件ごとの aggregated scores
    └── <condition>/
        └── run_001/
            ├── transcript.jsonl
            ├── interventions.jsonl   (full_proposal のみ)
            ├── snapshot.json         (full_proposal のみ)
            └── judge_reports.json
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from das.eval.conditions import (
    Condition,
    ConditionFullProposal,
    InterventionLogEntry,
    write_intervention_log,
)
from das.agents.stance_agent import StanceAgent, StanceMeasurement
from das.eval.citation import (
    CitationStats,
    aggregate_citation_stats,
    compute_citation_stats,
    compute_citation_stats_with_embeddings,
)
from das.eval.consensus import ConsensusReport, detect_consensus, detect_consensus_with_llm
from das.eval.controller import SessionConfig, SessionRunner
from das.eval.judge import (
    AggregatedScores,
    JudgeAgent,
    JudgeReport,
    aggregate_reports,
)
from das.eval.metrics import (
    GraphMetrics,
    TranscriptMetrics,
    graph_metrics,
    transcript_metrics,
)
from das.eval.persona import PersonaSpec
from das.eval.structural_metrics import (
    DiscussionStructuralMetrics,
    aggregate_structural_metrics,
    compute_structural_metrics,
)
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.logging import get_logger
from das.types import Utterance

ConditionFactory = Callable[[], Condition]
ProgressCallback = Callable[[str, int, int], Awaitable[None] | None]
"""``(condition_name, run_index, total_runs)`` を受け取る進捗コールバック。"""

EventEmitter = Callable[[dict], None]
"""UI 等に向けたイベント通知。``run_start`` / ``utterance`` / ``intervention`` /
``run_end`` の 4 種を発火する。同期コールバック想定。"""

_log = get_logger("das.eval.run_eval")


@dataclass(frozen=True)
class SingleRunResult:
    """1 ランの実行結果。"""

    run_id: str
    condition_name: str
    topic: str
    transcript: list[Utterance]
    transcript_metrics_: TranscriptMetrics
    graph_metrics_: GraphMetrics | None = None
    judge_reports: list[JudgeReport] = field(default_factory=list)
    intervention_log: list[InterventionLogEntry] | None = None
    snapshot: dict | None = None
    consensus: ConsensusReport | None = None
    """セッション終了時の合意検出レポート (until_consensus 有効時のみ非 None)。"""

    structural: DiscussionStructuralMetrics | None = None
    """AF + transcript から決定的に計算した構造指標 (DQI 風)。"""

    citation: CitationStats | None = None
    """提示情報の引用率 (source 別)。RQ4 の直接 evidence。"""

    stance: dict[str, dict[str, StanceMeasurement]] | None = None
    """ペルソナ別 × phase 別 (pre/post) の立場測定 (DEBATE benchmark 流)。
    形式: {persona_name: {"pre": StanceMeasurement, "post": StanceMeasurement}}
    """

    @property
    def n_turns(self) -> int:
        """実際に走ったターン数 (max_turns 未満で早期終了したかを判定可能)。"""

        return len(self.transcript)


@dataclass(frozen=True)
class EvalResult:
    """``n_runs`` x``len(conditions)`` 本の集積結果。"""

    eval_id: str
    topic: str
    personas: list[PersonaSpec]
    runs: list[SingleRunResult]
    eval_dir: Path | None = None

    def by_condition(self) -> dict[str, list[SingleRunResult]]:
        grouped: dict[str, list[SingleRunResult]] = {}
        for r in self.runs:
            grouped.setdefault(r.condition_name, []).append(r)
        return grouped

    def aggregate(self) -> dict[str, AggregatedScores]:
        """条件ごとに全ペルソナ x全ラン分の主観指標を平均する。"""

        result: dict[str, AggregatedScores] = {}
        for cond, runs in self.by_condition().items():
            reports = [rep for r in runs for rep in r.judge_reports]
            result[cond] = aggregate_reports(reports)
        return result


# --- 内部ヘルパ ----------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serialize_utterance(u: Utterance) -> dict:
    return {
        "turn_id": u.turn_id,
        "speaker": u.speaker,
        "text": u.text,
        "timestamp": u.timestamp.isoformat(),
    }


def _save_run(
    run_dir: Path,
    result: SingleRunResult,
) -> None:
    _ensure_dir(run_dir)

    transcript_path = run_dir / "transcript.jsonl"
    transcript_path.write_text(
        "\n".join(
            json.dumps(_serialize_utterance(u), ensure_ascii=False)
            for u in result.transcript
        ),
        encoding="utf-8",
    )

    if result.intervention_log:
        write_intervention_log(result.intervention_log, run_dir / "interventions.jsonl")

    if result.snapshot is not None:
        (run_dir / "snapshot.json").write_text(
            json.dumps(result.snapshot, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    if result.judge_reports:
        reports_payload = [
            {
                "persona_name": rep.persona_name,
                "condition_name": rep.condition_name,
                "topic": rep.topic,
                "scores": rep.scores.model_dump(),
            }
            for rep in result.judge_reports
        ]
        (run_dir / "judge_reports.json").write_text(
            json.dumps(reports_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ラン単位のメタ (収束情報やターン数を解析できるように残す)
    run_meta: dict = {
        "run_id": result.run_id,
        "condition_name": result.condition_name,
        "n_turns": result.n_turns,
    }
    if result.consensus is not None:
        run_meta["consensus"] = {
            "reached": result.consensus.consensus_reached,
            "signal": result.consensus.signal,
            "confidence": result.consensus.confidence,
            "fired_signals": list(result.consensus.fired_signals),
            "detected_at_turn": result.consensus.detected_at_turn,
            "rationale": result.consensus.rationale,
            "llm_judgement": result.consensus.llm_judgement,
        }
    if result.structural is not None:
        run_meta["structural_metrics"] = asdict(result.structural)
    if result.citation is not None:
        run_meta["citation"] = result.citation.to_dict()
    if result.stance:
        run_meta["stance"] = {
            persona: {
                phase: {
                    "public_stance": m.public_stance,
                    "private_stance": m.private_stance,
                    "public_reason": m.public_reason,
                    "private_reason": m.private_reason,
                    "public_private_gap": m.public_private_gap,
                }
                for phase, m in phases.items()
            }
            for persona, phases in result.stance.items()
        }
    (run_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _convergence_stats(runs: list[SingleRunResult]) -> dict:
    """ラン群を「収束したか / 何ターンで収束したか」で集計する。"""

    total = len(runs)
    if total == 0:
        return {
            "n_runs": 0,
            "n_converged": 0,
            "convergence_rate": 0.0,
            "mean_turns": 0.0,
            "mean_turns_to_consensus": None,
            "signals": {},
        }

    n_converged = 0
    sum_turns = 0
    converged_turns: list[int] = []
    signal_counts: dict[str, int] = {}

    for r in runs:
        sum_turns += r.n_turns
        if r.consensus is not None and r.consensus.consensus_reached:
            n_converged += 1
            if r.consensus.detected_at_turn is not None:
                converged_turns.append(r.consensus.detected_at_turn)
            signal_counts[r.consensus.signal] = signal_counts.get(r.consensus.signal, 0) + 1

    return {
        "n_runs": total,
        "n_converged": n_converged,
        "convergence_rate": n_converged / total,
        "mean_turns": sum_turns / total,
        "mean_turns_to_consensus": (
            sum(converged_turns) / len(converged_turns) if converged_turns else None
        ),
        "signals": signal_counts,
    }


def _aggregate_stance(stance_runs: list[dict]) -> dict:
    """ラン群の stance データから、公開/私的 × Pre/Post の平均と shift を集計。

    DEBATE benchmark 流の指標:
      - mean_public_shift: Post.public - Pre.public の平均 (議論で表明する立場の変化)
      - mean_private_shift: Post.private - Pre.private の平均 (内心の変化)
      - mean_pre_gap / mean_post_gap: |public - private| の平均 (見せかけの度合)
      - mean_post_gap が 0 に近いほど真の合意、大きいほど表層的合意の懸念
    """

    if not stance_runs:
        return {}
    pre_publics: list[int] = []
    pre_privates: list[int] = []
    post_publics: list[int] = []
    post_privates: list[int] = []
    pre_gaps: list[int] = []
    post_gaps: list[int] = []
    for stance in stance_runs:
        for _persona, phases in stance.items():
            pre = phases.get("pre")
            post = phases.get("post")
            if pre is not None:
                pre_publics.append(pre.public_stance)
                pre_privates.append(pre.private_stance)
                pre_gaps.append(abs(pre.public_stance - pre.private_stance))
            if post is not None:
                post_publics.append(post.public_stance)
                post_privates.append(post.private_stance)
                post_gaps.append(abs(post.public_stance - post.private_stance))

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    public_shift = _mean(post_publics) - _mean(pre_publics) if (pre_publics and post_publics) else 0.0
    private_shift = _mean(post_privates) - _mean(pre_privates) if (pre_privates and post_privates) else 0.0

    return {
        "n_persona_runs": len(pre_publics) if pre_publics else 0,
        "mean_pre_public": _mean(pre_publics),
        "mean_pre_private": _mean(pre_privates),
        "mean_post_public": _mean(post_publics),
        "mean_post_private": _mean(post_privates),
        "mean_public_shift": public_shift,
        "mean_private_shift": private_shift,
        "mean_pre_gap": _mean(pre_gaps),
        "mean_post_gap": _mean(post_gaps),
    }


def _save_eval_result(eval_dir: Path, result: EvalResult) -> None:
    aggregates = result.aggregate()
    grouped = result.by_condition()
    summary_payload = {
        "eval_id": result.eval_id,
        "topic": result.topic,
        "n_runs_total": len(result.runs),
        "by_condition": {
            cond: {
                "n_judge_reports": agg.n,
                "overall_satisfaction": [
                    agg.overall_satisfaction_mean,
                    agg.overall_satisfaction_std,
                ],
                "information_usefulness": [
                    agg.information_usefulness_mean,
                    agg.information_usefulness_std,
                ],
                "opposition_understanding": [
                    agg.opposition_understanding_mean,
                    agg.opposition_understanding_std,
                ],
                "confidence_change": [
                    agg.confidence_change_mean,
                    agg.confidence_change_std,
                ],
                "intervention_transparency": [
                    agg.intervention_transparency_mean,
                    agg.intervention_transparency_std,
                ],
                "convergence": _convergence_stats(grouped.get(cond, [])),
                "structural": aggregate_structural_metrics(
                    [r.structural for r in grouped.get(cond, []) if r.structural]
                ),
                "citation": aggregate_citation_stats(
                    [r.citation for r in grouped.get(cond, []) if r.citation]
                ),
                "stance": _aggregate_stance(
                    [r.stance for r in grouped.get(cond, []) if r.stance]
                ),
            }
            for cond, agg in aggregates.items()
        },
    }
    (eval_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --- main ----------------------------------------------------------


def _store_for_condition(condition: Condition) -> GraphStore | None:
    """ConditionFullProposal なら orchestrator の store を返す。それ以外は None。"""

    if isinstance(condition, ConditionFullProposal) and condition.orchestrator:
        return condition.orchestrator.store
    return None


async def _run_single(
    *,
    condition_name: str,
    condition: Condition,
    topic: str,
    personas: list[PersonaSpec],
    config: SessionConfig,
    docs_dir: Path | None,
    llm: OpenAIClient,
    judge: JudgeAgent | None,
    run_id: str,
    run_idx: int = 1,
    until_consensus: bool = False,
    consensus_kwargs: dict | None = None,
    event_emitter: EventEmitter | None = None,
    consensus_agent: object | None = None,
    consensus_min_interval: int = 4,
    stance_agent: StanceAgent | None = None,
) -> SingleRunResult:
    await condition.setup(docs_dir=docs_dir)

    consensus_kwargs = consensus_kwargs or {}

    def _emit(payload: dict) -> None:
        if event_emitter is None:
            return
        with contextlib.suppress(Exception):  # 防御的: emitter 失敗で本筋を止めない
            event_emitter(payload)

    _emit(
        {
            "type": "run_start",
            "condition": condition_name,
            "run_idx": run_idx,
            "run_id": run_id,
            "topic": topic,
            "personas": [
                {"name": p.name, "stance": p.stance, "focus": p.focus} for p in personas
            ],
        }
    )

    # Pre-discussion stance polling (DEBATE benchmark 流)
    stance_data: dict[str, dict[str, StanceMeasurement]] = {}
    if stance_agent is not None:
        for p in personas:
            try:
                pre = await stance_agent.measure(
                    persona=p, topic=topic, phase="pre"
                )
                stance_data[p.name] = {"pre": pre}
            except Exception as exc:  # pragma: no cover - 防御的
                _log.warning("stance.pre.failed", persona=p.name, error=str(exc))

    # until_consensus=True のときのみ stop_condition を組み立てる。
    # FullProposal なら orchestrator.store を渡して構造シグナルも使う。
    stop_condition = None
    if until_consensus:
        store_ref = _store_for_condition(condition)
        last_llm_check_at: dict[str, int] = {"turn": -1}
        last_report: dict[str, ConsensusReport | None] = {"r": None}

        if consensus_agent is None:
            # キーワード+構造シグナルだけの安価判定
            def _stop_sync(history: list[Utterance]) -> bool:
                return detect_consensus(
                    history, store=store_ref, **consensus_kwargs
                ).consensus_reached

            stop_condition = _stop_sync
        else:

            async def _stop_async(history: list[Utterance]) -> bool:
                # 安価な前段でシグナル無しなら LLM を呼ばずに False
                cheap = detect_consensus(history, store=store_ref, **consensus_kwargs)
                structural = any(
                    s in cheap.fired_signals
                    for s in ("new_claim_stalled", "no_new_attacks")
                )
                if not (cheap.consensus_reached or structural):
                    return False
                # 直近 LLM 呼び出しから min_interval ターン経っていなければスキップ
                turn = len(history)
                if turn - last_llm_check_at["turn"] < consensus_min_interval:
                    prev = last_report["r"]
                    return bool(prev and prev.consensus_reached)
                last_llm_check_at["turn"] = turn
                report = await detect_consensus_with_llm(
                    history,
                    topic=topic,
                    personas=personas,
                    agent=consensus_agent,
                    store=store_ref,
                    **consensus_kwargs,
                )
                last_report["r"] = report
                return report.consensus_reached

            stop_condition = _stop_async  # type: ignore[assignment]

    transcript: list[Utterance] = []
    prev_iv_count = 0
    runner = SessionRunner(personas, config, llm=llm)
    async for u in runner.run_streaming(
        info_provider=condition.info_provider, stop_condition=stop_condition
    ):
        # この発話の前に出された介入を先に通知 (タイムライン上は intervention → utterance の順)
        if isinstance(condition, ConditionFullProposal):
            new_log = condition.intervention_log
            for entry in new_log[prev_iv_count:]:
                _emit(
                    {
                        "type": "intervention",
                        "condition": condition_name,
                        "run_idx": run_idx,
                        "turn_id": entry.turn_id,
                        "kind": entry.kind,
                        "addressed_to": entry.addressed_to,
                        "brief": entry.brief,
                        "decision_reason": entry.decision_reason,
                        "items": entry.items,
                    }
                )
            prev_iv_count = len(new_log)

        transcript.append(u)
        _emit(
            {
                "type": "utterance",
                "condition": condition_name,
                "run_idx": run_idx,
                "turn_id": u.turn_id,
                "speaker": u.speaker,
                "text": u.text,
            }
        )

    t_metrics = transcript_metrics(transcript)

    g_metrics: GraphMetrics | None = None
    snapshot: dict | None = None
    intervention_log: list[InterventionLogEntry] | None = None
    final_store = _store_for_condition(condition)
    if final_store is not None:
        g_metrics = graph_metrics(final_store)
        snapshot = final_store.snapshot()
        if isinstance(condition, ConditionFullProposal):
            intervention_log = condition.intervention_log
    # flat_rag も intervention_log を残せるようになったので、citation 計算で使う。
    # (FullProposal でないので structural / snapshot は計算しない)
    elif hasattr(condition, "intervention_log"):
        log_attr = getattr(condition, "intervention_log", None)
        if log_attr:
            intervention_log = log_attr

    # 合意検出の最終レポート (until_consensus でない場合も「実際に合意していたか」を
    # 後付け判定して残すと分析しやすいので、常に算出する)。
    if consensus_agent is not None:
        consensus_report = await detect_consensus_with_llm(
            transcript,
            topic=topic,
            personas=personas,
            agent=consensus_agent,
            store=final_store,
            **consensus_kwargs,
        )
    else:
        consensus_report = detect_consensus(
            transcript, store=final_store, **consensus_kwargs
        )

    _emit(
        {
            "type": "run_end",
            "condition": condition_name,
            "run_idx": run_idx,
            "n_turns": len(transcript),
            "consensus_reached": consensus_report.consensus_reached,
            "consensus_signal": consensus_report.signal,
            "consensus_at": consensus_report.detected_at_turn,
        }
    )

    judge_reports: list[JudgeReport] = []
    if judge is not None:
        judge_reports = await judge.evaluate_session(
            personas,
            topic,
            transcript,
            condition_name=condition_name,
            info_log=intervention_log,
        )

    structural = compute_structural_metrics(transcript, final_store)

    # Post-discussion stance polling
    if stance_agent is not None:
        for p in personas:
            try:
                post = await stance_agent.measure(
                    persona=p, topic=topic, phase="post", transcript=transcript
                )
                if p.name not in stance_data:
                    stance_data[p.name] = {}
                stance_data[p.name]["post"] = post
            except Exception as exc:  # pragma: no cover - 防御的
                _log.warning("stance.post.failed", persona=p.name, error=str(exc))

    # 引用率: 介入で提示された情報が次発話で使われた率 (RQ4 直接指標)
    interventions_for_citation = (
        [
            {
                "kind": e.kind,
                "turn_id": e.turn_id,
                "addressed_to": e.addressed_to,
                "items": [
                    {
                        "source_text": it.get("source_text", ""),
                        "source_kind": it.get("source_kind", "unknown"),
                    }
                    for it in e.items
                ],
            }
            for e in intervention_log
        ]
        if intervention_log
        else []
    )
    # n-gram 一致 + embedding 類似度の両方で判定 (言い換え引用も捕捉)
    try:
        citation = await compute_citation_stats_with_embeddings(
            transcript, interventions_for_citation, embedder=llm
        )
    except Exception as exc:  # pragma: no cover - 防御的: embedding 失敗時は n-gram のみ
        _log.warning("citation.embedding_failed", error=str(exc))
        citation = compute_citation_stats(transcript, interventions_for_citation)

    return SingleRunResult(
        run_id=run_id,
        condition_name=condition_name,
        topic=topic,
        transcript=transcript,
        transcript_metrics_=t_metrics,
        graph_metrics_=g_metrics,
        judge_reports=judge_reports,
        intervention_log=intervention_log,
        snapshot=snapshot,
        consensus=consensus_report,
        structural=structural,
        citation=citation,
        stance=stance_data if stance_data else None,
    )


async def run_eval(
    topic: str,
    personas: list[PersonaSpec],
    condition_factories: dict[str, ConditionFactory],
    *,
    n_runs: int = 1,
    max_turns: int = 6,
    temperature: float = 0.7,
    docs_dir: Path | None = None,
    llm: OpenAIClient | None = None,
    judge: JudgeAgent | None = None,
    eval_dir: Path | None = None,
    eval_id: str | None = None,
    progress: ProgressCallback | None = None,
    until_consensus: bool = False,
    consensus_kwargs: dict | None = None,
    concurrency: int = 1,
    event_emitter: EventEmitter | None = None,
    consensus_agent: object | None = None,
    consensus_min_interval: int = 4,
    stance_agent: StanceAgent | None = None,
) -> EvalResult:
    """N 本のランを条件 xトピックで回し、集計結果を返す。

    ``until_consensus=True`` のとき、各セッションは ``detect_consensus`` が True を
    返した時点で早期終了する。``max_turns`` はその場合「安全上限」として機能する。
    ``consensus_kwargs`` は ``detect_consensus`` にそのまま渡されるため、
    ``agreement_window`` などのチューニングが可能。

    ``concurrency`` (>=1) を上げると、(condition, run) 単位の独立タスクを
    並列実行する。LLM 呼び出しは asyncio で多重化されるが、API のレート制限を
    超えないよう適切な値 (例: 4〜8) に抑えることを推奨。
    """

    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if not condition_factories:
        raise ValueError("condition_factories must not be empty")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    llm = llm or OpenAIClient()
    eval_id = eval_id or datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")
    target_dir = (eval_dir / eval_id) if eval_dir is not None else None
    if target_dir is not None:
        _ensure_dir(target_dir)

    config = SessionConfig(
        topic=topic, max_turns=max_turns, temperature=temperature
    )

    total = n_runs * len(condition_factories)
    semaphore = asyncio.Semaphore(concurrency)
    completed_counter = {"n": 0}

    async def _job(
        cond_name: str, factory: ConditionFactory, run_idx: int
    ) -> tuple[str, int, SingleRunResult]:
        async with semaphore:
            condition = factory()
            run_id = f"{cond_name}-run-{run_idx:03d}-{uuid4().hex[:6]}"
            _log.info("eval.run.start", run_id=run_id, condition=cond_name)
            result = await _run_single(
                condition_name=cond_name,
                condition=condition,
                topic=topic,
                personas=personas,
                config=config,
                docs_dir=docs_dir,
                llm=llm,
                judge=judge,
                run_id=run_id,
                run_idx=run_idx,
                until_consensus=until_consensus,
                consensus_kwargs=consensus_kwargs,
                event_emitter=event_emitter,
                consensus_agent=consensus_agent,
                consensus_min_interval=consensus_min_interval,
                stance_agent=stance_agent,
            )
            if target_dir is not None:
                run_dir = target_dir / cond_name / f"run_{run_idx:03d}"
                _save_run(run_dir, result)
            # asyncio は単一スレッドなので排他制御不要
            completed_counter["n"] += 1
            if progress is not None:
                ret = progress(cond_name, completed_counter["n"], total)
                if hasattr(ret, "__await__"):
                    await ret  # type: ignore[func-returns-value]
            return cond_name, run_idx, result

    tasks: list[asyncio.Task] = []
    cond_order = list(condition_factories.keys())
    for cond_name in cond_order:
        factory = condition_factories[cond_name]
        for run_idx in range(1, n_runs + 1):
            tasks.append(asyncio.create_task(_job(cond_name, factory, run_idx)))

    completed_results = await asyncio.gather(*tasks)
    # condition の順、run_idx の順に並べ直して再現性のある runs リストにする
    completed_results.sort(key=lambda triple: (cond_order.index(triple[0]), triple[1]))
    runs: list[SingleRunResult] = [t[2] for t in completed_results]

    eval_result = EvalResult(
        eval_id=eval_id,
        topic=topic,
        personas=personas,
        runs=runs,
        eval_dir=target_dir,
    )

    if target_dir is not None:
        # メタ情報
        meta = {
            "eval_id": eval_id,
            "topic": topic,
            "n_runs_per_condition": n_runs,
            "max_turns": max_turns,
            "temperature": temperature,
            "until_consensus": until_consensus,
            "consensus_kwargs": consensus_kwargs or {},
            "concurrency": concurrency,
            "personas": [asdict(p) for p in personas],
            "condition_names": list(condition_factories.keys()),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        (target_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        _save_eval_result(target_dir, eval_result)

    return eval_result


__all__ = [
    "ConditionFactory",
    "EvalResult",
    "ProgressCallback",
    "SingleRunResult",
    "run_eval",
]
