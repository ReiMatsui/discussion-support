"""保存済み eval ディレクトリの再採点 (rescore-everything, レビュー 03章 根本再設計)。

``das eval`` が生成した run ディレクトリ (transcript / 介入ログ / AF snapshot /
生 stance) を読み込み、会話を再生成せずに judge・構造指標・citation・consensus を
**再計算** して judge_reports.json / run_meta.json / summary.json を作り直す。

実行と採点の分離により:
  - judge プロンプトや citation 閾値を直したあと、API 再ランなしで再採点できる
  - 過去ランの再採点ができる (以前は不可能だった)

採点の本体は :func:`das.eval.run_eval.score_run` に一元化されており、実行時
(``_run_single``) と rescore はまったく同じ採点コードを通る。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from das.agents.stance_agent import StanceMeasurement
from das.eval.conditions import InterventionLogEntry
from das.eval.judge import JudgeAgent
from das.eval.metrics import graph_metrics, transcript_metrics
from das.eval.persona import PersonaSpec
from das.eval.run_eval import (
    EvalResult,
    SingleRunResult,
    _save_eval_result,
    _save_run_scores,
    score_run,
)
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.logging import get_logger
from das.types import Utterance

_log = get_logger("das.eval.rescore")


class RescoreError(Exception):
    """旧形式・破損 run を検出したときに送出する (黙って誤計算しないため)。"""


# --- ローダ (生データ → メモリ) --------------------------------------


def load_eval_meta(eval_dir: Path) -> dict[str, Any]:
    """eval_dir/meta.json を読み、rescore に必要なフィールドを検証する。

    旧形式 (personas / topic を持たない meta) は ``RescoreError`` にする。
    """

    meta_path = eval_dir / "meta.json"
    if not meta_path.exists():
        raise RescoreError(
            f"{meta_path} が見つかりません。この eval ディレクトリは rescore に必要な "
            "メタ情報 (personas / topic / consensus_kwargs) を持たない旧形式です。"
        )
    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [k for k in ("topic", "personas") if not meta.get(k)]
    if missing:
        raise RescoreError(
            f"{meta_path} に必須フィールド {missing} がありません "
            "(rescore は persona 定義と topic を保存済みメタから復元する必要があります)。"
            " この run は再採点できません。"
        )
    return meta


def personas_from_meta(meta: dict[str, Any]) -> list[PersonaSpec]:
    """meta.json の personas (asdict 形式) から ``PersonaSpec`` を復元する。"""

    personas: list[PersonaSpec] = []
    for raw in meta["personas"]:
        personas.append(
            PersonaSpec(
                name=raw["name"],
                stance=raw["stance"],
                focus=raw.get("focus", ""),
                personality=raw.get("personality", "落ち着いて論理的"),
                extra=raw.get("extra", ""),
                metadata=raw.get("metadata", {}) or {},
            )
        )
    return personas


def load_transcript(path: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterances.append(Utterance.model_validate(json.loads(line)))
    return utterances


def load_intervention_log(path: Path) -> list[InterventionLogEntry]:
    entries: list[InterventionLogEntry] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                InterventionLogEntry(
                    turn_id=data["turn_id"],
                    persona_name=data["persona_name"],
                    timestamp=data.get("timestamp", ""),
                    items=data.get("items", []),
                    kind=data.get("kind", "l1"),
                    addressed_to=data.get("addressed_to"),
                    brief=data.get("brief", ""),
                    decision_reason=data.get("decision_reason", ""),
                )
            )
    return entries


def load_store(snapshot_path: Path) -> GraphStore:
    """snapshot.json を GraphStore に復元する。"""

    from das.viz.render import load_snapshot

    return load_snapshot(snapshot_path)


def load_stance(run_meta: dict[str, Any]) -> dict[str, dict[str, StanceMeasurement]] | None:
    """run_meta.json の生 stance 測定を ``StanceMeasurement`` に復元する。

    stance は会話前後に測定された生データで rescore では再計算しない
    (集計 = paired diff のみ summary 段でやり直す)。
    """

    raw = run_meta.get("stance")
    if not raw:
        return None
    result: dict[str, dict[str, StanceMeasurement]] = {}
    for persona, phases in raw.items():
        result[persona] = {}
        for phase, m in phases.items():
            result[persona][phase] = StanceMeasurement(
                public_stance=m["public_stance"],
                private_stance=m["private_stance"],
                public_reason=m.get("public_reason", ""),
                private_reason=m.get("private_reason", ""),
            )
    return result


# --- 観測用 AF の後付け構築 (E4) ------------------------------------


async def build_observation_af(
    transcript: list[Utterance], llm: OpenAIClient
) -> GraphStore:
    """transcript から extraction+linking を走らせて **観測用 AF** を構築する。

    介入 (処置) には一切使わず、構造指標・合意の構造シグナルを **全条件同一の
    パイプライン** で計算するためのグラフ (レビュー H-1 / H-2)。none / flat_rag でも
    このグラフができるので、構造指標が全条件で比較可能になる。

    文書は投入しない (transcript のみ) — 対面でも録音から同じ意味で作れる設計。
    API コストが増えるため実行時ではなく rescore フェーズで走らせる。
    """

    from das.runtime import Orchestrator

    orch = Orchestrator.assemble(llm=llm)
    for utterance in transcript:
        await orch.bus.publish(utterance)
    await orch.bus.drain()
    return orch.store


# --- 1 ラン分の rescore ---------------------------------------------


async def rescore_run(
    run_dir: Path,
    *,
    condition_name: str,
    topic: str,
    personas: list[PersonaSpec],
    consensus_kwargs: dict[str, Any] | None,
    llm: OpenAIClient,
    judge: JudgeAgent | None,
    build_observation: bool = True,
    observation_store: GraphStore | None = None,
) -> SingleRunResult:
    """1 run ディレクトリを再採点し、採点出力を書き戻して結果を返す。

    ``build_observation`` (E4): True なら transcript から観測用 AF を後付け構築し、
    構造指標・合意の構造シグナルを全条件同一にそのグラフから計算する。テストや
    「構造指標を触らず judge だけ差し替えたい」ケースでは False にできる。
    ``observation_store`` を直接渡すとその構築をスキップして再利用する
    (テスト用の注入口)。
    """

    transcript_path = run_dir / "transcript.jsonl"
    if not transcript_path.exists():
        raise RescoreError(
            f"{transcript_path} が見つかりません。この run は transcript を保存して "
            "おらず (旧形式)、再採点できません。"
        )
    transcript = load_transcript(transcript_path)

    intervention_log: list[InterventionLogEntry] | None = None
    iv_path = run_dir / "interventions.jsonl"
    if iv_path.exists():
        intervention_log = load_intervention_log(iv_path)

    store: GraphStore | None = None
    snap_path = run_dir / "snapshot.json"
    if snap_path.exists():
        store = load_store(snap_path)

    # 生 stance は既存 run_meta から読む (再計算しない)。
    stance = None
    run_meta_path = run_dir / "run_meta.json"
    if run_meta_path.exists():
        stance = load_stance(json.loads(run_meta_path.read_text(encoding="utf-8")))

    # 観測用 AF: 明示注入 > 構築フラグ > なし
    obs_store = observation_store
    if obs_store is None and build_observation:
        obs_store = await build_observation_af(transcript, llm)

    scores = await score_run(
        transcript=transcript,
        condition_name=condition_name,
        topic=topic,
        personas=personas,
        intervention_log=intervention_log,
        store=store,
        llm=llm,
        judge=judge,
        consensus_agent=None,  # rescore は決定的なキーワード+構造合意判定を使う
        consensus_kwargs=consensus_kwargs,
        observation_store=obs_store,
    )

    result = SingleRunResult(
        run_id=run_dir.name,
        condition_name=condition_name,
        topic=topic,
        transcript=transcript,
        transcript_metrics_=transcript_metrics(transcript),
        graph_metrics_=graph_metrics(store) if store is not None else None,
        judge_reports=scores.judge_reports,
        intervention_log=intervention_log,
        snapshot=store.snapshot() if store is not None else None,
        consensus=scores.consensus,
        structural=scores.structural,
        structural_intervention=scores.structural_intervention,
        citation=scores.citation,
        stance=stance,
    )
    # 採点出力のみ書き戻す (生データには触れない)
    _save_run_scores(run_dir, result)
    return result


# --- eval_dir 全体の rescore ----------------------------------------


async def rescore_eval_dir(
    eval_dir: Path,
    *,
    llm: OpenAIClient,
    judge: JudgeAgent | None,
    build_observation: bool = True,
) -> EvalResult:
    """eval_dir 配下の全 run を再採点し、summary.json を作り直す。

    ``aqua-rescore`` と同じく condition/run_* をスキャンする。旧形式や破損 run は
    ``RescoreError`` を送出して停止する (黙って誤計算しない)。

    ``build_observation`` (E4): True なら全条件で観測用 AF を後付け構築して構造
    指標を統一する (API コスト増)。構造指標を触らず判定だけ再計算したいときは False。
    """

    meta = load_eval_meta(eval_dir)
    topic = meta["topic"]
    personas = personas_from_meta(meta)
    consensus_kwargs = meta.get("consensus_kwargs") or {}

    runs: list[SingleRunResult] = []
    n_scored = 0
    for cond_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            if not (run_dir / "transcript.jsonl").exists():
                # transcript の無いディレクトリは run ではない (スキップ)
                continue
            _log.info("rescore.run", condition=cond_dir.name, run=run_dir.name)
            result = await rescore_run(
                run_dir,
                condition_name=cond_dir.name,
                topic=topic,
                personas=personas,
                consensus_kwargs=consensus_kwargs,
                llm=llm,
                judge=judge,
                build_observation=build_observation,
            )
            runs.append(result)
            n_scored += 1

    if n_scored == 0:
        raise RescoreError(
            f"{eval_dir} 配下に transcript.jsonl を持つ run が見つかりませんでした。"
        )

    eval_result = EvalResult(
        eval_id=meta.get("eval_id", eval_dir.name),
        topic=topic,
        personas=personas,
        runs=runs,
        eval_dir=eval_dir,
    )
    _save_eval_result(eval_dir, eval_result)
    return eval_result


__all__ = [
    "RescoreError",
    "build_observation_af",
    "load_eval_meta",
    "personas_from_meta",
    "rescore_eval_dir",
    "rescore_run",
]
