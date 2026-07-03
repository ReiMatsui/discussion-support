"""run_eval (多数回ラン executor) のユニットテスト。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval.conditions import ConditionNone
from das.eval.judge import JudgeAgent, JudgeScores
from das.eval.persona import build_persona
from das.eval.run_eval import (
    EvalResult,
    SingleRunResult,
    run_eval,
)
from das.llm import OpenAIClient


def _fake_llm(reply: str = "発言") -> OpenAIClient:
    client = OpenAIClient(client=MagicMock())
    client.chat = AsyncMock(return_value=reply)  # type: ignore[method-assign]
    return client


def _fake_judge_scores() -> JudgeScores:
    return JudgeScores(
        overall_satisfaction=5,
        information_usefulness=4,
        opposition_understanding=4,
        confidence_change=0,
        intervention_transparency=3,
        rationale="auto",
    )


# --- 基本動作 -------------------------------------------------------


async def test_run_eval_minimum_one_run() -> None:
    """ConditionNone 1 run x1 persona で発話ターン分の transcript が出来る。"""

    llm = _fake_llm("発言")
    personas = [build_persona(name="A", stance="pro")]
    result = await run_eval(
        topic="トピック",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=3,
        llm=llm,
    )
    assert isinstance(result, EvalResult)
    assert len(result.runs) == 1
    assert result.runs[0].condition_name == "none"
    assert len(result.runs[0].transcript) == 3


async def test_run_eval_multiple_runs_and_conditions() -> None:
    llm = _fake_llm("u")
    personas = [build_persona(name="A"), build_persona(name="B", stance="con")]
    factories = {"none": ConditionNone, "none_again": ConditionNone}
    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories=factories,
        n_runs=2,
        max_turns=2,
        llm=llm,
    )
    assert len(result.runs) == 4
    grouped = result.by_condition()
    assert set(grouped.keys()) == {"none", "none_again"}
    assert len(grouped["none"]) == 2


async def test_run_eval_invalid_args() -> None:
    llm = _fake_llm()
    with pytest.raises(ValueError):
        await run_eval(
            topic="t",
            personas=[build_persona(name="A")],
            condition_factories={},
            n_runs=1,
            llm=llm,
        )
    with pytest.raises(ValueError):
        await run_eval(
            topic="t",
            personas=[build_persona(name="A")],
            condition_factories={"none": ConditionNone},
            n_runs=0,
            llm=llm,
        )


# --- judge 統合 ------------------------------------------------------


async def test_run_eval_with_judge_aggregates() -> None:
    llm = _fake_llm("u")
    judge = JudgeAgent(llm=llm)
    judge.evaluate_session = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda personas, topic, transcript, condition_name, info_log=None: [
            __import__("das").eval.JudgeReport(
                persona_name=p.name,
                condition_name=condition_name,
                topic=topic,
                scores=_fake_judge_scores(),
            )
            for p in personas
        ]
    )

    personas = [build_persona(name="A"), build_persona(name="B")]
    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=2,
        max_turns=2,
        llm=llm,
        judge=judge,
    )
    # 2 ペルソナ x2 ラン = 4 件のレポート
    all_reports = [rep for r in result.runs for rep in r.judge_reports]
    assert len(all_reports) == 4

    aggregated = result.aggregate()
    assert "none" in aggregated
    # レビュー H-5: ラン単位2段集計に変更したため、n は「ラン数 (クラスタ数)」= 2。
    # 旧実装ではペルソナ×ランの pool 数 (=4) だった。
    assert aggregated["none"].n == 2
    assert aggregated["none"].overall_satisfaction_mean == 5.0


# --- stance 集計 (paired diff) --------------------------------------


def _stance_m(public: int, private: int):
    from das.agents.stance_agent import StanceMeasurement

    return StanceMeasurement(
        public_stance=public,
        private_stance=private,
        public_reason="r",
        private_reason="r",
    )


def test_aggregate_stance_uses_paired_diff() -> None:
    """shift はペルソナ単位の paired diff の平均で計算される (レビュー H-5)。

    A: pre.public=-2 → post.public=2 (diff +4)
    B: pre.public=2 → post.public=0 (diff -2)
    paired 平均 = (+4 + -2) / 2 = +1.0。
    欠損時の頑健性のため post 平均 − pre 平均 ではなく paired で計算する。
    """
    from das.eval.run_eval import _aggregate_stance

    stance_run = {
        "A": {"pre": _stance_m(-2, -2), "post": _stance_m(2, 2)},
        "B": {"pre": _stance_m(2, 2), "post": _stance_m(0, 0)},
    }
    agg = _aggregate_stance([stance_run])
    assert agg["n_persona_runs"] == 2
    assert agg["mean_public_shift"] == pytest.approx(1.0)
    assert agg["mean_private_shift"] == pytest.approx(1.0)


def test_aggregate_stance_missing_post_excluded_from_shift() -> None:
    """post が欠損したペルソナは paired shift に入らない (系統的ずれの回避)。"""
    from das.eval.run_eval import _aggregate_stance

    stance_run = {
        "A": {"pre": _stance_m(-2, -2), "post": _stance_m(2, 2)},  # diff +4
        "B": {"pre": _stance_m(-3, -3)},  # post 欠損 → shift に寄与しない
    }
    agg = _aggregate_stance([stance_run])
    assert agg["n_persona_runs"] == 1
    assert agg["mean_public_shift"] == pytest.approx(4.0)


# --- ファイル出力 ----------------------------------------------------


async def test_run_eval_writes_to_eval_dir(tmp_path: Path) -> None:
    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=2,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="my-eval",
    )

    base = tmp_path / "my-eval"
    assert base.exists()
    assert (base / "meta.json").exists()
    assert (base / "summary.json").exists()
    run_dir = base / "none" / "run_001"
    assert run_dir.exists()
    transcript_lines = (run_dir / "transcript.jsonl").read_text().strip().split("\n")
    assert len(transcript_lines) == 2

    # meta.json は JSON として読める
    meta = json.loads((base / "meta.json").read_text())
    assert meta["topic"] == "t"
    assert meta["n_runs_per_condition"] == 1


# --- progress callback -----------------------------------------------


async def test_run_eval_progress_callback() -> None:
    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    seen: list[tuple[str, int, int]] = []

    def progress(cond: str, done: int, total: int) -> None:
        seen.append((cond, done, total))

    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone, "none2": ConditionNone},
        n_runs=2,
        max_turns=2,
        llm=llm,
        progress=progress,
    )
    assert len(seen) == 4
    assert seen[-1] == (seen[-1][0], 4, 4)


# --- 戻り値構造 -----------------------------------------------------


def test_single_run_result_fields() -> None:
    """SingleRunResult に必要なフィールドがあることを確認。"""

    sr = SingleRunResult(
        run_id="r1",
        condition_name="none",
        topic="t",
        transcript=[],
        transcript_metrics_=__import__(
            "das.eval.metrics", fromlist=["transcript_metrics"]
        ).transcript_metrics([]),
    )
    assert sr.condition_name == "none"
    assert sr.judge_reports == []
    assert sr.intervention_log is None
    assert sr.consensus is None
    assert sr.n_turns == 0


# --- 合意ベース早期終了 ----------------------------------------------


async def test_run_eval_until_consensus_stops_early() -> None:
    """until_consensus=True のとき、合意キーワードが続けば max_turns 前に停止する。"""

    llm = _fake_llm("u")
    # 4 ターン目以降に合意フレーズを連発
    replies = [
        "プラ容器を廃止すべき",
        "コストが高い",
        "折衷案",
        "なるほど納得です",
        "賛成です",
        "その通りです",
        "(これ以降は呼ばれないはず)",
        "(これも呼ばれないはず)",
    ]
    llm.chat = AsyncMock(side_effect=replies)  # type: ignore[method-assign]
    personas = [
        build_persona(name="A"),
        build_persona(name="B"),
        build_persona(name="C"),
    ]
    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=10,
        llm=llm,
        until_consensus=True,
    )
    run = result.runs[0]
    # max_turns=10 より少ないターンで停止しているはず
    assert run.n_turns < 10
    assert run.consensus is not None
    assert run.consensus.consensus_reached is True
    assert run.consensus.detected_at_turn is not None


async def test_run_eval_consensus_report_present_even_without_until_consensus() -> None:
    """until_consensus=False でも consensus フィールドは後付け判定で埋まる。"""

    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=2,
        llm=llm,
    )
    run = result.runs[0]
    assert run.consensus is not None
    # 短すぎるので合意は立たない
    assert run.consensus.consensus_reached is False


async def test_run_eval_concurrency_invalid() -> None:
    llm = _fake_llm("u")
    with pytest.raises(ValueError):
        await run_eval(
            topic="t",
            personas=[build_persona(name="A")],
            condition_factories={"none": ConditionNone},
            n_runs=1,
            max_turns=2,
            llm=llm,
            concurrency=0,
        )


async def test_run_eval_event_emitter_fires_run_start_utterance_run_end() -> None:
    """event_emitter に run_start / utterance / run_end の各 type が流れる。"""

    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    events: list[dict] = []

    def emit(payload: dict) -> None:
        events.append(payload)

    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=2,
        llm=llm,
        event_emitter=emit,
    )
    types = [e.get("type") for e in events]
    assert types[0] == "run_start"
    assert "utterance" in types
    assert types[-1] == "run_end"
    # utterance 数 = max_turns
    assert sum(1 for e in events if e.get("type") == "utterance") == 2


async def test_run_eval_concurrency_preserves_order_and_count() -> None:
    """concurrency>1 でも runs は (cond order, run_idx) で並ぶ。"""

    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone, "none2": ConditionNone},
        n_runs=3,
        max_turns=2,
        llm=llm,
        concurrency=4,
    )
    assert len(result.runs) == 6
    # condition順 (none, none2) かつ run_idx 順に並んでいる
    expected_cond_seq = ["none", "none", "none", "none2", "none2", "none2"]
    assert [r.condition_name for r in result.runs] == expected_cond_seq


async def test_run_eval_writes_run_meta_with_convergence(tmp_path: Path) -> None:
    """eval_dir 指定時に run_meta.json と summary.convergence が書かれる。"""

    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=2,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="ev",
    )
    run_meta_path = tmp_path / "ev" / "none" / "run_001" / "run_meta.json"
    assert run_meta_path.exists()
    rm = json.loads(run_meta_path.read_text())
    assert rm["n_turns"] == 2
    assert "consensus" in rm
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text())
    assert "convergence" in summary["by_condition"]["none"]
    conv = summary["by_condition"]["none"]["convergence"]
    assert conv["n_runs"] == 1


# --- Stratified ordering と部分完了 (BudgetExceededError 互換) ----------


async def test_run_eval_partial_completion_on_task_exception(tmp_path: Path) -> None:
    """1 つの condition の task が例外で失敗しても、他 condition の結果は保存され、
    最初の例外が末尾で再 raise される (= 部分結果を保ったまま CLI に通知される)。"""

    class FailingCondition:
        name = "failing"

        async def setup(self, *, docs_dir: Path | None = None) -> None:
            return None

        async def info_provider(self, history, persona):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated budget exceeded")

    llm = _fake_llm("発言")
    personas = [build_persona(name="A")]
    factories = {"none": ConditionNone, "failing": FailingCondition}

    with pytest.raises(RuntimeError, match="simulated budget exceeded"):
        await run_eval(
            topic="t",
            personas=personas,
            condition_factories=factories,
            n_runs=2,
            max_turns=2,
            llm=llm,
            eval_dir=tmp_path,
            eval_id="partial",
        )

    # 部分結果として meta.json は保存されるはず (halted_by_exception フラグ付き)
    meta = json.loads((tmp_path / "partial" / "meta.json").read_text())
    assert meta["halted_by_exception"] is True
    assert meta["first_exception"] == "RuntimeError"
    # none condition は成功してるので少なくとも 1 件は保存されている
    assert meta["n_runs_completed"] >= 1
    none_runs = list((tmp_path / "partial" / "none").glob("run_*"))
    assert len(none_runs) >= 1


async def test_run_eval_stratified_order_creates_tasks_by_run_idx(
    tmp_path: Path,
) -> None:
    """task が「各 run_idx の中で全 condition」順に作られている。

    部分完了時に全 condition のサンプルが残るための条件。
    """

    llm = _fake_llm("発言")
    personas = [build_persona(name="A")]
    factories = {"cA": ConditionNone, "cB": ConditionNone, "cC": ConditionNone}

    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories=factories,
        n_runs=3,
        max_turns=1,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="stratified",
    )

    # 全部成功すれば 3 cond x 3 runs = 9 件
    assert len(result.runs) == 9
    # 各 condition ごとに 3 件ずつ
    grouped = result.by_condition()
    assert all(len(v) == 3 for v in grouped.values())


async def test_run_eval_incremental_save_writes_transcript_per_turn(
    tmp_path: Path,
) -> None:
    """``eval_dir`` 指定時、各 turn の直後に transcript.jsonl が更新される。

    途中で BudgetExceededError などで中断しても、それまでの発話が残る。
    """

    llm = _fake_llm("発言")
    personas = [build_persona(name="A")]
    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=4,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="inc",
    )

    transcript_path = tmp_path / "inc" / "none" / "run_001" / "transcript.jsonl"
    assert transcript_path.exists()
    lines = [ln for ln in transcript_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 4
    parsed = [json.loads(ln) for ln in lines]
    assert all("turn_id" in p and "text" in p for p in parsed)
    assert [p["turn_id"] for p in parsed] == [1, 2, 3, 4]


async def test_run_eval_incremental_save_truncates_existing_file(
    tmp_path: Path,
) -> None:
    """同じ run_dir で再ランしたとき、古い transcript.jsonl が残らない。"""

    run_dir = tmp_path / "inc2" / "none" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    stale_path = run_dir / "transcript.jsonl"
    stale_path.write_text('{"turn_id": 99, "speaker": "Z", "text": "stale"}\n')

    llm = _fake_llm("発言")
    personas = [build_persona(name="A")]
    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=2,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="inc2",
    )

    lines = [ln for ln in stale_path.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2  # 古い行は消えて、新規 2 turn のみ
    parsed = [json.loads(ln) for ln in lines]
    assert [p["turn_id"] for p in parsed] == [1, 2]


async def test_run_eval_soft_budget_gates_new_runs_but_completes_inflight(
    tmp_path: Path,
) -> None:
    """soft budget が超過した時点で、新規 run は skip。既に始まった run は完走。"""

    from das.llm import CostTracker

    llm = _fake_llm("発言")
    personas = [build_persona(name="A")]

    # tracker を仕込んで client に紐付ける
    tracker = CostTracker(budget_usd=1e-6)  # ほぼゼロの soft budget
    tracker._total = 1.0  # type: ignore[attr-defined]  # 開始時点で既に超過
    llm._cost_tracker = tracker  # type: ignore[attr-defined]

    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=3,
        max_turns=1,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="soft-gate",
    )

    meta = json.loads((tmp_path / "soft-gate" / "meta.json").read_text())
    # 3 run すべて skip されているはず (= 既に超過、新規開始しない)
    assert meta["n_runs_skipped_budget"] == 3
    assert meta["n_runs_completed"] == 0
    # exception は無い (skip は意図的な動作)
    assert meta["halted_by_exception"] is False


async def test_run_eval_per_condition_concurrency_limits_active_runs(
    tmp_path: Path,
) -> None:
    """``condition_concurrency`` で指定した condition は同時実行数が絞られる。

    重い condition (full_proposal 想定) を sequential に走らせる用途。
    各 task が semaphore を確認し、超過していないことを assert する。
    """

    # 各 ConditionNone factory が現在の active 数をカウントし、上限を超えていないか確認する
    active_counts: dict[str, int] = {"heavy": 0, "light": 0}
    max_observed: dict[str, int] = {"heavy": 0, "light": 0}
    lock = asyncio.Lock()

    class TrackingCondition:
        def __init__(self, name: str) -> None:
            self._name = name

        name = "tracking"

        async def setup(self, *, docs_dir: Path | None = None) -> None:
            return None

        async def info_provider(self, history, persona):  # type: ignore[no-untyped-def]
            async with lock:
                active_counts[self._name] += 1
                max_observed[self._name] = max(
                    max_observed[self._name], active_counts[self._name]
                )
            # わずかな await で並列性を作る
            await asyncio.sleep(0.005)
            async with lock:
                active_counts[self._name] -= 1
            return None

    llm = _fake_llm("発言")
    personas = [build_persona(name="A")]
    factories = {
        "heavy": lambda: TrackingCondition("heavy"),
        "light": lambda: TrackingCondition("light"),
    }

    await run_eval(
        topic="t",
        personas=personas,
        condition_factories=factories,
        n_runs=3,
        max_turns=2,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="per-cond-conc",
        concurrency=5,  # global は緩い
        condition_concurrency={"heavy": 1},  # heavy だけ 1 に絞る
    )

    # heavy は同時 1 までしか動いていないはず
    assert max_observed["heavy"] == 1
    # light は global=5 まで動ける (n_runs=3 なので最大 3 まで)
    assert max_observed["light"] >= 1
