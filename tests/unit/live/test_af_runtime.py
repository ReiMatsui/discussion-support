"""AFRuntime (H1 フェーズ3: AF 常駐ランタイム) のユニットテスト。

実 LLM は使わず、extraction/linking を AsyncMock で差し替える。カーソル・epoch
ガード・レイテンシ計測・snapshot 保存を検証する。
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from das.asr.live._af_runtime import AFRuntime, _percentile
from das.graph.schema import Node
from das.llm import OpenAIClient
from das.types import Utterance


class _FakeState:
    def __init__(self, records: list[dict]) -> None:
        self.state_lock = threading.Lock()
        self.meeting_epoch = 0
        self.records = records
        self.stop = threading.Event()

    def disp_name(self, key: str) -> str:
        return key


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


def _runtime(records: list[dict], **kwargs: object) -> tuple[AFRuntime, _FakeState]:
    state = _FakeState(records)
    rt = AFRuntime(state, llm=_fake_llm(), **kwargs)  # type: ignore[arg-type]
    return rt, state


# --- _percentile --------------------------------------------------------


def test_percentile_basic() -> None:
    assert _percentile([], 50) is None
    assert _percentile([5.0], 90) == 5.0
    assert _percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)
    # 0..9 の p90 は 8.1 (線形補間)
    assert _percentile([float(i) for i in range(10)], 90) == pytest.approx(8.1)


def test_latency_summary_shape() -> None:
    rt, _ = _runtime([])
    rt.latencies_ms["total"] = [10.0, 20.0, 30.0]
    summary = rt.latency_summary()
    assert summary["total"]["n"] == 3
    assert summary["total"]["p50_ms"] == pytest.approx(20.0)
    assert summary["total"]["max_ms"] == 30.0
    assert summary["extraction"]["n"] == 0
    assert summary["extraction"]["p50_ms"] is None


# --- poll_once: cursor 前進 ---------------------------------------------


def test_poll_once_ingests_new_utterances_and_advances_cursor() -> None:
    records = [{"speaker": "A", "text": "こんにちは"}, {"speaker": "B", "text": "どうも"}]
    rt, _ = _runtime(records)
    rt.ingest_utterance = AsyncMock()  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    try:
        assert rt.poll_once(loop) == 2
        assert rt._cursor == 2
        # 再ポーリングでは新規なし
        assert rt.poll_once(loop) == 0
        # 新しい発話が来たら 1 件処理
        records.append({"speaker": "A", "text": "追加発言"})
        assert rt.poll_once(loop) == 1
        assert rt._cursor == 3
        # turn_id は連番で渡される
        turn_ids = [c.args[0].turn_id for c in rt.ingest_utterance.call_args_list]
        assert turn_ids == [1, 2, 3]
    finally:
        loop.close()


def test_poll_once_skips_agent_and_empty_records() -> None:
    records = [
        {"speaker": "ファシリテーター", "text": "AI の発話"},  # AGENT → 除外
        {"speaker": "A", "text": ""},  # 空 → 除外
        {"speaker": "A", "text": "本物の発話"},
    ]
    rt, _ = _runtime(records)
    rt.ingest_utterance = AsyncMock()  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    try:
        assert rt.poll_once(loop) == 1
    finally:
        loop.close()


def test_poll_once_resets_on_new_meeting_epoch() -> None:
    records = [{"speaker": "A", "text": "旧会議"}]
    rt, state = _runtime(records)
    rt.ingest_utterance = AsyncMock()  # type: ignore[method-assign]
    loop = asyncio.new_event_loop()
    try:
        rt.poll_once(loop)
        assert rt._cursor == 1
        # 新会議: epoch を進め records をクリア
        with state.state_lock:
            state.meeting_epoch = 1
            state.records = []
        assert rt.poll_once(loop) == 0  # リセットのみ
        assert rt._cursor == 0
        assert rt._epoch == 1
    finally:
        loop.close()


# --- ingest_utterance: レイテンシ計測 -----------------------------------


async def test_ingest_utterance_records_latency_and_adds_nodes() -> None:
    from das.agents.extraction import ExtractionOutput

    rt, _ = _runtime([])
    node = Node(text="主張", node_type="claim", source="utterance", author="A", turn_index=1)
    rt._orch.extraction.extract = AsyncMock(  # type: ignore[method-assign]
        return_value=ExtractionOutput(nodes=[node], edges=[])
    )
    rt._orch.linking.link_node = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rt.ingest_utterance(
        Utterance(turn_id=1, speaker="A", text="主張です")
    )

    assert len(rt.latencies_ms["total"]) == 1
    assert len(rt.latencies_ms["extraction"]) == 1
    assert len(rt.latencies_ms["linking"]) == 1
    assert next(iter(rt.store.nodes())).id == node.id
    rt._orch.linking.link_node.assert_awaited_once()


async def test_ingest_discards_result_when_epoch_changes_during_extraction() -> None:
    """T4: extraction (LLM) 中に会議リセットが起きたら、store に一切反映せず破棄する。"""
    from das.agents.extraction import ExtractionOutput

    rt, state = _runtime([])
    node = Node(text="旧会議主張", node_type="claim", source="utterance", author="A",
                turn_index=1)

    async def _extract(utt, context=None):  # type: ignore[no-untyped-def]
        state.meeting_epoch = 1  # 抽出中に会議リセット
        return ExtractionOutput(nodes=[node], edges=[])

    rt._orch.extraction.extract = AsyncMock(side_effect=_extract)  # type: ignore[method-assign]
    rt._orch.linking.link_node = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rt.ingest_utterance(
        Utterance(turn_id=1, speaker="A", text="x"), expected_epoch=0)

    assert list(rt.store.nodes()) == []              # store に追加されない
    assert len(rt.latencies_ms["total"]) == 0        # レイテンシも記録しない
    rt._orch.linking.link_node.assert_not_awaited()  # linking も呼ばれない


# --- 介入ノード・応答エッジの計測 (フェーズ5) ---------------------------


async def _ingest_with_embedding(rt, node, vec) -> None:
    """extraction=node, linking=no-op(ただし embedding を仕込む) で 1 発話取り込む。"""
    from das.agents.extraction import ExtractionOutput

    rt._orch.extraction.extract = AsyncMock(  # type: ignore[method-assign]
        return_value=ExtractionOutput(nodes=[node], edges=[])
    )

    async def _fake_link(n, store):  # type: ignore[no-untyped-def]
        rt._orch.linking._embeddings[n.id] = vec
        return []

    rt._orch.linking.link_node = AsyncMock(side_effect=_fake_link)  # type: ignore[method-assign]
    await rt.ingest_utterance(Utterance(turn_id=node.turn_index, speaker=node.author, text=node.text))


async def test_note_intervention_and_responds_to() -> None:
    """介入を記録し、類似発話が来たら responds_to (受容の痕跡) が張られる。"""
    rt, _ = _runtime([])
    rt._llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    rt.note_intervention("af_l1", "コスト懸念がある")

    node = Node(text="コスト懸念に同意", node_type="claim", source="utterance",
                author="B", turn_index=2)
    await _ingest_with_embedding(rt, node, [0.99, 0.01])  # 介入と高類似

    summ = rt.acceptance_summary()
    assert summ["n_interventions"] == 1
    assert summ["n_responded"] == 1
    assert summ["acceptance_rate"] == pytest.approx(1.0)


async def test_no_responds_to_when_dissimilar() -> None:
    """類似度が閾値未満なら responds_to は張られない。"""
    rt, _ = _runtime([])
    rt._llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    rt.note_intervention("af_l1", "コスト懸念がある")

    node = Node(text="全然別の話", node_type="claim", source="utterance",
                author="B", turn_index=2)
    await _ingest_with_embedding(rt, node, [0.0, 1.0])  # 直交

    summ = rt.acceptance_summary()
    assert summ["n_interventions"] == 1
    assert summ["n_responded"] == 0
    assert summ["acceptance_rate"] == pytest.approx(0.0)


def test_snapshot_includes_interventions(tmp_path: Path) -> None:
    snap = tmp_path / "sess.af.json"
    rt, _ = _runtime([], snapshot_path=snap)
    rt.note_intervention("af_l1", "提示テキスト")
    rt._response_edges.append({"intervention_id": "iv-001", "utterance_node_id": "x",
                               "similarity": 0.9, "responded_at_turn": 2})
    rt.save_snapshot()
    payload = json.loads(snap.read_text(encoding="utf-8"))
    assert len(payload["af_interventions"]) == 1
    assert payload["af_interventions"][0]["text"] == "提示テキスト"
    assert "embedding" not in payload["af_interventions"][0]
    assert len(payload["af_response_edges"]) == 1
    # interventions.jsonl も書かれる
    assert snap.with_suffix(".interventions.jsonl").exists()


# --- snapshot 保存 ------------------------------------------------------


def test_save_snapshot_writes_file(tmp_path: Path) -> None:
    snap = tmp_path / "sess.af.json"
    rt, _ = _runtime([], snapshot_path=snap)
    node = Node(text="主張", node_type="claim", source="utterance", author="A")
    rt.store.add_node(node)
    rt.save_snapshot()
    assert snap.exists()
    payload = json.loads(snap.read_text(encoding="utf-8"))
    assert len(payload["nodes"]) == 1
