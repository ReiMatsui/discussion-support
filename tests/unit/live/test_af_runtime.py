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
    rt, _ = _runtime([])
    node = Node(text="主張", node_type="claim", source="utterance", author="A", turn_index=1)
    rt._orch.extraction.extract = AsyncMock(return_value=[node])  # type: ignore[method-assign]
    rt._orch.linking.link_node = AsyncMock(return_value=[])  # type: ignore[method-assign]

    await rt.ingest_utterance(
        Utterance(turn_id=1, speaker="A", text="主張です")
    )

    assert len(rt.latencies_ms["total"]) == 1
    assert len(rt.latencies_ms["extraction"]) == 1
    assert len(rt.latencies_ms["linking"]) == 1
    assert next(iter(rt.store.nodes())).id == node.id
    rt._orch.linking.link_node.assert_awaited_once()


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
