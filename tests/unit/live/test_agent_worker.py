"""_run_agent_worker の介入リトライ集約（Bug 3）の回帰テスト.

実 RealtimeAgent/Partner/SessionState を使わず、最小限のフェイクで
ワーカーループの分岐挙動だけを検証する。
"""
from __future__ import annotations

import threading
import time

from das.asr.live._workers import _run_agent_worker


class FakeAgent:
    def __init__(self, mode: str = "facilitator") -> None:
        self._connected = True
        self.enabled = True
        self.mode = mode
        self.ai_speaking = False
        self._responding = False
        self._pending_intervention: dict | None = None
        self._last_noop_at = 0.0
        self.trigger_n = 10
        self.in_echo_window = False
        self._pending: list = []
        self.trigger_calls: list[dict] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def feed(self, speaker: str, text: str, **kw) -> None:  # pragma: no cover
        pass

    def trigger(self, *, topics=None, drift_reason=None) -> None:
        self.trigger_calls.append({"topics": topics, "drift_reason": drift_reason})
        # 実エージェントの挙動を模倣: トリガーで介入と保留発話を消費
        self._pending_intervention = None
        self._pending.clear()

    def interrupt(self) -> None:  # pragma: no cover
        pass


class FakePartner:
    def __init__(self) -> None:
        self._connected = True
        self.ai_speaking = False
        self._responding = False
        self.interrupts = 0

    def interrupt(self) -> None:
        self.interrupts += 1

    def inject_context(self, *a, **k) -> None:  # pragma: no cover
        pass


class FakeState:
    def __init__(self, agent, partner=None) -> None:
        self.stop = threading.Event()
        self.state_lock = threading.Lock()
        self.topics_lock = threading.Lock()
        self.topics: list = []
        self.records: list = []
        self.agent = agent
        self.partner = partner
        self.simulator = None
        self._last_utt_time = [time.monotonic()]
        self._was_in_echo = [False]
        self.agent_cursor = 0

    def disp_name(self, k):  # pragma: no cover
        return str(k)


def _run_worker_briefly(state, *, until, timeout=3.0) -> None:
    """ワーカーを別スレッドで起動し、until() が真になるかtimeoutまで待つ."""
    t = threading.Thread(target=_run_agent_worker, args=(state,), daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not until():
        time.sleep(0.05)
    state.stop.set()
    t.join(timeout=2.0)


def test_retry_fires_even_while_partner_speaking():
    """パートナー発話中でも、中断された介入はリトライされる（Bug 3集約後の核心挙動）.

    旧実装ではパートナー発話ガードの continue により retry に到達せず、
    会話中はリトライが永久に発火しなかった。
    """
    agent = FakeAgent()
    partner = FakePartner()
    partner.ai_speaking = True
    agent._pending_intervention = {
        "delivered": "中断された指摘", "created_at": time.monotonic(), "attempts": 1,
    }
    state = FakeState(agent, partner)

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "パートナー発話中でも中断介入はリトライされるべき"


def test_retry_waits_while_agent_busy():
    """agentが応答生成中(_responding)はリトライしない."""
    agent = FakeAgent()
    agent._responding = True
    agent._pending_intervention = {
        "delivered": "x", "created_at": time.monotonic(), "attempts": 1,
    }
    state = FakeState(agent, None)

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []


def test_no_retry_when_no_pending_intervention():
    """保留介入がなければ（沈黙も短ければ）トリガーしない."""
    agent = FakeAgent()
    state = FakeState(agent, None)

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []


def test_stall_breaker_fires_after_noop_silence():
    """介入不要後に沈黙が続いたら、本題に戻す一押しをトリガーする（Fix 10）."""
    agent = FakeAgent()
    agent._last_noop_at = time.monotonic()      # 直前に「介入不要」と判断
    state = FakeState(agent, None)
    state._last_utt_time[0] = time.monotonic() - 100  # 十分な沈黙を模擬

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "介入不要後の沈黙で一押しが入るべき"
    assert "止まって" in (agent.trigger_calls[0]["drift_reason"] or "")
    assert agent._last_noop_at == 0.0  # 発火後はマーカーを解除


def test_no_stall_breaker_without_noop():
    """介入不要の履歴がなければ、沈黙していても一押しはしない（通常の間は尊重）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state._last_utt_time[0] = time.monotonic() - 100

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []
