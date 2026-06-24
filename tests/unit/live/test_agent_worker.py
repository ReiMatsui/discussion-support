"""_run_agent_worker の介入リトライ集約（Bug 3）の回帰テスト.

実 RealtimeAgent/Partner/SessionState を使わず、最小限のフェイクで
ワーカーループの分岐挙動だけを検証する。
"""
from __future__ import annotations

import queue
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
        self.drift_cursor = 0
        self.drift_requests: queue.Queue[str] = queue.Queue()

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


def test_drift_request_triggers_with_reason():
    """drift_requestsに積まれた脱線要求を、agent_workerがdrift_reason付きでトリガーする（R2）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.drift_requests.put("ラーメンの雑談")

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "脱線要求でトリガーされるべき"
    assert agent.trigger_calls[0]["drift_reason"] == "ラーメンの雑談"


def test_drift_request_held_until_agent_free():
    """agentがbusy(応答中)の間は脱線介入を保持し、freeになってから発火する（R2）.

    保持状態はワーカースレッドのローカルに持つため、単一スレッドのまま
    busy→free に切り替えて検証する（本番もワーカーは1本の長命スレッド）。
    """
    agent = FakeAgent()
    agent._responding = True            # busy
    state = FakeState(agent, None)
    state.drift_requests.put("脱線理由")

    t = threading.Thread(target=_run_agent_worker, args=(state,), daemon=True)
    t.start()
    try:
        time.sleep(1.2)
        assert agent.trigger_calls == [], "busy中はトリガーされない（要求は保持）"
        agent._responding = False       # free化
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not agent.trigger_calls:
            time.sleep(0.05)
    finally:
        state.stop.set()
        t.join(timeout=2.0)

    assert agent.trigger_calls, "freeになったら保持していた要求で発火するべき"
    assert agent.trigger_calls[0]["drift_reason"] == "脱線理由"


def test_drift_intervention_cooldown_suppresses_repeats():
    """介入直後のクールダウン中は、続く脱線要求を抑制して連発を防ぐ（しつこさ緩和）."""
    from das.asr.live._constants import _INTERVENTION_COOLDOWN

    agent = FakeAgent()
    state = FakeState(agent, None)
    state.drift_requests.put("脱線1")

    t = threading.Thread(target=_run_agent_worker, args=(state,), daemon=True)
    t.start()
    try:
        # 1回目の脱線で介入が入る
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not agent.trigger_calls:
            time.sleep(0.05)
        assert len(agent.trigger_calls) == 1

        # クールダウン中に続けて脱線要求しても介入は増えない
        for _ in range(3):
            state.drift_requests.put("脱線2")
            time.sleep(0.6)
        assert len(agent.trigger_calls) == 1, "クールダウン中は連発しない"
    finally:
        state.stop.set()
        t.join(timeout=2.0)

    # クールダウンが十分長いこと（テストが秒単位で破綻しない範囲）を確認
    assert _INTERVENTION_COOLDOWN >= 10.0


def test_no_stall_breaker_without_noop():
    """介入不要の履歴がなければ、沈黙していても一押しはしない（通常の間は尊重）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state._last_utt_time[0] = time.monotonic() - 100

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []


# ---------------------------------------------------------------------------
# ドリフトチェッカーのウォームアップ（Fix 11）
# ---------------------------------------------------------------------------

def _run_drift_checker_briefly(state, monkeypatch, *, records, seconds=2.5):
    """check_drift をモックして _run_drift_checker を短時間動かし、呼び出しを記録."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_drift_checker

    calls: list = []

    def _fake_check(*_a, **_k):
        calls.append(1)
        return {"drift": False}

    monkeypatch.setattr(bootstrap, "check_drift", _fake_check)
    state.topics = [{"topic": "AI導入の是非", "speaker": "議題"}]
    state.records = records
    t = threading.Thread(target=_run_drift_checker,
                         args=(state, "key", "gpt-5-mini"), daemon=True)
    t.start()
    time.sleep(seconds)
    state.stop.set()
    t.join(timeout=2.0)
    return calls


def test_drift_warmup_skips_opening_greeting(monkeypatch):
    """開始直後（発話数 < ウォームアップ）は脱線判定しない（Fix 11）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    records = [{"speaker": "話者1", "text": "こんにちは、よろしくお願いします。"}]
    calls = _run_drift_checker_briefly(state, monkeypatch, records=records)
    assert calls == [], "開始直後の挨拶で脱線判定を走らせてはならない"


def test_drift_runs_after_warmup(monkeypatch):
    """ウォームアップ発話数に達したら脱線判定が走る（Fix 11）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    records = [
        {"speaker": "話者1", "text": "AI導入は段階的にやるべき"},
        {"speaker": "話者2", "text": "データ管理のルールが先だと思う"},
        {"speaker": "話者1", "text": "ところでリゾット食べたい"},
    ]
    calls = _run_drift_checker_briefly(state, monkeypatch, records=records)
    assert calls, "ウォームアップ後は脱線判定が走るべき"
