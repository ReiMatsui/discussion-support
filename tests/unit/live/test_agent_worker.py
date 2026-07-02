"""_run_agent_worker の介入リトライ集約（Bug 3）の回帰テスト.

実 RealtimeAgent/Partner/SessionState を使わず、最小限のフェイクで
ワーカーループの分岐挙動だけを検証する。
"""
from __future__ import annotations

import queue
import threading
import time

from das.asr.live._constants import _DRIFT_PENDING_TTL
from das.asr.live._workers import (
    _build_candidates,
    _log_intervention_event,
    _log_voice_call_diag,
    _PendingInterventions,
    _run_agent_worker,
)


class FakeAgent:
    def __init__(self, mode: str = "facilitator") -> None:
        self._connected = True
        self.enabled = True
        self.mode = mode
        self.ai_speaking = False
        self._responding = False
        self._pending_intervention: dict | None = None
        self.trigger_n = 10
        self.in_echo_window = False
        self._pending: list = []
        self.feeds: list[tuple[str, str]] = []
        self.trigger_calls: list[dict] = []
        self.connect_calls = 0
        self.interrupts = 0

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def feed(self, speaker: str, text: str, **kw) -> None:
        self.feeds.append((speaker, text))

    def trigger(self, *, topics=None, drift_reason=None, invite_target=None,
                fact_correction=None, manual_request=None,
                summary_focus=None, retry_intervention=None) -> None:
        self.trigger_calls.append({"topics": topics, "drift_reason": drift_reason,
                                   "invite_target": invite_target,
                                   "fact_correction": fact_correction,
                                   "manual_request": manual_request,
                                   "summary_focus": summary_focus,
                                   "retry_intervention": retry_intervention})
        # 実エージェントの挙動を模倣: トリガーで介入と保留発話を消費
        self._pending_intervention = None
        self._pending.clear()

    def interrupt(self) -> None:
        self.interrupts += 1

    def connect(self) -> None:
        self.connect_calls += 1
        self._connected = True


class FakePartner:
    def __init__(self) -> None:
        self._connected = True
        self.ai_speaking = False
        self._responding = False
        self.interrupts = 0
        self.injected: list = []

    def interrupt(self) -> None:
        self.interrupts += 1

    def inject_context(self, speaker=None, text=None, **k) -> None:
        self.injected.append((speaker, text))


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
        self.meeting_epoch = 0
        self.partial_text = ""
        self._last_partial_change = 0.0
        self.agent_cursor = 0
        self.drift_cursor = 0
        self.fact_cursor = 0
        self.drift_requests: queue.Queue[str] = queue.Queue()
        self.invite_requests: queue.Queue[str] = queue.Queue()
        self.factcheck_requests: queue.Queue[dict] = queue.Queue()
        self.manual_call_requests: queue.Queue[dict] = queue.Queue()
        self.summarize_requests: queue.Queue[dict] = queue.Queue()
        self.fac_events: queue.Queue = queue.Queue()
        self.proactivity = {"silence_summarize": 18.0, "cooldown": 25.0}
        self.intervention_enabled = True
        self.intervention_events: list[dict] = []
        self.written_events: list[dict] = []       # write_intervention_event の記録
        self.manual_statuses: list[dict] = []      # set_manual_call_status の記録

    def disp_name(self, k):  # pragma: no cover
        return str(k)

    def add_intervention_event(self, reason: str, detail: str = "",
                               metadata: dict | None = None) -> None:
        self.intervention_events.append({
            "reason": reason,
            "detail": detail,
            "metadata": metadata or {},
        })

    def write_intervention_event(self, event: dict) -> None:
        self.written_events.append(event)

    def set_manual_call_status(self, status: str, **kw) -> None:
        self.manual_statuses.append({"status": status, **kw})


def test_log_intervention_event_includes_review_context():
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.proactivity_name = "controlled"
    state.topics = [{"topic": "AI導入", "speaker": "議題"}]
    state.records = [
        {"speaker": "A", "text": f"発話{i}", "ms": i * 1000, "end_ms": i * 1000 + 500}
        for i in range(6)
    ]

    _log_intervention_event(state, "drift", "雑談")

    event = state.intervention_events[0]
    assert event["reason"] == "drift"
    assert event["metadata"]["mode"] == "facilitator"
    assert event["metadata"]["proactivity"] == "controlled"
    assert event["metadata"]["turn_count"] == 6
    assert event["metadata"]["topics"] == [{"topic": "AI導入", "speaker": "議題"}]
    assert [u["text"] for u in event["metadata"]["recent_utterances"]] == [
        "発話1", "発話2", "発話3", "発話4", "発話5",
    ]


def _run_worker_briefly(state, *, until, timeout=3.0) -> None:
    """ワーカーを別スレッドで起動し、until() が真になるかtimeoutまで待つ."""
    t = threading.Thread(target=_run_agent_worker, args=(state,), daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not until():
        time.sleep(0.05)
    state.stop.set()
    t.join(timeout=2.0)


def test_retry_waits_while_partner_speaking():
    """中断された介入の再送も、パートナー発話中は待つ."""
    agent = FakeAgent()
    partner = FakePartner()
    partner.ai_speaking = True
    agent._pending_intervention = {
        "delivered": "中断された指摘", "created_at": time.monotonic(), "attempts": 1,
    }
    state = FakeState(agent, partner)

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []


def test_retry_fires_after_pause():
    """中断された介入は、発話の切れ目ができたら再送できる."""
    agent = FakeAgent()
    agent._pending_intervention = {
        "delivered": "中断された指摘", "created_at": time.monotonic(), "attempts": 1,
    }
    state = FakeState(agent, None)
    state._last_utt_time[0] = time.monotonic() - 10

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls


def test_retry_waits_for_longer_pause_than_fact():
    """再送はしつこく見えやすいので、短い間では待つ."""
    agent = FakeAgent()
    agent._pending_intervention = {
        "delivered": "中断された指摘", "created_at": time.monotonic(), "attempts": 1,
    }
    state = FakeState(agent, None)
    state._last_utt_time[0] = time.monotonic() - 1.0

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []


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


def test_agent_worker_reconnects_disconnected_enabled_agent():
    """AI AgentのWebSocketが落ちたら、ワーカーが再接続を試みる."""
    agent = FakeAgent()
    agent._connected = False
    state = FakeState(agent, None)

    _run_worker_briefly(state, until=lambda: agent.connect_calls > 0, timeout=1.5)

    assert agent.connect_calls == 1
    assert agent._connected is True


def test_dead_air_alone_does_not_trigger_push():
    """デッドエアの一押し（旧 stall）は廃止済み。発言が溜まっていなければ黙る.

    以前は「介入不要」後の沈黙で一押ししていたが、Phase3 で Speaker から
    「介入不要」判断を外したため、その概念ごと廃止した。activeプロファイルで
    沈黙が続いても、pending が無ければ何もトリガーしない。
    """
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.proactivity = {"silence_summarize": 8.0, "cooldown": 15.0,
                         "drift_confirmations": 1}
    state._last_utt_time[0] = time.monotonic() - 100  # 十分な沈黙を模擬

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == [], "デッドエアの一押し（stall）は廃止済み"


def test_controlled_proactivity_no_silence_summarize():
    """controlledでは沈黙だけでは要約介入しない（過剰介入の抑制, S5）."""
    agent = FakeAgent()
    agent._pending = [{"speaker": "人間", "text": "x", "_count": True}]
    state = FakeState(agent, None)
    state.proactivity = {"silence_summarize": None, "cooldown": 40.0}
    state._last_utt_time[0] = time.monotonic() - 100  # 長い沈黙

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)
    assert agent.trigger_calls == []


def test_standard_proactivity_silence_summarize_fires():
    """standardでは沈黙が閾値を超えたら要約介入する（S5）."""
    agent = FakeAgent()
    agent._pending = [{"speaker": "人間", "text": "x", "_count": True}]
    state = FakeState(agent, None)  # default standard: silence_summarize=18
    state._last_utt_time[0] = time.monotonic() - 100

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))
    assert agent.trigger_calls


def test_intervention_disabled_skips_facilitator_but_keeps_partner_context():
    """介入オフでもAIパートナーには人間発話を渡し、進行役トリガーだけ止める。"""
    agent = FakeAgent()
    partner = FakePartner()
    partner.ai_speaking = True
    state = FakeState(agent, partner)
    state.intervention_enabled = False
    state.records = [{"speaker": "#1", "text": "この点は違うと思います", "ms": 0}]

    _run_worker_briefly(state, until=lambda: bool(partner.injected), timeout=1.5)

    assert agent.feeds == []
    assert agent.trigger_calls == []
    assert partner.interrupts == 1
    assert partner.injected == [("人間", "この点は違うと思います")]


def test_drift_request_triggers_with_reason():
    """連続した脱線要求で、agent_workerがdrift_reason付きでトリガーする（R2）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.drift_requests.put("ラーメンの雑談")
    state.drift_requests.put("ラーメンの雑談")

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "脱線要求でトリガーされるべき"
    assert agent.trigger_calls[0]["drift_reason"] == "ラーメンの雑談"
    event = state.intervention_events[0]
    assert event["reason"] == "drift"
    assert event["detail"] == "ラーメンの雑談"
    assert event["metadata"]["mode"] == "facilitator"
    assert event["metadata"]["turn_count"] == 0
    assert event["metadata"]["timing"]["kind"] == "drift"
    assert event["metadata"]["timing"]["policy"] == "drift_confirmation_pause"
    assert event["metadata"]["timing"]["candidate_wait_sec"] >= 0


def test_manual_call_triggers_with_request():
    """手動呼び出しが agent.trigger(manual_request=...) を呼び、ログに残る."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.manual_call_requests.put({"request": "ここまで整理して", "source": "ui"})

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "手動呼び出しでトリガーされるべき"
    assert agent.trigger_calls[0]["manual_request"]["request"] == "ここまで整理して"
    event = state.intervention_events[0]
    assert event["reason"] == "manual_call"
    # 観測性: source / request / 待ち時間 / 採択結果がログから追える
    timing = event["metadata"]["timing"]
    assert timing["source"] == "ui"
    assert timing["request"] == "ここまで整理して"
    assert timing["outcome"] == "selected"
    assert "candidate_wait_sec" in timing
    assert "queued_at" in timing
    # UIステータス: 発話直前に dispatched へ更新される
    assert any(s["status"] == "dispatched" for s in state.manual_statuses)
    assert event["detail"] == "ここまで整理して"
    assert event["metadata"]["timing"]["kind"] == "manual"
    assert event["metadata"]["timing"]["policy"] == "manual_call_pause"
    assert event["metadata"]["timing"]["source"] == "ui"


def test_manual_call_empty_request_uses_default_detail():
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.manual_call_requests.put({"request": "", "source": "ui"})

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls
    assert state.intervention_events[0]["detail"] == "直近の議論整理"


def test_manual_call_preferred_over_drift():
    """manual は drift より優先（明示呼び出しを尊重）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.manual_call_requests.put({"request": "整理して", "source": "ui"})
    state.drift_requests.put("式の話題")

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls
    assert agent.trigger_calls[0]["manual_request"] is not None
    assert agent.trigger_calls[0]["drift_reason"] is None
    assert state.intervention_events[0]["reason"] == "manual_call"


def test_manual_call_held_while_partner_speaking_then_fires():
    """パートナー発話中は保持し、空いたら手動呼び出しが発火する（すぐ捨てない）."""
    agent = FakeAgent()
    partner = FakePartner()
    partner.ai_speaking = True
    state = FakeState(agent, partner)
    state.manual_call_requests.put({"request": "整理して", "source": "ui"})

    t = threading.Thread(target=_run_agent_worker, args=(state,), daemon=True)
    t.start()
    try:
        time.sleep(1.0)
        assert agent.trigger_calls == [], "パートナー発話中は保持（発火しない）"
        partner.ai_speaking = False
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not agent.trigger_calls:
            time.sleep(0.05)
    finally:
        state.stop.set()
        t.join(timeout=2.0)

    assert agent.trigger_calls, "空いたら保持していた手動呼び出しで発火するべき"
    assert agent.trigger_calls[0]["manual_request"]["request"] == "整理して"


def test_manual_call_dropped_after_ttl():
    """TTLを超えた古い手動呼び出しは破棄して発火しない."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.manual_call_requests.put({
        "request": "整理して", "source": "ui",
        "created_at": time.monotonic() - 100,   # TTL(30s)超過
    })

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []
    # 観測性: 「呼んだのに反応しなかった」が trigger ログから追える
    expired = [e for e in state.intervention_events
               if e["reason"] == "manual_call_expired"]
    assert expired, "期限切れの手動呼び出しはログに残すべき"
    timing = expired[0]["metadata"]["timing"]
    assert timing["source"] == "ui"
    assert timing["request"] == "整理して"
    assert timing["outcome"] == "expired"
    assert timing["candidate_wait_sec"] > 30
    # UIステータスも expired へ
    assert any(s["status"] == "expired" for s in state.manual_statuses)


def test_manual_call_status_waiting_while_partner_busy():
    """パートナー発話中に保留された手動呼び出しは「待機中」として見える."""
    agent = FakeAgent()
    partner = FakePartner()
    partner.ai_speaking = True
    state = FakeState(agent, partner)
    state.manual_call_requests.put({"request": "整理して", "source": "ui"})

    _run_worker_briefly(
        state,
        until=lambda: any(s["status"] == "waiting" for s in state.manual_statuses),
        timeout=2.0)

    waiting = [s for s in state.manual_statuses if s["status"] == "waiting"]
    assert waiting, "保留中は waiting ステータスを更新するべき"
    assert waiting[0].get("wait_sec") is not None


def test_manual_call_status_cancelled_when_disabled():
    """介入オフで破棄された手動呼び出しは cancelled として見える."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.intervention_enabled = False
    state.manual_call_requests.put({"request": "整理して", "source": "ui"})

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []
    assert any(s["status"] == "cancelled" for s in state.manual_statuses)


def test_manual_call_ignored_when_intervention_disabled():
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.intervention_enabled = False
    state.manual_call_requests.put({"request": "整理して", "source": "ui"})

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []


def test_pending_manual_call_cleared_when_intervention_disabled():
    """worker内に保持された手動呼び出しも、介入オフで破棄する."""
    agent = FakeAgent()
    partner = FakePartner()
    partner.ai_speaking = True
    state = FakeState(agent, partner)
    state.manual_call_requests.put({"request": "整理して", "source": "ui"})

    t = threading.Thread(target=_run_agent_worker, args=(state,), daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not state.manual_call_requests.empty():
            time.sleep(0.05)
        assert state.manual_call_requests.empty(), "workerが手動呼び出しを保持している前提"

        state.intervention_enabled = False
        partner.ai_speaking = False
        time.sleep(0.4)
        state.intervention_enabled = True
        time.sleep(0.6)
    finally:
        state.stop.set()
        t.join(timeout=2.0)

    assert agent.trigger_calls == []


# ---------------------------------------------------------------------------
# Phase2: 音声での明示的なファシリテーター呼びかけ検出
# ---------------------------------------------------------------------------

def test_candidate_brief_includes_manual_tracking_fields():
    """review ログの manual 候補には source/request/queued_at が載る（観測性）."""
    from das.asr.live._workers import _build_candidates, _candidate_brief

    agent = FakeAgent()
    pending = _PendingInterventions()
    now = time.monotonic()
    pending.manual_call = {"request": "ここまで整理して", "source": "voice",
                           "created_at": now}
    pending.drift_reason = "話題ずれ"
    cands = _build_candidates(pending, agent, now=now)
    brief = _candidate_brief(next(c for c in cands if c.kind == "manual"))
    assert brief["source"] == "voice"
    assert brief["request"] == "ここまで整理して"
    assert brief["queued_at"] == now
    # 既存フィールドは不変
    other = _candidate_brief(next(c for c in cands if c.kind != "manual"))
    assert "source" not in other and "request" not in other


def test_voice_call_queues_manual_and_triggers():
    """triage が積んだ manual(source=voice) が trigger まで届く.

    呼びかけの検出自体は _run_triage_worker（LLM分類）の責務
    （tests/unit/live/test_human_mode.py 側で検証）。ここでは voice 由来の
    manual 要求が agent worker 経由で発火することを確認する。
    """
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.manual_call_requests.put({
        "request": "ここまで整理して", "source": "voice",
        "created_at": time.monotonic(),
    })

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "音声呼びかけで手動呼び出しが発火するべき"
    mr = agent.trigger_calls[0]["manual_request"]
    assert mr is not None
    assert mr["source"] == "voice"
    assert mr["request"] == "ここまで整理して"
    assert state.intervention_events[0]["reason"] == "manual_call"
    assert state.intervention_events[0]["metadata"]["timing"]["source"] == "voice"


def test_voice_call_diag_helper_writes_event():
    """音声呼びかけの検出 diag が jsonl に残る（不発/誤爆の事後検証用）."""
    agent = FakeAgent()
    state = FakeState(agent, None)

    _log_voice_call_diag(state, text="AIさん、ここまで整理して",
                         request="ここまでの整理")

    diags = [e for e in state.written_events if e["type"] == "voice_call_diag"]
    assert diags and diags[0]["detected"] is True
    assert diags[0]["request"] == "ここまでの整理"


def test_fact_request_triggers_before_drift():
    """高確信の事実補正は、会話を壊さない短い補足として優先的に発火する."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.factcheck_requests.put({
        "should_correct": True,
        "confidence": "high",
        "claim": "指標Xの計算式は分母を分子で割る",
        "correction": "指標Xは分子を分母で割ります。",
        "reason": "計算式が逆",
    })
    state.drift_requests.put("式の話題")

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls
    assert agent.trigger_calls[0]["fact_correction"]["correction"].startswith("指標Xは")
    assert agent.trigger_calls[0]["drift_reason"] is None
    assert agent.trigger_calls[0]["retry_intervention"] is False
    assert state.intervention_events[0]["reason"] == "fact"
    assert state.intervention_events[0]["metadata"]["timing"]["kind"] == "fact"
    assert state.intervention_events[0]["metadata"]["timing"]["policy"] == "fact_freshness_pause"


def test_fact_request_is_held_during_cooldown(monkeypatch):
    """事実補正はクールダウン中でも破棄せず、明けたら発火する."""
    import das.asr.live._workers as workers

    monkeypatch.setattr(workers, "_FACTCHECK_COOLDOWN", 1.0)
    agent = FakeAgent()
    state = FakeState(agent, None)

    t = threading.Thread(target=_run_agent_worker, args=(state,), daemon=True)
    t.start()
    try:
        state.factcheck_requests.put({
            "should_correct": True,
            "confidence": "high",
            "correction": "事物Aの高さは約3,000メートルです。",
        })
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(agent.trigger_calls) < 1:
            time.sleep(0.05)
        assert len(agent.trigger_calls) == 1

        state.factcheck_requests.put({
            "should_correct": True,
            "confidence": "high",
            "correction": "国Bの首都は都市Aです。",
        })
        time.sleep(0.3)
        assert len(agent.trigger_calls) == 1, "クールダウン中はまだ発火しない"

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and len(agent.trigger_calls) < 2:
            time.sleep(0.05)
    finally:
        state.stop.set()
        t.join(timeout=2.0)

    assert len(agent.trigger_calls) == 2
    assert agent.trigger_calls[1]["fact_correction"]["correction"].startswith("国B")


def test_fact_requests_are_drained_fifo_not_overwritten(monkeypatch):
    """busy中に複数の事実補正が積まれても、最後の1件で上書きせず順番に処理する."""
    import das.asr.live._workers as workers

    monkeypatch.setattr(workers, "_FACTCHECK_COOLDOWN", 0.0)
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.factcheck_requests.put({
        "should_correct": True,
        "confidence": "high",
        "correction": "1つ目の補正です。",
    })
    state.factcheck_requests.put({
        "should_correct": True,
        "confidence": "high",
        "correction": "2つ目の補正です。",
    })

    _run_worker_briefly(state, until=lambda: len(agent.trigger_calls) >= 2)

    corrections = [
        call["fact_correction"]["correction"]
        for call in agent.trigger_calls
    ]
    assert corrections == ["1つ目の補正です。", "2つ目の補正です。"]


def test_single_drift_request_is_held_until_confirmed():
    """controlledでは単発の脱線判定だけでは即介入しない。"""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.proactivity = {"silence_summarize": None, "cooldown": 40.0,
                         "drift_confirmations": 2}
    state.drift_requests.put("地元の雑談")

    _run_worker_briefly(state, until=lambda: False, timeout=1.0)

    assert agent.trigger_calls == []
    assert state.intervention_events == []


def test_unconfirmed_drift_does_not_starve_other_interventions():
    """確認待ちの脱線候補が hold でも、通常レーンの介入は飢餓しない（C1回帰）.

    drift_confirmations=2 で単発検出の drift が保留のまま残っても、
    価値判定済みの整理介入（summarize）は自身の pause を満たせば発火するべき。
    """
    agent = FakeAgent()
    agent._pending = [{"speaker": "人間", "text": str(i), "_count": True}
                      for i in range(agent.trigger_n)]
    state = FakeState(agent, None)
    state.proactivity = {"silence_summarize": None, "cooldown": 40.0,
                         "drift_confirmations": 2}
    state.drift_requests.put("地元の雑談")   # 単発 → 確認待ちで採択されない
    state.summarize_requests.put({"focus": "論点の整理"})

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "確認待ちdriftの保留中も summarize は発火するべき"
    assert state.intervention_events
    assert state.intervention_events[0]["reason"] == "summarize"


def test_stale_pending_drift_is_dropped_after_ttl():
    """確認待ちのまま古くなった脱線候補は TTL で破棄される（C1回帰）."""
    now = time.monotonic()
    pending = _PendingInterventions(drift_reason="地元の雑談", drift_count=1)
    pending.last_drift_request_at = now - (_DRIFT_PENDING_TTL + 1.0)

    pending.drop_stale_drift(now=now)

    assert pending.drift_reason is None
    assert pending.drift_count == 0


def test_fresh_pending_drift_is_kept():
    """寿命内の脱線候補は破棄されない."""
    now = time.monotonic()
    pending = _PendingInterventions(drift_reason="地元の雑談", drift_count=1)
    pending.last_drift_request_at = now - 5.0

    pending.drop_stale_drift(now=now)

    assert pending.drift_reason == "地元の雑談"


def test_drift_candidate_carries_expiry():
    """drift 候補にも expires_at が付き、Controller の期限切れ判定に乗る."""
    agent = FakeAgent()
    pending = _PendingInterventions(drift_reason="地元の雑談", drift_count=1)
    pending.last_drift_request_at = 100.0

    cands = _build_candidates(pending, agent, now=105.0)

    drift = next(c for c in cands if c.kind == "drift")
    assert drift.expires_at == 100.0 + _DRIFT_PENDING_TTL


def test_summarize_trigger_records_intervention_reason():
    """整理介入（価値判定済み summarize）の理由・焦点をUI用ログに残す（C3）."""
    agent = FakeAgent()
    agent._pending = [{"speaker": "人間", "text": str(i), "_count": True}
                      for i in range(agent.trigger_n)]
    state = FakeState(agent, None)
    state.summarize_requests.put({"focus": "論点の整理"})

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls
    assert agent.trigger_calls[0]["summary_focus"] == "論点の整理"
    assert state.intervention_events
    assert state.intervention_events[0]["reason"] == "summarize"
    assert state.intervention_events[0]["metadata"]["timing"]["kind"] == "summarize"
    assert state.intervention_events[0]["metadata"]["timing"]["policy"] == "structuring_value"


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
        deadline = time.monotonic() + 3.0
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


def test_invite_fires_at_pause_with_target():
    """声かけ要求は、沈黙の間が空いてから対象話者付きでトリガーされる（S4）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.invite_requests.put("参加者B")
    state._last_utt_time[0] = time.monotonic() - 100  # 十分な沈黙(間)

    _run_worker_briefly(state, until=lambda: bool(agent.trigger_calls))

    assert agent.trigger_calls, "沈黙の間で声かけがトリガーされるべき"
    assert agent.trigger_calls[0]["invite_target"] == "参加者B"


def test_invite_waits_for_pause():
    """沈黙の間が無い（直前に発話があった）うちは声かけしない（割り込まない）（S4）."""
    agent = FakeAgent()
    state = FakeState(agent, None)
    state.invite_requests.put("参加者B")
    state._last_utt_time[0] = time.monotonic()  # たった今発話があった → 間が無い

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


# ---------------------------------------------------------------------------
# ファシリテーターイベントワーカー（受信スレッドの切り離し）
# ---------------------------------------------------------------------------

def _run_event_worker_briefly(state, on_text, *, until, timeout=2.0):
    from das.asr.live._workers import _run_facilitator_event_worker
    t = threading.Thread(target=_run_facilitator_event_worker,
                         args=(state, on_text), daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not until():
        time.sleep(0.02)
    state.stop.set()
    t.join(timeout=1.0)


def test_event_worker_utterance_appends_and_reacts_partner():
    """utteranceイベントで議事録追記＋パートナー反応（割り込み＋注入）が起きる."""
    state = FakeState(FakeAgent(), None)
    p = FakePartner()
    p.ai_speaking = True
    state.partner = p
    texts: list = []
    state.fac_events.put(("utterance", "本題に戻しましょう"))
    _run_event_worker_briefly(state, texts.append, until=lambda: bool(texts))
    assert texts == ["本題に戻しましょう"]
    assert p.interrupts == 1
    assert p.injected and p.injected[0][1] == "本題に戻しましょう"


def test_event_worker_noop_utterance_does_not_react_partner():
    """「介入不要」発言では議事録には残すがパートナーは止めない."""
    state = FakeState(FakeAgent(), None)
    p = FakePartner()
    p.ai_speaking = True
    state.partner = p
    texts: list = []
    state.fac_events.put(("utterance", "（介入不要）"))
    _run_event_worker_briefly(state, texts.append, until=lambda: bool(texts))
    assert texts == ["（介入不要）"]
    assert p.interrupts == 0
    assert p.injected == []


def test_event_worker_speech_start_interrupts_partner():
    """speech_startイベントで、発話中のパートナーを割り込む."""
    state = FakeState(FakeAgent(), None)
    p = FakePartner()
    p._responding = True
    state.partner = p
    state.fac_events.put(("speech_start", None))
    _run_event_worker_briefly(state, lambda t: None, until=lambda: p.interrupts > 0)
    assert p.interrupts == 1


# ---------------------------------------------------------------------------
# 整理介入の価値判定チェッカー（C3）: count の無条件介入を置換
# ---------------------------------------------------------------------------

def _run_structuring_briefly(state, *, until, timeout=3.5):
    from das.asr.live._workers import _run_structuring_checker
    t = threading.Thread(target=_run_structuring_checker,
                         args=(state, "key", "model"), daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not until():
        time.sleep(0.05)
    state.stop.set()
    t.join(timeout=2.0)


def _at_threshold_state():
    agent = FakeAgent()
    agent._pending = [{"speaker": "人間", "text": str(i)}
                      for i in range(agent.trigger_n)]
    state = FakeState(agent, None)
    state.records = [{"speaker": "A", "text": "本題の議論"}]
    return state


def test_structuring_checker_skips_when_no_value(monkeypatch):
    """pending_count が閾値に達しても、価値なし判定なら summarize を積まない."""
    import das.asr.live._bootstrap as bs
    monkeypatch.setattr(bs, "check_summary_value",
                        lambda *a, **k: {"intervene": False, "focus": ""})
    state = _at_threshold_state()

    _run_structuring_briefly(state, until=lambda: False, timeout=2.5)

    assert state.summarize_requests.empty()


def test_structuring_checker_enqueues_when_valuable(monkeypatch):
    """価値あり判定なら focus 付きで summarize_requests に積む."""
    import das.asr.live._bootstrap as bs
    monkeypatch.setattr(bs, "check_summary_value",
                        lambda *a, **k: {"intervene": True, "focus": "論点整理"})
    state = _at_threshold_state()

    _run_structuring_briefly(
        state, until=lambda: not state.summarize_requests.empty())

    req = state.summarize_requests.get_nowait()
    assert req["focus"] == "論点整理"


def test_structuring_checker_does_not_rejudge_same_count(monkeypatch):
    """同じ pending_count では LLM 判定を繰り返さない（_last_judged_count）."""
    import das.asr.live._bootstrap as bs
    calls: list = []

    def _fake(*a, **k):
        calls.append(1)
        return {"intervene": False, "focus": ""}

    monkeypatch.setattr(bs, "check_summary_value", _fake)
    state = _at_threshold_state()

    _run_structuring_briefly(state, until=lambda: False, timeout=3.0)

    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 未確定話者の割り込み（C1）: 声紋が確定しない発話でもAIを止められる
# ---------------------------------------------------------------------------

def test_unconfirmed_speaker_interrupts_ai():
    """speaker='?'（未確定）の長い発話でも、AI発話中なら interrupt される."""
    agent = FakeAgent()
    agent.ai_speaking = True
    state = FakeState(agent, None)
    state.records = [{"speaker": "?", "text": "ちょっと待ってほしいのですが"}]

    _run_worker_briefly(state, until=lambda: agent.interrupts > 0)

    assert agent.interrupts == 1


def test_unconfirmed_speaker_not_fed_to_agent():
    """未確定話者の発話は割り込み判定には使うが、agent.feed には流さない."""
    agent = FakeAgent()
    agent.ai_speaking = True
    state = FakeState(agent, None)
    state.records = [{"speaker": "?", "text": "ちょっと待ってほしいのですが"}]

    _run_worker_briefly(state, until=lambda: agent.interrupts > 0)

    assert agent.feeds == []
