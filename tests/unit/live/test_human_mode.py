"""人間同士ファシリテーションモード（S1/S3）の配線テスト."""
from __future__ import annotations

import datetime
import threading
import time

from das.asr.live._bootstrap import LiveArgs


def test_liveargs_has_topic_field():
    assert LiveArgs().topic is None
    assert LiveArgs(topic="AI導入の是非").topic == "AI導入の是非"


def test_cli_has_topic_option():
    from das.asr.live import main
    names = {p.name for p in main.params}
    assert "topic" in names


def test_liveargs_has_proactivity():
    assert LiveArgs().proactivity == "standard"
    assert LiveArgs(proactivity="standard").proactivity == "standard"


def test_proactivity_default_survives_cli_option_removal():
    """積極性の既定は standard（CLIオプションは削除済み・変更はUIから）."""
    from das.asr.live._bootstrap import LiveArgs
    assert LiveArgs().proactivity == "standard"


def test_proactivity_profiles():
    from das.asr.live._constants import _PROACTIVITY_DEFAULT, _PROACTIVITY_PROFILES
    assert _PROACTIVITY_DEFAULT == "standard"
    assert _PROACTIVITY_PROFILES["controlled"]["silence_summarize"] is None
    assert _PROACTIVITY_PROFILES["controlled"]["drift_confirmations"] >= 2
    assert _PROACTIVITY_PROFILES["standard"]["silence_summarize"] == 18.0
    # 控えめ寄り: 標準でも以前の5秒よりかなり長い
    assert _PROACTIVITY_PROFILES["standard"]["silence_summarize"] > 10
    assert _PROACTIVITY_PROFILES["active"]["cooldown"] < \
        _PROACTIVITY_PROFILES["standard"]["cooldown"]
    # 旧 stall_breaker は Phase3 で廃止。プロファイルにキーを持たない。
    assert all("stall_breaker" not in p for p in _PROACTIVITY_PROFILES.values())


def test_cli_has_imbalanced_scenario():
    """一人で声かけ(invite)を試せる imbalanced シナリオが選択肢にある."""
    from das.asr.live import main
    for p in main.params:
        if p.name == "sim_scenario":
            assert "imbalanced" in p.type.choices
            break
    else:
        raise AssertionError("sim_scenario option not found")


def test_agenda_precedence_prefers_topic():
    """議題シードの優先順位は topic > debate > simulate."""
    # run_session の該当ロジックと同じ式を検証（args.topic or args.debate or args.simulate）
    a = LiveArgs(topic="T", debate="D", simulate="S")
    assert (a.topic or a.debate or a.simulate) == "T"
    b = LiveArgs(topic=None, debate="D", simulate="S")
    assert (b.topic or b.debate or b.simulate) == "D"
    c = LiveArgs(topic=None, debate=None, simulate="S")
    assert (c.topic or c.debate or c.simulate) == "S"
    d = LiveArgs()
    assert (d.topic or d.debate or d.simulate) is None


# ---------------------------------------------------------------------------
# S3: 冒頭アジェンダ自動検出
# ---------------------------------------------------------------------------

def test_detect_agenda_parses_agenda(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json",
                        lambda *a, **k: {"agenda": "AI導入の是非"})
    assert bootstrap.detect_agenda([{"speaker": "A", "text": "x"}],
                                   "key", "m") == "AI導入の是非"


def test_detect_agenda_returns_none(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json", lambda *a, **k: {"agenda": None})
    assert bootstrap.detect_agenda([{"speaker": "A", "text": "x"}], "key", "m") is None
    monkeypatch.setattr(bootstrap, "_post_chat_json", lambda *a, **k: None)
    assert bootstrap.detect_agenda([{"speaker": "A", "text": "x"}], "key", "m") is None


class _FakeAgent:
    """participation_checker が参照する最小限のエージェント."""
    enabled = True
    mode = "facilitator"


def _make_state(*, with_agent: bool = False):
    from das.asr.live._session_state import SessionState
    s = SessionState(
        args=object(), started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md", html_path="/tmp/o.html", diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns", wav_path="/tmp/o.wav",
    )
    if with_agent:
        s.agent = _FakeAgent()  # type: ignore[assignment]
    return s


def test_agenda_detector_seeds_when_detected(monkeypatch):
    """十分な発話がたまると、検出した議題をseedして停止する（S3）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_agenda_detector

    monkeypatch.setattr(bootstrap, "detect_agenda", lambda *a, **k: "AI導入の是非")
    state = _make_state()
    state.records = [{"speaker": "話者1", "text": f"発話{i}",
                      "ms": i * 1000, "end_ms": i * 1000 + 500} for i in range(4)]

    t = threading.Thread(target=_run_agenda_detector,
                         args=(state, "key", "gpt-5-mini"), daemon=True)
    t.start()
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not state.topics:
        time.sleep(0.05)
    state.stop.set()
    t.join(timeout=2)

    assert [tp["topic"] for tp in state.topics] == ["AI導入の是非"]


# ---------------------------------------------------------------------------
# S4: 参加度の声かけ
# ---------------------------------------------------------------------------

def test_check_participation_parses(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json",
                        lambda *a, **k: {"invite": True, "speaker": "参加者B",
                                         "reason": "発言が少ない"})
    r = bootstrap.check_participation(
        [{"speaker": "参加者B", "time_share": 0.1, "turns": 1, "silent_sec": 60}],
        [], "key", "m")
    assert r["invite"] is True
    assert r["speaker"] == "参加者B"


def test_check_participation_empty_is_no_invite():
    import das.asr.live._bootstrap as bootstrap
    assert bootstrap.check_participation([], [], "key", "m") == {"invite": False}


# ---------------------------------------------------------------------------
# 事実誤りの短い補正
# ---------------------------------------------------------------------------

def test_classify_utterance_parses(monkeypatch):
    """triage分類のLLM応答（fact候補/呼びかけ依頼）を構造化して返す."""
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json",
                        lambda *a, **k: {"factual_claim": True,
                                         "facilitator_request": "ここまでの整理"})
    r = bootstrap.classify_utterance(
        [{"speaker": "参加者A", "text": "AIさん、ここまで整理して"}], "key", "m")
    assert r == {"factual_claim": True, "facilitator_request": "ここまでの整理"}


def test_classify_utterance_marks_parse_failure_retryable(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json", lambda *a, **k: None)
    r = bootstrap.classify_utterance(
        [{"speaker": "参加者A", "text": "国Bの首都は都市Aです"}], "key", "m")
    assert r["retryable_error"] is True
    assert r["factual_claim"] is False


def test_classify_utterance_without_key_is_negative():
    import das.asr.live._bootstrap as bootstrap
    r = bootstrap.classify_utterance(
        [{"speaker": "参加者A", "text": "国Bの首都は都市Aです"}], "", "m")
    assert r == {"factual_claim": False, "facilitator_request": ""}


def _run_triage_briefly(state, *, until, timeout=3.0):
    from das.asr.live._workers import _run_triage_worker
    th = threading.Thread(target=_run_triage_worker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not until():
        time.sleep(0.05)
    state.stop.set()
    th.join(timeout=2)


def test_triage_worker_annotates_records_and_advances_cursor(monkeypatch):
    """確定発話ごとに1回だけLLM分類し、record に triage 注釈を付ける."""
    import das.asr.live._bootstrap as bootstrap

    calls = []

    def _fake_classify(utts, *_args):
        calls.append(utts)
        return {"factual_claim": utts[-1]["text"].endswith("です"),
                "facilitator_request": ""}

    monkeypatch.setattr(bootstrap, "classify_utterance", _fake_classify)
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "計算方法の話です", "ms": 0, "end_ms": 1000},
        {"speaker": "話者2", "text": "米よりパンのほうが好き", "ms": 1000, "end_ms": 2000},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 2)

    assert state.triage_cursor == 2
    assert state.records[0]["triage"] == {"factual_claim": True,
                                          "facilitator_request": ""}
    assert state.records[1]["triage"] == {"factual_claim": False,
                                          "facilitator_request": ""}
    assert len(calls) == 2


def test_triage_worker_discards_and_recovers_on_midflight_reset(monkeypatch):
    """分類中に会議リセット(meeting_epoch進行)が起きたらその結果を破棄し、
    以降の発話は正しく処理される (H2). リセット競合でカーソルが暴走せず、
    triage が止まって fact/呼びかけが連鎖停止するのを防ぐ。"""
    import das.asr.live._bootstrap as bootstrap

    calls: list = []

    def _fake_classify(utts, *_args):
        calls.append(utts)
        if len(calls) == 1:
            # スナップショット取得後・注釈書き戻し前にリセットが起きた状況を模す
            state.meeting_epoch += 1
        return {"factual_claim": False, "facilitator_request": ""}

    monkeypatch.setattr(bootstrap, "classify_utterance", _fake_classify)
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "計算方法の話です", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(
        state, until=lambda: state.records[0].get("triage") is not None
    )

    # 1回目の結果は epoch 不一致で破棄され、カーソルは進まない。
    # 2回目（epoch 安定後）で注釈が付きカーソルが進む。
    assert state.records[0]["triage"] == {"factual_claim": False,
                                          "facilitator_request": ""}
    assert state.triage_cursor == 1
    assert len(calls) == 2


def test_triage_worker_passes_recent_context_before_target(monkeypatch):
    """指示語・省略の補完のため、直前の発話を参照文脈として渡す."""
    import das.asr.live._bootstrap as bootstrap

    calls = []
    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda utts, *_a: calls.append(utts) or
                        {"factual_claim": False, "facilitator_request": ""})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "富士山の話をしましょう", "ms": 0, "end_ms": 1000},
        {"speaker": "話者2", "text": "高さは3000メートルです", "ms": 1000, "end_ms": 2000},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 2)

    assert calls[1] == [
        {"speaker": "参加者A", "text": "富士山の話をしましょう"},
        {"speaker": "参加者B", "text": "高さは3000メートルです"},
    ]


def test_triage_worker_local_gate_skips_very_short_utterances(monkeypatch):
    """極端に短い発話はLLMを呼ばず負の注釈を付ける（コスト0のゲート）.

    相槌・未確定は intervention_records が除外するため triage には届かない。
    """
    import das.asr.live._bootstrap as bootstrap

    calls = []
    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: calls.append(1) or
                        {"factual_claim": False, "facilitator_request": ""})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "5だ", "ms": 0, "end_ms": 500},
        {"speaker": "話者2", "text": "多分", "ms": 500, "end_ms": 900},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 2)

    assert calls == []
    assert state.triage_cursor == 2
    assert state.records[0]["triage"]["factual_claim"] is False
    assert state.records[1]["triage"]["factual_claim"] is False


def test_triage_worker_enqueues_facilitator_request(monkeypatch):
    """呼びかけ依頼を検出したら手動呼び出しキュー（source=voice）に積み、アック音を鳴らす."""
    import das.asr.live._bootstrap as bootstrap
    import das.asr.live._workers as workers

    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: {"factual_claim": False,
                                         "facilitator_request": "ここまでの整理"})
    chimes: list = []
    monkeypatch.setattr(workers, "_play_ack_chime", lambda: chimes.append(True))
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "AIさん、ここまで整理して", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(
        state, until=lambda: not state.manual_call_requests.empty())

    got = state.manual_call_requests.get_nowait()
    assert got["request"] == "ここまでの整理"
    assert got["source"] == "voice"
    assert chimes == [True]   # voice 呼びかけでアック音が鳴る


def test_triage_worker_topic_mention_does_not_chime(monkeypatch):
    """依頼なし（話題としての言及）ではアック音を鳴らさない."""
    import das.asr.live._bootstrap as bootstrap
    import das.asr.live._workers as workers

    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: {"factual_claim": False,
                                         "facilitator_request": ""})
    chimes: list = []
    monkeypatch.setattr(workers, "_play_ack_chime", lambda: chimes.append(True))
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "AIの導入について話しましょう", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 1)

    assert chimes == []


def test_play_ack_chime_never_raises():
    """再生に失敗しても（sounddevice不在など）例外を漏らさない."""
    from das.asr.live._workers import _play_ack_chime

    _play_ack_chime()   # 例外が上がらなければ合格（音は必須機能ではない）


def test_triage_worker_topic_mention_does_not_enqueue(monkeypatch):
    """AIを話題として言及しただけ（依頼なし）では呼び出しキューに積まない."""
    import das.asr.live._bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: {"factual_claim": False,
                                         "facilitator_request": ""})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "AIの導入について話しましょう", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 1)

    assert state.manual_call_requests.empty()
    assert state.triage_cursor == 1


def test_triage_worker_disabled_in_conversation_mode(monkeypatch):
    """conversation モードでは分類しない（fact/呼びかけ経路ごと停止, 二重応答回避）."""
    import das.asr.live._bootstrap as bootstrap

    calls = []
    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: calls.append(1) or
                        {"factual_claim": False, "facilitator_request": ""})
    state = _make_state(with_agent=True)
    state.agent.mode = "conversation"
    state.records = [
        {"speaker": "話者1", "text": "AIさん、ここまで整理して", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(state, until=lambda: False, timeout=1.0)

    assert calls == []
    assert state.manual_call_requests.empty()


def test_triage_worker_skips_when_intervention_disabled(monkeypatch):
    """介入オフでは分類も呼びかけ検出もしない."""
    import das.asr.live._bootstrap as bootstrap

    calls = []
    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: calls.append(1) or
                        {"factual_claim": False, "facilitator_request": ""})
    state = _make_state(with_agent=True)
    state.intervention_enabled = False
    state.records = [
        {"speaker": "話者1", "text": "AIさん、ここまで整理して", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(state, until=lambda: False, timeout=1.0)

    assert calls == []
    assert state.manual_call_requests.empty()


def test_triage_worker_retries_retryable_failure_before_advancing(monkeypatch):
    """LLM/API一時失敗では同じ発話を再試行し、成功したら注釈を付ける."""
    import das.asr.live._bootstrap as bootstrap

    calls = []

    def _fake_classify(utts, *_args):
        calls.append(utts)
        if len(calls) == 1:
            return {"factual_claim": False, "facilitator_request": "",
                    "retryable_error": True}
        return {"factual_claim": True, "facilitator_request": ""}

    monkeypatch.setattr(bootstrap, "classify_utterance", _fake_classify)
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "国Bの首都は都市Aです", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 1)

    assert len(calls) >= 2
    assert state.records[0]["triage"]["factual_claim"] is True
    assert state.triage_cursor == 1


def test_triage_worker_off_bulk_skips_then_resumes(monkeypatch):
    """介入オフ中の発話は LLM を呼ばず skipped=intervention_off で一括負注釈し、
    カーソルを最新まで進める。再有効化後は新規発話だけが通常分類され、オフ中に
    溜まった過去発話の呼びかけを誤発火させない（問題1）."""
    import das.asr.live._bootstrap as bootstrap

    calls: list = []
    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda utts, *_a: calls.append(utts) or
                        {"factual_claim": False, "facilitator_request": ""})
    state = _make_state(with_agent=True)
    state.intervention_enabled = False
    state.records = [
        {"speaker": "話者1", "text": "AIさん、ここまで整理して", "ms": 0, "end_ms": 1000},
        {"speaker": "話者2", "text": "そうですね、お願いします", "ms": 1000, "end_ms": 2000},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 2, timeout=2.0)

    # オフ中は分類されず、負注釈でカーソルが進む。呼びかけも積まれない。
    assert calls == []
    assert all(r["triage"]["skipped"] == "intervention_off" for r in state.records)
    assert state.triage_cursor == 2
    assert state.manual_call_requests.empty()

    # 再有効化: 新規発話だけが通常分類される（古い2件は再分類しない）。
    state.stop.clear()  # 前段の _run_triage_briefly が立てた stop を戻す
    state.intervention_enabled = True
    state.records.append(
        {"speaker": "話者1", "text": "新しい論点を出します", "ms": 2000, "end_ms": 3000})

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 3, timeout=2.0)

    assert len(calls) == 1
    assert calls[0][-1]["text"] == "新しい論点を出します"
    assert "skipped" not in state.records[2]["triage"]


def test_triage_worker_conversation_mode_bulk_skips(monkeypatch):
    """conversation モードでも未処理分を skipped で負注釈しカーソルを進める."""
    import das.asr.live._bootstrap as bootstrap

    calls: list = []
    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: calls.append(1) or
                        {"factual_claim": False, "facilitator_request": ""})
    state = _make_state(with_agent=True)
    state.agent.mode = "conversation"
    state.records = [
        {"speaker": "話者1", "text": "AIさん、ここまで整理して", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 1, timeout=2.0)

    assert calls == []
    assert state.records[0]["triage"]["skipped"] == "intervention_off"
    assert state.triage_cursor == 1
    assert state.manual_call_requests.empty()


def test_triage_worker_backlog_skips_old_and_processes_recent(monkeypatch):
    """バックログが上限超過なら古い分を skipped=backlog で分類せず飛ばし、直近分を
    連続処理して最新発話の呼びかけを発火する（問題2, 遅延の有界化）.

    内側ループは各件の間で sleep しないため、この catch-up は 1 tick で
    ``_TRIAGE_BACKLOG_MAX`` 件をまとめて処理する（1件/tick の上限を撤廃）."""
    import das.asr.live._bootstrap as bootstrap
    import das.asr.live._workers as workers

    monkeypatch.setattr(workers, "_TRIAGE_BACKLOG_MAX", 3)
    classified: list = []

    def _fake_classify(utts, *_a):
        target = utts[-1]["text"]
        classified.append(target)
        req = "AIさん整理して" if target.endswith("9番です") else ""
        return {"factual_claim": False, "facilitator_request": req}

    monkeypatch.setattr(bootstrap, "classify_utterance", _fake_classify)
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": f"これは発話{i}番です",
         "ms": i * 1000, "end_ms": i * 1000 + 500}
        for i in range(10)
    ]

    _run_triage_briefly(state, until=lambda: state.triage_cursor >= 10, timeout=5.0)

    assert state.triage_cursor == 10
    # 古い 10-3=7 件は分類されず skipped=backlog、直近3件だけ分類される。
    assert classified == ["これは発話7番です", "これは発話8番です", "これは発話9番です"]
    assert all(state.records[i]["triage"].get("skipped") == "backlog"
               for i in range(7))
    assert all("skipped" not in state.records[i]["triage"] for i in range(7, 10))
    # 最新発話の呼びかけは発火する。
    assert not state.manual_call_requests.empty()


def test_triage_worker_picks_up_unconfirmed_speaker_call(monkeypatch):
    """未確定話者(?)の呼びかけも triage が拾い manual_call(source=voice) に積む（修正5）.

    声紋未登録の参加者が何度AIを呼んでも無視される問題を解消する。"""
    import das.asr.live._bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: {"factual_claim": False,
                                         "facilitator_request": "ここまで整理して"})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "?", "text": "AIさん、ここまで整理して", "ms": 0, "end_ms": 1000},
    ]

    _run_triage_briefly(
        state, until=lambda: not state.manual_call_requests.empty(), timeout=2.0)

    req = state.manual_call_requests.get_nowait()
    assert req["source"] == "voice"
    assert req["request"] == "ここまで整理して"


def test_triage_worker_unconfirmed_factual_claim_not_fact_checked(monkeypatch):
    """未確定話者の事実断定は triage で factual_claim=True が付いても、fact checker は
    intervention_records を使うため check_fact を呼ばない（帰属不明の断定に訂正を
    打たない, 修正5）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_fact_checker

    monkeypatch.setattr(bootstrap, "classify_utterance",
                        lambda *a, **k: {"factual_claim": True,
                                         "facilitator_request": ""})
    fact_calls: list = []
    monkeypatch.setattr(bootstrap, "check_fact_correction",
                        lambda *a, **k: fact_calls.append(1) or
                        {"should_correct": False})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "?", "text": "国Bの首都は都市Aです", "ms": 0, "end_ms": 1000},
    ]

    # triage 注釈が付くまで triage worker を動かす（factual_claim=True が付く）。
    _run_triage_briefly(
        state, until=lambda: state.records[0].get("triage") is not None, timeout=2.0)
    assert state.records[0]["triage"]["factual_claim"] is True

    # fact checker を動かしても、未確定話者は intervention_records から外れるため
    # 候補にならず check_fact は呼ばれない。fact_cursor も進まない。
    state.stop.clear()
    th = threading.Thread(target=_run_fact_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    time.sleep(1.0)
    state.stop.set()
    th.join(timeout=2)

    assert fact_calls == []
    assert state.fact_cursor == 0


def test_check_fact_correction_accepts_high_confidence(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json",
                        lambda *a, **k: {"should_correct": True,
                                         "confidence": "high",
                                         "claim": "指標Xの計算式は分母を分子で割る",
                                         "correction": "指標Xは分子を分母で割ります。",
                                         "reason": "式が逆"})

    r = bootstrap.check_fact_correction(
        [{"speaker": "A", "text": "指標Xの計算式は分母を分子で割る"}], "key", "m")

    assert r["should_correct"] is True
    assert r["correction"].startswith("指標Xは")


def test_check_fact_correction_suppresses_low_confidence(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json",
                        lambda *a, **k: {"should_correct": True,
                                         "confidence": "medium",
                                         "correction": "補足"})

    assert bootstrap.check_fact_correction(
        [{"speaker": "A", "text": "たぶんそうだった気がします"}], "key", "m"
    ) == {"should_correct": False}


def test_check_fact_correction_suppresses_style_advice(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json",
                        lambda *a, **k: {"should_correct": True,
                                         "confidence": "high",
                                         "claim": "弾丸は襲いかかる",
                                         "correction": "弾丸は意思を持たないので、「飛んでいく」と表現するほうが正確です。",
                                         "reason": "表現の正確さ"})

    assert bootstrap.check_fact_correction(
        [{"speaker": "A", "text": "その弾丸は奴をめがけて襲いかかる。"}], "key", "m"
    ) == {"should_correct": False}


def test_check_fact_correction_marks_parse_failure_retryable(monkeypatch):
    import das.asr.live._bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_post_chat_json", lambda *a, **k: None)

    assert bootstrap.check_fact_correction(
        [{"speaker": "A", "text": "対象の値は100です"}], "key", "m"
    ) == {"should_correct": False, "retryable_error": True}


def test_check_fact_correction_marks_only_last_utterance_as_target(monkeypatch):
    import das.asr.live._bootstrap as bootstrap

    prompts = []

    def _fake_post(params, *_args, **_kwargs):
        prompts.append(params["messages"][0]["content"])
        return {"should_correct": False}

    monkeypatch.setattr(bootstrap, "_post_chat_json", _fake_post)

    bootstrap.check_fact_correction([
        {"speaker": "A", "text": "対象Aについて話しています"},
        {"speaker": "B", "text": "世界一高い山です"},
    ], "key", "m")

    assert "- [参照] A: 対象Aについて話しています" in prompts[0]
    assert "- [判定対象] B: 世界一高い山です" in prompts[0]


def test_fact_checker_enqueues_clear_formula_correction(monkeypatch):
    """式・定義っぽい発話だけを候補にし、高確信の補正をキューに積む."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_fact_checker

    calls = []

    def _fake_fact(utts, *_args):
        calls.append(utts)
        return {"should_correct": True,
                "confidence": "high",
                "claim": "指標Xの計算式は分母を分子で割る",
                "correction": "指標Xは分子を分母で割ります。",
                "reason": "式が逆"}

    monkeypatch.setattr(bootstrap, "check_fact_correction", _fake_fact)
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "計算方法の話です", "ms": 0, "end_ms": 1000,
         "triage": {"factual_claim": False, "facilitator_request": ""}},
        {"speaker": "話者2", "text": "指標Xの計算式は分母を分子で割る感じです",
         "ms": 1000, "end_ms": 2000, "triage": {"factual_claim": True, "facilitator_request": ""}},
    ]

    th = threading.Thread(target=_run_fact_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    got = None
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and got is None:
        try:
            got = state.factcheck_requests.get_nowait()
        except Exception:
            time.sleep(0.05)
    state.stop.set()
    th.join(timeout=2)

    assert got is not None
    assert got["correction"].startswith("指標Xは")
    assert calls == [[
        {"speaker": "参加者A", "text": "計算方法の話です"},
        {"speaker": "参加者B", "text": "指標Xの計算式は分母を分子で割る感じです"},
    ]]


def test_fact_checker_passes_recent_context_before_target(monkeypatch):
    """直前3発話は参照として渡し、判定対象は最後に置く."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_fact_checker

    calls = []

    def _fake_fact(utts, *_args):
        calls.append(utts)
        return {"should_correct": False}

    monkeypatch.setattr(bootstrap, "check_fact_correction", _fake_fact)
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "平均について話しましょう", "ms": 0, "end_ms": 1000,
         "triage": {"factual_claim": False, "facilitator_request": ""}},
        {"speaker": "話者1", "text": "計算方法の話です", "ms": 1000, "end_ms": 2000,
         "triage": {"factual_claim": False, "facilitator_request": ""}},
        {"speaker": "話者1", "text": "優先順位を決めましょう", "ms": 2000, "end_ms": 3000,
         "triage": {"factual_claim": False, "facilitator_request": ""}},
        {"speaker": "話者1", "text": "ランキングについて話しましょう", "ms": 3000,
         "end_ms": 4000, "triage": {"factual_claim": False, "facilitator_request": ""}},
        {"speaker": "話者1", "text": "対象の値は100です", "ms": 4000, "end_ms": 5000,
         "triage": {"factual_claim": True, "facilitator_request": ""}},
    ]

    th = threading.Thread(target=_run_fact_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not calls:
        time.sleep(0.05)
    state.stop.set()
    th.join(timeout=2)

    assert calls == [[
        {"speaker": "参加者A", "text": "計算方法の話です"},
        {"speaker": "参加者A", "text": "優先順位を決めましょう"},
        {"speaker": "参加者A", "text": "ランキングについて話しましょう"},
        {"speaker": "参加者A", "text": "対象の値は100です"},
    ]]


def test_fact_checker_retries_retryable_failure_before_advancing(monkeypatch):
    """LLM/API一時失敗では同じ発話を再試行し、成功したらキューに積む."""
    import das.asr.live._bootstrap as bootstrap
    import das.asr.live._workers as workers
    from das.asr.live._workers import _run_fact_checker

    monkeypatch.setattr(workers, "_FACTCHECK_CHECK_SEC", 0.1)
    calls = []

    def _fake_fact(utts, *_args):
        calls.append(utts)
        if len(calls) == 1:
            return {"should_correct": False, "retryable_error": True}
        return {"should_correct": True,
                "confidence": "high",
                "claim": "対象の値は100",
                "correction": "対象の値は200です。",
                "reason": "値が違う"}

    monkeypatch.setattr(bootstrap, "check_fact_correction", _fake_fact)
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "対象の値は100です", "ms": 0, "end_ms": 1000,
         "triage": {"factual_claim": True, "facilitator_request": ""}},
    ]

    th = threading.Thread(target=_run_fact_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    got = None
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and got is None:
        try:
            got = state.factcheck_requests.get_nowait()
        except Exception:
            time.sleep(0.05)
    state.stop.set()
    th.join(timeout=2)

    assert got is not None
    assert got["correction"] == "対象の値は200です。"
    assert len(calls) >= 2
    assert calls[0] == calls[1]
    assert state.fact_cursor == 1


def test_fact_checker_ignores_plain_opinion(monkeypatch):
    """triage が fact候補でないと判定した発話ではLLM判定すら呼ばない."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_fact_checker

    calls = []
    monkeypatch.setattr(bootstrap, "check_fact_correction",
                        lambda *a, **k: calls.append(1) or {"should_correct": False})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "米よりパンのほうが好きです", "ms": 0, "end_ms": 1000,
         "triage": {"factual_claim": False, "facilitator_request": ""}},
    ]

    th = threading.Thread(target=_run_fact_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    time.sleep(1.2)
    state.stop.set()
    th.join(timeout=2)

    assert calls == []
    assert state.factcheck_requests.empty()


def test_fact_checker_waits_for_unclassified_records(monkeypatch):
    """triage 注釈がまだ無い発話は消費せず、分類を待つ（カーソルを進めない）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_fact_checker

    calls = []
    monkeypatch.setattr(bootstrap, "check_fact_correction",
                        lambda *a, **k: calls.append(1) or {"should_correct": False})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "国Bの首都は都市Aです", "ms": 0, "end_ms": 1000},
    ]

    th = threading.Thread(target=_run_fact_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    time.sleep(1.0)
    state.stop.set()
    th.join(timeout=2)

    assert calls == []
    assert state.fact_cursor == 0


def test_participation_checker_enqueues_invite(monkeypatch):
    """発言の少ない人がいる時、LLM判定YESで声かけ要求をキューに積む（S4）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_participation_checker

    monkeypatch.setattr(bootstrap, "check_participation",
                        lambda *a, **k: {"invite": True, "speaker": "参加者B",
                                         "reason": "静か"})
    state = _make_state(with_agent=True)
    recs = []
    t = 0
    for i in range(8):   # 話者1 はよく話す（warmup超え）
        recs.append({"speaker": "話者1", "text": f"a{i}", "ms": t, "end_ms": t + 2000})
        t += 2000
    recs.append({"speaker": "話者2", "text": "少し補足があります", "ms": t, "end_ms": t + 200})
    state.records = recs

    th = threading.Thread(target=_run_participation_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    got = None
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and got is None:
        try:
            got = state.invite_requests.get_nowait()
        except Exception:
            time.sleep(0.05)
    state.stop.set()
    th.join(timeout=2)

    assert got == "参加者B"


def test_participation_checker_rejects_unreliable_invite_target(monkeypatch):
    """LLMが参加度表にない低信頼名を返しても声かけ要求を積まない."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_participation_checker

    monkeypatch.setattr(bootstrap, "check_participation",
                        lambda *a, **k: {"invite": True, "speaker": "発話者",
                                         "reason": "静か"})
    state = _make_state(with_agent=True)
    recs = []
    t = 0
    for i in range(8):
        recs.append({"speaker": "人物1", "text": f"a{i}", "ms": t, "end_ms": t + 2000})
        t += 2000
    recs.append({"speaker": "人物2", "text": "はい", "ms": t, "end_ms": t + 200})
    state.records = recs

    th = threading.Thread(target=_run_participation_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    time.sleep(2.5)
    state.stop.set()
    th.join(timeout=2)

    assert state.invite_requests.empty()


def test_participation_checker_skips_when_balanced(monkeypatch):
    """発話量が均衡していれば（事前ゲート）LLMを呼ばず声かけしない（S4）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_participation_checker

    calls = []
    monkeypatch.setattr(bootstrap, "check_participation",
                        lambda *a, **k: calls.append(1) or {"invite": False})
    state = _make_state(with_agent=True)
    recs = []
    t = 0
    for i in range(8):  # 話者1/話者2 が交互に同程度
        sp = "話者1" if i % 2 == 0 else "話者2"
        recs.append({"speaker": sp, "text": f"u{i}", "ms": t, "end_ms": t + 1000})
        t += 1000
    state.records = recs

    th = threading.Thread(target=_run_participation_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    time.sleep(2.5)
    state.stop.set()
    th.join(timeout=2)

    assert calls == []  # 均衡 → 事前ゲートでLLM未呼び出し


def test_agenda_detector_skips_when_topics_exist(monkeypatch):
    """既に論点があれば議題検出は走らない（S3）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_agenda_detector

    calls = []
    monkeypatch.setattr(bootstrap, "detect_agenda",
                        lambda *a, **k: calls.append(1) or "X")
    state = _make_state()
    state.topics = [{"topic": "既存論点", "speaker": "話者1"}]
    state.records = [{"speaker": "話者1", "text": f"u{i}",
                      "ms": i * 1000, "end_ms": i * 1000 + 500} for i in range(5)]

    t = threading.Thread(target=_run_agenda_detector,
                         args=(state, "key", "gpt-5-mini"), daemon=True)
    t.start()
    time.sleep(2.5)
    state.stop.set()
    t.join(timeout=2)

    assert calls == []
    assert [tp["topic"] for tp in state.topics] == ["既存論点"]
