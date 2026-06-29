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
    assert LiveArgs().proactivity == "controlled"
    assert LiveArgs(proactivity="standard").proactivity == "standard"


def test_cli_has_proactivity_option():
    from das.asr.live import main
    for p in main.params:
        if p.name == "proactivity":
            assert set(p.type.choices) == {"controlled", "standard", "active"}
            break
    else:
        raise AssertionError("proactivity option not found")


def test_proactivity_profiles():
    from das.asr.live._constants import _PROACTIVITY_DEFAULT, _PROACTIVITY_PROFILES
    assert _PROACTIVITY_DEFAULT == "controlled"
    assert _PROACTIVITY_PROFILES["controlled"]["silence_summarize"] is None
    assert _PROACTIVITY_PROFILES["controlled"]["drift_confirmations"] >= 2
    assert _PROACTIVITY_PROFILES["standard"]["silence_summarize"] == 18.0
    # 控えめ寄り: 標準でも以前の5秒よりかなり長い
    assert _PROACTIVITY_PROFILES["standard"]["silence_summarize"] > 10
    assert _PROACTIVITY_PROFILES["active"]["cooldown"] < \
        _PROACTIVITY_PROFILES["standard"]["cooldown"]


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

def test_fact_candidate_gate_is_only_a_thin_exclusion_filter():
    """明確な誤りかどうかはLLMに任せ、ローカルでは不要発話だけを落とす."""
    from das.asr.live._workers import _looks_like_fact_claim

    positives = [
        "指標Xの計算式は分母を分子で割る感じです",
        "なんか、量Aの2乗を量Bで割ると、どうのこうのみたいな",
        "量Bを量Aの2乗で割るんですよね",
        "指標Yの式は量Aを量Bで割るんですよね",
        "この用語の定義は、対象者が申請できる制度という意味です",
        "制度Xの上限は70%です",
        "事物Aの高さは200メートルです",
        "指標Y、いや、事物Aの高さは200メートルです",
        "都市Aは国Bの首都です",
        "都市Aは国Bの首都ではありません",
        "国Bの首都は都市Aです",
        "人物Aは組織BのCEOです",
        "人物Aは国B人です",
        "人物Aは国B出身です",
        "人物Aは大会Bを2勝しました",
        "作品Aの作者は人物Bです",
        "対象Aは世界で2番目に高い山です",
        "対象Aは世界で2番目に深い場所です",
        "対象Aは国内で3番目に大きい湖です",
        "対象Aは世界で2番目に人口が多いです",
        "人物Aはランキング1位です",
        "人物AはリーグCでトップ10です",
        "対象Aは世界一大きい島です",
        "対象Aを世界一高い山です",
        "対象Aは日本一長い川です",
        "対象Aは国内一広い湖です",
        "時代Aの次は時代Bです",
        "国Aは地域Bに属しています",
        "x = y / z",
        "単位はメートルです",
        "2乗が出てきました",
        "さっきは6回ほどかかりました",
        "最初の10個の発話ぐらいから",
        "式典は普通にやりました",
        "まず対象Aを1個置きます",
    ]
    negatives = [
        "平均について話しましょう",
        "123cmだとどれくらいですか",
        "対象Aは、123cmだとどれぐらいですか",
        "計算方法の話です",
        "米よりパンのほうが好きです",
        "いいとは思いますけど",
        "090-8165-1145 にかけてもいい？",
        "お酒はどこで飲むの、2回目のデート",
        "都市Aはきれいです",
        "人物Aはいい人です",
        "1つ目の論点を確認しましょう",
        "優先順位を決めましょう",
        "ランキングについて話しましょう",
        "対象Aは上位かもしれません",
    ]

    assert [_looks_like_fact_claim(t) for t in positives] == [True] * len(positives)
    assert [_looks_like_fact_claim(t) for t in negatives] == [False] * len(negatives)


def test_fact_candidate_gate_does_not_require_fact_pattern_keywords():
    """助詞や既知カテゴリに依存せず、断定発話はLLMに渡す."""
    from das.asr.live._workers import _looks_like_fact_claim

    texts = [
        "事物Aの高さは200メートルです",
        "対象Aを世界一高い山です",
        "時代Aの次は時代Bです",
        "国Aは地域Bに属しています",
        "対象Aは世界で2番目に人口が多いです",
        "人物Aは大会Bのランキング1位です",
        "これは未知カテゴリの断定文です",
    ]

    assert [_looks_like_fact_claim(t) for t in texts] == [True] * len(texts)


def test_fact_candidate_gate_ignores_low_value_utterances():
    """相槌・質問・曖昧・好み・会話運営だけを事前除外する."""
    from das.asr.live._workers import _looks_like_fact_claim

    texts = [
        "ランキングについて話しましょう",
        "優先順位を決めましょう",
        "1つ目の論点を確認しましょう",
        "トップになるにはどうすればいいですか",
        "対象Aは上位かもしれません",
        "順位は何位ですか",
        "計算方法の話です",
        "都市Aはきれいです",
        "人物Aはいい人です",
    ]

    for text in texts:
        assert not _looks_like_fact_claim(text), text


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
        {"speaker": "話者1", "text": "計算方法の話です", "ms": 0, "end_ms": 1000},
        {"speaker": "話者2", "text": "指標Xの計算式は分母を分子で割る感じです", "ms": 1000, "end_ms": 2000},
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
        {"speaker": "話者1", "text": "平均について話しましょう", "ms": 0, "end_ms": 1000},
        {"speaker": "話者1", "text": "計算方法の話です", "ms": 1000, "end_ms": 2000},
        {"speaker": "話者1", "text": "優先順位を決めましょう", "ms": 2000, "end_ms": 3000},
        {"speaker": "話者1", "text": "ランキングについて話しましょう", "ms": 3000, "end_ms": 4000},
        {"speaker": "話者1", "text": "対象の値は100です", "ms": 4000, "end_ms": 5000},
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


def test_fact_checker_ignores_plain_opinion(monkeypatch):
    """好みや単なる意見ではLLM判定すら呼ばない."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_fact_checker

    calls = []
    monkeypatch.setattr(bootstrap, "check_fact_correction",
                        lambda *a, **k: calls.append(1) or {"should_correct": False})
    state = _make_state(with_agent=True)
    state.records = [
        {"speaker": "話者1", "text": "米よりパンのほうが好きです", "ms": 0, "end_ms": 1000},
    ]

    th = threading.Thread(target=_run_fact_checker,
                          args=(state, "key", "gpt-5-mini"), daemon=True)
    th.start()
    time.sleep(1.2)
    state.stop.set()
    th.join(timeout=2)

    assert calls == []
    assert state.factcheck_requests.empty()


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
