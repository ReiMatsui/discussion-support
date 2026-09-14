"""DiscussionSimulator._parse_turn の話者分離テスト."""
from __future__ import annotations

from das.asr.live.agents._simulator import DiscussionSimulator


def _sim():
    return DiscussionSimulator(api_key="x", topic="テーマ")


def test_single_speaker_line():
    sp, utt = _sim()._parse_turn("参加者A: コストを議論しましょう。")
    assert sp == "参加者A"
    assert utt == "コストを議論しましょう。"


def test_fullwidth_colon():
    sp, utt = _sim()._parse_turn("参加者B：賛成です。")
    assert sp == "参加者B"
    assert utt == "賛成です。"


def test_multi_speaker_takes_first_only():
    """複数話者が混ざった応答でも、最初の1人だけを採用する（声の分離維持）."""
    text = "参加者A: メリットは効率化です。\n参加者B: でもコストが心配です。\n参加者C: 確かに。"
    sp, utt = _sim()._parse_turn(text)
    assert sp == "参加者A"
    assert utt == "メリットは効率化です。"
    assert "参加者B" not in utt  # 2人目以降は取り込まない


def test_unknown_speaker_is_rejected():
    assert _sim()._parse_turn("不明: なにか") == (None, None)


def test_non_speaker_format_rejected():
    assert _sim()._parse_turn("ただのテキスト") == (None, None)


def test_leading_blank_lines_skipped():
    sp, utt = _sim()._parse_turn("\n\n参加者C: そうですね。")
    assert sp == "参加者C"
    assert utt == "そうですね。"


# --- 音響リハーサル（--sim-acoustic）: 声はスピーカーだけ、入力はマイクから ----------


def test_acoustic_mode_plays_but_does_not_feed_the_pipeline(monkeypatch):
    import queue
    import threading

    monkeypatch.setattr("time.sleep", lambda s: None)
    sim = _sim()
    q: queue.Queue = queue.Queue()
    sim._audio_q = q
    sim._stop = threading.Event()
    sim._feed_pipeline = False
    played: list[bytes] = []

    class Out:
        def push(self, pcm):
            played.append(pcm)
    sim._play_out = Out()

    class Agent:
        _connected = True

        def __init__(self):
            self.fed = 0

        def feed_audio(self, pcm):
            self.fed += 1
    sim._agent_ref = Agent()

    sim._send_pcm(b"\x01\x00" * 16000)           # 1 秒
    sim._send_silence(0.5)
    assert q.empty()                              # STT には入れない
    assert sim._agent_ref.fed == 0                # GPT-Live にも入れない（マイクが拾う）
    assert len(played) >= 8                       # スピーカーには出す

    sim._feed_pipeline = True
    sim._send_pcm(b"\x01\x00" * 16000)
    assert not q.empty() and sim._agent_ref.fed > 0
