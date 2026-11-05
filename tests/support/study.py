"""研究用モードの人工判定と模擬通信。実API・マイク・音声出力は使わない。"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

from das.study.engine import Review, StudyConfig
from das.study.live import StudyRuntime


def review():
    return Review.model_validate(
        {
            "reasons": [
                {
                    "option": "D案",
                    "source": {"turn_id": "t1", "quote": "平日夜間に参加できます"},
                    "stance": "support",
                    "relevance": 1,
                    "handling": "unhandled",
                    "handled_by": None,
                }
            ],
            "proposal": {
                "option": "D案",
                "intent": "exclude",
                "source": {"turn_id": "t2", "quote": "D案は費用が重いので除外しよう"},
                "current_reason": {"turn_id": "t2", "quote": "費用が重い"},
                "already_handled": None,
                "discussion_in_progress": None,
            },
        }
    )


class Immediate:
    calls = 0

    def submit(self, fn, *args):
        self.calls += 1
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as exc:
            f.set_exception(exc)
        return f


class Agent:
    ready = True
    ai_speaking = False
    _responding = False
    in_echo_window = False

    def __init__(self):
        self.spoken = []
        self.stops = 0

    def trigger(self, **kw):
        self.spoken.append(kw)
        return True

    def stop_playback(self):
        self.stops += 1
        self.ai_speaking = False
        self._responding = False


def state():
    return SimpleNamespace(
        records=[
            {"speaker": "人物1", "text": "D案なら平日夜間に参加できます。", "end_ms": 1000},
            {"speaker": "人物2", "text": "D案は費用が重いので除外しよう。", "end_ms": 2000},
        ],
        state_lock=threading.RLock(),
        partial_text="",
        _last_partial_change=0,
        study_model="fake-no-api",
        study_api_key="not-a-real-key",
        agent=Agent(),
        intervention_enabled=True,
    )


def model(*_args):
    r = review()
    for x in [r.reasons[0].source, r.proposal.source, r.proposal.current_reason]:
        x.turn_id = {"t1": "turn-00001", "t2": "turn-00002"}[x.turn_id]
    return r


def runtime(tmp_path, condition="system"):
    return StudyRuntime(
        StudyConfig(session_id="synthetic", task_id="t", title="人工課題", options=["A案", "D案"],
                    pause_seconds=0),
        condition,
        str(tmp_path / "run.study.jsonl"),
    )

