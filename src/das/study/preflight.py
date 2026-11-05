"""20分相当の人工入力を仮想時計で流す。実API・録音・音声出力は使用しない。"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .engine import Review, StudyConfig
from .live import StudyRuntime


class ScriptedExecutor:
    def __init__(self):
        self.calls = 0

    def submit(self, fn, *args):
        self.calls += 1
        future = Future()
        future.set_result(fn(*args))
        return future


class SilentAgent:
    ready = True
    ai_speaking = False
    _responding = False
    in_echo_window = False

    def __init__(self):
        self.requests = []

    def trigger(self, **kwargs):
        self.requests.append(kwargs)
        return True

    def stop_playback(self):
        pass


def scripted_review(config, turns, *_):
    # Ground truth is constructed here, not inferred by an LLM. This tests the
    # controller and logging over a growing transcript, not semantic accuracy.
    reason, proposal = turns[-2:]
    option = config.options[((len(turns) // 2) - 1) % len(config.options)]
    return Review.model_validate({
        "reasons": [{
            "option": option, "source": {"turn_id": reason.id, "quote": reason.text},
            "stance": "support", "relevance": 1, "handling": "unhandled",
            "handled_by": None,
        }],
        "proposal": {
            "option": option, "intent": "exclude",
            "source": {"turn_id": proposal.id, "quote": proposal.text},
            "current_reason": None, "already_handled": None, "discussion_in_progress": None,
        },
    })


def run_preflight(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    for condition in ("system", "none"):
        clock = [0.0]
        log_path = out_dir / f"synthetic_20min_{condition}.jsonl"
        if log_path.exists():
            raise FileExistsError(f"既存の検証ログを上書きしません: {log_path}")
        config = StudyConfig(
            session_id=f"synthetic-20min-{condition}", task_id="synthetic-load",
            title="人工負荷試験", options=[f"{x}案" for x in "ABCDEF"],
        )
        with patch("das.study.live.time.monotonic", lambda clock=clock: clock[0]):
            runtime = StudyRuntime(config, condition, str(log_path))
            executor = ScriptedExecutor()
            state = SimpleNamespace(
                state_lock=threading.RLock(), records=[], partial_text="",
                _last_partial_change=0, agent=SilentAgent(), intervention_enabled=True,
                study_model="scripted-no-api", study_api_key="not-used",
            )
            # 120 proposals / 240 turns. Each option recurs every 60 seconds.
            # The normal 1.2-second quiet rule, 60-second interval and cap of 2
            # remain enabled. Tick every 0.1 virtual second for 20 minutes.
            for step in range(12001):
                clock[0] = step / 10
                if step < 12000 and step % 100 == 0:
                    i = step // 100
                    option = config.options[i % 6]
                    state.records.extend([
                        {"speaker": "人物1", "text": f"{option}には利点があります。",
                         "end_ms": i * 10000},
                        {"speaker": "人物2", "text": f"{option}を除外しよう。",
                         "end_ms": i * 10000 + 1},
                    ])
                runtime.tick(state, executor, scripted_review, now=clock[0])
            events = [json.loads(line) for line in log_path.read_text().splitlines()]
            delivered = [e for e in events if e["type"] == "voice_requested"]
            expected = 12 if condition == "system" else 0
            assert len(state.agent.requests) == len(delivered) == expected
            assert all(e["committed"] for e in delivered)
            assert executor.calls == (120 if condition == "system" else 0)
            assert runtime.engine.counts == (
                {option: 2 for option in config.options} if condition == "system" else {}
            )
            results.append({
                "condition": condition, "virtual_seconds": 1200,
                "ticks": 12001, "human_turns": len(state.records),
                "scripted_review_calls": executor.calls, "voice_requests": len(delivered),
                "request_counts": runtime.engine.counts, "log_bytes": log_path.stat().st_size,
            })
    report = {
        "data_kind": "synthetic", "status": "passed",
        "limits": "仮想時間・模擬意味判定・模擬音声送信。実機で20分動かした結果でも効果検証でもない。",
        "wall_seconds": round(time.perf_counter() - started, 3), "runs": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    print(json.dumps(run_preflight(parser.parse_args().out_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
