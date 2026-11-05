"""接続確認の永続利用記録。費用の計算・集計は既存CostTrackerへ委譲する。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path

from das.llm.cost import PRICING, CostTracker

MAX_CALLS = 3
DEFAULT_BUDGET_USD = 0.10


class ProbeLimitError(ValueError):
    """送信を止めた理由。本文はローカルで定義した説明だけを含む。"""


def _tracker_and_holds(entries):
    tracker = CostTracker()
    reservations = [e for e in entries if e.get("type", "reservation") == "reservation"]
    results = {e["attempt"]: e for e in entries if e.get("type") == "result"}
    held = 0.0
    unknown = 0
    for entry in reservations:
        result = results.get(entry["attempt"], {})
        usage = result.get("usage")
        if (
            isinstance(usage, dict)
            and type(usage.get("prompt_tokens")) is int
            and type(usage.get("completion_tokens")) is int
            and usage["prompt_tokens"] >= 0
            and usage["completion_tokens"] >= 0
        ):
            tracker.record(entry["model"], usage["prompt_tokens"], usage["completion_tokens"])
        else:
            unknown += 1
            # Includes old reservations made before result events were added.
            held += reserve_estimate(entry["model"], entry["request_bytes"], entry["max_output_tokens"])
    return tracker, held, unknown, reservations


def reserve_estimate(model, request_bytes, max_output):
    # Never use CostTracker's unknown-model price fallback for a budget gate.
    # UTF-8 request bytes + framing margin is a conservative planning estimate,
    # not an exact token count or a guarantee about the provider's bill.
    if model not in PRICING:
        raise ProbeLimitError("料金表に完全一致するモデルがないため、費用の予約を停止しました")
    return PRICING[model].cost(request_bytes + 1024, max_output)


class ProbeUsage:
    def __init__(self, path: Path, budget_usd=DEFAULT_BUDGET_USD):
        if not math.isfinite(budget_usd) or budget_usd <= 0:
            raise ValueError("予算は正の有限値にしてください")
        self.path = path
        self.budget_usd = budget_usd

    @staticmethod
    def _read(file):
        file.seek(0)
        return [json.loads(line) for line in file if line.strip()]

    @staticmethod
    def _append(file, entry):
        file.seek(0, os.SEEK_END)
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())

    def snapshot(self):
        if self.path.exists():
            with self.path.open(encoding="utf-8") as file:
                fcntl.flock(file, fcntl.LOCK_SH)
                entries = self._read(file)
        else:
            entries = []
        tracker, held, unknown, reservations = _tracker_and_holds(entries)
        return {
            "attempts": len(reservations), "remaining_calls": max(0, MAX_CALLS - len(reservations)),
            "reported_usage": tracker.snapshot(),
            "unknown_usage_attempts": unknown,
            "held_estimate_usd": held,
            "budget_usd": self.budget_usd,
            "remaining_estimate_usd": max(0, self.budget_usd - tracker.total_usd - held),
        }

    def reserve(self, model: str, body: bytes, max_output: int):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            entries = self._read(file)
            tracker, held, _, reservations = _tracker_and_holds(entries)
            if len(reservations) >= MAX_CALLS:
                raise ProbeLimitError("累計3回の上限です。自動では追加実行しません")
            if any(e["model"] != model for e in reservations):
                raise ProbeLimitError("利用記録の途中でモデルが変わったため停止しました")
            if any(e.get("budget_usd", self.budget_usd) != self.budget_usd for e in reservations):
                raise ProbeLimitError("保存済みの予算と指定が異なるため停止しました")
            estimate = reserve_estimate(model, len(body), max_output)
            if tracker.total_usd + held + estimate > self.budget_usd:
                raise ProbeLimitError("既知の費用と不明分の予約額が予算を超えるため送信しません")
            attempt = len(reservations) + 1
            self._append(file, {
                "type": "reservation", "attempt": attempt, "model": model,
                "request_bytes": len(body), "max_output_tokens": max_output,
                "request_sha256": hashlib.sha256(body).hexdigest(), "at_unix": time.time(),
                "budget_usd": self.budget_usd, "reserved_estimate_usd": estimate,
            })
            return attempt

    def finish(self, attempt: int, *, usage, status: str, http_status=None):
        with self.path.open("a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            entries = self._read(file)
            if any(e.get("type") == "result" and e["attempt"] == attempt for e in entries):
                raise ValueError("同じ試行結果は二重計上しません")
            if not any(e.get("type", "reservation") == "reservation" and e["attempt"] == attempt
                       for e in entries):
                raise ValueError("予約のない試行です")
            self._append(file, {
                "type": "result", "attempt": attempt, "status": status,
                "usage": usage, "http_status": http_status, "at_unix": time.time(),
            })
