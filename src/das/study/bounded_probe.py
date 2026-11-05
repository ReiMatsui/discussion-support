"""使用量を制限した人工会話の接続確認。既定は送信なし、--execute時のみ1回。"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .engine import Review, StudyConfig, StudyEngine, Turn, revision_of
from .live import build_review_params
from .probe_usage import (
    DEFAULT_BUDGET_USD,
    MAX_CALLS,
    ProbeLimitError,
    ProbeUsage,
    reserve_estimate,
)

MAX_REQUEST_BYTES = 32768
MAX_OUTPUT_TOKENS = 6000
DEFAULT_LEDGER = Path("study/probes/bounded_usage.jsonl")


def build_request(fixture: Path, model: str, max_output: int):
    if not 1 <= max_output <= MAX_OUTPUT_TOKENS:
        raise ValueError("出力上限は1〜6000トークンにしてください")
    data = json.loads(fixture.read_text(encoding="utf-8"))
    if data.get("data_kind") != "synthetic":
        raise ValueError("人工入力だけを受け付けます")
    config = StudyConfig.model_validate(data["config"])
    step = data["steps"][1]
    turns = [Turn.model_validate(t) for t in step["turns"]]
    new_ids = set(step["new_turn_ids"])
    params = build_review_params(
        config, turns, new_ids, model, max_output=max_output, reasoning_effort="low"
    )
    body = json.dumps(params, ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError("送信データが32KiBを超えるため停止しました")
    return config, turns, new_ids, params, body


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def send_once(body: bytes, key: str):
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    # No retry or redirect, including 400/401/429/timeouts. A lost response may
    # still have been billed, so its reserved slot is never refunded.
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        return json.loads(response.read())


def main(argv=None):
    from das.asr.live._bootstrap import load_env

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--fixture", type=Path, default=Path("study/synthetic_replay.json"))
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    args = parser.parse_args(argv)
    load_env()
    model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5.4-mini")
    config, turns, new_ids, params, body = build_request(
        args.fixture, model, args.max_output_tokens
    )
    ledger = ProbeUsage(DEFAULT_LEDGER, budget_usd=args.budget_usd)
    plan = {
        "mode": "execute" if args.execute else "dry_run_no_network",
        "model": model, "request_bytes": len(body),
        "max_request_bytes": MAX_REQUEST_BYTES, "max_output_tokens": args.max_output_tokens,
        "max_cumulative_calls": MAX_CALLS, "ledger": str(DEFAULT_LEDGER),
        "reasoning_effort": params.get("reasoning_effort"), "automatic_retries": 0,
        "estimated_reservation_usd": reserve_estimate(model, len(body), args.max_output_tokens),
        "cumulative_usage": ledger.snapshot(),
        "money_limit": "推定費用に基づく送信前の制限です。請求額の厳密な保証ではありません。",
    }
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.out is None:
        parser.error("実行時は --out で新しい結果ファイルを指定してください")
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or key.startswith("test-"):
        parser.error("ローカルのOPENAI_API_KEYを設定してください")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Claim the output before consuming budget; refuse accidental overwrite.
    with args.out.open("x", encoding="utf-8") as output:
        report = {"plan": plan, "data_kind": "synthetic"}
        started = time.monotonic()
        phase = "budget_reservation"
        try:
            report["attempt"] = ledger.reserve(model, body, args.max_output_tokens)
            phase = "api_request"
            response = send_once(body, key)
            phase = "response_validation"
            report["usage"] = response.get("usage")
            choice = response["choices"][0]
            report["finish_reason"] = choice.get("finish_reason")
            if choice.get("finish_reason") != "stop":
                raise ValueError("生成が完了していません。自動再試行しません")
            review = Review.model_validate_json(choice["message"]["content"])
            candidate, reason = StudyEngine(config).evaluate(
                turns, review, revision=revision_of(turns), now=time.monotonic(),
                new_turn_ids=new_ids,
            )
            report.update(
                status="completed", review=review.model_dump(), decision=reason,
                prepared_text=candidate.text if candidate else None,
            )
        except Exception as exc:
            # Never save provider error bodies or exception messages containing keys.
            report.update(status="failed", error_type=type(exc).__name__, failed_phase=phase)
            if isinstance(exc, urllib.error.HTTPError):
                report["http_status"] = exc.code
                report["message"] = (
                    "APIの認証・権限を確認してください。設定を直すまで再実行しないでください。"
                    if exc.code in (401, 403) else
                    "APIが要求を拒否しました。HTTPステータスを確認してください。"
                )
            elif isinstance(exc, ProbeLimitError):
                report["message"] = str(exc)
        if "attempt" in report:
            ledger.finish(
                report["attempt"], usage=report.get("usage"), status=report["status"],
                http_status=report.get("http_status"),
            )
        report["cumulative_usage"] = ledger.snapshot()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "result_file": str(args.out),
        "attempt": report.get("attempt"), "usage": report.get("usage"),
        "http_status": report.get("http_status"), "failed_phase": report.get("failed_phase"),
        "cumulative_usage": report.get("cumulative_usage"), "message": report.get("message"),
    }, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
