"""使用量を制限した人工会話の接続確認。既定は送信なし、--execute時のみ1回。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .engine import EXTRACTION_PROMPT, Review, StudyConfig, StudyEngine, Turn, revision_of

MAX_CALLS = 3
MAX_REQUEST_BYTES = 32768
MAX_OUTPUT_TOKENS = 6000
DEFAULT_LEDGER = Path("study/probes/bounded_usage.jsonl")


def build_request(fixture: Path, model: str, max_output: int):
    from das.asr.live._bootstrap import _build_chat_params

    if not 1 <= max_output <= MAX_OUTPUT_TOKENS:
        raise ValueError("出力上限は1〜6000トークンにしてください")
    data = json.loads(fixture.read_text(encoding="utf-8"))
    if data.get("data_kind") != "synthetic":
        raise ValueError("人工入力だけを受け付けます")
    config = StudyConfig.model_validate(data["config"])
    step = data["steps"][1]
    turns = [Turn.model_validate(t) for t in step["turns"]]
    new_ids = set(step["new_turn_ids"])
    params = _build_chat_params(
        model, "", max_out=max_output, temperature=0,
        schema_name="decision_review", schema=Review.model_json_schema(),
    )
    # The ordinary transport can retry an unsupported effort value. This probe
    # never retries; low is an explicit probe setting, recorded in the plan.
    if "reasoning_effort" in params:
        params["reasoning_effort"] = "low"
    params["messages"] = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": json.dumps({
            "options": config.options, "new_turn_ids": sorted(new_ids),
            "turns": [t.model_dump() for t in turns],
        }, ensure_ascii=False)},
    ]
    body = json.dumps(params, ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_REQUEST_BYTES:
        raise ValueError("送信データが32KiBを超えるため停止しました")
    return config, turns, new_ids, params, body


def reserve_call(path: Path, model: str, body: bytes, max_output: int):
    """Reserve before sending; failures and crashes consume a slot too."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.seek(0)
        entries = [json.loads(line) for line in file if line.strip()]
        if len(entries) >= MAX_CALLS:
            raise ValueError("累計3回の上限です。自動では追加実行しません")
        if any(e["model"] != model for e in entries):
            raise ValueError("利用記録の途中でモデルが変わったため停止しました")
        entry = {
            "attempt": len(entries) + 1, "model": model,
            "request_bytes": len(body), "max_output_tokens": max_output,
            "request_sha256": hashlib.sha256(body).hexdigest(), "at_unix": time.time(),
        }
        file.seek(0, os.SEEK_END)
        file.write(json.dumps(entry) + "\n")
        file.flush()
        os.fsync(file.fileno())
        return entry["attempt"]


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
    args = parser.parse_args(argv)
    load_env()
    model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5.4-mini")
    config, turns, new_ids, params, body = build_request(
        args.fixture, model, args.max_output_tokens
    )
    plan = {
        "mode": "execute" if args.execute else "dry_run_no_network",
        "model": model, "request_bytes": len(body),
        "max_request_bytes": MAX_REQUEST_BYTES, "max_output_tokens": args.max_output_tokens,
        "max_cumulative_calls": MAX_CALLS, "ledger": str(DEFAULT_LEDGER),
        "reasoning_effort": params.get("reasoning_effort"), "automatic_retries": 0,
        "money_limit": "金額上限ではありません。単価・実入力トークン数で費用が変わります。",
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
            report["attempt"] = reserve_call(DEFAULT_LEDGER, model, body, args.max_output_tokens)
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
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "result_file": str(args.out),
        "attempt": report.get("attempt"), "usage": report.get("usage"),
    }, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
