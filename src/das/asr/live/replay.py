"""Transcript replay harness for facilitator intervention checks.

This module replays saved ``*.turns.jsonl`` files without microphone/STT.
It is intentionally text-first: the goal is to make facilitator tuning
repeatable before adding heavier audio/TTS replay.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click

from das.asr.live._constants import (
    _DRIFT_CHECK_INTERVAL,
    _DRIFT_CHECK_WINDOW,
    _DRIFT_WARMUP,
    _FACTCHECK_WINDOW,
    _INVITE_QUIET_RATIO,
    _INVITE_WARMUP,
    AGENT_SPEAKER,
)
from das.asr.live._participation import participation_stats
from das.asr.live._speaker_policy import reliable_human_records
from das.asr.live._workers import _looks_like_fact_claim

CheckFact = Callable[[list[dict], str, str], dict]
CheckDrift = Callable[[list[dict], list[dict], str, str], dict]
CheckParticipation = Callable[[list[dict], list[dict], str, str], dict]

AGENT_SPEAKERS = {AGENT_SPEAKER, "AI", "パートナー"}


@dataclass
class ReplayOptions:
    """Options for deterministic transcript replay."""

    api_key: str = ""
    model: str = "gpt-5-mini"
    topic: str | None = None
    checks: set[str] = field(default_factory=lambda: {"fact", "drift", "invite"})
    no_api: bool = False
    limit: int | None = None
    include_agent: bool = False
    fact_cooldown_turns: int = 6


def load_turns(path: str | Path, *, include_agent: bool = False,
               limit: int | None = None) -> list[dict[str, Any]]:
    """Load turns.jsonl records in SessionState.write_turns() format."""
    turns: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            speaker = str(item.get("speaker") or "")
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            if not include_agent and speaker in AGENT_SPEAKERS:
                continue
            turns.append({
                "turn_id": item.get("turn_id"),
                "speaker": speaker,
                "text": text,
                "ms": item.get("ms"),
                "end_ms": item.get("end_ms"),
            })
            if limit is not None and len(turns) >= limit:
                break
    return turns


def _event(turn: dict, kind: str, detail: str, **extra) -> dict:
    return {
        "turn_id": turn.get("turn_id"),
        "ms": turn.get("ms"),
        "type": kind,
        "speaker": turn.get("speaker"),
        "text": turn.get("text"),
        "detail": detail,
        **extra,
    }


def _utterance_window(records: list[dict], size: int) -> list[dict]:
    return [{"speaker": r["speaker"], "text": r["text"]} for r in records[-size:]]


def _run_fact_check(
    records: list[dict],
    turn: dict,
    opts: ReplayOptions,
    check_fact: CheckFact,
) -> dict | None:
    if "fact" not in opts.checks:
        return None
    if not _looks_like_fact_claim(turn["text"]):
        return None
    if opts.no_api:
        return _event(turn, "fact_candidate", "式・数値・定義っぽい発話")
    result = check_fact(_utterance_window(records, _FACTCHECK_WINDOW), opts.api_key, opts.model)
    if result.get("should_correct"):
        correction = str(result.get("correction") or "").strip()
        if correction:
            return _event(
                turn,
                "fact",
                correction,
                claim=result.get("claim", ""),
                reason=result.get("reason", ""),
            )
    return None


def _run_drift_check(
    records: list[dict],
    turn: dict,
    opts: ReplayOptions,
    check_drift: CheckDrift,
) -> dict | None:
    if "drift" not in opts.checks or not opts.topic or opts.no_api:
        return None
    n = len(records)
    if n < _DRIFT_WARMUP or n % _DRIFT_CHECK_INTERVAL != 0:
        return None
    topics = [{"topic": opts.topic, "speaker": "議題"}]
    result = check_drift(_utterance_window(records, _DRIFT_CHECK_WINDOW),
                         topics, opts.api_key, opts.model)
    if result.get("drift"):
        return _event(turn, "drift", str(result.get("reason") or "脱線"))
    return None


def _run_invite_check(
    records: list[dict],
    turn: dict,
    opts: ReplayOptions,
    check_participation: CheckParticipation,
) -> dict | None:
    if "invite" not in opts.checks or opts.no_api:
        return None
    if len(records) < _INVITE_WARMUP or len(records) % _INVITE_WARMUP != 0:
        return None
    stats = participation_stats(reliable_human_records(records),
                                exclude_speakers=tuple(AGENT_SPEAKERS))
    if len(stats) < 2:
        return None
    equal = 1.0 / len(stats)
    if min(d["time_share"] for d in stats.values()) >= equal * _INVITE_QUIET_RATIO:
        return None
    now_ms = max((d["last_end_ms"] for d in stats.values()
                  if d["last_end_ms"] is not None), default=None)
    participation = []
    for speaker, data in stats.items():
        silent = ((now_ms - data["last_end_ms"]) / 1000.0
                  if now_ms is not None and data["last_end_ms"] is not None else 0.0)
        participation.append({
            "speaker": speaker,
            "time_share": data["time_share"],
            "turns": data["turns"],
            "silent_sec": silent,
        })
    result = check_participation(participation, _utterance_window(records, _DRIFT_CHECK_WINDOW),
                                 opts.api_key, opts.model)
    if result.get("invite") and result.get("speaker"):
        return _event(turn, "invite", str(result.get("speaker")),
                      reason=result.get("reason", ""))
    return None


def run_replay(
    turns: list[dict],
    opts: ReplayOptions,
    *,
    check_fact: CheckFact | None = None,
    check_drift: CheckDrift | None = None,
    check_participation: CheckParticipation | None = None,
) -> list[dict]:
    """Replay turns and return intervention candidate events."""
    from das.asr.live._bootstrap import (
        check_drift as default_check_drift,
        check_fact_correction as default_check_fact,
        check_participation as default_check_participation,
    )

    check_fact = check_fact or default_check_fact
    check_drift = check_drift or default_check_drift
    check_participation = check_participation or default_check_participation

    records: list[dict] = []
    events: list[dict] = []
    last_fact_event_at = -10_000
    for turn in turns:
        records.append(turn)
        fact_event = _run_fact_check(records, turn, opts, check_fact)
        if fact_event:
            if fact_event["type"] == "fact_candidate":
                events.append(fact_event)
            elif len(records) - last_fact_event_at >= opts.fact_cooldown_turns:
                events.append(fact_event)
                last_fact_event_at = len(records)
        drift_event = _run_drift_check(records, turn, opts, check_drift)
        if drift_event:
            events.append(drift_event)
        invite_event = _run_invite_check(records, turn, opts, check_participation)
        if invite_event:
            events.append(invite_event)
    return events


def _parse_checks(value: str) -> set[str]:
    checks = {v.strip() for v in value.split(",") if v.strip()}
    unknown = checks - {"fact", "drift", "invite"}
    if unknown:
        raise click.BadParameter(f"unknown checks: {', '.join(sorted(unknown))}")
    return checks


@click.command()
@click.argument("turns_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--topic", default=None, help="脱線判定の基準議題。未指定ならdriftは走りません")
@click.option("--checks", default="fact,drift,invite",
              help="実行する判定。カンマ区切り: fact,drift,invite")
@click.option("--model", default=None, help="OPENAI_MODEL_FASTの代わりに使うモデル")
@click.option("--out", default=None, type=click.Path(dir_okay=False),
              help="イベントJSONLの保存先。未指定なら標準出力")
@click.option("--limit", type=int, default=None, help="先頭N発話だけリプレイ")
@click.option("--include-agent", is_flag=True,
              help="保存済みのAI/ファシリテーター発話も入力に含める")
@click.option("--no-api", is_flag=True,
              help="APIを呼ばず、ローカル候補抽出だけ行う")
def main(turns_path: str, topic: str | None, checks: str, model: str | None,
         out: str | None, limit: int | None, include_agent: bool, no_api: bool) -> None:
    """Replay a saved turns.jsonl file and print intervention candidates."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    opts = ReplayOptions(
        api_key=api_key,
        model=model or os.environ.get("OPENAI_MODEL_FAST", "gpt-5-mini"),
        topic=topic,
        checks=_parse_checks(checks),
        no_api=no_api,
        limit=limit,
        include_agent=include_agent,
    )
    if not opts.no_api and not opts.api_key:
        raise click.ClickException("OPENAI_API_KEY is required unless --no-api is set")
    turns = load_turns(turns_path, include_agent=include_agent, limit=limit)
    events = run_replay(turns, opts)
    lines = [json.dumps(e, ensure_ascii=False) for e in events]
    text = "\n".join(lines) + ("\n" if lines else "")
    if out:
        Path(out).write_text(text, encoding="utf-8")
        click.echo(f"# replay events: {len(events)} -> {out}")
    else:
        click.echo(text, nl=False)


if __name__ == "__main__":
    main()
