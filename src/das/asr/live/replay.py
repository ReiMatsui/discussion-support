"""Transcript replay harness for facilitator intervention checks.

This module replays saved ``*.turns.jsonl`` files without microphone/STT.
It is intentionally text-first: the goal is to make facilitator tuning
repeatable before adding heavier audio/TTS replay.
"""
from __future__ import annotations

import json
import os
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import click

from das.asr.live._constants import (
    _DRIFT_CHECK_INTERVAL,
    _DRIFT_CHECK_WINDOW,
    _DRIFT_WARMUP,
    _INVITE_QUIET_RATIO,
    _INVITE_WARMUP,
    AGENT_SPEAKER,
)
from das.asr.live._participation import (
    participation_share_key,
    participation_share_label,
    participation_stats,
    quietest_participation_share,
)
from das.asr.live._speaker_policy import is_intervention_signal, reliable_human_records
from das.asr.live._workers import _looks_like_fact_claim

CheckFact = Callable[[list[dict], str, str], dict]
CheckDrift = Callable[[list[dict], list[dict], str, str], dict]
CheckParticipation = Callable[[list[dict], list[dict], str, str], dict]

AGENT_SPEAKERS = {AGENT_SPEAKER, "AI", "パートナー"}

REPLAY_INDEX_HTML = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>リプレイ検証</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --line:#e5e7eb; --ink:#1f2937;
    --muted:#6b7280; --fact:#0e7490; --drift:#b45309; --invite:#6d28d9; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, "Hiragino Sans", "Segoe UI", sans-serif; line-height: 1.55; }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 16px; }
  header { display:flex; align-items:baseline; gap:12px; margin-bottom:14px; }
  h1 { font-size: 1.08rem; margin:0; }
  .meta { color: var(--muted); font-size:.82rem; }
  .cols { display:grid; grid-template-columns: minmax(0,1fr) 340px; gap:14px; align-items:start; }
  @media (max-width: 860px) { .cols { grid-template-columns: 1fr; } }
  .panel { background: var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }
  .panel h2 { margin:0 0 .6rem; font-size:.86rem; color:#9ca3af; font-weight:600; }
  .turn { padding:.45rem .55rem; border-bottom:1px solid #f0f1f3; }
  .turn:last-child { border-bottom:0; }
  .turn.hit { background:#f8fafc; border-left:3px solid #94a3b8; }
  .ts { color:#9ca3af; font-size:.74rem; margin-right:.45rem; font-variant-numeric:tabular-nums; }
  .speaker { font-weight:700; margin-right:.45rem; }
  .event { border:1px solid var(--line); border-radius:8px; padding:.55rem .65rem; margin-bottom:.5rem; background:#fff; }
  .event .kind { font-weight:700; font-size:.78rem; }
  .event.fact .kind, .event.fact_candidate .kind, .event.fact_retryable_error .kind { color:var(--fact); }
  .event.drift .kind { color:var(--drift); }
  .event.invite .kind, .event.invite_rejected .kind { color:var(--invite); }
  .event .detail { margin-top:.25rem; }
  .event .quote { color:var(--muted); font-size:.78rem; margin-top:.3rem; }
  .review { border-top:1px solid var(--line); margin-top:.8rem; padding-top:.8rem; }
  .review-item { border:1px solid var(--line); border-radius:8px; padding:.55rem .65rem; margin-bottom:.5rem; background:#fff; }
  .review-item .status { color:var(--muted); font-size:.76rem; }
  .review-item .delivery { margin-top:.25rem; font-weight:600; }
  .review-item .context { color:var(--muted); font-size:.76rem; margin-top:.35rem; }
  .chips { display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.7rem; }
  .chip { border:1px solid var(--line); border-radius:999px; padding:.18rem .55rem; font-size:.78rem; background:#fff; }
  .empty { color:var(--muted); text-align:center; padding:1rem; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>リプレイ検証</h1>
    <span class="meta" id="summary">読み込み中...</span>
  </header>
  <div class="cols">
    <section class="panel">
      <h2>発話ログ</h2>
      <div id="turns"></div>
    </section>
    <aside class="panel">
      <h2>介入候補</h2>
      <div class="chips" id="chips"></div>
      <div id="events"></div>
      <div class="review">
        <h2>保存済み介入</h2>
        <div id="review"></div>
      </div>
    </aside>
  </div>
</div>
<script>
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ts = (ms) => {
  if (ms == null) return "--:--";
  const sec = Math.floor(ms / 1000);
  return `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
};
const label = (t) => ({
  fact: "事実補正",
  fact_candidate: "事実候補",
  fact_retryable_error: "事実判定失敗",
  drift: "脱線",
  invite: "声かけ",
  invite_rejected: "声かけ除外",
}[t] || t);
const statusLabel = (s) => ({
  delivered: "発話済み",
  missing_delivery: "発話未確認",
  orphan_delivery: "発火理由なし",
}[s] || s);
const flagLabel = (s) => ({
  missing_delivery: "発話なし",
  orphan_delivery: "理由なし発話",
  no_recent_context: "文脈なし",
  drift_without_topic: "議題なし脱線",
  long_delivery: "長い介入",
}[s] || s);
fetch("/api/replay").then((r) => r.json()).then((data) => {
  const events = data.events || [];
  const review = data.intervention_review || [];
  const reviewSummary = data.intervention_review_summary || {};
  const hitTurns = new Set(events.map((e) => e.turn_id));
  document.getElementById("summary").textContent =
    `${data.source} / ${data.turns.length}発話 / 候補${events.length}件 / 保存済み${review.length}件`;
  const counts = events.reduce((acc, e) => (acc[e.type] = (acc[e.type] || 0) + 1, acc), {});
  document.getElementById("chips").innerHTML = Object.keys(counts).length
    ? Object.entries(counts).map(([k,v]) => `<span class="chip">${esc(label(k))}: ${v}</span>`).join("")
    : `<span class="chip">候補なし</span>`;
  document.getElementById("events").innerHTML = events.length
    ? events.map((e) => `<div class="event ${esc(e.type)}">
        <div><span class="kind">${esc(label(e.type))}</span>
          <span class="meta">#${esc(e.turn_id)} ${esc(ts(e.ms))}</span></div>
        <div class="detail">${esc(e.detail)}</div>
        ${e.reason ? `<div class="meta">${esc(e.reason)}</div>` : ""}
        <div class="quote">${esc(e.speaker)}: ${esc(e.text)}</div>
      </div>`).join("")
    : `<div class="empty">介入候補はありません</div>`;
  document.getElementById("review").innerHTML = review.length
    ? `<div class="chips">
        <span class="chip">合計: ${esc(reviewSummary.total ?? review.length)}</span>
        <span class="chip">発話済み: ${esc(reviewSummary.status_counts?.delivered ?? 0)}</span>
        <span class="chip">要確認: ${esc(reviewSummary.flagged_count ?? 0)}</span>
        <span class="chip">10発話あたり: ${esc(reviewSummary.interventions_per_10_turns ?? "-")}</span>
      </div>` + review.map((r) => `<div class="review-item">
        <div><span class="kind">${esc(r.reason || "delivery")}</span>
          <span class="status">${esc(statusLabel(r.status))}</span></div>
        ${r.detail ? `<div class="detail">${esc(r.detail)}</div>` : ""}
        ${r.delivery_text ? `<div class="delivery">${esc(r.delivery_text)}</div>` : ""}
        ${r.quality_flags?.length ? `<div class="chips">${
          r.quality_flags.map((f) => `<span class="chip">${esc(flagLabel(f))}</span>`).join("")
        }</div>` : ""}
        <div class="context">turns: ${esc(r.turn_count ?? "-")}
          ${r.topics?.length ? ` / 論点: ${esc(r.topics.map((t) => t.topic).join(", "))}` : ""}</div>
        ${r.recent_utterances?.length ? `<div class="quote">${
          r.recent_utterances.map((u) => `${esc(u.speaker)}: ${esc(u.text)}`).join("<br>")
        }</div>` : ""}
      </div>`).join("")
    : `<div class="empty">保存済み介入ログはありません</div>`;
  document.getElementById("turns").innerHTML = data.turns.length
    ? data.turns.map((t) => `<div class="turn ${hitTurns.has(t.turn_id) ? "hit" : ""}">
        <span class="ts">${esc(ts(t.ms))}</span><span class="speaker">${esc(t.speaker)}</span>${esc(t.text)}
      </div>`).join("")
    : `<div class="empty">発話がありません</div>`;
});
</script>
</body>
</html>"""


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


def default_interventions_path(turns_path: str | Path) -> Path:
    """Return the sibling interventions.jsonl path for a turns file."""
    path = Path(turns_path)
    name = path.name
    if name.endswith(".turns.jsonl"):
        return path.with_name(name[:-len(".turns.jsonl")] + ".interventions.jsonl")
    if name.endswith(".jsonl"):
        return path.with_name(name[:-len(".jsonl")] + ".interventions.jsonl")
    return path.with_name(name + ".interventions.jsonl")


def load_interventions(path: str | Path | None) -> list[dict[str, Any]]:
    """Load saved intervention trigger/delivery events from JSONL."""
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _intervention_quality_flags(
    *,
    status: str,
    reason: str,
    delivery_text: str,
    metadata: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if status == "missing_delivery":
        flags.append("missing_delivery")
    if status == "orphan_delivery":
        flags.append("orphan_delivery")
    if not metadata.get("recent_utterances"):
        flags.append("no_recent_context")
    if reason == "drift" and not metadata.get("topics"):
        flags.append("drift_without_topic")
    if len(delivery_text) > 80:
        flags.append("long_delivery")
    return flags


def intervention_review_items(interventions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair trigger and delivery records for human review."""
    triggers = {
        str(e.get("event_id")): e
        for e in interventions
        if e.get("type") == "trigger" and e.get("event_id")
    }
    deliveries_by_trigger: dict[str, list[dict[str, Any]]] = {}
    orphans: list[dict[str, Any]] = []
    for event in interventions:
        if event.get("type") != "delivery":
            continue
        trigger_id = event.get("trigger_event_id")
        if trigger_id and str(trigger_id) in triggers:
            deliveries_by_trigger.setdefault(str(trigger_id), []).append(event)
        else:
            orphans.append(event)

    items: list[dict[str, Any]] = []
    for event_id, trigger in triggers.items():
        delivery = deliveries_by_trigger.get(event_id, [None])[0]
        metadata = trigger.get("metadata") if isinstance(trigger.get("metadata"), dict) else {}
        reason = str(trigger.get("reason") or "")
        delivery_text = str(delivery.get("text", "") if delivery else "")
        status = "delivered" if delivery else "missing_delivery"
        items.append({
            "event_id": event_id,
            "status": status,
            "reason": reason,
            "detail": trigger.get("detail", ""),
            "created_at": trigger.get("created_at"),
            "turn_count": metadata.get("turn_count"),
            "recent_utterances": metadata.get("recent_utterances", []),
            "topics": metadata.get("topics", []),
            "delivery_text": delivery_text,
            "quality_flags": _intervention_quality_flags(
                status=status,
                reason=reason,
                delivery_text=delivery_text,
                metadata=metadata,
            ),
            "trigger": trigger,
            "delivery": delivery,
        })
    for delivery in orphans:
        delivery_text = str(delivery.get("text", ""))
        items.append({
            "event_id": None,
            "status": "orphan_delivery",
            "reason": "",
            "detail": "",
            "created_at": delivery.get("created_at"),
            "turn_count": None,
            "recent_utterances": [],
            "topics": [],
            "delivery_text": delivery_text,
            "quality_flags": _intervention_quality_flags(
                status="orphan_delivery",
                reason="",
                delivery_text=delivery_text,
                metadata={},
            ),
            "trigger": None,
            "delivery": delivery,
        })
    return items


def intervention_review_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize saved intervention review items for run-level comparison."""
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    flagged_count = 0
    for item in items:
        status = str(item.get("status") or "unknown")
        reason = str(item.get("reason") or "delivery")
        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        flags = item.get("quality_flags") or []
        if flags:
            flagged_count += 1
        for flag in flags:
            flag = str(flag)
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    return {
        "total": len(items),
        "flagged_count": flagged_count,
        "status_counts": status_counts,
        "reason_counts": reason_counts,
        "flag_counts": flag_counts,
    }


def intervention_review_run_summary(
    items: list[dict[str, Any]],
    *,
    turn_count: int,
) -> dict[str, Any]:
    """Summarize a review with transcript-normalized metrics."""
    summary = intervention_review_summary(items)
    delivered = summary["status_counts"].get("delivered", 0)
    denominator = max(turn_count, 1)
    summary.update({
        "turn_count": turn_count,
        "interventions_per_10_turns": round(summary["total"] / denominator * 10, 3),
        "delivered_per_10_turns": round(delivered / denominator * 10, 3),
        "flagged_per_10_turns": round(summary["flagged_count"] / denominator * 10, 3),
    })
    return summary


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
    if not is_intervention_signal(turn):
        return None
    if not _looks_like_fact_claim(turn["text"]):
        return None
    if opts.no_api:
        return _event(turn, "fact_candidate", "LLM事実判定の対象")
    prior = [
        r for r in records[:-1]
        if is_intervention_signal(r)
    ][-3:]
    utts = _utterance_window([*prior, turn], 4)
    result = check_fact(utts, opts.api_key, opts.model)
    if result.get("retryable_error"):
        return _event(turn, "fact_retryable_error", "LLM事実判定の一時失敗")
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
    if quietest_participation_share(stats) >= equal * _INVITE_QUIET_RATIO:
        return None
    now_ms = max((d["last_end_ms"] for d in stats.values()
                  if d["last_end_ms"] is not None), default=None)
    participation = []
    valid_invite_targets: set[str] = set()
    share_key = participation_share_key(stats)
    share_label = participation_share_label(share_key)
    for speaker, data in stats.items():
        silent = ((now_ms - data["last_end_ms"]) / 1000.0
                  if now_ms is not None and data["last_end_ms"] is not None else 0.0)
        valid_invite_targets.add(str(speaker))
        participation.append({
            "speaker": speaker,
            "time_share": data["time_share"],
            "participation_share": data[share_key],
            "participation_share_label": share_label,
            "turns": data["turns"],
            "silent_sec": silent,
        })
    result = check_participation(participation, _utterance_window(records, _DRIFT_CHECK_WINDOW),
                                 opts.api_key, opts.model)
    if result.get("invite") and result.get("speaker"):
        target = str(result.get("speaker"))
        if target not in valid_invite_targets:
            return _event(turn, "invite_rejected", target,
                          reason="信頼できる参加者名ではない")
        return _event(turn, "invite", target, reason=result.get("reason", ""))
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
    from das.asr.live import _bootstrap

    check_fact = check_fact or _bootstrap.check_fact_correction
    check_drift = check_drift or _bootstrap.check_drift
    check_participation = check_participation or _bootstrap.check_participation

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


def replay_snapshot(source: str | Path, turns: list[dict], events: list[dict],
                    opts: ReplayOptions, interventions: list[dict] | None = None) -> dict:
    """Build the JSON object served by the replay UI."""
    interventions = interventions or []
    review_items = intervention_review_items(interventions)
    return {
        "source": str(source),
        "topic": opts.topic,
        "checks": sorted(opts.checks),
        "no_api": opts.no_api,
        "turns": turns,
        "events": events,
        "interventions": interventions,
        "intervention_review": review_items,
        "intervention_review_summary": intervention_review_run_summary(
            review_items,
            turn_count=len(turns),
        ),
    }


def serve_replay(snapshot: dict, *, port: int, open_browser: bool) -> None:
    """Serve replay results as an in-memory local UI."""
    payload = json.dumps(snapshot, ensure_ascii=False).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/replay":
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/" or self.path.startswith("/?"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(REPLAY_INDEX_HTML.encode("utf-8"))
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    click.echo(f"# replay UI: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


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
@click.option("--review-out", default=None, type=click.Path(dir_okay=False),
              help="保存済み介入レビューJSONLの保存先")
@click.option("--review-summary-out", default=None, type=click.Path(dir_okay=False),
              help="保存済み介入レビュー集計JSONの保存先")
@click.option("--serve", is_flag=True, help="結果をローカルUIで表示する")
@click.option("--port", type=int, default=8232, help="--serve のポート番号。0で自動割当")
@click.option("--open/--no-open", "open_browser", default=True,
              help="--serve 時にブラウザを開く")
@click.option("--limit", type=int, default=None, help="先頭N発話だけリプレイ")
@click.option("--include-agent", is_flag=True,
              help="保存済みのAI/ファシリテーター発話も入力に含める")
@click.option("--interventions", "interventions_path", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="保存済み介入ログJSONL。未指定ならturns隣の*.interventions.jsonlを読む")
@click.option("--no-api", is_flag=True,
              help="APIを呼ばず、ローカル候補抽出だけ行う")
def main(turns_path: str, topic: str | None, checks: str, model: str | None,
         out: str | None, review_out: str | None, review_summary_out: str | None,
         serve: bool, port: int, open_browser: bool, limit: int | None,
         include_agent: bool, interventions_path: str | None, no_api: bool) -> None:
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
    default_path = default_interventions_path(turns_path)
    interventions = load_interventions(interventions_path or default_path)
    snapshot = replay_snapshot(turns_path, turns, events, opts, interventions)
    if review_out:
        review_lines = [
            json.dumps(item, ensure_ascii=False)
            for item in snapshot["intervention_review"]
        ]
        Path(review_out).write_text(
            "\n".join(review_lines) + ("\n" if review_lines else ""),
            encoding="utf-8",
        )
        click.echo(f"# intervention review: {len(review_lines)} -> {review_out}")
    if review_summary_out:
        Path(review_summary_out).write_text(
            json.dumps(snapshot["intervention_review_summary"], ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        click.echo(f"# intervention review summary -> {review_summary_out}")
    if serve:
        serve_replay(snapshot, port=port, open_browser=open_browser)
        return
    lines = [json.dumps(e, ensure_ascii=False) for e in events]
    text = "\n".join(lines) + ("\n" if lines else "")
    if out:
        Path(out).write_text(text, encoding="utf-8")
        click.echo(f"# replay events: {len(events)} -> {out}")
    else:
        click.echo(text, nl=False)


if __name__ == "__main__":
    main()
