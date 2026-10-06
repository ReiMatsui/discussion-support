"""Metrics computed from event logs, with explicit missing-stage observations."""

from collections import Counter, defaultdict


def quantiles(values):
    if not values:
        return {"p50": None, "p90": None, "n": 0}
    ordered = sorted(values)

    def percentile(p):
        index = (len(ordered) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {"p50": percentile(0.5), "p90": percentile(0.9), "n": len(values)}


def compute(events, *, cost_usd=0.0, reserved_usd=0.0, issue_chars=20, position_chars=16):
    commits = [e for e in events if e.type == "display_commit"]
    duration = max((e.ms for e in events), default=0) - min((e.ms for e in events), default=0)
    counts = Counter(kind for e in commits for kind in e.data["changes"])
    labels = [
        (e.data["node"]["label"], e.data["node"]["kind"])
        for e in events
        if e.type == "node_created"
    ]
    labels += [(e.data["new_label"], "issue") for e in events if e.type == "issue_upgraded"]
    latencies = defaultdict(list)
    pending = {}
    previous_rows = {}
    previous_focus = None
    for e in events:
        if e.type in {
            "node_created",
            "stance_transition",
            "status_transition",
            "focus_transition",
            "issue_upgraded",
        }:
            node_id = e.data.get("node_id") or e.data.get("node", {}).get("id")
            if e.type == "focus_transition":
                node_id = e.data["current"]
            kind = {
                "node_created": "node",
                "stance_transition": "stance",
                "status_transition": "status",
                "focus_transition": "focus",
                "issue_upgraded": "upgrade",
            }[e.type]
            pending[(node_id, kind)] = (e.ms, e.data.get("origin_end_ms", e.ms))
        if e.type == "display_commit":
            rows = {n["id"]: n for n in e.data["state"]["nodes"]}
            focus_id = e.data["state"]["focus_id"]
            for key, (change_ms, origin_ms) in list(pending.items()):
                node_id, kind = key
                row, old = rows.get(node_id), previous_rows.get(node_id)
                changed = (kind == "focus" and focus_id != previous_focus) or (
                    row is not None
                    and (
                        (kind == "node" and old is None)
                        or (kind == "stance" and (old is None or row["marks"] != old["marks"]))
                        or (kind == "status" and (old is None or row["status"] != old["status"]))
                        or (kind == "upgrade" and (old is None or row["label"] != old["label"]))
                    )
                )
                if changed:
                    latencies[kind].append(e.ms - origin_ms)
                    latencies["change_to_display"].append(e.ms - change_ms)
                # Superseded changes were never displayed; do not fabricate latency.
                if row is not None or kind == "focus":
                    del pending[key]
            previous_rows, previous_focus = rows, focus_id
    return {
        "duration_ms": duration,
        "display_commits": len(commits),
        "display_changes_by_kind": dict(counts),
        "display_commits_per_minute": len(commits) * 60000 / duration if duration else None,
        "label_lengths": {
            kind: [len(t) for t, k in labels if k == kind] for kind in ("issue", "position")
        },
        "label_overflow_rate": sum(
            len(t) > (issue_chars if k == "issue" else position_chars) for t, k in labels
        )
        / len(labels)
        if labels
        else 0,
        "judgement_latency_ms": quantiles(
            [e.data["latency_ms"] for e in events if e.type == "judgement"]
        ),
        "slow_layer_latency_ms": quantiles(
            [e.data["latency_ms"] for e in events if e.type == "label_generation"]
        ),
        "display_latency_clock": "transcript virtual time; CPU/API durations reported separately",
        "rules_and_slow_layer_latency_ms": quantiles(
            [e.data["latency_ms"] for e in events if e.type == "rules_completed"]
        ),
        "display_latency_ms": {k: quantiles(v) for k, v in latencies.items()},
        "latency_stages": {
            "end_to_asr_final": None,
            "asr_final_to_fast": None,
            "note": "ASR finalization timestamps absent in offline turns; API durations use monotonic clock",
        },
        "unapplied_structure_proposals": sum(e.type == "structure_proposal" for e in events),
        "ai_speech": {
            "count": 0,
            "types": {},
            "lengths": [],
            "summary_duration_ms": None,
            "confirmations": 0,
        },
        "api_cost_usd": cost_usd,
        "api_reserved_usd": reserved_usd,
    }


def projection(snapshot):
    """Exact semantic tree expectation, independent of timings and API durations."""
    return {
        "focus_id": snapshot["internal"]["focus_id"],
        "nodes": [
            {k: n[k] for k in ("id", "kind", "label", "parent_id", "status")}
            | {
                "answer_type": n.get("answer_type"),
                "decided_answer": n.get("decided_answer"),
                "stances": {
                    uid: {"value": s["value"], "name": s["name"]} for uid, s in n["stances"].items()
                },
            }
            for n in snapshot["internal"]["nodes"]
        ],
    }


def scenario_check(scenario, snapshot, checkpoints):
    checks = []
    for name, expected, actual in [
        ("final", scenario["expected"], projection(snapshot)),
        *[
            (f"turn:{tid}", state, checkpoints.get(tid))
            for tid, state in scenario.get("checkpoints", {}).items()
        ],
    ]:
        checks.append(
            {"name": name, "matches": expected == actual, "expected": expected, "actual": actual}
        )
    return {"mismatches": sum(not c["matches"] for c in checks), "checks": checks}


def comparison(events, scripted):
    fields = defaultdict(list)
    for e in events:
        if e.type != "judgement" or e.turn_id not in scripted:
            continue
        for name, prediction in e.data["judgement"].items():
            if name not in scripted[e.turn_id]:
                continue
            probs = prediction["probabilities"]
            predicted = max(probs, key=probs.get)
            truth_probs = scripted[e.turn_id][name]["probabilities"]
            truth = max(truth_probs, key=truth_probs.get)
            fields[name].append((probs[predicted], predicted == truth))
    return {
        name: {
            "n": len(rows),
            "accuracy": sum(ok for _, ok in rows) / len(rows),
            "calibration": [
                {
                    "low": lo / 10,
                    "high": (lo + 2) / 10,
                    "n": len(bucket),
                    "accuracy": sum(ok for _, ok in bucket) / len(bucket) if bucket else None,
                    "mean_probability": sum(p for p, _ in bucket) / len(bucket) if bucket else None,
                }
                for lo in range(0, 10, 2)
                for bucket in [
                    [(p, ok) for p, ok in rows if lo / 10 <= p and (p < (lo + 2) / 10 or lo == 8)]
                ]
            ],
        }
        for name, rows in fields.items()
    }
