"""既存ASRの確定記録と音声エージェントを研究用の独立レーンへ接続する。"""

from __future__ import annotations

import datetime
import hashlib
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .engine import (
    EXTRACTION_PROMPT,
    Candidate,
    Review,
    StudyConfig,
    StudyEngine,
    Turn,
    revision_of,
)


def build_review_params(
    config: StudyConfig, turns: list[Turn], new_ids: set[str], model: str,
    *, max_output: int = 6000, reasoning_effort: str | None = None,
) -> dict:
    """実行時と接続確認で、抽出指示とJSON形式を共有する。"""
    from das.asr.live._llm_transport import _build_chat_params

    payload = {
        "options": config.options,
        "new_turn_ids": sorted(new_ids),
        "turns": [t.model_dump() for t in turns],
    }
    params = _build_chat_params(
        model,
        "",
        max_out=max_output,
        temperature=0,
        schema_name="decision_review",
        schema=Review.model_json_schema(),
    )
    params["messages"] = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    if reasoning_effort is not None and "reasoning_effort" in params:
        params["reasoning_effort"] = reasoning_effort
    return params


def request_review(
    config: StudyConfig, turns: list[Turn], new_ids: set[str], api_key: str, model: str
) -> Review:
    from das.asr.live._llm_transport import _post_chat_json

    params = build_review_params(config, turns, new_ids, model)
    result = _post_chat_json(
        params, api_key, timeout=30, label="study-review", raise_auth_errors=True
    )
    if result is None:
        raise RuntimeError("study review returned no valid JSON")
    return Review.model_validate(result)


class StudyRuntime:
    def __init__(self, config: StudyConfig, condition: str, log_path: str):
        if condition not in {"none", "system"}:
            raise ValueError("study condition must be none or system")
        self.config = config
        self.condition = condition
        self.engine = StudyEngine(config)
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._public: list[dict] = []
        self._status = "準備中"
        self._epoch = None
        self._observed_revision = ""
        self._last_revision_at = float("-inf")
        self._known: dict[str, str] = {}
        self._last_request_epoch = None
        self._future: Future | None = None
        self._request = None
        self.pending: Candidate | None = None
        self._auth_failed = False
        self.log(
            "session_config",
            condition=condition,
            config=config.model_dump(),
            prompt_sha256=hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest(),
        )

    def log(self, kind: str, **data):
        event = {
            "type": kind,
            "at": datetime.datetime.now(datetime.UTC).isoformat(),
            "monotonic": time.monotonic(),
            "session_id": self.config.session_id,
            **data,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def view(self):
        with self._lock:
            return {
                "title": self.config.title,
                "condition": self.condition,
                "status": self._status,
                "records": list(self._public) if self.condition == "system" else [],
            }

    @staticmethod
    def capture(state):
        from das.asr.live._constants import AGENT_SPEAKER
        from das.asr.live._speaker_policy import is_reliable_human_speaker

        with state.state_lock:
            turns = []
            for i, r in enumerate(state.records):
                if "speaker" not in r or r["speaker"] in {AGENT_SPEAKER, "パートナー"}:
                    continue
                text = str(r.get("text") or "").strip()
                if not text:
                    continue
                key = str(r["speaker"]) if is_reliable_human_speaker(r) else "発言者不明"
                turns.append(
                    Turn(
                        id=f"turn-{i + 1:05d}",
                        speaker=key,
                        text=text,
                        ms=max(0, int(r.get("end_ms") or r.get("ms") or 0)),
                    )
                )
            partial = str(state.partial_text or "")
            changed = float(state._last_partial_change)
        return turns, partial, changed

    def tick(self, state, executor, reviewer, *, now=None):
        tick_started = time.monotonic()
        now = tick_started if now is None else now
        turns, partial, partial_at = self.capture(state)
        revision = revision_of(turns)
        epoch = (revision, partial_at)
        agent = state.agent
        if epoch != self._epoch:
            if self.pending:
                self.log("candidate_cancelled", reason="new_input", revision=revision)
            self.pending = None
            self._epoch = epoch
            if revision != self._observed_revision:
                self.log(
                    "transcript_snapshot", revision=revision, turns=[t.model_dump() for t in turns]
                )
                self._observed_revision = revision
                self._last_revision_at = now
                with self._lock:
                    self._public = []  # Do not keep displaying a corrected/withdrawn old quote.
                    self._status = "発言を確認中"
                if agent and (agent.ai_speaking or agent._responding):
                    agent.stop_playback()
                    self.log("audio_cancel_requested", reason="confirmed_human_input")
        if self.condition == "none":
            return  # No LLM, no voice and no shared records in control sessions.
        if self._future is not None and self._future.done():
            future, request = self._future, self._request
            self._future = self._request = None
            try:
                review = future.result()
                self.log(
                    "review_result",
                    model=request[4],
                    latency_ms=round((now - request[3]) * 1000, 1),
                    revision=request[0][0],
                    review=review.model_dump(),
                )
                if request[0] != epoch:
                    self.log("abstain", reason="stale_llm_result", revision=request[0][0])
                else:
                    candidate, reason = self.engine.evaluate(
                        turns, review, revision=revision, now=now, new_turn_ids=request[2]
                    )
                    self._known = {t.id: t.model_dump_json() for t in turns}
                    self.pending = candidate
                    with self._lock:
                        self._public = self.engine.public_records(turns)
                        self._status = "議論中"
                    self.log(
                        "decision",
                        reason=reason,
                        revision=revision,
                        text=candidate.text if candidate else None,
                        history=[r.model_dump() for r in candidate.history] if candidate else [],
                    )
            except Exception as exc:
                self.log("review_error", error=type(exc).__name__, detail=str(exc))
                if getattr(exc, "code", None) in (401, 403):
                    self._auth_failed = True
                    self.pending = None
                    self.log("model_disabled", reason="authentication_failed")
                with self._lock:
                    self._status = "記録の更新を待っています"
        quiet = not partial and (
            now - max(partial_at, self._last_revision_at) >= self.config.pause_seconds
        )
        if self.pending:
            p = self.pending
            if now - p.created_at > self.config.candidate_ttl_seconds:
                self.log("candidate_cancelled", reason="expired", revision=p.revision)
                self.pending = None
            elif (
                quiet
                and agent
                and agent.ready
                and not agent.ai_speaking
                and not agent._responding
                and not agent.in_echo_window
            ):
                # Recheck immediately before enqueueing a voice instruction.
                latest, live_partial, latest_partial_at = self.capture(state)
                if (revision_of(latest), latest_partial_at) != epoch or live_partial:
                    self.pending = None
                    return
                if not getattr(state, "intervention_enabled", True):
                    self.pending = None
                    self.log("abstain", reason="disabled_by_operator")
                    return
                if agent.trigger(
                    prepared_text=p.text,
                    prepared_deadline=p.created_at + self.config.candidate_ttl_seconds,
                ):
                    # Dispatch can block while ASR continues on another thread. Recheck
                    # after sending, and count a request only against the current input.
                    after, after_partial, after_partial_at = self.capture(state)
                    input_changed = (revision_of(after), after_partial_at) != epoch or bool(
                        after_partial
                    )
                    commit_now = now + (time.monotonic() - tick_started)
                    committed = not input_changed and self.engine.commit(
                        p, revision=revision_of(after), now=commit_now
                    )
                    self.pending = None
                    self.log(
                        "voice_requested",
                        option=p.option,
                        text=p.text,
                        revision=revision,
                        committed=committed,
                    )
                    if not committed:
                        agent.stop_playback()
                        self.log(
                            "audio_cancel_requested",
                            reason="input_changed_during_dispatch" if input_changed
                            else "commit_rejected",
                        )
        if (
            turns
            and not self._auth_failed
            and self._future is None
            and self._last_request_epoch != epoch
            and quiet
        ):
            new_ids = {t.id for t in turns if self._known.get(t.id) != t.model_dump_json()}
            if not new_ids:
                return
            self._last_request_epoch = epoch
            self._request = (epoch, turns, new_ids, now, state.study_model)
            self.log(
                "review_requested",
                model=state.study_model,
                revision=revision,
                new_turn_ids=sorted(new_ids),
            )
            self._future = executor.submit(
                reviewer, self.config, turns, new_ids, state.study_api_key, state.study_model
            )


def run_study_worker(state):
    runtime = state.study
    runtime.log(
        "worker_started", model=state.study_model, voice_model=getattr(state.agent, "model", None)
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="study-review")
    try:
        while not state.stop.wait(0.1):
            try:
                runtime.tick(state, executor, request_review)
            except Exception as exc:
                runtime.pending = None
                runtime.log("worker_error", error=type(exc).__name__, detail=str(exc))
                state.stop.wait(1)
    finally:
        if runtime._future:
            runtime._future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        if state.agent:
            state.agent.stop_playback()
        runtime.log("worker_stopped")
