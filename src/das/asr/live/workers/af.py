"""常駐処理の責務: af。会話の判定規則・時刻条件は元実装を維持する。"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._session_state import SessionState

from .._constants import (
    AGENT_SPEAKER,
)
from .._facilitation import (
    FacilitationController,
    FacilitationInput,
    InterventionLogEntry,
)
from .._intervention import (
    _build_candidates,
    _intervention_enabled,
    _PendingInterventions,
    _suppressed_for,
)
from .._speaker_policy import (
    intervention_records,
    intervention_speaker_name,
)

_AF_L2_MIN_NEW_NODES = 4
_AF_L2_MAX_NEW_NODES = 16
_AF_L2_MAX_LEVEL = 2

def _af_l1_presentation(decision: Any) -> str:
    """AF L1 decision の items を関係ラベル付き提示文に整文する（H1 フェーズ4）."""
    label = {"support": "支持", "attack": "反論"}
    who = decision.addressed_to
    head = f"{who}さんの先ほどの発言に対する関連情報:" if who else "関連情報:"
    lines = [head]
    for it in decision.items:
        tag = label.get(it.relation, "参考")
        lines.append(f"- [{tag}] {it.source_text}")
    return "\n".join(lines)


def _af_l2_reason_type(reason: str) -> str:
    """af_l2 の理由文字列を粗い理由タイプに分類する (同種理由バックオフ用)。"""
    if "停滞" in reason:
        return "stalled"
    if "未応答" in reason or "偏り" in reason:
        return "bias"
    return "other"


def _af_checker_tick(
    state: SessionState, facil: Any, presented: set[str],
    af_gate: dict[str, Any] | None = None,
) -> int:
    """AF checker の 1 周分。af 候補を state.af_requests に積み、積んだ件数を返す。

    ``facil`` は :class:`FacilitationAgent`。``presented`` は提示済み source_text 集合
    (呼び出し側が meeting 世代ごとに保持)。``af_gate`` は af_l2 再発火ガードの状態
    ({"last_l2_node_count": int, "last_l2_reason_type": str, "l2_required_nodes": int,
    "last_l2_resp_count": int})。テスト容易性のため 1 周を関数化してある。
    """
    from das.types import Utterance

    if af_gate is None:
        af_gate = {}

    runtime = getattr(state, "af_runtime", None)
    if runtime is None:
        return 0  # AF 無効 (既定): 何もしない
    agent = state.agent
    if agent is None or getattr(agent, "mode", "facilitator") == "conversation":
        return 0
    epoch = state.meeting_epoch
    with state.state_lock:
        talk_rs = intervention_records([
            r for r in state.records
            if "speaker" in r and r.get("text")
            and r.get("speaker") != AGENT_SPEAKER
        ])
    if not talk_rs:
        return 0
    transcript = [
        Utterance(turn_id=i + 1, speaker=intervention_speaker_name(state, r),
                  text=str(r["text"]))
        for i, r in enumerate(talk_rs)
    ]
    try:
        store = runtime.store
        decision = facil.decide_intervention(transcript, store)
        if decision.kind == "l1":
            decision = facil.apply_l1_value_gate(
                decision, store, transcript, presented_source_texts=presented)
    except Exception as exc:  # pragma: no cover - 防御的
        print(f"# [af] decide error: {exc}", flush=True)
        return 0
    if decision.kind == "skip":
        return 0
    # 副作用（キュー投入）の直前で epoch 再確認（H2）。
    with state.state_lock:
        if state.meeting_epoch != epoch:
            return 0
    if decision.kind == "l1":
        for it in decision.items:
            presented.add(it.source_text.strip())
        state.af_requests.put({
            "kind": "af_l1",
            "brief": decision.reason,
            "af_text": _af_l1_presentation(decision),
            "target_speaker": decision.addressed_to,
        })
        print(f"# [af] → af_l1 候補を投入（{len(decision.items)}件）", flush=True)
        return 1
    # af_l2 再発火ガード + 同種理由バックオフ (課題2)。
    # 前回 af_l2 以降に新規発話ノードが「必要数」追加されるまで出さない。必要数は、
    # 同じ理由タイプで状態が改善しない (応答エッジも増えない) まま再発火が続くほど
    # 倍化する (4→8→16)。理由タイプが変わるか、応答エッジが増えたらベース(4)に戻す。
    n_utt = sum(1 for node in store.nodes() if node.source == "utterance")
    reason_type = _af_l2_reason_type(decision.reason)
    n_resp = len(getattr(runtime, "_response_edges", []) or [])
    last = af_gate.get("last_l2_node_count")
    last_type = af_gate.get("last_l2_reason_type")
    last_resp = int(af_gate.get("last_l2_resp_count", 0))
    level = int(af_gate.get("l2_backoff_level", 0))
    # 理由タイプが変わった / 応答エッジが増えた (介入が届いた) → バックオフをリセット。
    reset = last is None or reason_type != last_type or n_resp > last_resp
    if reset:
        level = 0
    required = min(_AF_L2_MIN_NEW_NODES * (2 ** level), _AF_L2_MAX_NEW_NODES)
    if last is not None and n_utt - last < required:
        print(
            f"# [af] af_l2 skip: 前回af_l2以降の新規発話ノード不足 "
            f"({n_utt - last}/{required}, 理由={reason_type}, lv={level})",
            flush=True,
        )
        return 0
    # 発火。同種理由・未改善のまま再発火した場合だけ、次回の必要数を倍化する。
    next_level = 0 if reset else min(level + 1, _AF_L2_MAX_LEVEL)
    af_gate["last_l2_node_count"] = n_utt
    af_gate["last_l2_reason_type"] = reason_type
    af_gate["last_l2_resp_count"] = n_resp
    af_gate["l2_backoff_level"] = next_level
    state.af_requests.put({
        "kind": "af_l2",
        "brief": decision.reason,
        "af_text": decision.brief,
        "target_speaker": None,
    })
    print(f"# [af] → af_l2 候補を投入 (理由={reason_type}, 必要数={required})", flush=True)
    return 1


class _AfDeliveryGate:
    """af 候補の配信ゲート（--af 有効時のみ・af 限定）.

    WHEN は Controller に委譲し、ここは「フロアが空いたら指示を送る」だけを担う。
    以前は pause 成立前に生成だけ先行させて音声を溜め、フロア成立で一斉再生する
    生成先行（hold/release）を持っていたが、GPT-Live は音声を実時間で流すので
    溜めるとモデルの時間とずれる。指示から声までの遅延は記録して詰める
    （docs/design/live_native_plan_2026-09.md §3.3）。

    ``status`` は Controller から得た af 候補の採否状態:
      - ``"deliver"`` : cooldown/arbitration/フロアを通過し **今** 配信してよい
      - ``"hold"``    : 採択見込みだが間待ち → 何もしない（次の tick で再評価）
      - ``"none"``    : af 候補なし / cooldown・期限切れ・他候補優先などで見送り
    """

    MIN_SILENCE = 0.3

    def tick(
        self, *, agent: Any, af: dict[str, Any] | None, status: str,
        silence: float, agent_busy: bool, topics: Any = None, **_unused,
    ) -> str:
        """1 ループ分の処理。行った操作名 (deliver/none) を返す。"""
        if agent_busy or silence < self.MIN_SILENCE or not af or status != "deliver":
            return "none"
        sent = agent.trigger(
            topics=topics,
            af_presentation=str(af.get("af_text") or ""),
            invite_target=af.get("target_speaker"),
        )
        if sent is False:                # 送れなかった（開き直し中・話し中）。候補は残す
            return "none"
        return "deliver"


def _af_gate_status(
    controller: FacilitationController,
    pending: _PendingInterventions,
    agent: Any,
    *,
    now: float,
    silence_elapsed: float,
    recent_interventions: list[InterventionLogEntry],
    cooldown: float,
    last_intervention_at: float,
    epoch: int,
    partner_busy: bool,
    in_echo_window: bool,
) -> tuple[str, dict[str, Any] | None]:
    """af 候補の採否状態を Controller に問い、ゲート駆動用の status を返す (WHEN を委譲).

    三層分離: WHAT/WHOM は AF (decide_intervention)、WHEN は Controller (cooldown/
    arbitration/フロア) が決める。ゲートは再生タイミングだけを担う。ここで Controller
    の ``arbitrate`` を af 候補にかけ、

      - 採択 (candidate_id が af)         → ``"deliver"`` (pause 成立・今出してよい)
      - 何も採らず af が awaiting_pause のみ → ``"hold"``    (採択見込み・間待ち)
      - それ以外 (cooldown/期限切れ/他優先) → ``"none"``

    を返す。af 候補が無い / conversation モードなら ``("none", None)``。af 候補は
    ``--af`` 有効時しか積まれないので、ルールベース経路には影響しない。
    """
    if not pending.af or getattr(agent, "mode", "facilitator") == "conversation":
        return "none", None
    cands = [c for c in _build_candidates(pending, agent, now=now)
             if c.kind in ("af_l1", "af_l2")]
    if not cands:
        return "none", None
    af_cand = cands[0]
    decision = controller.arbitrate(FacilitationInput(
        candidates=tuple(cands),
        recent_interventions=tuple(recent_interventions),
        silence_elapsed=silence_elapsed,
        snapshot_epoch=epoch,
        now=now,
        cooldown=cooldown,
        partner_busy=partner_busy,
        in_echo_window=in_echo_window,
        last_intervention_at=last_intervention_at,
        required_drift_confirmations=0,
    ))
    if decision.candidate_id == af_cand.id:
        return "deliver", pending.af
    if _suppressed_for(decision, candidate_id=af_cand.id, codes=("awaiting_pause",)):
        return "hold", pending.af
    return "none", pending.af


def _run_af_checker(state: SessionState, *, interval: float = 3.0) -> None:
    """AF ベース介入の候補生成ワーカー（H1 フェーズ4, AF ランタイム有効時のみ）.

    ``state.af_runtime`` (run_af_runtime がセット) が育てた AF を数秒周期で読み、
    ``decide_intervention`` + L1 価値ゲートで介入候補を決めて ``state.af_requests``
    に積む。trigger は行わず、既存の _run_agent_worker が Controller 採否を通す。
    AF ランタイムが無い (既定 OFF) なら何もしない = ルールベース挙動は不変。
    """
    from das.agents.facilitation import FacilitationAgent

    facil = FacilitationAgent(llm=None)  # decide_intervention は LLM を呼ばない (決定的)
    presented: set[str] = set()
    af_gate: dict[str, Any] = {}
    epoch = state.meeting_epoch
    while not state.stop.is_set():
        time.sleep(interval)
        if not _intervention_enabled(state):
            continue
        if state.meeting_epoch != epoch:
            epoch = state.meeting_epoch
            presented = set()
            af_gate = {}
            facil.reset()
        _af_checker_tick(state, facil, presented, af_gate)

