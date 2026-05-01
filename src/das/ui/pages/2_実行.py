"""eval 実行ページ (ライブ議論ビュー)。

Streamlit のフォームから ``das eval --emit-events`` をサブプロセス起動し、
stdout に流れる議論イベント (utterance / intervention / run_start / run_end) を
パースして「進行中の議論」をリアルタイムにカード表示する。

設計:
  - イベント駆動: CLI が ``__DAS_EVT__<json>`` 行を発行する仕様に依拠
  - 並列実行 (concurrency>1) のときはランごとに独立カラム
  - 生 CLI ログは折りたたみへ格下げ
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from das.settings import get_settings

st.set_page_config(page_title="実行", layout="wide")
st.title("eval を実行する")
st.caption(
    "提案手法を含むシミュレーション評価をここから起動。発話と介入はライブで時系列表示します。"
)

PRESETS: dict[str, str] = {
    "cafeteria": "大学のカフェテリアでプラスチック容器を廃止すべきか",
    "policy_ai": "生成 AI を大学の講義・レポート作成で許容すべきか",
}

CONDITIONS: list[str] = ["full_proposal", "none", "flat_rag"]
CONDITION_LABELS: dict[str, str] = {
    "full_proposal": "提案手法",
    "none": "情報なし",
    "flat_rag": "Flat RAG",
}

STANCE_BADGE: dict[str, str] = {
    "pro": "🟢 賛成",
    "con": "🔴 反対",
    "neutral": "⚪ 中立",
}

# --- フォーム ----------------------------------------------------------

with st.form("eval_form"):
    cols = st.columns(2)
    with cols[0]:
        preset = st.selectbox(
            "トピックプリセット",
            list(PRESETS.keys()),
            format_func=lambda k: f"{k} — {PRESETS[k]}",
        )
        n_runs = st.slider("各条件のラン数", 1, 10, 3)
        max_turns = st.slider("ターン上限 (安全上限)", 5, 40, 25)
        concurrency = st.slider("並列度", 1, 10, 3)
    with cols[1]:
        conditions = st.multiselect(
            "走らせる条件",
            CONDITIONS,
            default=CONDITIONS,
            format_func=lambda c: f"{CONDITION_LABELS[c]} ({c})",
        )
        until_consensus = st.checkbox("合意検出で早期終了", value=True)
        agreement_threshold = st.slider("合意キーワードしきい値", 0.2, 0.9, 0.4, 0.05)
        agreement_window = st.slider("合意判定の直近ターン数", 2, 8, 4)
        no_judge = st.checkbox("LLM-as-judge をスキップ", value=False)

    eval_id_input = st.text_input("eval_id (任意。空なら自動)", value="")
    submitted = st.form_submit_button("▶ 実行", type="primary")

# --- 実行 -------------------------------------------------------------


def _build_command(*, eval_id: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "das.cli",
        "eval",
        preset,
        "-n",
        str(n_runs),
        "-t",
        str(max_turns),
        "-j",
        str(concurrency),
        "--emit-events",
    ]
    if conditions:
        cmd += ["--conditions", ",".join(conditions)]
    if until_consensus:
        cmd += [
            "--until-consensus",
            "--agreement-threshold",
            str(agreement_threshold),
            "--agreement-window",
            str(agreement_window),
        ]
    if no_judge:
        cmd += ["--no-judge"]
    if eval_id:
        cmd += ["--eval-id", eval_id]
    return cmd


def _stance_for(speaker: str, personas: list[dict]) -> str:
    for p in personas:
        if p.get("name") == speaker:
            return p.get("stance", "neutral")
    return "neutral"


def _render_run(container, run_state: dict) -> None:
    """1 ラン分の状態をカードに描画する。"""

    cond = run_state["condition"]
    run_idx = run_state["run_idx"]
    status = run_state.get("status", "running")
    n_turns = run_state.get("n_turns", 0)
    consensus = run_state.get("consensus")

    label = f"{CONDITION_LABELS.get(cond, cond)} / run {run_idx:02d}"
    if status == "running":
        label = f"⏳ {label} ({n_turns} ターン進行中)"
    elif status == "done":
        if consensus and consensus.get("reached"):
            label = f"✅ {label} ({n_turns} ターン, 合意 turn {consensus.get('at')})"
        else:
            label = f"☑️ {label} ({n_turns} ターン, 合意なし)"

    with container.expander(label, expanded=(status == "running")):
        for evt in run_state["timeline"]:
            if evt["type"] == "utterance":
                stance = _stance_for(evt["speaker"], run_state.get("personas", []))
                with st.container(border=True):
                    c = st.columns([1, 9])
                    c[0].markdown(
                        f"**t{evt['turn_id']}**\n\n"
                        f"**{evt['speaker']}**\n\n{STANCE_BADGE.get(stance, '⚪')}"
                    )
                    c[1].markdown(evt["text"])
            elif evt["type"] == "intervention":
                kind = evt.get("kind", "l1")
                if kind == "skip":
                    continue
                if kind == "l2":
                    with st.container(border=True):
                        st.markdown(
                            f"🟪 **L2 議論の整理 (全体共有)** — _{evt.get('decision_reason', '')}_"
                        )
                        if evt.get("brief"):
                            st.markdown(f"> {evt['brief']}")
                else:
                    items = evt.get("items", [])
                    if not items:
                        continue
                    with st.container(border=True):
                        addressed = evt.get("addressed_to") or "次話者"
                        st.markdown(
                            f"🟦 **L1 → {addressed}** ({len(items)} 件)"
                        )
                        for it in items:
                            tag = "🟢 支持" if it.get("relation") == "support" else "🔴 反論"
                            kind_src = it.get("source_kind", "?")
                            st.markdown(
                                f"- {tag} ({kind_src}): {it.get('source_text', '')}"
                            )


def _render_all_runs(area, runs_state: dict[tuple[str, int], dict]) -> None:
    """すべてのラン状態を一気に再描画 (Streamlit のインクリメンタル更新)。"""

    with area.container():
        if not runs_state:
            st.caption("(まだ発言がありません)")
            return

        # 進行中を上、完了を下に
        sort_key = lambda kv: (kv[1].get("status") != "running", kv[0])  # noqa: E731
        for _, state in sorted(runs_state.items(), key=sort_key):
            _render_run(st, state)


if submitted:
    if not conditions:
        st.error("条件を 1 つ以上選んでください。")
        st.stop()

    fallback_eval_id = (
        eval_id_input.strip()
        or f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    cmd = _build_command(eval_id=eval_id_input.strip())

    with st.expander("実行コマンド", expanded=False):
        st.code(" ".join(cmd), language="bash")

    progress = st.progress(0.0, text="準備中...")
    discussion_area = st.empty()
    log_expander = st.expander("CLI 生ログ (デバッグ用)", expanded=False)
    log_box = log_expander.empty()

    settings = get_settings()
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # 状態
    runs_state: dict[tuple[str, int], dict] = {}
    log_lines: list[str] = []
    progress_re = re.compile(r"\[(\d+)/(\d+)\]")
    detected_eval_id = fallback_eval_id

    def _key(condition: str, run_idx: int) -> tuple[str, int]:
        return (condition, int(run_idx))

    EVT_PREFIX = "__DAS_EVT__"

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line.startswith(EVT_PREFIX):
                try:
                    evt = json.loads(line[len(EVT_PREFIX) :])
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t == "run_start":
                    runs_state[_key(evt["condition"], evt["run_idx"])] = {
                        "condition": evt["condition"],
                        "run_idx": evt["run_idx"],
                        "personas": evt.get("personas", []),
                        "topic": evt.get("topic", ""),
                        "timeline": [],
                        "n_turns": 0,
                        "status": "running",
                    }
                elif t in ("utterance", "intervention"):
                    state = runs_state.get(_key(evt["condition"], evt["run_idx"]))
                    if state is None:
                        # run_start を取りこぼした場合のフォールバック
                        state = {
                            "condition": evt["condition"],
                            "run_idx": evt["run_idx"],
                            "personas": [],
                            "topic": "",
                            "timeline": [],
                            "n_turns": 0,
                            "status": "running",
                        }
                        runs_state[_key(evt["condition"], evt["run_idx"])] = state
                    state["timeline"].append(evt)
                    if t == "utterance":
                        state["n_turns"] = max(state["n_turns"], int(evt.get("turn_id", 0)))
                elif t == "run_end":
                    state = runs_state.get(_key(evt["condition"], evt["run_idx"]))
                    if state is not None:
                        state["status"] = "done"
                        state["n_turns"] = evt.get("n_turns", state.get("n_turns", 0))
                        state["consensus"] = {
                            "reached": evt.get("consensus_reached"),
                            "at": evt.get("consensus_at"),
                            "signal": evt.get("consensus_signal"),
                        }

                # 議論ビューを再描画
                _render_all_runs(discussion_area, runs_state)

            else:
                # 生ログ
                log_lines.append(line)
                log_box.code("\n".join(log_lines[-60:]) or "(出力なし)")

                # 進捗
                m = progress_re.search(line)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    if total > 0:
                        progress.progress(done / total, text=f"{done}/{total} ラン完了")

                # 自動採番された eval_id を捕捉
                if "eval_id=" in line:
                    m2 = re.search(r"eval_id=([^\s]+)", line)
                    if m2:
                        detected_eval_id = m2.group(1)
    finally:
        rc = proc.wait()

    progress.progress(1.0, text="完了")
    # 最終描画
    _render_all_runs(discussion_area, runs_state)

    if rc == 0:
        eval_path = settings.data_dir / "eval" / detected_eval_id
        st.success(
            f"✅ 完了: `{detected_eval_id}` ({eval_path}) — メインの「議論レビュー」をリロードしてください"
        )
    else:
        st.error(f"❌ 失敗 (return code {rc})。下のログを確認してください。")
        log_expander.code("\n".join(log_lines[-200:]) or "(出力なし)")
