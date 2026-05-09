"""議論支援システム — 統合 UI。

サイドバーで「新規実行」と「過去の結果を見る」を切替えるだけ。
メインは縦 1 列で:

  [ヘッダ: トピック + 参加者]
        ↓
  [条件タブ: 提案手法 / 情報なし / Flat RAG]
        ↓
  [実行サマリ: 合意 / ターン進捗 / AF サイズ / 介入分布]
        ↓
  [議論タイムライン (発話 + 介入インライン)]
        ↓
  [折りたたみ: AF 可視化 / interventions / transcript]
        ↓
  [評価セクション (完了時のみ)]
        - ジャッジ評価
        - 客観構造指標
        - 条件横断比較

設計の要:
  - 新規実行は ``das eval --emit-events`` をサブプロセス起動
  - イベント (utterance / intervention / run_start / run_end) を
    ``__DAS_EVT__<json>`` 行で受け取り、議論をライブ表示
  - 完了後は同じ画面の下部に評価が現れる
  - 過去の eval を選ぶと、同じ表示が静的データから再構成される
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from das.settings import get_settings
from das.viz import load_snapshot, render_html


# --- 定数 ---------------------------------------------------------------

st.set_page_config(
    page_title="議論支援システム",
    layout="wide",
    initial_sidebar_state="expanded",
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
    "partial_pro": "🟢 部分賛成",
    "partial_con": "🔴 部分反対",
}

SUBJECTIVE_METRICS: list[tuple[str, str]] = [
    ("overall_satisfaction", "満足度"),
    ("information_usefulness", "情報有用性"),
    ("opposition_understanding", "反対理解"),
    ("confidence_change", "自信変化"),
    ("intervention_transparency", "介入透明性"),
]

EVT_PREFIX = "__DAS_EVT__"


# --- ヘルパ -------------------------------------------------------------


def _list_eval_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "summary.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
        if line.strip()
    ]


def _list_runs(eval_dir: Path, condition: str) -> list[Path]:
    cond_dir = eval_dir / condition
    if not cond_dir.exists():
        return []
    return sorted(p for p in cond_dir.iterdir() if p.is_dir())


def _load_eval_into_state(eval_dir: Path) -> None:
    """既存 eval を読み込み、session_state に組み立てる。"""

    meta = _load_json(eval_dir / "meta.json")
    summary = _load_json(eval_dir / "summary.json")
    runs_state: dict[tuple[str, int], dict] = {}

    declared_conds = meta.get("condition_names") or []
    available_conds = [
        d.name
        for d in eval_dir.iterdir()
        if d.is_dir() and d.name not in {"__pycache__"} and not d.name.startswith(".")
    ]

    for cond in declared_conds + sorted(available_conds):
        if cond not in available_conds:
            continue
        for run_dir in _list_runs(eval_dir, cond):
            run_idx = int(run_dir.name.split("_")[-1]) if "_" in run_dir.name else 1
            run_meta = _load_json(run_dir / "run_meta.json")
            transcript = _load_jsonl(run_dir / "transcript.jsonl")
            interventions = _load_jsonl(run_dir / "interventions.jsonl")
            judge = (
                json.loads((run_dir / "judge_reports.json").read_text(encoding="utf-8"))
                if (run_dir / "judge_reports.json").exists()
                else []
            )
            snapshot_path = run_dir / "snapshot.json"
            snap_stats = None
            if snapshot_path.exists():
                snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
                snap_stats = {
                    "n_nodes": len(snap.get("nodes", [])),
                    "n_edges": len(snap.get("edges", [])),
                    "n_support": sum(
                        1 for e in snap.get("edges", []) if e.get("relation") == "support"
                    ),
                    "n_attack": sum(
                        1 for e in snap.get("edges", []) if e.get("relation") == "attack"
                    ),
                }

            timeline: list[dict] = []
            iv_by_turn: dict[int, list[dict]] = {}
            for entry in interventions:
                iv_by_turn.setdefault(entry["turn_id"], []).append(entry)
            for u in transcript:
                # 発話前に出された介入を先に
                for iv in iv_by_turn.get(u["turn_id"] - 1, []):
                    if iv.get("kind") != "skip":
                        timeline.append(
                            {
                                "type": "intervention",
                                "kind": iv.get("kind", "l1"),
                                "items": iv.get("items", []),
                                "addressed_to": iv.get("addressed_to"),
                                "brief": iv.get("brief", ""),
                                "decision_reason": iv.get("decision_reason", ""),
                            }
                        )
                timeline.append(
                    {
                        "type": "utterance",
                        "turn_id": u["turn_id"],
                        "speaker": u["speaker"],
                        "text": u["text"],
                    }
                )

            cons = run_meta.get("consensus") or {}
            runs_state[(cond, run_idx)] = {
                "condition": cond,
                "run_idx": run_idx,
                "personas": meta.get("personas", []),
                "topic": meta.get("topic", ""),
                "timeline": timeline,
                "n_turns": run_meta.get("n_turns", len(transcript)),
                "status": "done",
                "consensus": {
                    "reached": cons.get("reached"),
                    "at": cons.get("detected_at_turn"),
                    "signal": cons.get("signal"),
                    "rationale": cons.get("rationale", ""),
                },
                "judge_reports": judge,
                "snap_stats": snap_stats,
                "snapshot_path": str(snapshot_path) if snapshot_path.exists() else None,
                "interventions_raw": interventions,
            }

    st.session_state.runs_state = runs_state
    st.session_state.eval_meta = meta
    st.session_state.eval_summary = summary
    st.session_state.eval_dir = str(eval_dir)
    st.session_state.completed = True
    st.session_state.executing = False


# --- レンダラ -----------------------------------------------------------


def _stance_for(speaker: str, personas: list[dict]) -> str:
    for p in personas:
        if p.get("name") == speaker:
            return p.get("stance", "neutral")
    return "neutral"


def _render_run_summary_bar(state: dict) -> None:
    """1 ラン上部の status bar。"""

    cb = st.columns(5)
    cons = state.get("consensus") or {}
    if state.get("status") == "running":
        cb[0].metric("状態", "⏳ 進行中")
    elif cons.get("reached"):
        cb[0].metric("状態", "✅ 合意", help=cons.get("rationale", ""))
    elif state.get("status") == "done":
        cb[0].metric("状態", "☑️ 完了 (合意なし)")
    else:
        cb[0].metric("状態", "—")

    cb[1].metric("ターン", state.get("n_turns", 0))

    if cons.get("at"):
        cb[2].metric("合意 turn", cons.get("at"))
    else:
        cb[2].metric("合意 turn", "-")

    snap = state.get("snap_stats") or {}
    if snap:
        cb[3].metric("AF (node/edge)", f"{snap['n_nodes']} / {snap['n_edges']}")
        cb[4].metric("支持 / 反論", f"{snap['n_support']} / {snap['n_attack']}")
    else:
        # 実行中: live でカウント
        n_nodes = state.get("live_n_nodes", 0)
        n_edges = state.get("live_n_edges", 0)
        cb[3].metric("AF (node/edge)", f"{n_nodes} / {n_edges}" if n_edges else "—")
        cb[4].metric("支持 / 反論", "—")


def _render_intervention_distribution(state: dict) -> None:
    interventions = [
        e for e in state["timeline"] if e["type"] == "intervention"
    ] + state.get("skip_records", [])
    if not interventions:
        return
    n = len(interventions)
    n_l1 = sum(1 for e in interventions if e.get("kind") == "l1")
    n_l2 = sum(1 for e in interventions if e.get("kind") == "l2")
    n_skip = n - n_l1 - n_l2
    ic = st.columns(3)
    ic[0].metric("L1 (個別)", f"{n_l1}")
    ic[1].metric("L2 (俯瞰)", f"{n_l2}")
    ic[2].metric("skip", f"{n_skip}")


def _render_timeline(state: dict) -> None:
    personas = state.get("personas", [])
    cons = state.get("consensus") or {}
    consensus_turn = cons.get("at")

    for evt in state["timeline"]:
        if evt["type"] == "utterance":
            stance = _stance_for(evt["speaker"], personas)
            with st.container(border=True):
                c = st.columns([1, 9])
                c[0].markdown(
                    f"**t{evt['turn_id']}**\n\n"
                    f"**{evt['speaker']}**\n\n{STANCE_BADGE.get(stance, '⚪')}"
                )
                c[1].markdown(evt["text"])
            if consensus_turn and evt["turn_id"] == consensus_turn:
                st.success(f"🎉 合意到達 (turn {consensus_turn})")
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
                addressed = evt.get("addressed_to") or "次話者"
                with st.container(border=True):
                    st.markdown(f"🟦 **L1 → {addressed}** ({len(items)} 件)")
                    for it in items:
                        tag = "🟢 支持" if it.get("relation") == "support" else "🔴 反論"
                        kind_src = it.get("source_kind", "?")
                        st.markdown(f"- {tag} ({kind_src}): {it.get('source_text', '')}")


def _render_run_detail(state: dict, *, key_prefix: str) -> None:
    _render_run_summary_bar(state)
    _render_intervention_distribution(state)
    st.markdown("#### 議論タイムライン")
    _render_timeline(state)

    # 折りたたみ
    if state.get("snapshot_path"):
        with st.expander("AF スナップショット (グラフ可視化)", expanded=False):
            if st.button("グラフを描画", key=f"{key_prefix}-graph"):
                store = load_snapshot(Path(state["snapshot_path"]))
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                    path = Path(f.name)
                try:
                    render_html(store, path)
                    components.html(
                        path.read_text(encoding="utf-8"),
                        height=520,
                        scrolling=False,
                    )
                finally:
                    path.unlink(missing_ok=True)
            else:
                st.caption("ノードが多いと描画に時間がかかるためボタン起動。")

    if state.get("interventions_raw"):
        with st.expander("全 interventions (skip 含む)", expanded=False):
            for entry in state["interventions_raw"]:
                kind = entry.get("kind", "l1")
                badge = {"l1": "🟦 L1", "l2": "🟪 L2", "skip": "⬜ skip"}.get(kind, kind)
                st.markdown(
                    f"**turn {entry['turn_id']}** {badge} → "
                    f"{entry.get('addressed_to') or '全員'}"
                )
                if entry.get("decision_reason"):
                    st.caption(f"理由: {entry['decision_reason']}")
                if entry.get("brief"):
                    st.markdown(f"> {entry['brief']}")
                for item in entry.get("items", []):
                    tag = "🟢" if item.get("relation") == "support" else "🔴"
                    reason = item.get("reason", "-")
                    st.markdown(
                        f"- {tag} {item.get('source_text', '')} "
                        f"(`{reason}` / conf {item.get('confidence', 0):.0%})"
                    )


def _render_judge_panel(state: dict, *, key_prefix: str) -> None:
    judge = state.get("judge_reports") or []
    if not judge:
        return
    rows: list[dict] = []
    for rep in judge:
        row: dict = {"persona": rep["persona_name"]}
        scores = rep.get("scores", {})
        for key, label in SUBJECTIVE_METRICS:
            row[label] = scores.get(key, "-")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("persona")
    st.dataframe(df, use_container_width=True)

    with st.expander("ペルソナ別ラショナル", expanded=False):
        for rep in judge:
            rationale = rep.get("scores", {}).get("rationale")
            if rationale:
                st.markdown(f"**{rep['persona_name']}**: _{rationale}_")


def _render_cross_condition_panel() -> None:
    """全条件横断の集計 (画面下部のまとめ)。"""

    summary = st.session_state.get("eval_summary") or {}
    by_cond = summary.get("by_condition") or {}
    if not by_cond:
        return

    st.markdown("### 主観評価 (条件横断)")
    rows: list[dict] = []
    for cond, payload in by_cond.items():
        for key, label in SUBJECTIVE_METRICS:
            mean, std = payload.get(key, [0.0, 0.0])
            rows.append(
                {
                    "condition": CONDITION_LABELS.get(cond, cond),
                    "metric": label,
                    "mean": mean,
                    "low": mean - std,
                    "high": mean + std,
                }
            )
    if rows:
        df = pd.DataFrame(rows)
        bar = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("condition:N", title=None, sort=None),
                y=alt.Y("mean:Q", title="平均"),
                color=alt.Color("condition:N", legend=None),
                column=alt.Column("metric:N", title=None, sort=None),
                tooltip=["condition", "metric", "mean"],
            )
            .properties(width=110, height=160)
        )
        err = (
            alt.Chart(df)
            .mark_errorbar()
            .encode(
                x="condition:N",
                y=alt.Y("low:Q", title="平均"),
                y2="high:Q",
                column=alt.Column("metric:N", title=None, sort=None),
            )
        )
        st.altair_chart(bar + err, use_container_width=False)

    # 収束統計
    conv_rows: list[dict] = []
    for cond, payload in by_cond.items():
        conv = payload.get("convergence") or {}
        if not conv:
            continue
        conv_rows.append(
            {
                "条件": CONDITION_LABELS.get(cond, cond),
                "ラン数": conv.get("n_runs", 0),
                "収束率": f"{conv.get('convergence_rate', 0) * 100:.0f}%",
                "平均ターン": round(conv.get("mean_turns", 0), 1),
                "平均到達ターン": (
                    f"{conv.get('mean_turns_to_consensus'):.1f}"
                    if conv.get("mean_turns_to_consensus") is not None
                    else "-"
                ),
            }
        )
    if conv_rows:
        st.markdown("### 合意形成までの時間 (条件横断)")
        st.dataframe(pd.DataFrame(conv_rows), use_container_width=True, hide_index=True)

    # 構造指標
    struct_rows: list[dict] = []
    for cond, payload in by_cond.items():
        s = payload.get("structural") or {}
        if not s:
            continue
        struct_rows.append(
            {
                "条件": CONDITION_LABELS.get(cond, cond),
                "参加偏在 (gini)": round(s.get("participation_gini_mean", 0), 3),
                "claim あたり premise": round(s.get("avg_premises_per_claim_mean", 0), 2),
                "未根拠 claim 率": f"{s.get('pct_unsupported_claims_mean', 0) * 100:.0f}%",
                "応答率": f"{s.get('response_rate_mean', 0) * 100:.0f}%",
                "再反論率": f"{s.get('pct_attacks_answered_mean', 0) * 100:.0f}%",
                "論証連鎖の深さ": round(s.get("avg_argument_chain_length_mean", 0), 2),
            }
        )
    if struct_rows:
        st.markdown("### 客観構造指標 (条件横断)")
        st.dataframe(pd.DataFrame(struct_rows), use_container_width=True, hide_index=True)


# --- 実行ロジック -------------------------------------------------------


def _build_command(*, preset: str, n_runs: int, max_turns: int, concurrency: int,
                   conditions: list[str], until_consensus: bool,
                   agreement_threshold: float, agreement_window: int,
                   no_judge: bool, web_search_enabled: bool,
                   max_web_searches: int, eval_id: str) -> list[str]:
    cmd = [
        sys.executable, "-m", "das.cli", "eval", preset,
        "-n", str(n_runs), "-t", str(max_turns), "-j", str(concurrency),
        "--emit-events",
    ]
    if conditions:
        cmd += ["--conditions", ",".join(conditions)]
    if until_consensus:
        cmd += [
            "--until-consensus",
            "--agreement-threshold", str(agreement_threshold),
            "--agreement-window", str(agreement_window),
        ]
    if no_judge:
        cmd += ["--no-judge"]
    if web_search_enabled:
        cmd += ["--web-search", "--max-web-searches", str(max_web_searches)]
    if eval_id:
        cmd += ["--eval-id", eval_id]
    return cmd


def _run_eval_streaming(cmd: list[str], placeholder: Any, log_box: Any,
                        progress: Any) -> tuple[int, str | None]:
    """サブプロセスを起動して stdout を逐次パースし session_state を更新。

    戻り値: (return_code, detected_eval_id)
    """

    proc = subprocess.Popen(
        cmd, cwd=str(Path.cwd()),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    log_lines: list[str] = []
    progress_re = re.compile(r"\[(\d+)/(\d+)\]")
    detected: str | None = None
    runs_state: dict[tuple[str, int], dict] = st.session_state.runs_state

    def _key(c: str, idx: int) -> tuple[str, int]:
        return (c, int(idx))

    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line.startswith(EVT_PREFIX):
                try:
                    evt = json.loads(line[len(EVT_PREFIX):])
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
                        "skip_records": [],
                        "n_turns": 0,
                        "status": "running",
                        "consensus": None,
                        "snap_stats": None,
                        "snapshot_path": None,
                        "interventions_raw": [],
                        "judge_reports": [],
                        "live_n_nodes": 0,
                        "live_n_edges": 0,
                    }
                elif t in ("utterance", "intervention"):
                    state = runs_state.get(_key(evt["condition"], evt["run_idx"]))
                    if state is None:
                        continue
                    if t == "utterance":
                        state["timeline"].append(evt)
                        state["n_turns"] = max(
                            state["n_turns"], int(evt.get("turn_id", 0))
                        )
                    else:  # intervention
                        state["interventions_raw"].append(evt)
                        if evt.get("kind") == "skip":
                            state["skip_records"].append(evt)
                        else:
                            state["timeline"].append(evt)
                elif t == "run_end":
                    state = runs_state.get(_key(evt["condition"], evt["run_idx"]))
                    if state is not None:
                        state["status"] = "done"
                        state["n_turns"] = evt.get("n_turns", state.get("n_turns", 0))
                        state["consensus"] = {
                            "reached": evt.get("consensus_reached"),
                            "at": evt.get("consensus_at"),
                            "signal": evt.get("consensus_signal"),
                            "rationale": "",
                        }
                # ライブ再描画 (重いので簡略表示のみ更新)
                _refresh_live_view(placeholder)
            else:
                log_lines.append(line)
                log_box.code("\n".join(log_lines[-30:]) or "")
                m = progress_re.search(line)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    if total > 0:
                        progress.progress(done / total, text=f"{done}/{total} ラン完了")
                if "eval_id=" in line:
                    m2 = re.search(r"eval_id=([^\s]+)", line)
                    if m2:
                        detected = m2.group(1)
    finally:
        rc = proc.wait()
    return rc, detected


def _refresh_live_view(placeholder: Any) -> None:
    """実行中のタブ状態をプレースホルダに再描画。"""

    runs_state: dict[tuple[str, int], dict] = st.session_state.runs_state
    if not runs_state:
        return

    by_condition: dict[str, list[dict]] = {}
    for (cond, _idx), state in sorted(runs_state.items()):
        by_condition.setdefault(cond, []).append(state)
    cond_order = [c for c in CONDITIONS if c in by_condition]

    with placeholder.container():
        if not cond_order:
            return
        labels = [
            CONDITION_LABELS.get(c, c) + (" ★" if c == "full_proposal" else "")
            for c in cond_order
        ]
        tabs = st.tabs(labels)
        for tab, cond in zip(tabs, cond_order, strict=True):
            with tab:
                for state in by_condition[cond]:
                    label = (
                        f"run {state['run_idx']:02d} "
                        f"({'⏳ 進行中' if state['status'] == 'running' else '✅ 完了'})"
                    )
                    with st.expander(label, expanded=(state["status"] == "running")):
                        _render_run_detail(state, key_prefix=f"live-{cond}-{state['run_idx']}")


# --- セッション初期化 ---------------------------------------------------

if "runs_state" not in st.session_state:
    st.session_state.runs_state = {}
if "eval_meta" not in st.session_state:
    st.session_state.eval_meta = {}
if "eval_summary" not in st.session_state:
    st.session_state.eval_summary = {}
if "eval_dir" not in st.session_state:
    st.session_state.eval_dir = None
if "completed" not in st.session_state:
    st.session_state.completed = False
if "executing" not in st.session_state:
    st.session_state.executing = False
if "selected_existing" not in st.session_state:
    st.session_state.selected_existing = None


# --- サイドバー (モード切替) -------------------------------------------

settings = get_settings()
default_eval_root = settings.data_dir / "eval"

with st.sidebar:
    st.markdown("### 議論支援システム")
    mode = st.radio(
        "モード",
        ["新しい評価を実行", "過去の結果を見る"],
        index=0,
        label_visibility="collapsed",
    )

    if mode == "新しい評価を実行":
        st.markdown("#### 新規実行")
        preset = st.selectbox(
            "トピック",
            list(PRESETS.keys()),
            format_func=lambda k: PRESETS[k],
        )
        col_a, col_b = st.columns(2)
        n_runs = col_a.number_input("ラン数", 1, 10, 2, 1)
        max_turns = col_b.number_input("ターン上限", 5, 40, 20, 1)
        concurrency = st.slider("並列度", 1, 8, 2)
        conditions = st.multiselect(
            "条件",
            CONDITIONS,
            default=CONDITIONS,
            format_func=lambda c: CONDITION_LABELS[c],
        )
        until_consensus = st.checkbox("合意検出で早期終了", value=True)
        with st.expander("詳細設定", expanded=False):
            agreement_threshold = st.slider("合意しきい値", 0.5, 0.95, 0.67, 0.01)
            agreement_window = st.slider("合意判定の直近ターン", 2, 8, 3)
            no_judge = st.checkbox("LLM-as-judge をスキップ (高速)")
            web_search_enabled = st.checkbox("Web 検索を有効化 (Tavily)")
            max_web_searches = st.slider("Web 検索上限/セッション", 1, 10, 3)
            eval_id_input = st.text_input("eval_id (空なら自動)")
        run_clicked = st.button("▶ 実行", type="primary", use_container_width=True)
    else:
        run_clicked = False
        eval_dirs = _list_eval_dirs(default_eval_root)
        if not eval_dirs:
            st.info("まだ評価結果がありません。「新しい評価を実行」から始めてください。")
            st.stop()
        names = [p.name for p in eval_dirs]
        selected_name = st.selectbox(
            "eval_id",
            names,
            index=0,
        )
        if st.session_state.selected_existing != selected_name:
            st.session_state.selected_existing = selected_name
            _load_eval_into_state(default_eval_root / selected_name)


# --- 実行ボタンが押されたら ---------------------------------------------

if mode == "新しい評価を実行" and run_clicked:
    if not conditions:
        st.error("条件を 1 つ以上選んでください。")
        st.stop()
    fallback_eval_id = (
        eval_id_input.strip()
        or f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    cmd = _build_command(
        preset=preset,
        n_runs=n_runs,
        max_turns=max_turns,
        concurrency=concurrency,
        conditions=conditions,
        until_consensus=until_consensus,
        agreement_threshold=agreement_threshold,
        agreement_window=agreement_window,
        no_judge=no_judge,
        web_search_enabled=web_search_enabled,
        max_web_searches=max_web_searches,
        eval_id=eval_id_input.strip(),
    )

    # フレッシュな state
    st.session_state.runs_state = {}
    st.session_state.eval_meta = {"topic": PRESETS[preset]}
    st.session_state.eval_summary = {}
    st.session_state.completed = False
    st.session_state.executing = True

    st.markdown(f"### 📋 {PRESETS[preset]}")

    progress = st.progress(0.0, text="準備中...")
    timeline_placeholder = st.empty()
    log_expander = st.expander("CLI 生ログ (デバッグ用)", expanded=False)
    log_box = log_expander.empty()

    rc, detected_eval_id = _run_eval_streaming(
        cmd, timeline_placeholder, log_box, progress
    )
    progress.progress(1.0, text="完了")

    eval_id = detected_eval_id or fallback_eval_id
    eval_path = settings.data_dir / "eval" / eval_id

    if rc == 0 and eval_path.exists():
        # 完了後は disk から再読込して static 表示に切り替え (judge / structural を含む)
        _load_eval_into_state(eval_path)
        st.session_state.selected_existing = eval_id
        st.success(f"✅ 完了: `{eval_id}`")
    else:
        st.error(f"❌ 失敗 (return code {rc})")
        st.session_state.executing = False


# --- メインビュー -------------------------------------------------------

if not st.session_state.runs_state:
    st.markdown("# 議論支援システム")
    st.info(
        "サイドバーで **新しい評価を実行** または **過去の結果を見る** を選んでください。"
    )
    st.stop()


# ヘッダ
meta = st.session_state.eval_meta or {}
topic = meta.get("topic", "(no topic)")
st.markdown(f"# 📋 {topic}")

personas = meta.get("personas", [])
if personas:
    chips = "  ".join(
        f"**{p.get('name', '?')}** {STANCE_BADGE.get(p.get('stance', 'neutral'), '⚪')} "
        f"({p.get('focus', '-')})"
        for p in personas
    )
    st.markdown(chips)

# 進捗状態
if st.session_state.executing:
    st.info("⏳ 実行中…")
elif st.session_state.completed:
    eval_dir = st.session_state.eval_dir
    st.caption(f"📚 結果: `{Path(eval_dir).name if eval_dir else '?'}`")


# --- 条件タブ -----------------------------------------------------------

runs_state: dict[tuple[str, int], dict] = st.session_state.runs_state
by_condition: dict[str, list[dict]] = {}
for (cond, _idx), state in sorted(runs_state.items()):
    by_condition.setdefault(cond, []).append(state)
cond_order = [c for c in CONDITIONS if c in by_condition]

if cond_order:
    labels = [
        CONDITION_LABELS.get(c, c) + (" ★" if c == "full_proposal" else "")
        for c in cond_order
    ]
    tabs = st.tabs(labels)
    for tab, cond in zip(tabs, cond_order, strict=True):
        with tab:
            states = by_condition[cond]
            if len(states) > 1:
                # 複数ランがあるとき selectbox で切替
                idx_options = [s["run_idx"] for s in states]
                selected_idx = st.selectbox(
                    "ラン",
                    idx_options,
                    format_func=lambda i: f"run {i:02d}",
                    key=f"runsel-{cond}",
                )
                state = next(s for s in states if s["run_idx"] == selected_idx)
            else:
                state = states[0]
            _render_run_detail(state, key_prefix=f"static-{cond}-{state['run_idx']}")

            # ジャッジ評価 (このラン)
            if state.get("judge_reports"):
                st.markdown("#### 評価エージェントによる評価 (このラン)")
                _render_judge_panel(state, key_prefix=f"judge-{cond}-{state['run_idx']}")


# --- 条件横断の集計 (画面下部) -----------------------------------------

if st.session_state.completed:
    st.divider()
    _render_cross_condition_panel()
