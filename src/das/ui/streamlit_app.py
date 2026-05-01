"""議論レビュー (メインページ)。

研究計画書 §4.4 に対応する「結果を読む / 分析する」ための単一画面 UI。

設計:
  - サイドバーで ``eval_id`` を 1 つ選ぶ
  - 上部に eval 全体のメタ情報 (トピック, 参加ペルソナ)
  - 条件 (full_proposal / none / flat_rag) はタブで切替。提案手法 = 1 番目
  - 各条件タブで: ラン選択 → 実行サマリ → 議論タイムライン (介入インライン) →
    評価エージェントの評価 → 折りたたみで AF stats / snapshot 可視化 / 全介入ログ
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from das.settings import get_settings
from das.viz import load_snapshot, render_html

st.set_page_config(
    page_title="議論レビュー",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ヘルパ -----------------------------------------------------------

SUBJECTIVE_METRICS: list[tuple[str, str]] = [
    ("overall_satisfaction", "満足度"),
    ("information_usefulness", "情報有用性"),
    ("opposition_understanding", "反対理解"),
    ("confidence_change", "自信変化"),
    ("intervention_transparency", "介入透明性"),
]

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


def _list_runs(eval_dir: Path, condition: str) -> list[Path]:
    cond_dir = eval_dir / condition
    if not cond_dir.exists():
        return []
    return sorted(p for p in cond_dir.iterdir() if p.is_dir())


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
        if line.strip()
    ]


# 条件タブを「提案手法を先頭」に並べる
def _ordered_conditions(meta: dict, eval_dir: Path) -> list[str]:
    declared = meta.get("condition_names") or []
    available = [
        d.name
        for d in eval_dir.iterdir()
        if d.is_dir() and d.name not in {"__pycache__"} and not d.name.startswith(".")
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    # 提案手法を先頭
    if "full_proposal" in declared or "full_proposal" in available:
        ordered.append("full_proposal")
        seen.add("full_proposal")
    for name in declared + sorted(available):
        if name in seen:
            continue
        if name not in available:
            continue
        ordered.append(name)
        seen.add(name)
    return ordered


# --- サイドバー: eval 選択 -----------------------------------------------

settings = get_settings()
default_eval_root = settings.data_dir / "eval"

with st.sidebar:
    st.header("Eval 選択")
    eval_dirs = _list_eval_dirs(default_eval_root)
    if not eval_dirs:
        st.warning(
            f"`{default_eval_root}` に評価結果がありません。\n\n"
            "`uv run das eval cafeteria -n 3 --until-consensus` で生成できます。"
        )
        st.stop()
    names = [p.name for p in eval_dirs]
    selected_name = st.selectbox("eval_id", names, index=0)
    eval_dir = next(p for p in eval_dirs if p.name == selected_name)
    st.caption(f"`{eval_dir}`")
    st.divider()
    st.caption(
        "他の集計指標は **集計** ページへ。条件のサマリ比較はそちらで。"
    )

# --- ヘッダー: eval メタ -------------------------------------------------

meta = _load_json(eval_dir / "meta.json")
summary = _load_json(eval_dir / "summary.json")

st.title("議論レビュー")
topic = meta.get("topic", "(no topic)")
st.markdown(f"### 📋 {topic}")

personas = meta.get("personas", [])
if personas:
    persona_chips = "  ".join(
        f"**{p.get('name', '?')}** {STANCE_BADGE.get(p.get('stance', 'neutral'), '⚪')} "
        f"({p.get('focus', '-')})"
        for p in personas
    )
    st.markdown(persona_chips)

stat_cols = st.columns(4)
stat_cols[0].metric("総ラン数", summary.get("n_runs_total", 0))
stat_cols[1].metric("条件数", len(summary.get("by_condition", {})))
stat_cols[2].metric("ターン上限", meta.get("max_turns", "-"))
stat_cols[3].metric(
    "until_consensus", "ON" if meta.get("until_consensus") else "OFF"
)

# --- 条件タブ ------------------------------------------------------------

cond_order = _ordered_conditions(meta, eval_dir)
if not cond_order:
    st.error("条件ディレクトリが見つかりません。")
    st.stop()

tab_labels = [CONDITION_LABELS.get(c, c) + (" ★" if i == 0 else "") for i, c in enumerate(cond_order)]
tabs = st.tabs(tab_labels)

for tab, cond in zip(tabs, cond_order, strict=True):
    with tab:
        runs = _list_runs(eval_dir, cond)
        if not runs:
            st.info(f"{cond} のランがありません。")
            continue

        # ラン選択 (複数あればセレクト)
        if len(runs) == 1:
            run_dir = runs[0]
            st.caption(f"ラン: `{run_dir.name}`")
        else:
            run_name = st.selectbox(
                "ラン",
                [p.name for p in runs],
                key=f"run-select-{cond}",
            )
            run_dir = next(p for p in runs if p.name == run_name)

        run_meta = _load_json(run_dir / "run_meta.json")
        transcript = _load_jsonl(run_dir / "transcript.jsonl")
        interventions = _load_jsonl(run_dir / "interventions.jsonl")
        judge_reports = json.loads(
            (run_dir / "judge_reports.json").read_text(encoding="utf-8")
        ) if (run_dir / "judge_reports.json").exists() else []

        # --- ラン要約バー ----------------------------------------------
        cons = run_meta.get("consensus") or {}
        consensus_reached = cons.get("reached")
        cb = st.columns(5)
        if consensus_reached:
            cb[0].metric("合意", "✅", help=cons.get("rationale", ""))
            cb[1].metric("到達ターン", cons.get("detected_at_turn") or "-")
        else:
            cb[0].metric("合意", "—")
            cb[1].metric("到達ターン", "-")
        cb[2].metric(
            "ターン",
            f"{run_meta.get('n_turns', len(transcript))}/{meta.get('max_turns', '-')}",
        )
        snapshot_path = run_dir / "snapshot.json"
        if snapshot_path.exists():
            snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
            n_nodes = len(snap.get("nodes", []))
            n_edges = len(snap.get("edges", []))
            n_supp = sum(1 for e in snap["edges"] if e.get("relation") == "support")
            n_atk = sum(1 for e in snap["edges"] if e.get("relation") == "attack")
            cb[3].metric("AF ノード/エッジ", f"{n_nodes} / {n_edges}")
            cb[4].metric("支持 / 反論", f"{n_supp} / {n_atk}")
        else:
            cb[3].metric("AF ノード/エッジ", "-")
            cb[4].metric("支持 / 反論", "-")

        # 介入分布 (full_proposal のみ意味がある)
        if interventions:
            n_iv = len(interventions)
            n_l1 = sum(1 for e in interventions if e.get("kind") == "l1")
            n_l2 = sum(1 for e in interventions if e.get("kind") == "l2")
            n_skip = sum(1 for e in interventions if e.get("kind") == "skip")
            ic = st.columns(4)
            ic[0].metric("介入記録数", n_iv)
            ic[1].metric("L1 (個別)", f"{n_l1} ({n_l1 / n_iv:.0%})")
            ic[2].metric("L2 (俯瞰)", f"{n_l2} ({n_l2 / n_iv:.0%})")
            ic[3].metric("skip", f"{n_skip} ({n_skip / n_iv:.0%})")

        st.divider()

        # --- 議論タイムライン (介入インライン) -------------------------
        st.markdown("### 議論タイムライン")
        # turn_id → intervention の辞書 (intervention は当該発話の「後」に出される)
        iv_by_turn: dict[int, list[dict]] = {}
        for entry in interventions:
            iv_by_turn.setdefault(entry["turn_id"], []).append(entry)

        consensus_turn = cons.get("detected_at_turn")

        for utt in transcript:
            stance = next(
                (
                    p.get("stance", "neutral")
                    for p in personas
                    if p.get("name") == utt["speaker"]
                ),
                "neutral",
            )
            with st.container(border=True):
                tc = st.columns([1, 9])
                tc[0].markdown(
                    f"**t{utt['turn_id']}**\n\n"
                    f"**{utt['speaker']}**\n\n{STANCE_BADGE.get(stance, '⚪')}"
                )
                tc[1].markdown(utt["text"])

            # この発話の後に出された介入 (= 次話者へ提示された情報)
            for entry in iv_by_turn.get(utt["turn_id"], []):
                kind = entry.get("kind", "l1")
                if kind == "skip":
                    continue  # skip はタイムラインに出さない (折りたたみで全件確認可能)
                if kind == "l2":
                    with st.container(border=True):
                        st.markdown(
                            f"🟪 **L2 議論の整理 (全体共有)** — _{entry.get('decision_reason', '')}_"
                        )
                        if entry.get("brief"):
                            st.markdown(f"> {entry['brief']}")
                else:  # l1
                    addressed = entry.get("addressed_to") or "次話者"
                    items = entry.get("items", [])
                    if not items:
                        continue
                    with st.container(border=True):
                        st.markdown(
                            f"🟦 **L1 → {addressed}** "
                            f"({len(items)} 件)"
                        )
                        for it in items:
                            tag = "🟢 支持" if it.get("relation") == "support" else "🔴 反論"
                            kind_src = it.get("source_kind", "?")
                            st.markdown(
                                f"- {tag} ({kind_src}): {it.get('source_text', '')}"
                            )

            if consensus_turn and utt["turn_id"] == consensus_turn:
                st.success(
                    f"🎉 **合意到達** (turn {consensus_turn} / signal: "
                    f"{cons.get('signal', '?')})"
                )

        # --- 評価エージェントの評価 ------------------------------------
        st.divider()
        st.markdown("### 評価エージェントの評価")
        if not judge_reports:
            st.info(
                "judge_reports.json がありません。`--no-judge` で走った可能性があります。"
            )
        else:
            score_rows: list[dict] = []
            for rep in judge_reports:
                row = {"persona": rep["persona_name"]}
                scores = rep.get("scores", {})
                for key, label in SUBJECTIVE_METRICS:
                    row[label] = scores.get(key, "-")
                score_rows.append(row)
            score_df = pd.DataFrame(score_rows).set_index("persona")
            st.dataframe(score_df, use_container_width=True)

            with st.expander("各ペルソナのラショナル", expanded=False):
                for rep in judge_reports:
                    rationale = rep.get("scores", {}).get("rationale")
                    if rationale:
                        st.markdown(f"**{rep['persona_name']}**: _{rationale}_")

        # --- 折りたたみ: 詳細データ ----------------------------------
        st.divider()
        with st.expander("AF スナップショット (グラフ可視化)", expanded=False):
            if not snapshot_path.exists():
                st.caption("snapshot.json なし")
            else:
                if st.button(
                    "グラフを描画 (pyvis)",
                    key=f"render-graph-{cond}-{run_dir.name}",
                ):
                    store = load_snapshot(snapshot_path)
                    with tempfile.NamedTemporaryFile(
                        suffix=".html", delete=False
                    ) as f:
                        path = Path(f.name)
                    try:
                        render_html(store, path)
                        components.html(
                            path.read_text(encoding="utf-8"),
                            height=600,
                            scrolling=False,
                        )
                    finally:
                        path.unlink(missing_ok=True)
                else:
                    st.caption(
                        "ノード/エッジが多いと描画に時間がかかるので、ボタンで明示起動してください。"
                    )

        with st.expander("全 interventions JSONL", expanded=False):
            if not interventions:
                st.caption("介入なし")
            else:
                for entry in interventions:
                    kind = entry.get("kind", "l1")
                    badge = {
                        "l1": "🟦 L1",
                        "l2": "🟪 L2",
                        "skip": "⬜ skip",
                    }.get(kind, kind)
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

        with st.expander("transcript.jsonl (raw)", expanded=False):
            st.code(
                "\n".join(json.dumps(u, ensure_ascii=False) for u in transcript),
                language="json",
            )


def main() -> None:
    """``uv run das ui`` から呼ばれる薄い entrypoint。

    Streamlit はモジュール全体を実行モデルなので、上のトップレベルコードが
    そのままページのレンダリングを行う。``main`` は CLI 経由起動の互換用。
    """

    # 既にトップレベルでページが組み立てられているので no-op で OK
