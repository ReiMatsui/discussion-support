"""評価ダッシュボード (M2.7)。

``data/eval/<eval_id>/`` を読んで、3 条件比較を可視化する:
  - 主観指標 (LLM-as-judge による 5 項目) の bar/box plot
  - 構造指標 (transcript / graph) の比較
  - 個別ランの transcript・judge rationale をドリルダウン

セッションを実際に走らせるのは ``das eval`` CLI または ``2_Compare`` ページに任せ、
このページは「読み込んで解析する」ことに集中する。
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from das.settings import get_settings

st.set_page_config(page_title="評価ダッシュボード", layout="wide")
st.title("評価ダッシュボード")
st.caption("data/eval/<eval_id>/ を読み込んで条件別指標を比較する")


# --- ヘルパ ---------------------------------------------------------


def _list_eval_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "summary.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )


def _load_summary(eval_dir: Path) -> dict:
    return json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))


def _load_meta(eval_dir: Path) -> dict:
    meta_path = eval_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _list_runs(eval_dir: Path, condition: str) -> list[Path]:
    cond_dir = eval_dir / condition
    if not cond_dir.exists():
        return []
    return sorted(p for p in cond_dir.iterdir() if p.is_dir())


SUBJECTIVE_METRICS: list[tuple[str, str]] = [
    ("overall_satisfaction", "議論満足度"),
    ("information_usefulness", "情報の有用性"),
    ("opposition_understanding", "反対意見の理解度"),
    ("confidence_change", "立場の自信度変化"),
    ("intervention_transparency", "介入の透明性"),
]


# --- サイドバー -----------------------------------------------------

settings = get_settings()
default_eval_root = settings.data_dir / "eval"

with st.sidebar:
    st.header("Eval 選択")
    eval_root_str = st.text_input(
        "Eval ディレクトリ",
        value=str(default_eval_root),
        help="data/eval をルートとして、サブディレクトリ (eval_id) を選択する",
    )
    eval_root = Path(eval_root_str)
    eval_dirs = _list_eval_dirs(eval_root)
    if not eval_dirs:
        st.warning(
            f"`{eval_root}` に評価結果がありません。\n\n"
            "`uv run das eval cafeteria -n 2` で生成できます。"
        )
        st.stop()
    names = [p.name for p in eval_dirs]
    selected_name = st.selectbox("eval_id", names, index=0)
    eval_dir = next(p for p in eval_dirs if p.name == selected_name)
    st.caption(f"path: `{eval_dir}`")

# --- メタ情報 -------------------------------------------------------

meta = _load_meta(eval_dir)
summary = _load_summary(eval_dir)

cols = st.columns(4)
cols[0].metric("eval_id", summary.get("eval_id", "-"))
cols[1].metric("条件数", len(summary.get("by_condition", {})))
cols[2].metric("総ラン数", summary.get("n_runs_total", 0))
cols[3].metric("ターン上限", meta.get("max_turns", "-"))

if meta.get("topic"):
    st.markdown(f"**トピック:** {meta['topic']}")
if meta.get("personas"):
    persona_names = ", ".join(p.get("name", "?") for p in meta["personas"])
    st.caption(f"参加ペルソナ: {persona_names}")
if meta.get("until_consensus"):
    st.caption("⏹ until_consensus: 合意検出で早期終了 (max_turns は安全上限)")

# --- 収束統計 -------------------------------------------------------

by_cond = summary.get("by_condition", {})
convergence_rows: list[dict] = []
for cond, payload in by_cond.items():
    conv = payload.get("convergence")
    if not conv:
        continue
    convergence_rows.append(
        {
            "condition": cond,
            "n_runs": conv.get("n_runs", 0),
            "n_converged": conv.get("n_converged", 0),
            "convergence_rate": conv.get("convergence_rate", 0.0),
            "mean_turns": conv.get("mean_turns", 0.0),
            "mean_turns_to_consensus": conv.get("mean_turns_to_consensus"),
            "signals": conv.get("signals", {}),
        }
    )
if convergence_rows:
    st.divider()
    st.markdown("## 合意形成までの時間")
    conv_df = pd.DataFrame(convergence_rows)
    display_df = conv_df.assign(
        convergence_rate=lambda d: (d["convergence_rate"] * 100).round(1).astype(str) + "%",
        mean_turns=lambda d: d["mean_turns"].round(2),
        mean_turns_to_consensus=lambda d: d["mean_turns_to_consensus"].apply(
            lambda v: "-" if v is None else f"{float(v):.2f}"
        ),
        signals=lambda d: d["signals"].apply(
            lambda sig: ", ".join(f"{k}:{v}" for k, v in sig.items()) if sig else "-"
        ),
    )
    st.dataframe(
        display_df.rename(
            columns={
                "condition": "条件",
                "n_runs": "ラン数",
                "n_converged": "収束数",
                "convergence_rate": "収束率",
                "mean_turns": "平均ターン",
                "mean_turns_to_consensus": "平均到達ターン",
                "signals": "シグナル内訳",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    chart = (
        alt.Chart(conv_df)
        .transform_calculate(rate="datum.convergence_rate * 100")
        .mark_bar()
        .encode(
            x=alt.X("condition:N", title="条件", sort=None),
            y=alt.Y("rate:Q", title="収束率 (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("condition:N", legend=None),
            tooltip=[
                "condition",
                alt.Tooltip("rate:Q", title="収束率(%)", format=".1f"),
                alt.Tooltip("mean_turns:Q", title="平均ターン", format=".2f"),
            ],
        )
        .properties(height=180)
    )
    st.altair_chart(chart, use_container_width=True)

# --- 主観指標の比較 -------------------------------------------------

st.divider()
st.markdown("## 主観指標 (LLM-as-judge)")
if not by_cond:
    st.info("by_condition が空です。--no-judge で走らせた eval ではジャッジ結果がありません。")
else:
    rows: list[dict] = []
    for cond, payload in by_cond.items():
        for key, label in SUBJECTIVE_METRICS:
            mean, std = payload.get(key, [0.0, 0.0])
            rows.append(
                {
                    "condition": cond,
                    "metric": label,
                    "metric_key": key,
                    "mean": mean,
                    "std": std,
                    "low": mean - std,
                    "high": mean + std,
                }
            )
    df = pd.DataFrame(rows)

    bar = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("condition:N", title="条件", sort=None),
            y=alt.Y("mean:Q", title="平均"),
            color=alt.Color("condition:N", legend=None),
            column=alt.Column("metric:N", title=None, sort=None),
            tooltip=["condition", "metric", "mean", "std"],
        )
        .properties(width=120, height=180)
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

    st.markdown("### 数値テーブル")
    pivot = df.pivot_table(
        index="metric", columns="condition", values=["mean", "std"], aggfunc="first"
    )
    st.dataframe(pivot, use_container_width=True)

# --- ラン詳細ドリルダウン ------------------------------------------

st.divider()
st.markdown("## ラン詳細")
condition_names = list(by_cond.keys()) or [
    p.name for p in eval_dir.iterdir() if p.is_dir() and p.name != "__pycache__"
]
if not condition_names:
    st.info("ラン詳細なし。")
    st.stop()

selected_cond = st.selectbox("条件", condition_names, key="run_drill_cond")
runs = _list_runs(eval_dir, selected_cond)
if not runs:
    st.info(f"{selected_cond} にランディレクトリがありません。")
    st.stop()
run_names = [p.name for p in runs]
selected_run_name = st.selectbox("ラン", run_names, key="run_drill_run")
run_dir = next(p for p in runs if p.name == selected_run_name)

# このランの収束情報 (run_meta.json があれば)
run_meta_path = run_dir / "run_meta.json"
if run_meta_path.exists():
    rm = json.loads(run_meta_path.read_text(encoding="utf-8"))
    rm_cols = st.columns(4)
    rm_cols[0].metric("実ターン数", rm.get("n_turns", "-"))
    cons = rm.get("consensus") or {}
    rm_cols[1].metric(
        "合意",
        "✅" if cons.get("reached") else "—",
    )
    rm_cols[2].metric("到達ターン", cons.get("detected_at_turn") or "-")
    rm_cols[3].metric("シグナル", cons.get("signal", "none"))
    if cons.get("rationale"):
        st.caption(f"判定根拠: {cons['rationale']}")

# 各ファイルがあれば読む
left, right = st.columns([1, 1])

with left:
    st.markdown("#### transcript")
    transcript_path = run_dir / "transcript.jsonl"
    if transcript_path.exists():
        for line in transcript_path.read_text(encoding="utf-8").strip().split("\n"):
            row = json.loads(line)
            st.markdown(f"**{row['speaker']}** [t{row['turn_id']}]")
            st.write(row["text"])
            st.markdown("---")
    else:
        st.caption("(transcript.jsonl なし)")

with right:
    st.markdown("#### judge_reports")
    reports_path = run_dir / "judge_reports.json"
    if reports_path.exists():
        reports = json.loads(reports_path.read_text(encoding="utf-8"))
        for rep in reports:
            scores = rep.get("scores", {})
            with st.container():
                st.markdown(f"**{rep['persona_name']}**")
                metric_cols = st.columns(5)
                for (key, label), col in zip(
                    SUBJECTIVE_METRICS, metric_cols, strict=True
                ):
                    val = scores.get(key, "-")
                    col.metric(label, val)
                if scores.get("rationale"):
                    st.caption(f"_{scores['rationale']}_")
                st.markdown("---")
    else:
        st.caption("(judge_reports.json なし、--no-judge で走った可能性)")

    interventions_path = run_dir / "interventions.jsonl"
    if interventions_path.exists():
        # 介入率の集計を上に表示
        all_entries = [
            json.loads(line)
            for line in interventions_path.read_text(encoding="utf-8").strip().split("\n")
            if line.strip()
        ]
        if all_entries:
            n = len(all_entries)
            n_l1 = sum(1 for e in all_entries if e.get("kind") == "l1")
            n_l2 = sum(1 for e in all_entries if e.get("kind") == "l2")
            n_skip = sum(1 for e in all_entries if e.get("kind") == "skip")
            st.markdown("#### 介入種別")
            ic = st.columns(4)
            ic[0].metric("総ターン", n)
            ic[1].metric("L1 (個別)", f"{n_l1} ({n_l1 / n:.0%})")
            ic[2].metric("L2 (俯瞰)", f"{n_l2} ({n_l2 / n:.0%})")
            ic[3].metric("skip", f"{n_skip} ({n_skip / n:.0%})")
        with st.expander("介入ログ (interventions.jsonl)"):
            for entry in all_entries:
                kind = entry.get("kind", "l1")
                badge = {"l1": "🟦 L1", "l2": "🟪 L2", "skip": "⬜ skip"}.get(kind, kind)
                st.markdown(
                    f"**turn {entry['turn_id']}** {badge} → "
                    f"{entry.get('addressed_to') or '全員'} "
                    f"({len(entry.get('items', []))} 件)"
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

    snapshot_path = run_dir / "snapshot.json"
    if snapshot_path.exists():
        with st.expander("AF スナップショット (snapshot.json)"):
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            st.markdown(
                f"ノード {len(data.get('nodes', []))} 件 / "
                f"エッジ {len(data.get('edges', []))} 件"
            )
            st.json(data, expanded=False)
