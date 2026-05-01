"""集計ビュー (副ページ)。

複数ランの主観指標と収束率を条件横断で比較する。個別ランのドリルダウンは
メインページ「議論レビュー」に集約済みなので、ここでは aggregate のみ。
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from das.settings import get_settings

st.set_page_config(page_title="集計", layout="wide")
st.title("集計比較")
st.caption("条件ごとの主観指標と合意形成までの時間を横断で比較する")

SUBJECTIVE_METRICS: list[tuple[str, str]] = [
    ("overall_satisfaction", "満足度"),
    ("information_usefulness", "情報有用性"),
    ("opposition_understanding", "反対理解"),
    ("confidence_change", "自信変化"),
    ("intervention_transparency", "介入透明性"),
]


def _list_eval_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "summary.json").exists()),
        key=lambda p: p.name,
        reverse=True,
    )


settings = get_settings()
default_eval_root = settings.data_dir / "eval"

with st.sidebar:
    st.header("Eval 選択")
    eval_dirs = _list_eval_dirs(default_eval_root)
    if not eval_dirs:
        st.warning(f"`{default_eval_root}` に評価結果がありません。")
        st.stop()
    selected_name = st.selectbox("eval_id", [p.name for p in eval_dirs], index=0)
    eval_dir = next(p for p in eval_dirs if p.name == selected_name)

summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
meta = (
    json.loads((eval_dir / "meta.json").read_text(encoding="utf-8"))
    if (eval_dir / "meta.json").exists()
    else {}
)

st.markdown(f"**トピック**: {meta.get('topic', '-')}")
by_cond = summary.get("by_condition", {})
if not by_cond:
    st.info("by_condition が空です。")
    st.stop()

# --- 主観指標 -----------------------------------------------------------

st.markdown("## 主観指標 (LLM-as-judge)")

rows: list[dict] = []
for cond, payload in by_cond.items():
    for key, label in SUBJECTIVE_METRICS:
        mean, std = payload.get(key, [0.0, 0.0])
        rows.append(
            {
                "condition": cond,
                "metric": label,
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

pivot = df.pivot_table(
    index="metric", columns="condition", values=["mean", "std"], aggfunc="first"
)
st.markdown("### 数値テーブル")
st.dataframe(pivot, use_container_width=True)

# --- 収束統計 -----------------------------------------------------------

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

st.caption("個別ランの詳細は **議論レビュー** ページで条件タブを切り替えて確認できます。")
