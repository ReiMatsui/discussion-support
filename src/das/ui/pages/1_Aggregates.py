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

# --- 客観構造指標 (Stage 1) -------------------------------------------

structural_rows: list[dict] = []
for cond, payload in by_cond.items():
    s = payload.get("structural")
    if not s:
        continue
    structural_rows.append(
        {
            "condition": cond,
            "participation_gini": s.get("participation_gini_mean", 0.0),
            "avg_premises_per_claim": s.get("avg_premises_per_claim_mean", 0.0),
            "pct_unsupported_claims": s.get("pct_unsupported_claims_mean", 0.0),
            "response_rate": s.get("response_rate_mean", 0.0),
            "pct_attacks_answered": s.get("pct_attacks_answered_mean", 0.0),
            "avg_argument_chain_length": s.get("avg_argument_chain_length_mean", 0.0),
            "n_total_edges": s.get("n_total_edges_mean", 0.0),
        }
    )

if structural_rows:
    st.divider()
    st.markdown("## 客観構造指標 (LLM-free, AF 由来)")
    st.caption(
        "DQI / Social Laboratory の流れに沿った客観指標。"
        "AF と transcript から決定的に計算できるため、対面・音声へも同じ意味で展開できる。"
    )
    sdf = pd.DataFrame(structural_rows)
    display = sdf.assign(
        participation_gini=lambda d: d["participation_gini"].round(3),
        avg_premises_per_claim=lambda d: d["avg_premises_per_claim"].round(2),
        pct_unsupported_claims=lambda d: (d["pct_unsupported_claims"] * 100).round(1).astype(str) + "%",
        response_rate=lambda d: (d["response_rate"] * 100).round(1).astype(str) + "%",
        pct_attacks_answered=lambda d: (d["pct_attacks_answered"] * 100).round(1).astype(str) + "%",
        avg_argument_chain_length=lambda d: d["avg_argument_chain_length"].round(2),
        n_total_edges=lambda d: d["n_total_edges"].round(1),
    ).rename(
        columns={
            "condition": "条件",
            "participation_gini": "参加偏在 (gini)",
            "avg_premises_per_claim": "claim あたり premise",
            "pct_unsupported_claims": "未根拠 claim 率",
            "response_rate": "応答率",
            "pct_attacks_answered": "反論への再反論率",
            "avg_argument_chain_length": "論証連鎖の平均深さ",
            "n_total_edges": "AF エッジ数 (平均)",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "**参加偏在 (gini)**: 0 = 完全平等, 1 = 一極集中。0.2-0.4 程度が望ましい。"
        " **応答率**: 各発話が以前の発話/文書に support/attack エッジを張った割合。"
        " **反論への再反論率**: 反論を受けた話者が後で反論し返した割合 (DQI: respect)。"
    )

st.caption("個別ランの詳細は **議論レビュー** ページで条件タブを切り替えて確認できます。")
