"""3 条件比較ページ。

同じトピック・ペルソナ・ターン数で None / FlatRAG / FullProposal を
順に走らせ、transcript と基本指標を side-by-side で並べる。

注意: 1 ボタンで 3 セッション分の OpenAI 呼び出しが発生する。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import streamlit as st

from das.eval import (
    Condition,
    ConditionFlatRAG,
    ConditionFullProposal,
    ConditionNone,
    GraphMetrics,
    PersonaSpec,
    SessionConfig,
    SessionRunner,
    TranscriptMetrics,
    cafeteria_personas,
    graph_metrics,
    policy_ai_lecture_personas,
    transcript_metrics,
)
from das.llm import OpenAIClient
from das.settings import get_settings
from das.types import Utterance

st.set_page_config(page_title="3 条件比較", layout="wide")
st.title("3 条件比較ラン")
st.caption(
    "同じトピック・ペルソナで 情報なし / フラット RAG / 提案手法 を順に走らせ、結果を並べて比較する"
)

PRESETS: dict[str, tuple[str, callable, str]] = {
    "cafeteria": (
        "プラ容器の廃止 (大学カフェテリア)",
        cafeteria_personas,
        "大学のカフェテリアでプラスチック容器を廃止すべきか",
    ),
    "policy_ai": (
        "生成 AI の大学講義利用",
        policy_ai_lecture_personas,
        "生成 AI を大学の講義・レポート作成で許容すべきか",
    ),
}

CONDITION_ORDER: list[tuple[str, str]] = [
    ("none", "なし"),
    ("flat_rag", "フラット RAG"),
    ("full_proposal", "提案手法"),
]


with st.sidebar:
    st.header("比較ラン設定")
    preset_key = st.selectbox(
        "トピックプリセット",
        list(PRESETS.keys()),
        format_func=lambda k: PRESETS[k][0],
    )
    label, persona_factory, default_topic = PRESETS[preset_key]
    topic = st.text_area("議論トピック", value=default_topic, height=80)
    # max_turns は合意検出による早期終了の安全上限としてのみ使う。UI では持たず固定値。
    max_turns = 12
    temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.1)
    docs_dir_str = st.text_input("ドキュメントディレクトリ", value=str(get_settings().docs_dir))

    st.caption(
        "コスト目安: 3 条件 x ターン数 x persona 数 の発話 + 提案手法のみ文書取り込み + 連結判定"
    )


personas: list[PersonaSpec] = persona_factory()
st.markdown("### 参加するペルソナ")
cols = st.columns(len(personas))
for col, p in zip(cols, personas, strict=True):
    with col:
        st.markdown(f"**{p.name}** _( {p.stance} )_")
        st.caption(f"重視点: {p.focus}")


def _make_condition(key: str, llm: OpenAIClient) -> Condition:
    if key == "none":
        return ConditionNone()
    if key == "flat_rag":
        return ConditionFlatRAG(llm=llm)
    if key == "full_proposal":
        return ConditionFullProposal(llm=llm)
    raise ValueError(f"unknown condition: {key}")


async def _run_one(
    condition_key: str,
    personas_: list[PersonaSpec],
    config: SessionConfig,
    docs_dir: Path | None,
    llm: OpenAIClient,
    progress_text,
    transcript_box,
) -> tuple[list[Utterance], Condition]:
    condition = _make_condition(condition_key, llm)
    progress_text.markdown(f"`{condition.name}` セットアップ中...")
    await condition.setup(docs_dir=docs_dir)
    progress_text.markdown(f"`{condition.name}` 開始 (最大 {config.max_turns} ターン)")
    transcript: list[Utterance] = []
    runner = SessionRunner(personas_, config, llm=llm)
    async for utterance in runner.run_streaming(info_provider=condition.info_provider):
        transcript.append(utterance)
        with transcript_box:
            st.markdown(f"**{utterance.speaker}** [t{utterance.turn_id}]")
            st.markdown(f"> {utterance.text}")
        progress_text.markdown(
            f"`{condition.name}` 進行中... {len(transcript)} / {config.max_turns}"
        )
    progress_text.markdown(f"✓ `{condition.name}` 完了")
    return transcript, condition


def _format_transcript_metrics(m: TranscriptMetrics) -> str:
    lines = [
        f"- ターン数: **{m.n_turns}**",
        f"- 総文字数: **{m.n_chars_total:,}**",
        f"- 平均文字/ターン: **{m.avg_chars_per_turn:.1f}**",
        f"- 発言バランス Gini: **{m.gini_speaker_balance:.2f}** (0=均等)",
    ]
    if m.speaker_turn_counts:
        balance_str = ", ".join(f"{name}: {n}" for name, n in m.speaker_turn_counts.items())
        lines.append(f"- 話者別: {balance_str}")
    return "\n".join(lines)


def _format_graph_metrics(g: GraphMetrics) -> str:
    ratio_str = f"{g.support_attack_ratio:.0%}" if g.support_attack_ratio is not None else "-"
    return "\n".join(
        [
            f"- 統合 AF ノード: **{g.n_nodes}** "
            f"(発話 {g.n_utterance_nodes} / 文書 {g.n_document_nodes} / Web {g.n_web_nodes})",
            f"- エッジ: **{g.n_edges}** (支持 {g.n_support_edges} / 反論 {g.n_attack_edges})",
            f"- 支持/(支持+反論): **{ratio_str}**",
        ]
    )


if st.button("3 条件を順番に走らせる", type="primary", use_container_width=True):
    if not topic.strip():
        st.error("トピックを入力してください。")
        st.stop()

    config = SessionConfig(
        topic=topic.strip(),
        max_turns=max_turns,
        temperature=temperature,
    )
    docs_dir = Path(docs_dir_str) if docs_dir_str else None
    llm = OpenAIClient()

    progress_text = st.empty()
    columns = st.columns(len(CONDITION_ORDER))
    transcripts: dict[str, list[Utterance]] = {}
    conditions: dict[str, Condition] = {}

    # 各列のヘッダ + transcript ボックス
    transcript_boxes: dict[str, object] = {}
    for col, (key, label) in zip(columns, CONDITION_ORDER, strict=True):
        with col:
            st.markdown(f"#### {label}  `({key})`")
            transcript_boxes[key] = st.container()

    async def _run_all() -> None:
        for key, _label in CONDITION_ORDER:
            t, cond = await _run_one(
                key,
                personas,
                config,
                docs_dir,
                llm,
                progress_text,
                transcript_boxes[key],
            )
            transcripts[key] = t
            conditions[key] = cond

    try:
        asyncio.run(_run_all())
    except Exception as exc:  # pragma: no cover - UI 用
        st.error(f"比較ラン中にエラー: {exc}")
        st.exception(exc)
        st.stop()

    progress_text.markdown("✓ 全条件完了")

    st.divider()
    st.markdown("## 指標サマリ")
    metric_cols = st.columns(len(CONDITION_ORDER))
    for col, (key, label) in zip(metric_cols, CONDITION_ORDER, strict=True):
        with col:
            st.markdown(f"#### {label}")
            t_metrics = transcript_metrics(transcripts[key])
            st.markdown(_format_transcript_metrics(t_metrics))
            cond = conditions[key]
            if isinstance(cond, ConditionFullProposal) and cond.orchestrator:
                g_metrics = graph_metrics(cond.orchestrator.store)
                st.markdown("---")
                st.markdown(_format_graph_metrics(g_metrics))

    st.divider()
    with st.expander("transcript JSONL (全条件)"):
        for key, label in CONDITION_ORDER:
            st.markdown(f"##### {label}")
            payload = "\n".join(
                json.dumps(
                    {
                        "turn_id": u.turn_id,
                        "speaker": u.speaker,
                        "text": u.text,
                    },
                    ensure_ascii=False,
                )
                for u in transcripts[key]
            )
            st.code(payload, language="json")
