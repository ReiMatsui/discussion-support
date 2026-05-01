"""ライブセッション ページ。

ペルソナ + トピック + 情報提供条件 を選んで、議論をリアルタイムに走らせる。
発話が出来るたびに画面に追記される。

注意: 実 OpenAI API を呼ぶのでコストが発生する。
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
    PersonaSpec,
    SessionConfig,
    SessionRunner,
    cafeteria_personas,
    policy_ai_lecture_personas,
)
from das.llm import OpenAIClient
from das.settings import get_settings
from das.types import Utterance

st.set_page_config(page_title="ライブセッション", layout="wide")
st.title("ライブセッション")
st.caption("ペルソナ同士の議論をリアルタイムに走らせる")

# --- プリセット ---------------------------------------------------------

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

CONDITION_LABELS: dict[str, str] = {
    "none": "なし (情報提供なし)",
    "flat_rag": "フラット RAG",
    "full_proposal": "提案手法 (統合議論グラフ)",
}

# --- サイドバー --------------------------------------------------------

with st.sidebar:
    st.header("セッション設定")
    preset_key: str = st.selectbox(
        "トピックプリセット",
        list(PRESETS.keys()),
        format_func=lambda k: PRESETS[k][0],
    )
    label, persona_factory, default_topic = PRESETS[preset_key]
    topic = st.text_area("議論トピック (編集可)", value=default_topic, height=80)
    max_turns = st.slider("最大ターン数", 3, 20, 8)
    temperature = st.slider("temperature", 0.0, 1.5, 0.7, 0.1)

    st.divider()
    st.subheader("情報提供条件")
    condition_key: str = st.selectbox(
        "条件",
        list(CONDITION_LABELS.keys()),
        format_func=lambda k: CONDITION_LABELS[k],
    )
    docs_dir_str = st.text_input(
        "ドキュメントディレクトリ",
        value=str(get_settings().docs_dir),
        help="flat_rag / full_proposal で参照する事前文書",
    )

    st.caption(
        "注意: 実 OpenAI API を呼ぶためコストが発生します。"
        "full_proposal は文書取り込み + 連結判定で追加 LLM 呼び出しが多いです。"
    )

personas: list[PersonaSpec] = persona_factory()

st.markdown("### 参加するペルソナ")
cols = st.columns(len(personas))
for col, p in zip(cols, personas, strict=True):
    with col:
        st.markdown(f"**{p.name}** _( {p.stance} )_")
        st.caption(f"重視点: {p.focus}")
        if p.extra:
            st.caption(p.extra)


# --- 実行 -------------------------------------------------------------


def _format_utterance(u: Utterance) -> str:
    return f"**{u.speaker}** [turn {u.turn_id}]\n\n{u.text}"


def _make_condition(key: str, llm: OpenAIClient) -> Condition:
    if key == "none":
        return ConditionNone()
    if key == "flat_rag":
        return ConditionFlatRAG(llm=llm)
    if key == "full_proposal":
        return ConditionFullProposal(llm=llm)
    raise ValueError(f"unknown condition: {key}")


async def _run(
    runner: SessionRunner,
    condition: Condition,
    transcript_box,
    progress_text,
    counter_metric,
    info_box,
) -> list[Utterance]:
    transcript: list[Utterance] = []

    async def info_provider_with_display(history, persona):
        info = await condition.info_provider(history, persona)
        if info:
            with info_box:
                st.markdown(
                    f"**[{persona.name} 向けの参考情報]**\n\n```\n{info}\n```"
                )
        return info

    async for utterance in runner.run_streaming(
        info_provider=info_provider_with_display
    ):
        transcript.append(utterance)
        with transcript_box:
            st.markdown(_format_utterance(utterance))
            st.markdown("---")
        progress_text.markdown(
            f"進行中... {len(transcript)} / {runner.config.max_turns}"
        )
        counter_metric.metric("発話数", len(transcript))
    return transcript


if st.button("セッション開始", type="primary", use_container_width=True):
    if not topic.strip():
        st.error("トピックを入力してください。")
        st.stop()

    config = SessionConfig(
        topic=topic.strip(),
        max_turns=max_turns,
        temperature=temperature,
    )

    llm = OpenAIClient()
    try:
        runner = SessionRunner(personas, config, llm=llm)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    condition = _make_condition(condition_key, llm)
    docs_dir: Path | None = Path(docs_dir_str) if docs_dir_str else None

    setup_text = st.empty()
    setup_text.markdown(
        f"条件 '{condition.name}' のセットアップ中... (docs={docs_dir})"
    )
    try:
        asyncio.run(condition.setup(docs_dir=docs_dir))
    except Exception as exc:  # pragma: no cover - UI 用
        st.error(f"セットアップで失敗: {exc}")
        st.exception(exc)
        st.stop()
    setup_text.markdown(f"✓ 条件 '{condition.name}' のセットアップ完了")

    counter_col, _ = st.columns([1, 5])
    with counter_col:
        counter_metric = st.empty()
        counter_metric.metric("発話数", 0)

    progress_text = st.empty()
    progress_text.markdown(f"開始... 最大 {max_turns} ターン")

    main_col, side_col = st.columns([2, 1])
    with main_col:
        st.markdown("### transcript")
        transcript_box = st.container()
    with side_col:
        st.markdown("### 提示された参考情報")
        info_box = st.container()

    try:
        transcript = asyncio.run(
            _run(
                runner,
                condition,
                transcript_box,
                progress_text,
                counter_metric,
                info_box,
            )
        )
    except Exception as exc:  # pragma: no cover - UI 用
        st.error(f"セッション中にエラーが発生しました: {exc}")
        st.exception(exc)
    else:
        progress_text.markdown(f"✓ 完了 ({len(transcript)} ターン)")

        with st.expander("transcript JSONL (コピペ用)"):
            payload = "\n".join(
                json.dumps(
                    {
                        "turn_id": u.turn_id,
                        "speaker": u.speaker,
                        "text": u.text,
                    },
                    ensure_ascii=False,
                )
                for u in transcript
            )
            st.code(payload, language="json")

        # full_proposal の場合は構築された AF も snapshot として表示できるようにしておく
        if isinstance(condition, ConditionFullProposal) and condition.orchestrator:
            store = condition.orchestrator.store
            n_nodes = len(list(store.nodes()))
            n_edges = len(list(store.edges()))
            st.success(
                f"統合議論グラフ: ノード {n_nodes} 件 / エッジ {n_edges} 件"
            )
