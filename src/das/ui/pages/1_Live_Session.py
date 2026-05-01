"""ライブセッション ページ。

ペルソナ + トピック + 情報提供条件 を選んで、議論をリアルタイムに走らせる。
発話が出来るたびに transcript に追記され、FullProposal 条件では統合議論グラフが
育つ様子も右パネルでライブ更新される。

注意: 実 OpenAI API を呼ぶのでコストが発生する。
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from das.eval import (
    Condition,
    ConditionFlatRAG,
    ConditionFullProposal,
    ConditionNone,
    FlatRAGItem,
    InfoItem,
    PersonaSpec,
    SessionConfig,
    SessionRunner,
    cafeteria_personas,
    policy_ai_lecture_personas,
)
from das.graph.store import GraphStore
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

    st.divider()
    save_snapshot = st.checkbox(
        "終了後にスナップショットを保存",
        value=True,
        help="full_proposal 条件のとき data/runs/<タイムスタンプ>/ に snapshot.json と graph.html を保存",
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


_SOURCE_BADGE: dict[str, str] = {
    "utterance": "🗣️ 発話",
    "document": "📄 文書",
    "web": "🌐 Web",
}


def _truncate(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _render_full_proposal_info(persona_name: str, items: list[InfoItem]) -> None:
    st.markdown(f"#### {persona_name} さんへの参考情報")
    if not items:
        st.caption("(関連する支持・反論なし)")
        return
    for item in items:
        badge = _SOURCE_BADGE.get(item.source_kind, item.source_kind)
        author = item.source_author or "?"
        target_short = _truncate(item.target_text)
        rationale_part = f"\n\n_「{item.rationale}」_" if item.rationale else ""
        body = (
            f"**{badge} `{author}`** | conf {item.confidence:.0%}\n\n"
            f"> {item.source_text}\n\n"
            f"あなたの「{target_short}」への "
            f"{'支持' if item.relation == 'support' else '反論'}"
            f"{rationale_part}"
        )
        if item.relation == "support":
            st.success("🟢 支持\n\n" + body)
        else:
            st.error("🔴 反論\n\n" + body)


def _render_flat_rag_info(persona_name: str, items: list[FlatRAGItem]) -> None:
    st.markdown(f"#### {persona_name} さんへの参考情報 (FlatRAG)")
    if not items:
        st.caption("(類似チャンクなし)")
        return
    for item in items:
        st.info(f"📄 **{item.doc_id}** | similarity {item.score:.2f}\n\n> {item.text}")


def _make_condition(key: str, llm: OpenAIClient) -> Condition:
    if key == "none":
        return ConditionNone()
    if key == "flat_rag":
        return ConditionFlatRAG(llm=llm)
    if key == "full_proposal":
        return ConditionFullProposal(llm=llm)
    raise ValueError(f"unknown condition: {key}")


def _render_store_html(store: GraphStore) -> str:
    """``GraphStore`` を pyvis HTML 文字列にする (一時ファイル経由)。"""

    from das.viz import render_html

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = Path(f.name)
    try:
        render_html(store, path, height="500px")
        return path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)


def _update_af_panel(condition: Condition, af_metric, af_box) -> None:
    """FullProposal なら最新の AF をライブで描き直す。"""

    if not isinstance(condition, ConditionFullProposal):
        return
    orch = condition.orchestrator
    if orch is None:
        return
    store = orch.store
    n_nodes = len(list(store.nodes()))
    n_edges = len(list(store.edges()))
    af_metric.metric("AF (ノード / エッジ)", f"{n_nodes} / {n_edges}")
    if n_nodes == 0:
        return
    try:
        html = _render_store_html(store)
    except Exception as exc:  # pragma: no cover - UI 用
        with af_box.container():
            st.warning(f"AF 描画に失敗: {exc}")
        return
    with af_box.container():
        components.html(html, height=520, scrolling=False)


async def _run(
    runner: SessionRunner,
    condition: Condition,
    transcript_box,
    progress_text,
    counter_metric,
    info_placeholder,
    af_metric,
    af_box,
) -> list[Utterance]:
    transcript: list[Utterance] = []

    async def info_provider_with_display(history, persona):
        info = await condition.info_provider(history, persona)
        # 最新ターンの参考情報だけを表示 (前のターンの分は上書き消去)
        with info_placeholder.container():
            if isinstance(condition, ConditionFullProposal):
                _render_full_proposal_info(persona.name, condition.last_items)
            elif isinstance(condition, ConditionFlatRAG):
                _render_flat_rag_info(persona.name, condition.last_items)
            else:
                st.caption("(情報提供なし)")
        _update_af_panel(condition, af_metric, af_box)
        return info

    async for utterance in runner.run_streaming(info_provider=info_provider_with_display):
        transcript.append(utterance)
        with transcript_box:
            st.markdown(_format_utterance(utterance))
            st.markdown("---")
        progress_text.markdown(f"進行中... {len(transcript)} / {runner.config.max_turns}")
        counter_metric.metric("発話数", len(transcript))

    # ループ終了後にも一度更新 (最終ターンの AF が反映済みであることを保証)
    _update_af_panel(condition, af_metric, af_box)
    return transcript


def _save_snapshot(condition: Condition) -> Path | None:
    """FullProposal の最終 AF を ``data/runs/<ts>/`` に保存する。"""

    if not isinstance(condition, ConditionFullProposal):
        return None
    orch = condition.orchestrator
    if orch is None:
        return None

    from das.viz import dump_snapshot, render_html

    settings = get_settings()
    run_id = datetime.now(timezone.utc).strftime("live-%Y%m%dT%H%M%SZ")
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_snapshot(orch.store, run_dir / "snapshot.json")
    render_html(orch.store, run_dir / "graph.html")
    return run_dir


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
    setup_text.markdown(f"条件 '{condition.name}' のセットアップ中... (docs={docs_dir})")
    try:
        asyncio.run(condition.setup(docs_dir=docs_dir))
    except Exception as exc:  # pragma: no cover - UI 用
        st.error(f"セットアップで失敗: {exc}")
        st.exception(exc)
        st.stop()
    setup_text.markdown(f"✓ 条件 '{condition.name}' のセットアップ完了")

    metrics_cols = st.columns([1, 1, 4])
    with metrics_cols[0]:
        counter_metric = st.empty()
        counter_metric.metric("発話数", 0)
    with metrics_cols[1]:
        af_metric = st.empty()
        af_metric.metric("AF (ノード / エッジ)", "- / -")

    progress_text = st.empty()
    progress_text.markdown(f"開始... 最大 {max_turns} ターン")

    is_full_proposal = isinstance(condition, ConditionFullProposal)

    if is_full_proposal:
        main_col, side_col = st.columns([1, 1])
    else:
        main_col, side_col = st.columns([2, 1])

    with main_col:
        st.markdown("### transcript")
        transcript_box = st.container()
    with side_col:
        if is_full_proposal:
            st.markdown("### 統合議論グラフ (リアルタイム)")
            st.caption(
                "**ノード**: 🔵 発話 / 🟩 文書 / 🟧 Web   "
                "**エッジ**: 🟢 支持 / 🔴 反論 (太さ = 信頼度)"
            )
            af_box = st.empty()
            with af_box.container():
                st.info("発話が始まると AF が育っていきます。")
            st.divider()
        else:
            af_box = st.empty()  # 描画されない placeholder
        st.markdown("### 現在の参考情報")
        info_placeholder = st.empty()
        with info_placeholder.container():
            st.caption("(まだ情報なし)")

    try:
        transcript = asyncio.run(
            _run(
                runner,
                condition,
                transcript_box,
                progress_text,
                counter_metric,
                info_placeholder,
                af_metric,
                af_box,
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

        if save_snapshot:
            saved_dir = _save_snapshot(condition)
            if saved_dir is not None:
                st.success(
                    f"AF スナップショットを保存しました: `{saved_dir}` (snapshot.json + graph.html)"
                )
