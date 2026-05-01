"""Streamlit 製の議論グラフ ビューア。

起動:
    uv run das ui
        or
    uv run streamlit run src/das/ui/streamlit_app.py

機能:
    - ``data/runs/<run_id>/snapshot.json`` を一覧から選択
    - ノード/エッジの基本統計
    - pyvis でグラフを HTML 埋め込み
    - ノード/エッジ一覧をフィルタ付きで表示

公開向けの 4 層提示 (L1/L2/L3) ではなく、研究プロトタイプ用のデバッグ/ブラウズ用 UI。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from das.graph.schema import Edge, Node
from das.graph.store import GraphStore
from das.settings import get_settings
from das.viz import load_snapshot, render_html


def _list_run_directories(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )


def _stats(nodes: list[Node], edges: list[Edge]) -> dict[str, int]:
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "utterance": sum(1 for n in nodes if n.source == "utterance"),
        "document": sum(1 for n in nodes if n.source == "document"),
        "web": sum(1 for n in nodes if n.source == "web"),
        "support": sum(1 for e in edges if e.relation == "support"),
        "attack": sum(1 for e in edges if e.relation == "attack"),
    }


def _node_dataframe(nodes: list[Node]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": str(n.id)[:8],
                "type": n.node_type,
                "source": n.source,
                "author": n.author or "",
                "text": n.text,
            }
            for n in nodes
        ]
    )


def _edge_dataframe(edges: list[Edge], nodes: list[Node]) -> pd.DataFrame:
    by_id = {n.id: n for n in nodes}
    rows: list[dict[str, object]] = []
    for e in edges:
        src_text = by_id[e.src_id].text if e.src_id in by_id else f"({e.src_id})"
        dst_text = by_id[e.dst_id].text if e.dst_id in by_id else f"({e.dst_id})"
        rows.append(
            {
                "src": src_text,
                "→": "→",
                "dst": dst_text,
                "relation": e.relation,
                "confidence": round(e.confidence, 3),
                "rationale": e.rationale,
            }
        )
    return pd.DataFrame(rows)


def _render_graph_html(store: GraphStore) -> str:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        path = Path(f.name)
    render_html(store, path)
    html = path.read_text(encoding="utf-8")
    path.unlink(missing_ok=True)
    return html


def main() -> None:
    st.set_page_config(
        page_title="議論グラフ ビューア",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("議論グラフ ビューア")
    st.caption("Discussion Argumentation Support — 統合議論グラフのブラウズ用ビューア")

    settings = get_settings()
    runs_dir = settings.runs_dir

    with st.sidebar:
        st.header("セッション選択")
        run_dirs = _list_run_directories(runs_dir)
        if not run_dirs:
            st.warning(
                f"`{runs_dir}` にセッションがありません。\n\n"
                "`uv run das run-session <transcript.jsonl>` を実行するとここに出てきます。"
            )
            st.stop()

        names = [p.name for p in run_dirs]
        selected_name = st.selectbox("Run ID", names, index=0)
        selected_dir = next(p for p in run_dirs if p.name == selected_name)
        snapshot_path = selected_dir / "snapshot.json"
        if not snapshot_path.exists():
            st.error(f"`snapshot.json` が見つかりません: {snapshot_path}")
            st.stop()
        st.caption(f"snapshot: `{snapshot_path}`")

    store = load_snapshot(snapshot_path)
    nodes = list(store.nodes())
    edges = list(store.edges())
    stats = _stats(nodes, edges)

    # --- 統計 -----------------------------------------------------------
    cols = st.columns(7)
    cols[0].metric("ノード数", stats["nodes"])
    cols[1].metric("エッジ数", stats["edges"])
    cols[2].metric("発話", stats["utterance"])
    cols[3].metric("文書", stats["document"])
    cols[4].metric("Web", stats["web"])
    cols[5].metric("Support", stats["support"])
    cols[6].metric("Attack", stats["attack"])

    # --- グラフ ---------------------------------------------------------
    st.subheader("統合議論グラフ")
    if not nodes:
        st.info("ノードがありません。")
    else:
        html = _render_graph_html(store)
        components.html(html, height=820, scrolling=False)

    # --- 一覧 ------------------------------------------------------------
    tab_nodes, tab_edges = st.tabs(["ノード一覧", "エッジ一覧"])

    with tab_nodes:
        if not nodes:
            st.write("なし")
        else:
            df_nodes = _node_dataframe(nodes)
            available_sources = ["all", *sorted({n.source for n in nodes})]
            selected_source = st.selectbox(
                "source フィルタ", available_sources, key="node_source_filter"
            )
            if selected_source != "all":
                df_nodes = df_nodes[df_nodes["source"] == selected_source]

            search = st.text_input("テキスト検索 (部分一致)", "")
            if search:
                df_nodes = df_nodes[df_nodes["text"].str.contains(search, case=False, na=False)]

            st.dataframe(df_nodes, hide_index=True, use_container_width=True)

    with tab_edges:
        if not edges:
            st.write("なし")
        else:
            df_edges = _edge_dataframe(edges, nodes)
            relations = ["all", "support", "attack"]
            selected_relation = st.selectbox(
                "relation フィルタ", relations, key="edge_relation_filter"
            )
            if selected_relation != "all":
                df_edges = df_edges[df_edges["relation"] == selected_relation]

            min_confidence = st.slider("最小 confidence", 0.0, 1.0, 0.0, 0.05)
            if min_confidence > 0:
                df_edges = df_edges[df_edges["confidence"] >= min_confidence]

            st.dataframe(df_edges, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
