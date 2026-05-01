"""E2E スモークテスト。

実 OpenAI API を呼ぶため、明示的な opt-in が必要。

実行方法:
    OPENAI_API_KEY=sk-... OPENAI_INTEGRATION=1 \\
        uv run pytest tests/integration -m integration

OPENAI_INTEGRATION=1 が立っていない場合は自動的に skip。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.runtime import Orchestrator
from das.types import Utterance
from das.viz import dump_snapshot, render_html

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "data" / "docs"
TRANSCRIPT = REPO_ROOT / "tests" / "fixtures" / "cafeteria_transcript.jsonl"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("OPENAI_INTEGRATION") != "1",
        reason="set OPENAI_INTEGRATION=1 (and OPENAI_API_KEY) to enable",
    ),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", "").startswith("sk-"),
        reason="OPENAI_API_KEY (sk-...) not set in environment",
    ),
]


def _load_transcript(path: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utterances.append(Utterance.model_validate(json.loads(line)))
    return utterances


async def test_cafeteria_session_smoke(tmp_path: Path) -> None:
    """カフェテリア例 (5 ターン + 4 文書) を最小構成で 1 セッション通す。

    検証する不変量:
      - パイプラインが例外なく完走する
      - 文書ノードが少なくとも 1 件 AF 化される
      - 発話ノードが少なくとも 1 件 (extraction が空にしすぎていない)
      - snapshot.json と graph.html が書き出せる
    エッジが何件出るかは LinkingAgent の精度に依存するため、ここではアサートしない
    (M2 の評価で別途測定する)。
    """

    assert DOCS_DIR.exists(), f"sample docs missing: {DOCS_DIR}"
    assert TRANSCRIPT.exists(), f"transcript fixture missing: {TRANSCRIPT}"

    transcript = _load_transcript(TRANSCRIPT)
    assert len(transcript) >= 3, "fixture が壊れている可能性"

    llm = OpenAIClient()
    store = NetworkXGraphStore(db_path=tmp_path / "graph.sqlite")
    orch = Orchestrator.assemble(llm=llm, store=store, top_k=5, threshold=0.6)

    # ドキュメント取り込み
    doc_nodes = await orch.ingest_documents(DOCS_DIR)
    assert len(doc_nodes) >= 1, "ドキュメントから 1 ノード以上抽出できるはず"
    assert all(n.source == "document" for n in doc_nodes)

    # 議論セッション
    await orch.run_session(transcript)

    nodes = list(store.nodes())
    edges = list(store.edges())
    utterance_nodes = [n for n in nodes if n.source == "utterance"]
    document_nodes = [n for n in nodes if n.source == "document"]

    assert len(utterance_nodes) >= 1, "発話由来のノードがゼロ"
    assert len(document_nodes) >= 1, "文書由来のノードがゼロ"

    # 出力もファイルに落ちる
    snapshot_path = dump_snapshot(store, tmp_path / "snapshot.json")
    html_path = render_html(store, tmp_path / "graph.html")
    assert snapshot_path.exists()
    assert html_path.exists()

    # 進捗を tee 風に出しておく (pytest -s で見える)
    print(
        f"\n[smoke] nodes={len(nodes)} (utterance={len(utterance_nodes)} "
        f"document={len(document_nodes)}) edges={len(edges)}"
    )
    print(f"[smoke] snapshot={snapshot_path}")
    print(f"[smoke] html={html_path}")
