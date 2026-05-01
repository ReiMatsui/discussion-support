"""LinkingAgent のユニットテスト。

OpenAI 呼び出しは ``embed`` / ``embed_one`` / ``chat_structured`` を AsyncMock で差し替える。
発表資料 §3 のカフェテリア例に近い構成で検証する。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.agents.linking import LinkingAgent, _LinkJudgment, cosine_similarity
from das.graph.schema import Node
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


@pytest.fixture
def store() -> NetworkXGraphStore:
    return NetworkXGraphStore()


@pytest.fixture
def cafeteria() -> dict[str, Node]:
    a1 = Node(
        text="プラ容器を廃止すべき",
        node_type="claim",
        source="utterance",
        author="A",
    )
    a2 = Node(
        text="紙容器はコストが 3 倍で値上げにつながる",
        node_type="claim",
        source="utterance",
        author="B",
    )
    a3 = Node(
        text="X 大学では紙容器導入 2 年目にコスト増が解消した",
        node_type="premise",
        source="document",
        author="x_univ_case",
    )
    a4 = Node(
        text="今日は雨が降っている",
        node_type="premise",
        source="utterance",
        author="C",
    )
    return {"a1": a1, "a2": a2, "a3": a3, "a4": a4}


# --- cosine_similarity --------------------------------------------------


def test_cosine_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_identical() -> None:
    assert cosine_similarity([1.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)


def test_cosine_zero_vector() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_dimension_mismatch() -> None:
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0


# --- 候補選定 -----------------------------------------------------------


async def test_link_node_returns_no_edges_when_store_empty(store: NetworkXGraphStore) -> None:
    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock()  # 呼ばれないはず  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm)
    target = Node(text="孤独な発話", node_type="claim", source="utterance", author="A")
    store.add_node(target)

    edges = await agent.link_node(target, store)
    assert edges == []
    llm.chat_structured.assert_not_awaited()


async def test_top_k_filters_by_embedding_similarity(
    store: NetworkXGraphStore, cafeteria: dict[str, Node]
) -> None:
    """top_k=1 のとき、類似度最大の 1 件だけが LLM 判定に渡されることを確認。"""

    nodes = cafeteria
    for n in (nodes["a2"], nodes["a3"], nodes["a4"]):
        store.add_node(n)
    target = nodes["a1"]
    store.add_node(target)

    llm = _fake_llm()
    # target と最も近いベクトルを a2 に割り当て、a3 と a4 は遠ざける
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0, 0.0])  # type: ignore[method-assign]
    embeddings_by_text = {
        nodes["a2"].text: [0.99, 0.01, 0.0],
        nodes["a3"].text: [0.10, 0.95, 0.0],
        nodes["a4"].text: [0.00, 0.00, 1.0],
    }

    async def fake_embed(texts: list[str], **kwargs: object) -> list[list[float]]:
        return [embeddings_by_text[t] for t in texts]

    llm.embed = AsyncMock(side_effect=fake_embed)  # type: ignore[method-assign]
    judged: list[tuple[str, str]] = []

    async def fake_judge(messages: list[dict], **kwargs: object) -> _LinkJudgment:
        # メッセージ内容に含まれるテキストから候補を抽出
        user_content = messages[1]["content"]
        for text in (nodes["a2"].text, nodes["a3"].text, nodes["a4"].text):
            if text in user_content:
                judged.append((target.text, text))
        return _LinkJudgment(relation="none", confidence=0.9, rationale="-")

    llm.chat_structured = AsyncMock(side_effect=fake_judge)  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=1, threshold=0.6)
    await agent.link_node(target, store)

    # 1 件だけ判定が呼ばれ、それは a2 (最も近い)
    assert len(judged) == 1
    assert judged[0][1] == nodes["a2"].text


# --- 関係推定とエッジ書き込み ------------------------------------------


def _judgment(rel: str, conf: float = 0.9) -> _LinkJudgment:
    return _LinkJudgment(relation=rel, confidence=conf, rationale="rationale")  # type: ignore[arg-type]


async def test_a_attacks_b_creates_edge_target_to_candidate(
    store: NetworkXGraphStore, cafeteria: dict[str, Node]
) -> None:
    nodes = cafeteria
    target = nodes["a2"]  # B の発話
    cand = nodes["a1"]  # A の発話
    store.add_node(cand)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(return_value=_judgment("a_attacks_b", 0.9))  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5, threshold=0.6)
    edges = await agent.link_node(target, store)

    assert len(edges) == 1
    edge = edges[0]
    assert edge.src_id == target.id
    assert edge.dst_id == cand.id
    assert edge.relation == "attack"
    assert edge.confidence == pytest.approx(0.9)
    assert edge.created_by == "linking"


async def test_b_attacks_a_creates_edge_candidate_to_target(
    store: NetworkXGraphStore, cafeteria: dict[str, Node]
) -> None:
    """文書 (a3) がコスト懸念 (target=a2) を反論する想定。"""

    nodes = cafeteria
    target = nodes["a2"]
    cand = nodes["a3"]
    store.add_node(cand)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(return_value=_judgment("b_attacks_a", 0.85))  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5, threshold=0.6)
    edges = await agent.link_node(target, store)

    assert len(edges) == 1
    edge = edges[0]
    assert edge.src_id == cand.id
    assert edge.dst_id == target.id
    assert edge.relation == "attack"


async def test_b_supports_a_creates_support_edge(
    store: NetworkXGraphStore, cafeteria: dict[str, Node]
) -> None:
    """Web ノード (premise) が claim を支持。"""

    web = Node(
        text="バイオプラは +40% コストで生分解可能",
        node_type="premise",
        source="web",
        author="example.com",
    )
    target = cafeteria["a1"]
    store.add_node(web)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(return_value=_judgment("b_supports_a", 0.7))  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5, threshold=0.6)
    edges = await agent.link_node(target, store)

    assert len(edges) == 1
    assert edges[0].src_id == web.id
    assert edges[0].dst_id == target.id
    assert edges[0].relation == "support"


async def test_none_relation_creates_no_edge(
    store: NetworkXGraphStore, cafeteria: dict[str, Node]
) -> None:
    target = cafeteria["a1"]
    cand = cafeteria["a4"]  # 関係ない発話
    store.add_node(cand)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(return_value=_judgment("none", 0.95))  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5, threshold=0.6)
    edges = await agent.link_node(target, store)
    assert edges == []


async def test_below_threshold_creates_no_edge(
    store: NetworkXGraphStore, cafeteria: dict[str, Node]
) -> None:
    target = cafeteria["a2"]
    cand = cafeteria["a1"]
    store.add_node(cand)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(return_value=_judgment("a_attacks_b", 0.3))  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5, threshold=0.6)
    edges = await agent.link_node(target, store)
    assert edges == []


# --- 埋め込みキャッシュ -------------------------------------------------


async def test_embedding_cache_avoids_recomputation(
    store: NetworkXGraphStore, cafeteria: dict[str, Node]
) -> None:
    nodes = cafeteria
    for n in (nodes["a2"], nodes["a3"]):
        store.add_node(n)
    target = nodes["a1"]
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[0.9, 0.1], [0.1, 0.9]])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(return_value=_judgment("none", 0.9))  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5)
    await agent.link_node(target, store)

    # 2 回目: ストアに新ノードを追加して再 link → embed は新規分のみ呼ばれる
    new_node = Node(
        text="ゴミの量が問題",
        node_type="premise",
        source="utterance",
        author="D",
    )
    store.add_node(new_node)
    llm.embed.reset_mock()
    llm.embed.return_value = [[0.5, 0.5]]
    await agent.link_node(target, store)

    # embed は new_node の 1 件だけが渡されるはず
    assert llm.embed.await_count == 1
    called_texts = llm.embed.await_args.args[0]
    assert called_texts == [new_node.text]
