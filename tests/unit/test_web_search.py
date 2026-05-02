"""WebSearchAgent のユニットテスト。

外部 (Tavily) は呼ばず、TavilyClient.search を MagicMock で差し替える。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from das.agents.web_search import WebSearchAgent
from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore


def _fake_tavily(results: list[dict]) -> MagicMock:
    client = MagicMock()
    client.search = MagicMock(return_value={"results": results})
    return client


@pytest.fixture(autouse=True)
def _set_tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
    from das.settings import reset_settings

    reset_settings()


# --- search() -----------------------------------------------------------


async def test_search_returns_web_nodes_from_tavily_results() -> None:
    agent = WebSearchAgent(api_key="fake")
    agent._client = _fake_tavily(
        [
            {
                "url": "https://example.com/page1",
                "title": "Plastic policy review",
                "content": "Single-use plastic ban analysis content...",
            },
            {
                "url": "https://x.example.org/y",
                "title": "Cost study",
                "content": "Cost increase observed in case studies",
            },
        ]
    )
    nodes = await agent.search("プラスチック容器の廃止")
    assert len(nodes) == 2
    assert all(n.source == "web" for n in nodes)
    assert all(n.node_type == "premise" for n in nodes)
    assert nodes[0].metadata.get("url") == "https://example.com/page1"
    assert nodes[0].author == "example.com"


async def test_search_caches_query() -> None:
    """同じクエリを 2 回叩いても Tavily 呼び出しは 1 回。"""

    agent = WebSearchAgent(api_key="fake")
    fake = _fake_tavily([{"url": "u", "title": "t", "content": "c"}])
    agent._client = fake

    await agent.search("Q")
    await agent.search("Q")
    assert fake.search.call_count == 1
    assert agent.n_searches_done == 1


async def test_search_respects_global_cap() -> None:
    agent = WebSearchAgent(api_key="fake", max_searches_per_session=2)
    fake = _fake_tavily([{"url": "u", "title": "t", "content": "c"}])
    agent._client = fake

    await agent.search("Q1")
    await agent.search("Q2")
    nodes = await agent.search("Q3")  # cap 越え
    assert nodes == []
    assert fake.search.call_count == 2


async def test_search_returns_empty_when_disabled() -> None:
    agent = WebSearchAgent(api_key=None)
    agent._client = None  # 無効化
    nodes = await agent.search("Q")
    assert nodes == []


# --- maybe_search_for_node -----------------------------------------------


async def test_maybe_search_skips_when_node_has_enough_edges() -> None:
    agent = WebSearchAgent(api_key="fake", min_existing_edges=1)
    fake = _fake_tavily([{"url": "u", "title": "t", "content": "c"}])
    agent._client = fake

    store = NetworkXGraphStore()
    target = Node(text="主張", node_type="claim", source="utterance", author="A")
    other = Node(text="文書", node_type="premise", source="document", author="d1")
    store.add_node(target)
    store.add_node(other)
    # 既に 2 エッジ → 検索不要
    store.add_edge(Edge(src_id=other.id, dst_id=target.id, relation="support", confidence=0.9))
    store.add_edge(Edge(src_id=target.id, dst_id=other.id, relation="attack", confidence=0.7))

    new = await agent.maybe_search_for_node(target, store)
    assert new == []
    assert fake.search.call_count == 0


async def test_maybe_search_fires_for_isolated_claim() -> None:
    agent = WebSearchAgent(api_key="fake", min_existing_edges=1)
    fake = _fake_tavily(
        [{"url": "https://a.com/x", "title": "T", "content": "Relevant snippet"}]
    )
    agent._client = fake

    store = NetworkXGraphStore()
    target = Node(text="孤立主張", node_type="claim", source="utterance", author="A")
    store.add_node(target)

    new = await agent.maybe_search_for_node(target, store)
    assert len(new) == 1
    assert new[0].source == "web"
    # store にも追加されている
    assert any(n.id == new[0].id for n in store.nodes())


async def test_maybe_search_skips_non_claim_nodes() -> None:
    agent = WebSearchAgent(api_key="fake", min_existing_edges=1)
    fake = _fake_tavily([{"url": "u", "title": "t", "content": "c"}])
    agent._client = fake
    store = NetworkXGraphStore()

    # premise (claim ではない) → 対象外
    premise = Node(text="前提", node_type="premise", source="utterance", author="A")
    store.add_node(premise)
    new = await agent.maybe_search_for_node(premise, store)
    assert new == []
    assert fake.search.call_count == 0


async def test_reset_clears_state() -> None:
    agent = WebSearchAgent(api_key="fake")
    agent._client = _fake_tavily([{"url": "u", "title": "t", "content": "c"}])
    await agent.search("Q1")
    assert agent.n_searches_done == 1
    agent.reset()
    assert agent.n_searches_done == 0
    assert "Q1" not in agent._cache
