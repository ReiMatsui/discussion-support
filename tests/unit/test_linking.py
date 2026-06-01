"""LinkingAgent のユニットテスト。

OpenAI 呼び出しは ``embed`` / ``embed_one`` / ``chat_structured`` を AsyncMock で差し替える。
発表資料 §3 のカフェテリア例に近い構成で検証する。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.agents.linking import (
    _BatchJudgment,
    _IndexedJudgment,
    _LinkJudgment,
    LinkingAgent,
    cosine_similarity,
)
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

    async def fake_judge(messages: list[dict], **kwargs: object) -> _BatchJudgment:
        # バッチ 1 回の呼び出し。メッセージ内容に含まれるテキストから候補を抽出
        user_content = messages[1]["content"]
        matched = [
            text
            for text in (nodes["a2"].text, nodes["a3"].text, nodes["a4"].text)
            if text in user_content
        ]
        judged.extend((target.text, text) for text in matched)
        return _batch_none(len(matched))

    llm.chat_structured = AsyncMock(side_effect=fake_judge)  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=1, threshold=0.6)
    await agent.link_node(target, store)

    # 候補は 1 件 (a2) だけ判定に渡される
    assert len(judged) == 1
    assert judged[0][1] == nodes["a2"].text


# --- 関係推定とエッジ書き込み ------------------------------------------


def _judgment(rel: str, conf: float = 0.9) -> _LinkJudgment:
    return _LinkJudgment(relation=rel, confidence=conf, rationale="rationale")  # type: ignore[arg-type]


def _batch1(rel: str, conf: float = 0.9) -> _BatchJudgment:
    """単一候補ぶんのバッチ結果 (index=0)。バッチ既定の judge 戻り値に使う。"""

    return _BatchJudgment(
        judgments=[
            _IndexedJudgment(index=0, relation=rel, confidence=conf, rationale="rationale")  # type: ignore[arg-type]
        ]
    )


def _batch_none(n: int) -> _BatchJudgment:
    """候補 n 件すべてを relation="none" で返すバッチ結果。"""

    return _BatchJudgment(
        judgments=[
            _IndexedJudgment(index=i, relation="none", confidence=0.9, rationale="-")
            for i in range(n)
        ]
    )


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
    llm.chat_structured = AsyncMock(return_value=_batch1("a_attacks_b", 0.9))  # type: ignore[method-assign]

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
    llm.chat_structured = AsyncMock(return_value=_batch1("b_attacks_a", 0.85))  # type: ignore[method-assign]

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
    llm.chat_structured = AsyncMock(return_value=_batch1("b_supports_a", 0.7))  # type: ignore[method-assign]

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
    llm.chat_structured = AsyncMock(return_value=_batch1("none", 0.95))  # type: ignore[method-assign]

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
    llm.chat_structured = AsyncMock(return_value=_batch1("a_attacks_b", 0.3))  # type: ignore[method-assign]

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
    llm.chat_structured = AsyncMock(return_value=_batch1("none", 0.9))  # type: ignore[method-assign]

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


# --- model_override (cheap linking 用) ------------------------------


async def test_linking_uses_model_override_when_set(store: NetworkXGraphStore) -> None:
    """``model_override`` を指定すると judge の chat_structured に model 引数が渡る。"""

    a = Node(text="主張1", node_type="claim", source="utterance", author="A")
    b = Node(text="主張2", node_type="claim", source="utterance", author="B")
    store.add_node(a)
    store.add_node(b)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[0.9, 0.1]])  # type: ignore[method-assign]
    captured = AsyncMock(return_value=_batch1("none", 0.9))
    llm.chat_structured = captured  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5, model_override="gpt-5-nano")
    await agent.link_node(a, store)

    # chat_structured に model="gpt-5-nano" が渡されている
    captured.assert_awaited()
    for call in captured.await_args_list:
        assert call.kwargs.get("model") == "gpt-5-nano"


async def test_linking_no_model_override_means_default_model(
    store: NetworkXGraphStore,
) -> None:
    """``model_override`` 未指定なら chat_structured には model=None が渡る (=既定)。"""

    a = Node(text="主張1", node_type="claim", source="utterance", author="A")
    b = Node(text="主張2", node_type="claim", source="utterance", author="B")
    store.add_node(a)
    store.add_node(b)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[0.9, 0.1]])  # type: ignore[method-assign]
    captured = AsyncMock(return_value=_batch1("none", 0.9))
    llm.chat_structured = captured  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5)
    await agent.link_node(a, store)

    for call in captured.await_args_list:
        assert call.kwargs.get("model") is None


# --- source 別 top-k (Fix A) ----------------------------------------


async def test_top_k_per_source_picks_n_from_each_bucket(
    store: NetworkXGraphStore,
) -> None:
    """``top_k_per_source=1`` のとき、utterance / document / web から 1 件ずつ取る。

    混合 top-k なら utterance だけで埋まる配置でも、source 別なら文書・Web も拾う。
    """

    target = Node(text="新しい発話", node_type="claim", source="utterance", author="A")
    utt1 = Node(text="他の発話1", node_type="claim", source="utterance", author="B")
    utt2 = Node(text="他の発話2", node_type="claim", source="utterance", author="C")
    doc1 = Node(text="文書1", node_type="premise", source="document", author="d1")
    doc2 = Node(text="文書2", node_type="premise", source="document", author="d2")
    web1 = Node(text="Web1", node_type="premise", source="web", author="example.com")
    for n in (utt1, utt2, doc1, doc2, web1):
        store.add_node(n)
    store.add_node(target)

    # target に対する類似度: utterance >> document >> web の順に高く設計
    # 混合 top_k=3 なら utt1, utt2, doc1 を取るが doc2, web1 は落ちる
    # 一方 top_k_per_source=1 なら utt1, doc1, web1 を取り「各 source 1 件」になる
    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0, 0.0, 0.0])  # type: ignore[method-assign]
    embeddings_by_text = {
        utt1.text: [0.99, 0.10, 0.0, 0.0],
        utt2.text: [0.95, 0.20, 0.0, 0.0],
        doc1.text: [0.50, 0.50, 0.50, 0.0],
        doc2.text: [0.40, 0.50, 0.50, 0.0],
        web1.text: [0.20, 0.0, 0.0, 0.80],
    }

    async def fake_embed(texts: list[str], **kwargs: object) -> list[list[float]]:
        return [embeddings_by_text[t] for t in texts]

    llm.embed = AsyncMock(side_effect=fake_embed)  # type: ignore[method-assign]

    judged_texts: list[str] = []

    async def fake_judge(messages: list[dict], **kwargs: object) -> _BatchJudgment:
        user_content = messages[1]["content"]
        matched = [text for text in embeddings_by_text if text in user_content]
        judged_texts.extend(matched)
        return _batch_none(len(matched))

    llm.chat_structured = AsyncMock(side_effect=fake_judge)  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k_per_source=1, threshold=0.6)
    await agent.link_node(target, store)

    # source 別に各 1 件ずつ取れているはず: utt1 (最高 utterance), doc1 (最高 document), web1 (唯一 web)
    assert sorted(judged_texts) == sorted([utt1.text, doc1.text, web1.text])


async def test_top_k_per_source_overrides_top_k(store: NetworkXGraphStore) -> None:
    """``top_k_per_source`` を指定すると ``top_k`` は無視される。"""

    target = Node(text="新発話", node_type="claim", source="utterance", author="A")
    utts = [
        Node(text=f"u{i}", node_type="claim", source="utterance", author=f"S{i}")
        for i in range(5)
    ]
    doc = Node(text="d1", node_type="premise", source="document", author="d1")
    for n in utts + [doc]:
        store.add_node(n)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]

    async def fake_embed(texts: list[str], **kwargs: object) -> list[list[float]]:
        # すべてのノードに類似度を返す。utterance を高く、document を低くしておく。
        out = []
        for t in texts:
            if t == "d1":
                out.append([0.3, 0.7])
            else:
                out.append([0.9, 0.1])
        return out

    llm.embed = AsyncMock(side_effect=fake_embed)  # type: ignore[method-assign]

    captured_contents: list[str] = []

    async def fake_judge(messages: list[dict], **kwargs: object) -> _BatchJudgment:
        content = messages[1]["content"]
        captured_contents.append(content)
        # 候補数 = "  text:" 行数 - 1 (A=target ぶんを差し引く)
        n_candidates = content.count("  text:") - 1
        return _batch_none(n_candidates)

    llm.chat_structured = AsyncMock(side_effect=fake_judge)  # type: ignore[method-assign]

    # top_k=100 だが top_k_per_source=2 が優先される
    agent = LinkingAgent(llm=llm, top_k=100, top_k_per_source=2, threshold=0.6)
    await agent.link_node(target, store)

    # バッチなので候補数によらず chat completion は 1 回だけ
    assert llm.chat_structured.await_count == 1
    # 候補数 = utterance 2 + document 1 = 3 (web は無いので skip)
    assert captured_contents[0].count("  text:") - 1 == 3


async def test_link_node_runs_judges_in_parallel(store: NetworkXGraphStore) -> None:
    """``batch=False`` のレガシー経路で候補の judge を並列に走らせることを確認する。

    候補ノードを 4 つ用意し、各 judge が遅延を持つようにしておいて、
    逐次なら 4×delay、並列なら 1×delay 程度で終わることを観測する。
    (バッチ既定では呼び出しが 1 回に集約されるため、この回帰は per-pair 経路向け。)
    """

    import time

    target = Node(text="t", node_type="claim", source="utterance", author="A")
    cands = [
        Node(text=f"c{i}", node_type="claim", source="utterance", author=f"S{i}")
        for i in range(4)
    ]
    for n in cands:
        store.add_node(n)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(  # type: ignore[method-assign]
        return_value=[[0.9, 0.1] for _ in cands]
    )

    delay = 0.05  # 50ms

    async def slow_judge(*args: object, **kwargs: object) -> _LinkJudgment:
        import asyncio as _asyncio

        await _asyncio.sleep(delay)
        return _judgment("none", 0.9)

    llm.chat_structured = AsyncMock(side_effect=slow_judge)  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=10, threshold=0.6, batch=False)

    t0 = time.monotonic()
    await agent.link_node(target, store)
    elapsed = time.monotonic() - t0

    # 4 並列ならおおむね delay の 2 倍未満で終わる (overhead を考慮しても 4×delay の半分)。
    # 逐次なら 4×delay = 0.20s かかる。
    assert elapsed < delay * 3, (
        f"判定が並列化されていない可能性。elapsed={elapsed:.3f}s "
        f"(逐次想定 {delay * 4:.3f}s / 並列想定 {delay:.3f}s)"
    )
    assert llm.chat_structured.await_count == 4


# --- judge timeout (hang 保護) ---------------------------------------


async def test_hung_judge_times_out_and_yields_none_edge(
    store: NetworkXGraphStore,
) -> None:
    """``batch=False`` の per-pair 経路で 1 件が hang しても他候補は edge になる。

    並列バッチの中の 1 件が long-hang する OpenAI 障害シナリオを模擬する。
    対象が 3 つあり、うち 1 つだけ無限待ち、残り 2 つは即座に support を返す。
    judge_timeout=0.1 を設定すると、hang した 1 件のみ none 化され、
    他 2 件は通常通り edge が作られる。(バッチ既定では呼び出しが 1 回なので、
    per-pair の粒度で timeout を切れるのはこの経路のみ。)
    """

    import asyncio as _asyncio

    target = Node(text="t", node_type="claim", source="utterance", author="A")
    fast1 = Node(text="fast1", node_type="claim", source="utterance", author="B")
    fast2 = Node(text="fast2", node_type="claim", source="utterance", author="C")
    slow = Node(text="slow", node_type="claim", source="utterance", author="D")
    for n in (fast1, fast2, slow):
        store.add_node(n)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(  # type: ignore[method-assign]
        return_value=[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]
    )

    async def judge_dispatch(messages: list[dict], **kwargs: object) -> _LinkJudgment:
        user_content = messages[1]["content"]
        if "slow" in user_content:
            # 永久に終わらない (timeout でキャンセルされる想定)
            await _asyncio.sleep(60.0)
            return _judgment("a_supports_b", 0.9)  # ここには到達しない
        return _judgment("a_supports_b", 0.9)

    llm.chat_structured = AsyncMock(side_effect=judge_dispatch)  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=10, threshold=0.6, judge_timeout=0.1, batch=False)

    import time

    t0 = time.monotonic()
    edges = await agent.link_node(target, store)
    elapsed = time.monotonic() - t0

    # hang した 1 件が timeout=0.1 で打ち切られ、バッチ全体も 1 秒未満で抜ける
    assert elapsed < 1.0, f"timeout が効いていない。elapsed={elapsed:.3f}s"
    # fast 2 件だけ edge になり、slow は edge にならない
    assert len(edges) == 2
    edge_dsts = {e.dst_id for e in edges}
    assert fast1.id in edge_dsts
    assert fast2.id in edge_dsts
    assert slow.id not in edge_dsts


async def test_judge_timeout_none_passes_through(store: NetworkXGraphStore) -> None:
    """``judge_timeout=None`` なら従来通り wait_for ラップなしで動く (回帰防止)。"""

    target = Node(text="t", node_type="claim", source="utterance", author="A")
    cand = Node(text="c", node_type="claim", source="utterance", author="B")
    store.add_node(cand)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[0.9, 0.1]])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_batch1("a_supports_b", 0.9)
    )

    agent = LinkingAgent(llm=llm, top_k=10, threshold=0.6, judge_timeout=None)
    edges = await agent.link_node(target, store)
    assert len(edges) == 1


# --- バッチ判定 (top-k 候補をまとめて 1 回) -----------------------------


def _parse_index_to_text(content: str) -> dict[int, str]:
    """バッチ user メッセージから ``index -> candidate text`` の対応を抜き出す。"""

    import re

    mapping: dict[int, str] = {}
    cur: int | None = None
    for line in content.splitlines():
        stripped = line.strip()
        m = re.fullmatch(r"\[(\d+)\]", stripped)
        if m:
            cur = int(m.group(1))
        elif cur is not None and stripped.startswith("text:"):
            mapping[cur] = stripped[len("text:") :].strip()
            cur = None
    return mapping


async def test_batch_single_chat_call_for_k_candidates(store: NetworkXGraphStore) -> None:
    """候補 k 件に対し chat completion 呼び出しは 1 回だけ (O(top_k)→O(1))。"""

    target = Node(text="t", node_type="claim", source="utterance", author="A")
    cands = [
        Node(text=f"c{i}", node_type="claim", source="utterance", author=f"S{i}")
        for i in range(6)
    ]
    for n in cands:
        store.add_node(n)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(  # type: ignore[method-assign]
        return_value=[[0.9, 0.1] for _ in cands]
    )
    llm.chat_structured = AsyncMock(return_value=_batch_none(6))  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=6, threshold=0.6)
    await agent.link_node(target, store)

    assert llm.chat_structured.await_count == 1


async def test_batch_aligns_judgments_despite_missing_and_reordered(
    store: NetworkXGraphStore,
) -> None:
    """LLM が一部候補を省略 / 順序を入れ替えても index で正しくアラインする。

    候補 3 件 (hi / mid / lo) に対し、バッチ応答は順序を逆転させ mid を省略する。
    欠けた mid は none 扱いでエッジ無し、hi / lo は index 経由で正しくエッジ化される。
    """

    target = Node(text="target", node_type="claim", source="utterance", author="A")
    hi = Node(text="hi", node_type="claim", source="utterance", author="B")
    mid = Node(text="mid", node_type="claim", source="utterance", author="C")
    lo = Node(text="lo", node_type="premise", source="web", author="example.com")
    for n in (hi, mid, lo):
        store.add_node(n)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    embeddings_by_text = {
        hi.text: [1.0, 0.0],
        mid.text: [0.8, 0.6],
        lo.text: [0.6, 0.8],
    }

    async def fake_embed(texts: list[str], **kwargs: object) -> list[list[float]]:
        return [embeddings_by_text[t] for t in texts]

    llm.embed = AsyncMock(side_effect=fake_embed)  # type: ignore[method-assign]

    async def fake_judge(messages: list[dict], **kwargs: object) -> _BatchJudgment:
        mapping = _parse_index_to_text(messages[1]["content"])
        out: list[_IndexedJudgment] = []
        for idx, text in mapping.items():
            if text == hi.text:
                out.append(
                    _IndexedJudgment(index=idx, relation="a_attacks_b", confidence=0.9)
                )
            elif text == lo.text:
                out.append(
                    _IndexedJudgment(index=idx, relation="b_supports_a", confidence=0.9)
                )
            # mid は意図的に省略
        out.reverse()  # 順序ズレを模擬
        return _BatchJudgment(judgments=out)

    llm.chat_structured = AsyncMock(side_effect=fake_judge)  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=5, threshold=0.6)
    edges = await agent.link_node(target, store)

    assert llm.chat_structured.await_count == 1
    by_pair = {(e.src_id, e.dst_id): e for e in edges}
    # hi: a_attacks_b → target -> hi (attack)
    assert by_pair[(target.id, hi.id)].relation == "attack"
    # lo: b_supports_a → lo -> target (support)
    assert by_pair[(lo.id, target.id)].relation == "support"
    # mid は省略されたので none 扱い、エッジ無し
    assert mid.id not in {e.src_id for e in edges} | {e.dst_id for e in edges}
    assert len(edges) == 2


async def test_batch_timeout_yields_no_edges(store: NetworkXGraphStore) -> None:
    """バッチ呼び出しが hang したら全候補 none 扱いでエッジ無し、即座に復帰する。"""

    import asyncio as _asyncio
    import time

    target = Node(text="t", node_type="claim", source="utterance", author="A")
    cands = [
        Node(text=f"c{i}", node_type="claim", source="utterance", author=f"S{i}")
        for i in range(3)
    ]
    for n in cands:
        store.add_node(n)
    store.add_node(target)

    llm = _fake_llm()
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(  # type: ignore[method-assign]
        return_value=[[0.9, 0.1] for _ in cands]
    )

    async def hang(*args: object, **kwargs: object) -> _BatchJudgment:
        await _asyncio.sleep(60.0)
        return _batch_none(3)  # 到達しない

    llm.chat_structured = AsyncMock(side_effect=hang)  # type: ignore[method-assign]

    agent = LinkingAgent(llm=llm, top_k=10, threshold=0.6, judge_timeout=0.1)

    t0 = time.monotonic()
    edges = await agent.link_node(target, store)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"batch timeout が効いていない。elapsed={elapsed:.3f}s"
    assert edges == []
