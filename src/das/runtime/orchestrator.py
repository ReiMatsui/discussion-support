"""エージェント協調動作のオーケストレータ。

外向きの API は同期的に見える (``run_session(transcript)`` で全部終わるまで待つ)
が、内部は ``EventBus`` を通じた pub/sub になっている。

関心事の分離:
  - エージェント: それぞれの専門処理 (extract / link など) のみ知る
  - オーケストレータ: バスへのハンドラ登録、ストア書き込み、二次イベントの publish
  - CLI: orchestrator を組み立てて run_session を呼ぶ
"""

from __future__ import annotations

from pathlib import Path

from das.agents import DocumentAgent, ExtractionAgent, LinkingAgent
from das.agents.web_search import WebSearchAgent
from das.graph.schema import Node
from das.graph.store import GraphStore, NetworkXGraphStore
from das.llm import OpenAIClient
from das.logging import get_logger
from das.runtime.bus import EventBus
from das.types import NodeAdded, Utterance


class Orchestrator:
    """エージェント・バス・ストアを束ねる軽量コーディネータ。"""

    def __init__(
        self,
        *,
        store: GraphStore,
        bus: EventBus,
        extraction: ExtractionAgent,
        document: DocumentAgent,
        linking: LinkingAgent,
        web_search: WebSearchAgent | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._extraction = extraction
        self._document = document
        self._linking = linking
        self._web_search = web_search
        self._log = get_logger("das.runtime.orchestrator")

    # --- 組み立て -----------------------------------------------------

    @classmethod
    def assemble(
        cls,
        *,
        llm: OpenAIClient | None = None,
        store: GraphStore | None = None,
        threshold: float | None = None,
        top_k: int = 5,
        web_search: WebSearchAgent | None = None,
    ) -> Orchestrator:
        """既定構成のオーケストレータを 1 つ作る。

        ``web_search`` を渡すと、新しい utterance/claim ノードに対して
        既存エッジ数が少ないとき自動で Web 検索 → AF 化が走る。
        """

        llm = llm or OpenAIClient()
        store = store or NetworkXGraphStore()
        bus = EventBus()

        extraction = ExtractionAgent(llm=llm)
        document = DocumentAgent(llm=llm)
        linking = LinkingAgent(llm=llm, threshold=threshold, top_k=top_k)

        orch = cls(
            store=store,
            bus=bus,
            extraction=extraction,
            document=document,
            linking=linking,
            web_search=web_search,
        )
        orch._wire_handlers()
        return orch

    def _wire_handlers(self) -> None:
        """エージェントを Bus のハンドラとして登録する。"""

        self._bus.subscribe(Utterance, self._on_utterance)
        self._bus.subscribe(NodeAdded, self._on_node_added)

    # --- ハンドラ -----------------------------------------------------

    async def _on_utterance(self, event: Utterance) -> None:
        nodes = await self._extraction.extract(event)
        for node in nodes:
            self._store.add_node(node)
            await self._bus.publish(NodeAdded(node_id=node.id, source=node.source))

    async def _on_node_added(self, event: NodeAdded) -> None:
        node = self._store.get_node(event.node_id)
        if node is None:  # pragma: no cover - 防御的
            return
        # 連結エージェントは新ノードに対し、既存ノード群との関係を推定する
        await self._linking.link_node(node, self._store)

        # 必要なら Web 検索エージェントが新規根拠を取りに行く。
        # 戻ってきた web ノードもバスに publish して連結を走らせる
        # (= 議論ノード ↔ web ノード のエッジが張られる)。
        if self._web_search is not None and self._web_search.is_enabled:
            new_web_nodes = await self._web_search.maybe_search_for_node(
                node, self._store
            )
            for web_node in new_web_nodes:
                await self._bus.publish(
                    NodeAdded(node_id=web_node.id, source=web_node.source)
                )

    # --- 公開 API -----------------------------------------------------

    @property
    def store(self) -> GraphStore:
        return self._store

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def extraction(self) -> ExtractionAgent:
        return self._extraction

    @property
    def document(self) -> DocumentAgent:
        return self._document

    @property
    def linking(self) -> LinkingAgent:
        return self._linking

    @property
    def web_search(self) -> WebSearchAgent | None:
        return self._web_search

    async def ingest_documents(self, directory: Path) -> list[Node]:
        """事前文書を AF 化してストアに追加する。

        議論前の準備フェーズで呼ぶ想定。バスは経由しない (= ingest 中に
        document ノード同士の連結は走らない)。発話が入ってきたタイミングで
        LinkingAgent が候補として拾う。
        """

        nodes = await self._document.ingest_directory(directory, store=self._store)
        self._log.info("orchestrator.docs_ingested", n_nodes=len(nodes))
        return nodes

    async def run_session(
        self,
        transcript: list[Utterance],
    ) -> GraphStore:
        """発話列を順に流し、すべての二次イベントが収まるまで待つ。"""

        self._log.info("orchestrator.run_session.start", n_utterances=len(transcript))
        for utterance in transcript:
            await self._bus.publish(utterance)
        await self._bus.drain()
        self._log.info(
            "orchestrator.run_session.done",
            n_nodes=len(list(self._store.nodes())),
            n_edges=len(list(self._store.edges())),
        )
        return self._store


__all__ = ["Orchestrator"]
