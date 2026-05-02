"""Web 検索エージェント (研究計画書 §4.1 — 5 つ目の専門エージェント)。

事前資料に無い論点や時事情報を **リアルタイムで検索** し、議論グラフに
``source="web"`` のノードとして追加する。直後に ``LinkingAgent`` が
新しい web ノードと既存ノード (発話 / 文書) との支持・攻撃関係を判定する。

設計上の選択:
  - 検索バックエンドは Tavily (LLM-friendly な Search API)
  - 検索発火条件は「**新発話 claim に既存 AF からの繋がりが少ない**」
    (既に十分な根拠があれば追加検索しない = コスト抑制)
  - グローバルキャップ ``max_searches_per_session`` でコスト爆発を防ぐ
  - クエリのキャッシュで重複検索を防ぐ
  - Tavily 未インストール / API キー欠如のときは静かに no-op

検索結果は **そのまま渡さず**、``Node(source="web", node_type="premise")``
として AF に整形して入れる (=「広い知識」も同じ AF 上で扱える)。
"""

from __future__ import annotations

import asyncio
from typing import Any

from das.agents.base import BaseAgent
from das.graph.schema import Node
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.settings import get_settings


class WebSearchAgent(BaseAgent):
    """Tavily を使った Web 検索 + AF 化エージェント。"""

    name = "web_search"

    def __init__(
        self,
        llm: OpenAIClient | None = None,
        *,
        max_searches_per_session: int = 5,
        min_existing_edges: int = 1,
        max_results_per_query: int = 3,
        api_key: str | None = None,
    ) -> None:
        """
        Parameters:
          - ``max_searches_per_session``: セッション全体の検索回数上限
          - ``min_existing_edges``: claim ノードがこれ以下の隣接エッジしか持たないとき
            だけ検索 (= 既存知識で議論が回っているなら検索しない)
          - ``max_results_per_query``: 1 検索あたりに採用する web ノード数
          - ``api_key``: Tavily API キー。省略時は ``Settings.tavily_api_key``
        """

        super().__init__(llm=llm)
        self._max_searches = max_searches_per_session
        self._min_edges = min_existing_edges
        self._max_results = max_results_per_query
        self._n_searches_done = 0
        self._cache: dict[str, list[Node]] = {}

        settings = get_settings()
        resolved_key = api_key or settings.tavily_api_key
        self._client: Any | None = None
        if resolved_key:
            try:
                from tavily import TavilyClient

                self._client = TavilyClient(api_key=resolved_key)
            except ImportError:
                self.log.warning("web_search.tavily_not_installed")
                self._client = None
        else:
            self.log.info("web_search.disabled_no_api_key")

    @property
    def is_enabled(self) -> bool:
        """Tavily クライアントが使える状態かどうか。"""

        return self._client is not None

    @property
    def n_searches_done(self) -> int:
        """このインスタンスがこれまで実行した検索回数。"""

        return self._n_searches_done

    def reset(self) -> None:
        """セッション切替時にカウンタとキャッシュをリセット。"""

        self._n_searches_done = 0
        self._cache.clear()

    # --- 検索本体 ----------------------------------------------------

    async def search(self, query: str) -> list[Node]:
        """``query`` で Web 検索し、結果を web ノードのリストとして返す。

        - クライアント未初期化なら空リスト
        - キャッシュヒットならキャッシュ結果を返す
        - グローバルキャップを超えていれば空リスト
        """

        if not self._client:
            return []
        if self._n_searches_done >= self._max_searches:
            self.log.info("web_search.cap_reached", n=self._n_searches_done)
            return []

        normalized = query.strip()[:200]
        if normalized in self._cache:
            return list(self._cache[normalized])

        self._n_searches_done += 1
        self.log.info(
            "web_search.query", query=normalized[:80], idx=self._n_searches_done
        )

        try:
            # Tavily は同期 SDK なので thread に逃がす
            response = await asyncio.to_thread(
                self._client.search,
                query=normalized,
                max_results=self._max_results,
                search_depth="basic",
                include_answer=False,
            )
        except Exception as exc:  # pragma: no cover - 防御的
            self.log.warning("web_search.failed", error=str(exc))
            self._cache[normalized] = []
            return []

        results = response.get("results", []) if isinstance(response, dict) else []
        nodes: list[Node] = []
        for item in results[: self._max_results]:
            text = (item.get("content") or item.get("title") or "").strip()
            if not text:
                continue
            url = item.get("url", "")
            domain = url.split("/")[2] if "://" in url else url[:60]
            nodes.append(
                Node(
                    text=text[:500],
                    node_type="premise",
                    source="web",
                    author=domain,
                    metadata={
                        "url": url,
                        "title": item.get("title", ""),
                        "query": normalized,
                    },
                )
            )

        self._cache[normalized] = nodes
        self.log.info(
            "web_search.results", query=normalized[:80], n=len(nodes)
        )
        return nodes

    async def maybe_search_for_node(
        self,
        node: Node,
        store: GraphStore,
    ) -> list[Node]:
        """``node`` に必要十分な根拠が無いとき検索し、結果を ``store`` に追加する。

        条件:
          - node が utterance/claim (= 議論側の主張)
          - 既存の隣接エッジ数 ≤ ``min_existing_edges``
          - 検索キャップ未達

        戻り値は store に追加された web ノードのリスト (Linking 側がさらに処理する)。
        """

        if not self.is_enabled:
            return []
        if node.source != "utterance" or node.node_type != "claim":
            return []

        existing_edges = sum(
            1 for e in store.edges() if e.dst_id == node.id or e.src_id == node.id
        )
        if existing_edges > self._min_edges:
            return []

        new_nodes = await self.search(node.text)
        for n in new_nodes:
            store.add_node(n)
        return new_nodes


__all__ = ["WebSearchAgent"]
