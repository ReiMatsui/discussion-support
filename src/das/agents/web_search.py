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
  - **cooldown**: 直近 N 秒以内に検索済みなら発火しない (レイテンシ制御)
  - **lazy モード**: stalled シグナルが来たときだけ検索する (対面議論用)
  - Tavily 未インストール / API キー欠如のときは静かに no-op

検索結果は **そのまま渡さず**、``Node(source="web", node_type="evidence")``
として AF に整形して入れる (=「広い知識」も同じ AF 上で扱える)。Web 結果は
立場を持つ主張ではなく中立な事実なので evidence 扱いとし、特定の主張への
支持/攻撃 (スタンス) は連結エージェントが対象主張ごとにエッジで判定する。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.graph.schema import Node
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.settings import get_settings

_QUERY_SYSTEM_PROMPT = (
    "あなたは検索クエリ生成器です。与えられた主張文を、検索エンジンで裏取りするための"
    "短い検索クエリ (固有名詞・数値・論点語を含むキーワード列) に変換してください。"
    "主張文の当為表現 (〜すべき) や主観は落とし、事実確認に有効な語を残します。"
    "日本語はそのまま日本語で。1 行のクエリだけを返します。"
)


class _SearchQuery(BaseModel):
    """検索クエリ生成の構造化出力。"""

    query: str = Field(description="検索エンジン向けの短いクエリ (キーワード列)")


SearchPolicy = Literal["eager", "lazy"]
"""検索ポリシー。
- eager: claim がエッジ不足なら即検索 (シミュレーション向け)
- lazy: stalled シグナルが来たときだけ検索 (対面リアルタイム向け)
"""


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
        cooldown_seconds: float = 0.0,
        policy: SearchPolicy = "eager",
    ) -> None:
        """
        Parameters:
          - ``max_searches_per_session``: セッション全体の検索回数上限
          - ``min_existing_edges``: claim ノードがこれ以下の隣接エッジしか持たないとき
            だけ検索 (= 既存知識で議論が回っているなら検索しない)
          - ``max_results_per_query``: 1 検索あたりに採用する web ノード数
          - ``api_key``: Tavily API キー。省略時は ``Settings.tavily_api_key``
          - ``cooldown_seconds``: 連続検索を抑制する秒数。直近の検索からこの秒数以内は
            ``maybe_search_for_node`` が no-op になる。対面リアルタイム向けに 10-30 秒
            程度を設定するとレイテンシが改善する。既定 0 (無制限)。
          - ``policy``: 検索発火ポリシー。
            - "eager": claim がエッジ不足なら即検索 (従来動作、シミュレーション向け)
            - "lazy": ``signal_stalled()`` が呼ばれたときだけ検索を許可する
              (対面リアルタイム向け)。FacilitationAgent の stalled 検知と組み合わせる。
        """

        super().__init__(llm=llm)
        self._max_searches = max_searches_per_session
        self._min_edges = min_existing_edges
        self._max_results = max_results_per_query
        self._n_searches_done = 0
        self._cache: dict[str, list[Node]] = {}
        self._cooldown_seconds = cooldown_seconds
        self._last_search_time: float = 0.0
        self._policy = policy
        self._stalled_signal: bool = False
        self._pending_queries: list[tuple[Node, GraphStore]] = []

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
        self._last_search_time = 0.0
        self._stalled_signal = False
        self._pending_queries.clear()

    def signal_stalled(self) -> None:
        """lazy ポリシー時に「議論が停滞した」シグナルを受け取る。

        FacilitationAgent が stalled を検知したとき呼ぶことで、
        pending_queries に溜まった検索を次の ``flush_pending()`` で実行する。
        eager ポリシーでは no-op。
        """
        self._stalled_signal = True

    async def flush_pending(self) -> list[Node]:
        """lazy ポリシーで溜めた pending queries を実行する。

        ``signal_stalled()`` の後に orchestrator / facilitation が呼ぶ。
        戻り値は新しく作られた web ノード群。
        """
        if not self._stalled_signal or not self._pending_queries:
            return []
        self._stalled_signal = False
        all_nodes: list[Node] = []
        for node, store in self._pending_queries:
            new_nodes = await self._do_search_for_node(node, store)
            all_nodes.extend(new_nodes)
        self._pending_queries.clear()
        return all_nodes

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
                    node_type="evidence",
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
          - cooldown 経過済み
          - policy が eager、または lazy で stalled シグナル受信済み

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

        # Lazy policy: キューに溜めて signal_stalled() 後に flush する
        if self._policy == "lazy":
            self._pending_queries.append((node, store))
            self.log.info(
                "web_search.lazy_queued",
                node_id=str(node.id),
                pending=len(self._pending_queries),
            )
            return []

        # Cooldown check
        if self._cooldown_seconds > 0:
            elapsed = time.monotonic() - self._last_search_time
            if elapsed < self._cooldown_seconds:
                self.log.info(
                    "web_search.cooldown_skip",
                    elapsed=round(elapsed, 1),
                    cooldown=self._cooldown_seconds,
                )
                return []

        return await self._do_search_for_node(node, store)

    async def _do_search_for_node(self, node: Node, store: GraphStore) -> list[Node]:
        """実際に検索を実行して store に追加する内部メソッド。"""

        # G4: 主張原文をそのままクエリにせず、検索エンジン向けクエリに変換する
        query = await self._generate_query(node.text)
        new_nodes = await self.search(query)
        self._last_search_time = time.monotonic()
        for n in new_nodes:
            store.add_node(n)
        return new_nodes

    async def _generate_query(self, claim_text: str) -> str:
        """主張文を検索エンジン向けクエリ (キーワード列) に変換する (G4, レビュー M-4)。

        失敗・空のときは主張原文にフォールバックする (検索自体は止めない)。
        """

        messages = [
            {"role": "system", "content": _QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": claim_text},
        ]
        try:
            result = await self.llm.chat_structured(
                messages,  # type: ignore[arg-type]
                response_format=_SearchQuery,
            )
        except Exception as exc:  # pragma: no cover - 防御的: 失敗時は原文で検索
            self.log.warning("web_search.query_gen_failed", error=str(exc))
            return claim_text
        query = result.query.strip()
        if not query:
            return claim_text
        self.log.info("web_search.query_generated", claim=claim_text[:60], query=query)
        return query


__all__ = ["SearchPolicy", "WebSearchAgent"]
