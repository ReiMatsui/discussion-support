"""ライブセッションに統合議論グラフ (AF) を常駐させるランタイム (H1 フェーズ3)。

確定発話を専用スレッドで取り込み、extraction → linking で AF を育てる。

このフェーズでは **介入は一切行わない** (Controller 接続・AF checker はフェーズ4)。
目的は 2 つ:
  1. ライブセッション中に AF を常駐・成長させる基盤を用意する
  2. 「発話確定 → ノード追加 → エッジ追加」のレイテンシを実測してログに残す
     (エッジ追加が 10 秒を超えるならフェーズ4前に linking 対象の窓絞りで最適化する)

既存の checker (topic/drift/fact...) と同じ「records カーソル + meeting epoch ガード」
パターンを踏襲する。live 側は asyncio を持たないので、専用スレッド内に event loop を
1 つ立てて extraction/linking を同期的に回す。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from das.agents.linking import cosine_similarity
from das.graph.store import GraphStore, NetworkXGraphStore
from das.llm import OpenAIClient
from das.logging import get_logger
from das.runtime import Orchestrator
from das.types import Utterance

from ._constants import AGENT_SPEAKER
from ._speaker_policy import intervention_records, intervention_speaker_name

_log = get_logger("das.asr.live.af_runtime")


def _percentile(values: list[float], pct: float) -> float | None:
    """線形補間なしの単純な百分位数 (p50/p90 の粗い実測用)。"""

    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


class AFRuntime:
    """ライブセッションに常駐する AF (store + Orchestrator)。介入はしない。"""

    def __init__(
        self,
        state: Any,
        *,
        llm: OpenAIClient,
        docs_dir: str | Path | None = None,
        poll_interval: float = 0.5,
        snapshot_debounce_sec: float = 60.0,
        snapshot_path: str | Path | None = None,
        linking_model: str | None = None,
    ) -> None:
        self._state = state
        self._llm = llm
        self._linking_model = linking_model
        self._docs_dir = Path(docs_dir) if docs_dir else None
        self._poll_interval = poll_interval
        self._snapshot_debounce = snapshot_debounce_sec
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None

        self._store: GraphStore = NetworkXGraphStore()
        self._orch = Orchestrator.assemble(
            llm=llm, store=self._store, linking_model=linking_model
        )
        self._cursor = 0
        self._epoch = getattr(state, "meeting_epoch", 0)
        # 指示語解決の参照文脈に使う直近発話 (G2)
        self._recent_utts: list[Utterance] = []
        # レイテンシ実測 (ms)。フェーズ3 完了時に p50/p90 を報告する。
        self.latencies_ms: dict[str, list[float]] = {
            "extraction": [],
            "linking": [],
            "total": [],
        }
        # 介入ノード・応答エッジの計測 (フェーズ5, B4)。制御には使わず記録のみ。
        # AF の argumentation スキーマ (support/attack) を汚さないよう、Node/Edge では
        # なく計測専用の構造で保持し snapshot / interventions.jsonl に落とす。
        self._interventions: list[dict[str, Any]] = []
        self._response_edges: list[dict[str, Any]] = []
        self._intervention_lock = threading.Lock()
        self._responds_threshold = 0.6

    def note_intervention(self, kind: str, text: str) -> None:
        """worker が af 介入を配信したときに呼ぶ (別スレッド安全)。受容計測用に記録する。

        提示された介入を「intervention ノード」相当として記録し、以降の発話取り込みで
        embedding 類似が閾値超なら responds_to (受容の痕跡) を張る = ライブ版 citation。
        """
        if not text:
            return
        with self._intervention_lock:
            iv_id = f"iv-{len(self._interventions) + 1:03d}"
            self._interventions.append({
                "id": iv_id,
                "kind": kind,
                "text": text,
                "at": time.time(),
                "presented_at_turn": self._cursor,
                "embedding": None,  # 遅延埋め込み (af_runtime スレッドで計算)
            })

    @property
    def store(self) -> GraphStore:
        return self._store

    # --- 取り込み ------------------------------------------------------

    async def ingest_documents(self) -> None:
        """事前文書を evidence として投入する (あれば)。"""

        if self._docs_dir is not None and self._docs_dir.exists():
            await self._orch.ingest_documents(self._docs_dir)
            _log.info("af_runtime.docs_ingested", docs_dir=str(self._docs_dir))

    async def ingest_utterance(
        self, utterance: Utterance, context: list[Utterance] | None = None
    ) -> None:
        """1 発話を extraction → linking で AF に取り込み、レイテンシを記録する。

        「ノード追加」「エッジ追加」を分けて計測するため、bus.drain ではなく
        extraction / linking を直接呼ぶ (フェーズ3 は介入なしなので web_search /
        NodeAdded 連鎖は不要)。フェーズ4 で Controller に繋ぐ際に bus 経路へ戻す。
        ``context`` は指示語解決の参照文脈 (G2)。
        """

        t0 = time.monotonic()
        result = await self._orch.extraction.extract(utterance, context=context)
        for node in result.nodes:
            self._store.add_node(node)
        for edge in result.edges:
            self._store.add_edge(edge)
        nodes = result.nodes
        t_node = time.monotonic()
        for node in nodes:
            await self._orch.linking.link_node(node, self._store)
        t_edge = time.monotonic()

        ext_ms = (t_node - t0) * 1000.0
        link_ms = (t_edge - t_node) * 1000.0
        total_ms = (t_edge - t0) * 1000.0
        self.latencies_ms["extraction"].append(ext_ms)
        self.latencies_ms["linking"].append(link_ms)
        self.latencies_ms["total"].append(total_ms)
        # 受容の痕跡 (responds_to) を計測する (フェーズ5, 制御には使わない)
        await self._detect_responds_to(nodes)
        _log.info(
            "af_runtime.ingested",
            turn_index=utterance.turn_id,
            speaker=utterance.speaker,
            n_nodes=len(nodes),
            extraction_ms=round(ext_ms, 1),
            linking_ms=round(link_ms, 1),
            total_ms=round(total_ms, 1),
        )

    async def _detect_responds_to(self, new_nodes: list[Any]) -> None:
        """新規発話ノードが過去の af 介入に応答しているか (embedding 類似) を記録する。

        閾値超なら responds_to (受容の痕跡) を計測レイヤに追加する。linking が計算
        済みの発話ノード embedding を再利用し、介入テキストのみ遅延埋め込みする。
        """

        with self._intervention_lock:
            interventions = list(self._interventions)
        if not interventions or not new_nodes:
            return
        node_vecs = self._orch.linking.embeddings  # {node_id: vec}
        for iv in interventions:
            if iv.get("embedding") is None:
                try:
                    iv["embedding"] = await self._llm.embed_one(str(iv["text"]))
                except Exception:  # pragma: no cover - 防御的
                    continue
            for node in new_nodes:
                vec = node_vecs.get(node.id)
                if vec is None:
                    continue
                sim = cosine_similarity(iv["embedding"], vec)
                if sim >= self._responds_threshold:
                    self._response_edges.append({
                        "intervention_id": iv["id"],
                        "utterance_node_id": str(node.id),
                        "similarity": round(sim, 3),
                        "responded_at_turn": node.turn_index,
                    })

    def latency_summary(self) -> dict[str, Any]:
        """各フェーズの p50/p90/件数を返す (レイテンシ実測レポート用)。"""

        out: dict[str, Any] = {}
        for phase, values in self.latencies_ms.items():
            out[phase] = {
                "n": len(values),
                "p50_ms": _percentile(values, 50),
                "p90_ms": _percentile(values, 90),
                "max_ms": max(values) if values else None,
            }
        return out

    # --- 発話ソース ----------------------------------------------------

    def _build_utterance(self, record: dict[str, Any], turn_id: int) -> Utterance:
        speaker = intervention_speaker_name(self._state, record)
        return Utterance(
            turn_id=turn_id,
            speaker=speaker,
            text=str(record.get("text") or ""),
            timestamp=datetime.now(UTC),
        )

    def _reset_for_new_meeting(self, epoch: int) -> None:
        """会議世代が変わったら store と cursor をリセットする (H2 と同思想)。"""

        self._store = NetworkXGraphStore()
        self._orch = Orchestrator.assemble(
            llm=self._llm, store=self._store, linking_model=self._linking_model
        )
        self._cursor = 0
        self._epoch = epoch
        self._recent_utts = []
        with self._intervention_lock:
            self._interventions = []
        self._response_edges = []
        _log.info("af_runtime.reset", epoch=epoch)

    def acceptance_summary(self) -> dict[str, Any]:
        """ライブ版 citation: 配信した af 介入のうち応答された割合 (受容性指標)。"""

        with self._intervention_lock:
            n_iv = len(self._interventions)
        responded = {e["intervention_id"] for e in self._response_edges}
        return {
            "n_interventions": n_iv,
            "n_responded": len(responded),
            "acceptance_rate": (len(responded) / n_iv) if n_iv else 0.0,
            "n_response_edges": len(self._response_edges),
        }

    def save_snapshot(self) -> None:
        if self._snapshot_path is None:
            return
        try:
            # snapshot に介入・応答エッジの計測を同梱する (embedding は保存しない)。
            with self._intervention_lock:
                ivs = [
                    {k: v for k, v in iv.items() if k != "embedding"}
                    for iv in self._interventions
                ]
            payload = {
                **self._store.snapshot(),
                "af_interventions": ivs,
                "af_response_edges": list(self._response_edges),
            }
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self._snapshot_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            # 受容の痕跡 (responds_to) を interventions.jsonl にも落とす (段階C資産)
            iv_path = self._snapshot_path.with_suffix(".interventions.jsonl")
            iv_path.write_text(
                "\n".join(json.dumps(e, ensure_ascii=False) for e in self._response_edges),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - 防御的
            _log.warning("af_runtime.snapshot_failed", error=str(exc))

    # --- ポーリング ----------------------------------------------------

    def poll_once(self, loop: asyncio.AbstractEventLoop) -> int:
        """1 回分のポーリング: 新しい確定発話を取り込む。取り込んだ件数を返す。

        run() から周期的に呼ばれる。テストからも直接呼べるよう分離してある。
        """

        state = self._state
        with state.state_lock:
            epoch = state.meeting_epoch
            talk_rs = intervention_records(
                [
                    r
                    for r in state.records
                    if "speaker" in r
                    and r.get("text")
                    and r.get("speaker") != AGENT_SPEAKER
                ]
            )

        # 会議リセット跨ぎ: store/cursor を新世代に合わせて初期化して抜ける
        if epoch != self._epoch:
            self._reset_for_new_meeting(epoch)
            return 0

        n = len(talk_rs)
        if n <= self._cursor:
            return 0

        new = talk_rs[self._cursor : n]
        ingested = 0
        for offset, record in enumerate(new, start=1):
            turn_id = self._cursor + offset
            utterance = self._build_utterance(record, turn_id)
            context = self._recent_utts[-3:]
            try:
                loop.run_until_complete(self.ingest_utterance(utterance, context))
                ingested += 1
            except Exception as exc:  # pragma: no cover - 防御的
                _log.warning("af_runtime.ingest_failed", error=str(exc))
            self._recent_utts.append(utterance)

        # cursor 書き戻し (副作用) の直前に epoch を再確認 (H2)。
        with state.state_lock:
            if state.meeting_epoch != epoch:
                return ingested
            self._cursor = n
        return ingested

    def run(self) -> None:
        """専用スレッドのエントリポイント。停止まで records をポーリングする。"""

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.ingest_documents())
        except Exception as exc:  # pragma: no cover - 防御的
            _log.warning("af_runtime.docs_failed", error=str(exc))

        last_snapshot = time.monotonic()
        state = self._state
        _log.info("af_runtime.started", poll_interval=self._poll_interval)
        while not state.stop.is_set():
            time.sleep(self._poll_interval)
            self.poll_once(loop)
            now = time.monotonic()
            if now - last_snapshot >= self._snapshot_debounce:
                self.save_snapshot()
                last_snapshot = now

        self.save_snapshot()
        loop.close()
        _log.info("af_runtime.stopped", latency=self.latency_summary())


def run_af_runtime(
    state: Any,
    oai_key: str,
    oai_model: str,
    *,
    docs_dir: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    linking_model: str | None = None,
) -> None:
    """スレッド target。AFRuntime を組み立てて state に保持し、ポーリングを回す。

    ``oai_key`` は既存 checker と揃えたシグネチャのため受け取るが、``OpenAIClient`` は
    settings (.env) から鍵を読むため直接は使わない。``oai_model`` も同様 (extraction/
    linking は settings の既定モデルを使う)。
    """

    llm = OpenAIClient()
    runtime = AFRuntime(
        state,
        llm=llm,
        docs_dir=docs_dir,
        snapshot_path=snapshot_path,
        linking_model=linking_model,
    )
    # ライブビュー等から参照できるよう state に保持する
    state.af_runtime = runtime
    runtime.run()


__all__ = ["AFRuntime", "run_af_runtime"]
