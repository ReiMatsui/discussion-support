"""L3 自然言語要約 (議論区切りでの全員共有)。

研究計画書 §4.4 L3 に対応する実装:
  「現在の論点は A と B の対立に整理され、A を支持する根拠は 3 件、
   B を支持する根拠は 2 件、応答されていない反論が 1 件あります」

実装方針:
  - ``programmatic_summary`` は LLM 不要で、graph_metrics を文字列化する決定的サマリ
  - ``llm_summary`` は LLM を使って自然な日本語の 3〜5 文の要約に整える
  - ``summarize_session`` は llm が None なら programmatic にフォールバック

L3 はリアルタイム提示ではなく区切り提示なので、smart モデル + temperature 低めで OK。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from das.eval.metrics import graph_metrics, transcript_metrics
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.types import Utterance


@dataclass(frozen=True)
class SessionSummary:
    """L3 要約の出力。``text`` を全員に提示する。"""

    text: str
    structural_lines: list[str] = field(default_factory=list)
    n_nodes: int = 0
    n_edges: int = 0
    n_support: int = 0
    n_attack: int = 0
    unanswered_attacks: int = 0


def _count_unanswered_attacks(store: GraphStore) -> int:
    """攻撃を受けたまま応答されていない発話 claim の総数。"""

    count = 0
    for node in store.nodes():
        if node.source != "utterance":
            continue
        incoming = [
            e for e in store.neighbors(node.id, direction="in") if e.relation == "attack"
        ]
        if not incoming:
            continue
        # この発話自身が他者を反撃しているか
        outgoing = [
            e for e in store.neighbors(node.id, direction="out") if e.relation == "attack"
        ]
        if not outgoing:
            count += len(incoming)
    return count


def programmatic_summary(
    store: GraphStore,
    transcript: list[Utterance],
) -> SessionSummary:
    """LLM を使わない決定的なサマリ。"""

    g = graph_metrics(store)
    t = transcript_metrics(transcript)
    unanswered = _count_unanswered_attacks(store)

    structural_lines = [
        f"発言ターン数: {t.n_turns}",
        f"主張・前提のノード数: {g.n_nodes} (発話 {g.n_utterance_nodes} / "
        f"文書 {g.n_document_nodes} / Web {g.n_web_nodes})",
        f"支持エッジ {g.n_support_edges} 件 / 反論エッジ {g.n_attack_edges} 件",
        f"未応答の反論: {unanswered} 件",
    ]

    if g.n_nodes == 0:
        text = "まだ議論が始まっていません。"
    else:
        sentences = [
            f"これまでに {t.n_turns} ターンの発話があり、{g.n_nodes} 個の主張・根拠が出ています。"
        ]
        if g.n_support_edges > 0 or g.n_attack_edges > 0:
            sentences.append(
                f"支持関係 {g.n_support_edges} 件、反論関係 {g.n_attack_edges} 件が観察されています。"
            )
        if unanswered > 0:
            sentences.append(f"{unanswered} 件の反論が未応答のまま残っています。")
        else:
            sentences.append("提示された反論はすべて応答されています。")
        if t.gini_speaker_balance > 0.4 and t.speaker_turn_counts:
            top_speaker = max(t.speaker_turn_counts.items(), key=lambda kv: kv[1])
            sentences.append(
                f"発言量に偏りがあり、{top_speaker[0]} さんの発話が多い状況です。"
            )
        text = "".join(sentences)

    return SessionSummary(
        text=text,
        structural_lines=structural_lines,
        n_nodes=g.n_nodes,
        n_edges=g.n_edges,
        n_support=g.n_support_edges,
        n_attack=g.n_attack_edges,
        unanswered_attacks=unanswered,
    )


async def llm_summary(
    store: GraphStore,
    transcript: list[Utterance],
    llm: OpenAIClient,
    *,
    max_recent_turns: int = 8,
) -> SessionSummary:
    """``programmatic_summary`` の事実を踏まえ、LLM で自然な日本語に整形する。"""

    base = programmatic_summary(store, transcript)
    recent = transcript[-max_recent_turns:] if transcript else []
    quotes = "\n".join(f"- {u.speaker}: {u.text}" for u in recent)
    stats = "\n".join(f"- {line}" for line in base.structural_lines)

    messages = [
        {
            "role": "system",
            "content": (
                "あなたは議論ファシリテータです。これまでの議論の現在地を、"
                "全員に共有する短い自然言語の要約 (3〜5 文) としてまとめてください。\n"
                "- 論点の対立構造と未応答の反論を中心に整理する\n"
                "- 誰が何を言ったかは細かく書かない (代わりに「賛成側」「反対側」のように)\n"
                "- 客観的・中立的な口調を保つ\n"
                "- 出力は要約本文のみ。前置き・後付けは不要"
            ),
        },
        {
            "role": "user",
            "content": (
                f"## 構造的な状況\n{stats}\n\n"
                f"## 直近の発言\n{quotes if quotes else '(発言なし)'}\n\n"
                f"## 出力\n3〜5 文の日本語要約:"
            ),
        },
    ]
    text = await llm.chat(
        messages,  # type: ignore[arg-type]
        model=llm.smart_model,
        temperature=0.3,
    )
    return SessionSummary(
        text=text.strip() or base.text,
        structural_lines=base.structural_lines,
        n_nodes=base.n_nodes,
        n_edges=base.n_edges,
        n_support=base.n_support,
        n_attack=base.n_attack,
        unanswered_attacks=base.unanswered_attacks,
    )


async def summarize_session(
    store: GraphStore,
    transcript: list[Utterance],
    *,
    llm: OpenAIClient | None = None,
) -> SessionSummary:
    """L3 サマリを返す。``llm`` が None なら programmatic にフォールバック。"""

    if llm is None:
        return programmatic_summary(store, transcript)
    return await llm_summary(store, transcript, llm)


__all__ = [
    "SessionSummary",
    "llm_summary",
    "programmatic_summary",
    "summarize_session",
]
