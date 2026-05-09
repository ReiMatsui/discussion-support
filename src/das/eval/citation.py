"""提示情報の引用率 (citation_rate) 計算 — RQ4 の直接 evidence。

「ファシリテータが提示した情報 (L1 アイテムの ``source_text``) が、
その後の発話で実際に参照されたか」を客観的・決定的に計算する。

Flat RAG と FullProposal の差は、**関係ラベル付き提示が外部知識の活用度を
高めるか** に現れるはず。これを source 別 (発話 / 文書 / Web) に分解して測る。

判定方法:
  - **n-gram coverage**: 提示テキストの文字 n-gram のうち、
    対象発話に出現する割合。日本語は単語境界がないので文字 n-gram が頑健。
  - 既定 ``n=4`` (短すぎ: 偶然一致が増える, 長すぎ: 言い換えに弱い)。
  - 既定しきい値 ``coverage_threshold=0.15`` (ゆるめ; 一文一致なら届く)。

LLM judge ノイズに左右されない、対面実験にも同じ計算で適用できる
(録音 transcript と提示 source_text を直接照合する)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from das.types import Utterance


_WS = re.compile(r"\s+")


def _ngrams(text: str, n: int) -> set[str]:
    """文字レベル n-gram の集合。空白は除去。"""

    cleaned = _WS.sub("", text)
    if len(cleaned) < n:
        return set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def _coverage(source: str, target: str, n: int) -> float:
    """``source`` の n-gram のうち ``target`` に現れる割合 (0-1)。

    源が短すぎて n-gram が取れない場合は 0 を返す。
    """

    sa = _ngrams(source, n)
    if not sa:
        return 0.0
    sb = _ngrams(target, n)
    return len(sa & sb) / len(sa)


def is_cited(
    source_text: str,
    utterance_text: str,
    *,
    n: int = 4,
    coverage_threshold: float = 0.15,
) -> bool:
    """``source_text`` が ``utterance_text`` に引用されているとみなせるか。"""

    return _coverage(source_text, utterance_text, n) >= coverage_threshold


@dataclass(frozen=True)
class CitationByKind:
    """source kind 別の集計。"""

    source_kind: str  # "utterance" / "document" / "web"
    n_items_presented: int = 0
    n_items_cited: int = 0

    @property
    def rate(self) -> float:
        if self.n_items_presented == 0:
            return 0.0
        return self.n_items_cited / self.n_items_presented


@dataclass(frozen=True)
class CitationStats:
    """1 ラン分の引用率集計。"""

    n_items_presented: int = 0
    n_items_cited: int = 0
    by_kind: dict[str, CitationByKind] = field(default_factory=dict)

    @property
    def overall_rate(self) -> float:
        if self.n_items_presented == 0:
            return 0.0
        return self.n_items_cited / self.n_items_presented

    def to_dict(self) -> dict:
        return {
            "n_items_presented": self.n_items_presented,
            "n_items_cited": self.n_items_cited,
            "overall_rate": self.overall_rate,
            "by_kind": {
                k: {
                    "n_items_presented": v.n_items_presented,
                    "n_items_cited": v.n_items_cited,
                    "rate": v.rate,
                }
                for k, v in self.by_kind.items()
            },
        }


def compute_citation_stats(
    transcript: list[Utterance],
    interventions: list[dict],
    *,
    n: int = 4,
    coverage_threshold: float = 0.15,
) -> CitationStats:
    """transcript と介入ログから ``CitationStats`` を計算する。

    interventions は ``InterventionLogEntry`` を ``asdict`` した形を想定:
    - ``turn_id``: 介入が発火したターン (= 介入直後のターンが「次発話」)
    - ``addressed_to``: L1 のときの宛先発話者名
    - ``items``: list of dict with ``source_text`` and ``source_kind``
    - ``kind``: "l1" / "l2" / "skip"

    ロジック:
      L1 のとき: 介入の turn_id を T、addressed_to を S とすると、
                  S が次に話すターン T' (T < T') の発話を target にする。
                  各 item の source_text を target に対して引用判定。
      L2 のとき: 介入後の **次の任意の発話**を target にする (全員向けなので)。
      skip のとき: スキップ。
    """

    # turn_id → (speaker, text) 高速参照
    by_turn: dict[int, Utterance] = {u.turn_id: u for u in transcript}

    # speaker → 発話の turn_id 順リスト
    speaker_turns: dict[str, list[int]] = {}
    for u in transcript:
        speaker_turns.setdefault(u.speaker, []).append(u.turn_id)

    n_items_total = 0
    n_cited_total = 0
    by_kind: dict[str, dict[str, int]] = {}

    def _bump(kind: str, *, presented: bool, cited: bool) -> None:
        d = by_kind.setdefault(kind, {"presented": 0, "cited": 0})
        if presented:
            d["presented"] += 1
        if cited:
            d["cited"] += 1

    for entry in interventions:
        kind = entry.get("kind")
        if kind not in ("l1", "l2"):
            continue
        items = entry.get("items") or []
        if not items:
            continue
        trigger_turn = int(entry.get("turn_id", 0))

        # target 発話の決定
        target_text: str | None = None
        if kind == "l1":
            addressed = entry.get("addressed_to")
            if addressed and addressed in speaker_turns:
                future = [t for t in speaker_turns[addressed] if t > trigger_turn]
                if future:
                    target_text = by_turn[future[0]].text
        else:  # l2
            future = [t for t in by_turn if t > trigger_turn]
            if future:
                target_text = by_turn[min(future)].text

        if not target_text:
            continue

        for it in items:
            src_text = it.get("source_text", "")
            src_kind = it.get("source_kind", "unknown")
            if not src_text:
                continue
            n_items_total += 1
            cited = is_cited(
                src_text,
                target_text,
                n=n,
                coverage_threshold=coverage_threshold,
            )
            if cited:
                n_cited_total += 1
            _bump(src_kind, presented=True, cited=cited)

    by_kind_objs = {
        k: CitationByKind(
            source_kind=k,
            n_items_presented=v["presented"],
            n_items_cited=v["cited"],
        )
        for k, v in by_kind.items()
    }
    return CitationStats(
        n_items_presented=n_items_total,
        n_items_cited=n_cited_total,
        by_kind=by_kind_objs,
    )


def aggregate_citation_stats(stats_list: list[CitationStats]) -> dict:
    """複数ラン分の citation 集計を平均する (集計表示用)。"""

    if not stats_list:
        return {}
    n = len(stats_list)
    total_presented = sum(s.n_items_presented for s in stats_list)
    total_cited = sum(s.n_items_cited for s in stats_list)

    by_kind_agg: dict[str, dict[str, int]] = {}
    for s in stats_list:
        for k, v in s.by_kind.items():
            d = by_kind_agg.setdefault(k, {"presented": 0, "cited": 0})
            d["presented"] += v.n_items_presented
            d["cited"] += v.n_items_cited

    return {
        "n_runs": n,
        "n_items_presented_total": total_presented,
        "n_items_cited_total": total_cited,
        "overall_rate": (total_cited / total_presented) if total_presented else 0.0,
        "by_kind": {
            k: {
                "n_items_presented": v["presented"],
                "n_items_cited": v["cited"],
                "rate": (v["cited"] / v["presented"]) if v["presented"] else 0.0,
            }
            for k, v in by_kind_agg.items()
        },
    }


__all__ = [
    "CitationByKind",
    "CitationStats",
    "aggregate_citation_stats",
    "compute_citation_stats",
    "is_cited",
]
