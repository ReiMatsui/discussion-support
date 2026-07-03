"""抽出 (G2) の目視比較スクリプト (雑でよい)。

文脈付き抽出 (指示語解決 + 発話内エッジ) の効果を確認するため、サンプル発話列を
「文脈なし」「文脈あり」の両方で抽出して並べて表示する。実 LLM (.env の
OPENAI_API_KEY) を使う。

使い方:
    python scripts/compare_extraction.py               # 内蔵サンプルで比較
    python scripts/compare_extraction.py <transcript.jsonl> [n]

旧実装との比較ではない (G2 で extraction を置換したため)。代わりに、参照文脈の
有無で claim テキストの自己完結度がどう変わるか / intra_edges が張られるかを見る。
"""

import asyncio
import json
import sys

from das.agents.extraction import ExtractionAgent
from das.llm import OpenAIClient
from das.types import Utterance

# 指示語・省略が効く内蔵サンプル (5 発話)
_SAMPLE = [
    ("B", "紙容器はコストが 3 倍かかるので、導入は慎重にすべきです。"),
    ("A", "それはちょっと違うと思います。長期的にはむしろ安くなります。"),
    ("C", "さっきの話ですけど、リユース容器なら両立できるんじゃないですか。"),
    ("B", "でも回収の手間が増えますよね。"),
    ("A", "その手間はアプリで管理すれば減らせます。"),
]


def _load(path: str, n: int) -> list[Utterance]:
    utts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            utts.append(Utterance(turn_id=d["turn_id"], speaker=d["speaker"], text=d["text"]))
            if len(utts) >= n:
                break
    return utts


def _fmt(result) -> str:
    lines = [f"    - [{n.node_type}] {n.text}" for n in result.nodes]
    for e in result.edges:
        lines.append(f"    - edge: unit→unit {e.relation} (created_by={e.created_by})")
    return "\n".join(lines) or "    (なし)"


async def main() -> None:
    if len(sys.argv) > 1:
        utts = _load(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    else:
        utts = [Utterance(turn_id=i + 1, speaker=s, text=t) for i, (s, t) in enumerate(_SAMPLE)]

    agent = ExtractionAgent(llm=OpenAIClient())
    for i, u in enumerate(utts):
        context = utts[:i]
        print(f"\n=== turn {u.turn_id} [{u.speaker}] {u.text}")
        no_ctx = await agent.extract(u, context=None)
        with_ctx = await agent.extract(u, context=context)
        print("  [文脈なし]")
        print(_fmt(no_ctx))
        print("  [文脈あり]")
        print(_fmt(with_ctx))


if __name__ == "__main__":
    asyncio.run(main())
