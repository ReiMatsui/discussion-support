"""A vs B 検証スクリプト（浪江町 F-REI コーパス）.

目的:
  A（類似度 top-k 取得 → 取得後に支持/攻撃を判定）だけで「攻撃すべき事実」が
  候補 top-k に入るのか、それとも取得段で埋もれて取りこぼすのかを、
  リポジトリの実埋め込み（das.llm.OpenAIClient.embed）で定量化する。

  併せて、事実どうしの類似度を出し、「事前に張れる事実間の矛盾エッジ」が
  この浪江町コーパスに存在しそうか（=純粋な B が発火するか）も見る。

実行:
  uv run python scripts/validate_a_vs_b.py
  # OPENAI_API_KEY が必要（embed のみ。chat は呼ばない）

判断の目安:
  - 「攻撃すべき事実」の類似度順位が top_k 以内 → A の取得で拾える（Aで足りる）
  - top_k 圏外 → A は取りこぼす。主張相対の取得拡張（リコール強化）が要る
"""

from __future__ import annotations

import asyncio
import itertools
import math

from das.llm import OpenAIClient

TOP_K = 3  # 既定の linking top_k 相当。必要なら変える

# 浪江町 F-REI コーパス（docs/af_example_namie_frei.md より）
CLAIMS = {
    "C1": "浪江町は復興資源を F-REI を核とした先端研究・産業の集積に集中させるべきだ",
    "C2": "まず帰還住民の生活インフラ（医療・買い物・交通）の再建を優先すべきだ",
    "C3": "F-REI の雇用が地元の人口回復に本当につながるのかは疑わしい",
}
EVIDENCE = {
    "E1": "F-REI は2023年に浪江町へ設立。7年間で約1000億円、2029年度までに研究者・職員600人体制を目指す",
    "E2": "2025年の住民意向調査で「戻らないと決めている」51.0%、「戻りたい」10.1%、「判断がつかない」24.7%",
    "E3": "棚塩産業団地（約49ha）に FH2R（世界最大級の水素製造拠点）と福島ロボットテストフィールドが立地・稼働",
    "E4": "町内居住者は約2,400人（震災時人口 約21,500、住民登録 約14,000、帰還率 約17%）",
    "P1": "すでに棚塩産業団地に水素・ロボットの研究拠点が集積している",
}
# 資料が想定する「正解の関係」（対象主張ごと）
INTENDED = {
    "C1": {"support": ["E1", "E3", "P1"], "attack": ["E2"]},
    "C2": {"support": ["E2", "E4"], "attack": []},
    "C3": {"support": [], "attack": ["E1"]},
}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def main() -> None:
    llm = OpenAIClient()
    keys = list(CLAIMS) + list(EVIDENCE)
    texts = list(CLAIMS.values()) + list(EVIDENCE.values())
    vecs = await llm.embed(texts)
    emb = dict(zip(keys, vecs))

    print(f"=== A の取得検証（top_k={TOP_K}）: 攻撃すべき事実は候補に入るか ===")
    miss = 0
    for c, rels in INTENDED.items():
        ranked = sorted(EVIDENCE, key=lambda e: cosine(emb[c], emb[e]), reverse=True)
        scores = {e: round(cosine(emb[c], emb[e]), 3) for e in ranked}
        print(f"\n{c}: {CLAIMS[c][:30]}…")
        print("  類似度順:", [f"{e}={scores[e]}" for e in ranked])
        for a in rels["attack"]:
            rank = ranked.index(a) + 1
            ok = "OK(取得される)" if rank <= TOP_K else "★取りこぼし(Aの穴)"
            if rank > TOP_K:
                miss += 1
            print(f"  攻撃すべき事実 {a}: 類似度 {rank} 位 / {len(EVIDENCE)} → {ok}")
    print(f"\n→ top_k={TOP_K} で取りこぼす攻撃関係: {miss} 件")

    print("\n=== B の検証: 事実どうしに『事前に張れる矛盾』はあるか ===")
    print("(事実ペアの類似度。高くても“同時に真”なら矛盾ではない点に注意)")
    for a, b in itertools.combinations(EVIDENCE, 2):
        print(f"  {a}-{b}: cos={round(cosine(emb[a], emb[b]), 3)}")
    print(
        "\n※ 浪江町では E1〜E4 は同時に真で事実間の矛盾はほぼ無い。"
        "\n  純粋な B（事前の事実間矛盾エッジ）は発火しにくい→ 効く B は主張相対の取得拡張。"
    )


if __name__ == "__main__":
    asyncio.run(main())
