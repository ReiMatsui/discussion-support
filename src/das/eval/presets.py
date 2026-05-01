"""議論シミュレーション用の固定 Persona プリセット。

研究プロトタイプの再現性を確保するため、トピックごとに 3〜5 名の
代表的な persona を事前定義する。Parameterized 版 (``build_persona``) と
組み合わせて使うと、ベースライン + 揺らぎという構成にもできる。
"""

from __future__ import annotations

from das.eval.persona import PersonaSpec, build_persona


def cafeteria_personas() -> list[PersonaSpec]:
    """カフェテリアのプラ容器廃止議論用 (発表資料 p7 と整合)。"""

    return [
        build_persona(
            name="A",
            stance="pro",
            focus="環境負荷とサステナビリティ",
            personality="原則重視で、長期的な視点を強調する",
            extra="プラスチック廃棄の影響に強い問題意識を持つ",
        ),
        build_persona(
            name="B",
            stance="con",
            focus="学食価格とコスト負担",
            personality="現実主義で、運用面の懸念を強く示す",
            extra="学生の経済負担を最優先で考える",
        ),
        build_persona(
            name="C",
            stance="neutral",
            focus="折衷案や代替策",
            personality="調整役で、両論を整理しつつ第 3 の選択肢を探す",
            extra="バイオプラやリユース容器など、選択肢の多さに関心",
        ),
    ]


def policy_ai_lecture_personas() -> list[PersonaSpec]:
    """政策論題: 「生成 AI の大学講義利用を許容すべきか」用。"""

    return [
        build_persona(
            name="教員 X",
            stance="con",
            focus="学習プロセスの空洞化と評価の公正性",
            personality="慎重で、伝統的な学習価値を重視する",
            extra="AI が生成した文章を学生のものと区別できないことに不安を感じている",
        ),
        build_persona(
            name="学生 Y",
            stance="pro",
            focus="学習効率と将来の職業適応",
            personality="実務志向で、新技術にオープン",
            extra="AI を使えないと社会に出てから不利になると考えている",
        ),
        build_persona(
            name="教育工学者 Z",
            stance="neutral",
            focus="設計と運用ルールのバランス",
            personality="データ志向で、ルール設計の効果検証を重視する",
            extra="AI 利用と評価方法を組み合わせた研究知見を引用しがち",
        ),
        build_persona(
            name="保護者 W",
            stance="con",
            focus="高い学費に見合う教育の質",
            personality="率直で、コストパフォーマンスの観点で発言する",
            extra="AI に頼った学習で本当に力が付くのか疑問視している",
        ),
    ]


__all__ = [
    "cafeteria_personas",
    "policy_ai_lecture_personas",
]
