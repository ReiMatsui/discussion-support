# 実行計画: 関連研究と比較可能な状態に揃える

最終更新: 2026-05-13
担当: Rei

研究方針の確認 (2026-05-13 合意):
- **engine は固定**: AF 構築 (発話 + 文書 + Web をすべて claim/premise 化)、LinkingAgent、FacilitationAgent はそのまま
- **動かすのは「情報の使い方」だけ**: LLM ペルソナへの渡し方、対面会議の人間への見せ方
- **まず「比較できる状態」を作る**: 関連研究と同じ指標で同じ実験を回せるところまで持っていく
- **その後で presentation の variation を系統的に動かす** (= 論文の主結果)

---

## 1. 全体像: Phase 1 (5 週間) → Phase 2 (variation 実験)

```
Phase 1 (Week 1-5): 比較可能な状態に揃える
  ├─ Week 1   段階 A — ARIES で Linking macro-F1
  ├─ Week 1-2 AQuA 20-indicator を LLM-judge で自前実装
  ├─ Week 2-3 政策トピック × n=10 で 3 条件再ラン
  ├─ Week 3-4 Social Laboratory 心理計量指標を judge に追加
  └─ Week 4-5 PTFA / Empty Chair を別トラックで実装

Phase 2 (Week 6+): presentation variation 実験
  └─ 出典タグ / 関係ラベル強度 / rationale / 件数 / タイミング /
     ソース層化 / 対面用 visualization を 1 つずつ動かす
```

---

## 2. 各週の詳細

### Week 1: 段階 A — Linking macro-F1

**目的**: citation_rate が低かったとき、リンク精度のせいか提示のせいかを切り分けられる「保険」を作る。

**やること**:
- ARIES (Gemechu et al. ArgMining 2024) の 1 データセット (推奨: Persuasive Essays または AAEC) を load
- `src/das/eval/component_eval/linking_benchmark.py` を新設
- LinkingAgent を ARIES の各ペアに適用し、5 値 (a_supports_b / a_attacks_b / b_supports_a / b_attacks_a / none) を 3 値 (support / attack / none) に縮約して macro-F1
- 結果を `data/component_eval/linking_aries_<timestamp>.json` に保存

**外部依存**: ARIES データ取得 (GitHub 公開済み、要確認)

**完了基準**: 「LinkingAgent の macro-F1 は ARIES で X」と論文に書ける 1 行が出る

### Week 1-2: AQuA 20-indicator (LLM-judge 自前実装)

**目的**: 外部の deliberation 評価枠組みで das の transcript を採点 → 既存 3 条件の比較に外部基準を入れる。

**やること**:
- Behrendt et al. (DELITE @ LREC-COLING 2024) の 20 deliberation indicator 一覧を抜き出し
- `src/das/eval/aqua.py` を新設
- 各 indicator を LLM-judge で 0-1 スコア化する prompt を設計
- 既存の `tier12-smoke` を AQuA で再採点

**判断**: 公開コード (mabehrendt/AQuA) はドイツ語 adapter なので、option (3) = AQuA-inspired として 20 軸だけ採用、das の OpenAI client で各 indicator を直接スコアする。論文には "AQuA-inspired 20 indicators" と書く。

**完了基準**: 既存 3 条件の AQuA スコア平均と分散が summary.json に並ぶ

### Week 2-3: 政策トピック × n=10 で 3 条件再ラン

**目的**: n=3 → n=10 にして統計的に意味のある比較にする。これが Phase 1 の本丸。

**やること**:
- `policy_ai_lecture_personas` × max_turns=22 × n=10 / 条件 × 3 条件 = 30 ラン
- 全指標を 1 度に出す:
  - 主観 5 軸 (JudgeAgent)
  - 構造 8 指標 (structural_metrics)
  - citation_rate (source 別)
  - cross_source_edge_rate
  - stance shift (pre/post × public/private)
  - AQuA 20-indicator
  - consensus 検出
- `data/eval/phase1-main/` に集約

**完了基準**: 「3 条件比較」の表 1 枚 + 図 2〜3 枚が論文用に出せる

**コスト見積もり**: 1 ラン ≈ $0.50 (gpt-5-mini, 22 turn × 4 persona × extraction/linking) と仮定して、30 ラン × $0.50 + judge/AQuA で **約 $30〜50**。要確認

### Week 3-4: Social Laboratory 心理計量指標を judge に追加

**目的**: Social Laboratory (arXiv 2510.01295) と直接比較可能にする。

**やること**:
- `judge.py` に Empathy Score と Cognitive Dissonance を追加
- 既存 transcript を再採点 (新しく走らせる必要なし)
- 政策トピック n=10 のデータに対しても採点

**完了基準**: Social Laboratory の 3 指標 (Empathy / Cognitive Dissonance / Stance Shift) で 3 条件比較が出る

### Week 4-5: PTFA / Empty Chair を別トラックで実装

**目的**: 「先行手法と提案手法」の section を埋める。**本筋 (presentation 比較) とは交絡するので別表に分ける**。

**やること**:
- `ConditionPTFA`: Six Thinking Hats の 6 ロール (White/Red/Black/Yellow/Green/Blue) を LLM persona として並列に短コメント、議論の脇に表示
- `ConditionEmptyChair`: 議論中盤に不在ペルソナ (例: 「AI 企業」「学習障害学生」) を 1 ターン挿入
- 同じ評価パイプラインで採点
- 結果を別の summary に保存

**完了基準**: 論文の「related work と直接比較」表が 1 枚

---

## 3. Phase 2: presentation variation 実験 (Week 6+)

engine 固定 (FullProposal) のまま、以下の次元を 1 つずつ動かす。

### 動かす次元

| 次元 | variation 例 | 期待される効果 |
|---|---|---|
| **出典タグ** | なし / `[文書: XX より]` / `[Web: domain.com より]` | citation_rate (document) が上がる |
| **関係ラベルの強度** | `[支持]` / `[支持/反論]` / `[Aさんを補強する事例]` のような自然文 | 介入透明性 + RQ2 反対理解 |
| **rationale 表示** | なし / 1 文 / 詳細 | 介入透明性 |
| **件数** | 1 / 2 / 3 | citation_rate と overwhelm のトレードオフ |
| **タイミング** | 毎ターン / 偏り検知時のみ / 停滞時のみ | 認知負荷 |
| **ソース層化** | 発話のみ / 文書のみ / 混合 (発話 1 + 文書 1) | cross_source_edge_rate |
| **対面用 visualization** | テキスト挿入 / グラフ可視化 / ハイブリッド | UEQ + NASA-TLX |

### 設計上の変更

`info_provider(history, persona) -> str | None` を `PresentationStrategy` 抽象に拡張:

```python
class PresentationStrategy(Protocol):
    name: str
    async def render(
        self,
        decision: InterventionDecision,
        persona: PersonaSpec,
        modality: Literal["llm_prompt", "human_visual"],
    ) -> str | dict | None: ...
```

これで「同じ FacilitationAgent の出力を、persona LLM 向けには文字列で、対面被験者向けには可視化用 dict で」返せる。

### Phase 2 の評価

Phase 1 で固めた評価パイプラインをそのまま使う。**評価方法を変えずに、variation を 1 つずつ走らせて差分を測る**。

---

## 4. 段階 C (対面実験) は Phase 2 と並行で設計

- Phase 2 で「効きそうな presentation」がデータから見えてくる
- それを 2〜3 種類に絞り込んで対面実験条件にする
- IRB 用 `docs/study_protocol.md` を Phase 2 期間中に並行作成
- 5-8 名 × 60 分 × 政策トピック、被験者内デザイン
- UEQ + NASA-TLX + 自由記述

---

## 5. 決定済みの方針

- AQuA は公開 adapter (ドイツ語) ではなく、**20 軸だけ採用して LLM-judge で自前採点** (option 3)
- PTFA / Empty Chair は **本筋とは別表**で扱う (engine が違うので交絡)
- 段階 A (Linking F1) は **Phase 1 最初の 1 週間**で必ず通す (citation_rate 解釈の保険)
- n は **政策トピック × n=5** で固める ($1 制約のため n=10 から縮小)
- **`--budget` (soft)** で実コストを enforce する: `das eval --budget 1.5` のように指定すると、累積 cost が 1.5 USD を超えた時点で **新規 run の開始だけが gate** される。**進行中の run は最後まで完走**するので部分結果が確実に残る (in-flight run の暴走防止に **`--hard-budget`** を併用するのが安全)
- **`--linking-top-k 3 --linking-model gpt-5-nano`** の節約策で per-fp-run コストを $1.05 → ~$0.30 に削減

---

## 6. 未決事項

- **OpenAI コスト**: Week 2-3 の n=10 × 3 条件で $30〜50 見込み。承認要
- **ARIES データ**: GitHub から取得可能か確認、ライセンス確認
- **対面実験の予算**: 段階 C 開始までに研究室予算と相談
