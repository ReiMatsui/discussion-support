# マルチエージェントによる議論グラフ統合型 議論支援

> Argumentation-Graph–Integrated Multi-Agent Discussion Support  
> 松井 玲 / 伊藤研究室

---

## 1. 研究目的と問い

### ゴール

**対面会議 (face-to-face) のリアルタイム議論を、議論グラフ (Argumentation Framework, AF) を構築・提示することで支援する**システムを提案する。LLM エージェント同士のシミュレーションは、人間を集めずに仕組みの動作と効果を検証するための **proxy 環境**であり、研究の主役ではない。

### 提案手法の核 — 3 本の貢献

提案手法は次の 3 点で他のアプローチと差別化される:

**① 議論ログと外部知識を「同じ AF」に統合する**

発話・事前文書・Web 検索結果のいずれも論証単位 (claim / premise) のノードとして同一グラフに置き、それらの間に支持/攻撃エッジを張る。「議論」と「外部知識」が論証的に連続する。

**② 関係ラベル (支持/攻撃) を持って情報を提示する**

参考情報を「類似文章」ではなく、**「あなたの主張への反論」「Bさんの立場を補強する事例」**のように、関係を明示して提示する。これにより参加者は提示情報の意味を即座に理解できる。

**③ ファシリテータが「いつ・誰に・何を」を AF 状態だけで判断する**

`decide_intervention(history, store)` は次話者 / 話者順を引数に取らず、グラフ状態だけで意思決定する。テキスト・音声・対面どのモダリティでも同じ判断ロジックが使える。

### リサーチクエスチョン

- **RQ1 議論の質への影響**: 統合 AF を活用する議論支援は、フラット RAG / 支援なしと比べて議論の質を改善するか
- **RQ2 反対意見の理解**: 支持・攻撃を構造的にバランス提示することで、参加者の反対意見理解と過剰自信は変化するか
- **RQ3 介入の透明性**: 介入根拠が論証構造として明示されることで、納得感は変化するか
- **RQ4 外部知識の活用 (新規)**: 関係ラベル付き提示 (vs 類似度のみ) は、外部知識 (文書・Web) の参加者による活用度を高めるか — これが貢献①②を直接測る軸

### 解決したい既存議論支援の課題

| | 課題 | 対応 |
|---|---|---|
| C1 | RAG が **発話に同調**する情報を返す (確証バイアス助長) | 関係 (支持/攻撃) を明示した提示 |
| C2 | 議論ログと外部知識の **論証的不接続** | 統合 AF で両者を支持/攻撃エッジ連結 |
| C3 | 介入根拠の **不透明性** | 介入ログの完全 trace + reason 付き |
| C4 | 事前資料の網羅性の限界 | Web 検索エージェント (M3 完了) でリアルタイム検索 → AF ノード化 |

### 検証戦略 (2 段)

```
Phase 1 (現在): LLM シミュレーションで仕組みの動作と効果を確認
  - 3 条件 (None / FlatRAG / FullProposal) で議論を回す
  - 評価エージェントで主観評価、AF/transcript から客観指標
  - **指標は人間でも同じ意味で測れるものを選択** (citation_rate, stance shift,
    response_rate 等) — 対面実験への transferability を確保

Phase 2 (将来): 対面参加者実験で人間に対する効果を直接測る
  - 5-8 名 × 60 分 × 政策論題、被験者内デザイン
  - 主観: 議論満足度・納得感・介入の透明性 (アンケート)
  - 客観: 同一の AF 構造指標 + UEQ + NASA-TLX
```

LLM シミュレーションは proxy であり、論文の主軸ではない。**システムは最初からリアルタイム動作前提で設計されている** (SessionRunner は round-robin だが、Facilitator 本体は次話者を仮定しない)。

---

## 2. アプローチの全体像

```
                ┌──────────────┐
                │  発話 (text/  │
                │  音声/対面)   │
                └──────┬───────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  論証抽出エージェント         │  ← LLM
        │  (Utterance → claim/premise) │
        └──────────────┬───────────────┘
                       │ NodeAdded
                       ▼
        ┌──────────────────────────────┐       ┌─────────────────────┐
        │  連結エージェント             │ ←──→ │ 文書知識エージェント │
        │  (cosine top-k → LLM 5値判定)│       │ (事前文書を AF 化)   │
        └──────────────┬───────────────┘       └─────────────────────┘
                       │ Edge 追加
                       ▼
        ┌────────────────────────────────────────────┐
        │     統合議論グラフ (AF, NetworkX + SQLite)  │
        │     - utterance / document / web ノード     │
        │     - support / attack エッジ              │
        └──────────────┬─────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  ファシリテーション エージェント  │
        │  - 偏り検知 (BiasReport)         │
        │  - ステージ検知 (graph-based)    │
        │  - decide_intervention →         │
        │    {SKIP / L1 / L2}              │
        └──────────────┬───────────────────┘
                       │
                       ▼
              情報提示 (4 層設計)
              - L1: 個別通知 (1〜2 件)
              - L2: 議論の俯瞰整理 (LLM 整文)
              - L3: 事後の自然言語要約
              - L4: 個別振り返り
```

### 5 つの専門エージェント

| エージェント | 責務 | 実装 |
|---|---|---|
| 論証抽出 (Extraction) | 発話を claim/premise に分解 | `agents/extraction.py` |
| 文書知識 (Document) | 事前資料を AF 化 | `agents/document.py` |
| Web 検索 | リアルタイム検索 (M3 予定) | 未実装 |
| 連結 (Linking) | embedding top-k → 5 値関係判定 (a_supports_b / a_attacks_b / b_supports_a / b_attacks_a / none) | `agents/linking.py` |
| ファシリテーション | グラフ全体を読み「**いつ・誰に・何を**」提示するか中央調停 | `agents/facilitation.py` |

### 評価エージェント (シミュレーション専用)

| エージェント | 役割 |
|---|---|
| Persona | 立場 (pro/con/neutral) を持つ参加者を演じる | `eval/persona.py` |
| Judge | 議論終了後にペルソナ視点で 5 軸主観評価 | `eval/judge.py` |
| Consensus | LLM-judge による合意検出 (stance + agreement) | `agents/consensus_agent.py` |

---

## 3. ファシリテーションの「**いつ介入するか**」設計

毎ターン無条件に情報を出すのではなく、グラフ状態を見て介入の要否を判断する 2 段構成:

```
decide_intervention(transcript, store) → InterventionDecision
                                          ├─ SKIP: 介入しない
                                          ├─ L1:   個別通知 (1-2 件)
                                          └─ L2:   俯瞰整理 (全員共有)
```

### トリガー条件

- **SKIP**: 直前介入後に新エッジ追加なし / 最新発話の関連エッジが空
- **L1**: 最新発話の隣接エッジがあるとき (デフォルト経路、最大 2 件)
- **L2**:
  - 直近窓で新 claim/attack が両方ゼロ (= stalled)
  - または `imbalance_ratio > 0.4` かつ weak_claims が存在
  - 連続 L2 抑制のため `l2_min_interval=5` ターン

### モダリティ非依存性

研究計画書の最終目標 (対面・音声・テキスト全モダリティ対応) を見据え、Facilitator の判断は AF 状態だけに依存させる:

- ✅ 「次話者」を引数に取らない (round-robin 仮定なし)
- ✅ 出力 `InterventionDecision` に `addressed_to` を持たせ、配信チャネル非依存
- ✅ stalled 検知はテキスト重複ではなく **AF 追加レート**で判定
- ✅ シミュレーション固有の整形は `ConditionFullProposal.info_provider` のアダプタ層に局所化

---

## 4. 評価設計 (3 条件比較)

研究計画書 §5.2 段階 B (シミュレーション評価)。

### 条件

| 条件 | 内容 |
|---|---|
| **None** | 情報提供なし (ベースライン) |
| **FlatRAG** | 段落単位 embedding top-k で類似度の高い文書チャンクをそのまま提示 (関係ラベルなし) |
| **FullProposal** | 提案手法。AF 構築 + ファシリテータが support/attack を関係ラベル付きで提示 |

### 主観指標 (Judge: LLM がペルソナをロールプレイ)

研究計画書 §5.1 から抽出した 5 軸:

| 指標 | 範囲 |
|---|---|
| 議論満足度 (overall_satisfaction) | 1-7 |
| 情報の有用性 (information_usefulness) | 1-7 |
| 反対意見の理解度 (opposition_understanding) | 1-7 |
| 立場の自信度変化 (confidence_change) | -3..+3 |
| 介入の透明性 (intervention_transparency) | 1-7 |

### 客観指標 (Stage 1: AF + transcript から決定的計算)

DQI (Steenbergen et al., 2003) と The Social Laboratory (2025) の流れを踏まえた **LLM 不要** の構造指標:

| 指標 | 意味 | DQI 対応 |
|---|---|---|
| participation_gini | 話者間の発話偏在 (0-1) | 平等性 |
| avg_premises_per_claim | claim あたりの根拠ノード数 | 正当化レベル |
| pct_unsupported_claims | 根拠 0 の claim 率 | 正当化レベル |
| response_rate | 過去発話に応答した発話の割合 | 尊重・応答性 |
| pct_attacks_answered | 反論を受けた話者が再反論した率 | 尊重・応答性 |
| avg_argument_chain_length | 支持/攻撃チェーンの平均深さ | 議論深度 |
| n_isolated_claims | 孤立 claim 数 | 議論密度 |

### RQ4 を測る指標 (貢献①②を直接検証)

「議論ログと外部知識の同一 AF 統合」と「関係ラベル付き提示」が実際に効果を持つかを測る:

| 指標 | 計算 | 何を示すか |
|---|---|---|
| **citation_rate (source 別)** | 提示された源 (utterance/document/web) 別に、source_text が次発話に出現した率 | 提示情報が実際に議論で使われたか (RQ4 主要 evidence) |
| **異種ソース間エッジ密度** | 全エッジのうち発話↔文書 / 発話↔Web のエッジが占める割合 | 議論ログと外部知識の論証的接続が成立しているか (貢献①) |
| **未反映外部知識率** | 文書/Web ノードのうち、終了時にどのエッジも持たないものの割合 | 外部知識が議論に統合されたか |
| **stance-aligned citation** | 反対派が「反論」ラベルで提示された情報を引用した率 | 立場と異なる視点に触れた率 (RQ2 とも関連) |

これらは **AF + transcript + 介入ログ** から決定的に計算できるので LLM judge ノイズに左右されない。

### 評価指標の対面実験への transferability

LLM シミュレーション → 対面実験への移行で同じ意味で測れる必要がある。各指標の取り方を 2 列で:

| 指標 | LLM シミュレーション | 対面実験 |
|---|---|---|
| citation_rate | 提示 source_text と次発話の n-gram 一致 | 録音 transcript と提示文の照合 |
| stance shift | 評価エージェントによる per-persona Likert | 参加者へのアンケート (議論前後) |
| response_rate | AF エッジから計算 | 同じ計算 (録音 → AF) |
| 介入透明性 | 評価エージェントによる Likert | 参加者へのアンケート |
| 異種ソース間エッジ密度 | snapshot 集計 | 同じ計算 |

これにより Phase 2 の対面実験に実装を変えずそのまま適用できる。

### 合意検出 (Consensus)

文献:
- 表面キーワードでは LLM の **譲歩前置き「確かに〜が、」** を誤判定する
- 標準は **stance + agreement detection** (Sirota et al., SIGDIAL 2025)

実装は **2 段ハイブリッド**:

1. **Stage 1 (cheap)**: AF の構造シグナル (新 claim / attack 停止) を前段トリガーとして使用
2. **Stage 2 (expensive)**: 構造シグナルが立ったとき **だけ** ConsensusAgent (LLM) を起動し、各ペルソナの立場 (pro / con / partial_pro / partial_con / neutral) と全体合意を structured output で取得
3. 全員一致 + confidence ≥ 0.7 のときのみ合意成立

これにより:
- LLM 呼び出しは議論全体で数回程度に抑制
- 判定根拠 (rationale, stances) が透明
- キーワードに依存しない文脈判断

### 実装規模

| カテゴリ | ファイル数 | テスト |
|---|---|---|
| 中核 (graph / agents / runtime) | ~12 | 100+ |
| 評価 (eval) | ~9 | 60+ |
| UI (Streamlit) | 3 ページ | 手動 |
| 合計テスト | — | **176 passed** |

---

## 5. 現状の進捗

### 完了したマイルストーン

- ✅ **M1**: AF スキーマ / Orchestrator / Linking / 可視化 / E2E スモーク
- ✅ **M2.1-M2.3**: Persona / Controller / 3 条件 (None / FlatRAG / FullProposal)
- ✅ **M2.5**: LLM-as-judge 主観指標
- ✅ **M2.6**: 多数回ラン executor (`das eval`) + 並列実行
- ✅ **M2.7**: 評価ダッシュボード (議論レビュー / 集計 / 実行)
- ✅ FacilitationAgent: skip/L1/L2 介入判断 + モダリティ非依存
- ✅ Persona prompt の議論進行強化 (4 つの応答パターン必須化)
- ✅ ConsensusAgent: LLM-judge による合意検出
- ✅ 合意検出の誤検出修正 (逆接フィルタ + 二段ハイブリッド)
- ✅ **M2.4**: AF 由来の客観構造指標 (Stage 1)
- ✅ GPT-5 mini 対応 (temperature 互換性レイヤ)

### 残タスク

完了:
- ✅ **M2.0** 政策トピック (生成 AI 講義許容問題) のフィクスチャ作成
- ✅ **M3** Web 検索エージェント

進行中 / 次:
- ✅ **Tier 1 (RQ4 を直接測る指標群) — 完了**
  - citation_rate (source 別: utterance/document/web) — `src/das/eval/citation.py`
  - per-metric judge rationale (各スコアごとに reason) + UI 改修
  - 異種ソース間エッジ密度 (貢献①の直接証拠) — `cross_source_edge_rate`
- ⏳ Tier 2 (Planted Contradiction シナリオ + stance polling)
- ⏳ 段階 A 技術検証 (手動アノテーションで extraction / linking の F1 測定)
- ⏳ Stage 2 (DQI-inspired 主観評価)
- ⏳ 段階 C: 対面参加者実験 (5-8 名 × 60 分)

---

## 6. 既知の知見と限界

### 予備実験 `eval-20260501T082055Z` から得られた示唆 (n=2 × 3 条件)

| 指標 | none | flat_rag | full_proposal |
|---|---|---|---|
| 満足度 | 5.33 | 5.33 | **5.00** |
| 情報有用性 | 5.50 | 5.50 | 5.17 |
| 反対理解 | 6.17 | 6.33 | 5.83 |
| **介入透明性** | 4.00 | 3.33 | **4.67** ✓ |
| 介入数 (full_proposal のみ) | — | — | 16 件 / 11 ターン |

**仮説に沿った結果**: 介入透明性は提案手法が最高 (RQ3 を支持する方向)。

**仮説に反する結果**:
- 満足度・反対理解は提案手法が最低。**B (反対派) が full_proposal で 3-4 と低下**
- support エッジ 73 vs attack 26 と support 偏重
- Facilitator の `balance_correction` / `stage_alignment` が 1 度も発火していない (全介入が `adjacent`)

これにより本実装で **(1) 介入頻度の制御 (skip 機能)** と **(2) 偏り検知のしきい値再調整** を行った。

### 限界

1. **n が小さい**: 統計的結論には n=10 以上が必要
2. **ペルソナが LLM**: 実人間とは応答ダイナミクスが異なる (社会的圧力での合意傾向が出やすい)
3. **トピック 1 つのみ**: cafeteria のみで検証中。政策 AI トピックは未実装
4. **Web 検索未実装**: 動的情報の影響は未評価
5. **対面実験未実施**: 段階 C は今後の課題

---

## 7. 関連研究

### 議論支援システム

- **PTFA** (Gu et al., PRICAI 2025): Six Thinking Hats を 6 役割の LLM が並行して担い、合意形成を自動ファシリテート。
- **The Empty Chair** (Fulay & Roy, MIT 2025): 対面市民集会で不在ステークホルダーを LLM ペルソナで補完。

### Multi-agent debate / consensus

- **Du et al. (2023)**: Multi-agent debate で factuality と reasoning を改善。多数決で集約。
- **ReConcile** (Chen et al., ACL 2024): Round-table 形式で信頼度重み付き投票。
- **Voting or Consensus?** (Kaesberg et al., ACL 2025): 投票プロトコルが推論で +13.2%, 合意プロトコルが知識で +2.8%。
- **A-HMAD** (Springer 2025): Adaptive consensus optimizer で +4-6% 精度。
- **Adaptive Stability Detection** (arXiv 2510.12697): Beta-Binomial + KS 統計で議論を自動停止。

### 合意 / agreement detection

- **Finding Common Ground** (Sirota et al., SIGDIAL 2025): LLM による zero-shot agreement detection (stance + polarity)。**我々の ConsensusAgent はこの路線**。

### 議論質評価

- **Discourse Quality Index (DQI)** (Steenbergen et al., 2003 → Oxford 2024): 政治学の標準。5 次元 (平等性 / 正当化 / 内容 / 尊重 / 建設性)。
- **AQuA** (Behrendt et al., LREC-COLING 2024): DQI 派生の 20 指標を LLM で自動採点 + 専門家/非専門家評価で重み付け。
- **The Social Laboratory** (arXiv 2510.01295, 2025): Cognitive Dissonance / Empathy / Stance Shift など心理計量指標。
- **ChatEval** (Chan et al., ICLR 2024): 複数 judge agent の合議で評価品質向上。
- **ARS Framework** (Hinton & Wagemans): 引数単位の Acceptability / Relevance / Sufficiency。
- **Mirzakhmedova et al. (2024)**: LLM の引数質アノテーション信頼性。κ ≈ 0.71 (appropriateness, global relevance)。

### 議論グラフ (AF) と Argumentation Frameworks

- **Dung (1995)**: Abstract Argumentation Framework の古典。
- **Bench-Capon (2003)**: Value-based AAF。
- **MDPI Systems 2025**: AAF と DQI の統合的応用 (我々の方向と類似)。
- **ARGORA** (arXiv 2601.21533): 引数グラフを **因果モデル**として扱い、ablation で load-bearing な引数を特定。
- **ArgMed-Agents** (arXiv 2403.06294): 臨床判断における argumentation scheme + symbolic solver。
- **Can LLMs Judge Debates?** (EMNLP 2025 Findings): QuAD semantics による非線形引数推論評価。LLM は線形推論に強く非線形 AF に弱いと報告。

### AI-mediated public deliberation

- **Polis** (Taiwan, vTaiwan): 投票クラスタリング型の合意検出。**外部知識統合と関係ラベルは持たない**。
- **OECD 2025 レポート**: AI 公共参加ツールの体系的レビュー。
- **Knight First Amendment Institute**: AI mediation の効果と倫理。

### Argument visualization の有効性 (HCI / 教育)

- **Argument mapping のメタ分析**: Critical thinking の効果量 ~0.70 SD (通常授業の約 2 倍)。
- **Khartabil et al. (CGF 2021)**: AG vs テキスト表示の UEQ + NASA-TLX 比較。**未習熟トピックで AG が好まれる**。
- **Frontiers Education (2025)**: argument visualization のレビュー。

### 本研究の位置取り

| 観点 | 発話のみ debate (Du, ReConcile…) | Polis | Flat RAG | ARGORA / ArgMed | **本研究** |
|---|---|---|---|---|---|
| 議論ログを AF 化 | × | × | × | ◯ (内部のみ) | ◯ |
| 外部知識を AF 化 | × | × | △ (類似のみ) | △ (限定) | ◯ |
| 議論↔知識を **同一 AF に統合** | × | × | × | × | **◯** |
| 関係ラベル (支持/攻撃) | △ | × | × | ◯ | ◯ |
| **リアルタイム介入** | × | △ | △ | × | **◯** |
| **対面会議展開可能** | × | △ | × | × | **◯ (設計上)** |

本研究の独自性は「**議論ログと外部知識を同じ AF に統合し、関係ラベル付きで提示し、リアルタイムに介入する**」の 4 要素を全て満たす点。

---

## 8. プロジェクト構成

```
src/das/
├── agents/                # 5 専門エージェント
│   ├── extraction.py      # 論証抽出
│   ├── document.py        # 文書知識
│   ├── linking.py         # 連結 (embedding + LLM)
│   ├── facilitation.py    # ファシリテーション (中央調停)
│   └── consensus_agent.py # LLM-judge 合意検出
├── graph/                 # AF スキーマ + ストア
│   ├── schema.py          # Node / Edge (pydantic)
│   └── store.py           # NetworkX + SQLite append-only
├── eval/                  # シミュレーション評価
│   ├── persona.py         # PersonaSpec / PersonaAgent
│   ├── controller.py      # SessionRunner
│   ├── conditions.py      # None / FlatRAG / FullProposal
│   ├── consensus.py       # detect_consensus + LLM 統合
│   ├── judge.py           # 主観 5 軸評価
│   ├── structural_metrics.py  # AF 由来 客観指標 (DQI 風)
│   └── run_eval.py        # 多数回ラン executor
├── runtime/               # EventBus + Orchestrator
├── presentation/          # L3/L4 (要約・振り返り)
├── ui/                    # Streamlit
│   ├── streamlit_app.py   # 議論レビュー (メイン)
│   └── pages/
│       ├── 1_Aggregates.py     # 集計比較
│       └── 2_実行.py            # eval ライブ実行
├── llm/                   # OpenAI ラッパ (リトライ + 構造化出力)
└── cli.py                 # Typer CLI

data/
├── docs/                  # 事前文書 (cafeteria トピック)
└── eval/                  # eval 結果 (gitignored)
```

---

## 9. 使用方法 (再現用)

```bash
# 環境構築
uv sync --all-extras
cp .env.example .env  # OPENAI_API_KEY を設定

# 評価実行 (CLI)
uv run das eval cafeteria \
  -n 5 -t 25 --until-consensus -j 5 \
  --llm-consensus

# UI で結果確認
uv run das ui
# → http://localhost:8501
#    - 議論レビュー (メイン): タブで条件切替、タイムラインに介入インライン
#    - 集計比較: 主観 5 軸 + 構造指標 + 収束率
#    - 実行: フォームから eval を起動 (ライブ議論ビュー)
```

---

## 10. 引用

本実装で参考にした主要文献は本ドキュメント §7 を参照。
