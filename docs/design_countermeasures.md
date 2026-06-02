# 4 つの懸念に対する設計対策

## 1. IBIS との差異と研究的位置づけ

### IBIS (Issue-Based Information System) との構造的差異

| 比較軸 | IBIS | 本手法 (統合議論グラフ) |
|--------|------|------------------------|
| ノード型 | Issue / Position / Argument | claim / premise / evidence |
| 外部知識の扱い | フレームワーク外 (議論ログのみ) | evidence ノードとして同一グラフ上に統合 |
| 関係の表現 | Position→Issue, Argument→Position (固定木構造) | 任意ノード間の支持/攻撃。同じ事実が主張Aを支持し主張Bを攻撃する (対象主張ごとの相対的エッジ) |
| グラフ構築 | 人手 or 事後整理 | LLM エージェントによる自動リアルタイム構築 |
| 活用方法 | 議論の可視化・整理 | 構造的偏り検知 → 選択的介入 (バイアス補正) |

### 本手法でのみ可能になる介入

1. **over_supported_claims への攻撃エビデンス自動提示**: 支持ばかり集まった主張に対し、攻撃関係にある外部事実を優先提示。IBIS は support/attack の区別がないため不可能。
2. **cross_source_edge_rate による統合度の定量化**: 議論ノードと外部知識ノードの間のエッジ比率を測定し、外部知識の活用度を客観指標化。
3. **相対的エッジによる多面的事実提示**: 同一事実が主張 A を支持し主張 B を攻撃することをエッジで明示。IBIS の木構造では表現不可。

### Dung の Abstract Argumentation Framework (AF) との関係

本手法は Dung (1995) の AF を拡張し、ノードに source 属性 (utterance/document/web) を持たせたもの。純粋な AF は攻撃関係のみだが、本手法は支持も含む bipolar AF に近い。ASPIC+ (Modgil & Prakken, 2014) の構造化論証とも関連するが、本手法はより実用寄りで LLM による自動構築を前提としている。


## 2. RAG の有効性: hybrid retrieval + retrieval 品質ログ

### 問題

embedding 類似度は「トピック的に近い」を拾うが、「論証的に反証する」ものは意味が対立するため distance が遠くなりがち。攻撃関係の候補が top-k から漏れるリスク。

### 対策: hybrid retrieval (embedding + BM25)

```
final_score = α × embedding_normalized + (1-α) × bm25_normalized
```

- BM25 はキーワード一致を見るため、同じ用語を使いつつ反対の結論を述べる文 (典型的な反論) を拾いやすい
- α = 0.7 (embedding 重視) をデフォルトとし、実験で最適値を探索
- `top_k_per_source` との併用も可能

### 対策: retrieval 品質ログ (`RetrievalQualityLog`)

各 `link_node` 呼び出しで top-k 候補の判定結果を記録:
- `n_support` / `n_attack` / `n_none` の内訳
- `hit_rate`: support + attack の割合 (retrieval の精度)
- `attack_ratio`: 有効候補のうち attack の比率 (反論の拾い具合)
- `candidate_sources`: source 別の候補内訳

これにより「embedding-only vs hybrid」の retrieval 精度を定量比較できる。


## 3. ファシリテーションの有効性

### 問題

「隣接エッジを提示するだけで議論の質が変わるか」の検証が不十分。介入が逆効果になるケースの検知がない。

### 対策 A: ablation 条件 (`ConditionGraphlessFacilitation`)

グラフなしで LLM に直接ファシリテーション生成させる条件を追加:
- 外部知識: FlatRAG と同じ embedding 検索
- 介入判断: LLM が議論履歴を見て「何を提示すべきか」を直接生成
- グラフ構築・構造的偏り検知・攻撃エッジ追跡なし

4 条件比較: None < FlatRAG ≤ Graphless < FullProposal の序列が示せれば、グラフの貢献が切り分けられる。

### 対策 B: 負の影響指標

`DiscussionStructuralMetrics` に追加:
- `speaker_dropout_count`: 議論途中で発話が途絶えた話者数 (萎縮の指標)
- `utterance_length_decline_rate`: 前半/後半の平均発話長の変化率
- `longest_silence_turns`: 最も長く沈黙した話者の連続非発話ターン数

これらが条件間で有意差を持たないことを示す (= 介入が萎縮を引き起こしていない)。


## 4. Web 検索エージェントの発火制御

### 問題

対面リアルタイムでは毎 claim 発火するとレイテンシが大きく、情報過多にもなる。

### 対策: cooldown + lazy policy

- `cooldown_seconds`: 直近の検索から N 秒以内は発火抑制 (対面向けに 10-30 秒)
- `policy="lazy"`: `signal_stalled()` が呼ばれるまで検索をキューに溜め、議論停滞時にまとめて実行。FacilitationAgent の stalled 検知と組み合わせる。
- `policy="eager"`: 従来動作。シミュレーション向け。

レイテンシ vs 情報量のトレードオフはパラメータで制御可能にし、実験で最適設定を探る。
