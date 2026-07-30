# 設計書: agents層（抽出・連結・整文）の修正

実施者向け。根拠: `docs/review-2026-07-02/02_agents_graph_llm.md`（Critical/High）。グラフの質＝介入の質の上流なので、H1統合（フェーズ4）の**前に**入るのが望ましい。検証はすべてシミュ評価・単体テストで可能（対面不要）。品質基準は共通。

## G1 `fix(agents): evidence↔claim エッジの方向制約をコードで強制する`（02章 Critical）

### 問題
linking は LLM の返す5値方向をそのままエッジ化し、「evidence が常に src」という設計不変条件はプロンプト指示のみ。逆向きエッジは L1 選定（direction="in"）と weak_claims 判定から静かに漏れる。
### 修正
`linking.py` の `_maybe_make_edge`（該当箇所は実装時に確認）で、ペアの一方が evidence の場合: 方向が「claim→evidence」を意味する判定だったら **evidence→claim に正規化**（a_supports_b で a=claim, b=evidence なら b_supports_a 相当として扱う——意味論は「事実が主張を支持/攻撃する」に固定）。正規化した件数を統計ログに残す（プロンプト遵守率の観測）。
### テスト
4方向×(claim,evidence)組合せの正規化テーブルテスト。逆向きが store に入らないこと。

## G2 `feat(agents): 抽出を文脈付き・発話内関係込みの1呼び出しにする`（02章 High・最上流の精度損失）

### 問題
extraction は単発話・文脈なしで claim/premise ノードのみ返す。(a) 日本語会話の指示語・省略（「それは違うと思います」）を解決できない、(b) 発話内の premise→claim 支持関係を捨て、後段 linking が別LLM呼び出しで再発見している。
### 修正
- extraction の入力に直近 K 発話（K=3、話者名付き）を参照文脈として追加。プロンプトに「参照は指示語・省略の解決のみに使い、ノード化するのは判定対象発話のみ」を明記（fact判定/triage と同じ規約）
- 出力スキーマに発話内エッジ `intra_edges: [{src_idx, dst_idx, relation}]` を追加し、`EdgeCreator="extraction"`（schema に定義済み・未使用）でエッジ化。linking は発話内ペアをスキップ（重複判定の削減＝コスト減）
- 指示語解決の結果、claim テキストは**自己完結文**（「それは違う」→「コスト見積もりが違うという主張」）に正規化させる。これは A2 の soft-merge（embedding類似）の精度も上げる
### テスト
- フェイクLLMで: 文脈が渡ること、intra_edges がエッジ化されること、linking が発話内ペアを再判定しないこと
- 回帰: 既存 extraction テストのスキーマ互換（intra_edges 欠落時は空として扱う）
### 評価
シミュのサンプル transcript 5件で新旧の抽出結果を並べて目視比較するスクリプト（`scripts/compare_extraction.py`、雑でよい）を作り、結果を PR 説明に貼る。

## G3 `refactor(agents): L2整文ヘルパーの一元化と dead branch 除去`（02章 High）

L2 の LLM 整文が呼び出し側2箇所に複製され、`self.llm is None` の dead branch がある。整文を facilitation 側の単一ヘルパーに集約（H1 設計書 §3 が同ヘルパーを流用する前提なので、先にやるとH1が楽になる）。dead branch は削除。テストは既存 L2 テストの移設で足りるはず。

## G4 `fix(agents): Web検索クエリの生成を主張原文から検索語へ`（02章 Medium）

web_search が主張原文をそのままクエリにしている。LLM で「検索エンジン向けクエリ（固有名詞・数値・論点語を含む短句）」に変換する1段を追加（structured output、既存パターン）。テスト: フェイクLLMでクエリ変換が呼ばれること。

## G5 `fix(llm): CostTracker のログ到達不能バグ修正`（02章）

02章指摘のログ到達不能分岐（該当箇所は cost.py を読んで特定）を修正し、集計が全呼び出しを拾うことをテストで固定。実験のコスト見積もり（E5）の信頼性に直結。

## 実施順序と検証

G1（小・Critical）→ G3（小・H1の前提）→ G2（中・本丸）→ G4/G5（小）。
全体を通した検証: `uv run das run-session tests/fixtures/cafeteria_transcript.jsonl` 相当のE2E（フェイクLLM単体テスト＋、可能ならAPI付きスモーク1回）でグラフが構築されること。G2 は挙動変更なので、eval 側の既存テストが壊れたら「意図した変化か」を必ず判断してから直す（黙って期待値を書き換えない）。
