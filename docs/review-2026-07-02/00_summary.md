# 徹底コードレビュー 総括（2026-07-02）

対象: `src/das/` 全体（レガシー `transcriber/`・`live_transcriber.html`・`experiments/`・`scripts/` は除外、削除候補として扱う）。
観点: (a) コード品質（保守性・可読性）、(b) 目的「AIで対面会議に介入する」に対するロジックの本質性・UI/UX、(c) 外部ツール選定。
主要指摘の file:line は抜き取り検証済み（実在確認）。

詳細は各章を参照:

- [01_live_pipeline.md](01_live_pipeline.md) — src/das/asr/live/
- [02_agents_graph_llm.md](02_agents_graph_llm.md) — agents / graph / llm / prompts
- [03_eval_cli.md](03_eval_cli.md) — eval / cli
- [04_ui_ux.md](04_ui_ux.md) — ライブWeb UI / Streamlit
- [05_external_tools.md](05_external_tools.md) — 外部ツール選定（2026年7月時点のWeb調査付き）

---

## 総合評価

コード品質そのものは研究プロトタイプとして高水準（DI、structured output、決定的フォールバック、テスト網羅、コスト計測）。一方で、**研究の核と実装の断絶**、**評価の妥当性を壊すバグ**、**regex による意味判定の肩代わり**という3つの構造問題が、個々の品質の高さを打ち消している。fix コミットの積み重なり（closed roster / manual call / voice call / fact prefilter）は、いずれもこの構造問題の対症療法として説明できる。

## 最重要指摘（横断トップ7）

### 1. [Critical] 研究の核（統合AF）とライブ介入パイプラインが完全断絶
`src/das/asr/live/` は `das.graph` / `das.agents` を一切 import しない。対面ライブUIの介入は AF と無関係の drift/fact/invite ルール（`FacilitationController`）で動き、AF ベースの `FacilitationAgent.decide_intervention`（facilitation.py:301）は eval と `listen-soniox` の別系統のみ。**貢献②③（AF状態に基づくモダリティ非依存の介入判断）が、本命の対面モダリティで一度も検証されていない。**
→ 再設計: AF 判断を `InterventionCandidate(kind="af_l1/l2")` を生成する checker として既存 Controller 経路（候補生成→調停→整文の3層）に統合する。

### 2. [Critical] LLM judge が条件名と期待方向を知らされている（評価無効の疑い）
judge.py:130-134 が条件名をプロンプトに明示し、prompts/judge.md の intervention_transparency 定義が「情報提供がない条件では 1、提案手法なら高めに」と採点方向を直接指示。RQ3 の予備実験結果は誘導の産物の疑いが濃い。条件盲検化とスコア定義の中立化が必須。あわせて citation_rate の照合対象不一致（facilitation.py:412 の addressed_to と conditions.py の round-robin 注入のずれ）、flat_rag 提示が全部「[反論]」表示になる judge.py:88 のバグ、疑似反復（ペルソナ×ランの pool）も評価の妥当性を直接損なう。

### 3. [Critical] drift 候補 hold による全介入レーンの無期限飢餓
確認待ち drift 候補が抑制されると `hold` → `continue`（_workers.py:1710-1711）で通常レーン（沈黙要約・声かけ）の評価ごとスキップされる。drift 候補に TTL がなく、標準プロファイルは confirmations=2 のため、「脱線が1回検出され自然に戻った」だけで以後の介入が会議終了まで全停止しうる。drift に expires_at を付与し、hold を全レーン停止ではなくフォールスルーに。

### 4. [High] regex による意味判定の肩代わり（対症療法の集積地）
fact prefilter の regex 群（_workers.py:100-124、「二丁拳銃」等の特定テスト会話への過学習を含む）と音声呼びかけ検出の三段 regex（呼称 "AI" がデモ議題「AIツール導入」と衝突）。根本原因はコスト回避のため意味判定を regex に落としたこと。発話ごと1回の軽量 LLM 分類（fact候補/呼びかけ/その他を同時判定する structured output）に統合すれば regex 群ごと削除できる。

### 5. [High] 抽出パイプライン最上流の精度損失
extraction は単発話・文脈なしで claim/premise ノードのみ返し、発話内の premise→claim 関係を捨て、後段 linking が別の LLM 呼び出しで再発見している。日本語会話の指示語・省略（「それは違うと思います」）を解決できず、以降の全段が劣化を引き継ぐ。直近文脈付き・関係込みの1呼び出しへ。

### 6. [High] レイテンシ予算の未設計
fast/smart 両方が reasoning モデル gpt-5-mini（settings.py:26-27）で、reasoning_effort 制御も per-call timeout もない（wait_for 保護は linking のみ）。発話→介入提示までの遅延が構造的に数秒以上乗り、「ライブ介入に耐える」ことが計測されていない。段ごとのレイテンシ予算を決めて計測をログに残すべき。なお外部ツール調査によると gpt-5-mini/nano は2世代前で、現行 GPT-5.4 mini/nano への更新を推奨（05章）。

### 7. [High] 無音故障とUIの構造問題
STT切断・再接続がUIに一切見えない（_webapp.py:728 の空 onerror、api_snapshot に健全性フィールドなし）— 文字起こしが黙って止まっても画面は「ライブ」のまま。また進行中のAI発話を止める手段がない（「無視しやすい介入」の核心の欠落）。構造面ではホスト用操作卓と参加者向け表示の分離（/console と /board）が最小工数で最も効く。

## 対症療法パターンの根本原因（分析）

fix コミットが集中した3領域には共通構造がある: **判断の正しい置き場所（LLM/AF/Controller）にコストやレイテンシの都合で置けず、手前の層に表層ルールを足した**こと。

1. fact prefilter → 意味判定を regex に肩代わり（上記4）
2. closed roster / speaker policy → 話者確定ロジックが4箇所に分散し、二重実装
3. manual/voice call → Controller 移行が途中で止まり、legacy selector 併存・抑制理由の日本語部分文字列マッチ（`"直前の介入から間隔不足"` で分岐、_workers.py:946-951）・pause/cooldown 定数の3箇所重複

いずれも「新しい層を足す」のではなく「移行を完了させて旧層を消す」ことが本質的な解。

## コード品質の主な指摘（抜粋）

- `_workers.py`（2153行）: レーン別 worker + 候補生成 checker + 整文で分割可能。抑制理由は文字列でなく構造化コードに
- `_session_state.py`（1126行）: save() が state_lock 保持のままディスクI/O（O(n²)）。リセットとワーカーカーソルの無ロック競合（epoch 導入で解決）
- `run_eval.py`: `_run_single` が8責務300行。「実行と採点の分離（rescore-everything 化）」にすると judge 修正を API 再ラン不要で再適用できる
- L2 の LLM 整文が呼び出し側2箇所に複製、dead branch あり
- `_webapp.py`: 755行のHTML埋め込み文字列 → 別ファイル化。毎秒フルスナップショットSSE＋innerHTML全置換 → 差分配信へ

## 外部ツール（05章の結論のみ）

Soniox stt-rt-v5・pyannoteAI・gpt-realtime-2・Streamlit+pyvis は2026年7月時点で妥当（維持）。要対応は: LLM既定を GPT-5.4 mini/nano へ更新、AssemblyAI 分離プロバイダは日本語ストリーミング非対応のため実験用と明記か削除、Soniox 採用根拠の日本語一次ベンチ（vs Speechmatics / AmiVoice）を一度取得して論文の選定根拠にする。

## 推奨着手順

1. 評価の Critical 修正（judge 盲検化・citation 照合・flat_rag 表示バグ）— 予備実験の結論に直結、修正コスト小
2. drift hold 飢餓の修正 — 1会議を壊すバグ、修正コスト小
3. AF とライブ介入の統合 — 研究の核。Controller 3層への checker 統合として段階的に
4. regex 群の LLM 分類への置換と legacy selector 撤去 — 対症療法の清算
5. UI: 接続健全性の可視化・発話キャンセル・/console と /board 分離
6. モデル更新（GPT-5.4系）と段別レイテンシ計測
