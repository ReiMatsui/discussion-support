# コードレビュー報告 第2回 (2026-07-07b)

> **[アーカイブ]** 未修正指摘は docs/fix-instructions-2026-07-09_round2.md に統合済み。本書は詳細の参照用

> 作成: Claude Fable 5 (claude-fable-5), 2026-07-07

対象: T1-T10修正マージ後の main 全体。前回（code-review-2026-07-07.md）とは独立に5系統のサブエージェントで再レビューし、高優先指摘は本体で裏取り済み。前回指摘の修正はいずれも正しく機能しており、デグレは検出されなかった。

---

## A. 裏取り済みの新規指摘（要修正）

### A-1. [高] global スコープの cooldown が kind 別設定値を無視
`_facilitation.py:291-299` — `cooldown_scope == "global"` の分岐が `policy.cooldown`（summarize=30s、af_l2=60s「俯瞰は頻発させない」）を使わず、共通の `inp.cooldown` のみで判定。**af_l2（俯瞰介入）が設計意図より高頻度に発火しうる**。設定テーブルと実挙動の乖離で、前回A-3の議論とも関係する本丸。修正: global 判定でも `max(policy.cooldown, inp.cooldown)` 等で kind 別値を尊重。

### A-2. [高] 会議リセットで `af_requests` キューがドレインされない
`_session_state.py:126, 775-781` — `reset_for_new_meeting()` のドレイン対象が drift/invite/factcheck/manual_call/summarize の5キューのみで、`af_requests` だけ漏れている。リセット直前にキューされた af_l1/af_l2 候補が**新しい会議に持ち越されて配信されうる**。T3（epoch保護）と同系統の追記漏れ。

### A-3. [高] `_as_bool` の水平展開漏れ（3箇所）
`_workers.py:1080, 1453, 1511` — T9で導入した `_as_bool` が factcheck 系にのみ適用され、`result.get("drift")`（脱線）、`result.get("invite")`（声かけ）、`result.get("intervene")`（整理）は生の truthiness 判定のまま。LLMが文字列 `"false"` を返すと**脱線介入・声かけ・整理介入が偽陽性で発火**する。同じ正規化を適用するだけ。

### A-4. [中→高] participation checker に epoch ガードがない
`_workers.py`（`_run_participation_checker`）— 他の全ワーカー（agenda/topic/drift/triage/fact/structuring/af_checker）はLLM呼び出し後に `meeting_epoch` を再確認するが、このワーカーだけ epoch を一切参照せず、リセット跨ぎで旧会議の参加者への `invite_requests.put()` が新会議に混入する。T3の適用漏れ。

---

## B. 要確認の新規指摘（実在すれば中程度）

- **[中] partner のテキストエコー判定が常時有効** `_recv_loop.py:92-107` — agent はエコー窓内のみ類似度判定するが、partner は窓ガードなしで常時判定。過去AI発話と偶然似た**人間の発話が破棄されうる**。docstring の設計方針（安全網はエコー窓中のみ）と矛盾。意図的か要確認。
- **[中] `interrupt()` が `_end_speech()` を経由しない** `_partner.py:189-220` — AI発話区間の記録タイミングが割り込み時刻より遅延しうる。実害は限定的。
- **[中] consensus の prefix 再評価と store の整合性** `consensus.py:283-299` — `_first_consensus_turn` は transcript を prefix 化するが store は全期間のまま。AF由来シグナルが prefix 外のノードに影響される可能性。T6修正の残課題として、store を使うシグナルの扱いを確認。
- **[中] `_INTERRUPT_MIN_CHARS=8` が短い制止を拾わない** — 「ちょっと待って」等の8文字前後の制止発話が割り込み対象から漏れうる。相槌除外(T7)は正常動作。対面での体感に関わるため実地で確認を。
- **[中] evidence 支持判定の自己包含** `agents/facilitation.py:672-677` — `_item_has_tension` の `n_support <= 1` 判定に評価対象の evidence 自身が含まれる疑い。

## C. 低優先（品質改善）

- `networkx_store.py:200` — リプレイ時の孤立エッジを無ログで破棄（ログ追加推奨）
- `_webapp.py` SSEハンドラ — `JSON.parse` 失敗時の catch なし／`_bootstrap.py:815` — `suppress(Exception)` が広すぎる
- `citation.py:272` — `id(it)` キーの embedding キャッシュ（内容ベースのキーに）
- `judge.py:146-157` — ペルソナ評価が逐次実行（gather で並列化可能）
- `openai_client.py` — `response.choices[0]` の防御なし
- `structural_metrics.py:219-232` — pct_attacks_answered の重複スキップ意図をコメント化

## D. 誤検知と判断した指摘

- 「run_eval 実行時に観測用AF未構築（E4未達成）」→ rescore フェーズで `build_observation_af` がデフォルト構築し、**rescore後summaryを正とする設計**（experiment_runbook 記載）どおり。実行時スコアは暫定値であり問題なし。
- 「linking.py `max(default=1.0) or 1.0` の二重防御」→ 冗長だが無害。
- 前回指摘のT1-T10修正はすべて正しく動作しており、デグレなし。

---

## 推奨対応順

1. **A-1〜A-4 を第2次修正としてまとめて実施**（いずれも既存パターンの横展開で小さい。テスト込みで半日規模）
2. B の partner エコー判定と consensus prefix 整合性は、設計意図を確認してから判断
3. C はシミュ本実験のブロッカーではないので後回し可

**評価実験への影響**: A-1〜A-4 はすべてライブ介入側で、シミュレーション評価（run_eval系）には影響しない。シミュ本実験はこの修正を待たず開始してよい。対面パイロット前には A-1〜A-4 を修正しておくこと。
