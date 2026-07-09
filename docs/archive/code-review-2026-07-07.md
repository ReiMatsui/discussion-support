# コードレビュー報告 (2026-07-07)

> **[アーカイブ]** 指摘は第1次修正(T1-T10)で対応済み。現役の指示書は docs/fix-instructions-2026-07-09_round2.md

> 作成: Claude Fable 5 (claude-fable-5), 2026-07-07

対象: src/das 全体 + transcriber。観点は (1) バグ・意図しない挙動、(2) 介入ロジックの不自然さ、(3) プロンプト/パース整合性、(4) 評価の信頼性。
サブエージェント4系統でレビューし、高優先の指摘は本体で裏取り済み。裏取りで誤検知と判明したものは末尾に記載。

---

## A. 介入ロジックの不自然さ（最重要観点）

### A-1. [高・裏取り済] Partner同席時に沈黙閾値がプロファイル無視で15秒固定
`_workers.py` ~496-509 / `_constants.py:394-395`

```python
silence_thresh = (_AGENT_DEBATE_SILENCE if partner_present else silence_summarize)
```

- `partner_present` が真だと、積極性プロファイル（active=8.0 / controlled=None 等）を無視して常に 15.0 秒。
- 特に `controlled` プロファイルは「沈黙要約なし (None)」の設計なのに、Partner同席時は沈黙候補が復活する。設定と実挙動が乖離。
- 修正案: `partner_present` 時は `max(silence_summarize, _AGENT_DEBATE_SILENCE)`（Noneなら無効のまま）等、プロファイルを尊重する合成に。

### A-2. [中] barge-in（割り込み）判定の非対称
`_workers.py` ~2042 — facilitator側の割り込みキャンセルは「8文字超」だけで判定し、partner側にある相槌除外（`_BACKCHANNEL_RE`）を通していない。「そうですね〜」程度の相槌でファシリテーターの発話がキャンセルされうる。対面議論では相槌が頻発するので、体感上「AIが言いかけてやめる」不自然さに直結。

### A-3. [中] cooldownスコープの不統一で「仕切りすぎ防止」が骨抜きに
`_facilitation.py` ~178-196

- `summarize` は global スコープ（他介入直後は抑制）だが、同 priority 帯の `af_l1` は kind スコープ。summarize と af_l1 が交互に発火すると global cooldown の意図（連続介入の防止）が実質無効化される。
- また `manual`（明示呼び出し）は cooldown 5秒 kind スコープで global 25秒をバイパス。明示呼び出し優先は妥当だが、5秒間隔の連打を許すのは過剰かは要判断。

### A-4. [中] priority同値時のconfidence比較の意味論
`_facilitation.py` — `af_l1`=`summarize`(4)、`af_l2`=`invite`(6) が同値で、同時成立時は confidence で決着。fact系のconfidenceは high/medium/low 由来、af系は別スケールの可能性があり、種別間で比較可能な設計になっているか要確認。

### A-5. [中] participation checker（沈黙者への声かけ）
- `_workers.py` ~1355-1365 — 待機時間 `_INVITE_CHECK_SEC` を経ずに即LLM判定が発火しうる（要確認）。
- `_PARTICIPATION_PROMPT` の返す話者名を `valid_invite_targets` と完全一致でしか照合せず、表記揺れで声かけが不発になる可能性。
- `invite_payload["same_as_last_invited"]` は下流で未参照のデッドコードの疑い。

### A-6. [低] 沈黙系トリガーの相対優先
`_constants.py:392-396` — `_AGENT_CONV_SILENCE(1.5s) < _INVITE_SILENCE(2.0s)` のため、モード混在時に「会話継続」が「声かけ」より先に成立しやすい。意図通りか要確認。

---

## B. バグ・意図しない挙動

### B-1. [高] AFランタイムの会議リセット(epoch)保護が不完全
`_af_runtime.py:339-385` — `poll_once` はロック解放後に LLM 呼び出し込みの `ingest_utterance` を実行。epochチェックはループ先頭と cursor 書き戻し時のみで、**store への追加自体は保護されない**。会議リセット直後、旧会議の発話が新グラフに混入しうる。

### B-2. [高] meeting_epoch リセット漏れがワーカーに散在
`_workers.py` — `_run_agenda_detector`(~900-922) は他ワーカーと異なり epoch リセットチェックなし。`_retry_counts` / `_recent_corrections`(~1153-1240) も epoch をまたいで残留。「新しい会議」直後に前会議の議題・訂正履歴が漏れる。

### B-3. [高] `state.topics` のロック不統一
`_workers.py` 2110-2111 vs 2172-2174 — 読み出しがロック有無混在。ドリフトチェッカーと競合し不整合なリストを読む可能性。

### B-4. [中] `StopIteration` でワーカークラッシュ
`_workers.py` ~794-797, 860 — `next(c for c in cands if c.id == decision.candidate_id)` にデフォルトなし。LLMが不正な candidate_id を返すとワーカースレッドが落ちる。`next(..., None)` + スキップに。

### B-5. [中] factcheck の confidence 未使用
`_workers.py` ~1290 — プロンプトは high/medium/low を返させ、docstring は「high のみ採用」だが、実装は `should_correct` しか見ていない。低確度の訂正が対面議論に流れうる = 誤介入リスク。

### B-6. [中] パイプライン異常終了時の発話ロス
`_listen.py:183-195` — `_live_mod.main()` 例外終了時に即 EOS(None) が積まれ、キュー内の未処理発話が捨てられる可能性。

### B-7. [中] transcriber のリソースリーク
`transcriber/server.py:89-100` — OpenAI WS 接続失敗時に `http_session.close()` が呼ばれない。

### B-8. [低〜中] その他
- `_workers.py` 各所 — 広範な `except Exception` 握りつぶし（トレースバック非出力箇所あり）。障害解析コスト増。
- `_workers.py` ~1189 — `bool(result.get("factual_claim"))`: 文字列 `"false"` が True になる（LLM出力が文字列で来た場合。要確認）。
- `_realtime.py:610-613` — `_played_bytes` の更新元が見当たらず `_barely_played` が常時 True の疑い（要確認）。
- `_simulator.py:201-217` — パース失敗時に同一ターンで無限リトライしうる（stop 以外の脱出経路なし）。
- `_session_state.py:1046-1108` — `write_html` が O(n²)。長時間会議で 2 秒間隔更新がブロッキング。
- `_af_runtime.py:224-226` — embedding 失敗を無ログで握りつぶし、毎回再試行。
- `orchestrator.py:131-132` — ノード取得失敗時に linking / Web publish がスキップされたまま終了する可能性。

---

## C. 評価系（研究結果の信頼性）

- [高] `consensus.py:243,303` — `detected_at_turn` が **常に最終ターン固定**（裏取り済）。「いつ合意したか」の分析には使えない。合意時刻を使う集計があるなら結果が歪む。
- [中] structural_metrics — timestamp を文字列比較しており形式依存（要確認）。
- [中] stance 集計 — pre/post が揃わないペルソナの扱いでペア差分のサンプルサイズが乖離しうる（要確認）。
- [中] facilitation.py:368-374 — 最新発話の「ノード化完了」判定が extraction 完了のみで linking 未完了を考慮していない疑い（要確認）。

---

## D. 裏取りの結果、誤検知と判断した指摘

- linking.py のバッチ判定「index重複で判定消失」→ 既定 none で埋めてから上書きする設計で、コメント通り重複・欠落に頑健。**問題なし**。
- web_search.py の「エッジ二重カウント」→ `or` 条件は1エッジ1カウント。自己ループ以外で二重計上なし。**問題なし**。

---

## 推奨対応順

1. A-1（Partner時のプロファイル無視）と B-5（confidence未使用）— 介入の質に直結
2. B-1/B-2（epoch保護）— 「新しい会議」機能の信頼性
3. B-4（StopIteration）— クラッシュ系で修正が1行
4. C の consensus.detected_at_turn — 評価に使うなら先に修正
5. A-2/A-3 — 対面での体感品質
