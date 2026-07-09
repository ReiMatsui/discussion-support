# 修正作業指示書（コードレビュー 2026-07-07 対応）

> **[アーカイブ]** 指摘は第1次修正(T1-T10)で対応済み。現役の指示書は docs/fix-instructions-2026-07-09_round2.md

> 作成: Claude Fable 5 (claude-fable-5), 2026-07-07

背景: `docs/code-review-2026-07-07.md` のレビュー結果に基づく修正タスク。本リポジトリは「対面議論をライブ文字起こしし、AIファシリテーターが議論グラフ(AF)を基に介入する」研究プロトタイプ。

## 作業ルール

- タスクは番号順（優先度順）に着手する。1タスク=1コミットを目安に。
- 各タスク完了時に `uv run pytest -q` が通ること（実APIは呼ばない）。既存テストを壊さない。
- 挙動を変える修正には必ずユニットテストを追加・更新する（tests/unit/live/ 等に既存の対応テストあり）。
- 「要確認」と書かれたタスクは、まずコードを読んで問題が実在するか確認し、誤検知なら修正せず理由を報告して次へ進む。
- 判断に迷う設計変更（閾値の値など）は勝手に決めず、選択肢と推奨を報告に書く。

---

## タスク1: Partner同席時の沈黙閾値がプロファイルを無視する問題【高】

- 場所: `src/das/asr/live/_workers.py` ~496-509、`src/das/asr/live/_constants.py:394-395`
- 現状: `silence_thresh = _AGENT_DEBATE_SILENCE if partner_present else silence_summarize` により、Partner同席時は積極性プロファイル（active=8.0 / controlled=None）を無視して常に15.0秒。controlled は「沈黙要約なし」の設計なのに沈黙候補が復活する。
- 修正方針:
  - `silence_summarize is None` の場合は Partner同席時も沈黙候補を出さない。
  - それ以外は `max(silence_summarize, _AGENT_DEBATE_SILENCE)` とする（Partner会話を邪魔しない意図は保ちつつ、プロファイルの無効化設定を尊重）。
- テスト: partner_present × プロファイル3種（active/standard/controlled）の組で候補生成を検証。

## タスク2: factcheck の confidence が未使用【高】

- 場所: `src/das/asr/live/_workers.py` ~1290 付近（`_FACTCHECK_PROMPT` の結果処理）
- 現状: プロンプトは confidence(high/medium/low) を返させ、docstring は「high のみ採用」と書くが、実装は `should_correct` のみ参照。低確度の訂正が対面議論に流れる。
- 修正方針: `should_correct and confidence == "high"` のときのみ訂正を発火。confidence が欠落・不正値のときは発火しない（安全側）。docstring と挙動を一致させる。
- テスト: confidence=medium/low/欠落 で訂正が出ないこと。

## タスク3: meeting_epoch（会議リセット）保護漏れ【高】

- 場所: `src/das/asr/live/_workers.py`
  - `_run_agenda_detector`（~900-922）: 他ワーカーにある epoch リセットチェックがない。
  - `_retry_counts` / `_recent_corrections`（~1153-1240）: epoch をまたいでリセットされない。
- 修正方針: 他ワーカーと同じ epoch チェックパターンに揃え、epoch 変化時に上記の内部状態をクリアする。
- テスト: epoch を進めた後に旧状態が参照されないこと。

## タスク4: AFランタイムの epoch 保護が store 追加を守っていない【高】

- 場所: `src/das/asr/live/_af_runtime.py:339-385`（`poll_once`）
- 現状: ロック解放後に LLM 呼び出し込みの `ingest_utterance` を実行。epoch チェックはループ先頭と cursor 書き戻しのみで、処理中にリセットが起きると旧会議の発話が新グラフに混入しうる。
- 修正方針: `ingest_utterance` 完了後・store 反映前に epoch を再確認し、変わっていたら結果を破棄する。ループ各イテレーション先頭でも epoch を確認して早期 break。
- テスト: ingest 中に epoch が変わったケースで store に追加されないこと（LLM はモック）。

## タスク5: 不正 candidate_id で StopIteration クラッシュ【中・修正は小】

- 場所: `src/das/asr/live/_workers.py` ~794-797, ~860
- 現状: `next(c for c in cands if c.id == decision.candidate_id)` にデフォルトなし。LLM が不正 ID を返すとワーカースレッドが落ちる。
- 修正方針: `next((...), None)` にして None なら警告ログを出しスキップ。
- テスト: 不正 ID でクラッシュせずスキップされること。

## タスク6: consensus の detected_at_turn が常に最終ターン【高（評価を使う場合）】

- 場所: `src/das/eval/consensus.py:243, 303`
- 現状: `detected_at_turn=transcript[-1].turn_id if consensus else None` で、合意検出時刻が常に最終ターンに固定。
- 修正方針: 実際に合意が成立したターンを特定して記録する。判定ロジック上ターン特定が不可能な場合は、フィールドを `None` にして docstring に「未対応」と明記する（偽の時刻を記録しない）。
- テスト: 途中ターンで合意が成立するフィクスチャで正しいターンが記録される（または None になる）こと。

## タスク7: barge-in 判定の非対称（相槌でキャンセル）【中】

- 場所: `src/das/asr/live/_workers.py` ~2042
- 現状: facilitator 発話への割り込みキャンセルが「8文字超」のみで、partner 側にある `_BACKCHANNEL_RE`（相槌除外）を通していない。相槌でファシリテーター発話がキャンセルされる。
- 修正方針: facilitator 側にも `_BACKCHANNEL_RE` による相槌除外を適用し、partner 側と判定を共通化（ヘルパー関数に抽出）。
- テスト: 「そうですね」等の相槌でキャンセルされないこと。

## タスク8: cooldown スコープの不統一【中・要設計判断】

- 場所: `src/das/asr/live/_facilitation.py` ~178-196（`_KIND_POLICY`）
- 現状: `summarize` は global スコープだが同 priority 帯の `af_l1` は kind スコープ。交互発火で「連続介入防止」が骨抜きになりうる。`manual` は cooldown 5秒で global をバイパス（連打可能）。
- 修正方針: `af_l1` も global スコープにするのが第一候補。manual の連打については現状維持でよいが、判断根拠をコメントに明記。**変更前に既存テスト（tests/unit/live/test_facilitation_controller.py）の意図を確認し、意図的な設計なら変更せず報告のみ。**

## タスク9: 例外処理・小バグ群【中〜低・まとめて1コミット可】

1. `src/das/asr/live/_workers.py` ~1189: `bool(result.get("factual_claim"))` — LLM 出力が文字列 `"false"` の場合 True になる。文字列/bool 両対応の正規化関数を通す。（要確認）
2. `src/das/cli/_listen.py:183-195`: パイプライン異常終了時に即 EOS が積まれ未処理発話がロスしうる。EOS 投入前にキュー消化を待つ or EOS をキュー末尾投入に。（要確認）
3. `transcriber/server.py:89-100`: OpenAI WS 接続失敗時に `http_session.close()` 未呼び出し。try/finally 構造を修正。
4. `src/das/asr/live/_af_runtime.py:224-226`: embedding 失敗の無ログ握りつぶし。warning ログを追加（毎回再試行のコストも一言コメント）。
5. `src/das/asr/live/_workers.py` 2110-2111 vs 2172-2174: `state.topics` 読み出しのロック有無が不統一。ロック取得で統一。
6. `src/das/runtime/orchestrator.py:131-132`: ノード取得失敗時に linking / Web publish が黙ってスキップされる。warning ログ追加。

## タスク10: 要確認事項の調査（修正は実在確認後）

以下はレビューで「要確認」とされた項目。それぞれ実在するか調査し、実在すれば最小修正、誤検知なら報告のみ:

1. `_realtime.py:610-613` — `_played_bytes` が更新されず `_barely_played` が常時 True の疑い。
2. `_workers.py` ~1355-1365 — participation checker が `_INVITE_CHECK_SEC` の待機なしで発火する疑い。
3. `_workers.py` ~511-523 — `invite_payload["same_as_last_invited"]` が下流未参照のデッドコード疑い。
4. `_simulator.py:201-217` — LLM 出力パース失敗時に同一ターンで無限リトライしうる（リトライ上限を追加）。
5. `agents/facilitation.py:368-374` — 最新発話の「ノード化完了」判定が linking 未完了を考慮していない疑い。
6. eval: structural_metrics の timestamp 文字列比較、stance 集計の pre/post 不揃いペルソナの扱い。

---

## 完了報告フォーマット

各タスクについて: 対応内容（1-2文）／変更ファイル／追加テスト／誤検知・見送りの場合はその理由。最後に `uv run pytest -q` の結果を貼る。
