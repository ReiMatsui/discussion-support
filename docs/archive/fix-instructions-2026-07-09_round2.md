# 修正作業指示書 第2次（2026-07-09）

> 作成: Claude Fable 5 (claude-fable-5), 2026-07-09

背景: 第1次修正（T1-T10、マージ済み）後の再レビューおよび介入・割り込みロジックの徹底トレース（旧 code-review-2026-07-07b / 07c、docs/archive/ に保管）で確認された**未修正の指摘**を統合した実行可能な指示書。これ1枚が現役。

## 作業ルール（第1次と同じ）

- 番号順に着手。1タスク=1コミット。`uv run pytest -q` 必須。挙動変更にはテスト追加。
- 「要確認」は実在確認→誤検知なら理由を報告して次へ。設計判断は勝手に決めず選択肢を報告。

**実験への影響**: 全タスクがライブ介入側。シミュ評価には無影響。対面パイロット（9月）前に R1〜R4 の完了を強く推奨。

---

## R1【高】「生成中・未発声」のAIに割り込めない（stale delivery）

- 場所: `_workers.py` ~2097（`if _human_spoke and agent.ai_speaking:`）、partner優先ブロック ~2126-2130
- 現状: 割り込み条件が `ai_speaking` のみ。`interrupt()` 自体は `ai_speaking or _responding` で動作可能（`_realtime.py:581`）。trigger送信〜音声開始前の窓に人間が発話するとカーソルだけ進み永久に割り込み不発 → 発話前文脈の介入が後追い再生され人間の発話に被さる。
- 修正: 割り込み条件を `agent.ai_speaking or agent.responding`（`_responding` の公開プロパティ追加）に拡大。partner優先ブロックも同様。
- テスト: `_responding=True, ai_speaking=False` で実質発話→interruptが呼ばれる。

## R2【高】`interrupt()` のスレッド安全性

- 場所: `_realtime.py:571-672`、`_partner.py:189-220`
- 現状: 2スレッド（agent worker / UIイベント系）から呼ばれるが、キュー排出→300ms再投入→終端マーカー、cancel/truncate送信、`_current_item_id` クリアが無ロック。partner側は全く無ロック。二重interruptで「終端マーカー後に旧音声チャンク再投入」が理論上成立。
- 修正: interrupt() 本体を `_state_lock` で保護し、再入は先勝ちで即return。facilitator→partner interrupt の二重経路（`_workers.py:1888-1891` と `2126-2130`）もどちらかに一本化。
- テスト: 2スレッド同時interruptでキュー整合性が保たれる（終端マーカー後に音声チャンクがない）。

## R3【高】会議リセットで agent worker のローカル状態が持ち越される

- 場所: `_workers.py` `_run_agent_worker`（`_pending`/`_last_intervention_at`/`_recent_interventions`/`_last_invited`、~2011-2027）、`_session_state.py:775-781`（ドレイン対象に `af_requests` がない）
- 現状: tick単位のepochチェックは発話破棄のみで、ワーカーローカル状態をリセットしない。旧会議の介入候補が新会議冒頭で配信されうる。`af_requests` キューもドレイン漏れ。
- 修正: ループ先頭のepoch変化検知時にローカル状態を一括クリア＋`reset_for_new_meeting()` のドレイン対象に `af_requests` を追加。
- テスト: epoch変化後、旧 `_pending` 由来の候補が配信されない。

## R4【高】cooldown設計の2つの乖離（1タスクで対応、要設計判断）

- (a) globalスコープが kind別設定値を無視: `_facilitation.py:290-304` — global分岐は `inp.cooldown` のみ参照し、表の summarize=30s / af_l2=60s は未使用。af_l2が意図より高頻度化。修正: globalスコープでは `max(policy.cooldown, inp.cooldown)`。
- (b) 全介入種が同一の `_last_intervention_at` を更新: fact（2秒周期）や retry の発火が drift/summarize/invite/af_l2 の global cooldown 起点を毎回リセットし飢餓を起こす。修正方針（要判断）: global起点の更新を globalスコープ種別＋summarize等の「発話量の大きい介入」に限定する案を第1候補として、既存テストの意図と突き合わせて提案・実装。
- テスト: af_l2 が60秒間隔を守る／fact連発中でも invite が出せる。

## R5【中】_as_bool の水平展開漏れ（3箇所）

- 場所: `_workers.py:1080`（drift）、`:1453`（invite）、`:1511`（intervene）
- 修正: `_as_bool()` を適用（文字列 "false" での偽陽性発火を防ぐ）。テスト: 文字列 "false" で発火しない。

## R6【中】participation checker に epoch ガードがない

- 場所: `_workers.py` `_run_participation_checker`
- 修正: 他ワーカーと同じ「LLM呼び出し後・`invite_requests.put()` 前」のepoch再確認を追加。

## R7【中・要確認】partner のテキストエコー判定が常時有効

- 場所: `_recv_loop.py:92-107` — agentはエコー窓内のみ類似度判定、partnerは窓ガードなしで常時判定。人間の発話が誤破棄されうる。docstringの設計方針と矛盾。意図的か確認の上、agentと同じ窓ガードに統一。

## R8【中・要確認】割り込み発話のエコー誤破棄

- 場所: `_recv_loop.py:82-120` — 割り込み発話は必ずAI再生と重なるため類似度0.35の安全網に毎回かかる。まず echo_drop 診断ログ（類似度値と対象テキスト）を強化し、パイロットで「割り込みが無視される」事象が出たら interrupt トリガー発話を除外対象に。

## R9【低・まとめて1コミット可】

1. summarize 候補にTTLがない（`_workers.py` ~479-491）→ 他種と同様のTTL付与
2. リトライ上限が「連続」のみ（`_realtime.py:604-625`）→ 同一介入の総リトライ回数上限を追加
3. 沈黙タイマー起点の非対称（partner即時 vs facilitator遅延、`_workers.py:1930` vs 1840-1861）→ 意味論を統一しコメント明記
4. `networkx_store.py:200` 孤立エッジの無ログ破棄 → warning追加
5. priority同値時のタイブレークが候補追加順依存（`_facilitation.py:243`）→ 明示的タイブレークキー追加
6. `citation.py:272` `id(it)` キーのembeddingキャッシュ → 内容ベースのキーに
7. `_webapp.py` SSEの `JSON.parse` にcatch追加／`_bootstrap.py:815` `suppress(Exception)` の限定化

## 参考: 誤検知として棄却済み（再調査不要）

- run_eval実行時の観測用AF未構築 → rescoreを正とする設計どおり
- linking.py index重複／web_search.py エッジ二重カウント → 問題なし
- `_played_bytes` 未更新疑い → `_base.py:137` で加算済み

## 完了報告フォーマット

第1次と同じ（対応内容／変更ファイル／追加テスト／棄却理由、最後に pytest 結果）。
