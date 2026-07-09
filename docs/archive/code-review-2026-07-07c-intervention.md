# 介入・割り込みロジック徹底トレースレビュー (2026-07-07c)

> **[アーカイブ]** 未修正指摘は docs/fix-instructions-2026-07-09_round2.md に統合済み。本書は詳細の参照用

> 作成: Claude Fable 5 (claude-fable-5), 2026-07-07

対象: 介入決定パイプライン（候補生成→Controller裁定→配信→cooldown記帳）と、人間の割り込み（barge-in）処理フロー。7つの割り込みシナリオと6つのパイプライン観点をコード上でトレースした。高優先指摘は本体で裏取り済み。

## 総評

通常ケース（AI発話中の実質発話による割り込み、相槌の除外、中断介入のTTL付き再送）は緻密に作られており堅牢。破綻は「**生成中だが未発声**」という短い時間窓と、「**会議リセット・cooldown記帳の境界**」に集中している。

---

## A. 裏取り済みの重要指摘

### A-1. [高] 「生成中・未発声」のAIに割り込めず、古い介入が後追い再生される（stale delivery）
`_workers.py:2097` — 割り込み条件が `agent.ai_speaking` のみ。`interrupt()` 自体は `ai_speaking or _responding` で動作可能（`_realtime.py:581` で裏取り済み）なのに、呼び出し側が能力より狭い。人間の発話が「trigger送信後〜最初の音声delta受信前」に到着すると割り込み不発、しかもそのtickでカーソルが進むため**以後も永久に割り込まれず**、人間の発話前の文脈に基づく介入が後追いで再生され「無視された」ように聞こえる。partner優先ブロック（2126-2130）も同根。
**修正**: 割り込み条件を `agent.ai_speaking or agent._responding`（公開プロパティ化推奨）に広げる。

### A-2. [高] `interrupt()` のスレッド安全性不足
`_realtime.py:571-672` / `_partner.py:189-220` — interrupt() は2スレッド（agent worker と UI/イベントworker）から呼ばれるが、`_pending_intervention` 部分以外（キュー排出→300ms分再投入→終端マーカー、`response.cancel`/`truncate` 送信、`_current_item_id` クリア）はロックなし。partner側は全くロックなし。二重interruptで「終端マーカーの後に旧音声チャンクが再投入され、止めたはずの音声が再生される」順序が理論上成立（`_base.py:130-137` の再生スレッドは実音声チャンクのepochを見ない）。
**修正**: interrupt() 本体を `_state_lock` で保護（再入は先勝ちで即return）。

### A-3. [高] 全介入種が同一の `_last_intervention_at` を更新し、globalクールダウンが飢餓を起こす
`_workers.py`（fact:2311 / manual:2342 / drift:2362 / retry:2380 / summarize:2512 / silence:2527 / invite:2542 / af:2567）— fact（2秒周期で連発しうる）や retry の発火がそのたびに global cooldown の起点を更新するため、事実訂正が続く議論では drift/summarize/invite/af_l2 が**長時間出せなくなる**。
**修正**: global cooldown の起点更新を「global スコープの種別」または「発話量の大きい介入」に限定する設計判断が必要。

### A-4. [高] 会議リセットで `_run_agent_worker` のローカル状態が持ち越される
`_workers.py:2011-2027` — tick単位の epoch チェック（発話破棄・カーソル保護）はあるが（裏取り済み）、epoch 変化時に `_pending`（drain済み候補）・`_last_intervention_at`・`_recent_interventions`・`_last_invited` をリセットしない。**旧会議の介入候補が新会議冒頭で配信されうる**／旧会議由来のcooldownが新会議に食い込む。07bのA-2（af_requestsドレイン漏れ）と同根で、こちらが本体。
**修正**: ループ先頭の epoch 検知時にワーカーローカル状態を一括クリア。

### A-5. [高] globalスコープの cooldown が kind 別設定値を無視（07b A-1 の再確認＋詳細）
`_facilitation.py:290-304` — global/kind が if/else 排他で、summarize=30s・af_l2=60s の表値は**一切参照されない**。docstringは「kind cooldown に加え global も」と両方効くように読めるが実装は片方のみ。drift/invite は表値がたまたま `_INTERVENTION_COOLDOWN` と同値のため症状が隠れている。
**修正**: global スコープでも `max(policy.cooldown, inp.cooldown)` を採用し、両方の制約を満たす形に。

### A-6. [中] 割り込み発話がエコー判定で誤破棄されうる
`_recv_loop.py:82-120` — 割り込み発話は定義上AI再生区間と重なるため、必ず類似度0.35の安全網にかかる。AIと同じ話題・語彙の正当な割り込みが「エコー」として**無音で捨てられ**、recordsに残らない。さらに interrupt 時に delivered テキストを `_recent_ai_texts` に追加する安全網（F1）が比較コーパスを増やし誤判定率を漸増させる。
**修正案**: interrupt を発生させた発話（=割り込みトリガーと同一のraw record）はエコー破棄の対象外にする、または割り込み直後のみ閾値を引き上げる。

---

## B. 中程度・設計判断が必要な指摘

- **[中] レーン間で優先度表が機能しない** `_workers.py:2291-2294` — barge-inレーンで drift が hold のとき、通常レーンの summarize が先に配信されうる（優先度表では drift > summarize）。飢餓防止の意図的フォールスルーだが「脱線指摘より要約が先」は対面として不自然な場面がありうる。意図の明文化 or driftがhold中は通常レーンからdrift未満の優先度を除外。
- **[中] summarize 候補に TTL がない** `_workers.py:479-491` — 他種は30-90秒のTTL付きだが summarize は無期限。長い抑制の後に古い focus のまま配信されるリスク。モード切替（conversation→facilitate）で古い pending が発火する経路も同じ。TTL付与を推奨。
- **[中] リトライ上限が「連続」のみ** `_realtime.py:604-625` — attempts は送信成功でリセットされるため、「中断→再送成功→また中断」のサイクルに総量上限がない。同一介入の言い直しがセッション全体で無制限。総回数 or 同一テキストの再送上限を検討。
- **[中] 沈黙タイマー起点の非対称** — partner発話は確定の瞬間に `_last_utt_time` を更新（`_workers.py:1930`）、facilitator発話はエコーウィンドウ経由の遅延更新のみ。`_last_utt_time` の意味論（人間の最終発話か、発話イベント一般か）が経路間で不一致。統一を推奨。
- **[中] facilitator→partner interrupt の二重経路** `_workers.py:1888-1891` と `2126-2130` — イベント駆動とポーリングの2系統から別スレッドでほぼ同時に呼ばれうる。A-2修正とあわせて一本化推奨。

## C. 低優先

- `_realtime.py:634-669` — truncate 境界と graceful yield 300ms のズレ（実害微小、docstringとの厳密な不一致のみ）
- `_facilitation.py:243` — priority同値時のタイブレークが `_build_candidates` の追加順依存（現状は決定的だがリファクタで静かに変わる。明示キー追加を推奨）
- 言い直しが常に冒頭からの再現になる件（設計意図どおりだが、複数回中断時の体験は要観察）

---

## 修正の優先順位（推奨）

1. **A-1 + A-2**（割り込みゲート拡大とロック保護）— 対面体験の「AIが人の発話に被せてくる」問題の根治。セットで半日規模
2. **A-4**（会議リセット時のワーカーローカル状態クリア）— 07b A-2 と統合して1タスクに
3. **A-5 + A-3**（cooldown設計の整理）— 両方とも cooldown 記帳の設計判断を伴うため1タスクで。af_l2 頻発と drift/invite 飢餓の両方に効く
4. A-6（エコー誤破棄）— 対面パイロットで「割り込みが無視される」事象が出たらこれを疑う。先に診断ログ（echo_drop時の類似度と対象テキスト）を強化してから判断でも可
5. B群はパイロットの観察結果を見てから

**実験への影響**: すべてライブ介入側。シミュ評価には影響なし。対面パイロット（9月）前に 1〜3 の修正を強く推奨。
