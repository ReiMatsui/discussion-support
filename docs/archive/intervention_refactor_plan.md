# 介入・割り込みロジック 修正＆リファクタリング計画

対象: `src/das/asr/live/` の `_realtime.py` / `_partner.py` / `_workers.py` / `_recv_loop.py` / `_session_state.py`

目的: (A) 既知バグの修正、(B) 競合リスクの解消、(C) 状態管理とトリガー経路を構造的に整理し、今後ガードを継ぎ足さなくても破綻しない設計にする。

---

## 全体方針

現状の弱点は2つに集約される。

1. **状態フラグが無ロックで複数スレッドから読み書きされている** — `ai_speaking` / `_responding` / `_interrupted` / `_pending_intervention` / `_current_item_id` / `_played_bytes`。
2. **`trigger()` を呼ぶスレッドが2系統ある**（`_run_agent_worker` と `_run_drift_checker`）。両者が同じ状態を見て同じ介入をリトライするため、二重発火・取りこぼしが起きる。

この計画は「まず低リスクでバグを潰す（Phase 1-2）→ テスト基盤を敷く（Phase 0は先行）→ 構造を直す（Phase 3）」の順で進める。各フェーズは独立してマージ可能。

---

## Phase 0: テスト基盤（リファクタの前提・先行実施）

live配下の介入ロジックには現在ユニットテストが存在しない。リファクタの安全網として最優先で整備する。

- **FakeWebSocket** を作る: `send()` で送ったJSONを記録し、`recv()` で台本のイベント列を返すスタブ。
- **FakeAudioQueue / 再生スレッドのモック**: `sounddevice` を使わずに `_played_bytes` を進められるようにする。
- 検証したいシナリオをテストケース化:
  1. 通常の介入（feed → trigger → audio.delta → transcript.done → on_ai_utterance 発火）
  2. 「（介入不要）」応答で音声が1チャンクも再生されず、`on_speech_start` が呼ばれないこと（← Bug 1 の回帰テスト）
  3. 割り込み（interrupt）後に `_pending_intervention` が保存され、リトライで1回だけ再送されること
  4. drift検出 → trigger に `drift_reason` が渡ること
  5. 送信例外時に `_pending` が失われないこと（← Bug 2 の回帰テスト）

成果物: `tests/unit/live/test_realtime_agent.py`, `test_partner.py`, `test_agent_worker.py`

---

## Phase 1: 確実なバグ修正（低リスク・即効）

### Fix 1 — 「介入不要」の音声漏れ＆パートナー誤中断

- 場所: `_realtime.py` `_handle`（output_audio_transcript.delta）/ `_flush_preflight` / 定数 `_preflight_chars`
- 問題: `_preflight_chars=3` がガード文言「（介入不要）」確定（5文字目）より短く、判定前にflush＝再生開始＆partner中断。
- 対応:
  - プリフライト保留条件を「文字数」から「**ガード文言のprefix判定**」に変更。現バッファが `（介入不要）` のいずれかのprefixである間はflushしない。
  - 安全側に倒すため、`_CANCEL_MARKERS = ("介入不要",)` を定数化し、`_is_possible_cancel_prefix(buf)` ヘルパーで判定。
  - フォールバックとして閾値も「マーカー最大長＋余裕」へ引き上げ（例: 6）。
- 回帰テスト: Phase 0 シナリオ2。

### Fix 2 — `trigger()` 送信失敗時の発話・介入消失

- 場所: `_realtime.py` `trigger()`
- 問題: `ws.send()` の前に `_pending.clear()` と `_pending_intervention=None` を実行しており、送信例外で内容が消える。
- 対応:
  - 送信用の `conv` 構築までは行うが、`_pending` の**確定クリアは送信成功後**に移動。
  - 送信失敗時は `_pending` を元に戻す（または「未送信」フラグで次回再構築）。`_pending_intervention` も送信成功まで保持。
  - `_responding=True` は送信成功時のみ立てる（現状通り）。

### Fix 3 — `_pending_intervention` リトライの二重化を解消

- 場所: `_workers.py` `_run_agent_worker`(302-307) と `_run_drift_checker`(109-116)
- 問題: 同一の保留介入を2スレッドがリトライ。
- 対応（Phase 3の布石）: **リトライ責務を `_run_agent_worker` 単一に集約**し、drift_checker側のリトライブロックを削除。drift_checkerは「新規drift検出」専任にする。
  - 暫定的にここで削除しておくと、Phase 3でのトリガー経路単一化がスムーズになる。

---

## Phase 2: 競合リスクの局所修正

### Fix 4 — `_responding` の test-and-set をアトミック化

- 場所: `_realtime.py` `trigger()`（348行のガードと399行のセット）
- 対応: `_responding` のチェックとセットを `self._lock` 配下に入れ、`response.create` 送信の直前に `_responding=True` を確定。二重 `response.create` の窓を閉じる。
- 注: Phase 3でトリガー経路を単一化すれば本質的に不要になるが、それまでの安全策として先に入れる。

### Fix 5 — パートナーの cancel→create 競合

- 場所: `_workers.py`(265-270) / `_partner.py` `interrupt` + `inject_context` + `_handle`(error)
- 問題: `response.cancel` 直後の `response.create` が「already has active response」を誘発し得る。`_handle` は `no active response` しか無視しない。
- 対応:
  - エラーハンドリングで `already has an active response` も警告に格下げ（無視ログ）。
  - 可能なら、`inject_context(request_response=True)` を「直前のcancelの response.done を待ってから create」する軽い順序制御を入れる（受信側で cancelled を確認したら pending create を flush）。まずはエラー握りつぶしの拡張で実害を消し、順序制御は様子見。

### Fix 6 — `ai_speaking` の世代（epoch）管理

- 場所: `_realtime.py`（および同型の `_partner.py`）の再生スレッド／`_handle`
- 問題: 応答が重なった際、旧応答のNone終端マーカーが新応答再生中に `ai_speaking=False` を誤セットしうる。
- 対応:
  - 各 `response.output_item.added` で `_response_epoch += 1`。
  - 再生キューに積むのを `(epoch, chunk)` / 終端は `(epoch, None)` に変更。
  - 再生スレッドは「**終端マーカーのepochが最新のときだけ** `ai_speaking=False`」にする。
- これにより echo window の早期崩壊（人間音声のエコー誤除去・誤再トリガー）を防ぐ。

---

## Phase 3: 中期リファクタ（構造の是正）

目的: 「無ロック共有フラグ」と「二重トリガー経路」を設計レベルで解消する。

### R1 — エージェント状態を単一ロック配下に集約

- `RealtimeAgent` / `ConversationPartner` に `self._state_lock` を導入。
- 可変状態（`ai_speaking`, `_responding`, `_interrupted`, `_pending_intervention`, `_current_item_id`, `_played_bytes`, `_response_epoch`）を**専用の小さな状態オブジェクト**（例: `@dataclass AgentRuntimeState`）にまとめ、読み書きは必ずロック経由。
- 外部から参照される `ai_speaking` / `in_echo_window` はプロパティでスナップショットを返す。
- 既存の `_lock`（`_pending`用）と統合するか役割分担を明確化。

### R2 — トリガー経路の単一化（Coordinator化）

- 介入トリガーの判断ロジックを **1スレッド（agent_worker）に集約**する。
- `_run_drift_checker` は「drift を検出したら `InterventionRequest(reason=...)` を `queue.Queue` に積む」だけにする。
- agent_worker が毎ループでキューを drain し、`_pending_intervention` のリトライ／drift要求／沈黙・カウントトリガーを**一箇所で**裁定して `trigger()` を呼ぶ。
- 結果: Fix 3 / Fix 4 が構造的に不要になり、二重発火の余地が消える。

### R3 — interrupt / cancel の状態遷移を明示的なステートマシンに

- 現状 `_interrupted` / `_responding` / `ai_speaking` の組み合わせで暗黙的に表現している状態を、`IDLE / RESPONDING / SPEAKING / INTERRUPTED / CANCELLING` のような明示的enumに整理。
- 各 Realtime イベント（output_item.added / audio.delta / transcript.done / audio.done / response.done / error）での遷移を表で定義し、コメントではなくコードで保証する。
- `_realtime.py` と `_partner.py` で重複している再生スレッド・声紋登録・truncate計算を**共通基底クラス**に抽出（DRY化、バグの片側修正漏れを防止）。

### R4 — タイミング定数の集約と可観測性

- `_echo_cooldown` / `_AGENT_*_SILENCE` / `_DRIFT_CHECK_*` / `_preflight_chars` 等を1箇所にまとめ、意味と相互関係をドキュメント化。
- 既存の `# [drift]` / `# [diag]` ログを構造化（state遷移ログ）し、回帰時に原因追跡できるようにする。

---

## 進め方・検証

- 各 Fix は小さなコミットに分け、Phase 0 のテストを回帰チェックに使う。
- Phase 3 は Phase 1-2 完了後に着手（バグ修正で挙動を安定させてから構造を動かす）。
- 受け入れ確認: 実機テストで「会話中の脱線 → 20秒以内に介入」「介入不要時に無音」「割り込み後の再介入が1回だけ」を確認。

## 推奨着手順

1. Phase 0（テスト基盤）
2. Fix 1 → Fix 2 → Fix 3
3. Fix 6 → Fix 4 → Fix 5
4. Phase 3（R1 → R2 → R3 → R4）
