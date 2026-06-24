# Phase 3 計画: 介入・割り込みロジックの構造リファクタ

ブランチ: `refactor/phase3-structure`
前提: Phase 1・2 + Fix 7〜11 がmainにマージ済み（`tests/unit/live` 51件パス）。

## 目的

Fix 1〜11 は「ガードを足して塞ぐ」対症的修正だった。Phase 3 では、今後ガードを
継ぎ足さなくても破綻しないよう、以下の構造的弱点を是正する。

1. 状態フラグ（`ai_speaking` / `_responding` / `_interrupted` / `_pending_intervention`
   / `_current_item_id` / `_played_bytes` / `_play_epoch` / `_last_noop_at`）が
   複数スレッドから無ロックで読み書きされている。
2. `RealtimeAgent` と `ConversationPartner` に重複コードが多い
   （`_q_put` / `_on_playback_terminator` / `_start_playback_thread` / `_recv_loop`
   / 声紋登録 / `close` / `set_tracker` / `_best_similarity` 等）。
3. interrupt/cancel の状態遷移が暗黙的（フラグの組み合わせで表現）。
4. タイミング定数が分散し、相互関係が不明瞭。

## 原則（Phase 1・2と同じ進め方）

- 各ステップは独立してマージ可能・テスト付き・小さなコミット。
- まず `tests/unit/live` を緑に保ったまま進める（リファクタの安全網）。
- 振る舞いを変えない「純粋リファクタ」と、振る舞いに関わる変更を分けてコミットする。
- 各ステップ後に立ち止まれる。実機確認を挟める。

---

## ステップ定義

### R4-先行: タイミング定数の集約 + 状態遷移ログ（低リスク・地固め）

R1/R2 のデバッグを楽にするため、観測性を先に上げる。

- `_constants.py` のタイミング定数（`_AGENT_*_SILENCE` / `_DRIFT_CHECK_*` /
  `_DRIFT_WARMUP` / `_STALL_*` / `_echo_cooldown` 等）を1ブロックに集約し、
  意味と相互関係をコメントで明記。
- `RealtimeAgent` / `ConversationPartner` の状態遷移（IDLE→RESPONDING→SPEAKING→
  INTERRUPTED→CANCELLING）を1関数で記録できる軽量ログを追加（`# [state]` 行）。
- 振る舞いは変えない。テスト: 既存51件が緑のまま。

### R1: エージェント状態を単一ロック配下に集約

- `RealtimeAgent` に `self._state_lock` を導入し、可変フラグ群を専用の小さな
  状態オブジェクト（`@dataclass _AgentRuntime`）にまとめる。
- 読み書きはロック経由。外部公開の `ai_speaking` / `in_echo_window` は
  スナップショットを返すプロパティにする。
- **デッドロック回避の鉄則**: ロック保持中に `ws.send()` やコールバック
  （`on_speech_start` / `on_ai_utterance`）を呼ばない。ロックはフラグの
  読み書きの最小区間だけ。
- まず `RealtimeAgent` のみ対象（`_partner` は R3 で基底クラス化する際に同じ仕組みに乗せる）。
- テスト: 既存の trigger/interrupt/cancel テストが緑のまま。
  競合の単体テスト（test-and-setの不可分性）を1〜2件追加。

### R2: トリガー経路の単一化（Coordinator化）

- 介入トリガーの判断を **`_run_agent_worker` 1スレッドに集約**。
- `_run_drift_checker` は「脱線を検出したら `InterventionRequest(reason=...)` を
  `queue.Queue` に積む」だけにする（自分では `trigger()` を呼ばない）。
- `_run_agent_worker` が毎ループでキューを drain し、
  ①中断介入のリトライ ②drift要求 ③沈黙/カウント ④stall-breaker
  を**一箇所で**裁定して `trigger()` を呼ぶ。
- 結果: Fix 3/Fix 4 の対症ガードが構造的に不要になる（二重発火の余地が消える）。
- テスト: drift検出→キュー投入→worker側でtrigger、の流れを
  FakeAgent/FakeStateで検証。既存のstall/retryテストを新構造に合わせて更新。

### R3: 共通基底クラスの抽出（重複解消）

- `_RealtimeBase` を新設し、共通実装を集約:
  - WebSocket接続・`_recv_loop` の骨組み
  - 再生スレッド `_start_playback_thread` + epochキュー（`_q_put` /
    `_on_playback_terminator`）
  - 声紋登録（`_try_enroll_*`）、truncate計算、`close`、`set_tracker`、
    `_best_similarity`、`in_echo_window`、R1の状態ロック
- `RealtimeAgent` / `ConversationPartner` は固有部分のみ保持:
  - `_handle`（イベント別処理）、セッション設定（VAD有無・プロンプト）、
    `feed`/`trigger`（facilitator固有）、`feed_audio`/`inject_context`（partner固有）
- 純粋リファクタとして実施（振る舞い不変）。テスト: 既存51件が緑のまま。
  これが最大の変更なので、R1/R2 で振る舞いを安定させた後の最後に行う。

### R4-本体: 構造化ログの仕上げ

- R4-先行で入れたログを整理し、`# [state]` / `# [trigger]` / `# [drift]` の
  粒度を統一。回帰時の原因追跡を容易にする。

---

## 推奨順序とリスク

```
R4-先行(観測性) → R1(ロック集約) → R2(トリガー単一化) → R3(基底クラス) → R4-本体(ログ整理)
```

- R4-先行は地固め（低リスク）。R1・R2 が中核（並行性の是正）。R3 は最大だが
  純粋リファクタなので振る舞い安定後に回す。
- 各ステップでテストを緑に保ち、要所で実機確認を挟む。
- いつでも途中で止めてマージ可能。

## やらないこと（スコープ外・別タスク）

- アジェンダ自動検出（会議冒頭でLLMが議題判定→更新）は別の機能開発として分離。
- STT/声紋まわりの変更は対象外（介入・割り込みロジックに限定）。
