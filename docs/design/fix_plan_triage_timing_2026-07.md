# 修正計画書: triage統合の運用課題と介入タイミングの下地修正（2026-07-02）

実施者向けの作業指示書。**着手前に必読**: `docs/HANDOVER-2026-07-02.md`（環境の罠・テスト実行コマンド・品質基準）と `docs/review-2026-07-02/01_live_pipeline.md`（背景）。

## 目的

`docs/research/near_term_validation_plan_2026-07.md` の検証（S1〜S3）を始める前に、実運用で確実に踏む問題を潰す。対象は5件。修正1・2は直近の regex→LLM分類置換（コミット c453fe9）が持ち込んだ問題、修正3は既知レース（レビューH2）のtriageへの波及、修正4はタイミング計測の下地、修正5は仕様変更（ユーザー合意済み: **未確定話者の呼びかけも拾う**）。

**共通ルール**: コミットは修正単位で分割。各コミットで `tests/unit` 全通過＋`ruff check` クリーン＋mypy strictエラー数が変更前から増えないこと（絶対数はもともと大きい。before/after比較で確認）。対症療法（regex・フラグ・特殊ケース）を足さない。

---

## 修正3を最初に実施: meeting epoch によるリセット競合ガード

（修正1・2がtriageループを書き換えるため、土台のこれを先にやる）

### 問題
「新しい会議」リセット（`_session_state.py` `reset_for_new_meeting`、カーソル類を0に戻す）と、各workerの無ロックなカーソル書き戻し（例: `_workers.py` triage の `state.triage_cursor = idx + 1`、agent worker の `state.agent_cursor = n`）が競合する。workerがリセット直前のスナップショットから計算した古いカーソル値をリセット後に書き戻すと、新会議でカーソルが発話数を超え、`idx >= n` で永久待機。**triage が止まると fact checker が注釈待ちで連鎖停止**するため、新会議で fact補正・音声呼びかけが全滅する。

### 修正方針
1. `SessionState.__init__` に `self.meeting_epoch = 0` を追加。`reset_for_new_meeting` の `with self.state_lock:` ブロック内で `self.meeting_epoch += 1`
2. カーソルを持つ全worker（`_run_agent_worker` / `_run_topic_worker` / `_run_drift_checker` / `_run_fact_checker` / `_run_triage_worker`）で次のパターンを徹底:
   - スナップショット取得時（既存の `with state.state_lock:` 内）に `epoch = state.meeting_epoch` を一緒に読む
   - **副作用（agent.feed / キュー投入 / カーソル書き戻し）の直前**に `with state.state_lock:` で `state.meeting_epoch == epoch` を確認。ずれていたらそのtickの結果を全て破棄して `continue`
   - カーソル書き戻し自体も state_lock 内で行う
3. FakeState（`tests/unit/live/test_agent_worker.py` ほか）に `meeting_epoch = 0` を追加。**`getattr(state, "meeting_epoch", 0)` のような防御は書かない**（レビューM7: 暗黙インターフェースの禁止）

### テスト
- 機能回帰: worker実行中に `reset_for_new_meeting()` を呼び、その後の新規発話が処理される（triage注釈が付く / agent.feed される）ことを実SessionStateで確認
- 単体: epoch不一致時にカーソルが書き戻されないこと（スナップショット後にepochを手で進めるフェイクで検証）

### 受け入れ基準
リセット後に投入した発話が3秒以内にtriage注釈・agent feedされる（テストで until 待ち）。既存テスト全通過。

---

## 修正1＋2: triageループの再構成（1コミットで実施）

### 問題1: 復帰後のバックログ誤発火
`_workers.py:944-948` — 介入オフ / conversationモード中は `continue` でカーソルが止まる。復帰すると溜まった過去発話を順にLLM分類し、数分前の呼びかけを `created_at=time.monotonic()`（=今）で manual_call に積む（`:1001-1008`）。TTL30秒が無効化され、AIが古い依頼に突然応答する。復帰直後はLLM呼び出しのバーストも起きる。

### 問題2: スループット上限
1 tick（0.25s）で1発話しか処理しないため、処理速度 ≈ 1発話/(0.25s＋LLM往復) ≈ 1〜2秒/発話。活発な会話で恒常的にバックログが伸び、fact・呼びかけの遅延が有界でない。

### 修正方針（ループ全体をこう作り直す）
```
毎tick:
  key/agent/enabled チェック（従来どおり continue）
  if 介入オフ or conversationモード:
      未処理分[cursor..n) を LLM を呼ばずに一括で負注釈
      {"factual_claim": False, "facilitator_request": "", "skipped": "intervention_off"}
      を付け、cursor = n（epoch確認の上で）。キュー投入なし。continue
  スナップショット取得（epoch付き）
  backlog = n - cursor
  if backlog > _TRIAGE_BACKLOG_MAX:
      古い (backlog - _TRIAGE_BACKLOG_MAX) 件を負注釈
      {"skipped": "backlog"} でスキップし cursor を進める（1回だけ警告ログ）
  残りを内側ループで連続処理（tickあたり最大 _TRIAGE_BACKLOG_MAX 件）:
      各件: 従来どおり 最小文字数ゲート → LLM分類 → 注釈 → cursor+1 → 呼びかけenqueue
      retryable_error は従来どおり同一発話でbreakして次tickへ（リトライ上限も従来どおり）
      各件の間で state.stop / epoch を確認（stop時即座に抜ける）
```
- `_constants.py` に `_TRIAGE_BACKLOG_MAX = 8` を追加（コメント: 遅延を有界にする。8件×2秒≈最悪16秒の追いつき時間）
- 注釈dictに `"skipped"` キーが増えるのは fact checker に影響しない（`factual_claim` False として扱われる）ことを確認
- **呼びかけ enqueue の鮮度ガード**: バックログ処理中でも、enqueue するのは分類した record が「スナップショット末尾から `_TRIAGE_BACKLOG_MAX` 件以内」のもののみ…とはせず、上のbacklogスキップで古い分は分類自体されないため追加ガード不要。設計をこれ以上複雑にしない

### テスト
- 介入オフ中に発話を積む → 負注釈（skipped=intervention_off）が付き、classify が呼ばれず、manual_call_requests が空のまま。再有効化後の新規発話は通常分類される
- conversationモードでも同様
- 10件積んで `_TRIAGE_BACKLOG_MAX=8`（monkeypatchで小さくしてよい）→ 古い2件が skipped=backlog、残りが分類され、最新発話の呼びかけは発火する
- 1 tickで複数件処理されること（fake classifyで時間を止めて件数を確認）
- 既存の triage テスト（test_human_mode.py）が意味を保ったまま通ること

### 受け入れ基準
「介入オフ→5分後に復帰」を模したテストで、過去の呼びかけが発火しない。バックログがあっても最新発話の呼びかけ遅延が有界（テストでは件数で担保）。

---

## 修正5: 未確定話者の呼びかけを拾う（仕様変更・ユーザー合意済み）

### 問題
triage の入力が `intervention_records`（`_speaker_policy.py`）のため、未確定話者（`?` / `未確定`）の発話が分類対象にすら入らず、声紋未登録の参加者は何度AIを呼んでも無言で無視される。呼びかけは話者が誰かに依存しない操作なので、これは不自然。

### 修正方針
1. `_speaker_policy.py` に追加:
   - `is_triage_signal(record)`: `is_intervention_signal` から**未確定話者の除外だけを外した**判定（text非空・bcでない・相槌regexに一致しない）。docstringに「呼びかけ検出は話者同一性に依存しないため未確定を含める。fact/drift/invite の材料としての利用は従来どおり `is_intervention_signal` 側で制限される」と明記
   - `triage_records(records)`: 上記でフィルタ
2. `_run_triage_worker` のスナップショットを `intervention_records` → `triage_records` に差し替え（コンテキスト部も同じリストを使う）。話者名は既存の `intervention_speaker_name` が未確定を汎用名「発話者」に落とすのでそのまま
3. fact checker は従来どおり `intervention_records` を使う（未確定話者の factual_claim 注釈は付くが消費されない。これは意図どおり — 帰属不明の断定に訂正を打つと誤爆リスクがあるため）
4. **注意**: triage_cursor のインデックスは新フィルタのリストに対するものになる。fact_cursor（intervention_recordsベース）とはリストが異なるが、fact checker は record オブジェクトの `triage` キーを直接読むため整合する。ここを崩さないこと（インデックスの共有をしない）

### テスト
- speaker="?" の「AIさん、ここまで整理して」→ manual_call(source=voice) が積まれる
- speaker="?" の事実断定 → triage で factual_claim=True が付いても fact checker が check_fact を呼ばない
- 相槌（bc / _BACKCHANNEL_RE 一致）は未確定話者でも従来どおり除外

### 受け入れ基準
未確定話者の呼びかけが発火し、未確定話者の fact 訂正は発火しない。

---

## 修正4: partial受信で「沈黙」タイマーを更新（最後に実施）

### 問題
`_last_utt_time` はSTT**確定**レコードの観測時のみ更新される（`_workers.py:1381`）。長い発話の途中では確定が来ないため「喋っている最中に沈黙が伸び」、pause判定を満たした介入が発話に被さり得る（レビューM6）。検証計画の被り率計測の前に入れないと数字が歪む。

### 修正方針
- `SessionState.show_partial` 内（`_session_state.py`。partial文字列を保持している場所）で、**partialテキストが非空かつ前回のpartialから変化した場合のみ** `self._last_utt_time[0] = time.monotonic()` を更新する。「変化した場合のみ」なのは、同一partialの再送で沈黙が永久に0に張り付くのを防ぐため
- 実装はrecv_loop側でなく show_partial 側に置く（単体テストしやすく、呼び出し元が増えても一貫する）
- 既知のトレードオフをコメントに明記: エコーウィンドウ中のAI自身の声のpartialでもタイマーが更新され得るが、フロア判定を保守側（介入を待つ側）に倒すだけなので許容する

### テスト
- show_partial("...", 非空) でタイマーが更新される / 同一文字列の再送では更新されない / 空文字では更新されない
- 既存のworkerテストで沈黙依存のもの（retry/silence系）が壊れないこと（FakeStateはshow_partialを通らないため影響しない想定。壊れた場合は原因を確認してから直す）

### 受け入れ基準
上記単体テスト＋既存テスト全通過。

---

## 実施順序とコミット構成

| # | コミット | 内容 |
|---|---|---|
| 1 | `fix(live): meeting epochでリセット競合をガード (H2)` | 修正3 |
| 2 | `fix(live): triageのバックログ制御と復帰時の誤発火防止` | 修正1＋2 |
| 3 | `feat(live): 未確定話者のファシリテーター呼びかけを拾う` | 修正5 |
| 4 | `fix(live): partial受信で沈黙タイマーを更新 (M6)` | 修正4 |

各コミット後: `tests/unit` 全通過（サンドボックスなら HANDOVER 記載の unshare コマンド、Mac なら `uv run pytest -q`）＋ `ruff check src/das/asr/live tests/unit/live`。最後に mypy の before/after 比較（`--python-version 3.12`、エラー数が増えていないこと）。

## やらないこと（スコープ外）
- 話者分離の閾値・ポリシー統合（M3/M9）— 検証で基準を割ってから
- epoch/deadline の本実装（H5）、採否込みリプレイ（M5）— 事象が観測されてから
- VAD導入 — 修正4で不足だった場合の次の手
- triage の並列LLM呼び出し化 — `_TRIAGE_BACKLOG_MAX` で足りなければ検討
