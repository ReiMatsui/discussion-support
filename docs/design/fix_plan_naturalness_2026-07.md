# 修正計画書: 話者分離と介入挙動の自然さ改善（2026-07-02 深掘りレビュー対応）

実施者向け。着手前に必読: `docs/HANDOVER-2026-07-02.md`（環境・テスト実行・品質基準）、根拠は `docs/review-2026-07-02/06_speaker_attribution_deep.md` / `07_intervention_behavior_deep.md` / `08_deep_review_summary.md`。

**共通ルール**: コミット分割、各コミットで `tests/unit` 全通過＋`ruff check` クリーン＋mypy増分ゼロ（before/after比較）。regex・フラグ・特殊ケースの追加禁止。行番号は執筆時点のもの — 実装前に必ず現物を確認すること。

構成は2フェーズ。**Phase 1（4コミット）は対面パイロット前に必須**。Phase 2 は Phase 1 完了後に続けて実施してよいが、途中で止めても Phase 1 だけで出荷可能な状態を保つこと。

---

# Phase 1（必須）

## P1-1 `fix(live): 未確定話者の発話でもAIの発話を中断できるようにする (C1)`

### 問題
割り込み検出（`_workers.py` agent worker の feed 節）が `intervention_records`（未確定話者除外）を材料にしているため、声紋が確定しない参加者が「ちょっと待って」と遮ってもAIは話し続ける。AI発話との重畳時は新ラベル発行＋エコー窓中の蓄積抑止で帰属が最も壊れやすく、「遮る場面ほど止まらない」。

### 修正
`_run_agent_worker` の新規レコード処理節（現状 `talk_rs = [...]`（raw, AGENT/パートナー除外のみ）→ `new_records = intervention_records(talk_rs[state.agent_cursor:])`）で:
- **割り込み判定だけ raw スライスを使う**: `_raw_new = talk_rs[state.agent_cursor:]` を取り、`_human_spoke = any(len(str(r.get("text","")).strip()) > _INTERRUPT_MIN_CHARS for r in _raw_new)` に変更（発話の存在は話者不明でも確実。8文字超の相槌は稀なので bc 除外は不要）
- partner への割り込み・inject_context 判定（`_real_utterances`、_BACKCHANNEL_RE フィルタあり）も raw ベースに変更
- `agent.feed` と `_last_utt_time` 更新は従来どおり `intervention_records` ベースを維持（発話内容として使う側の信頼ポリシーは変えない）

### テスト
- speaker="?" の9文字以上の発話で `agent.interrupt()` が呼ばれる（FakeAgent の interrupt を記録可能にする。既存 FakeAgent.interrupt は pass なのでカウンタ追加）
- 未確定発話が `agent.feed` されないこと（従来挙動の維持）

---

## P1-2 `fix(live): 会議リセットで有効化済みの実名声紋を維持する (C2)`

### 問題
`VoiceProfiles.reset_session`（`_voice_profiles.py:191-208`）が `_active_keys` をAI声紋のみに縮小するため、**ユーザーがUIで有効化した実名プロファイルまで非活性化**される。照合は `_active_human()`（activeのみ）に対して行われるので、リセット後の新会議では全発話が誰とも照合されず、closed roster（auto=False）では全て未確定に落ちる（`:453-463`）。未確定は介入シグナルからも除外されるため、AIが実質盲目化する。対面実験の条件間リセットで確実に踏む。

### 修正
`reset_session` の `_active_keys` 縮小を「**匿名（ANON=人物N系）だけ落とす**」に変更:
```python
self._active_keys = {k for k in self._active_keys
                     if (k.startswith("__") and k.endswith("__"))
                     or not self.ANON.match(k)}
```
セッション由来の状態（sp_map / label_embs / pool / n_anon / own_sims）のクリアは従来どおり。docstring を「有効化済みの実名プロファイルは次の会議へ引き継ぐ（同じ参加者で会議を続けるのが通常のため）。匿名人物Nはセッション限りなので落とす」と更新。ANON が「人物N」と「#ラベル」の両方をカバーしているか実装前に確認し、漏れる形式があれば条件に含める。

### テスト
- 実名プロファイルを enroll→activate した状態で `reset_session()` → `_active_human()` に実名が残る／匿名 人物N は消える／AI声紋は残る
- リセット後、その実名話者の声（同一embedding）が classify で実名に一致する（closed roster: auto=False でも）

---

## P1-3 `feat(live): countを「価値判定つき整理介入」に置き換える (C3)`

### 問題
count は「10発話たまったら無条件に介入」（価値判定なし・cooldown 0）で、ファシリテータープロンプトは「『話すかどうか』は考えず…必ず述べよ」（`_constants.py:326-328`）。健全に流れている議論にも定期介入が入る「仕切りすぎ」の構造要因。系のどこにも「今は黙っているべき」という判断が存在しない。

### 修正方針（checker→候補→Controller の既存構造に載せる）
1. `_constants.py` に `_SUMMARY_VALUE_PROMPT` を追加。入力: 直近発話（12件程度）＋論点一覧。判定: 「**今、短い整理・要約の介入が議論に価値を足すか**」。true の目安: 論点が拡散して噛み合っていない／同じ主張の繰り返しが続く／決定が先送りされ続けている。false の目安: 議論が順調に深まっている・具体案の詰めに入っている・直前に整理があった。出力 `{"intervene": bool, "focus": "介入するなら焦点を短く"}`。**迷ったら false**（過剰介入の回避を明記）
2. `_bootstrap.py` に `check_summary_value(utterances, topics, api_key, model) -> dict`（既存 `_build_chat_params`＋`_post_chat_json` パターン、schema付き、temperature 0）
3. `_workers.py` に新checker `_run_structuring_checker`（1s tick）: `agent.pending_count >= agent.trigger_n` かつ前回判定時から発話が進んでいる時だけ LLM 判定（`_last_judged_count` をローカルに持ち再判定の連打を防ぐ）。intervene=true なら `state.summarize_requests`（Queue, SessionState に追加・resetでdrain）へ `{"focus": ...}` を積む
4. `_build_candidates`: `pending.summarize`（drainで保持）から kind="summarize" 候補を生成（brief=focus）。**count 候補の生成と dispatch 分岐は削除**。`_NORMAL_KINDS` の "count" を "summarize" に置換
5. `_facilitation.py` `_KIND_POLICY`: `"summarize": _KindPolicy(4, _INTERVENTION_PAUSE_COUNT相当(1.5), 30.0, 2000, "wait_for_pause", "global")`（同種連発防止に kind cooldown 30s、global scope で他介入直後も抑制）。"count" エントリは削除
6. dispatch: summarize 採択時は `agent.trigger(topics=..., summary_focus=focus)`。`_realtime.py` の trigger に summary_focus 用のコンテキスト節を1つ追加（「議論の整理が求められています。焦点: {focus}。一言で整理してください」程度）
7. `_PROMPT_FACILITATOR` は上流で価値判定済みになるため大枠維持でよいが、「価値を足す最小限の発言に留めてください」の直後に「**足すべき価値が薄いと感じたら、無理に整理せず一言の相槌程度に留めて構いません**」を追加（最後の安全弁）
8. 観測: summarize の trigger ログ・review ログは既存の仕組みに自然に乗る（kind が変わるだけ）。`check_summary_value` の false 判定もログに1行残す（`# [structuring] skip: 介入価値なし`）— 「なぜ黙ったか」の追跡は本研究の核

### テスト
- pending_count が trigger_n に達しても check_summary_value=false なら trigger されない（旧 count テストの反転）
- true なら summarize として発火し、intervention_events の reason が "summarize"
- 同じ pending_count で LLM 判定が繰り返されない（_last_judged_count）
- リセットで summarize_requests が drain される
- 既存の count 前提テスト（test_agent_worker の count 系）を summarize 前提に書き換え

### 注意
- silence（18秒沈黙）トリガーは**変更しない**（沈黙自体が既に意味のあるシグナル）
- conversation モードは対象外（従来どおり）

---

## P1-4 `feat(live): 呼びかけへの即時アック音 (H)`

### 問題
「AIさん、整理して」から音声応答まで3〜7秒（STT確定＋triage LLM＋pause＋生成）。その間のフィードバックは画面ステータスのみで、話者は「聞こえたのか」が分からず言い直す→二重呼び出しになる。

### 修正
- triage worker が manual_call(source=voice) を enqueue した直後に、**短いアック音（チャイム）**を鳴らす。`_workers.py` に `_play_ack_chime()`: sounddevice で 150ms 程度の減衰サイン波（880Hz, 音量控えめ）をワンショット再生。失敗は握りつぶし（`contextlib.suppress(Exception)`、音は必須機能ではない）。別スレッド不要（150msブロックは許容）だが、triage ループを止めないよう `threading.Thread(daemon=True)` で投げてよい
- UI 由来の manual_call（ボタン）はUIに既にフィードバックがあるため鳴らさない
- 定数 `_ACK_CHIME_ENABLED = True` を `_constants.py` に置き、無効化できるようにする（実験条件によっては音を消したい）
- **本格的な二段応答（「はい」と声で応じてから本応答）は今回やらない**（Realtime API の応答2連発はエコー・状態管理の複雑化を招く。チャイムで「聞こえた」は伝わる）
- 既知の限界をコードコメントに明記: triage バックログ超過時（>8件）に古い呼びかけが分類されず落ちる経路は残る（バックログ時は警告ログ済み。S2/S3計測で頻度を見る）

### テスト
- voice 呼びかけ enqueue でチャイム関数が呼ばれる（monkeypatchで記録）／UI 由来では呼ばれない／チャイム失敗が例外を漏らさない

---

# Phase 2（Phase 1 完了後）

## P2-1 `fix(live): エコー判定をAI再生区間との重なりベースにする`
テキスト安全網とエコー窓中の蓄積抑止のゲートが「flush時の壁時計」（`in_echo_window`/`ai_speaking`）基準のため、STT確定が遅れて窓の外に出た回り込みが両防御を素通りする（16:42障害の残余経路）。修正: SessionState にAI再生区間の記録 `note_ai_speech_interval(start, end)` を追加（agents の再生開始/終了フックから、**マイク音声のmsタイムライン**で記録。`asr_pcm_buf_offset`＋buf長から現在msを算出するヘルパーを SessionState に置く）。`_recv_loop.flush` のエコー系ゲートを「発話区間 `[cur_ms, cur_end]` がAI再生区間（±300ms マージン）と重なるか」に置換。壁時計判定はフォールバック（ms が無い場合）として残す。テスト: 窓を過ぎてflushされた重なり発話がエコー判定されること。

## P2-2 `fix(live): エコー窓中も声紋照合は行い、蓄積・登録だけ止める`
`classify` の `count=False`（`_voice_profiles.py:374` の `elif count and ...`）は照合ごとスキップするため、AI発話直後2秒に集中する人間の返答が声紋補正なしのラベル追従になる。修正: `classify` に `enroll: bool = True` を追加し、埋め込み計算・照合（AI声紋チェック含む）は `count` 条件から分離して常行、蓄積ブロック（`if self.auto and chars > 0:` 節と `label_embs` への追加）だけを `enroll` でゲート。`_recv_loop` は `count=not _is_backchannel, enroll=(not _is_backchannel) and not _ai_active` を渡す。テスト: エコー窓中の人間発話が声紋一致で正しい実名になる／蓄積は増えない。

## P2-3 `fix(live): リトライを「本当に途切れた発話の再開」に限定する`
(a) 中断時のリトライ素材が未再生 transcript を含む問題: `interrupt()` で `_pending_intervention` を作る際、`_played_bytes` 由来の再生率が極端に低い（例: 10%未満 = ほぼ何も聞こえていない）場合はリトライでなく破棄する。(b) 中断注記が次の異種トリガー（fact等）に混入して1発話2意図になる問題（`_realtime.py` trigger 内の中断コンテキスト節）: 中断内容の再開は reason=retry の trigger のみに含め、他種別のトリガーでは含めない。(c) 会話が進んだ後の蒸し返し防止: interrupt 後に `feed` された発話数を数え、4発話を超えたら `_pending_intervention` を破棄（TTL 60s と併用）。テスト各分岐。

## P2-4 `fix(live): パートナー切断後もエコー参照を短時間保持する`
`_detach_partner`（`_workers.py:1398-1406`）で `state.partner=None`→`close()` するとテキスト・声紋両方のエコー防御が即消える。修正: SessionState に `retired_echo_texts: deque[(monotonic, text)]`（TTL 10秒）を追加し、detach 時に partner の `_recent_ai_texts` を積む。`_recv_loop` のテキスト安全網は TTL 内の retired テキストも照合対象に含める（類似度関数は agents/_base の `_best_text_similarity` を再利用）。声紋 `__PARTNER__` は既に tracker 側に残るため対応不要（確認すること）。

## P2-5 `fix(live): 事前登録に品質ゲートを入れる`
`/api/enroll`（`_ui.py` の enroll エンドポイント）が (a) AI発話中でも実行でき、(b) 音声区間の検査なしで末尾N秒を平均する。修正: enroll 実行時に AI が発話中/エコー窓中（P2-1 の区間記録があればそれで判定）ならエラーを返す（「AIの発話が終わってからもう一度お願いします」）。加えて録音セグメントの実効長（無音を除いた長さ）が下限未満なら reject。UI にエラー文言を表示。テスト: AI発話中の enroll が拒否される。

---

## 実施順序まとめ

| # | コミット | 規模 |
|---|---|---|
| P1-1 | 未確定話者の割り込み | 小 |
| P1-2 | リセットで実名声紋維持 | 小 |
| P1-3 | count→価値判定つきsummarize | 中 |
| P1-4 | 呼びかけアック音 | 小 |
| P2-1 | エコー判定の区間ベース化 | 中 |
| P2-2 | 照合と蓄積の分離 | 中 |
| P2-3 | リトライの限定 | 中 |
| P2-4 | detach後のエコー参照保持 | 小 |
| P2-5 | 事前登録の品質ゲート | 小 |

## スコープ外（測ってから／別計画）
- PCMベースの簡易VAD（endpoint跨ぎの思考ポーズ対策の根治）— S3 計測で被り率が基準超過なら着手
- invite 後の「返答を待つ・拾う」状態機械、介入後フォローアップ、宛先明示の強化 — AF×ライブ統合（H1）の設計と一体で扱う方が本質的なため、そちらの計画に含める
- 本格的な二段応答（声のアック）
