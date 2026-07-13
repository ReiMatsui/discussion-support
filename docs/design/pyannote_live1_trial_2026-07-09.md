> 作成: Claude Fable 5 (claude-fable-5), 2026-07-09

# pyannoteAI Live-1 検証: バッチ diarization オフラインベンチマーク

実施者向け。背景: pyannoteAI が Live-1（ストリーミング話者分離。バッチ (`precision-2`) と精度同等と主張）を出した。本システムは対面議論を Soniox STT ＋声紋照合（`src/das/asr/live/_voice_profiles.py` 等）で話者帰属しており、外部 diarization を差し込むフックとして `--diarization pyannote|assemblyai` が既にある（`src/das/asr/live/_pyannote_diarization.py` はストリーミングWS実装、`src/das/asr/live/_diarization.py` が統合・評価ロジック）。Live-1へ本格移行する前に、**まずバッチAPIで過去録音を分離し、現行の話者タイムライン（`transcripts/*.turns.jsonl`）とどれだけ食い違うか**をオフラインで確認し、乗り換えの価値があるかの一次判断材料を作る。

## 0. 対象と非対象

**やること**: `transcripts/*.wav`（16kHz mono 前提。実データは概ね16kHz/mono/16bitだったが、セッションによって異なる可能性があるため `scripts/benchmark_pyannote.py` 実行時に自動で警告する）を pyannoteAI のバッチ `POST /v1/diarize` に投げ、結果を現行 `turns.jsonl` と突き合わせる。

**やらないこと**: Live-1（`POST /v1/live` WebSocket）を実際に叩く比較（今回はバッチのみ）。ライブ本統合（`--diarization pyannote` の中身をLive-1に差し替える）そのものは別タスク（本ドキュメント末尾に見取り図のみ記載）。

## 1. pyannoteAI API 要点（docs.pyannote.ai 調査、2026-07時点）

- **認証**: `Authorization: Bearer <API_KEY>` （JWT bearer）。既存コード (`src/das/asr/live/_bootstrap.py:565`) は環境変数名 `PYANNOTEAI_API_KEY` を使っている。本リポジトリの `.env` にも既にキーが入っている（`PYANNOTEAI_API_KEY=sk_...`）。`scripts/benchmark_pyannote.py` はこの名前を優先して読み、依頼書指定の `PYANNOTE_API_KEY` があればフォールバックとして使う。
- **ジョブ投入**: `POST https://api.pyannote.ai/v1/diarize`
  - body: `{"url": "<公開URL または media://object-key>", "numSpeakers"?: int, "minSpeakers"?: int, "maxSpeakers"?: int, "model"?: "precision-2"|"community-1"}`
  - レスポンス: `{"jobId": "...", "status": "created"}`
  - `numSpeakers` を指定すると精度が上がるとドキュメントに明記あり（未指定なら自動推定）。今回の対話音声は概ね2〜4名程度で recording ごとに既知なので、指定できるセッションでは指定する。
- **ローカルファイルのアップロード**（音声が公開URLでない場合、今回はこちら）:
  1. `POST /v1/media/input` body `{"url": "media://<object-key>"}` → `{"url": "<presigned PUT URL>"}`
  2. その presigned URL へ生バイトを `PUT`（アップロードは48時間で自動削除、チーム内のみアクセス可）
  3. diarize リクエストの `url` に `media://<object-key>` を指定
- **結果取得（ポーリング）**: `GET /v1/jobs/{jobId}`
  - `status`: `pending|created|running|succeeded|failed|canceled`
  - `succeeded` になったら `output.diarization` に `[{"speaker": "SPEAKER_00", "start": 15.0, "end": 30.5}, ...]`（秒単位）が入る。
  - **結果は完了後24時間で自動削除**されるため、自前で保存する必要あり（`scripts/benchmark_pyannote.py` は `transcripts/<session>.pyannote_bench.json` に保存する）。
  - Webhookも使えるが、今回はオフラインバッチ検証なのでポーリングで十分。
- **課金**: 成功したジョブのみ課金。`/diarize` は音声秒数課金、**最低20秒課金**。Live-1 (`/live`) も秒数課金で同様の最低課金がある。詳細レートはダッシュボード次第（後述の費用試算を参照）。

## 2. 使い方

### 2.1 前提: APIキー取得

1. https://dashboard.pyannote.ai でアカウント作成（本リポジトリの `.env` に既に `PYANNOTEAI_API_KEY` が入っているため、多くの場合これを使えばよい。失効/別チームの場合はダッシュボードの API Keys から再発行）。
2. `export PYANNOTEAI_API_KEY=sk_...`（`.env` を読み込む実行系ならそちらでも可）。

### 2.2 実行例（3セッション分）

```bash
# 話者数が既知なら --num-speakers を付けると精度が上がる
uv run python scripts/benchmark_pyannote.py \
  --session 2026-06-25_1543 \
  --session 2026-06-29_2145 \
  --session 2026-06-24_1518

# 話者数不明でOK（自動推定）
uv run python scripts/benchmark_pyannote.py --session 2026-06-25_1543
```

wav パスを直接渡すことも可能（同名の `.turns.jsonl` を自動解決）:

```bash
uv run python scripts/benchmark_pyannote.py --wav transcripts/2026-06-25_1543.wav
```

APIキーなしでロジックだけ確認する場合:

```bash
# モックJSON（POST /v1/diarize output か diarization配列そのもの）を使う
uv run python scripts/benchmark_pyannote.py --session 2026-06-25_1543 --dry-run --from-json mock.json

# モックJSONも用意せず、比較ロジックの配線だけ確認（合成モックを内部生成）
uv run python scripts/benchmark_pyannote.py --session 2026-06-25_1543 --dry-run
```

出力は標準出力のサマリと `transcripts/<session>.pyannote_bench.json`（詳細: 話者マッピング・不一致ターン全件・pyannote生セグメント全件）。複数セッション指定時は全体サマリ（マイクロ平均・セッション平均の一致率、話者数一致セッション数、`未確定`区間の分離状況）も出る。

### 2.3 データの注意

- `transcripts/*.turns.jsonl` は途中でスキーマが変わっている。古いセッション（例: `2026-06-12_*`）は `end_ms` を持たず `ms`（開始時刻）のみ。`end_ms` がないターンは pyannote と時間比較できないため比較対象から除外される（`n_turns_timed` / `n_turns_compared` の差として出力に出る）。**比較セッションを選ぶときは `end_ms` が入っている＝比較的新しいセッションを優先すること。**
- `AI` / `パートナー` / `ファシリテーター` / `[Partner]` は音声を伴わない合成発話（`ms`が`null`）なので自然に比較対象外になる。
- 現行が話者を特定できなかった区間は `未確定`（セッションによっては別ラベルの可能性もあるため `--unknown-label` で変更可）というラベルで記録されている。

## 3. 判断基準

主要指標は2つ:

1. **ターン単位一致率**（貪欲法でpyannote話者ラベル→現行話者ラベルに最適1対1マッピングした上での、`end_ms`が確定しているターンのうち現行話者ラベルと一致した割合）。
2. **重なり発話区間・短発話（相槌等）での差**（不一致ターンリストの `overlap_ratio` とテキスト内容から、相槌の取りこぼしや割り込み時の帰属の差を確認）。

判断の流れ:

- **一致率が高い（目安95%以上）セッションが大半**: 既存の声紋照合と同等以上の可能性が高い。乗り換えの実益は薄いか、`未確定`区間の解消効果（下記）次第。
- **一致率がある閾値（目安80%）を下回るセッションが一定数ある**: その不一致ターン（`mismatches`）を人手で聴取確認する。
  - pyannote側の判定が正しい事例（現行の声紋照合ミス）が多数 → **乗り換え/併用を検討**（特に声が似ている話者ペア、長時間セッションでの声紋ドリフトに強いかがポイント）。
  - pyannote側もズレている、あるいは現行が正しい事例が多数 → 乗り換えの根拠なし。特にオーバーラップ発話・短い相槌（「うん」「そう」等）はdiarization全般が苦手なので、`overlap_ratio`が低い（重なりが薄い＝短発話や境界付近）ケースの傾向を見る。
- **現行が `未確定` 扱いにした区間**（`unresolved_findings`）: pyannoteが自信を持って単一/複数話者に割れているか（`split_into_multiple` と `covered_ratio`）を確認する。ここが大きく改善するなら、たとえ全体一致率が同程度でも「声紋が取れなかった区間の救済」という明確な価値がある。
- **話者数の一致**（`speaker_count_matches`）: pyannoteが数を過大/過小推定していないか。ずれている場合は `--num-speakers` 指定での再実行で改善するか確認する。

**推奨する検証セット**: 最低3セッション、できれば「参加者が少なく綺麗に録れているセッション」「参加者が多い/声が近いセッション」「長時間で声紋ドリフトが疑われるセッション」を1件ずつ選び、上記指標を比較する。

## 4. Live-1 本統合する場合の差し替え箇所（見取り図）

現状のライブ音声パイプラインでの外部diarizationの位置づけ:

```
STT (Soniox, 高精度日本語認識)  ──┐
                                  ├─ SpeakerResolver.resolve() (_diarization.py)
声紋照合 (_voice_profiles.py)  ──┤    優先順位: 1.声紋高信頼 → 2.外部diarization → 3.STTラベル
                                  │
外部diarization provider ────────┘
  (DiarizationProvider Protocol: start/send_audio/drain_events/active_events/close)
  実装: _pyannote_diarization.PyannoteStreamingDiarizationProvider (WS: POST /v1/live)
        _assemblyai_diarization.py (同型)
```

Live-1 は既に `_pyannote_diarization.py` の `PyannoteStreamingDiarizationProvider` が `POST /v1/live` を呼ぶ形で実装済み（`--diarization pyannote` で有効化）。バッチ検証の結果が良好なら、**新規実装は不要**で以下の確認だけが残る:

1. `_pyannote_diarization.py` の WS メッセージパース（`diarization_speaker_start` / `diarization_speaker_end`）が現行のLive-1 APIレスポンス仕様と一致しているか（ドキュメント更新の可能性があるため実運用前に軽く突き合わせ）。
2. `SpeakerResolver`（`_diarization.py`）の `diarization_min_overlap` 等の閾値は Speechmatics/AssemblyAI 想定で調整されている可能性があるため、pyannoteの出力特性（境界の遅延・粒度）に合わせた再チューニングが必要か、本ベンチマークの不一致傾向（境界付近のズレ量）から見積もる。
3. `score_diarization`（`_diarization.py`）を使ったオンラインでのA/B（現行 vs pyannote-live1）は別途、実セッションでの並走記録が必要（本書のオフラインベンチマークはそのための事前スクリーニング）。
4. コスト影響（下記）を踏まえ、既定を`--diarization none`のままにするか`pyannote`に倒すかを判断。

## 5. 費用試算

pyannoteAIの正確な単価はダッシュボードの契約プラン依存だが、公開情報ベースの目安は **概ね €0.17〜0.20 / 時間**（diarizationの音声処理時間課金、最低20秒/ジョブ）。想定利用パターン:

- 1セッション平均60分・1日5セッション運用と仮定 → 約5時間/日 → **概算 €0.85〜1.00/日**（月20営業日で €17〜20/月）。
- Live-1（ストリーミング）も同様の秒数課金体系（最低20秒/セッション課金）のため、バッチと大きな価格差は出ない想定。ただし実際の契約単価はダッシュボードの Billing ページを都度確認すること（Developer/Starterプランには月間無料枠あり）。
- 声紋照合（自前・無料）で十分な精度が出ている場合、上記コストに見合う精度改善（特に「未確定」区間の解消・声紋ドリフト耐性）があるかが乗り換え判断の分岐点になる。

## 6. 次のアクション

1. 本ドキュメントの手順で最低3セッションをベンチマーク実行し、`transcripts/*.pyannote_bench.json` を確認。
2. 一致率が低いセッションの `mismatches` を数件、実音声で聴取確認（どちらが正しいかの一次判定）。
3. 結果を踏まえ、乗り換え/併用/現状維持を判断し、必要なら `--diarization pyannote` のライブ本番投入計画（上記4節）に進む。

## 7. Live-1 実測手順（2026-07-09追記）

上記4節の「1. WSメッセージパースが現行API仕様と一致しているか」を確認するため、
`src/das/asr/live/_pyannote_diarization.py` を pyannoteAI Live-1 の正式仕様
（`docs.pyannote.ai/tutorials/streaming-real-time` および
`docs.pyannote.ai/api-reference/{create-stream,streaming}`）に合わせて更新した
（16kHz mono pcm_f32le・100ms固定チャンク送信、`end_of_stream` 終了シーケンス、
`error` イベントのログ出力）。既存の録音wavを実際にWSへ流して実測するための
スクリプトを `scripts/test_pyannote_live.py` として新規追加した。

### 7.1 実行コマンド例

```bash
# transcripts/2026-06-25_1554.wav + .turns.jsonl を解決し、先頭5分だけ実時間で流す
uv run python scripts/test_pyannote_live.py --session 2026-06-25_1554

# 先頭2分だけ（動作確認を素早く回したい時）
uv run python scripts/test_pyannote_live.py --session 2026-06-25_1554 --head-minutes 2

# wavを直接指定
uv run python scripts/test_pyannote_live.py --wav transcripts/2026-06-25_1554.wav
```

標準出力に受信イベントが `[mm:ss] SPEAKER_XX ...` 形式で逐次表示され、終了時に
`transcripts/<session>.pyannote_live.json`（確定した話者区間一覧）を保存し、
流した範囲の `turns.jsonl` と突き合わせたサマリ（`scripts/benchmark_pyannote.py`
の `compare_session` 等を再利用）を表示する。Ctrl-C で安全に中断でき、
その場合は `end_of_stream` を送ってサーバの残り確定イベントを受け切ってから
接続を閉じる（送信済みぶんまでの結果は保存・比較される）。

### 7.2 実時間制約と頭出し推奨

Live-1 はサーバ側が「実時間 + 最大5秒バッファ」までしか先行受信を許容しない
仕様のため、本スクリプトは100msチャンクをwall-clockで100ms間隔にペーシングして
送信する。**録音時間分だけ実時間がかかる**（26分のセッションなら26分待つ）ため、
動作確認や一次スクリーニングでは既定の `--head-minutes 5`（先頭5分のみ）を
推奨する。長時間セッションでの声紋ドリフト耐性など、頭出し5分では判断できない
観点を確認したい場合のみ `--head-minutes` を伸ばすか `0`（全編）を指定する。

### 7.3 費用目安

Live-1もバッチ (`/diarize`) と同様の秒数課金体系で、目安は**概ね
€0.17〜0.20/時間**（最低20秒/セッション課金、詳細は5節参照）。頭出し5分の
実測なら1回あたり数セント程度に収まる想定。

### 7.4 バッチベンチのモデルは再実行不要（調査結果）

`scripts/benchmark_pyannote.py` が使っているバッチ diarizationモデル
（既定 `precision-2`）は、2026-07-09時点の `docs.pyannote.ai/models` 確認でも
引き続き最新・最高精度モデルだった。すなわち**過去に実行済みのバッチベンチ結果は
モデル陳腐化が理由の再実行は不要**（3節「使い方」参照）。今回の作業対象は
あくまでLive-1（ストリーミング）側のプロトコル実装確認・実測であり、バッチ側の
やり直しではない。

### 7.5 判定の見方

`compare_session` が返す指標の読み方は3節と同じ（ターン単位一致率・話者数一致・
`未確定`区間の解消状況）。Live-1実測では追加で次を確認する:

- **境界タイミングの遅延**: ストリーミングは确定イベント（`diarization_speaker_end`）
  がバッチより数百ms〜数秒遅れて届く特性があるため、`overlap_ratio` が低い
  不一致が頭出し範囲の終端付近に集中していないか確認する（範囲外に及ぶターンは
  `clip_turns_to_range` で比較対象から自動除外しているが、境界直前のターンは
  残るため誤差として出うる）。
- **頭出し5分での話者数**: セッション全体の話者数と、先頭5分だけの話者数は
  一致しないことがある（後半から参加する話者がいる等）。`speaker_count_matches`
  が不一致でも、単に「まだ登場していない」だけの可能性がある点に注意する。
- バッチとの精度同等性そのものを厳密に検証したい場合は、`--head-minutes 0`
  （全編）で1〜2セッション流し、`scripts/benchmark_pyannote.py` の同一セッション
  結果と一致率・話者マッピングを見比べる。

---

## 8. 実測結果と判定（2026-07-09、Claude Fable 5 記載）

### 実測サマリ
- **バッチ（precision-2）**: 実会議2本で一致率27.9%／27.4%。pyannoteは3-4話者に集約、現行は6-8ラベル。未確定区間の回収力は高い（1614: 150件中70件を分離、1554: 28件中14件）
- **ライブ（Live-1）**: 5分間の実時間ストリーミング成功（話者イベント326件）。52秒地点でサーバ都合の1011切断が1回（自動再接続で継続）。ラベル空間を分けた再解析でも一致率21%で、バッチと同傾向
- **取り違えの構造**: 現行の複数ラベル（話者1/話者2/ペンタて/わっち/松井）がpyannote側では「としや」相当の1話者に吸収される。現行の過分割か、pyannoteの結合しすぎかは、**音声のground truthなしには確定できない**

### 判定: 現時点では乗り換えのメドは立たず → ブランチ凍結
1. 「明確に改善する」という証拠が得られなかった（一致率の低さは優劣不明の食い違い）
2. Live-1はリリース直後でサーバ切断（1011）が発生しており、実会議中の話者分離基盤としては運用リスクがある
3. masterplan §3 のスコープ方針（話者分離は基準を満たしたら触らない）どおり、クリティカルパスを優先する

### 再開条件と手順（ブランチ try/pyannote-live1 に一式保存済み）
- **再開条件**: 9月対面パイロットで「話者の取り違えが介入品質を損ねる」事象が観測された場合
- 手順: benchmark_pyannote.py（バッチ比較）→ test_pyannote_live.py（ストリーミング実測）→ 不一致ターンの聴取裁定（docs/design/pyannote_adjudication_2026-07-09.md）
- 技術メモ: Live-1のイベントに start==end の縮退セグメントが多数含まれる。再開時は provider のイベントペアリング（speaker_start/end の対応付け）を要確認。切断への自動再接続は test スクリプトに実装済みだが、本統合時は provider 本体に持たせること
- 副次的知見: pyannoteの「未確定区間の回収」と「重複発話の分離」は現行より明確に優れており、**ライブ用途でなく事後分析（修論の実装評価・リプレイ評価の補助）に precision-2 バッチを使う**のは低リスクで価値がある

### 8.1 実地観察の追記（2026-07-09、`--diarization pyannote` での単独発話テスト）

一人での発話が 参加者A/B/D/E の複数話者として議事録化された。原因の推定:
(1) Live-1 のセッション序盤のクラスタ不安定（ラベル揺れ）を SpeakerResolver が新規参加者として即登録する、
(2) 抑揚の大きい独話でクラスタが分裂しやすい、(3) イベントタイムスタンプと録音タイムラインの整合ずれの可能性。
→ **凍結判定を実地でも確認**。現行（フラグなし）運用を継続。
再開時の必須対応: SpeakerResolver 側に「新ラベルは N 秒/累積発話量の様子見をしてから参加者化する」ヒステリシスを入れること（縮退セグメント対策・再接続対応と合わせて）。

### 8.2 フェア再検証用の最適化実装（2026-07-13追記）

8.1 で確認した「単独発話が複数参加者に分裂する」問題に対し、ブランチ
`try/pyannote-live1` 上で以下を実装した（`--diarization pyannote` 使用時のみ
影響。Soniox/AssemblyAI経路の挙動は変えない）。

1. **参加者化ヒステリシス**（`_constants.PYANNOTE_PARTICIPANT_HYSTERESIS_S`
   既定3.0秒、`SessionState.key_for_diarization_speaker` /
   `key_for_stt_fallback_speaker`）: pyannote provider使用時のみ、生ラベル
   （`source:speaker`）ごとの累積発話msが閾値に達するまで `@diar:N` を新規発行
   せず `UNSURE_SPEAKER`（未確定）に留める。閾値到達後は通常どおり `@diar:N`
   を発行し、以後同じ生ラベルには安定して同じキーを返す（既に登録済みの
   ラベルの遡及付け替えは行わない）。他providerは従来どおり即時登録。
2. **縮退セグメント（start==end）対策**（`_pyannote_diarization.py`
   `_parse_message`）: (a) 対応する `speaker_start` が無い `speaker_end` は
   区間を再構成できないため破棄しログに残す、(b) `end_ms <= start_ms` と
   なるペアも破棄する。従来はフォールバックで `end` の timestamp を
   start扱いにしており、0ms区間を量産していた（下流の `DiarizationEvent.closed()`
   自体はこれを弾くため実害は限定的だったが、`drain_events()` 経由の生ログ・
   統計には混入していた）。
3. **自動再接続**（`_pyannote_diarization.py` `auto_reconnect`/
   `max_reconnects`引数、既定 有効/3回）: `send_audio()` 中の送信失敗を検知
   すると新しいLive-1セッションを作り直し、タイムスタンプに送信済み累計msを
   オフセット加算して連続タイムラインに補正する。再接続後のラベルは
   `R{epoch}:` を前置して旧セッションのラベルと衝突しないようにする。再接続
   直後にラベル空間が変わっても、上記ヒステリシスが新ラベルの即時参加者化を
   抑えるため、既存参加者が別の `@diar:N` に一時的に分裂するリスクは
   「猶予期間中はUNSURE扱いになる」形で緩和される（ただし猶予後は新しい
   `@diar:N` として確定するため、頻繁な再接続が起きるセッションでは同一人物に
   複数の参加者IDが割り当たり得る点は残存の既知限界）。
4. **話者数ヒント配線**（`_bootstrap.py`）: `PyannoteStreamingDiarizationProvider`
   生成時に `max_speakers=args.diarization_max_speakers` を渡すよう配線した
   （AssemblyAI経路と同じ形に統一）。ただし `POST /v1/live` のボディスキーマは
   プロパティを持たない `object` 型であり、pyannoteAI側には送信されない
   （2026-07-09時点のAPI仕様確認済み、`_pyannote_diarization.py` docstring
   参照）。セッションレベルでの人間話者数抑制は既存の
   `SessionState.constrain_human_speaker_key` / `VoiceProfiles.set_max_human_speakers`
   経路が引き続き機能する。

追加テスト: `tests/unit/live/test_pyannote_diarization.py`
（縮退セグメント非emit2種、`max_speakers`保持とAPI非送信の確認）、
`tests/unit/live/test_session_state.py`（ヒステリシス閾値未満/到達後の挙動、
非pyannote providerでの従来挙動維持）。

**再検証コマンド**:

```bash
# バッチ再検証（話者数ヒントあり）
uv run python scripts/benchmark_pyannote.py --session 2026-06-25_1554 --num-speakers 4

# ライブ単独発話テスト（一人で話し続け、偽参加者が出ないか確認）
uv run python -m das.asr.live --diarization pyannote --diarization-max-speakers 2
```

**期待される結果**:
- ライブ単独発話テストで、ヒステリシス猶予（3秒）を過ぎても参加者が1人の
  ままであること（8.1で観測された参加者A/B/D/Eへの分裂が再発しないこと）。
  再接続が発生した場合は再接続直後の3秒程度だけ未確定表示が続くのは想定内。
- バッチ再検証の一致率が実装前（27.9%/27.4%、本ドキュメント冒頭参照）から
  改善するかは、あくまで参考値として記録する（バッチ側のロジックは今回
  変更していないため大きな変化は想定していない。変化があれば
  `--num-speakers` 指定の効果、または比較対象セッションの差によるものと
  考えられる）。
- 縮退セグメント対策・自動再接続はいずれもログ（`logger.warning`）で動作が
  追跡できるため、再検証時は標準エラー出力も合わせて確認すること。

### 8.3 盲検聴取による裁定結果（2026-07-09、松井氏による一次判定 n=20）

対象: 2026-06-25_1554 の不一致ターン（num_speakers=4 のバッチ結果）から無作為20件。
ブラインド形式（システム判定を隠して「誰の声か」を選択）。

- **現行のみ正解: 4 / pyannoteのみ正解: 7 / 両方不正解: 9**
- 争点区間での現行の正解率は 4/20=20%、pyannote は 7/20=35%
- 現行の誤りには幽霊ラベルへの帰属（正解ペンタて→としや、正解わっち→ペンタて等）や、
  「その他/不明」が正解の区間への確信的割当（4件）を含む
- 粗い全体推定（一致ターン65件を両者正解と仮定）: 現行≈46%、pyannote≈56%

**判定の更新: 凍結を解除し、ハイブリッド構成の検討を正式に再開する。**
根拠: (1) 争点でpyannote優位（7:4）、(2) 未確定回収・重複分離は一貫してpyannote優位、
(3) ライブのラベル揺れはヒステリシス実装で解消済み（実地確認済み）。
残る課題: Live-1のサーバ切断(1011)の運用リスク、ストリーミング粒度の確認、
声紋照合（名前付け）との統合設計。

注: 本裁定は不一致ターンのみのサンプルであり全体精度ではない。裁定者は著者1名
（修論記載時はこの限界を明記。シートは transcripts/clips/checklist.html、再生成可能）。

### 8.4 盲検裁定 第2セッション（2026-06-25_1614, n=20）と一貫性分析

表面スコア: 現行のみ正解4 / pyannoteのみ正解1 / 両方不正解15。
ただしラベル→名前のマッピングは現行ラベル基準の貪欲対応であり、現行の誤りが
pyannote側の成績に伝播する構造だった。**生ラベルの一貫性**で再分析:

- 裁定者が「わっち」と確定した9件: pyannote生ラベル SPEAKER_01 が 7/9 (78%) で一貫。
  現行の名前付けは わっち 4/9 (44%)、残りは としや/人物6/話者2 に分散
- 「話者1」4件: pyannote SPEAKER_03 3/4 (75%) vs 現行 2/4 (50%)

**結論（2セッション総合）**: pyannoteのクラスタリング（同一声の束ね）は現行の
声紋名前付けより一貫して優秀。弱点は名前対応付けのみ。
→ **ハイブリッド構成を実装する**: pyannote Live-1 が「区間と話者クラスタ」を出し、
名前付けは既存の声紋照合（登録プロファイルとのマッチング）をクラスタ単位で行う。
クラスタ単位の照合は発話断片単位より安定するため、両者の弱点を相殺できる見込み。

補足: さらに厳密な一貫性測定が必要なら、既知話者のYouTube動画等を流す統制テストが
可能（裁定者の確信度問題を回避）。現時点では設計判断には不要と判断。
