# 直近の検証計画: ライブ介入システムの妥当性確認（2026-07-02版）

`research_plan_2026-07.md` のトラックB Week 1-3 を具体化したもの。対象は4領域: **①介入ロジック ②介入タイミング ③呼びかけ応答 ④話者分離**。方針は「**先に測る、直すのは基準を割った所だけ**」— 観測ログ（interventions.jsonl / intervention_review.jsonl / diag.jsonl）とリプレイハーネスが既に揃っているので、改修前にまず現状の数字を出す。

---

## 0. 共通: データ取りの段取り（最初にやる）

すべての検証は同じ録りデータを使い回せる。取るべきセッションは3種:

| セッション | 内容 | 得られるもの |
|---|---|---|
| S1: 一人シミュレート | `uv run python -m das.asr.live --simulate 'AIツール導入の是非' --sim-scenario derailed` と `imbalanced` を各1回 | drift/invite が発火する基本動作の確認（無料に近い、即日可能） |
| S2: 台本読み上げ | 呼びかけ・話者分離用の台本（後述§3・§4）を1〜2人で読み上げ | 呼びかけ precision/recall、話者帰属の正解付きデータ |
| S3: 実会話 20〜30分 × 2本 | 研究室の2〜3人で普通に議論（パイロット#1を兼ねる。議題は政策AIトピック推奨） | 全4領域の実地データ。`transcripts/` に turns.jsonl / interventions.jsonl / intervention_review.jsonl / diag.jsonl が自動保存される |

S3の直後に毎回、参加者に3問だけ聞く（各介入について 適切/早すぎ/遅すぎ/不要、呼びかけに反応したか、名前の取り違えに気づいたか）。**人手ラベルが全ての基準値になる。**

---

## 1. 介入ロジックの妥当性（何を・いつ介入するか）

### 現状の仕組み（確認済みの実装）
- 候補生成: drift（1s毎・確認2回/20秒窓・TTL30s）/ fact（triage分類→high確信のみ）/ invite（公平シェア0.5未満の人がいる時のみLLM判定・8s間隔）/ count（10発話）/ silence（standard 18s）/ manual（UI・音声）/ retry
- 採否: FacilitationController が優先度・pause・cooldown・期限で一元裁定。全採否が review ログに残る

### 検証すること（S1＋S3）
1. **適合率**: 発火した各介入を人手で「適切/不要/遅い/早い」ラベル → kind別の適合率。**基準: 「不要」が2割以下**（それ以上なら閾値でなくトリガー条件自体を疑う）
2. **不発の内訳**: intervention_review.jsonl の suppressed code 分布（awaiting_pause / cooldown_global / awaiting_drift_confirmation / expired）。「人間が介入してほしかったのに黙っていた」場面を参加者ヒアリングから逆引きし、どの code で死んだか特定
3. **drift の確認回数(2回/20秒)の妥当性**: derailed シミュレートで、脱線開始→介入までの発話数を測る。5発話以上かかるなら窓かconfirmationsを調整
4. **invite の宛先妥当性**: 声かけ対象が本当に「静かな人」だったか（diag の participation_stats と突き合わせ）。話者誤帰属由来の誤声かけが1件でもあれば§4を優先

### 既知の懸念（レビュー由来、測ってから判断）
- epoch/deadline の安全装置が実質不発（レビュー01 H5）→ **「話題が変わった後に古い介入を喋る」事象がS3で観測されたら**着手。観測されなければ後回し
- replay が採否層を再現しない（M5）→ 閾値を弄る必要が出た時点で「採否込みリプレイ」を実装（それまでは不要）

### 使う道具
```bash
uv run python -m das.asr.live.replay transcripts/<日時>.turns.jsonl --no-api --serve   # 介入レビューUI
```
＋ 軽い集計スクリプトを1本書く: interventions.jsonl から kind別件数・suppressed code分布・人手ラベルとの突合表を出す（30分仕事、最初に作る価値が高い）

---

## 2. 介入タイミング（「間」の質）

### 現状の仕組み
- kind別 pause（fact 0.9s / manual 1.0s / count 1.5s / drift 1.8s / invite 2.0s / retry 2.4s）、global cooldown 25s（standard）
- timing metadata が毎介入に記録済み: `pause_required_sec` / `pause_actual_sec` / `candidate_wait_sec` / `speak_start_latency_ms`

### 検証すること（S3）
1. **被り率**: 介入の発話開始時に人間が既に話し始めていた率（turns.jsonl の ms と介入 delivery 時刻の突合）。**基準: 1割以下**
2. **体感遅延**: trigger→初回音声の `speak_start_latency_ms` の分布（p50/p90）。**p50が1.5秒を超えるなら**再設計案D（生成先行・再生ゲート）を検討
3. **fact の鮮度**: `candidate_wait_sec`（誤り発話→補正発話までの待ち）。話題が次に進んでから訂正していないか人手ラベルと突合

### 既知の懸念
- 「沈黙」をSTT確定時刻で測っている（M6）→ pause_actual が実際の間とずれる。**被り率が基準を割った場合の第一手**は partial トークン受信時刻での `_last_utt_time` 更新（小改修）。VAD導入はその次

---

## 3. 呼びかけ応答（音声での手動呼び出し）

**先週 regex→LLM分類（triage）に置き換えたばかりで実地未検証。今回の検証の最優先項目。**

### 現状の仕組み
- `_run_triage_worker` が確定発話ごとに1回 `classify_utterance` を呼び、`facilitator_request` 非空なら manual_call(source=voice) を投入。TTL30秒、status遷移（queued→waiting→dispatched→delivered/expired/cancelled）はUIに出る。検出は voice_call_diag としてログに残る

### 検証すること（S2: 台本40〜60文）
台本の構成: (a) 明示呼びかけ+依頼 15文（「AIさん、ここまで整理して」「ファシリテーター、話を戻して」、句読点なし・言い淀みありも含む）、(b) 話題言及 15文（「AIは便利だよね」「AIの導入について話しましょう」）、(c) 境界例 10文（「AIさんに後で聞こうか」「進行役って必要かな」）、(d) 依頼だが呼称なし 5文（「ちょっとまとめてくれる？」→ 現仕様では拾わない想定を確認）

1. **Precision / Recall**: 基準 **precision ≥ 0.9（誤爆が会議を壊すので優先）、recall ≥ 0.8**
2. **応答レイテンシ**: 呼びかけ発話終了→AI発話開始。**基準: p50 ≤ 3秒**（STT確定＋triage LLM＋pause 1.0s＋生成が乗る。voice_call_diag と timing metadata で分解できる）
3. **TTL失敗率**: queued→expired になった率。高ければ「間が取れない」問題（§2）が原因

### 調整先
- 精度が低い → `_TRIAGE_PROMPT`（`src/das/asr/live/_constants.py`）の呼びかけ定義・例を調整。**regexを足さないこと**
- レイテンシが遅い → triage の tick(0.25s)やモデル、manual の pause(1.0s) を順に疑う

---

## 4. 話者分離（Soniox＋声紋）

### 現状の仕組みと方針
- Sonioxの話者ラベル＋ReDimNet声紋照合＋SpeakerResolver。判定の全根拠が diag.jsonl に落ちる。**段階Cは closed roster＋事前登録（読み上げ登録UI実装済み）で固定**という方針は決定済み — なのでオープンセット自動登録の精度改善はやらない
- 採点道具は既にある: `scripts/make_overlap_testset.py`（テストセット生成）→ `scripts/score_overlap_test.py`（最適1対1対応での発話帰属正解率・重なりあり/なし内訳・取りこぼし率）

### 検証すること（S2＋S3、**全員事前登録した状態で**）
1. **発話帰属正解率**: score_overlap_test.py で採点。**基準: 重なりなし ≥ 95%、重なりあり ≥ 80%**
2. **誤帰属率（他人に確定）**: **基準: ≤ 2%**。未確定（?）に落ちるのは許容、他人への確定は invite の誤声かけ・引用の誤記に直結するので厳しく見る
3. **未確定率**: 高すぎる（>15%）と participation 統計が痩せて invite が機能しない
4. **実会話での体感**: S3後のヒアリング「自分の発言が他人名義になった瞬間があったか」

### 判断ルール
- 基準を満たす → **触らない**（閾値群M9・ポリシー分散M3のリファクタは段階Cの後でよい）
- 誤帰属が基準超え → 閾値を上げて「未確定に落ちやすく」する方向のみ調整（closed roster前提なら未確定は安全側）。調整は必ず score_overlap_test.py の数字で回帰確認
- 重なり時だけ悪い → 仕様として受容し、段階Cの司会進行で「同時発話を避ける」運用に含める

---

## 5. 進め方（1.5〜2週間）

| 日程 | やること |
|---|---|
| Day 1 | 集計スクリプト作成（§1）＋ S1シミュレート2本 → 明らかな異常の有無 |
| Day 2 | S2台本作成・読み上げ収録 → §3呼びかけ・§4話者分離の数字を出す |
| Day 3-4 | 基準割れ箇所の修正（想定: _TRIAGE_PROMPT調整が主。話者閾値は触らない可能性が高い） |
| Day 5 | S2再測定（修正の回帰確認） |
| Week 2 | S3実会話×2本（パイロット#1兼用）→ §1適合率・§2タイミングの数字＋参加者ヒアリング |
| Week 2末 | 結果を1枚にまとめ、(a) 段階Cに向け許容できる水準か (b) AF×ライブ統合（H1）に進んでよいか を判断 |

### この計画の出口
「介入・タイミング・呼びかけ・話者分離の各基準値と実測値の表」が1枚できる。これは (1) H1統合に進む判断材料、(2) 中間発表の「システム実現可能性」セクション、(3) 修論の実装評価節、の3か所でそのまま使える。
