# Discussion Support — コマンドガイド

## 前提

```bash
cd ~/discussion-support
```

環境変数（`.env`に設定）:

| 変数 | 用途 |
|---|---|
| `OPENAI_API_KEY` | AIファシリテーター・TTS音声生成 |
| `SONIOX_API_KEY` | Soniox リアルタイムSTT |
| `SPEECHMATICS_API_KEY` | Speechmatics STT（オプション） |

---

## 1. ライブ書き起こし（本番用）

### マイクからリアルタイム書き起こし（最もシンプル）

```bash
uv run python -m das.asr.live
```

ブラウザUI（`http://127.0.0.1:8231/`）が自動で開き、議事録がライブ更新される
（Server-Sent Eventsで差分配信。旧来の2秒ごと全リロードは廃止）。
終了はUIの「終了」ボタン、または `Ctrl-C`。

### ブラウザUIでできること

| 操作 | 説明 |
|---|---|
| モード切替 | 議事録のみ / AIと会話 / 人間に介入 を実行中に切替（後述） |
| 議題 | 脱線判定の基準テーマを入力・変更（空欄なら冒頭から自動推定） |
| 話者の名前を登録 | 「話者1」等に名前を付ける（声紋で次回以降も認識） |
| 発言量 | 話者別の発話時間・文字数・発話回数 |
| 論点 | 会話から自動抽出された論点 |
| 新しい会議 | アプリを止めず、STT接続を作り直して次の会議へ（声紋・名前は引き継ぐ） |
| 終了 | アプリを終了 |

### 主要オプション

```bash
# Speechmaticsを使う
uv run python -m das.asr.live --stt speechmatics

# ブラウザを開かない
uv run python -m das.asr.live --no-open

# 開始前設定をスキップしてすぐ開始
uv run python -m das.asr.live --no-setup

# 声紋照合を無効化
uv run python -m das.asr.live --no-vp
```

### 声の登録

起動中にブラウザUIの「参加者の名前を登録」から名前を入れると、次回以降は `voices.json` から自動認識。

---

## 2. AIファシリテーター（Realtime API v2）

AIファシリテーターは既定で有効。脱線したら本題に戻し、発言の少ない人に
声をかける。起動後はブラウザUIで3モードを切り替えられる。

### 3つのモード

| モード | 内容 |
|---|---|
| 議事録のみ | 文字起こし＋話者分離だけ（介入なし） |
| AIと会話 | AIと音声で議論しつつ、進行も手伝う |
| 人間に介入 | 人同士の議論を進行役として支援（脱線戻し・声かけ） |

```bash
# 推奨: 起動後、ブラウザで参加人数を確認してから開始
uv run python -m das.asr.live

# 議題を指定（脱線判定の基準。UIから後で変更も可）
uv run python -m das.asr.live --topic 'AIツール導入の是非'

# 介入の積極性（controlled=既定・控えめ / standard=標準 / active=積極的）
uv run python -m das.asr.live --proactivity standard

# AIと音声で議論する相手を付ける（AIと会話モードで起動）
uv run python -m das.asr.live --debate 'AIツール導入の是非'

# 声を変える（alloy/ash/ballad/coral/echo/sage/shimmer/verse/marin/cedar）
uv run python -m das.asr.live --agent-voice sage

# AIファシリテーターを使わない
uv run python -m das.asr.live --no-agent
```

モードは**ブラウザUI上で実行中に切替**できる（AIパートナーの接続/切断も含む）。
議題もUIから設定・変更でき、次回以降の脱線判定に即反映される。

エコー防止方式（E+B）:
- **E**: AIのテキストはSTTを経由せず直接議事録に挿入
- **B**: AI音声をマイクが拾った場合、話者ラベルで自動除去

### ファシリテーターを呼ぶ（手動呼び出し）

自動介入を待たずに、参加者側からファシリテーターを呼べる（人間に介入モード）。

- **UIから**: 「介入」パネルの入力欄に依頼を書いて「呼ぶ」（空なら直近の議論整理）。
  ボタンの下に進行状況が出る: 受付済み → 待機中（発話の切れ目待ち）→ 発話済み。
- **音声で**: 冒頭で明示的に呼びかけ＋依頼を言う。
  例: 「ファシリテーター、ここまで整理して」「進行役さん、次どうしましょう」「AI、Aさんにも聞いて」

反応しない主な理由:

- 介入がオフ（介入パネルの「オン」を確認）
- 「AIと会話」モード中（音声呼びかけ検出は人間に介入モード専用。会話モードでは普通に話しかければ応答する）
- 誰かが話し続けていて間が取れない（待機中のまま。30秒間が取れないと破棄され「応答できませんでした」と表示）
- 誤爆防止で落ちた（「AIについて…」のような話題化や、呼びかけのみで依頼表現が無い発話は拾わない）

呼び出しの経緯は `<日時>.interventions.jsonl`（`manual_call` / `manual_call_expired` /
`voice_call_diag`）と `<日時>.intervention_review.jsonl`（候補の採否）で後から検証できる。

### シミュレーション（一人で動作確認）

実際の参加者を集めなくても、AI3人の議論を生成して挙動を確認できる。

```bash
# 脱線→本題に戻す挙動の確認
uv run python -m das.asr.live --simulate 'AIツール導入の是非' --sim-scenario derailed

# 発言量の偏り→声かけの確認
uv run python -m das.asr.live --simulate 'AIツール導入の是非' --sim-scenario imbalanced
```

### 録音後の介入レビュー

ライブ実行後、`transcripts/` には議事録だけでなく、ターン単位ログと介入ログも残る。
`replay` UIを開くと、保存済み介入の「発火理由」「直近文脈」「実際に届いた発話」を並べて確認できる。

ファシリテーター呼び出しも同じ画面で検証できる:

- 手動呼び出し（UI/音声）は保存済み介入に「手動呼び出し」として出る（依頼・待ち秒数つき）。
  間が取れず破棄されたものは「呼び出し不発（期限切れ）」として区別される。
- 「音声呼びかけ診断」欄で、呼びかけとして検出された発話と、
  誤爆防止で無視された発話（話題化/依頼表現なし）を見返せる。

```bash
# APIを使わず、保存済み介入ログだけを見返す
uv run python -m das.asr.live.replay transcripts/<日時>.turns.jsonl --no-api --serve

# 保存済みログに加えて、同じturnsから介入候補を再判定する
uv run python -m das.asr.live.replay transcripts/<日時>.turns.jsonl \
  --topic 'AIツール導入の是非' --serve
```

`<日時>.interventions.jsonl` が同じ場所にあれば自動で読み込まれる。別ファイルを使う場合:

```bash
uv run python -m das.asr.live.replay transcripts/<日時>.turns.jsonl \
  --interventions transcripts/<別名>.interventions.jsonl --no-api --serve
```

レビューUIでは、delivery欠落・発火理由なし発話・文脈欠落・長すぎる介入などを
軽い品質フラグとして表示する。

集計や後処理に回す場合はレビュー項目だけをJSONLで書き出せる:

```bash
uv run python -m das.asr.live.replay transcripts/<日時>.turns.jsonl \
  --no-api --review-out transcripts/<日時>.intervention_review.jsonl
```

集計だけ欲しい場合:

```bash
uv run python -m das.asr.live.replay transcripts/<日時>.turns.jsonl \
  --no-api --review-summary-out transcripts/<日時>.intervention_review_summary.json
```

集計JSONには、件数に加えて `interventions_per_10_turns` などの10発話あたり指標も入る。

---

## 3. AIファシリテータ付きモード（dasオーケストレータ）

議論をAIがリアルタイム分析し、介入提案を出す統合モード。

```bash
uv run das listen-soniox
```

### 主要オプション

```bash
# ドキュメント事前読み込みをスキップ（素早く開始）
uv run das listen-soniox --skip-docs

# 書き起こし側にオプションを渡す
uv run das listen-soniox --skip-docs --soniox-args "--stt soniox"

# ファイル観戦 + ファシリテータ
uv run das listen-soniox --skip-docs --soniox-args "--wav data/overlap_test/C_heavy.wav --play"
```

---

## 4. ファイル再生テスト

### 直接注入（観戦モード）— パイプラインの実力測定

```bash
uv run python -m das.asr.live --wav data/overlap_test/C_heavy.wav --play
```

wav内の音声を直接STTに送信。スピーカーからも再生されるので聞きながら確認できる。マイクは使わない。

### マイク経由 — 実環境テスト

ターミナル1:
```bash
uv run python -m das.asr.live
```

ターミナル2:
```bash
afplay data/overlap_test/C_heavy.wav
```

スピーカー→部屋→マイクの実経路を通した認識精度を測定。

### 参加モード（--join）— TTS討論に乱入

```bash
uv run python -m das.asr.live --wav data/overlap_test/C_heavy.wav --join
```

wavをスピーカー再生しつつ、自分のマイクも同時に拾う。**イヤホン推奨**（ハウリング防止）。

---

## 5. テストセット生成・採点

### TTS討論音声の生成

```bash
uv run python scripts/make_overlap_testset.py
```

`data/overlap_test/` に3条件を生成:
- `A_clean.wav` — 重なりゼロ
- `B_mild.wav` — 軽い重なり（20%確率）
- `C_heavy.wav` — 過酷な重なり（40%確率 + 同時発話3箇所）

各条件に `.answer.json`（正解データ）が付属。生成済み音声はキャッシュされるため再実行は高速。

### 採点

```bash
uv run python scripts/score_overlap_test.py data/overlap_test/C_heavy.answer.json transcripts/<日時>.turns.jsonl
```

話者帰属正解率（重なり有無別）とテキスト類似度を出力。

---

## 6. その他のCLIコマンド

```bash
uv run das version              # バージョン表示
uv run das ingest-docs data/docs  # 文書をAF化して取り込み
uv run das eval                 # 評価実行
uv run das ui                   # Streamlit UI起動
uv run das visualize            # 議論グラフ可視化
```

---

## 出力ファイル

書き起こし結果は `transcripts/` に保存される:

| ファイル | 内容 |
|---|---|
| `<日時>.md` | Markdown議事録 |
| `<日時>.html` | 静的議事録（`file://`表示用。ライブUIはサーバーが配信） |
| `<日時>.wav` | 録音（「新しい会議」で会議ごとに分割） |
| `<日時>.turns.jsonl` | ターン単位データ（採点用） |

ライブ表示はブラウザUI（`http://127.0.0.1:8231/`）が `/api/state`・SSEで配信する。
「新しい会議」（リセット）すると、新しいタイムスタンプで別ファイルとして保存される。
