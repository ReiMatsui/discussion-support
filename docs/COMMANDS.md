# Discussion Support — コマンドガイド

## 前提

```bash
cd ~/discussion-support
```

環境変数（`.env`に設定）:

| 変数 | 用途 |
|---|---|
| `OPENAI_API_KEY` | TTS音声生成・清書 |
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
# 清書なし（高速・API節約）
uv run python -m das.asr.live --no-polish

# Speechmaticsを使う
uv run python -m das.asr.live --stt speechmatics

# ブラウザを開かない
uv run python -m das.asr.live --no-open

# 声紋照合を無効化
uv run python -m das.asr.live --no-vp
```

### 声の登録

起動中にターミナルで `1=松井` のように入力すると、「人物1」が「松井」に実名化される。次回以降は `voices.json` から自動認識。

---

## 2. AIファシリテーター（Realtime API v2）

AIファシリテーターを `--agent` で有効化する。脱線したら本題に戻し、発言の
少ない人に声をかける。起動後はブラウザUIで3モードを切り替えられる。

### 3つのモード

| モード | 内容 |
|---|---|
| 議事録のみ | 文字起こし＋話者分離だけ（介入なし） |
| AIと会話 | AIと音声で議論しつつ、進行も手伝う |
| 人間に介入 | 人同士の議論を進行役として支援（脱線戻し・声かけ） |

```bash
# AIファシリテーターを有効化（議題は冒頭から自動推定）
uv run python -m das.asr.live --agent

# 議題を指定（脱線判定の基準。UIから後で変更も可）
uv run python -m das.asr.live --agent --topic 'AIツール導入の是非'

# 介入の積極性（controlled=控えめ / standard=既定 / active=積極的）
uv run python -m das.asr.live --agent --proactivity controlled

# AIと音声で議論する相手を付ける（AIと会話モードで起動）
uv run python -m das.asr.live --agent --debate 'AIツール導入の是非'

# 声を変える（alloy/ash/ballad/coral/echo/sage/shimmer/verse/marin/cedar）
uv run python -m das.asr.live --agent --agent-voice sage
```

モードは**ブラウザUI上で実行中に切替**できる（AIパートナーの接続/切断も含む）。
議題もUIから設定・変更でき、次回以降の脱線判定に即反映される。

エコー防止方式（E+B）:
- **E**: AIのテキストはSTTを経由せず直接議事録に挿入
- **B**: AI音声をマイクが拾った場合、話者ラベルで自動除去

### シミュレーション（一人で動作確認）

実際の参加者を集めなくても、AI3人の議論を生成して挙動を確認できる。

```bash
# 脱線→本題に戻す挙動の確認
uv run python -m das.asr.live --simulate 'AIツール導入の是非' --sim-scenario derailed --agent

# 発言量の偏り→声かけの確認
uv run python -m das.asr.live --simulate 'AIツール導入の是非' --sim-scenario imbalanced --agent
```

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
uv run das listen-soniox --skip-docs --soniox-args "--no-polish"

# ファイル観戦 + ファシリテータ
uv run das listen-soniox --skip-docs --soniox-args "--wav data/overlap_test/C_heavy.wav --play --no-polish"
```

---

## 4. ファイル再生テスト

### 直接注入（観戦モード）— パイプラインの実力測定

```bash
uv run python -m das.asr.live --wav data/overlap_test/C_heavy.wav --play --no-polish
```

wav内の音声を直接STTに送信。スピーカーからも再生されるので聞きながら確認できる。マイクは使わない。

### マイク経由 — 実環境テスト

ターミナル1:
```bash
uv run python -m das.asr.live --no-polish
```

ターミナル2:
```bash
afplay data/overlap_test/C_heavy.wav
```

スピーカー→部屋→マイクの実経路を通した認識精度を測定。

### 参加モード（--join）— TTS討論に乱入

```bash
uv run python -m das.asr.live --wav data/overlap_test/C_heavy.wav --join --no-polish
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
| `<日時>.html` | 静的議事録（`file://`表示・清書用。ライブUIはサーバーが配信） |
| `<日時>.wav` | 録音（「新しい会議」で会議ごとに分割） |
| `<日時>.turns.jsonl` | ターン単位データ（採点用） |
| `<日時>.final.md` | 清書版（`--no-polish`未指定時） |

ライブ表示はブラウザUI（`http://127.0.0.1:8231/`）が `/api/state`・SSEで配信する。
「新しい会議」（リセット）すると、新しいタイムスタンプで別ファイルとして保存される。
