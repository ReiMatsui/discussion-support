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

ブラウザに議事録が自動表示され、2秒ごとに更新される。`Ctrl-C`で終了。

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

## 2. AIエージェント参加モード（Realtime API v2）

AIファシリテーターが会議に音声で参加する。`--agent` フラグで有効化。

```bash
# マイクライブ + AIエージェント
uv run python -m das.asr.live --agent

# 声を変える（alloy/ash/ballad/coral/echo/sage/shimmer/verse）
uv run python -m das.asr.live --agent --agent-voice sage

# 応答頻度を調整（既定10発話ごと）
uv run python -m das.asr.live --agent --agent-trigger 5
```

エコー防止方式（E+B）:
- **E**: AIのテキストはSTTを経由せず直接議事録に挿入
- **B**: AI音声をマイクが拾った場合、話者ラベルで自動除去

ブラウザ上で🤖トグルからON/OFFを切り替え可能。

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
| `<日時>.html` | ブラウザ表示用（ライブ更新対応） |
| `<日時>.wav` | 録音 |
| `<日時>.turns.jsonl` | ターン単位データ（採点用） |
| `<日時>.final.md` | 清書版（`--no-polish`未指定時） |
