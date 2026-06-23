# discussion-support

マルチエージェントによる議論グラフ統合型 議論支援システム (das = Discussion Argumentation Support).

議論ログ側 (立場を持つ claim / premise) と外部知識側 (中立な事実 = evidence) を
支持・攻撃エッジで連結した「統合議論グラフ」を、複数の専門エージェント
(論証抽出 / ドキュメント知識 / Web 検索 / 連結 / ファシリテーション) が
分業して構築・運用する研究プロトタイプ。

ノード型は議論側の `claim` / `premise` と知識側の `evidence` の 2 系統。外部の文書・Web
は「主張」に分解せず中立な事実として扱い、その事実が各主張を支持するか攻撃するかは
**対象主張ごとのエッジ** で表現する (同じ事実が主張 A を支持し主張 B を攻撃しうる)。

- **論証抽出 (Extraction)**: 発話を claim / premise に分解 (議論側、立場あり)
- **ドキュメント知識 (Document)**: 事前文書を中立な事実 (evidence) に分解
- **Web 検索 (WebSearch)**: リアルタイム検索結果を事実 (evidence) ノード化
- **連結 (Linking)**: 事実 → 主張の関係を対象主張ごとに支持/攻撃/中立で判定しエッジ化
- **ファシリテーション (Facilitation)**: グラフ全体を読み「いつ・誰に・何を」提示するか中央調停

## クイックスタート

```bash
# 依存関係をインストール
uv sync --all-extras

# 環境変数を設定
cp .env.example .env
$EDITOR .env  # OPENAI_API_KEY を設定

# 単体テストが通ることを確認 (実 API は呼ばない)
uv run pytest -q

# サンプル議論ログから AF を構築 (実 API を呼ぶ)
uv run das run-session tests/fixtures/cafeteria_transcript.jsonl

# Streamlit ビューアで結果をブラウズ
uv run das ui
```

## 対面議論のライブ入力 (Soniox + 声紋プロファイル)

speaker-attribution 由来の「誰が何を言ったか」文字起こしを統合済み (`das/asr/live.py`)。

```bash
uv sync --extra soniox
echo "SONIOX_API_KEY=..." >> .env
uv run das listen-soniox            # 録音→話者特定→統合AF構築→ライブ介入
# 実行中: 「1=松井」で話者の実名登録 / Ctrl-C で停止
# 介入(💡)はターミナルとライブ議事録HTML(2秒自動更新)の両方に出る
#   --facilitate-interval 3.0  介入判定の周期(0で無効)
#   --min-utt-chars 7          相槌をAF構築から除外する文字数しきい値
# 議事録(MD/HTML/turns.jsonl)は transcripts/ に自動保存
# バッチでも可: uv run das run-session transcripts/<日時>.turns.jsonl
```

## CLI

```bash
uv run das version                    # バージョン
uv run das ingest-docs data/docs/     # 文書を evidence ノード化して保存
uv run das run-session <file>.jsonl   # 議論ログを流して統合 AF を構築
uv run das listen                     # マイクからのリアルタイム議論を AF 化 (asr extras)
uv run das visualize <snapshot.json>  # snapshot を pyvis HTML に
uv run das ui                         # Streamlit ビューア
```

## リアルタイム音声入力 (`das listen`)

WhisperLiveKit を使ってマイク音声を逐次文字起こしし、そのまま統合 AF を組み
立てる。Apple Silicon 向けに mlx-whisper バックエンドを既定にしている。

```bash
# extras を入れる (PyTorch / mlx-whisper / sounddevice が入る、~3GB)
uv sync --extra asr

# 録音開始 (既定: large-v3 / 日本語)。Ctrl-C で停止し snapshot を保存。
uv run das listen

# モデル・言語の上書き
uv run das listen --model large-v3-turbo --language ja
```

設定は `.env` でも上書きできる:

```env
DAS_ASR_BACKEND=mlx-whisper      # CUDA なら faster-whisper
DAS_ASR_MODEL=large-v3
DAS_ASR_LANGUAGE=ja
```

注意: 話者ダイアライゼーション (複数人の自動区別) は初期実装では無効。すべての
発話が `speaker_1` で記録される。多人数議論サポートは追加予定。

## テスト

単体テスト (実 API を呼ばない、AsyncMock でフェイク):

```bash
uv run pytest -q
```

E2E スモーク (実 OpenAI API を呼ぶため、明示的な opt-in が必要):

```bash
OPENAI_API_KEY=sk-... OPENAI_INTEGRATION=1 \
    uv run pytest tests/integration -m integration -s
```

## 開発

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src/das
uv run pytest --cov=das
```

## ディレクトリ

実装計画は `docs/implementation_plan.md` を参照。
