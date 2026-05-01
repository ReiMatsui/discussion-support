# discussion-support

マルチエージェントによる議論グラフ統合型 議論支援システム (das = Discussion Argumentation Support).

議論ログ側 AF と外部知識側 AF を支持・攻撃エッジで連結した「統合議論グラフ」を、
複数の専門エージェント (論証抽出 / ドキュメント知識 / Web 検索 / 連結 / ファシリテーション) が
分業して構築・運用する研究プロトタイプ。

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

## CLI

```bash
uv run das version                    # バージョン
uv run das ingest-docs data/docs/     # 文書を AF 化して保存
uv run das run-session <file>.jsonl   # 議論ログを流して統合 AF を構築
uv run das visualize <snapshot.json>  # snapshot を pyvis HTML に
uv run das ui                         # Streamlit ビューア
```

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
