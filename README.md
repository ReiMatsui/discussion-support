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

# テストが通ることを確認
uv run pytest -q

# サンプル議論ログから AF を構築 (M1.7 以降)
uv run das run-session tests/fixtures/cafeteria.jsonl
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
