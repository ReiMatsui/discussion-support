# DEVELOPMENT

## 目的

この文書は、開発者とCodexが同じ前提で作業できるようにするための開発手順を定義する。

このリポジトリでは、iOS、backend、frontendを同時に扱う。通信仕様は `protocol/` を正本とし、各実装はその仕様に従う。

## 基本構成

```text
protocol/
  通信仕様の正本

backend/
  WebSocket、検証、文字起こし連携、viewer配信

frontend/
  viewer画面、リアルタイムイベント表示

iOS/
  会議デバイスクライアント、音声送信、視覚特徴送信

docs/
  開発計画、運用手順、リリース確認
```

## 開発の原則

変更はできるだけ縦切りで行う。

例として、`visual.features` を追加する場合は、iOS送信、backend受信、frontend表示、protocol更新、テスト追加までを1つの作業単位にする。

ただし、単純なlint修正、文書修正、テスト修正は個別作業として扱ってよい。

## 作業前の確認

リポジトリ直下で次を確認する。

```bash
git status --short
```

意図しない変更がある場合は、先に内容を確認する。

```bash
git diff
```

## backend 開発

backendでは `uv` を使う。

初回セットアップは次の通り。

```bash
cd backend
uv sync
```

テストは次の通り。

```bash
cd backend
uv run pytest
```

静的チェックは次の通り。

```bash
cd backend
uv run ruff check .
```

`.venv/` はローカル開発用の仮想環境であり、Git管理には含めない。削除しても `uv sync` で再作成できる。

## frontend 開発

初回セットアップは次の通り。

```bash
cd frontend
npm install
```

開発サーバーは次の通り。

```bash
cd frontend
npm run dev
```

lintは次の通り。

```bash
cd frontend
npm run lint
```

buildは次の通り。

```bash
cd frontend
npm run build
```

`node_modules/` と `dist/` はGit管理に含めない。

## iOS 開発

iOSはXcodeで開発する。

確認すべき内容は次の通り。

```text
プロジェクトが開けること
Swiftの型定義が protocol と一致していること
WebSocket接続ができること
マイク権限の説明文が Info.plist にあること
カメラを使う場合はカメラ権限の説明文が Info.plist にあること
実機で音声送信を確認できること
```

Xcodeのユーザー固有ファイルはGit管理に含めない。

```text
xcuserdata/
*.xcuserstate
DerivedData/
```

## protocol 更新手順

通信仕様を変える場合は、最初に `protocol/` を変更する。

最低限、次を更新する。

```text
protocol/REALTIME_PROTOCOL.md
protocol/realtime.schema.json
protocol/examples/*.json
```

その後、必要に応じて次を更新する。

```text
backend/app/realtime/schemas.py
frontend/src/types/realtime.ts
iOS/.../RealtimeMessage.swift
backend tests
frontend tests
iOS build確認
```

## 追加してよいもの

Git管理に入れてよいものは次の通り。

```text
ソースコード
テストコード
通信仕様
サンプルJSON
ドキュメント
設定テンプレート
lockファイル
```

## 追加してはいけないもの

Git管理に入れてはいけないものは次の通り。

```text
.env
.env.local
node_modules/
dist/
.venv/
.pytest_cache/
.ruff_cache/
.DS_Store
xcuserdata/
*.xcuserstate
DerivedData/
ログファイル
秘密鍵
APIキー
```

## 完了条件

通常の変更では、少なくとも次を確認する。

```bash
cd backend
uv run ruff check .
uv run pytest

cd ../frontend
npm run lint
npm run build
```

iOSを変更した場合は、Xcodeでビルドできることを確認する。

## コミット単位

コミットは、意味のある小さな単位にする。

例は次の通り。

```text
chore: add protocol source of truth docs
fix: resolve frontend lint error
feat: add visual.features backend validation
test: validate protocol example messages
```

## Codexに依頼する場合の形式

Codexへは、目的、変更範囲、完了条件を明示する。

例は次の通り。

```text
目的:
visual.features を iOS から backend に送信し、frontend で受信状態を表示できるようにする。

変更範囲:
protocol/
backend/
frontend/
iOS/

完了条件:
backend の ruff が通る
backend の pytest が通る
frontend の lint が通る
frontend の build が通る
iOS のビルドが通る
README または docs に確認手順がある
```
