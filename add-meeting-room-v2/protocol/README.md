# protocol

## 目的

`protocol/` は、`meeting-room-v2` の backend と frontend が共有するリアルタイム通信仕様の正本です。

通信仕様を変更する場合は、先にこのディレクトリを更新し、その後で backend の Pydantic schema と frontend の TypeScript 型を合わせます。

## 含まれるファイル

```text
protocol/
  README.md
  ROOM_REALTIME_PROTOCOL.md
```

## 運用ルール

`ROOM_REALTIME_PROTOCOL.md` は人間が読む初期仕様です。

最初は Soniox realtime speech-to-text の正規化イベントだけを定義します。

Soniox の raw payload は frontend protocol にしません。backend が project-owned event へ正規化してから viewer へ配信します。

## 重要な前提

初期入力は backend 側で取得する Mac / room microphone の 1 本音声です。

raw audio は初期版では保存しません。

`speaker_label` は Soniox 由来の仮の話者クラスタです。参加者本人の確定情報として扱いません。

raw video、顔画像、会議映像、視覚特徴は v2 初期版には入れません。

## 変更時の確認

```bash
cd backend
uv run pytest
uv run ruff check .

cd ../frontend
npm run lint
npm run build
```
