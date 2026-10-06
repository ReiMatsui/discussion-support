# 議論構造のオフライン試作

ライブ基盤を import せず、保存した発話を終端時刻の順に再生する。状態の更新と
タイマーは `ms` / `end_ms`、通信・ラベル生成の処理時間だけは単調時計を使う。

```sh
.venv/bin/python -m das.discussion_structure.replay \
  tests/fixtures/discussion_structure/upgrade_concerns.turns.jsonl \
  --agenda '学食の環境対策' --backend scripted \
  --scenario tests/fixtures/discussion_structure/upgrade_concerns.json \
  --config configs/discussion_structure.toml

.venv/bin/python -m das.discussion_structure.compare --backend scripted
.venv/bin/python -m pytest -q tests/discussion_structure
.venv/bin/python -m ruff check src/das/discussion_structure tests/discussion_structure
```

`replay` は `--out` がなければ `data/discussion_structure/<入力のstem>/` に
`events.jsonl`、`snapshot.json`、`timeline.txt`、`metrics.json` を作る。
`--scenario` があれば `scenario_check.json` と判定項目別の比較結果も作る。
途中の通信失敗でも完了した部分のログを保存し、終了コード 1 になる。
人間の発話の時刻欠損、重複 turn_id、不正な確率はエラーにする。
旧議事録の AI 発話（「ファシリテーター」「パートナー」）は判定しない。
その時刻が null なら直前の発話終端に置き、正規化イベントを残す。
新しい入力では `role` と `backchannel`（旧形式の `bc` も可）を明示できる。

## 判定器とラベル生成器

- `scripted`: JSON の `judgements[turn_id]` と `labels[turn_id][purpose]` を読む。
  ラベル用途は `issue`、`position`、`upgrade`。ラベルの配列で再生成の結果も指定できる。
  シナリオには手で記述した最終木と途中の期待木を持ち、生成器の出力から期待値を作らない。
- `openai_logprobs`: 各項目につき英大文字 1 トークンの選択肢を出させ、
  `top_logprobs` を指数変換する。打ち切り・選択肢外の質量は `__other__` に残す。
  20 選択肢を超えたら階層的に判定する。判定確率は較正されたものとは扱わない。
- `jev`: 公式の `POST /v1/systemone` に Choice の質問をまとめて送り、
  各 `answers[*].probabilities` を検査する。`TYPESAFE_API_KEY` がなければ通信前に停止する。
  Jev 自体はラベルを生成しないため、実行時の遅い層には OpenAI を使う。

CLI は `.env` を環境変数を上書きせずに読む。キーは状態・ログに保存しない。
速い層は仕様 03 の 7 項目に加え、規則が必要とするきっかけ・親・応答先・乗り換え・
是非型の決定内容を独立した選択質問として返す。`pending:<turn_id>` は応答待ちの参照 ID。
ラベル生成の入力は根拠発話、焦点の周辺、親までの経路、兄弟ラベルに限る。

## 実 API の比較

```sh
.venv/bin/python -m das.discussion_structure.replay \
  transcripts/2026-06-25_1654.turns.jsonl --agenda '学食の環境対策' \
  --backend openai_logprobs --out data/discussion_structure/real

.venv/bin/python -m das.discussion_structure.compare \
  --backend openai_logprobs --out data/discussion_structure/comparison
```

`compare` は全シナリオの速い層・遅い層で **1 つの USD 2 予算を共有**し、
項目別一致率、0.2 幅の確率区間ごとの正解率、p50/p90、費用を
`comparison.json` に出す。単独 `replay` の予算はその実行だけなので、実データと比較を
合計 USD 2 以内で実施する場合は、`run` と `run_suite` に同じ `OpenAITransport` を渡す。

通信前に入力の UTF-8 バイト数と最大出力長から保守的な費用を予約する。
SDK の再試行を無効にし、通信失敗で請求の有無が不明なら予約を残す。
既定モデルは `gpt-4.1-mini`。モデルを変更する場合は入力・出力の単価も TOML で変更する。
Jev の実通信費用は未計測（今回の環境ではキーがない）。

`scenario_check` は内部の木・ラベル・型・親・状態・話者別の印・焦点を完全比較する。
API 再生ではノード生成が異なると ID がずれるので、対象 ID の一致率には構造の誤差も含まれる。
純粋な分類精度の測定には、今後、固定した正解文脈での評価も必要になる。

表示遅延は議事録の仮想時間で計算する。速い層・遅い層の実処理時間は別に記録する。
ASR 確定時刻がないため、ASR 区間の遅延は `null`。描画や AI の発話は行わない。
