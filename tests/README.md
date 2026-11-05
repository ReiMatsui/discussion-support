# テストの入口

普段は目的に合う範囲だけ実行する。`das check` は既存のpytestを呼ぶ入口であり、検査の内容を別に実装したものではない。実API向けのintegrationは全プロファイルで除外する。

|コマンド|範囲|使う場面|
|`uv run das check`|基本起動・CLI・費用集計・研究用の判定と予算|変更後の最初の確認|
|`uv run das check study`|採否判断支援全体。音声・表示・仮想20分も含む|現在の提案手法を変更したとき|
|`uv run das check core`|グラフ・エージェント・費用・CLI等の共通部分|共通基盤を変更したとき|
|`uv run das check live`|音声認識・話者帰属・従来の介入・UI|音声本体を変更したとき|
|`uv run das check all`|実API向けを除く全自動テスト|広い変更・統合前|

`--list` は対象を表示するだけでテストを実行しない。`--durations 10` で遅い上位10件を表示できる。プロファイルの選択は `src/das/cli/_checks.py` 一か所で管理する。異なる領域の変更時には該当する範囲を組み合わせる。quickだけの通過を、全体の通過と読み替えない。

## 配置

```text
tests/
  unit/
    test_*.py               共通基盤・CLI
    study/
      test_engine.py        引用・採否・理由の固定判定
      test_live.py          実行制御、訂正、待機、通信失敗
      test_entrypoints.py   研究用起動・HTTP画面・条件固定
      test_voice.py         読み上げ指示・無断発話・期限
      test_preflight.py     仮想20分の連続動作
      test_bounded_probe.py 手動接続確認の実行・上限
      test_probe_usage.py   累計使用量・費用予約・互換性
    live/                   既存の音声処理・話者帰属・UI等
  support/                  共通の人工データ・スタブ（test_*.pyから借りない）
  fixtures/                 保存された人工入力
  integration/              実APIを使う可能性がある別枠
```

テストの件数は、検査対象やパラメータの組合せの数でもある。今回、既存の検査は削除していない。研究用テストの移動前後で関数の存在と本体を照合した。変更した既存テストの動作は、HTTPサーバー終了の待機周期を0.5秒から0.01秒へ短縮した点である。本番の会話待機・音声タイミングは変えていない。

## 手動の実API確認との区別

- `scripts/check_facilitation.py`：従来のファシリテーションを実LLMで確認する手動実験。通常の自動テストではない。旧 `python -m tests.test_facilitation_harness` の転送入口は削除し、このスクリプトへ一本化した。今回の累計3回制限はこの旧手動実験には適用されない。
- `python -m das.study.bounded_probe`：現在の採否判断支援の少量接続確認。既定は送信なし。本人が `--execute` を付けた場合のみ送信する。
- `python -m das.study.preflight`：意味判定・音声を模擬した連続動作確認。介入効果や実音声の性能検査ではない。

本体の起動方法は [コマンドガイド](../docs/COMMANDS.md)、現時点の未確認事項は [進捗一覧](../docs/progress/STATUS_2026-11-05.md) を参照。
