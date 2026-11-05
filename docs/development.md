# コードと確認作業の見取り図

2026年11月5日。起動・実行・人工入力検査・実API確認を区別するための入口。

## 起動

- **現在の研究用モード**：`uv run das study`。`src/das/cli/_study.py` が課題・条件を下層の音声本体へ渡す。従来のグラフ構築を起動しない。
- **従来の統合システム**：`uv run das listen-soniox`。`src/das/cli/_listen.py`。既存の動作を維持している。
- **開発者向けの確認**：`uv run das check`。範囲と使い分けは [テストの入口](../tests/README.md)。

## コードの役割

|場所|役割|
|`src/das/cli/`|コマンドの解釈と既存処理への引渡し|
|`src/das/study/engine.py`|研究用モードの固定判定・文面・提示の確定|
|`src/das/study/live.py`|研究用の意味抽出と会話の更新・音声要求の制御|
|`src/das/study/view.py`|参加者向けの共有表示|
|`src/das/study/bounded_probe.py`、`probe_usage.py`|本人が実行する接続確認と、永続的な回数・予算の管理|
|`src/das/study/preflight.py`、`replay.py`|人工入力を用いたオフラインの検査|
|`src/das/llm/cost.py`|共通のトークン数・推定費用集計|
|`src/das/llm/openai_client.py`|既存の非同期LLMクライアント|
|`src/das/asr/live/_bootstrap.py`|音声セッションの準備・起動、従来のLLM支援の組立て|
|`src/das/asr/live/_llm_transport.py`|同期のLLMパラメータ・通信。起動処理から分離|
|`src/das/asr/live/_session_state.py`、`_workers.py`|音声本体の共有状態・常駐処理|
|`src/das/asr/live/stt/`、話者帰属関連モジュール|音声認識・話者帰属|
|`src/das/agents/`、`graph/`、`runtime/`|従来の議論グラフ・エージェント・統合処理|
|`eval/`|既存の録音・コーパスの評価経路。採点の正本は `_pipeline.py`|
|`tests/support/`|複数の自動テストから使う人工判定・スタブ|

同期通信の既存利用箇所との互換性のため、`_bootstrap` から使われていたヘルパー名は再公開している。研究用モードでは起動モジュールを経由せず通信処理を参照できる。既存の再試行方式、モデル選択、話者帰属の校正値は今回変更していない。

## 今回の整理の範囲

LLM通信の分離、DASからの研究用起動、テスト範囲の選択、研究用テストの役割別配置、人工データの共有、手動API実験の別置き、HTTPテストの不要な終了待ちの短縮を行った。

`_session_state.py` や `_workers.py` には引き続き大きなクラス・処理が残る。音声・話者訂正・スレッド間の共有状態が密に関係するため、一括移動では整理を終えず、責務を特定して関連テストと一緒に分ける必要がある。今回の整理でこれらの内部を分解済みとは扱わない。
