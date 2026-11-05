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
|`src/das/asr/live/_session_state.py`|音声本体の共有状態|
|`src/das/asr/live/workers/`|議論判定・分類・グラフ支援・介入制御・音声接続・終了処理|
|`src/das/transcripts.py`|共通のUtterance形式JSONL読込み|
|`src/das/audio/pcm.py`|TTS検査用のPCM・WAV変換|
|`src/das/asr/live/stt/`、話者帰属関連モジュール|音声認識・話者帰属|
|`src/das/agents/`、`graph/`、`runtime/`|従来の議論グラフ・エージェント・統合処理|
|`eval/`|既存の録音・コーパスの評価経路。採点の正本は `_pipeline.py`|
|`tests/support/`|複数の自動テストから使う人工判定・スタブ|

同期通信の既存利用箇所との互換性のため、`_bootstrap` から使われていたヘルパー名は再公開している。研究用モードでは起動モジュールを経由せず通信処理を参照できる。既存の再試行方式、モデル選択、話者帰属の校正値は今回変更していない。

## 今回の整理の範囲

LLM通信の分離、DASからの研究用起動、テスト範囲の選択、研究用テストの役割別配置、人工データの共有、手動API実験の別置き、HTTPテストの不要な終了待ちの短縮を行った。

同日の追加整理で `_workers.py` を廃止し、`workers/` に責務別の実装を置いた。呼び出し元とテストは各モジュールを直接参照する。旧パスを残す転送レイヤーは追加していない。

`_session_state.py` には引き続き共有状態に関係する大きな処理が残る。話者訂正・音声区間・スレッド間の共有を切り離す際は、具体的な不変条件と検査を伴わせる。

## 常駐処理の配置

- `workers/checks.py`：議題・論点・脱線・事実・参加・構造化の判定。
- `workers/triage.py`：発話の分類、呼びかけ依頼、応答済み判定。
- `workers/af.py`：議論グラフ支援の提示とゲート。
- `workers/agent.py`：進行役の介入制御。
- `workers/voice.py`：音声エージェント接続・イベント・相手役の切替。
- `workers/lifecycle.py`：標準入力の処理と終了。
- `workers/common.py`：予期しない例外からワーカーを再開する処理。

手動実API実験は `scripts/check_facilitation.py`、少量の接続確認は `python -m das.study.bounded_probe` に一本化した。旧 `tests.test_facilitation_harness` と `das.study.probe` の転送入口は削除済み。
