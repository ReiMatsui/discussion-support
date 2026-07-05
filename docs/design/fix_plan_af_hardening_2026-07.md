# 修正計画書: AF統合コードの堅牢化（09章レビュー対応）＋小バグ回収

実施者向け。根拠: `docs/review-2026-07-02/09_af_integration_review.md`（file:line・修正案の詳細はそちら）。品質基準は共通（コミット分割・全テスト通過・ruff・mypy増分ゼロ・ルールベースモード挙動不変・行番号は実装前に現物確認）。

## A1 `fix(live): hold中の通常経路による介入の握りつぶしを解消`（09章 Critical）
af生成先行のhold中（最大8秒）、通常レーンがagentのbusyを見ずに採択→`trigger()`が`_responding`ガードで黙って返る→呼び出し側は候補消費・ログ記録済み、という「ログ上は発話済み・実際は無発話」。修正方針: `trigger()`が送信できたかを bool で返し、dispatch側は False のとき候補を消費せずログも `suppressed(agent_busy)` として記録する（介入ログは研究の一次データなので虚偽記録の解消が本質）。加えて通常レーン評価の前段に「hold中（`_af_gate`保持中）は barge-in系（fact/manual）以外を評価しない」ガードを検討（fact/manualはcancel_held→即時経路を優先してよい。設計判断は09章の修正案に従う）。テスト: hold中にsummarize採択が起きた場合の候補非消費とログ整合。

## A2 `fix(live): 会議リセットで af_requests をクリア対象に追加`（09章 High）
`reset_for_new_meeting` のキュー drain タプル（`_session_state.py:775-781`）に `self.af_requests` が漏れている。1行追加＋回帰テスト（リセット後に旧会議のaf候補が発火しないこと）。

## A3 `fix(live): agent worker のAFローカル状態を epoch リセットに対応`（09章 High）
`_pending.af`・`_AfEarlyGenGate`（hold中フラグ・保持テキスト）・af_l2バックオフ状態が meeting_epoch 変化でリセットされない。既存の drift/fact 系の epoch 整合パターンに合わせ、epoch 変化検知時にこれらを破棄（hold中なら `cancel_held()` も呼ぶ）。テスト: リセットを跨いだ hold が発話に至らないこと。

## A4 `fix(live): interrupt の hold 考慮とロック規律`（09章 Medium×2、1コミットで可）
- `interrupt()` 冒頭で hold 状態なら `cancel_held()` に委譲（二重cancel防止の冪等化）
- `_detect_responds_to` のロック外での要素dict書き換えを、ロック内更新 or 深いコピーに修正

## A5 `fix(eval): rescore の summary 生成に structural 集約を追加`
phase1-main で観測: ラン単位 `run_meta.json` の `structural_metrics` は全条件で算出済みなのに、`summary.json` の `by_condition.*.structural` が空。rescore の summary 集約に構造指標（mean±SD、ラン単位）を追加。既存 phase1-main に対して rescore を再実行して埋まることを確認（API再呼び出し無しで集約だけ再計算できる経路があればそれを使う）。

## A6 （確認）3b: note_intervention の実質部分埋め込み
前回指示済みだが未コミットの模様。未着手なら本バッチに含める: 応答エッジ検出の埋め込みをボイラープレート全文でなく提示項目の実質テキストに変更＋`af.interventions.jsonl` の kind=None 空行混入の修正。

## 実施順序
A2（1行・即効）→ A1（本丸）→ A3 → A4 → A5 → A6。完了後に停止・報告。
