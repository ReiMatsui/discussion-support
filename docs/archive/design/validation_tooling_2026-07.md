# 設計書: 検証ツール整備（S1〜S3の実施を1日仕事にする）

実施者向け。根拠: `docs/research/near_term_validation_plan_2026-07.md`。目的: ユーザーが実会議・台本読み上げをした**後の集計・採点を全自動**にする。コーディングのみで完結（対面不要）。品質基準は共通。scripts/ 配下は mypy 対象外なら型は緩くてよいが、ruff は通すこと。

## V1 `feat(scripts): 介入セッション集計スクリプト`

`scripts/analyze_session.py <transcripts/日時>`（拡張子なしプレフィックス）で、1セッションの全ログ（turns / interventions / intervention_review / diag）から以下を1枚のMarkdownレポートに出力:
- 介入一覧: kind / trigger時刻 / pause_required vs actual / candidate_wait / speak_start_latency / 被り判定（発話開始時に turns 上で誰かの発話区間内だったか）
- 抑制の内訳: suppressed code 別件数（awaiting_pause / cooldown_* / expired / awaiting_drift_confirmation）
- 呼びかけ: voice_call_diag の検出一覧と、queued→dispatched/expired の遷移・所要秒
- triage: 注釈済み発話数 / factual_claim率 / skipped(backlog・intervention_off)件数
- 話者: 話者別発話数・時間シェア・未確定率、echo_drop 件数
- 人手ラベル欄: 各介入に「適切/早い/遅い/不要」を書き込む空欄列（Markdown表）を出し、記入後に `--with-labels <file>` で適合率を再集計できる
出力先: `transcripts/<日時>.analysis.md`。**S3実施日はこのスクリプトを回すだけ**、が受け入れ基準。テスト: 16:42セッションの実ログをフィクスチャにコピーして期待値固定（被り判定が manual#1 で真になること）。

## V2 `feat(scripts): S2台本パッケージと自動採点`

- `data/validation/s2_scripts.md`: 呼びかけ検証台本を**成果物として同梱**する。構成: (a)明示呼びかけ+依頼 15文 (b)話題言及 15文 (c)境界例 10文 (d)呼称なし依頼 5文。各文に期待ラベル（call/not_call）を付与。near_term_validation_plan §3 の例文を核に、言い淀み・句読点なし・全角/カナ表記ゆれを含める
- `scripts/score_s2.py <turns.jsonl> <interventions.jsonl> --script data/validation/s2_scripts.md`: 読み上げ結果の turns と台本を突合（テキスト類似で対応付け）、呼びかけ precision/recall と応答レイテンシ（呼びかけ発話 end_ms → 対応する manual trigger/delivery 時刻）を出力
- 話者分離用には既存 `scripts/score_overlap_test.py` があるため新規不要。ただし README 的な1節を s2_scripts.md 末尾に足し、S2実施手順（読み上げ→2コマンド）を書く

## V3 `feat(scripts): 話者分離閾値の校正スクリプト常設`（01章 M9対応の第一歩）

`scripts/calibrate_voiceprint.py <answer.json> <turns.jsonl>...`: 既存 score_overlap_test の採点をコアに、**閾値セット（thresh/margin/short系）をグリッドで振って混同行列と帰属正解率の表**を出す。diag.jsonl の埋め込み類似値を再利用できる場合は再計算を省く（実装時に diag の中身を確認）。目的は「閾値変更の根拠を測定結果ファイルに移す」こと（コードコメントの日付入りチューニング履歴からの脱却）。DEFAULTS の変更自体は**この設計書のスコープ外**（数字が出てから別途判断）。

## V4 `feat(live): 段別レイテンシの常時計測ログ`（05章・logic_review C13の観測基盤）

発話1件が「STT確定 → triage完了 → （fact判定完了）→ 介入trigger → 発話開始」の各段をいつ通過したかを、既存ログから横断できるよう **相関ID（record の ms）** を各ログ行に含める。不足しているのは: triage 完了時刻（record への注釈時に diag へ1行）、fact 判定完了時刻。dispatch 側の timing metadata は既にあるので、V1 の集計スクリプトが段別 p50/p90 を出せるようになる。ログ追加は各1行で、洪水にならないこと（triage は発話ごと1行で許容）。

## 実施順序
V1（中・最優先——S3の前提）→ V2（小）→ V4（小）→ V3（中）。V1+V2 完了時点でユーザーに報告（S2/S3 が実施可能になったタイミング）。
