# 作業依頼: 決定者本人の決定の発話を本人の同意として扱う

- 状態: approved（2026-10-06、高田）
- 依頼元: Claude Code（設計）／担当: Codex（実装）
- 前提: [2026-10-06_rules_update2.md](2026-10-06_rules_update2.md) の実装（コミット `320f1e3` まで）
- 正本: `docs/discussion_structure/`（コミット `df96b7d`）。仕様書（`spec/`・`decisions/`）は編集しない。

## 変更

[07_resolution.md](../spec/07_resolution.md) §1.2 に追加した規則を実装する。

- **決定を言った本人の決定の発話は、本人の同意として扱う**。決定者本人に決定に反する立場（「する」なら
  採用する案への懸念、「しない」なら元の提案への賛成）が残っていても、決定を止めない。
- 他の人の反する立場の扱い（決定後の明示的な同意がなければ決定候補）は変えない。
- 決定者本人の立場の印は変えない（決定の発話から立場を推測しない）。根拠（`resolution_evidence`）には、
  決定者本人の発話が本人の同意として扱われたことが分かるように記録する。

## シナリオ・テスト

- `decision_guards.json` の t11 と最終状態の i1 の期待結果を、前回の更新前と同じ `decided/no` に戻す
  （A は自分で「しない」と決定し、B が同意している）。前回の報告で変更した理由が今回の規則で解消される。
- 決定者本人の反する立場では止まらず、他の人の反する立場では止まることを、「する」「しない」の両方で
  確かめる単体テストを追加する。

## 完了条件・約束

- `scripted` で全シナリオ不一致 0、`tests/discussion_structure` 全件成功、全体 `pytest` と `ruff` で新規の
  失敗・警告なし。OpenAI・Jev の実通信はしない。
- ブランチ `codex/discussion-structure-rules3`（作成済み）で作業し、切り替えない。`.git` に書けなければ
  未コミットで残し、報告にコミット計画を書く。push・PR はしない。`spec/`・`decisions/`、
  `src/das/discussion_structure` 外のモジュールは変更しない。
- 報告は `docs/discussion_structure/work/2026-10-06_rules_update3_report.md` に書く。
