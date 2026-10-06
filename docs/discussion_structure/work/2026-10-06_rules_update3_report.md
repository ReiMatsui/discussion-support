# 決定者本人の同意規則 実装報告（2026-10-06）

approved の `2026-10-06_rules_update3.md` を実装した。参照した正本は
`spec/07_resolution.md` §1.2、前回報告は `2026-10-06_rules_update2_report.md`。

ブランチ `codex/discussion-structure-rules3`、開始時・完了時 HEAD `f154dea`。
`.git` は `os.access('.git', os.W_OK) == False` のため、変更は未コミットで残す。
ブランチ変更、push、PR、ネットワーク呼出しは行っていない。
既存 `.venv/bin/python` のみを使い、uv およびホームの uv キャッシュは使用していない。
実装変更は `src/das/discussion_structure/` 内に限定した。
`spec/`・`decisions/`・他モジュール・開始時から未追跡の `tmp/` は変更していない。

## 実装

- `resolution.py`: `blockers` に決定発話を渡し、決定者の UID を同意者として扱う。
  決定待機中の `advance` と話者訂正後の `revalidate` の両方に同じ規則を適用する。
  「する」の本人の懸念、「しない」の本人の賛成は決定を止めない。
  他者の反する立場には従来どおり、その本人の決定後の明示同意が必要。
  他者による独立した同意1件以上、10秒の異議待ちの条件は維持した。
- `models.py`: `ResolutionEvidence.decision_as_agreement` を追加。
  自動決定時に `true` を保存し、`decision_turn` と合わせて本人の同意の根拠を示す。
  同じ根拠を状態遷移イベントにも出力する。省略時は `false` で旧形式を読める。
  保留・取り下げでは `false`。`agreements`（独立した他者の同意）と
  `explicit_agreements`（決定後の agreement 判定）は変更せず、本人の決定を混入させない。
- 立場の更新規則は変更していない。決定判定が `stance=none` なら本人の既存の印を保持する。
  話者訂正は既存の `decision_turn` 訂正処理を使い、本人の同意も訂正後の UID に追随する。
- `src/das/discussion_structure/README.md` に規則と根拠欄を説明した。

## シナリオ・単体テスト

`decision_guards.json` の t11 と最終状態の i1 を `decided/no` に戻した。
A 自身の「しない」決定が本人の同意になり、B の後続同意もあるため決定できる。
A・B の賛成印、発話、scripted 判定、他のチェックポイントは変更していない。

前回の `test_no_decision_requires_supporters_consent_including_decider` は
他者 B の同意を要求するテストに更新した（旧パラメータの A 1件を置換）。
新規 `test_rules_update3.py` の15件で次を確認した。

- 是非型の「する」「しない」、選択型・自由回答型の採用で、本人の反する立場は止めず、
  他者の反する立場は別の人 C の同意があっても止める。その本人の明示同意後には決定する。
- 10秒の境界、立場の全フィールドの保持、本人の決定と他者の同意の根拠の分離、
  根拠の JSON 往復および状態遷移イベントへの記録。
- 本人の同意だけでは独立した他者の同意要件を満たさない。
- 「する」「しない」の決定待機中・決定後の話者訂正でも、本人の同意は追随し、
  誤った確認フラグを付けない。
- 保留の根拠には本人の決定による同意の印を付けない。

## 検証結果

- `.venv/bin/python -m pytest -q tests/discussion_structure`: **137 passed**（0.79秒）。
  開始時の123件から旧期待1件を更新・統合し、新規15件を追加したため、純増14件。
- `.venv/bin/python -m pytest -q`: 変更前 **1205 passed / 30 failed / 4 skipped / 474 warnings**
  （95.30秒）、変更後 **1219 passed / 30 failed / 4 skipped / 474 warnings**
  （94.98秒、終了値1）。失敗したテストIDの集合は完全一致し、新規失敗0件、
  成功は純増14件。警告件数も増えていない。
- `.venv/bin/python -m ruff check .`: 変更前・変更後とも **8 errors、終了値1**。
  ログがバイト単位で一致し、新規指摘0件。既存 `transcriber/server.py` の
  I001 2件、F401 1件、E402 2件、F541 2件、SIM117 1件。
- `.venv/bin/python -m ruff check src/das/discussion_structure tests/discussion_structure`:
  **All checks passed**。
- `.venv/bin/python -m das.discussion_structure.compare --backend scripted --out /private/tmp/discussion-structure-rules3-scripted`:
  **終了値0、全8シナリオ・47照合、不一致0、API費用0**。
- `git diff --check`: **終了値0**。

| scripted シナリオ | 照合 | 不一致 |
|---|---:|---:|
| concern_identity_rules | 5 | 0 |
| decision_guards | 4 | 0 |
| decisions_reopen | 7 | 0 |
| focus_agreement_rules | 3 | 0 |
| open_stances_focus | 7 | 0 |
| pending_additions | 5 | 0 |
| rules_update2 | 10 | 0 |
| upgrade_concerns | 6 | 0 |
| 合計 | 47 | 0 |

ログは `/private/tmp/rules3-{baseline,final}-{pytest,ruff}.log`。
scripted の成果物は `/private/tmp/discussion-structure-rules3-scripted/`。

既存の30失敗は `tests/unit/live/test_modes.py` の1件と `test_ui_api.py` の29件。
HTTP bind がサンドボックスで拒否される `PermissionError: [Errno 1] Operation not permitted`
によるもの。474 warnings は既存 `src/das/graph/store/networkx_store.py` の
`datetime.utcnow()` の非推奨警告で、変更前後で警告内容・件数は同一。
依頼範囲外の既存失敗・指摘は修正していない。

## コミット計画

`.git` 書込み不可のためコミットIDはない。権限がある環境で同じブランチ上に、
次のファイルだけを明示して1コミットにする。
コミット名: `feat(discussion-structure): treat decision as decider consent`。

- `src/das/discussion_structure/resolution.py`
- `src/das/discussion_structure/models.py`
- `src/das/discussion_structure/README.md`
- `tests/discussion_structure/test_rules_update2.py`
- `tests/discussion_structure/test_rules_update3.py`
- `tests/fixtures/discussion_structure/decision_guards.json`
- `docs/discussion_structure/work/2026-10-06_rules_update3_report.md`

`git add .` は使わず、既存 `tmp/`・機密情報・実データ・生成出力を含めない。
push・PR は行わない。
