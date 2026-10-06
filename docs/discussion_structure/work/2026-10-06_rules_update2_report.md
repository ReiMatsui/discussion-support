# 規則の追加更新 実装報告（2026-10-06）

approved の `2026-10-06_rules_update2.md` の規則 1〜5 を実装した。
参照した正本は spec/07 §1.1・§1.2・§4.1、spec/10 §5、ADR-010 の
2026-10-06 追記、前回の `2026-10-06_rules_update_report.md`。

ブランチは `codex/discussion-structure-rules2`、開始時・完了時の HEAD は `341eeb0`。
`.git` は `os.access('.git', os.W_OK) == False` のため、変更は未コミットで残した。
ブランチ変更、push、PR、OpenAI/Jev の実通信は行っていない。
実装は `src/das/discussion_structure/` 内に限定し、既存の他モジュール、
特に `src/das/asr/live/`、`spec/`、`decisions/` は変更していない。
開始時から未追跡だった `tmp/` は変更していない。`.env`、キー、音声、`data/` の
出力は変更・コミット候補に含めていない。uv は使わず、ホームの uv キャッシュへ書き込んでいない。

## 規則と実装箇所

| 規則 | 実装箇所（src/das/discussion_structure 内） | 内容 |
|---|---|---|
| 1 | `resolution.py`: `blockers` / `process` / `advance`、`models.py`: `ResolutionEvidence` | 「する」は採用する案への懸念、「しない」は元の提案への賛成を判定。本人の決定後の `agreement=yes` が確率0.8以上なら止めない。前の賛成とは別に `explicit_agreements` を保存し、決定待機・保存済み根拠とも話者訂正に追随する |
| 2 | `resolution.py`: `blockers` | 「しない」の懸念は阻害条件にも同意にも使わず、立場の印を保持する |
| 3 | `nodes.py`: `unanswered_concerns` / `process` | 保留中も未応答候補を保持。一覧に `during_discussion=false`、`summary_role=take_home` を付ける。保留からの `reopen` と別人の応答で未解決へ戻せるようにし、その後は `during_discussion=true`、`summary_role=unanswered_concern` に戻る |
| 4 | `models.py`: `Issue`、`tracker.py`: `identify_speaker` / `snapshot`、`resolution.py`: `revalidate` | 話者訂正が影響した問いの最新の自動決定根拠を再検証。別人の同意が失われた場合、または反する立場が同意なしで残る場合に `needs_confirmation` と理由を記録。`internal.decision_confirmations` を決定候補と並ぶ確認一覧として出す。状態・決定内容・案の採否を維持し、フラグは表示へ出さない |
| 5 | `resolution.py`: `observe` / `entered_focus` / `process`、`tracker.py`: `process` | 解決済みの target を問いへ正規化し、配下の案を含む連続した話の流れの賛成を焦点の表示待ち中も保存。他の問いへの実質発話で切り、相槌・判定対象外は無視する。表示の切替では保存済みの連続部分を消さない |

別人の同意条件も決定タイマー時点で再検証し、待機中の話者訂正で同じクラスタになった
同意者を誤って独立した同意者と数えない。決定後の立場 `support` のみを明示同意に
代用していた経路は、spec/07 §1.1 に合わせて `agreement` 判定へ統一した。

`src/das/discussion_structure/README.md` に追加した内部情報と扱いを記載した。
音声の生成やまとめ確認の対話フローは今回の範囲に含めない。

## シナリオと期待結果

新規 `rules_update2.{json,turns.jsonl}` は34発話、途中9箇所と最終状態を照合する。

- 「しない」に反する賛成者 B の沈黙中は未解決・決定候補。C の同意だけでは閉じず、B の後の明示同意で決定する。
- 「する」に懸念のある B の沈黙中は決定候補。C の同意だけでは閉じず、B の留保つき明示同意で決定する。懸念印は残す。
- 告知の懸念を保留後60秒以上保持し、持ち帰り事項として残す。再開への別人の応答で未解決へ戻ると途中の対象へ戻る。
- ポスターへの賛成は焦点表示の切替より前に出るが、連続した話の流れなので同意に数えて決定する。
- 掲示板の決定で、異なるクラスタの同意者が後から決定者と同じ UID に訂正される。決定状態を維持したまま `needs_confirmation` が付き、確認一覧に入る。

既存シナリオの変更:

| シナリオ | 変更 | 理由・期待結果 |
|---|---|---|
| `decision_guards.json` | t11 のチェックポイントと最終状態の i1 を `decided/no` から `open/null` へ変更 | A は元の提案への賛成者で、「しない」の決定後に明示同意していない。B の同意だけでは決定せず候補に残す。発話・scripted 判定は変更しない |
| `concern_identity_rules.{json,turns.jsonl}` | t9 の沈黙を A の「やりたかったが、しない決定に同意します」に変更し、`agreement=yes` とクラスタ a を設定 | 賛成印を持つ A 本人の明示同意を加え、従来の後半（D の別人同意で決定、蒸し返し、失効、名前の確定）を引き続き検証する。既存の木の期待結果は変更しない。沈黙で止まる場合は新規シナリオで検証 |
| `decisions_reopen.json` | t4「その決定に賛成します」に `agreement=yes` を追加 | 決定後の賛成立場だけで明示同意を代用しないため。発話・立場・木の期待結果は変更しない |

他の4シナリオの期待結果は変更していない。

新規 `test_rules_update2.py` の26件は、是非・選択・自由回答での対称な阻害条件、
本人の同意（決定者を含む）、0.79 の同意と決定前の同意の除外、保留と配下の案の懸念、
遡及訂正時の状態・表示・シリアライズ、正しい訂正での誤検出防止、待機中の同一クラスタ訂正、
焦点待ち中の賛成、相槌・対象外発話の無視、他の問いによる分断、新しい問いの開始を検証する。
新規シナリオのパラメータ化テスト1件を合わせて、今回の追加は27件。

scripted CLI（出力先 `/private/tmp/discussion-structure-rules2-scripted`）:

| シナリオ | 発話 | 照合 | 不一致 |
|---|---:|---:|---:|
| upgrade_concerns | 10 | 6 | 0 |
| decisions_reopen | 13 | 7 | 0 |
| pending_additions | 11 | 5 | 0 |
| open_stances_focus | 19 | 7 | 0 |
| decision_guards | 14 | 4 | 0 |
| focus_agreement_rules | 14 | 3 | 0 |
| concern_identity_rules | 17 | 5 | 0 |
| rules_update2 | 34 | 10 | 0 |
| 合計 | 132 | 47 | 0 |

## 検証結果

全コマンドは既存 `.venv/bin/python` を使用した。scripted と既存のスタブ・注入された
トランスポートのみを使い、OpenAI/Jev の実通信・API 費用はない。

- `.venv/bin/python -m pytest -q tests/discussion_structure`: **123 passed**（0.69秒）。既存96件と今回追加27件が成功。
- `.venv/bin/python -m pytest -q`: 作業前 **1178 passed / 30 failed / 4 skipped / 474 warnings**（92.33秒）、作業後 **1205 passed / 30 failed / 4 skipped / 474 warnings**（94.54秒、終了値1）。失敗した test ID の集合は完全一致し、新規失敗0件。成功は今回追加した27件分増加。
- `.venv/bin/python -m ruff check .`: 作業前・作業後とも **8 errors、終了値1**。ログがバイト単位で一致し、新規指摘0件。全8件は既存 `transcriber/server.py` の I001 2件、F401 1件、E402 2件、F541 2件、SIM117 1件。
- `.venv/bin/python -m ruff check src/das/discussion_structure tests/discussion_structure`: **All checks passed**。
- `.venv/bin/python -m das.discussion_structure.compare --backend scripted --out /private/tmp/discussion-structure-rules2-scripted`: **終了値0、8本・47照合で不一致0**。
- `git diff --check`: **終了値0**。

比較ログ: `/private/tmp/rules2-{baseline,final}-{pytest,ruff}.log`。
既存の30失敗は `tests/unit/live/test_modes.py` の1件と `test_ui_api.py` の29件で、
HTTP bind がサンドボックスに拒否される `PermissionError: [Errno 1] Operation not permitted` によるもの。
474 warnings は既存 `src/das/graph/store/networkx_store.py` の `datetime.utcnow()` 非推奨警告で、件数は増えていない。

## コミット計画

`.git` が書込み不可のためコミットIDはない。次の順番で、ファイルを明示してコミットする計画。
`git add .` は使わず、既存の `tmp/`、機密情報、実データ、生成出力を含めない。

1. `feat(discussion-structure): apply symmetric decision guards and confirmation tracking`
   - `src/das/discussion_structure/models.py`
   - `src/das/discussion_structure/nodes.py`
   - `src/das/discussion_structure/resolution.py`
   - `src/das/discussion_structure/tracker.py`
2. `test(discussion-structure): cover rules update2 with offline scenarios`
   - `tests/discussion_structure/test_rules_update2.py`
   - `tests/fixtures/discussion_structure/rules_update2.json`
   - `tests/fixtures/discussion_structure/rules_update2.turns.jsonl`
   - `tests/fixtures/discussion_structure/concern_identity_rules.json`
   - `tests/fixtures/discussion_structure/concern_identity_rules.turns.jsonl`
   - `tests/fixtures/discussion_structure/decision_guards.json`
   - `tests/fixtures/discussion_structure/decisions_reopen.json`
3. `docs(discussion-structure): document rules update2 and offline validation`
   - `src/das/discussion_structure/README.md`
   - `docs/discussion_structure/work/2026-10-06_rules_update2_report.md`

## 曖昧な点と採用した解釈

1. **決定者自身の反する立場**: §1.2 は「その人」の例外を設けていないため、決定者本人も決定後の `agreement` が必要。決定発話自体から立場の変更・同意を推測しない。`decision_guards` の期待結果変更はこの解釈に基づく。
2. **保留から未解決への戻し方**: 新しい API を作らず、既存の `reopen` と別人の応答を保留にも適用する。未応答の懸念の同じ ID・根拠を保持し、状態から伝える対象かを毎回計算する。
3. **途中で伝える対象の意味**: `during_discussion` は保留による除外を表す。既存 Controller の間隔・焦点移動・1回までの発声の採否は実装しない。`summary_role` で持ち帰り事項を抽出できる。
4. **連続の単位**: 表示上の焦点ではなく、Nodes が解決した target の問い（配下の案を含む）を単位にする。新しい問いは作成時から。判定済みの `relevance=off` と相槌・AI・無音は区切りに数えず、その他の実質発話が有効な問いを指さなければ連続を切る。
5. **確認フラグの寿命**: 最新の自動決定を検証し、フラグは後の話者訂正だけでは自動解除しない。まとめで明示的に確認すべきという §4.1 に従う。既存の蒸し返しで未解決へ戻った際はフラグを解除し、過去の根拠履歴は残す。まとめ確認の音声・応答による更新フローは今回追加していない。
6. **表示を維持する範囲**: 再検証は決定状態・内容・案の採否を変えず、確認フラグによる `display_commit` は発生しない。既存の話者訂正による立場の名前や衝突整理は従来どおり反映する。
