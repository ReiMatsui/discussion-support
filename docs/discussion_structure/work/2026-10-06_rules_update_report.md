# 規則更新 実装報告（2026-10-06）

作業依頼 `2026-10-06_rules_update.md` の規則 1〜8 を反映した。
ブランチは `codex/discussion-structure-rules`、HEAD は開始時から `0aaf487` のまま。
`.git` の書込み可否は `os.access('.git', os.W_OK) == False` で確認した。
変更は未コミットの作業ツリーに残し、以下にコミット計画を記載する。
ブランチ変更・push・PR・OpenAI/Jev の実通信は行っていない。

参照した正本は spec/02・05・06・07・10、ADR-009・ADR-010 の 2026-10-06 追記、
前回の `2026-10-06_offline_prototype_report.md`。
`spec/`、`decisions/`、既存モジュール（特に `src/das/asr/live/`）は変更していない。
開始時からあった未追跡の `tmp/` は変更していない。
`.env`・キー・音声・`data/` の出力は変更・コミット候補に含めていない。

## 変更ファイルと規則の対応

| 規則 | 実装箇所 | 反映した内容 |
|---|---|---|
| 1 | `resolution.py` の `process` / `entered_focus` / `upgraded`、`tracker.py` の焦点切替処理 | 現在の焦点区間の採用案への賛成発話を保存し、決定者以外の発話を同意に採用。焦点移動で区切り、本人・他案・過去の区間の賛成を除外。格上げでは元の案へ移す |
| 2 | `models.py` の `Judgement.agreement`、`judgement.py` の選択質問、`resolution.py` | 「しない」には前の懸念/賛成を使わず、後の明示同意または他者の同じ内容の決定発話を使う。元の提案への立場とは分離 |
| 3 | `models.py` の `ResolutionEvidence` / `Issue.resolution_evidence`、`resolution.py`、`tracker.py` | 決定内容・対象・決定発話・同意した話者と発話・まとめ確認の保存欄を、立場から独立した履歴として保持。保留/取り下げも記録。状態変更イベントにも根拠を出す。蒸し返しでも履歴を残す |
| 4 | `nodes.py` の `expire`、`tracker.py` の `advance` | 蒸し返しも暗黙の問いと同じ `pending_ms = 60000`。境界を含めて失効し、ちょうど60秒後の応答は認めない |
| 5 | `nodes.py` の応答処理 / `clean_concerns`、`tracker.py` の処理/タイマー/`switch_agenda` | 応答・本人の立場変更/訂正・問いの決定/取り下げ・議題切替で未応答の保持を終了。応答だけでは本人の懸念の印を変えない。議題切替 API は前の木を保存し、新しい木を開始 |
| 6 | `nodes.py` の `unanswered_concerns`、`tracker.py` の `snapshot` | `internal.unanswered_concerns` にノード別の件数・候補ID・話者・名前・根拠発話・本文をまとめる。内容の同一性判定や音声生成は行わない |
| 7 | `config.py`、設定 TOML、`models.py` の `Turn.identified`、`stances.py`、`tracker.py` の `identify_speaker` | `name_threshold = 0.5`。低確信度/unsure は「?」、確信度欠損は旧挙動を維持。後の確定発話/遡及訂正で名前を更新し、印の衝突は新しい方を残す |
| 8 | `models.py` の `speaker_key` / `different_person`、`nodes.py` の応答、`resolution.py` の同意判定 | 同一 UID / 同一クラスタは別人にしない。双方が確定しているか、双方のクラスタが異なると分かる場合だけ数える。unsure の発話ごとの unknown ID や speaker_uid だけでは独立性を認めない |

上表の Python ファイルはすべて `src/das/discussion_structure/` 内。
変更ファイルの一覧:

- `src/das/discussion_structure/{models,config,judgement,stances,nodes,resolution,tracker}.py`
- `src/das/discussion_structure/README.md`（入力・根拠・新しい API の説明）
- `configs/discussion_structure.toml`
- `tests/discussion_structure/test_rules_update.py`（新規）
- `tests/discussion_structure/{test_backends,test_replay}.py`
- `tests/fixtures/discussion_structure/decision_guards.json`
- `tests/fixtures/discussion_structure/focus_agreement_rules.{json,turns.jsonl}`（新規）
- `tests/fixtures/discussion_structure/concern_identity_rules.{json,turns.jsonl}`（新規）
- `docs/discussion_structure/work/2026-10-06_rules_update_report.md`（本報告）

## シナリオ・テスト

新規 `focus_agreement_rules`: 同じ流れの前の賛成で「する」が決定、本人の賛成では決定しない、
別の問いへ移って戻った後の古い賛成では決定せず候補に留まる。

新規 `concern_identity_rules`: 同じノードの2人の懸念をまとめる、応答で未応答だけを消す、本人の立場変更で
消す、「しない」は後の明示同意まで決定しない、決定で未応答を消す、立場と根拠を別に保持する、
蒸し返しの60秒失効、クラスタのない unsure の応答を除外、異なるクラスタの応答を認める、
0.49 の印を「?」にし後の 0.5 の発話で名前を付ける。

既存 `decision_guards` は t10「その決定に賛成します」の scripted 判定を
`stance=support` から `stance=none, agreement=yes` に変更した。
「しない決定への同意」を「元の提案への賛成」と誤って同一視しないため。
B の既存の賛成印は t6 に由来し、そのまま残るため、既存5本の最終木・途中チェックポイントの
期待結果は一切変更していない。既存 `decision_guards.turns.jsonl` の発話内容も変更していない。

既存テストはスタブの新しい `agreement` 質問（判定13項目・ラベル1回）と、シナリオ数の増加に対応。
新規単体テストは是非/選択/自由回答の前の賛成、他案/本人の除外、焦点の再訪、格上げ時の根拠と懸念の移動、
60秒の境界、懸念の構造による終了と長時間保持、確信度境界/欠損、UID・クラスタの組合せ、
別ノードを話す際の名前更新、遡及訂正/衝突/未応答の重複整理、まとめ確認欄のシリアライズを検証する。

scripted CLI の実測（出力先 `/private/tmp/discussion-structure-rules-scripted`）:

| シナリオ | 発話 | 照合 | 不一致 |
|---|---:|---:|---:|
| upgrade_concerns | 10 | 6 | 0 |
| decisions_reopen | 13 | 7 | 0 |
| pending_additions | 11 | 5 | 0 |
| open_stances_focus | 19 | 7 | 0 |
| decision_guards | 14 | 4 | 0 |
| focus_agreement_rules | 14 | 3 | 0 |
| concern_identity_rules | 17 | 5 | 0 |
| 合計 | 98 | 37 | 0 |

## 検証結果

全コマンドは既存 `.venv/bin/python` で実行した。uv は使わず、ホームの uv キャッシュに書き込んでいない。
OpenAI/Jev のテストは既存のスタブ/注入済みトランスポートのみ。scripted の API 費用は 0。

- `.venv/bin/python -m pytest -q tests/discussion_structure`: **96 passed**（0.39秒）。
  既存51件と今回の追加45件が成功。
- `.venv/bin/python -m pytest -q`: **1178 passed / 30 failed / 4 skipped / 474 warnings**（92.78秒、終了値1）。
  作業前は **1133 passed / 30 failed / 4 skipped / 474 warnings**（92.95秒）。
  failed test ID の集合は完全一致し、新規失敗0件、成功は追加45件分増加。
- `.venv/bin/python -m ruff check .`: **8 errors、終了値1**。
  作業前の出力と作業後の出力はバイト単位で一致し、新しい指摘は0件。
  全8件は既存の `transcriber/server.py`（I001 2件、F401 1件、E402 2件、F541 2件、SIM117 1件）。
- `.venv/bin/python -m ruff check src/das/discussion_structure tests/discussion_structure`:
  **All checks passed**。
- `.venv/bin/python -m das.discussion_structure.compare --backend scripted --out /private/tmp/discussion-structure-rules-scripted`:
  **終了値0、7本・37照合で不一致0**。
- `git diff --check`: **終了値0**。

比較用ログは `/private/tmp/rules-{baseline,final}-{pytest,ruff}.log`。
既存の30失敗は `tests/unit/live/test_modes.py` の1件と `test_ui_api.py` の29件。
HTTP bind がサンドボックスで拒否される `PermissionError: [Errno 1] Operation not permitted` によるもの。
474 warnings は既存の `src/das/graph/store/networkx_store.py` の `datetime.utcnow()` 非推奨警告で、件数は増えていない。

## コミット計画

`.git` が書込み不可のためコミットIDはない。以下の順番でファイルを明示してコミットする計画。
`git add .` は使わず、既存の `tmp/` や機密・実データ・出力を含めない。

1. `feat(discussion-structure): apply focus-scoped agreement and speaker identity rules`
   - `src/das/discussion_structure/{models,config,judgement,stances,nodes,resolution,tracker}.py`
   - `configs/discussion_structure.toml`
2. `test(discussion-structure): cover updated agreement concern and identity rules`
   - `tests/discussion_structure/{test_rules_update,test_backends,test_replay}.py`
   - `tests/fixtures/discussion_structure/decision_guards.json`
   - `tests/fixtures/discussion_structure/focus_agreement_rules.{json,turns.jsonl}`
   - `tests/fixtures/discussion_structure/concern_identity_rules.{json,turns.jsonl}`
3. `docs(discussion-structure): report rules update and offline validation`
   - `src/das/discussion_structure/README.md`
   - `docs/discussion_structure/work/2026-10-06_rules_update_report.md`

## 曖昧な点と採用した解釈・設計側への質問

1. **焦点の開始**: 平滑化が実際に焦点を切り替えた発話の時点から数える。
   焦点候補を観測し始めた過去の時点へ遡らない。切替発話そのものの賛成は含める。
2. **前の賛成の現在性**: 本人の立場が既に変わったり訂正/乗り換えで消えたりした賛成は同意に使わない。
   焦点を再訪した際の決定発話は、新しい流れの決定として扱い、古い待機決定への同意と混同しない。
3. **「しない」への懸念**: 元の提案の懸念印は「しない決定」への異議ではない。
   後の明示同意と10秒の異議なしを満たせば、懸念印を残したまま「しない」を決定できる。
   これは「採用する案に懸念なし」という条件を「する/案の採用」に適用した解釈。
   設計側には、この非対称な扱いが意図どおりか確認してほしい。
4. **話者の確定**: 名前と同じ閾値を確定判定にも使う。別の stable UID だけでは unsure の独立性を証明せず、
   双方の既知クラスタが異なる場合に認める。同一クラスタは、名前が異なっていても別人にしない。
   一方だけクラスタが分かる場合は、双方の名前が確定していなければ数えない。
5. **懸念の単位**: ノード×話者の現在の懸念を1件として、繰り返し発話では根拠を更新する。
   複数人の本文が同じかどうかは判定しない。保留は終了条件に列挙されていないため未応答を保持する。
6. **応答の判断**: 既存の `response_to` が懸念を指す実質発話で応答を認定し、子の問いの作成規則も維持する。
   「未応答を消すこと」と「本人の懸念の印を消すこと」は別の操作。
7. **根拠の形とまとめ**: `resolution_evidence` は過去の決定を失わない追記履歴。
   確認の発話を保持できる `summary_confirmation` 欄は追加・往復テスト済みだが、前回試作にない
   まとめ確認フロー/音声生成は今回も追加していない。scripted 再生ではこの欄は空。
8. **遡及訂正**: 同じ UID/クラスタの後の確定は自動、それ以外は `identify_speaker` API に明示的な対応を渡す。
   訂正は印・未応答候補・待機中の同意・保存済み根拠の話者を更新する。
   確定済みの解決状態を遡及的に自動取り消す規則は仕様にないため追加していない。
   設計側には、訂正で決定者と同意者が同一人物になった場合の再確認方針を今後決めてほしい。

上記は今回の作業を止める質問ではなく、採用した解釈と後続設計の確認事項。
