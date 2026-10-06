# オフライン試作 実装報告（2026-10-06）

指定された `codex/discussion-structure-offline`（HEAD `edbdb38`）上で実装した。
`.git/index.lock` の作成が `Operation not permitted` で拒否されたため、コミットはなく、
新規ファイルは未コミットの作業ツリーに残っている。ブランチ変更、push、PR は行っていない。
既存モジュール、`spec/`、`decisions/` は変更していない。作業開始前からあった `tmp/` も変更していない。
`.env`、キー、音声、`data/` の出力はコミット候補に含めない。

## 構成

```text
configs/discussion_structure.toml                  閾値・モデル・単価の上書き設定
src/das/discussion_structure/
  __init__.py, models.py, config.py                 immutable record、木、設定
  views.py, judgement.py, focus.py                  入力制限、3 backend、信念と平滑化
  nodes.py, labels.py, stances.py, resolution.py    作成・格上げ・ラベル・立場・解決規則
  display.py, tracker.py                           表示状態と議事録時刻のタイマー
  replay.py, metrics.py                            CLI、保存、照合、指標
  compare.py                                      全シナリオで予算を共有する比較 CLI（追加）
  README.md                                       実行方法、入力契約、評価の制限（追加）
tests/discussion_structure/
  test_replay.py, test_backends.py
tests/fixtures/discussion_structure/
  upgrade_concerns.{json,turns.jsonl}
  decisions_reopen.{json,turns.jsonl}
  pending_additions.{json,turns.jsonl}
  open_stances_focus.{json,turns.jsonl}
  decision_guards.{json,turns.jsonl}
docs/discussion_structure/work/2026-10-06_offline_prototype_report.md
```

§4.1 の目安に `compare.py` と README を追加した。既存パッケージや依存設定は変更していない。
シナリオ JSON は発話・scripted 判定・ラベル・手で作った期待木を含む。
同名の JSONL は CLI 用の発話だけである。

## 提案するコミット計画（順番・ファイル・メッセージ）

すべて新規ファイルのみ。`git add .` を使わず、下記のファイルを明示的に選ぶ。

1. `feat(discussion-structure): define offline records, configuration and focus filter`
   - `src/das/discussion_structure/{__init__,models,config,views,focus}.py`
   - `configs/discussion_structure.toml`
2. `feat(discussion-structure): add injectable probability and label backends`
   - `src/das/discussion_structure/{judgement,labels}.py`
3. `feat(discussion-structure): implement tree, stance, resolution and display rules`
   - `src/das/discussion_structure/{nodes,stances,resolution,display,tracker}.py`
4. `feat(discussion-structure): add replay artifacts and budgeted backend comparison`
   - `src/das/discussion_structure/{replay,metrics,compare}.py`
5. `test(discussion-structure): cover Japanese replay scenarios and stubbed APIs`
   - `tests/discussion_structure/{test_replay,test_backends}.py`
   - `tests/fixtures/discussion_structure/` の上記 10 ファイル
6. `docs(discussion-structure): document prototype usage and sandbox validation`
   - `src/das/discussion_structure/README.md`
   - この報告書

## 検証結果

すべて既存 `.venv/bin/python` を使用した。uv のホームキャッシュには書き込んでいない。

- 新規テスト: **51 passed**。通信はスタブで、実 API は呼ばない。
- 全体 pytest の作業前: **1082 passed / 30 failed / 4 skipped / 474 warnings**。
- 全体 pytest の作業後: **1133 passed / 30 failed / 4 skipped / 474 warnings**（92.07 秒）。
  作業前後の failed test ID 集合は完全一致、新規失敗 0 件。新規 warnings 0 件。
- 既存の 30 件は HTTP サーバーの bind がサンドボックスで拒否される
  `PermissionError: [Errno 1] Operation not permitted`。
  対象は `tests/unit/live/test_modes.py` の 1 件と `test_ui_api.py` の 29 件。
- 既存 warnings は `src/das/graph/store/networkx_store.py` の `datetime.utcnow()` の非推奨警告。
- 全体 ruff: **既存 8 件 / 新規 0 件**。作業前と作業後の出力は完全一致した。
  8 件はすべて `transcriber/server.py`（I001、F401、E402、SIM117）。全体コマンドの終了値は 1。
- 新規ディレクトリの ruff: **All checks passed**。

実行コマンド:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q tests/discussion_structure
.venv/bin/python -m ruff check src/das/discussion_structure tests/discussion_structure
.venv/bin/python -m das.discussion_structure.compare --backend scripted \
  --out data/discussion_structure/scripted
```

## 完了条件 1〜6

| 条件 | 結果 | 根拠 |
|---|---|---|
| 1 scripted 全シナリオ一致 | 達成 | 5 本・67 発話・最終 5 件と途中 24 件、合計 29 照合で不一致 0 |
| 2 規則の単体テスト・全体 pytest | 新規は達成、全体 green はサンドボックスのため未達 | 新規 51 件成功。既存の HTTP bind 失敗を作業前と比較 |
| 3 新しい ruff 警告を増やさない | 達成 | 既存 8 件のみ、出力完全一致 |
| 4 実議事録の OpenAI 再生 | 指示に従いスキップ | 入力正規化後、最初の通信で APIConnectionError。実 API 出力完了を主張しない |
| 5 OpenAI 全シナリオ比較 | 指示に従いスキップ | 条件 4 の通信失敗により実施せず。集計 CLI とスタブのテストは実装済み |
| 6 Jev キーなしエラー・スタブテスト | 達成 | キーなしは通信前に明確なエラー。公式 payload と確率、HTTP エラーをスタブで検証 |

| scripted シナリオ | 発話数 | 期待結果の照合数 | 不一致 |
|---|---:|---:|---:|
| upgrade_concerns | 10 | 6 | 0 |
| decisions_reopen | 13 | 7 | 0 |
| pending_additions | 11 | 5 | 0 |
| open_stances_focus | 19 | 7 | 0 |
| decision_guards | 14 | 4 | 0 |

条件 5 は未実行なので、一致率・実 API 遅延・較正の実測値はない。
scripted の一致率を OpenAI の数値として扱わない。
API 成功件数 0、usage で確認できた費用 USD 0。
接続失敗分の費用予約 USD **0.0005656** は請求有無不明として残した。
再試行は行わず、USD 2 上限を超える呼び出しはしていない。
Jev の実通信はキーがないため一度も行っていない。

出力はすべて Git 管理外の `data/discussion_structure/` にある。
実 API プローブの出力は途中までのログで、条件 4 成功の出力ではない。

## timeline.txt の抜粋（格上げ）

`data/discussion_structure/scripted/upgrade_concerns/timeline.txt`:

```text
[21.000s] display_commit (structure)
- 学食の環境対策 [open]
  ▶ 紙ストローに切り替えるか？ [open] ○A □B
    - いつから切り替えるか？ [open]
    - 紙の費用は問題ないか？ [open]

[25.000s] display_commit (structure, label, stance)
- 学食の環境対策 [open]
  ▶ ストローを何にするか？ [open]
    - いつから切り替えるか？ [open]
    - 紙の費用は問題ないか？ [open]
    - 紙ストロー [proposed] ○A □B
    - バイオプラ [proposed] ○D

[29.000s] display_commit (stance)
- 学食の環境対策 [open]
  ▶ ストローを何にするか？ [open]
    - いつから切り替えるか？ [open]
    - 紙の費用は問題ないか？ [open]
    - 紙ストロー [proposed] □B
    - バイオプラ [proposed] ○D ○A
```

同一 Issue ID のまま型とラベルを変え、立場の完全な記録を元の案に移す。
子の親と既存兄弟の相対順序は維持する。
紙の費用の子を紙の案へ移すことは `structure_proposal` にだけ記録し、自動適用しない。

## 仕様の曖昧さと採用した解釈

1. **問いの形の提案**: 05 §2 の例を優先し、「切り替えない？」でも単独提案なら応答待ち。
   明示的な全員向けの問いとは、追加した `trigger` 選択質問で区別する。
   聞き返し・進行・反語・合意確認は新規 Issue にしない。
2. **仕様 03 の選択肢だけでは作成規則の分岐が足りない**:
   `trigger`、`parent`、`response_to`、`switch`、`decision_answer` を独立した選択質問として追加。
   モデルが任意の木の編集命令を返す形にはしない。
3. **「既存か新規か拮抗」の幅**: 0.05 以内なら既存を優先。設定で変更可能。
4. **焦点遷移の数値**: 継続 prior 0.8、転換合図あり 0.45。
   移動は木の距離の二乗に反比例。観測は文字数 / 20 を 0.5〜2 に制限して重み付け。
   事後確率 > 0.7、2 発話以上かつ最初の候補観測から 10 秒で切替。
5. **応答待ちの期限**: 暗黙の問いだけ 60 秒（境界を含む）で失効。
   懸念・決定再開には規定の期限がないため保持。本人が懸念を解消したら待ち候補を外す。
6. **決定の同意**: 決定発話以降の別話者の明示的同意を必要とし、古い賛成印や沈黙で代用しない。
   異議は 10 秒の境界も含めて拒否する。採用先が特定されない choice/open の決定は適用しない。
7. **「しない」決定**: 明示的内容を `decision_answer` で保持し、`決定: しない` と記録できる。
   是非型の印の意味は従来どおり元の提案に対する立場として扱う。
8. **EOF の扱い**: EOF 後に 10 秒を捏造して決定しない。表示の未反映分だけ 3 秒規則で flush。
9. **表示 3 段**: 焦点の上の Issue 1 段、焦点、下 1 段を基本とする。
   Position は Issue の段数に数えず、上位の経路は breadcrumb に縮める。
   15 ノード上限では焦点経路を保護し、距離・解決状態・最終言及時刻の順で折りたたむ。
10. **話者確信度**: 新しい名前表示用の閾値は設けず、`unsure` を優先。
    安定 ID がない未確定話者は発話ごとの unknown ID として、他の未確定話者と混同しない。
11. **旧議事録**: AI の null 時刻は直前の終端へ置いて判定から除外し、正規化イベントに残す。
    人間の時刻欠損はエラー。重なった発話は安定な終端時刻順で処理する。
12. **ラベル生成待ち**: オフライン処理を逐次にし、生成完了まで次の発話を消費しない。
    未確定ラベルの仮表示はしない。違反時は一度再生成し、二度失敗したら構造提案へ。
13. **遅延評価**: ASR 確定時刻がない区間は null。
    表示・タイマーは議事録の仮想時間、判定・ラベル生成の所要時間は別の実測値として記録する。
    ライブの実時間の end-to-display 遅延に相当するものとしては報告しない。
14. **対数確率**: 上位候補外の質量は `__other__` に保持し、選択肢だけで再正規化して
    確信度を膨らませない。候補数 > 20 は階層判定にする。
15. **理由**: 懸念本文と根拠発話を内部に保存し、表示からは除外。
    理由の別途 LLM 要約は生成していない。

## 設計側への質問・次に決めてほしいこと

- 決定の「同意 1 件」は、決定発話より前の賛成を認めるか。
  「しない」決定では同意と、元の提案への賛成/懸念をどう別に表すか。
- 焦点 prior、観測重み、既存優先の幅、3 段の配分を実データで調整するか。
- 懸念・再開候補の失効、複数の未応答懸念の同一性をどう決めるか。
- 名前を出す話者確信度の閾値と、unknown ID しかない話者の応答認定をどうするか。
- ラベルの疑問形検査、個人名検査、内部理由要約の粒度をどこまで厳密にするか。
- 追加した atomic 判定項目のスキーマと、階層 logprobs の確率を評価設計で採用するか。
- 条件 4・5 をネットワーク利用可能な環境で実施してほしい。
  分類精度と構造生成誤差を分けるため、固定した正解文脈での比較も追加するか。

参照した外部 API の正本（2026-10-06 に確認）:
[TypeSafe HTTP API](https://docs.typesafe.ai/api)、
[OpenAI Chat API](https://developers.openai.com/api/reference/resources/chat)、
[GPT-4.1 mini モデル・単価](https://developers.openai.com/api/docs/models/gpt-4.1-mini)。
