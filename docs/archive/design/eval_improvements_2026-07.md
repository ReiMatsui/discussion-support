# 設計書: 評価系の改良とシミュ本実験の実行準備（トラックA）

実施者向け。根拠: `docs/review-2026-07-02/03_eval_cli.md`（High残り）、`docs/research/research_plan_2026-07.md` トラックA。品質基準は共通（コミット分割・全テスト通過・ruff・mypy増分ゼロ）。judge盲検化・citation照合・関係ラベル表示は修正済み（0a8fb6a, 02c7471）——本書はその続き。

## E1 `refactor(eval): 実行と採点を分離する（rescore-everything化）`

### 問題（03章 根本再設計）
`run_eval.py` の `_run_single` が会話生成と採点を一体で実行するため、judge/指標の修正のたびにAPI再ランが必要。judge修正済みの今、過去ランの再採点もできない。

### 修正
- 実行フェーズ: 会話生成のみを行い、**採点に必要な生データを全て保存**する（transcript、条件、提示ログ（受信者付き）、AFスナップショット、ペルソナ定義、seed）。保存形式は既存 run ディレクトリ構造を拡張（欠けているものだけ足す。何が欠けているかは `_run_single` を読んで特定）
- 採点フェーズ: 新CLI `das eval-rescore <run_dir>`（既存 `aqua-rescore` と同じパターン）が保存データから judge・構造指標・citation・stance集計を再計算し、summary を再生成する
- 既存の一体実行は「実行→即rescore」の合成として残す（利用者の手順は変わらない）

### テスト/受け入れ
- 同一runに対する rescore が決定的部分（構造指標・citation）で同一結果を返す
- judge プロンプトを変えて rescore すると judge スコアだけが変わる
- 旧形式 run ディレクトリを読んだ場合は明確なエラーメッセージ（黙って誤計算しない）

## E2 `fix(eval): ラン単位集計に変更する`

### 問題（03章 H-5前半）
ペルソナ×ランを独立サンプルとして pool し n を過大申告。
### 修正
summary の主観指標は「ラン内でペルソナ平均→ラン間で平均±SD」の2段集計に変更。ペルソナ別の内訳は参考値として残す。UI（Aggregates ページ）の表示も追随。
### テスト
2ラン×3ペルソナの固定データで手計算と一致すること。

## E3 `feat(eval): dose統制ablation条件 full_proposal_unlabeled`

### 問題（03章 H-5後半）
flat_rag（毎ターン3件）と full_proposal（トリガー時≤2件）は提示量・頻度が交絡し、関係ラベルの効果が分離できない。
### 修正
第4条件 `full_proposal_unlabeled` を追加: full_proposal と**同一のトリガー・同一の選定・同一の件数**で、提示文から関係ラベルだけを除去（`relation_label` ヘルパーの中立表記を全項目に使う）。conditions.py に既存 FullProposal のサブクラス/パラメータとして実装し、選定ロジックの複製を作らない。
### テスト
同一seedで full_proposal と unlabeled の提示タイミング・件数・対象が一致し、ラベル文字列のみ異なること。

## E4 `feat(eval): none/flat_rag への post-hoc AF構築（観測用グラフ）`

### 問題（03章 H-3）
構造指標（response_rate等）が none/flat_rag で常に0となり条件比較が成立しない。合意検出の停止規則も full_proposal のみ構造シグナルが効き、停止規則が処置と共変。
### 修正
- 全条件で、会話終了後に transcript から extraction/linking を実行して**観測用AF**を構築し（介入には一切使わない）、構造指標はこの観測用AFから全条件同一に計算する
- full_proposal は「介入用AF」と「観測用AF」を持つことになる——指標は観測用に統一（条件間で構築条件が揃うため）。介入用AFの指標は参考出力に格下げ
- `--until-consensus` の構造シグナルゲートも観測用AF基準に統一（停止規則の共変を解消）
### テスト
none 条件のrunで構造指標が非ゼロになる／full_proposal の指標が観測用AF由来になる。
### 注意
post-hoc 構築はAPIコスト増（1runあたり抽出+連結一式）。rescore フェーズ（E1）に置き、実行時ではなくバッチで走らせられるようにする。

## E5 実験実行手順（コードでなくREADME: `docs/research/experiment_runbook.md` を新設）

E1〜E4完了後の本実験手順を1ページに:
1. スモーク: `das eval <政策トピックpreset> -n 2` → judge reason を目視（盲検化後の分布確認）
2. 本実験: n=10 × 4条件（none / flat_rag / full_proposal / full_proposal_unlabeled）、`--until-consensus`、seed固定、予算見積もり（1ラン$0.5基準で約$20＋rescore）
3. rescore → summary → 図表出力（既存UI/Aggregates）
4. 結果の置き場: `data/eval/phase1-main/`
presetの正確な名前・オプションは presets.py を確認して記載すること（推測で書かない）。

## 実施順序
E2（小）→ E1（中・基盤）→ E3（小〜中）→ E4（中）→ E5（文書）。E1完了時点で一度区切って報告。
