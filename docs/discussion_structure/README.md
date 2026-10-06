# リアルタイム議論構造（議論グラフ）

最終更新: 2026-10-06
担当: 高田
状態: 設計検討中（実装なし）

対面議論の音声から、操作者なしでリアルタイムに「問いと案の木」を構築し、参加者と
AI ファシリテーターに提示する仕組みの設計文書一式である。

> 操作者なしで動く、リアルタイムの対面議論の議論グラフ

## 文書の構成

```text
docs/discussion_structure/
├── README.md              この文書（入口・表記・読む順番）
├── spec/                  仕様（何をするか）
├── decisions/             設計判断の記録（なぜそうしたか。ADR）
├── research/              先行研究のまとめと文献一覧
└── work/                  Codex への作業依頼（受け渡しの記録）
```

### 仕様（`spec/`）

| 文書 | 内容 |
|---|---|
| [01_overview.md](spec/01_overview.md) | 目的、全体像、要求、遅延目標、設計原則 |
| [02_data_model.md](spec/02_data_model.md) | ノード、問いの答えの型、木、議題、解決状態、立場、記録 |
| [03_utterance_judgement.md](spec/03_utterance_judgement.md) | 発話ごとの判定（速い層・遅い層）、判定モデル、入力の範囲 |
| [04_focus.md](spec/04_focus.md) | 焦点（いま何を議論しているか）の追跡 |
| [05_nodes.md](spec/05_nodes.md) | ノードの作成・同一性・配置・格上げ・ラベル |
| [06_stances.md](spec/06_stances.md) | 立場（賛成・懸念）の判定・更新・表示 |
| [07_resolution.md](spec/07_resolution.md) | 解決状態の判定と、まとめの段階 |
| [08_drift.md](spec/08_drift.md) | 議論と無関係な話（脱線）の検知 |
| [09_display.md](spec/09_display.md) | 画面（静かな画面の規則と表示要素） |
| [10_ai_voice.md](spec/10_ai_voice.md) | AI ファシリテーターの声（GPT-Live） |
| [11_evaluation.md](spec/11_evaluation.md) | 評価 |
| [12_implementation.md](spec/12_implementation.md) | 実装方針、QUD 試作との関係、段階 |
| [open_issues.md](spec/open_issues.md) | 未決事項 |

### 設計判断の記録（`decisions/`）

各判断について、背景・選択肢・決定・理由・根拠・影響を記録する。一覧と決定の日付は
[decisions/README.md](decisions/README.md)。

### 先行研究（`research/`）

| 文書 | 内容 |
|---|---|
| [related_work.md](research/related_work.md) | 先行研究のまとめ（分野ごとの流れと、本研究の位置づけ） |
| [references.md](research/references.md) | 文献一覧（各文献から使った内容と、原文の確認範囲） |

### 作業依頼（`work/`）

| 文書 | 状態 |
|---|---|
| [2026-10-06_offline_prototype.md](work/2026-10-06_offline_prototype.md) | オフライン試作（議事録の再生から木を作る）。完了（報告: [report](work/2026-10-06_offline_prototype_report.md)） |
| [2026-10-06_rules_update.md](work/2026-10-06_rules_update.md) | 試作の規則を 2026-10-06 の決定に合わせる。完了（報告: [report](work/2026-10-06_rules_update_report.md)） |
| [2026-10-06_rules_update2.md](work/2026-10-06_rules_update2.md) | 試作の規則の追加更新（決定の対称化ほか）。完了（報告: [report](work/2026-10-06_rules_update2_report.md)） |
| [2026-10-06_rules_update3.md](work/2026-10-06_rules_update3.md) | 決定者本人の決定の発話を本人の同意として扱う |

## 読む順番

1. [spec/01_overview.md](spec/01_overview.md) で全体をつかむ
2. 関心のある機能の仕様を読む
3. 「なぜそうしたか」は各仕様の冒頭にある ADR へのリンクから辿る
4. 研究上の位置づけは [research/related_work.md](research/related_work.md)

## 表記

仕様の各節の見出しに状態を付ける。

| 表記 | 意味 |
|---|---|
| 【合意】 | 高田と合意済み。変更するときは ADR を追加・更新する |
| 【暫定値】 | 方針は合意済みだが、数値は実データで見直す |
| 【検討中】 | 方針がまだ決まっていない |
| 【未確認の提案】 | 設計側（Claude Code）の提案で、まだ確認を取っていない |

## 運用

- 設計は Claude Code、実装は Codex が担当する。本ディレクトリを受け渡しの正本とし、受け渡しは
  ブランチ名とコミット ID で行う。
- 作業ブランチは `takada/discussion-structure`（`main` から作成）。push と PR は高田の指示が
  あるまで行わない。
- 共同研究者のブランチ（`origin/feat/live-native` など）は変更しない。

## 関連する既存文書

| 文書 | 関係 |
|---|---|
| `docs/design/QUD_INQUIRY_DESIGN.md`（`takada/qud-inquiry` ブランチ） | 本設計の前身の QUD 試作 |
| `docs/design/presentation_policy.md` | 「グラフは見せない」既存方針（[ADR-011](decisions/ADR-011-presentation.md)） |
| `docs/design/STATUS.md` | 話者帰属の精度の正本 |
| `docs/research/RESEARCH.md`、`docs/research/significance_2026-07.md` | 研究全体の RQ と仮説 |
| `docs/research/thesis_path_online_2026-09.md` | 修論の方針（オンライン視聴実験） |
| `docs/research/gpt_live_api_2026-09.md`（`origin/feat/live-native`） | GPT-Live の仕様と実測 |
