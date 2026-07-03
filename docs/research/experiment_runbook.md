# 実験実行 runbook: シミュ本実験 (段階B / Phase 1)

E1〜E4 (評価系の改良) 完了後に、政策トピックでシミュ本実験を回すための手順。
コマンドとオプションは `src/das/cli/_eval.py` / `src/das/eval/presets.py` の現物に基づく
(2026-07 時点)。実行は必ずリポジトリルートで、`uv run das ...` として行う。

前提: `OPENAI_API_KEY` が設定済み。UI を使う場合は `uv sync --extra ui`。
出力ベースディレクトリは既定で `./data/eval` (`DAS_DATA_DIR` で変更可)。

---

## 1. スモーク (盲検化後の分布確認)

judge を C-1 修正 (盲検化) 済みの状態で、reason の分布を目視する。

```
uv run das eval policy_ai -n 2 \
  --conditions none,flat_rag,full_proposal,full_proposal_unlabeled \
  --eval-id smoke
```

- `policy_ai` プリセット = トピック「生成 AI を大学の講義・レポート作成で許容すべきか」、
  persona 4 名 (教員 X=con / 学生 Y=pro / 教育工学者 Z=neutral / 保護者 W=con)、
  文書サブディレクトリ `docs_policy`。
- 確認: `data/eval/smoke/<condition>/run_00N/judge_reports.json` の各 `*_reason` を読み、
  条件名に依存した誘導 (例: full_proposal だけ透明性が機械的に高い) が消えているか、
  reason が transcript の中身に即しているかを目視する。
- スモークで judge プロンプトを直したら、**会話を回し直さず** rescore で反映できる (手順 3)。

## 2. 本実験

n=10 × 4 条件。`--until-consensus` で合意到達までのターンも測る (`--max-turns` は安全上限)。

```
uv run das eval policy_ai -n 10 \
  --conditions none,flat_rag,full_proposal,full_proposal_unlabeled \
  --until-consensus --max-turns 20 \
  --stance-polling \
  --eval-dir data/eval --eval-id phase1-main \
  --budget 20 --hard-budget 30 \
  --cond-concurrency full_proposal=1,full_proposal_unlabeled=1
```

- 条件は 4 つ: `none` / `flat_rag` / `full_proposal` / `full_proposal_unlabeled` (E3 で追加)。
  `full_proposal_unlabeled` は full_proposal と同一選定・件数で関係ラベルのみ除いた
  dose 統制条件で、RQ4 (関係ラベルの寄与) の主検定に使う。
- 予算見積もり: 1 ラン $0.5 基準で 10×4=40 ラン ≒ **$20**。`--budget 20` は超過後の新規ラン
  開始を止める (in-flight は完走)。`--hard-budget 30` は API 呼び出しごと即停止の保険。
  linking がコストの 80-90% を占めるので、必要なら `--linking-model gpt-5-nano` で 70-80%
  削減できる (モデル名は model_update 反映後の値に合わせる)。
- `--cond-concurrency` で重い full_proposal 系を sequential にし、予算切れ時の部分結果を守る。

### 注意 (再現性・既知の制約)

- **seed 固定は現状 CLI に無い** (`--seed` オプションは未実装)。persona/judge は
  temperature ベースのサンプリングで完全な決定性は無い。再現性は meta.json に保存される
  全設定 (topic / personas / conditions / consensus_kwargs / temperature) と、後段の
  固定 rescore で担保する。厳密な seed 固定が必要なら別途 CLI 拡張が要る (要相談)。
- `--until-consensus` の停止判定は **実行時には介入用 AF のみ** 構造シグナルを使うため、
  条件間で停止規則が非対称になりうる (レビュー H-1)。E4 の観測用 AF 統一は rescore
  フェーズで効くので、**合意/収束の条件間比較は rescore 後の summary を正**とする。
  厳密に揃えたい場合は `--until-consensus` を外し固定ターンで回して time-to-consensus を
  事後解析にする選択肢もある (H-1(c))。

## 3. rescore → summary → 図表

本実験の transcript を、観測用 AF を全条件で後付け構築して再採点し summary を作り直す。

```
uv run das eval-rescore data/eval/phase1-main
```

- 既定で全条件の transcript から観測用 AF を構築し、構造指標と合意の構造シグナルを
  全条件同一に計算する (none/flat_rag でも構造指標が非ゼロになる, E4)。介入用 AF 由来の
  構造指標は run_meta の `structural_metrics_intervention_af` に参考として残る。
- judge プロンプトだけ直した場合など、構造を触らず再採点したいときは `--no-observation-af`。
- judge モデルを別系統にするなら `--model <judge_model>` (M-1: judge は persona と別モデル推奨)。
- rescore は summary.json を作り直す。主観指標はラン単位 2 段集計 (E2)、n はラン数。

図表 (Aggregates ページ) は既存 UI で確認する:

```
uv run das ui            # http://localhost:8501 を開き、phase1-main を選択
```

- 参照する主要出力: 主観評価 (ラン間 SD 付き)、合意形成、議論の中身の特徴 (構造指標)、
  提示情報の引用率 (RQ4)、立場の変化と見せかけ合意 (stance)。ペルソナ別内訳は
  「参考: ペルソナ別の内訳」expander で確認 (立場ごとの交互作用チェック)。

## 4. 結果の置き場

- 本実験の生成物・採点結果・summary は `data/eval/phase1-main/` にまとまる。
- `data/` は git 管理外 (成果物はコミットしない)。共有が必要なら summary.json と
  meta.json のみ別途アーカイブする。
