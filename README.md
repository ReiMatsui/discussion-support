# discussion-support

対面議論のリアルタイム議事録＋話者特定＋AIファシリテーション (das = Discussion
Argumentation Support)。Soniox のリアルタイム文字起こしに、pyannote の話者分離と
ReDimNet の声紋照合を重ねて「誰が何を言ったか」を特定し、AIファシリテーターが
脱線戻し・声かけ・事実確認で議論を支援する。

## セットアップ

```bash
uv sync --all-extras
cp .env.example .env    # SONIOX_API_KEY / OPENAI_API_KEY /
                        # PYANNOTEAI_API_KEY（--hybrid 用）を設定
uv run das check       # 普段の基本確認（実API向けのテストを除外）
```

## 使い方

3人での試行を、カメラ確認・録画・議論の開始まで画面から行う場合は次を使います。

```sh
uv run das pilot
```

画面を開くだけでは撮影や外部送信は始まりません。支援ありを選ぶ前に、有効なAPIキーで接続確認が必要です。
[11月12日の実施手順・配布資料](docs/pilot_2026-11-12/README.md) に、30分の進め方と使用量制限、未確認事項をまとめています。

現在の研究用モードを起動する基本コマンドは次です。

```sh
uv run das study
```

`uv run das study --condition none` で無支援条件を起動できます。
本体はマイクと外部APIを使用します。接続確認スクリプトの累計3回・費用予算は本体には適用されません。
従来の統合システムは次のコマンドで起動します。

```bash
# 従来の統合システム（既定で推奨構成: Soniox + pyannote + 声紋。人数指定は不要）
uv run das listen-soniox

# 書き起こし＋話者特定だけ（統合AF構築なし）
uv run python -m das.asr.live

# 録音ファイルで再実験（マイク不要）
uv run das listen-soniox --wav transcripts/<日時>.wav

# 検証向け: AI介入・論点抽出などLLMを一切使わない（トークン消費ゼロ）
uv run das listen-soniox --wav <ファイル>.wav --no-intervention

# pyannoteを使わないSoniox単独（PYANNOTEAI_API_KEY が無い環境でも起動は止まらず
# 自動でこの構成に縮退する）
uv run python -m das.asr.live --diarization none
```

起動するとブラウザUI（`http://127.0.0.1:8231/`）が開き、ライブ議事録・
モード切替（議事録のみ／AIと会話／人間に介入）・話者の名前登録・議題編集・
「新しい会議」・停止が行える。議事録（md / turns.jsonl / diag.jsonl / wav）は
`transcripts/` に自動保存される。

参加人数は設定しなくても動く（設定すると未確定が減る）。話者はプロファイル
ゼロから自動学習され、名前はUIから後付けできる。詳しい操作とオプションは
`docs/COMMANDS.md` を参照。

## 進捗と資料

現在の実装・未検証事項は [進捗一覧](docs/progress/STATUS_2026-11-05.md)、
研究記録と発表資料の入口は [資料の索引](docs/README.md) を参照してください。

## 精度と検証

話者帰属の成績・採用済みの仕組み・却下済みの案は `docs/design/STATUS.md` が
正本（校正セットで文字正解 91.5%、人数未指定でも 85.5%）。経緯は
`docs/design/handoff_2026-07-14_unregistered_speakers.md`、研究記録は
`docs/research/` にある。採点・オフライン再生・正解アノテーションのツールは
`eval/`（案内は `eval/README.md`）。

## 開発

```bash
uv run ruff check .     # lint（既存エラーを増やさない）
uv run das check        # 普段の基本確認
uv run das check study  # 現在の研究用モード
uv run das check all    # 実API向けを除く全自動テスト
```

詳細は [テストの入口](tests/README.md) と [コードの見取り図](docs/development.md) を参照してください。

- 実験・挙動変更は測定とセットで（1変更=1コミット、回帰テストを付ける）
- 再現と採点は `eval/_pipeline.py` だけを使う
- しきい値は `src/das/asr/live/_constants.py` が正本（校正表つき）

## 修論実験向けの採否判断支援

通常の介入と分離した実験用モードを追加しています。起動と未検証事項は
`docs/design/study_mode_2026-11-03.md` を参照してください。

```sh
uv run das study --config study/facility.json --condition system
uv run das study --config study/facility.json --condition none
```

研究者用画面は `/`、参加者用の理由記録は `/study` です。課題・報酬・終了手順の
妥当性や、人への支援効果はまだ確認していません。

実験前の動作検査と現在の未確認事項は
`docs/design/study_mode_verification_2026-11-05.md` を参照してください。
人工入力による連続動作の検査は、実API・録音・再生なしで実行できます。

実LLMの確認を手動で少量ずつ行う場合は、送信なしが既定の
`python -m das.study.bounded_probe` を使用してください。
累計3回まで・自動再試行なしの手順は
`docs/design/bounded_llm_probe_2026-11-05.md` にあります。
この制限は通常の実験システムには適用されません。
