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
uv run pytest -q        # 単体テスト（実APIは呼ばない）
```

## 使い方

```bash
# 推奨構成（Soniox + pyannote + 声紋）。参加人数は指定しなくてよい
uv run python -m das.asr.live --diarization pyannote --vp-cluster-naming

# 統合AF構築＋ライブ介入まで含めたフル構成（--hybrid は上記構成の短縮形）
uv run das listen-soniox --hybrid

# 録音ファイルで再実験（マイク不要）
uv run das listen-soniox --hybrid --wav transcripts/<日時>.wav
```

起動するとブラウザUI（`http://127.0.0.1:8231/`）が開き、ライブ議事録・
モード切替（議事録のみ／AIと会話／人間に介入）・話者の名前登録・議題編集・
「新しい会議」・停止が行える。議事録（md / turns.jsonl / diag.jsonl / wav）は
`transcripts/` に自動保存される。

参加人数は設定しなくても動く（設定すると未確定が減る）。話者はプロファイル
ゼロから自動学習され、名前はUIから後付けできる。詳しい操作とオプションは
`docs/COMMANDS.md` を参照。

## 精度と検証

話者帰属の成績・採用済みの仕組み・却下済みの案は `docs/design/STATUS.md` が
正本（校正セットで文字正解 91.5%、人数未指定でも 85.5%）。経緯は
`docs/design/handoff_2026-07-14_unregistered_speakers.md`、研究記録は
`docs/research/` にある。採点・オフライン再生・正解アノテーションのツールは
`eval/`（案内は `eval/README.md`）。

## 開発

```bash
uv run ruff check .     # lint（既存エラーを増やさない）
uv run pytest -q        # 全テスト
```

- 実験・挙動変更は測定とセットで（1変更=1コミット、回帰テストを付ける）
- 再現と採点は `eval/_pipeline.py` だけを使う
- しきい値は `src/das/asr/live/_constants.py` が正本（校正表つき）
