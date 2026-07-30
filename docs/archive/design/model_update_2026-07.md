# 設計書: LLM既定モデルの更新（小規模）

実施者向け。根拠: `docs/review-2026-07-02/05_external_tools.md`（要対応1: gpt-5-mini/nano は2世代前。現行 GPT-5.4 mini/nano へ）。品質基準は共通。

## M1 `chore(llm): 既定モデルを GPT-5.4 系へ更新`

1. `settings.py` の `openai_model_fast` / `openai_model_smart` 既定を `gpt-5.4-mini` へ（正確なモデル文字列は実装時に OpenAI ドキュメントで確認——05章調査時点の記載は mini $0.75/$4.50, nano $0.20/$1.25。**「gpt-5.5-mini」は存在しない**ことに注意）
2. `llm/cost.py` の料金表に 5.4 系を追加（旧5系の行は残す——過去runのrescore用）
3. `_build_chat_params` 系のモデル名分岐（`name.startswith("gpt-5")` 等）が 5.4 系でも正しい枝に入るか確認（reasoning_effort / max_completion_tokens の扱い）。`_bootstrap.py` と `llm/openai_client.py` の両方にある点に注意
4. `.env.example` のコメント更新

## M2 回帰確認（API課金あり・小規模）

- 単体テスト全通過（フェイクLLMなのでモデル名非依存のはず。ハードコードがあれば修正）
- APIスモーク: `das eval <preset> -n 1` を新旧モデルで1回ずつ回し、(a) structured output のパース失敗が起きないこと (b) triage/fact/drift の judge 系がJSONを返すこと (c) コスト集計が新料金で出ること、を確認。差分の目視で明らかな品質劣化がないかを見る（厳密比較は本実験のスモークに委ねる）
- live 系は `--simulate` で1セッション（triage・summarize が動くこと）

## 注意
- Realtime（`gpt-realtime-2`）と embedding は**変更しない**（05章で現行妥当と判定済み）
- モデル名は env で上書き可能なので、ユーザーの .env に旧名が残っていると効かない——完了報告に「.env の OPENAI_MODEL_FAST/SMART を消すか更新」と明記すること
