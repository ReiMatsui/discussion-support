# 未決事項

## 1. 研究上の判断・共同研究者との調整

- `presentation_policy.md`（グラフは見せない）との関係。比較条件として扱うことの合意
  （[ADR-011](../decisions/ADR-011-presentation.md)）
- 人を対象とした評価の条件と規模（[11_evaluation.md](11_evaluation.md)）
- Controller への介入の種類の追加と、`feat/live-native` との統合の時期・方法（Rei との調整。
  [10_ai_voice.md](10_ai_voice.md)、[12_implementation.md](12_implementation.md)）

## 2. 確認待ちの提案

- 議題の「修正」と「次の議題へ」を UI の別操作に分けること（[02_data_model.md](02_data_model.md) §4）
- 格上げ時の立場の移し方（[05_nodes.md](05_nodes.md) §5）
- 焦点と解決状態の見せ分け（[09_display.md](09_display.md) §6）

## 3. 設計の細部

- 画面の描き方（字下げの一覧か、線で結ぶ図か）（[09_display.md](09_display.md) §8）
- 「更新中」の表示と「新」印の要否（[09_display.md](09_display.md) §3）
- ラベルの文体の規則（[05_nodes.md](05_nodes.md) §7）
- 並行して複数の話題が進む場合の焦点の扱い（[04_focus.md](04_focus.md)）
- 議題外の経過時間を表示し始める閾値（[08_drift.md](08_drift.md)）
- 立場の印に名前を出す話者の確信度の閾値（[06_stances.md](06_stances.md) §6）

## 4. 実測で決める値

| 値 | 現在の暫定値 | 仕様 |
|---|---|---|
| 判定モデルの選択 | Jev / Decisions API / 通常の LLM | [03](03_utterance_judgement.md) |
| 焦点の平滑化 | 事後確率 0.7、2 発話かつ 10 秒 | [04](04_focus.md) |
| 暗黙の問いの応答待ち | 60 秒 | [05](05_nodes.md) |
| 立場の印の閾値 | 確率 0.8 | [06](06_stances.md) |
| 決定の異議待ち | 10 秒 | [07](07_resolution.md) |
| まとめに入る閾値 | 終了 5 分前、出尽くし 3 分 | [07](07_resolution.md) |
| ラベル長 | 問い 20 字、案 16 字 | [01](01_overview.md) |
| 画面の反映間隔・段数・ノード数 | 3 秒、3 段、15 | [09](09_display.md) |
| 遅延目標 | 新しい問い p90 8 秒など | [01](01_overview.md) |
| GPT-Live の委任・指示の守り方 | — | [10](10_ai_voice.md) |
