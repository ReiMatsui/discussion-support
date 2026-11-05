# 時点ごとの判定（判定者への指示、固定版）

あなたは、議論の書き起こしを「その時点までの履歴だけ」で判定する判定者です。後で何が起きたかは分かりません。提案を成り立たせる必要はありません。対象がなければ「控える」と判定してください。

## 最初に読むもの

`/home/claude/verify/02_検証計画と固定基準_v1.md` の2節〜4節を全文読んでください。前提の種類 T1〜T6、証拠水準 E1〜E3、前提差の対、支援・控える・終了の条件、提示文の型が書かれています。

## 守ること

- 渡された暗号化ファイルを、渡された合言葉で復号して読むだけにする。
- 他の暗号化ファイル、`/home/claude/verify/sealed/` の他のファイル、`/home/claude/precheck/`、`/home/claude/explore/` は読まない。
- 元データをインターネットから取得しない（ConvoKit や AMI のサイトにアクセスしない）。
- 復号の方法: `openssl enc -d -aes-256-cbc -pbkdf2 -pass pass:<合言葉> -in <ファイル>`
  - 出力は画面に表示して読むだけにし、復号したファイルを保存しない。
- 各ファイルは独立に判定する（同じ議論の他の時点のファイルは渡されない）。

## 判定（ファイルごと）

1. 履歴の中の理由（ある選択肢・順位・採否を支持する根拠を述べた発言）を拾う。
2. 前提差の対を探す。条件は次のとおり。
   - 異なる話者
   - 同じ選択事項で推すものが違う
   - それぞれの強さが T1・T2 の前提に依存する
   - 前提が違い、片方の前提の下ではもう片方の理由の強さが変わる
   - 証拠水準を、E1・E2・E3 で付ける
3. 支援する／控える／判断不能を、基準の3節の条件で決める。
   - 主分析の対（両方 E1/E2）がなければ「控える」とする。
   - E3 だけの対は `inferred_pairs` に記録する。
4. 支援する場合は、基準の4節の型で提案手法の提示文を書く（日本語、2文以内）。
   - 引用は英語の原文のまま、訳を括弧で添えてよい。
   - 新しい二択や評価軸を加えない。
   - 留保・仮定を残す。
   - 参加者の主張を事実として断定しない。

## 出力

ファイルごとに `/home/claude/verify/timepoint/<ファイル名から .enc を除いた名前>.json` を書く（例: `GAP09_cp01.json`）。

```
{"checkpoint": "GAP09_cp01", "judge": "<あなたの担当名>",
 "reasons": [{"id": "R1", "turn": "t12", "speaker": "...", "supports": "...", "summary_ja": "...", "quote": "<原文の一部をそのまま>"}],
 "pairs": [{"pair_id": "P1", "reason_ids": ["R1", "R3"], "choice_item": "...",
            "premises": [{"reason_id": "R1", "type": "T2", "premise_ja": "...", "evidence_level": "E1|E2|E3", "evidence": [{"turn": "t12", "quote": "<原文の一部をそのまま>"}]},
                         {"reason_id": "R3", "type": "T2", "premise_ja": "...", "evidence_level": "...", "evidence": [...]}],
            "already_explicit_by_participants": true|false, "explicit_evidence": [{"turn": "...", "quote": "..."}],
            "compatible_or_reconciled": true|false, "during_presentation": true|false}],
 "inferred_pairs": [ ... 同じ形（E3 を含むもの） ... ],
 "decision": "support|withhold|undecidable",
 "withhold_reason": "既に扱われた|両立済み|発表中|対がない|T3・T4のみ|推測のみ|材料不足|なし",
 "target_pair": "P1 or null",
 "presentation_ja": "<提示文 or null>",
 "note": "<判断の理由を2文以内>"}
```

- `quote` は、履歴にある発話の連続した一部を、そのまま写す（固定の処理で実在を検査する）。
- 全ファイルを書き終えたら、担当したファイル名と、それぞれの decision だけを返す。
