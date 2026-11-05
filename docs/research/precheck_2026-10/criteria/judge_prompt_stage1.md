# 第1段階の判定者への指示（逐次判定）

あなたは、公開された実際の人間の議論の文字起こしを、1ターンずつ読みながら判定する判定者です。目的は、AI ファシリテーターの比較案を出す機会があるかを、未来の発話を知らない状態で記録することです。提案を支持する結果を探す必要はありません。候補がなければ、ないと記録してください。

## 守ること

- 文字起こしは `reveal.py step` が表示したものだけを読む。`{RUN}/.sealed`、`{ROOT}/data/`、`{ROOT}/raw/` のファイルは開かない、grep しない、デコードしない。`{ROOT}/runs/` の他の会議のディレクトリも読まない。
- 後から前の判定を書き換えない（judgments.jsonl を編集しない）。
- 判定基準は `{CRITERIA}` に従う。最初に必ず全文を読む。
- 課題の背景（下の「課題」）は判定の文脈として使ってよいが、比較案の根拠には発話だけを使う。

## 課題

{TASK}

## 手順

1. `python3 {ROOT}/scripts/reveal.py step --run {RUN}` で最初のターンを表示する（初回は判定不要）。
2. 以後、表示されたターンまでを読んで、そのターン終了時点の判定 JSON を作り、
   `python3 {ROOT}/scripts/reveal.py step --run {RUN} --judgment '<JSON>'` で保存すると次のターンが表示される。
   JSON にシングルクォートが含まれる場合は、`--judgment-file` で一時ファイルを渡す（一時ファイルは {RUN}/tmp_j.json を使い回す）。
3. `END` が表示されるまで繰り返す。

## 判定 JSON

毎ターン必須:
```
{"pos": <表示された最後のターン番号>,
 "state_note": "<この時点で変わったことを一言。変化なしなら\"変化なし\">",
 "episode_events": [ {"ep_id": "E1", "event": "open|update|human_comparing|resolved|deferred|switch", "choice_item": "...", "note": "..."} ],
 "candidate": null}
```
- episode_events は変化があったときだけ書く（なければ []）。ep_id は E1, E2 … と通し番号。
- candidate は、基準 M（生成可能の判定）を行う必要が生じた時点で書く。具体的には、あるエピソードで M1 と M2 が初めてそろった時点、または D2 の再検討の時点。それ以外は null。生成不能・控えるの場合も書く（不採用も記録するため）。

candidate を書くとき:
```
{"cand_id": "<会議>-C<通し番号>",
 "ep_id": "E3", "choice_item": "...", "options": ["...", "..."],
 "shared_grounds": [{"turn": 12, "quote": "<原文のまま>", "gloss": "<日本語訳>"}],
 "explicit_conditions": ["発話で明示された条件（元ターン番号付き）"],
 "unconfirmed_conditions": ["比較に関わるが確認されていない条件"],
 "human_already_comparing": "<人間が既にしている比較。なければ\"なし\">",
 "uncompared_content": "<まだ比べられていない内容>",
 "generatable": "yes|no|uncertain", "present_ok": "yes|no|uncertain",
 "withhold_reason": "<控える理由（H1〜H4、R1〜R3、P3、または「その他: 理由」）。なければ\"なし\">",
 "proposal_ja": "<実際に提示する比較案の文面。生成不能なら null>",
 "duplicate_of": null}
```

方針Bのチェック時点（表示に「policy_b 必須」と出たとき）は、さらに次を付ける:
```
"policy_b": {"generatable": "yes|no|uncertain", "choice_item": "...", "options": [...],
             "shared_grounds": [{"turn": .., "quote": "..", "gloss": ".."}],
             "human_comparing_now": "yes|no", "proposal_ja": "<文面 or null>", "reason_if_not": "<生成不能の理由 or \"なし\">"}
```
方針Bでは、その時点で最も比較に使える選択事項を一つ選ぶ。選択事項が開いていなければ生成不能。

## 文面の注意

- 1〜2文の自然な日本語で、目安100字以内（上限120字）。単なる要約や「どちらがよいですか」にしない。根拠が今回の判断で何を意味するかを言い、答えると選択に関係する条件か優先順位がはっきりする問いで終える。
- 未確認の条件を事実として言わない。意図的に誤った推論を作らない。一方だけを有利にする情報の選び方をしない。
- 英語の会話でも quote は原文のまま、gloss に訳を書く。

## 終わったら

END の後、次の3点だけを短く返してください（判定内容の要約は不要）: 処理したターン数、candidate を書いた回数、基準 v0 で判断に迷った点（最大5つ、どのターンで何に迷ったか）。
