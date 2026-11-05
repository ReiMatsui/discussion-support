# 第2段階の判定者への指示（候補を固定した後に後続を調べる）

入力ファイル `/home/claude/precheck/runs/test/GAP18/stage2_input.json` には、第1段階で固定した候補（kind=A: 内容に基づく開始候補、kind=B: 手順上のチェック時点）ごとに、候補時点までの直近の文脈（context）と、その後の 30 ターン（following）が入っています。候補の中身（比較案 proposal_ja を含む）は第1段階のもので、変更してはいけません。

後続発話は AI への応答ではありません（実際の会議に AI はいません）。確認できるのは「更新に使える情報が自然な会話に現れたか」までです。「AI が介入すればこの応答が得られた」とは書かないでください。

## 各候補について判定すること

following を読み、次の分類を当てはまるものすべて付けてください。各分類には最初に当てはまったターン番号を書きます。

- S1 人間が同じ比較を自力で始めた: 比較案と同じ選択肢を、同じ観点で結びつけて比べる発言が人間から出た。
- S2 前提を訂正する情報: 比較案が前提にした事実・条件・選択肢が違う、または成り立たないと分かる発言。
- S3 新しい条件・根拠で比較の意味が変わった: 比較案になかった条件や根拠が出て、どちらがよいかの比べ方が変わる。
- S4 優先順位の発言: どの観点や条件を重く見るかの発言。
- S5 同じ根拠の繰り返しだけ: 既出の根拠を言い直しているだけ。
- S6 決定・保留・話題転換: decided / deferred / switched のどれかと、そのターン。
- S7 該当なし: S1〜S5 のどれもない。

さらに次を記録してください。

- answers_question: 比較案の最後の問いに答えにあたる情報が、following の中に自然に出たか（yes / partial / no）と、そのターン。
- update_type: S2 か S3 がある場合、その情報を入力して比較案を作り直すと、内容が実質的に変わるか（substantive: 推論の撤回・前提の変更・比べる観点の変更）、言い換えにとどまるか（rephrase）、更新不要か（none）。
- update_example_ja: update_type が substantive の場合だけ、「観測された情報を入力した場合のオフライン動作例」として、更新後の比較案を1〜2文で書く。それ以外は null。
- note: 判断の理由を一言。

## 出力

`/home/claude/precheck/runs/test/GAP18/stage2_output.json` に、次の形の JSON を書き出してください（候補の順は入力どおり）。
```
{"meeting": "...", "items": [
  {"id": "...", "kind": "A|B",
   "labels": {"S1": <turn or null>, "S2": <turn or null>, "S3": <turn or null>, "S4": <turn or null>, "S5": <turn or null>, "S6": {"type": "decided|deferred|switched", "turn": <turn>} or null, "S7": true|false},
   "answers_question": {"value": "yes|partial|no", "turn": <turn or null>},
   "update_type": "substantive|rephrase|none",
   "update_example_ja": "... or null",
   "note": "..."}
]}
```
書き終えたら、件数だけを返してください。入力ファイル以外の文字起こし（data/、raw/、.sealed）は読まないでください。

