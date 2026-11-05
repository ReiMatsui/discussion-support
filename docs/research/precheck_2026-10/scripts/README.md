# 再現手順

作業フォルダを `precheck/` とする。Python 3.10 以上が必要で、追加のパッケージは要らない。

## 1. 取得
```
mkdir -p raw/gap raw/ami data runs out
curl -o raw/gap/gap-corpus.zip https://zissou.infosci.cornell.edu/convokit/datasets/gap-corpus/gap-corpus.zip
(cd raw/gap && unzip gap-corpus.zip)
curl -o raw/ami/ami_manual.zip https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
(cd raw/ami && unzip ami_manual.zip)
```

## 2. 発話列の作成（同じ話者の連続をターンにまとめ、元の ID を保持）
```
python3 -I scripts/prep.py gap:1 gap:7 gap:8 gap:18 gap:24 ami:ES2002c ami:ES2003c ami:ES2004c
```

## 3. 第1段階（逐次判定。未読の発話を表示しない）
```
python3 -I scripts/reveal.py init --meeting GAP07 --run runs/test/GAP07
python3 scripts/make_prompt.py GAP GAP07 runs/test/GAP07 criteria_v1.md > runs/test/GAP07/prompt.md
```
`prompt.md` を判定者（今回は Claude のサブエージェント）に渡す。判定者は次のコマンドを繰り返す。
```
python3 scripts/reveal.py step --run runs/test/GAP07 --judgment '<JSON>'
```
判定は `judgments.jsonl` に追記され、表示した発話は `shown.log` に残る。

## 4. 第2段階（候補を固定した後に後続30ターンを分類）
```
python3 -I scripts/stage2_pack.py runs/test/GAP07
python3 scripts/make_prompt2.py runs/test/GAP07 > runs/test/GAP07/prompt2.md
```
出力は `stage2_output.json`。

## 5. 品質評価（生成に関わらない判定者を2つ）
```
python3 -I scripts/quality_pack.py runs/test/GAP07
python3 scripts/make_prompt_q.py GAP runs/test/GAP07 J1 > runs/test/GAP07/prompt_q_J1.md
```
J2 も同様。出力は `quality_J1.json` と `quality_J2.json`。

## 6. 集計
```
python3 -I scripts/aggregate.py
python3 -I scripts/report_stats.py
```
出力は `out/candidates_all.csv`、`out/per_meeting.csv`、`out/quality_agreement.json`。

## 判定基準と指示文

- `criteria/criteria_v0.md`: 基準整備用の草案
- `criteria/criteria_v1.md`: 固定版。変更履歴は末尾
- `criteria/criteria_v1.sha256`: 固定時のハッシュ
- `criteria/judge_prompt_stage1.md`、`judge_prompt_stage2.md`、`judge_prompt_quality.md`: 判定者への指示

## CSV の主な列（candidates_all.csv）

- **識別:**
  - `kind`: A = 内容に基づく開始候補、B = 方針Bのチェック時点
  - `pos`: 判定したターン
  - `time_s`: そのターンの終了時刻
- **根拠:**
  - `ground_turns` / `ground_source_ids`: 根拠の発話（ターン番号と元データの ID）
  - `grounds_quote_original`: 原文
  - `grounds_gloss_ja`: 訳
- **第1段階の判定:** `generatable`、`present_ok`、`withhold_reason`、`proposal_ja`
- **第2段階の分類:**
  - `S1_*`〜`S5_*`: 最初に当てはまったターン、候補からのターン差・秒差
  - `S6`: 決定・保留・話題転換とそのターン
  - `answers_question`: 問いへの答えにあたる情報が自然に出たか
  - `update_type`: 実質的な更新か言い換えか
  - `update_example_offline_ja`: 観測された情報を入力した場合のオフライン動作例
- **品質評価:**
  - `Q1_J1`〜`Q8_J2`: 各判定者の評価
  - `Qn_final`: 保守的に統合した値
  - `Q_fail_items` / `Q_uncertain_items`: 満たさない項目と不確かな項目
