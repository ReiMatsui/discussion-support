# 対面実験・実会議録音の当日手順（2026-10）

対面の参加者実験と、10 月のゼミ録音（実マイクでの精度測定）の両方で使う。
声の登録は当日その場で行い、本議論は登録済みの声を有効化して始める。
根拠: docs/research/enroll_eval_2026-09-28.md（登録で序盤の正解 70.6%→96.3%、
誤り 6.6%→0.4%）。

## 前日

```
uv run python scripts/preflight.py --device 内蔵
```

✗ が 1 つでもあれば直す。ReDimNet のキャッシュ警告が出たら一度起動して止める。
電源アダプタ、予備の録音（scripts/record_backup.py）も用意する。

## 当日、1 組ごと

### 1. 声の登録（5 分）

一人ずつ、その場のマイクで 30 秒話してもらって登録する（自己紹介や今日の
予定など、何でもよい。相槌ではなく話し続けてもらう）:

```
uv run python scripts/enroll_voices.py --voices voices.json --record 田中 --seconds 30
uv run python scripts/enroll_voices.py --voices voices.json --record 佐藤 --seconds 30
uv run python scripts/enroll_voices.py --voices voices.json --record 鈴木 --seconds 30
uv run python scripts/preflight.py --skip-mic --voices voices.json --expect 田中,佐藤,鈴木
```

「小さい」「クリップ」と出たら録り直す。録音は data/voices/<名前>.wav に残る
（後で別条件の再生評価にも使える）。voices.json には過去の登録も残っているが、
本議論では今日の 3 人だけを `--activate` で有効化するので照合には入らない。

同じ名前の人が別の日にもいる場合は上書きされる（その日の声で登録し直す）。

### 2. 練習議論（5 分）

本議論と同じ起動で 5 分話してもらう。目的は、参加者がシステムの声と
止め方（人が話し始めると止まる、「ちょっと待って」で止まる）に慣れること、
こちらが議事録の名前付けとスピーカー音量を確認すること。

```
uv run python -m das.asr.live --device 内蔵 --topic "練習: 今週あったこと" \
  --voices voices.json --activate 田中,佐藤,鈴木 --diarization-max-speakers 3 \
  --out transcripts/g01_practice.md
```

確認すること: 3 人の発言が最初の 1 分以内に本人の名前で出るか。出ないなら
登録の録音を確認して 1. に戻る。介入の声が聞こえ、参加者が話し始めたら
止まるか。止まらない・自分の声に反応して話し続けるなら、スピーカーの音量を
下げる（回り込み対策。AI の声紋は自動で登録され判定からは除かれるが、
GPT-Live 自身が自分の声を聞いて応じることがある）。

### 3. 本議論（30 分）

介入あり条件:

```
uv run python -m das.asr.live --device 内蔵 --topic "研究室の備品購入の優先順位" \
  --voices voices.json --activate 田中,佐藤,鈴木 --diarization-max-speakers 3 \
  --out transcripts/g01_main.md
```

介入なし条件（録音と話者同定だけ。LLM を使わない）:

```
uv run python -m das.asr.live --device 内蔵 --no-agent --no-llm \
  --voices voices.json --activate 田中,佐藤,鈴木 --diarization-max-speakers 3 \
  --out transcripts/g01_main.md
```

終了は Ctrl-C。transcripts/g01_main.{md,html,turns.jsonl,diag.jsonl,wav,
interventions.jsonl} が残る。wav は採点と介入の聞き直し（質問紙の後）に使う。

### 4. 質問紙、介入の聞き直し

replay（介入箇所の頭出し）で介入ごとに聞き直して評定してもらう。

## 事後: 話者同定の採点（ゼミ録音と実験の両方）

同じ wav を登録なしでも流して、登録あり／なしを同じ表で比べる:

```
uv run python -m das.asr.live --wav transcripts/g01_main.wav --no-agent --no-llm \
  --diarization-max-speakers 3 --out transcripts/g01_main_none.md
uv run python eval/annotate.py g01_main --minutes 10
uv run python eval/eval_speaker_gt.py eval/gt_g01_main.json
uv run python eval/eval_speaker_gt.py eval/gt_g01_main.json g01_main_none
uv run python eval/enroll_breakdown.py --gt eval/gt_g01_main.json \
  --none g01_main_none --enroll g01_main --minutes 10
```

正解付けは 10 分で 20〜30 分の作業。介入時点（遡及訂正前）の精度は
eval/decision_time.py。

## 未確認（リハーサルで潰す）

- 内蔵スピーカーの音量と回り込み: GPT-Live が自分の声に応じないか（A3〜A9）
- 30 分の連続運転: pyannote の再接続が起きたときにクラスタが分断しても
  登録済みの名前で戻るか（B）。2026-09-28 に 14 分の再生で確認済み: サーバ都合の
  切断（1011）が 3 回起きても再接続して続き、登録ありの精度は 97.9%（切断なしの
  ランと同じ）。切断の連鎖で分離が死ぬバグと、開始時に繋がらないと会議ごと
  落ちるバグは同日に修正（`yt_r25_enroll_2026-09-28.md` §切断）
- 「ちょっと待って」で本当に止まるか、止まったあと再開しないか（C）
- 介入なし条件の起動（--no-agent --no-llm）でも UI と録音が同じに残るか
