# YouTube 討論で「別の動画から登録した声」の話者同定を測る手順（2026-09-28）

千葉コーパスの検証（enroll_eval_2026-09-28.md）は同じ収録の声で登録していた。
ここでは別の動画（単独出演）から登録し、3 人討論に流す。対面実験の
「練習議論で登録→本議論」より条件は厳しい（別日・別マイク）。

## 素材の候補

3 人討論（本番）: ReHacQ の生配信。出演者 3 人とも単独の動画が豊富。

- ひろゆき × 高橋弘樹 × 西田亮介: https://www.youtube.com/watch?v=pA4KCoXaX_4
- 予備: ひろゆき × 斎藤幸平 × 高橋弘樹: https://www.youtube.com/watch?v=Skj9lUjLar0

登録用（単独で話している動画。各人 30〜60 秒のきれいな部分を切り出す）:

- 高橋弘樹: まったり雑談生配信（一人）https://www.youtube.com/watch?v=etA0wdL68mU
- ひろゆき: 本人チャンネル hiroyuki の一人配信ならどれでもよい
  https://www.youtube.com/channel/UC0yQ2h4gQXmVUFWZSqlMVOA
- 西田亮介: 対談動画から西田さんが一人で話している区間を切る
  例 https://www.youtube.com/watch?v=jc5jg6Orh5I（安野貴博との対談）

切り出す区間は聴いて決める。相槌や重なりのない、本人だけが 30 秒以上
続けて話しているところ。BGM やナレーションが乗っている部分は避ける。

## 手順（Mac）

音声の取得と 16 kHz mono への変換:

```
mkdir -p data/yt/clips
yt-dlp -x --audio-format wav -o "data/yt/%(id)s.%(ext)s" https://www.youtube.com/watch?v=pA4KCoXaX_4
yt-dlp -x --audio-format wav -o "data/yt/%(id)s.%(ext)s" https://www.youtube.com/watch?v=etA0wdL68mU
ffmpeg -i data/yt/pA4KCoXaX_4.wav -ar 16000 -ac 1 data/yt/rehacq_pA4K_16k.wav
```

登録用の切り出し（開始秒と長さは聴いて決める。例: 90 秒目から 60 秒）:

```
ffmpeg -ss 90 -t 60 -i data/yt/etA0wdL68mU.wav -ar 16000 -ac 1 data/yt/clips/takahashi.wav
ffmpeg -ss 120 -t 60 -i data/yt/<ひろゆきの動画>.wav -ar 16000 -ac 1 data/yt/clips/hiroyuki.wav
ffmpeg -ss 300 -t 60 -i data/yt/jc5jg6Orh5I.wav -ar 16000 -ac 1 data/yt/clips/nishida.wav
```

声紋の登録:

```
uv run python scripts/enroll_voices.py --voices data/yt/voices_rehacq_pA4K.json \
  --add 高橋=data/yt/clips/takahashi.wav \
  --add ひろゆき=data/yt/clips/hiroyuki.wav \
  --add 西田=data/yt/clips/nishida.wav
```

再生ラン（本番の先頭 10 分を切り出して流す。2 条件）:

```
ffmpeg -t 600 -i data/yt/rehacq_pA4K_16k.wav data/yt/rehacq_pA4K_10min.wav
uv run das listen-soniox --no-intervention --wav data/yt/rehacq_pA4K_10min.wav \
  --soniox-args "--voices data/yt/voices_rehacq_pA4K.json --activate all --out transcripts/rehacq_pA4K_enroll.md"
uv run das listen-soniox --no-intervention --wav data/yt/rehacq_pA4K_10min.wav \
  --soniox-args "--out transcripts/rehacq_pA4K_none.md"
```

登録なしの条件で人数を与えるかは §49.6 の知見（開始前の雑音があると
人数なしが優位）に従い、まず与えない。与える場合は `--max-speakers 3`。

## 採点

正解付け（片方のランに付ければ、もう片方は同じ音声なので時間で突合できる）:

```
uv run python eval/annotate.py rehacq_pA4K_none --minutes 10
uv run python eval/eval_speaker_gt.py eval/gt_rehacq_pA4K_none.json
uv run python eval/eval_speaker_gt.py eval/gt_rehacq_pA4K_none.json rehacq_pA4K_enroll
```

10 分ぶんの正解付けは 20〜30 分の作業。誤りだけをマークする 8 月の方式
（下界）でもよいが、登録あり／なしの未確定率を比べるには正解付けが要る。
時間帯別の内訳は eval/enroll_breakdown.py を YouTube 用に一般化して出す（未着手）。

## この検証で分かること・分からないこと

- 分かる: 別日・別マイクで録った声を登録しても、討論の序盤から本人に
  付くか。付くなら対面実験の登録手順は「同じ部屋で 5 分前」より緩くてもよい。
- 分からない: 会議室の卓上マイクの音響。これは 10 月のゼミ録音で測る
  （そのときも練習 5 分で登録してから本議論を流し、登録あり／なしを比べる）。
