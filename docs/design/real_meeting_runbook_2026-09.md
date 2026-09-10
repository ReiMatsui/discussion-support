# 実会議の録音と同定の測定 手順書（2026-09）

> 目的: ゼミ等の実際の会議で、システムが落ちても録音が残り、後から日常会話
> コーパスと同一手順で話者同定の精度を測れる状態を作る。介入はまだ出さない。
> 測定方法は 7月の `mic_verification_runbook_2026-07.md` と同じ（変えていない）。
> 用語は「話者同定」に統一する（原稿と同じ。旧文書の「帰属」は同じ意味）。

## 0. 前提として知っておくこと

本体（`python -m das.asr.live`）が残す `transcripts/<日時>.wav` は **STT へ送れた
音声だけ** が入る（接続断の間の音声は入らない。入れると発話 ms とずれて採点に
使えなくなるため。§30）。またヘッダは正常終了時に確定するので、途中で落ちると
0 秒の wav に見える（中身は残っているので `--repair` で直せる）。

したがって実会議では、本体とは別プロセスで同じマイクから生の音声を録る
（`scripts/record_backup.py`）。本体が落ちても、接続が切れても、この予備録音を
`--wav` で本体に流し直せば、同じ手順で同定の記録（diag/turns）が作れる。

役割分担:

| 録音 | 用途 |
|---|---|
| 予備録音 `transcripts/backup_<日時>.wav` | 生の音声の正本。落ちない。後から `--wav` で再生して測る |
| 本体の `transcripts/<日時>.wav` + diag/turns | その場でライブ実行できた場合の記録。diag の ms と揃っているので、そのまま注釈と採点に使える |

## 1. 前日まで（1回だけ）

```bash
cd ~/discussion-support
uv run pytest -q                                   # グリーンが基準
uv run python -m das.asr.live --no-agent --no-open  # 一度起動して止める
```

一度起動しておくのは、声紋モデル ReDimNet の初回ダウンロード（GitHub, 20MB）を
会議室でやらないため。起動して議事録画面が出たら Ctrl-C でよい。

- `.env` に `SONIOX_API_KEY` と `PYANNOTEAI_API_KEY`。介入を出さない間は
  `OPENAI_API_KEY` は要らない
- 参加者の同意（録音・研究利用）。声の登録は求めない（登録者ゼロで始める）

## 2. 当日、開始前（5分）

ターミナルを2つ開く。どちらも `cd ~/discussion-support`。

```bash
# (1) 点検。✗ が1つでもあれば開始しない
uv run python scripts/preflight.py --no-agent
# 外付けマイクを使うなら名前の一部で指定（一覧: scripts/record_backup.py --list）
uv run python scripts/preflight.py --no-agent --device "USB"
```

点検で見るもの: APIキー、Soniox/pyannote への到達、マイクの入力レベル
（-40 dBFS より小さければ近づける）、ReDimNet のキャッシュ、ディスク残量、
`transcripts/` への書き込み、電源。

```bash
# (2) 予備録音を先に開始（ターミナル1）。Ctrl-C で止めるまで録り続ける
uv run python scripts/record_backup.py                # 既定マイク
uv run python scripts/record_backup.py --device "USB" # 本体と同じマイクを指定
```

10秒ごとにレベルが出る。30秒以上ほぼ無音なら警告が出るので、そのときは
マイクの選択とミュートを見る。

```bash
# (3) 本体を開始（ターミナル2）。介入なし・LLM なし・想定人数を実人数に
uv run python -m das.asr.live --no-agent --no-llm --diarization-max-speakers 4
```

`--diarization-max-speakers` は実際の参加人数に合わせる（7月の実会話3本は
上限1のまま録れていて事後まで気づけなかった。§20-3）。開始直後に

```bash
head -1 transcripts/$(ls -t transcripts/*.diag.jsonl | head -1 | xargs basename)
```

で構成の行を見て、`diarization: pyannote` と人数が入っていることを確認する。
ここで `PYANNOTEAI_API_KEY が未設定` の行が議事録に出ていたら止めて直す。

会議中は何もしない。議事録画面は参加者に見せなくてよい（見せると振る舞いが
変わる。今回は同定の測定が目的）。

## 3. 終了直後（1分）

1. 本体を UI の「終了」か Ctrl-C で止める → 次に予備録音を Ctrl-C で止める
   （この順。本体の wav のヘッダが確定してから）
2. セッション名を控える

```bash
SESSION=$(basename "$(ls -t transcripts/*.turns.jsonl | head -1)" .turns.jsonl)
echo $SESSION
uv run python eval/diagnose_live_session.py --session $SESSION   # 一次診断
uv run python scripts/check_audio.py transcripts/$SESSION.wav    # 音質（SNR・帯域）
```

コンソールに `録音: STTへ送れず録音に含めなかった音声 N秒` が出ていたら、
その分は本体の記録に穴がある。N が数秒なら本体の記録をそのまま使ってよい。
数十秒以上なら、§5 の予備録音からの再生に切り替える。

## 4. 採点（1本あたり 注釈20〜30分 + 採点1分）

```bash
uv run python eval/annotate.py $SESSION          # 正解付け（ブラウザ）。eval/gt_$SESSION.json に自動保存
uv run python eval/decompose_attribution.py --run $SESSION --detail   # 正解・誤同定・未確定と内訳
```

`decompose_attribution.py` が原稿の数字（91.5%/86.8%）と同じ `_pipeline` で
採点する。見る順は7月の手順書 §3 と同じ: 実質発話の正解率と誤同定（実用ライン
正解 ≥80%・誤同定 ≤10%）、ラベル爆発の有無、名指し中の正しさ。

介入時点（遡及前）の断面も見るなら `eval/decision_time.py` の `GROUPS` に
この日付の接頭辞を足す（例 `("実会議", "2026-09")`）。

## 5. 本体が落ちた／接続が切れたとき

慌てなくてよい。予備録音が正本になる。

```bash
# 本体の wav が 0 秒に見えるときは、まずヘッダを直す（中身は残っている）
uv run python scripts/record_backup.py --repair transcripts/$SESSION.wav

# 予備録音を本体に実時間で流し直す（会議と同じ時間がかかる。介入なし）
uv run python -m das.asr.live --wav transcripts/backup_<日時>.wav \
    --no-agent --no-llm --no-setup --no-open --diarization-max-speakers 4
```

再生で新しいセッション（diag/turns/wav）ができるので、以降は §3〜§4 をその
セッション名で行う。再生は STT と話者分離を同じ経路で通すので「同一手順」の
測定になる。ただし話者分離と音声認識は API 側の状態で結果がわずかに揺れるため、
ライブと再生で数字が完全一致はしない（校正9本でも同じ性質）。

再生中は本体が予備録音を同じ 16kHz mono で読むので変換は要らない。別の機器で
録った音声（スマホ等）を使う場合は `ffmpeg -i in.m4a -ar 16000 -ac 1 out.wav`。

## 6. 何本録るか

9月は 10分以上の会議を 3本以上（メンバーや人数が違う日を含む）。1本の数字で
結論を出さない。STATUS の「新しい録音」（§49 の残課題）に対応する。

3本そろったら、日常会話コーパス（校正9本・持ち越し4本）と並べた表を STATUS に
足し、原稿の第8節「実会議での同定の測定」の結果とする。

## 7. してはいけないこと（7月の手順書 §6 の再掲）

- 実会議の記録での閾値チューニング（ホールドアウト保護。§15.3）
- 予備録音を止めてから本体を止める順（本体の wav が壊れることはないが、
  予備録音の末尾が欠ける）
- 参加者への声の登録の依頼（今回は登録者ゼロの条件で測る）
