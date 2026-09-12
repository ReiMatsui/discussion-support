# GPT-Live-1 API の調査と組み込み方針（2026-09-12）

7月の `gpt-live_impact_2026-07-09.md` は「API 未提供・修論では乗り換えない」だった。
9/10 に API が公開されたので、事実関係を更新し、試験的に組み込んだ。

## 1. 公開された事実

- 公開日 2026-09-10。モデル名 `gpt-live-1`。音声層は 1分 0.05 ドル（秒単位課金）。
  背後で使うモデル（委任先）とツールの料金は別。
- 全二重: 聞きながら話す。割り込み・間・雑音を1つのモデルで扱う。
  gpt-realtime-2.1 より応答が速く、割り込みの扱いが8割改善と発表。
- 接続: WebSocket `wss://api.openai.com/v1/live/sessions`、`Authorization: Bearer`。
  他に WebRTC（ブラウザ）、SIP、既存セッションに横から付く sideband。
- セッション: `session.start` に model / instructions / audio.format（PCM 24kHz
  または 16kHz）/ audio.output.voice（既定 marin）/ delegation。
  **instructions と voice は開始後に変えられない**。
- 音声入力 `session.input_audio.append`（base64 PCM16）。
  音声出力 `session.output_audio.delta`（start_ms/end_ms 付き）。
  **出力の終端イベントは無い**（文書は「無音は送られない」とするが実測は §1.1）。
- 文字: `session.input_transcript.delta` / `session.output_transcript.delta`。
  区切りは音声の都合で、ターンの終わりを示すイベントは無い。
- テキストの注入は3種類、各 500 トークンまで:
  `session.instructions.append`（指示）、`session.thinking.append`（黙って読む文脈）、
  `session.commentary.append`（話すべき情報）。ack は注入の確認であって、
  モデルが読んだ・話したことの保証ではない。
- 委任: `delegation.type` が `responses`（OpenAI 側で Responses モデルとツールを回す）
  か `client`（`session.delegation.created` を受けてこちらが処理し、上の append で返す）。
- 制約: ライブ側のモデルは文脈窓が小さい。話者識別・多人数会話への言及は
  どの文書にも無い。対応言語の一覧も無く、「プロンプトを話させたい言語で書く」
  とだけある（日本語は実測で確かめる）。
- gpt-realtime は廃止の告知なし。併存。

出典: OpenAI の発表と開発者向けガイド（Getting started / WebSockets /
Delegation / Prompting）、Azure Foundry の GPT-Live リファレンス、gihyo.jp の記事。

## 1.1 疎通で分かった、文書と違う挙動（2026-09-12、`scripts/live_smoke.py`）

- **セッションの時間は入力音声の長さで進む**。音声を送らないと、指示や文脈の
  注入は「予定」のまま実行されず、usage も 0 秒のまま。マイクを送らない運用でも
  100ms ごとに無音を送って時計を進める必要がある（`LiveAgent._start_clock`）。
- **出力音声は無音も含む連続ストリーム**。文書には「無音は省かれる」とあるが、
  話していない間も実時間で無音の delta が届き続ける。発話の終わりは delta の
  途切れでは分からず、音量（RMS）で声の区間を切る（`_on_live_audio`）。
- 文字（output_transcript.delta）は再生の進みに合わせて遅れて届く。音声を
  まとめて先に受け取っても、文字は後から来るので、終端の確定は再生が追いつき
  文字も止まってからにする。
- 日本語は問題なく話す。指示から最初の声までは 0.5〜4.6 秒とばらつく
  （4回の実測: 484 / 613 / 682 / 4577 ms）。Realtime 版と同じ計測なので比較可能。
- マイクを30秒聞かせても指示なしには話さなかった（1回の観察）。入力の文字起こし
  は届く（「はい」）。

## 2. 本システムにとっての意味

7月の整理はそのまま成り立つ。GPT-Live は「いつ・誰に・何を」を決める層では
なく、話す層である。話者同定は提供されないので、Soniox＋pyannote＋声紋の
第1段はそのまま要る。介入の判断（Controller）も変えない。

得られるのは3つ。割り込まれたときの止まり方が自然になる（モデル側で処理）、
発話開始が速い、`_realtime.py` の cancel/truncate まわりが要らなくなる。

失うものが1つ。実験の統制。全二重モデルは「会話相手」として設計されて
いるので、室内の音声を聞かせると指示なしに話し出す可能性がある。
これは指示文で抑えられるかを実測で確かめる（§4 の確認 3）。

## 3. 組み込み（`--agent-engine live`）

`src/das/asr/live/agents/_live.py` の `LiveAgent`。`RealtimeAgent` を継承し、
公開 API（connect / feed / trigger / interrupt / apply_config / close）は同じ。
Controller・workers・UI は変更なし。

| Realtime 版 | Live 版 |
|---|---|
| `session.update` で指示と声 | `session.start` で固定（変更不可） |
| `conversation.item.create` ＋ `response.create` | 文脈を `thinking.append`、指示を `instructions.append` |
| `response.output_audio.delta` / `response.done` | `session.output_audio.delta`（無音も連続で届く）。音量で声の区間を切り、0.7秒声が途切れ再生が追いついたら終端 |
| `response.cancel` ＋ `truncate` | 再生キューを捨て、「今すぐやめて聞く」を `instructions.append` |
| 室内の音声は送らない（文字だけ） | `feed_audio` で送る。既定 `listen="idle"`＝AI が話している間は送らない |

`listen="idle"` にした理由: WebSocket 経由ではエコー除去が無く、スピーカーの
音がマイクに回り込んでモデルが自分の声を聞く。全二重の割り込み検出はその
代わりに STT 側の従来の仕組み（`interrupt()`）に任せる。エコー除去のある
機材なら `listen="always"` で全二重に任せられる。

委任は `client` にし、`delegation.created` には「追加処理は不要」と返して閉じる
（介入の内容は Controller が決めるため）。

## 4. 確認の手順（`scripts/live_smoke.py`、Mac、5分）

```
uv run python scripts/live_smoke.py            # 1〜2, 4
uv run python scripts/live_smoke.py --mic 30   # 3 も
```

1. 接続と `session.started` までの時間
2. 指示を送って日本語で話すか、指示から最初の音声までの遅延（Realtime 版の
   `_last_speak_latency_ms` と同じ計測）、文字起こしの内容
3. マイクを30秒聞かせて、指示なしに話し出さないか
4. 終了時の usage

3 で勝手に話すなら、指示文（`_PROMPT_LIVE`）を締めるか、`listen="never"`
（文字だけ渡す＝Realtime 版と同じ運用）に落とす。

## 4.1 シミュレーションでの通し（2026-09-12、3回）

`--agent-engine live --simulate 'AIツール導入の是非' --sim-scenario imbalanced`。

- 1回目: 声かけの介入は名前を呼んで成立（「参加者Cさん、ここまでの導入方針に
  ついて…」）。ただし介入中に話者分離が切れた。原因はシミュレータが介入中に
  音声を止めること（pyannote は5秒無音声で切断）と、再接続後に溜まった音声を
  一気に送って「実時間より5秒先行」で再び切られる悪循環。後者は実会議でも
  起こり得る弱点だったので `_pyannote_diarization.send_audio` に先行分を捨てて
  追従する処理を入れ、前者はシミュレータが待機中も無音を流すようにした。
- 2回目: 分離は切れなくなったが、介入の発話が4つに割れた（壁時計で終端を
  測っていたため、受信の空白を終端と誤認）。終端をストリーム時間の無音で
  測るように変更。
- 3回目: 「では、参加者Cさん、今の話題について、どうお考えですか?」が1発話で
  議事録に入り、参加者Cが応答。分離も安定。AI の声紋は3秒で登録。
  指示なしの発話なし。

## 5. 判断

- 修論の実験は当面 Realtime 版（既定）のまま。Live 版は §4・§4.1 が通ったので、
  次は実マイクで2〜3人・5分の試行（介入あり）を行い、発話開始の遅延の分布と
  割り込み時の止まり方を Realtime 版と並べてから、実験で使うかを決める。
  途中で切り替えない。
- 論文では、7月の整理（会話担当と深い処理の分離が業界の設計と同型）に加え、
  「全二重モデルが公開されても話者同定は提供されない」という事実を関連研究と
  考察に書ける。
