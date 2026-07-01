# 作業計画: ファシリテーター手動呼び出し / 呼びかけ

ブランチ案: `codex/manual-facilitator-call`

## 目的

現在のファシリテーターは、自動判定（事実補正・脱線・沈黙・発言量偏りなど）でのみ介入する。
会議中には「今まとめてほしい」「論点を整理してほしい」「次に何を決めるべきか聞きたい」
という、参加者側から明示的にファシリテーターを呼びたい場面がある。

この作業では、参加者が自然にファシリテーターを呼び出せる経路を追加する。

目標は次の3点。

- 参加者が必要な時だけAIを呼べる。
- 自動介入の自然さ・抑制ロジックを壊さない。
- 呼び出し時の応答は短く、会議を乗っ取らず、直近文脈に沿った支援に限定する。

## 背景

現状の主な介入経路:

- fact: 高確信の事実誤りを短く補正する。
- drift: 議題から明確に逸れた時に戻す。
- retry: 割り込まれた介入を必要なら再送する。
- count / silence: 一定発話数・沈黙後に整理を促す。
- invite: 発言が少ない参加者に自然に話を振る。
- conversation mode: AIと会話するモードでは通常返答する。

ただし `facilitate` モードでは、参加者が明示的に「ファシリテーター、整理して」と言っても、
それを専用の呼び出しとして扱う経路がない。自動判定に乗らなければ反応しない。

## 方針

段階的に実装する。

### Phase 1: UIからの手動呼び出し

まずは低リスクな手動ボタンを実装する。

- UIの「介入」パネルに「呼ぶ」ボタンと、任意の依頼テキスト入力を追加する。
- 例:
  - ボタン: `ファシリテーターを呼ぶ`
  - 入力 placeholder: `例: ここまでを整理して / 次に決めることを確認して`
- 入力が空なら「直近の議論を短く整理して、次に進める一言」を依頼する。
- 入力がある場合は、その依頼をファシリテーターに渡す。
- APIは新規 `POST /api/facilitator/call` を追加する。
- 既存の `/api/intervention` は設定変更用のままにし、呼び出しとは分ける。

理由:

- 手動呼び出しは誤爆しない。
- 音声ウェイクワードより検証しやすい。
- 既存の `RealtimeAgent.trigger()` / delivery log / Partner interrupt 経路を再利用できる。

### Phase 2: 音声での呼びかけ

Phase 1 が安定した後に、STT transcript から呼びかけを検出する。

対象例:

- `ファシリテーター、ここまで整理して`
- `AI、今の論点をまとめて`
- `進行役さん、次に何を決めればいい？`

ただし、音声呼びかけは誤爆リスクがあるため、最初は保守的にする。

- 「ファシリテーター」「進行役」「AI」などの明示呼称がある時だけ候補化する。
- 直後に依頼動詞がある時だけ発火する。
  - `まとめて`
  - `整理して`
  - `確認して`
  - `次どうする`
  - `意見を聞いて`
  - `話を振って`
- 「AIについて話そう」「ファシリテーター機能が...」のようなメタ話題は発火しない。
- `conversation` モードでは通常会話に任せ、専用呼び出し検出は原則不要。

## 非目標

この作業でやらないこと:

- 自動介入の閾値調整。
- fact / drift / participation checker の作り直し。
- AECや音響処理の追加。
- RealtimeAgent の大規模リファクタ。
- UI全体の再デザイン。
- ファシリテーターに長い議事録要約を喋らせること。

## 望ましいユーザー体験

### 手動呼び出し

1. 会議中に右側の「介入」パネルを見る。
2. 必要なら依頼欄に短く入力する。
3. `呼ぶ` を押す。
4. ファシリテーターが1〜2文で短く話す。
5. 介入理由ログに `manual_call` として残る。

入力なしの場合の発話イメージ:

> ここまでの話では、候補を広げる方向と、まず条件を絞る方向が出ています。次は「今日決める範囲」を一つ確認すると進めやすそうです。

入力ありの場合の発話イメージ:

依頼: `Aさんにも意見を聞いて`

> Aさん、この条件で進める場合に気になる点はありますか。

### 音声呼びかけ

参加者:

> ファシリテーター、ここまで整理して。

ファシリテーター:

> 今は、費用を優先するか、使いやすさを優先するかが主な分かれ目です。次にどちらを先に決めるか確認しましょう。

## 設計

### 1. 新しい介入種別

`_facilitation.py` に `manual` kind を追加する。

優先度案:

- `fact` よりは低い。
- `drift` / `retry` より高いか同等。
- `invite` より高い。

推奨:

```text
fact: 0
manual: 1
drift: 2
retry: 3
count: 4
silence: 5
invite: 6
conversation: 7
```

理由:

- ユーザーが明示的に呼んだものは基本的に尊重する。
- ただし、直前に明確な事実誤り補正が必要な場合は fact を優先する。

`manual` のタイミング案:

- pause: `0.8`〜`1.2` 秒。
- cooldown: 同種で `5` 秒程度。
- global cooldown の影響は受けない、または緩くする。
- deadline: `3000` ms 程度。
- urgency: `wait_for_pause`。

注意:

- 手動呼び出しでも、人が話している最中に強引に割り込まない。
- ただし通常の drift/invite よりは早く反応してよい。
- echo window / partner busy は既存と同じく尊重する。

### 2. 状態とキュー

`SessionState` に手動呼び出し用キューを追加する。

例:

```python
self.manual_call_requests: queue.Queue[dict] = queue.Queue()
```

payload案:

```python
{
    "request": "ここまで整理して",
    "source": "ui" | "voice",
    "created_at": time.monotonic(),
}
```

リセット時には既存の `drift_requests` / `invite_requests` / `factcheck_requests` と同様にクリアする。

### 3. API

`_ui.py` に `POST /api/facilitator/call` を追加する。

入力:

```json
{
  "request": "ここまで整理して"
}
```

`request` は任意。空文字可。

バリデーション:

- `intervention_enabled == false` の場合は 400 または `{ok:false}`。
- `agent is None` または `agent.mode == "off"` の場合は `{ok:false}`。
- `request` は最大100文字程度に切り詰める。
- 改行は空白に正規化する。

成功時:

```json
{
  "ok": true,
  "queued": true
}
```

副作用:

- `state.manual_call_requests.put(...)`
- `state.add_intervention_event("manual_call", detail, metadata={...})` は実際に採択された時に行うのが望ましい。
  - ただしUI即時フィードバック用に `manual_call_queued` を残してもよい。
  - trigger log と delivery log の対応を壊さないよう注意する。

### 4. Worker統合

`_workers.py` の `_run_agent_worker` で、他の pending と同じように manual call を drain する。

実装方針:

- `_PendingInterventions` に `manual_call: dict | None` を追加する。
- `_build_candidates(...)` で manual candidate を作る。
- manual candidate の payload:

```python
{
    "request": "...",
    "source": "ui" | "voice",
}
```

- Controller が manual を選んだ時だけ queue から消費する。
- cooldown / pause 不足で hold された場合は、短時間保持する。
- TTL は 30秒程度。古すぎる呼び出しは破棄する。

重要:

- 既存の fact/drift/invite と同じ採否・review log の枠に乗せる。
- manual だけ別スレッドから直接 `agent.trigger()` しない。
- 二重発火やログ不整合を避けるため、発話経路は `_run_agent_worker` に集約する。

### 5. RealtimeAgent

`RealtimeAgent.trigger()` に `manual_request: dict | None` または `manual_instruction: str | None` を追加する。

コンテキスト例:

```text
[手動呼び出し]
参加者がファシリテーターに明示的に助けを求めています。
依頼: ここまで整理して
直近の発話を踏まえ、1〜2文で短く支援してください。
会議を乗っ取らず、必要な確認・整理・声かけだけを行ってください。
```

入力が空の場合:

```text
依頼: 直近の議論を短く整理し、次に進める一言を述べる
```

`_retry_fallback_text()` でも manual の意図を保存できるようにする。

### 6. UI

`_webapp.py` の介入パネルに追加する。

最小UI:

- テキスト入力
- `呼ぶ` ボタン
- 送信中/受付済み表示

例:

```html
<div class="manual-call-row">
  <input id="manual-call-text" placeholder="例: ここまで整理して">
  <button id="manual-call-btn" class="btn">呼ぶ</button>
</div>
<div id="manual-call-status" class="intervention-summary"></div>
```

UX注意:

- ボタンは `intervention_enabled=false` の時は disabled。
- `mode=transcribe` または agent 無効時も disabled にする。
- Enter で送信できるとよい。
- 成功時は `ファシリテーターに依頼しました` と短く表示。
- 失敗時は理由を表示。

### 7. 音声呼びかけ検出

Phase 2 で実装する。

候補関数案:

```python
def _detect_facilitator_call(text: str) -> str | None:
    ...
```

置き場所:

- `_workers.py` に置いてもよいが、テストしやすさを優先するなら小さな専用関数として切る。

検出方針:

- 明示呼称が必要:
  - `ファシリテーター`
  - `進行役`
  - `AI`
  - `AIさん`
- 依頼表現が必要:
  - `まとめ`
  - `整理`
  - `確認`
  - `次`
  - `振って`
  - `聞いて`
  - `助けて`
- 疑問・メタ話題だけでは発火しない。

通す例:

- `ファシリテーター、ここまで整理して`
- `進行役さん、次に決めることを確認して`
- `AI、Aさんにも意見を聞いて`

落とす例:

- `AIについて話しましょう`
- `ファシリテーター機能って便利ですね`
- `進行役は誰がやりますか`
- `整理すると、AIの話ですね`（AIへの呼びかけではない）

音声呼びかけを検出した場合:

- `manual_call_requests` に `source="voice"` で積む。
- 元の参加者発話自体は通常どおり transcript に残す。
- 自動介入と同じ Controller で採否する。

## ログ

`.interventions.jsonl`:

- trigger type:
  - `reason: "manual_call"`
  - `detail: request or "直近の議論整理"`
  - metadata:

```json
{
  "source": "ui",
  "request": "ここまで整理して"
}
```

`.intervention_review.jsonl`:

- candidate kind: `manual`
- payload に `request` / `source`
- dispatched true/false を既存通り記録。

## テスト計画

最低限追加するテスト:

### Unit: Controller

`tests/unit/live/test_facilitation_controller.py`

- manual は invite/count/silence より優先される。
- fact と manual が同時なら fact が優先される。
- pause 不足なら hold。
- 同種 cooldown 中なら hold。
- expired manual は採択されない。

### Unit: Worker

`tests/unit/live/test_agent_worker.py`

- manual queue に依頼があると agent.trigger(manual...) が呼ばれる。
- hold 時はすぐ捨てない。
- TTL 超過時は捨てる。
- intervention disabled / agent off では発火しない。
- drift/invite と同時でも Controller 優先順位に従う。

### Unit: RealtimeAgent

`tests/unit/live/test_realtime_agent.py`

- `trigger(manual_request=...)` が `[手動呼び出し]` コンテキストを送る。
- 空依頼でもデフォルト依頼が入る。
- manual は retry fallback に意図を残す。
- fact/manual/drift/invite のコンテキストが混ざる時に破綻しない。

### Unit: UI API

`tests/unit/live/test_ui_api.py`

- `POST /api/facilitator/call` が queue に積む。
- request を trim / length limit する。
- intervention disabled では拒否する。
- agent 無効時は拒否する。

### UI smoke

可能なら軽いDOMテスト、難しければ手動確認でよい。

- 介入パネルに入力欄とボタンが出る。
- オフ時に disabled になる。
- 成功/失敗メッセージが出る。

### Phase 2: 音声検出

検出関数の純粋テスト:

- 通す例 / 落とす例を明示。
- メタ話題を誤爆させない。

## 検証コマンド

```bash
uv run pytest tests/unit/live/test_facilitation_controller.py -q
uv run pytest tests/unit/live/test_agent_worker.py -q
uv run pytest tests/unit/live/test_realtime_agent.py -q
uv run pytest tests/unit/live/test_ui_api.py -q
uv run ruff check src/das/asr/live tests/unit/live
```

可能なら最後に:

```bash
uv run pytest tests/unit/live -q
```

## 実装順序

1. `manual` kind と Controller policy を追加する。
2. `SessionState` に `manual_call_requests` を追加し、reset 時にクリアする。
3. `_workers.py` に pending/candidate/drain/consume を追加する。
4. `RealtimeAgent.trigger()` に manual context を追加する。
5. `POST /api/facilitator/call` を追加する。
6. `_webapp.py` に手動呼び出しUIを追加する。
7. テストを追加して通す。
8. 実機で、ボタン押下から発話開始までの latency とログを確認する。
9. Phase 1 をマージ後、Phase 2 の音声呼びかけ検出へ進む。

## 完了条件

Phase 1:

- UIからファシリテーターを呼べる。
- 依頼文あり/なしの両方で自然な短い介入になる。
- 自動介入の既存テストが壊れない。
- intervention log / review log に manual call として追跡できる。
- 直接 `agent.trigger()` せず、既存の worker + Controller 経路に乗っている。

Phase 2:

- 明示的な音声呼びかけでファシリテーターを呼べる。
- メタ話題や偶然の「AI」発話では誤爆しない。
- UI手動呼び出しと同じ manual call 経路に乗っている。

## 実装担当AIへのプロンプト

このリポジトリで「ファシリテーター手動呼び出し / 呼びかけ」機能を実装してください。

まず `docs/design/manual_facilitator_call_plan.md` を最後まで読み、その指示に従ってください。

重要な方針:

- まず Phase 1 の UI手動呼び出しだけを実装してください。
- 音声呼びかけ検出（Phase 2）は、Phase 1 が小さく安全に完了してから別コミットまたは別ブランチで進めてください。
- 手動呼び出しは、別スレッドやAPIハンドラから直接 `agent.trigger()` しないでください。
- 既存の `_run_agent_worker` + `FacilitationController` + `RealtimeAgent.trigger()` 経路に統合してください。
- 自動介入の fact / drift / invite / silence の挙動は不要に変えないでください。
- 介入ログと review ログで manual call を追跡できるようにしてください。

主に見るファイル:

- `src/das/asr/live/_facilitation.py`
- `src/das/asr/live/_workers.py`
- `src/das/asr/live/_session_state.py`
- `src/das/asr/live/_ui.py`
- `src/das/asr/live/_webapp.py`
- `src/das/asr/live/agents/_realtime.py`
- `tests/unit/live/test_facilitation_controller.py`
- `tests/unit/live/test_agent_worker.py`
- `tests/unit/live/test_realtime_agent.py`
- `tests/unit/live/test_ui_api.py`

推奨ブランチ:

```bash
git switch -c codex/manual-facilitator-call
```

完了時に最低限実行するコマンド:

```bash
uv run pytest tests/unit/live/test_facilitation_controller.py -q
uv run pytest tests/unit/live/test_agent_worker.py -q
uv run pytest tests/unit/live/test_realtime_agent.py -q
uv run pytest tests/unit/live/test_ui_api.py -q
uv run ruff check src/das/asr/live tests/unit/live
```

可能なら最後に:

```bash
uv run pytest tests/unit/live -q
```
