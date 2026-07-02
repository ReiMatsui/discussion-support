# 深掘りレビュー: 介入時の実挙動（自然さ・本質性）

対象: `src/das/asr/live/`（HEAD、drift TTL / 抑制コード化 / triage LLM分類 / meeting epoch / partialフロアゲート等の直近修正込み）
方法: 会話シナリオ T1〜T10 のコードトレース。既存レビュー（docs/review-2026-07-02/01_live_pipeline.md）の既知指摘（M5/M6/M10/M11/H5 等）は繰り返さず、修正後の残余のみ再評価する。
行番号はすべて実際に確認したもの。

---

## シナリオ別トレース

### T1. AI発話開始直後の衝突 → 中断 → リトライ

**経路**: 人間の新規確定発話（>8文字）で `agent.interrupt()`（`_workers.py:1516-1519`）→ 発話済みバッファを `_pending_intervention` に保存（`_realtime.py:524-554`、TTL 60秒・再試行2回）→ Controller が retry 候補（`_workers.py:393-402`、pause 2.4秒 `_facilitation.py:184`）→ `trigger()` が「[システム注記: …中断されました。まだ重要であれば、簡潔に再度伝えてください] あなたの中断された発言: {delivered}」を注入（`_realtime.py:448-460`）。

**確認した問題**:

1. **「言いかけた内容」が実際に聞こえた内容とずれる**。保存される `delivered` は割り込み時点の transcript バッファ `_ai_text_buf`（`_realtime.py:525`）だが、transcript は音声再生より先行してストリームされる。API 側履歴は再生済みバイトで truncate している（`_realtime.py:584-591`）のに、リトライ注記には**誰も聞いていない後半のテキスト**が「あなたの中断された発言」として渡る。モデルが「先ほど申し上げた通り…」と、実際には発していない内容を既出扱いで再開する不自然さが生じる。

2. **文脈が進んだ後のリトライに内容陳腐化ガードがない**。retry 候補は `_build_candidates` で `expires_at` 未設定（`_workers.py:395-402`、0.0=無期限）で、期限判定は `trigger()` 内の TTL 60秒だけ（`_realtime.py:453`）。60秒＝活発な会話なら10ターン以上。しかも Phase3 で Speaker から「介入不要」判断を撤去し、プロンプトは「『話すかどうか』は考えず…述べてください」（`_constants.py:326-328`）と強制発話を指示する一方、リトライ注記は「まだ重要であれば」（`_realtime.py:454-457`）と裁量を与えており**矛盾**。実挙動はほぼ確実に再発話で、2ターン進んだ後の「先ほどの続きですが」は会話の流れを逆行させる。

3. **中断内容が次のあらゆる介入に混入する**。`trigger()` は fact/manual/drift/invite どのトリガーでも、`_pending_intervention` が新鮮なら注記を必ず追加する（`_realtime.py:448-460`、トリガー種別による分岐なし）。例えば fact 補正（「短い一言で。追加論点の展開はしない」指示 `_realtime.py:424-431`）に「中断された発言の再送」が同居し、**1発話に2つの意図が混ざった長い発話**が生成されうる。手動呼び出しへの回答に前の脱線指摘の続きが混ざるのも同型。

4. 期限切れの retry が Controller に採択された場合、`trigger()` 内で破棄されるが `_pending` が非空なら汎用コンテキストだけで送信され（`_realtime.py:462-471` → 送信続行）、ログ上は "retry" なのに実発話は count 相当という不整合が残る（観測性のみの問題）。

5. **中断反応そのものが STT 確定待ちで遅い**。割り込み検出は確定 record 依存（`_workers.py:1516`）なので、人間が話し始めてから 0.5〜1.5 秒（endpoint+flush）は AI が喋り続け、そこから graceful yield 300ms（`_realtime.py:556-576`）。人間側の体感は「AI がなかなか譲らない」。既知 M6 の変種だが、衝突シナリオでは partial ゲート（受け側）でなく**攻め側（interrupt）が partial を見ていない**ことが残余リスク。partial は `show_partial` で既に手元にある（`_session_state.py:834-850`）。

### T2. 手動呼びかけ「AIさん、論点整理して」→ 応答までの全経路

**レイテンシ内訳**（実測系の定数・タイムアウトから）:

| 区間 | 該当箇所 | 目安 |
|---|---|---|
| 発話終了 → STT endpoint 確定・flush | `_recv_loop.py:277-279` | 0.5〜1.5s |
| triage tick + LLM 分類 | `_workers.py:987`（0.25s tick）, `1068`（timeout 10s、通常1〜3s） | 1.3〜3.3s |
| manual キュー投入 → worker drain | `_workers.py:1104-1110`, `1554` | 〜0.25s |
| pause 判定（沈黙 1.0s 必要） | `_facilitation.py:182` `_INTERVENTION_PAUSE_MANUAL` | ※沈黙は record 観測時点から進むため triage 遅延と並行消化 |
| `trigger()` → 最初の音声 | `_realtime.py:243-246`（`_last_speak_latency_ms` 実測） | 1〜3s |

**合計: 発話終了から音声応答まで約 3〜7 秒**。この間のフィードバックは `manual_call_status`（queued/waiting/dispatched/…）の UI テキストのみ（`_session_state.py:754-780`）。**音声のアック（「はい」等）が一切ない**。対面会議で参加者は画面を見ていないため、5秒の無反応＝「聞こえていない」と解釈して呼び直す。呼び直しは最新依頼で上書きされる（`_workers.py:222-230`）ので二重応答はないが、**1回目が dispatch 済みの直後に言い直した場合**、manual の kind cooldown 5秒（`_facilitation.py:182,289-297`）で保留され、体感さらに悪化する。

**待機失敗時**: 会話が続いて 1.0 秒の間が 30 秒取れないと TTL で破棄され（`_workers.py:667-681`）、status="expired" の小さな表示のみ。「呼んだのに何も起きなかった」ことが音声では伝わらない。

**応答が的外れな場合の修正手段**: もう一度呼ぶ以外にない。Realtime セッションは自分の発話履歴を保持しているので「さっきのは違う、〇〇を整理して」は文脈上は通じる設計だが、上記 cooldown 5 秒と再度の 3〜7 秒レイテンシを踏む。会話的な修正ループとしては重い。

**補足**: fact(priority 0) は manual(priority 1) に勝つ（`_facilitation.py:179-182`）。ユーザーが明示的に呼んだ直後でも、たまたま補正候補が残っていればそちらが先に喋る。コメント上は意図的だが、「呼んだのに別の話をされる」体験であり優先順位は再考の余地がある。

### T3. fact 補正発火の寸前に話題が変わる

fact 候補の鮮度管理は **時間ベースのみ**: `expires_at = queued_at + 30s`（`_facilitation.py:326-327`）、pause 0.9s、kind cooldown 2.0s（`_constants.py:401-405`）。deadline_ms 1500 は依然未使用（既知 H5、修正されていない）。

パイプライン遅延を積むと: triage 注釈（tick+LLM 1〜3s）→ fact checker（0.5s ゲート `_workers.py:1168` + LLM timeout 15s 通常1〜3s）→ キュー投入は**発話から 3〜6 秒後**。そこから 0.9s の間待ち。活発な会話では partial フロアゲートにより沈黙が積まれず、**訂正対象発話から 10〜25 秒後・話題転換後**に発火するのが現実的な挙動。この時:

- 候補破棄は expires 30 秒だけ。**「話題がもう移ったか」の内容判定はどこにもない**。30 秒は 2 論点ぶん進む長さ。
- `fact_note`（`_realtime.py:420-432`）は claim を明示するので「先ほどの◯◯という点ですが」と繋げる素材はあるが、**「話題が既に移っている可能性があるので、必要なら一言で触れるだけにせよ」という指示がない**。モデルは「今の話」として訂正する。
- facts deque は先頭 1 件しか候補化しない（`_workers.py:348-363`）ため、古い補正が pause を待つ間、新しい（より文脈に近い）補正が後ろで待たされる head-of-line blocking がある。
- kind cooldown 2.0 秒は非常に短く、誤り断定が続く発話列で**訂正の連発（2秒間隔）**が可能。重複判定は正規化テキスト完全一致のみ（`_workers.py:1198-1204`）。「揚げ足取り AI」の体感リスク。

### T4. 沈黙18秒 → silence 要約。だが全員が資料を読んでいた

silence 候補は `pending_count > 0` かつ閾値（standard 18s）だけで生成され（`_workers.py:424-436`）、採否も `silence_elapsed >= pause_required` の時間判定のみ（`_facilitation.py:298-303`）。**沈黙の質（思考中・資料読み・気まずい停滞）を区別する信号は一切ない**。マイク PCM はあるが VAD なし、画面・資料の文脈もなし。

さらに悪いのは、**トリガー理由がモデルに伝わらない**こと。silence dispatch は `agent.trigger(topics=_topics)` のみ（`_workers.py:1863`）で、drift/invite/fact/manual のような注記がない。モデルは「直近発話のバッチ」だけ渡されて「介入すべきと判断された場面だ、必ず何か言え」（`_constants.py:326-328`）と指示される。結果、資料読み中の 18 秒に**要約や新論点の提示**という最も重い介入が落ちる。緩和要素は「トリガーで pending が消費されるため連続発火しない」ことだけで、最初の一発は必ず刺さる。

**あるべき挙動**: 沈黙トリガーは「短い確認の問いかけ」（「続けますか、それとも一度整理しましょうか？」）に限定し、silence である旨をコンテキスト注記で渡す。VAD/資料共有がない以上、質の判別は不可能なので、**介入の重さを下げることで誤爆コストを下げる**のが現実解。

### T5. invite 直後、本人が答える前に count/drift が発火しうるか

invite dispatch 後は `_last_intervention_at` 更新（`_workers.py:1879`）により global cooldown（standard 25s）で **drift と再 invite は 25 秒抑制される**（`_facilitation.py:183,187,284-288`）。同一人物への連続声かけも `same_as_last_invited` で抑制（`_workers.py:438-450`, `_facilitation.py:304-306`）。ここは設計通り。

しかし **count と fact は global cooldown の対象外**（kind scope、count は cooldown 0.0 `_facilitation.py:185`）:

- invite トリガーで `_pending` は消費されるため count 即発は起きないが、招かれた人が考えている間に他の参加者が 10 発話すれば、**本人がまだ答えていなくても count 介入が発火**し、声かけの意味を潰す。
- 招かれた人がためらいがちに答え始めた際の思考ポーズ（>0.9s、endpoint 跨ぎ）に **fact 補正がバージインしうる**。最も発言のハードルが高い参加者の初発言に最優先介入が被さるのは、participation 支援の目的と正面衝突する。
- **「声かけした相手の応答を待つ／確認する」状態がどこにもない**。`_last_invited` は連続回避にしか使われず（`_workers.py:1880`）、invitee が実際に発言したかの追跡・フォローアップ（「〇〇さん、いかがですか」→無応答→別の形で拾う）は存在しない。

### T6. モード切替の瞬間（transcribe / facilitate / converse）

`set_session_mode`（`_workers.py:1409-1430`）のトレース:

1. **converse→facilitate をパートナー発話中に実行**: `_detach_partner`（`_workers.py:1398-1406`）が `state.partner=None` → `p.close()`。`close()`（`_base.py:165-179`）は stop フラグで再生スレッドを即終了させ**音声がぶつ切り**（graceful yield なし。`ConversationPartner.interrupt` の即停止設計は AI 間制御用だが、ここではユーザー向け挙動になる）。喋りかけの transcript は record されず消える。さらに深刻なのは**エコー保護の同時消滅**: flush のテキスト類似エコー判定は `partner is None` でスキップされ（`_recv_loop.py:86-88`）、`__PARTNER__` 声紋も close() で削除される（`_base.py:171-176`）。**切断直前に再生されたパートナー音声の残響（〜2秒）が人間の発話として議事録に載り、triage/fact/topic の入力になる**。

2. **facilitate→transcribe を AI 発話中に実行**: `apply_config(mode="off")`（`_realtime.py:184-197`）は interrupt しないため、**モードを切ったのに AI は最後まで喋り続ける**。対照的に介入トグル OFF は `agent.interrupt()` を呼ぶ（`_session_state.py:717-726`）。同じ「止めたい」操作で挙動が非一貫。

3. **agent panel から conversation モードへ切替**: triage は未処理分を一括負注釈して安全（`_workers.py:1012-1021`）、drift/fact/participation checker も停止する（`_workers.py:910,1138,1235`）。しかし **切替前から `_pending` に残っていた drift/fact/manual は破棄されない**（clear_all は disabled/off のみ `_workers.py:1468-1470,1531-1545`）。`_build_candidates` は fact/manual/drift をモード無関係に生成し（`_workers.py:348-391`）、barge-in レーンにモードチェックがない（`_workers.py:1561-1722`）ため、**conversation プロンプトの下で脱線介入や事実補正が発火しうる**（TTL 30 秒以内）。

4. エコー窓は agent インスタンスに紐づくため切替を跨いで正しく持続する（問題なし）。

### T7. パートナーとファシリテーターの同時トリガー／脱線への付き合い

- 同時発話の防御は 3 点で確認: worker ループ（`_workers.py:1546-1551`）、speech_start イベント（`_workers.py:1335-1338`）、utterance イベント（`_workers.py:1327-1334`）。トリガー経路も worker 1 本なので二重発話は構造的に起きない。マイク→パートナーの供給もファシリテーターのエコー窓中は遮断（`_workers.py:1932-1936`）。ここは堅い。
- **脱線に付き合うパートナー**: drift checker はパートナー発話を判定対象に含む設計（`_workers.py:920-928`）で検出自体は可能。しかし (a) パートナーのプロンプトは「雑談に自然に付き合え、議題に無理に戻すな」（`_constants.py:466-468`）と**脱線を能動的に延命する**指示であり、(b) drift 候補は `partner_busy` で抑制され（`_facilitation.py:270-272`）、server VAD のパートナーは人間発話後 ~500ms で応答を始めるため **1.8 秒の pause がほぼ発生しない**。drift 確認 2 回 + pause 待ちの間に TTL 30 秒（`_workers.py:271-284`）で候補が破棄されるループに入りやすい。**converse モードでは脱線が AI 燃料で自走し、ファシリテーターの介入機会が構造的に痩せる**。ファシリテーターは優先権（partner.interrupt）を持っているのに、drift 採択の物理条件がそれを使わせない。drift 採択時に「パートナーを能動的に interrupt してから話す」経路（urgency=barge_in 相当）が欠けている。
- パートナー中断は即時ぶつ切り（`_partner.py:189-220`）。AI 同士とはいえ人間には「片方の AI がもう片方を無作法に遮った」ように聞こえる。graceful yield（300ms）は RealtimeAgent 側にしかない。

### T8. 発話の長さ・頻度・言い回しの同型性

- **count トリガーは「10 発話たまった」だけで発火し（`_workers.py:404-413`）、cooldown 0.0（`_facilitation.py:185`）、しかも Phase3 で「介入不要」判断を撤去したためモデルは必ず喋る**（`_constants.py:326-328`「『話すかどうか』は考えず」）。健全に回っている議論でも約 10 発話ごとに強制介入が入る。「言う価値があるか」の判定は checker 側にある建前（`_realtime.py:44-46` docstring）だが、**count/silence の checker は無判定のカウンタ／タイマー**であり、系全体から価値判定が消えている。これが「仕切りすぎ」の構造要因。
- 発話長の上限は「30 秒以内」（`_constants.py:329`）。日本語 30 秒 ≈ 150〜200 字で「一言」とは言えない。count（数分おき）× 30 秒で会議時間を有意に占有しうる。conversation は 15 秒（`_constants.py:340`）なのにファシリテーターの方が長い設定は逆。
- **言い回しの同型化**: drift/invite/fact/manual の注記は固定テンプレート（`_realtime.py:407-444`）で、count/silence は理由なし。プロンプトに「前回と同じ切り出しを避ける」「宛先（誰への発言か）を明示する」指示がない。Realtime セッションが自分の発話履歴を持つことが唯一の変化源で、介入は「ここまでの議論を整理すると…」型に収束しやすい。
- fact の kind cooldown 2.0 秒（T3 で詳述）は頻度面でも訂正連発を許す。

### T9. triage 分類が遅延した場合（LLM 3秒）

- **カーソル整合は正しい**: fact checker は未注釈 record で必ず停止し（`_workers.py:1153-1156` の `break`）、追い越しはない。epoch チェックも副作用直前に一貫して入っている。設計通り。
- **呼びかけ応答**: 分類遅延はそのまま T2 のレイテンシに加算。さらに manual の `created_at` は**分類完了時**に打たれる（`_workers.py:1104-1107`）ため、TTL 30 秒は発話時点でなく分類時点から数える。遅延 3 秒なら応答が発話から最大 33 秒後まで許容され、鮮度の意味が弱まる。
- **リトライの head-of-line blocking**: 一時失敗は同一発話を tick 跨ぎで最大 2 回再試行し、その間**後続発話（新しい呼びかけを含む）の分類が全停止**する（`_workers.py:1069-1075` の `break`）。
- **backlog ドロップで呼びかけが無言で消える**: バックログ >8 件で古い分を `skipped=backlog` 負注釈する（`_workers.py:1023-1039`）。**ドロップ範囲に「AIさん、〜して」が含まれていても、分類されないため manual 化されず、voice_call_diag も status も残らない**。白熱した議論の最中（まさに backlog が積まれる状況）に呼んだ声だけが選択的に無視される。遅延有界化の設計意図は正しいが、少なくとも「スキップされた発話に呼びかけ様のテキストがないか」の安価なチェック（あるいはスキップ時の UI 通知）が欠けている。
- fact 鮮度: 注釈遅延ぶん、訂正が誤り発話からさらに遠ざかる（T3 に合流）。

### T10. ファシリテーター発話に ms が無い

`_on_agent_text` は `{"ms": None, "end_ms": None, ...}` で record する（`_workers.py:1293-1295`）。パートナーも同様（`_workers.py:1362-1364`）。表示は "--:--"（`_constants.py:471-475`）。

影響:
- **研究データ**: turns.jsonl 上で AI 発話の時間位置・長さが不明。介入タイミングの分析（発話ギャップとの関係、被り率、AI の発話時間シェア）が records だけでは不可能。interventions.jsonl の delivery には壁時計 `created_at` があるが、records 側は壁時計も持たないため、突合はテキスト一致頼み。
- **リプレイ**: 既知 M5 の「時刻駆動の採否込みリプレイ」を将来作る際、ファシリテーターターンを仮想クロック上に置けない。
- **レビュー**: 「介入が発話に被ったか」を turns.jsonl から事後検証できない（今回の実障害事例のような分析は wav を聞くしかない）。

修正は容易: `add_facilitator_delivery_event` 側は時刻を持っており、state は `pcm_total_bytes`（16kHz×2byte → ms 換算）で STT の ms 軸に載せられる。speech_start 時刻と再生バイト数（`_played_bytes`、24kHz）から ms/end_ms を近似できる。

### 実障害事例（11秒発話中の1秒の間に manual 発火）の残余リスク評価

partial フロアゲート（`_workers.py:81-96` + `_session_state.py:834-850`）は「STT が転写継続中」の間だけ有効。残余:

1. **Soniox endpoint（`<end>`）で flush されると `cur_text` がクリアされ、直後の `show_partial` で `partial_text` が空になる**（`_recv_loop.py:277-279,304-305` → `flush` 247-249）。endpoint は 0.5〜1 秒程度の間で発火するため、**発話の途中の「endpoint を超えるが話し終えてはいない」思考ポーズ（1.5〜2.5 秒）では保護が切れる**。manual の pause 1.0s + record 観測からの沈黙カウントで、保護総量は endpoint 閾値+1.0 秒 ≈ 約 2 秒。2.5 秒の「えーっと」には依然被る。根治は VAD（既知 M6）で、partial ゲートは緩和策と位置づけるのが正確。
2. `_PARTIAL_FLOOR_MAX_AGE` 10 秒: partial が 10 秒不変なら stale 扱いでフロア解放（`_workers.py:94`）。話し続けているのに partial が 10 秒不変という状況は稀で、保険として妥当。
3. エコー窓中の AI 自身の partial でタイマーが更新される既知トレードオフ（`_session_state.py:845-846`）は保守側なので許容で正しい。

---

## 指摘一覧

### Critical

**C-1. 「話さない」という判断が系から消えている（count/silence の無判定強制発話）**
`_workers.py:404-436`（無判定の候補生成）, `_facilitation.py:185-186`（count cooldown 0）, `_constants.py:326-328`（「話すかどうかは考えず」）, `_workers.py:1849,1863`（理由なしトリガー）。
Phase3 で Speaker の「介入不要」を撤去した際、その判断責務が count/silence の checker に移されなかった。結果、価値判定ゼロのカウンタ／タイマーが LLM に強制発話させる。「10 発話ごとに必ず何か言う AI」「資料読みの沈黙に要約を被せる AI」はファシリテーションとして本質を欠き、ユーザーの信頼を最も損なう挙動。T4/T8 参照。

### High

**H-1. リトライ設計が「自然な再開」になっていない**（T1）
`_realtime.py:448-460, 524-532`, `_workers.py:393-402`。①未再生テキストを「中断された発言」として再送する、②内容陳腐化のガードが TTL 60 秒のみ、③「まだ重要であれば」と言いつつプロンプトが沈黙を禁止しているため実質必ず再発話、④中断内容がその後のあらゆるトリガー（fact 含む）に混入し 1 発話 2 意図の長い発話を生む。

**H-2. 手動呼びかけの応答体験: 3〜7 秒の無音＋音声アックなし＋backlog 時の無言ドロップ**（T2/T9）
`_workers.py:1046-1110`（triage 経由）, `_workers.py:1023-1039`（backlog スキップに呼びかけ検査なし）, `_session_state.py:754-780`（UI status のみ）。対面会議では画面 status は見えない。呼びかけは唯一「ユーザーが明示的に AI に期待する」瞬間であり、ここでの無反応・期限切れ・選択的無視は他のどの介入不発より体験を壊す。なお UI ボタン経由は triage を通らないが、音声呼びかけは分類 LLM 1 往復ぶん構造的に遅い。

**H-3. converse→facilitate 切替でパートナーのエコー保護が同時消滅し、AI の残響が人間発話として記録される**（T6）
`_workers.py:1398-1406`, `_recv_loop.py:86-88`, `_base.py:171-176`。切替直前のパートナー音声の室内残響（〜2 秒）がテキスト類似・声紋の両フィルタを失った状態で STT に確定され、議事録・triage・fact・topic に流入する。切断を「エコー窓（2 秒）が明けてから」に遅延させるだけで塞げる。

**H-4. トリガー種別・理由がモデルに伝わらない（count/silence）**（T4/T8）
`_workers.py:1849,1863` → `_realtime.py:351-444`（count/silence には注記節がない）。モデルは「なぜ呼ばれたか」を知らずに「必ず何か言え」と言われるため、場面に合わない重い介入（要約・新論点）を選びがち。drift/invite/fact/manual は注記があるのに count/silence だけ欠けているのは実装漏れに近い。

### Medium

**M-1. fact 補正の鮮度・頻度設計**（T3）
`_facilitation.py:326-327`（TTL 30s）, `_constants.py:402`（cooldown 2.0s）, `_workers.py:348-363`（deque 先頭のみ）。話題転換の内容判定なし・連発可・head-of-line blocking。fact_note に「話題が移っている可能性があれば一言で」の指示追加と、TTL 短縮（10〜15s）または「候補生成後 N 発話進んだら破棄」の発話数ベース期限が妥当。

**M-2. invite 後の「待つ」状態がない**（T5）
`_workers.py:1866-1882`。声かけ後、invitee の応答有無を追跡せず、count/fact が被さりうる。invite 後 20〜30 秒は count を保留し fact の pause を引き上げる「awaiting_invitee」状態（Controller への 1 入力追加）が欲しい。フォローアップ（無応答時の拾い直し）も欠けている。

**M-3. converse モードで drift 介入が構造的に飢餓する＋パートナーの役割矛盾**（T7）
`_facilitation.py:270-272`, `_constants.py:466-468`。パートナーが脱線を延命し、その発話中は drift が抑制され、pause 1.8s が生まれず TTL 30s で候補が死ぬ。drift 確認済みの場合はファシリテーターがパートナーを能動 interrupt して介入する経路（既存の優先権機構の流用）を追加すべき。

**M-4. モード off 切替が進行中の発話を止めない非一貫**（T6）
`_realtime.py:184-197` vs `_session_state.py:717-726`。`apply_config` でモードが off になったら interrupt を呼ぶべき。

**M-5. conversation モード切替後も保留 fact/drift/manual が発火しうる**（T6）
`_workers.py:348-391, 1561-1722`（barge-in レーンにモードゲートなし）。モード変更時に `_pending.clear_all()` するか、`_build_candidates` でモードにより fact/drift を落とす。

**M-6. topic 抽出がファシリテーター/パートナー自身の発話を論点として拾う**
`_workers.py:853-857`（AGENT_SPEAKER を除外していない唯一の checker）。AI の整理発言が「論点」として抽出され、以後の drift 判定の基準に混入する自己強化ループ。他 checker と同じ除外フィルタを足すだけ。

**M-7. partial フロアゲートの残余: endpoint 跨ぎの思考ポーズには依然被る**（実障害の再評価）
`_recv_loop.py:277-279` + `_workers.py:81-96`。保護は「STT が発話継続とみなす間」に限られ、endpoint 閾値を超える 1.5〜2.5 秒のポーズで manual(1.0s)/fact(0.9s) は衝突可能。根治は PCM ベースの簡易 VAD（既知 M6）で、それまでの暫定として「直前 record の end_ms から実時間 N 秒以内はフロア占有扱い」も安価。

**M-8. ファシリテーター/パートナー発話の ms=None が研究データ・リプレイを損なう**（T10）
`_workers.py:1293-1295, 1362-1364`。`pcm_total_bytes` から ms 軸への載せ替えで小工数で解消できる。

**M-9. 割り込み検出が STT 確定依存で 0.5〜1.5 秒遅い**（T1）
`_workers.py:1516-1519`。partial（`_last_partial_change`）は既に手元にあるので、「AI 発話中に人間の partial が N 文字/秒以上伸びたら interrupt」の前倒しが可能。

### Low

**L-1. 期限切れ retry 採択時に汎用発話が "retry" としてログされる**（`_realtime.py:462-471`）。研究ログの介入種別が実態とずれる。
**L-2. `_AGENT_SILENCE = 5.0` が未使用**（`_constants.py:367`、grep で参照ゼロ）。silence 閾値は proactivity プロファイルに一本化されており、死んだ定数が閾値体系の誤読を招く。
**L-3. 「介入不要」文字列チェックの残骸**（`_workers.py:1327`, `_session_state.py:1126`）。Phase3 で介入不要応答は生成されない前提なのに文字列プロトコルが残存。
**L-4. 発話長上限 30 秒は「一言」と乖離**（`_constants.py:329`）。15 秒程度への短縮と「1 介入 1 メッセージ」の明示を推奨。
**L-5. fact(0) > manual(1) の優先順位**（`_facilitation.py:179-182`）。ユーザーの明示呼び出し直後に別件の訂正が先行しうる。manual を最優先にするか、manual 応答内に fact を併合する方が自然。

---

## 本質性の評価（介入種別ごと）

| 種別 | 価値 | 評価 |
|---|---|---|
| manual | ◎ 最も本質的 | ユーザー主導・誤爆ゼロ。ここに最良のレイテンシと確実性を投資すべき（現状は逆に最も待たされる経路） |
| fact | ○ 条件付き | high confidence 限定は正しい。鮮度・頻度・言い方（M-1）を締めれば価値が立つ |
| drift | ○ | 確認 2 回 + TTL は良設計。converse での飢餓（M-3）だけ塞ぐ |
| invite | ○ | 参加支援は研究目的に整合。「声かけ後に待つ・拾う」（M-2）が付いて初めて完結する |
| silence | △ | 質の判別が原理的に不可能な以上、「重い要約」でなく「軽い確認の問いかけ」に格下げすべき（C-1/T4） |
| count | ✕ 削除または再定義 | 「N 発話経過」はファシリテーションの理由にならない。残すなら「N 発話ごとに『介入する価値があるか＋何を』を安価な LLM で判定し、価値がある時だけ候補化」する periodic-review に再定義する |

**欠けている自然な振る舞い**:
1. **呼びかけへの即時アック**（相槌）: manual 検出時に固定短音声「はい」を即再生してから本応答を生成する。人間のファシリテーターは呼ばれて 5 秒黙らない。
2. **介入後のフォローアップ**: invite 後の無応答の拾い直し、drift 指摘後に会話が戻ったかの確認（現状は言いっぱなし）。
3. **宛先の明示**: プロンプトに「必要なら発言を particular な参加者に向ける」指示がなく、常に全体向けの放送になる。
4. **トリガー理由の自己認識**（H-4）と**言い回しの多様化指示**。
5. **会議フェーズの認識**: 冒頭（agenda 検出中）と終盤で介入の重みを変える概念がない。

---

## 「自然で本質的な動き」のための優先改善トップ5

1. **count/silence に価値判定を戻す（C-1, H-4）** — 工数: 中
   count は廃止または「periodic review checker」（N 発話ごとに軽量 LLM が『介入価値あり＋要旨』を返した時だけ候補化）へ再定義。silence は「短い確認の問いかけ」に限定し、count/silence にもトリガー理由注記を渡す（`trigger()` に注記節を 2 つ足すだけ）。喋りすぎの構造要因を断つ、体感効果が最大の変更。

2. **手動呼びかけの即時アック＋確実性強化（H-2）** — 工数: 小〜中
   検出時に固定の短い応答音声（事前生成 PCM「はい、少々お待ちを」）を即再生。UI ボタン経路の pause を短縮（ユーザーが「今」呼んでいる＝フロアを譲る意思表示）。triage backlog スキップ時は呼称を含む発話だけ分類対象に残す（1 つの substring チェックで可）。

3. **リトライの再設計（H-1）** — 工数: 小〜中
   ①保存する delivered を「再生済みバイト数に対応する transcript 接頭辞」に切り詰める、②retry 候補に expires_at（15〜20 秒 or 発話数 N）を設定、③pi 注記の混入をトリガー種別で制限（fact/manual では付けない）、④retry 注記を「以降の会話が進んでいる。続きをそのまま言うのではなく、今の文脈で必要な形に直すか、価値が薄ければごく短く済ませよ」に書き換える。

4. **モード切替の後始末を揃える（H-3, M-4, M-5）** — 工数: 小
   `_detach_partner` はエコー窓明けまで声紋・recent_ai_texts を残して遅延切断（または state 側に echo 参照を退避）。`apply_config(mode="off")` と conversation 切替で `interrupt()` + `_pending.clear_all()`。いずれも局所変更。

5. **フロア判定と割り込みの物理根拠強化（M-7, M-9）＋AI 発話の ms 記録（M-8）** — 工数: 中〜大
   PCM ベース簡易 VAD（エネルギー閾値で十分）を `_last_voice_at` として state に持たせ、`_effective_silence` と interrupt 前倒し（AI 発話中の人間 partial 成長で即 interrupt）の両方に使う。合わせてファシリテーター/パートナー record に `pcm_total_bytes` 由来の ms/end_ms を刻み、被り分析を事後検証可能にする。実障害（発話中の介入被り）系の残余リスクを根から消す変更。
