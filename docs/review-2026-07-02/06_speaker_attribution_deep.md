# 深掘りレビュー: 話者分離・話者帰属チェーン（シナリオ駆動）

対象: `src/das/asr/live/` の `_voice_profiles.py` / `_diarization.py` / `_pyannote_diarization.py` / `_assemblyai_diarization.py` / `_speaker_policy.py` / `_recv_loop.py` / `_session_state.py` / `agents/_base.py`（＋トレースに必要な範囲で `_workers.py` / `agents/_realtime.py` / `agents/_partner.py` / `_ui.py` / `_bootstrap.py`）
HEAD: `8a28437`（D1/D2/D3 修正込み）。既存レビュー `docs/review-2026-07-02/01_live_pipeline.md` の既知指摘（M3/M9/L4等）は原則繰り返さない。行番号はすべて現HEADで実際に確認したもの。

---

## 1. シナリオ別トレース結果

### S1. 3人会議、2人は事前登録済み、1人が未登録のまま30分話す

**closed roster（対面実験の想定モード, `tracker.auto=False`）では、未登録者は30分間ずっと「未確定」になり、その発話内容がファシリテーターから完全に見えなくなる。**

- 未登録者の発話 ≥1秒: `_voice_profiles.py:374` の照合で誰にも一致せず、`auto=False` なので蓄積もされない（`_voice_profiles.py:420` で `kind="未確定"`）。`_voice_profiles.py:457-461` の closed roster 分岐で `UNSURE_SPEAKER` に落ちる。`_session_state.py:270-272` の `constrain_human_speaker_key` でも同様。ここまでは設計どおり。
- 問題は下流: `_speaker_policy.py:21-22` の `is_intervention_signal` が未確定話者を**内容ごと**除外するため、topic/drift/fact/participation/agent feed の全ワーカー（例: `_workers.py:1495` `intervention_records(...)`）にこの人の発話が一切届かない。参加度統計も `_session_state.py:464-465` で未確定を除外。
- 結果: ファシリテーターは3人会議の1/3を聞いていない状態で「整理」し（議論の要約が欠落）、発言していない登録者に声かけし（未登録者が一番喋っていても invite 対象は登録者のみ）、議事録の1/3が「未確定」で埋まる。唯一 `is_triage_signal`（`_speaker_policy.py:26-40`）だけは未確定を通すため、AIへの呼びかけだけは反応する — 「話は聞いてないのに呼べば来る」という不自然な人格になる。
- open roster（カジュアル利用）では45文字蓄積で「人物N」登録され概ね動くが、登録前の発話の遡及リネームは commit を起こした Soniox ラベル1本だけ（`_voice_profiles.py:314` の `"#"+sp`、適用は `_recv_loop.py:161-162`）。登録前に複数ラベルが振られていた場合、他ラベルの `#label` レコードは幽霊参加者（参加者X）として残る。

→ 指摘 **C2 / H5 / M4**

### S2. AとBが同時に話す（重なり）、直後にAだけが続ける

**重なり検出が「後からflushされた側」にしか効かない非対称バグがある。**

- `overlaps_other`（`_recv_loop.py:58-62`）は `recent_segs` と比較するが、`recent_segs` への追加は classify の**後**（`_recv_loop.py:220-222`）。A・Bの重なり区間で先にflushされた発話は、相手の区間がまだ `recent_segs` に無いため `overlapped=False` でフル照合される（`_recv_loop.py:140`）。
- 単一マイクなので切り出しwav（`_recv_loop.py:112-119`）にはA+Bの混合音声が入る。混合embeddingで①即時判定が通れば誤帰属＋`own_sims` 汚染（`_voice_profiles.py:398-399`、以後の person_th がずれる）、通らなければ pool に混合サンプル＋文字数が入る（窓分割の一貫性チェック `_voice_profiles.py:251-269` が部分的に防ぐが、1.5秒未満の重なりは1サンプルのまま素通り）。
- 後からflushされた側は正しく「重なりスキップ」になり、直前のラベル対応に追従（`_voice_profiles.py:372, 451-453`）— ただし追従先が上記の誤帰属なら誤りが連鎖する。
- 直後のA単独発話 ≥1秒は正常に照合され回復する。短い（<1秒）継続の場合は short 経路（`_voice_profiles.py:427-450`）が働き概ね妥当。

→ 指摘 **H5**

### S3. AIファシリテーター発話中にAが割り込む（エコー修正後の経路）

**修正 D1/D2 には3つの穴が残る。最も深刻なのは「割り込み検出そのものが話者帰属の成功に依存している」こと。**

1. **未確定話者はAIを止められない。** AIへの自動割り込みは `_workers.py:1516-1519` のみで、判定材料 `new_texts` は `intervention_records`（`_workers.py:1495`）由来 = 未確定話者を除外済み。AI発話と人声が重なる区間は Soniox が新ラベルを振りやすく、D2 の count=False（下記2）で声紋照合もスキップされるため、closed roster では割り込み発話が高確率で「未確定」→ `_human_spoke=False` → **AIは interrupt されず話し続ける**。「割り込まれたら引く」という自然さの根幹が、帰属が最も壊れやすい音響条件（AI音声との重畳）に賭けられている。
2. **D2 の count=False が広すぎる。** `_recv_loop.py:132-142` の `_ai_active`（AI発話中＋エコー窓2秒）中は classify に `count=False` が渡り、`_voice_profiles.py:374` のゲートで**AI声紋チェック（①, :387）も人間の照合も丸ごと**スキップされる。つまり(a) 割り込んだ登録者Aの発話も声紋検証なしのラベル追従になり、Sonioxがラベルを取り違えていれば補正が効かない。(b) AI回り込みがテキスト網（sim≤0.35、転写崩れで外れうる）をすり抜け、かつ新ラベルだった場合、`"#"+sp` キーで records に残る — open mode では**「幻の参加者X」の発言レコードとして議事録に載る**（声紋蓄積は防いだが、記録汚染は元障害の症状のまま残余）。closed では未確定行き。さらに facilitate モードでは人間の返答はAI発話直後（エコー窓2秒以内）に集中するため、正当な発話の照合スキップが常態化し、新規参加者の自動登録も体感的に大きく遅れる。
3. **D1 は逆方向の誤爆を増やす。** `interrupt()` が `_ai_text_buf` を `_recent_ai_texts` に積む（`_realtime.py:524-531`）ため、割り込んだ人間が**AIの言葉を引用して反論する**（「いや、整理はいらなくて…」等、割り込み時に最も自然な発話）と、テキスト網（`_recv_loop.py:86-110`、`_voice_profiles.py:67-69` の部分包含は即1.0）に引っかかって人間の発話ごと破棄されるリスクが上がった。エコー参照を厚くするほど quote-drop が増えるトレードオフが未管理。

また、割り込み検出は final トークン＋flush 駆動（partial では interrupt しない）ため、人間が声を発してからAIが黙るまで Soniox のエンドポイント遅延（1秒前後〜）が必ず乗る。graceful yield 300ms（`_realtime.py:555-576`）と合わせ、体感では「Aが話し始めても1.5秒くらいAIが被せてくる」挙動になる。

→ 指摘 **C1 / H1 / H2 / H3 / M5**

### S4. 会議中にAが席を移動してマイクとの距離・角度が変わる

**プロファイル凍結＋person_th の生存者バイアスで、復帰経路がない。**

- person_th は「受理された一致sim」の中央値−0.12（`_voice_profiles.py:334-339`、履歴追加は :396-399）。近距離で高simが続くと閾値が締まり、席移動でsimが帯域外に落ちると①即時判定も継続性チェック（`_voice_profiles.py:408-417`、named prev は即 `prev=None`）も通らない。
- open mode: 新位置の声が45文字蓄積 → `_commit_profile`（`_voice_profiles.py:294-318`）で旧プロファイルとの dedupe（redimnet 0.50）に届かなければ**同一人物が「人物N」として二重登録**され、参加度・声かけ・議事録が分裂する。届けば「合流」するがプロファイルは凍結のままなので、以後も低sim追従が続く。
- closed roster: 永久に未確定（S8と同型）。own_sims はセッション限りで下がる方向に適応しないため、会議中に自然回復しない。

→ 指摘 **M2**

### S5. 「新しい会議」リセット直後、前の会議の参加者が続けて話す

**closed roster ではリセックス（reset）後、全員が永久に「未確定」になる Critical バグ。**

- `reset_for_new_meeting`（`_session_state.py:584-659`）→ `tracker.reset_session()`（:623-624）→ `_voice_profiles.py:206-208` で `_active_keys` から **`__..__`（AI声紋）以外の全キー＝事前登録済みの実名プロファイルも**外される。
- `tracker.auto` はリセットで変わらない（locked のまま）ので、次の発話から `constrain_human_speaker_key`（`_session_state.py:270-272`）の roster チェックが空集合（正確には `__AI__` 等のみ、下記M1）に対して行われ、**全員が未確定**。UIで一人ずつ再有効化しない限り新会議は全損する。対面実験では条件間の区切りで「新しい会議」を必ず押すはずで、確実に踏む。
- さらに `reset_for_new_meeting` の docstring（`_session_state.py:587`「声紋プロファイル・話者名・色は引き継ぐ」）は実装（:613-616 で `names={}`, `colors={}` クリア、tracker全非活性化）と**矛盾**しており、運用者の期待を裏切る。
- open mode では: sp_map/pool クリア＋`n_anon=0` により再蓄積から始まる（意図どおり）が、旧会議の「人物N」プロファイルが `self.profiles` に残留したまま `n_anon=0` に戻るため、新会議の「人物1」が旧「人物1」を黙って上書きする（`_voice_profiles.py:308-310`）。records はクリア済みなので実害は小さいが、残留プロファイルは無期限に溜まる。

→ 指摘 **C2（前段）/ L1**

### S6. Soniox が同一人物に別ラベル / 別人に同一ラベルを振る

概ね設計どおり吸収される（別ラベル→①即時判定か dedupe 合流、同一ラベル別人→≥1秒発話で「補正」）。ただし2点:

- **外部 diarization 併用時の identity 分裂。** `_recv_loop.py:212-219`: provider があり、voiceprint kind が高信頼でなく（相槌追従・低信頼追従等）、diarization 重なりも薄い場合、**確定済みの名前キーですら** `key_for_stt_fallback_speaker` で `@diar:N` へ**リキーされて名前が消える**。同一人物の発話が `田中` / `人物2` / `@diar:3` / `#7` の4名前空間に分裂し、`rekey`/`remap` はこの間を橋渡ししない（`_session_state.py:288-303` は voiceprint 側の sp_map を一切参照しない）。
- 短い発話（<1秒）はラベル追従が既定なので、ラベルが人を跨いだ瞬間の短いラリーは誤帰属する。相槌は `_recv_loop.py:231-234` で未確定化されるので致命ではない。

→ 指摘 **M3 / M4**

### S7. 事前登録の読み上げ中に別の人が口を挟む

**混入チェックが皆無のまま、汚染声紋が voices.json に永続化される。**

- `/api/enroll`（`_ui.py:245-277`）は `pcm_buf` の**末尾N秒をそのまま**切り出す（:258-260）。口を挟んだ人の声も入る。
- `enroll_from_audio`（`_voice_profiles.py:579-604`）は2秒窓の**単純平均**で、classify 側の自動登録が持つ窓間一貫性チェック（`_segment_samples`＋`ecs`）に相当する品質ゲートが無い。混入窓もそのまま平均に入る。
- さらにこのAPIは会議中いつでも叩ける。直前にAIが喋っていれば**AI音声混じりの「人間」声紋**が作れてしまい、以後 `_ai_echo`（`_voice_profiles.py:232-236`）の `ai_sim > best_human` 条件を汚染プロファイルが押し上げ、**エコー除去自体を破る**連鎖がある。
- 既知 L4（同名上書き）はこの経路で未修正のまま（`_voice_profiles.py:600-601` で無条件上書き）。S7の混入と合わせ「本人の良い声紋を、混入した悪い声紋で黙って置き換える」事故が1操作で起きる。

→ 指摘 **H4**

### S8. closed roster で登録者の風邪声・小声

S4と同型。simが `thresh`（＋person_th）を下回った瞬間から未確定になり、その人の発話は内容ごと介入シグナルから消える（S1と同じブラックホール）。継続性チェック（`_voice_profiles.py:408-417`）は named プロファイルに対して `person_th` フル値を要求する（ANON より厳しい）ため、「昨日登録した本人が今日は少し掠れている」だけで脱落する。会議中の回復手段は /api/enroll の取り直しだが、それはS7の汚染リスクを踏む。→ **C2 / M2**

### S9. AI声紋自動登録（3秒）前のマイク回り込み（残余リスク）

- 登録は再生スレッドが**再生済みPCM**を3秒分蓄積してから（`agents/_base.py:81-101, 127-132`）。この間の防御はテキスト網＋count=False のみ。テキスト網は `_ai_text_buf`（transcript delta）が届く前の冒頭数百msに参照が無く、転写が大きく崩れた回り込みも 0.35 を下回りうる。その場合、D2により**声紋蓄積は防がれる**が、レコードは `#label`（→ open では参加者X表示）として議事録に残る（S3の2と同じ残余）。
- flush が壁時計でエコー窓（`_ECHO_COOLDOWN=2.0`, `_constants.py:420`; 窓判定 `_realtime.py:603-610`）を過ぎてから届いた回り込み断片は、`_ai_active=False`（count=True）かつ agent のテキスト網ゲート（`_recv_loop.py:89-90` は `in_echo_window` 時のみ有効）も閉じているため、**両方の網を素通り**する。AI声紋未登録（開始3秒前）ならこの断片が pool に蓄積される。エコー判定の時間軸が「音声の発生時刻」ではなく「flush時の壁時計」であることが根本原因（H2）。
- 登録用音声はスピーカー経由でない**クリーン合成PCM**なので、部屋の音響（残響・スピーカー特性）が乗った実際の回り込みとの照合は AI_THRESH（`_voice_profiles.py:105`）を外しやすい — D2コミット自身が認めるとおりで、count=False はこの緩和として妥当。ただし「マイクで拾った回り込み自体から登録し直す/追加する」方が原理的に正しい。

→ 指摘 **H2**（＋現状の多層防御は妥当と評価）

### S10. パートナーモード（converse）で両AI声が存在

- 声紋二重化（`__AI__`/`__PARTNER__`）と `_ai_echo` の best_human 比較は妥当。マイク→partner の自己エコーは `_workers.py:1929-1936` のゲートで遮断。
- **問題: partner のテキスト安全網が時間無制限。** `_recv_loop.py:89-90` のエコー窓ゲートは `agent` のみで、partner は `_recent_ai_texts`（**deque 20件＝会話のほぼ全履歴**, `_partner.py:56`）に対し常時 sim>0.35 で照合される。converse モードでは人間がパートナーの主張を復唱・引用するのが議論の自然な形（「さっき君が言ったコスト面の話だけど」）で、部分包含（`_voice_profiles.py:67-69`）や trigram coverage で 1.0/高値が出て**人間の発話が黙って破棄される**。破棄は diag に echo_drop で残るが、議事録・介入文脈からは消える。
- また `_partner.py:249-257` は中断された応答のtranscriptも全量 `_recent_ai_texts` に積むため、参照集合は肥大する一方。会話が長いほど false-drop 率が単調増加する構造。

→ 指摘 **H3**

---

## 2. 指摘一覧

### Critical

#### C1. AIへの割り込み検出が「話者帰属の成功」に依存し、未確定話者はAIを止められない
- **該当**: `_workers.py:1495`（`intervention_records` で未確定除外）→ `_workers.py:1516-1519`（`_human_spoke` 判定と `agent.interrupt()`）、`_speaker_policy.py:21-22`
- **問題**: interrupt のトリガー母集団が「話者が確定した発話」に限定される。AI発話との重畳は帰属が最も壊れる条件（新ラベル発行＋D2のcount=False＋closed rosterの未確定化）で、まさに割り込みシーンで interrupt が発火しない。
- **なぜ危険か**: 「参加者が遮ったのにAIが話し続ける」のはファシリテーターとして最悪の不作法で、実験の受容性評価を直接毀損する。エコー修正(D2)が count=False で帰属を弱めたぶん、この依存は**修正後にむしろ悪化**した。
- **修正案**: interrupt 判定を帰属から切り離す。(1) `new_texts` を `intervention_records` でなく「bc以外の全確定発話（未確定含む）」から取る（interruptは“誰か”が話した事実だけで足りる。エコーは flush 前に破棄済みなのでAI自声で自爆しない）。(2) 併せて partial ベースの早期 interrupt（`show_partial` で非空 partial が `_INTERRUPT_MIN_CHARS` を超えたら interrupt）を入れると遅延も解決する（M5）。

#### C2. 「新しい会議」リセットが closed roster を全滅させる（＋docstring が実装と矛盾）
- **該当**: `_voice_profiles.py:206-208`（reset_session が実名プロファイルも非活性化）、`_session_state.py:623-624`（呼び出し）、`_session_state.py:270-272`（roster照合で全員未確定へ）、docstring矛盾 `_session_state.py:587` vs `:613-616`
- **問題**: リセット後、`tracker.auto=False` のまま roster が空になり、事前登録者全員の発話が未確定に落ちる。S1で示したとおり未確定は介入シグナルからも消えるため、新会議ではファシリテーターが完全に沈黙・盲目化する。
- **なぜ危険か**: 対面実験の運用手順（条件間で「新しい会議」）で100%再現する。「事前登録＋closed roster」という実験の前提構成が、リセット1回で静かに壊れる。
- **修正案**: `reset_session` で外すのは匿名「人物N」と sp_map/pool/own_sims のみとし、**実名のアクティブプロファイルは維持**する（`self._active_keys = {k for k in self._active_keys if not self.ANON.match(k)}` 相当）。docstring（引き継ぐ/クリアする）はどちらに寄せるか決めて実装と一致させる。

### High

#### H1. D2 の count=False がエコー窓中の「人間」の声紋検証・補正まで止める（範囲過大）
- **該当**: `_recv_loop.py:132-142`、`_voice_profiles.py:374`（count ゲートが照合全体を包む）、fallback `_voice_profiles.py:451-464`
- **問題**: `_ai_active` 中は AI声紋チェック(①)も人間の照合・補正・蓄積も全てスキップされ、ラベル追従のみになる。facilitate モードでは人間の応答はAI発話直後2秒に集中するため、(a) 取り違え補正が効かない時間帯が常態化、(b) 新ラベルの発話（人間・AI回り込みの両方）が `#label` として記録に残る（open modeでは「幻の参加者X」の発言として議事録汚染 — 元障害の症状の残余）、(c) 新規話者の登録が体感で大きく遅延。
- **修正案**: count の意味を分解する。「蓄積・自動登録をしない」（D2の意図）と「照合をしない」を分け、エコー窓中も**照合（AI声紋①＋人間②の読み取り専用マッチ）は行う**。classify に `enroll=False` 相当のフラグを追加し、①②と補正は実行、pool/own_sims への書き込みだけ抑止するのが最小変更。
- **既知指摘との関係**: 01_live_pipeline.md M3（ポリシー4箇所分散）は、D2でflush内にエコー判定点がもう1つ増えたことで**さらに悪化**している（現在: テキスト網→classify内AI声紋→`__..__`ドロップ→resolver→bc→constrain の6段直列）。

#### H2. エコー窓・`_ai_active` の判定が「flush時の壁時計」で、音声の発生時刻とずれる
- **該当**: `_recv_loop.py:89-90`（テキスト網の agent ゲート）、`_recv_loop.py:132-137`（`_ai_active`）、`_realtime.py:603-610` / `_partner.py:82-88`（in_echo_window）
- **問題**: STTのエンドポイント遅延・FLUSH_TIMEOUT により、AI発話中に発生した音声がエコー窓終了後にflushされると count=True＋テキスト網ゲート閉で両防御を素通りする（S9）。逆にAI発話前の人間発話が窓中にflushされると不当に count=False を食らう。発話には `cur_ms/cur_end`（会議絶対時刻）が既にあるのに使っていない。
- **修正案**: agent/partner に「AI音声の再生区間ログ」（speech_start〜last_speech_end を ms で記録、`stt_abs_ms` と同じ時計に写像）を持たせ、`_ai_active`／テキスト網ゲートを「発話区間 [cur_ms, cur_end] がAI再生区間＋cooldown と重なるか」で判定する。D3 が partial にやったのと同じ「直接表現への置換」をエコーにも適用する形。

#### H3. Partner のテキスト安全網が時間無制限＋参照20件で、人間の引用・復唱発話を黙って破棄する
- **該当**: `_recv_loop.py:86-91`（partner はゲートなし）、`_voice_profiles.py:64-77`（部分包含で即1.0）、`_partner.py:56, 249-257, 271-273`
- **問題**: converse モードで人間がパートナーの直近でない発言を引用しても sim>0.35 で議事録から消える。D1 で agent 側の参照も増えたため、割り込み直後にAIの言葉を引いた反論（S3で最も自然な発話）も落ちやすくなった。破棄はUI通知なし（diagのみ）。
- **なぜ危険か**: 「言ったのに議事録に無い」「AIが自分の反論を無視した」— 幻の参加者と対をなす“発話の蒸発”で、信頼を直接損なう。
- **修正案**: (1) partner にもエコー窓ゲートを適用し、参照は「直近再生分＋窓内」に限定（deque全history照合をやめる）。(2) 窓外は閾値を大きく上げる（0.35→0.7等）か包含ルールを外す。(3) echo_drop 時にUI上「エコーとして除外」と薄く表示し、誤爆を人間が発見・救済できるようにする。

#### H4. 事前登録（読み上げ）に品質ゲートがなく、混入・AI音声入り声紋が voices.json に永続化される
- **該当**: `_ui.py:245-277`（末尾N秒無検査切り出し・会議中も実行可）、`_voice_profiles.py:579-604`（単純平均・一貫性チェックなし・無条件上書き＝既知L4未修正）
- **問題**: S7のとおり。closed roster 運用は事前登録の品質が全て（照合の分母）なのに、登録経路だけ品質保証が無い。汚染プロファイルは `_ai_echo` の best_human 比較（`_voice_profiles.py:232-236`）まで壊す。
- **修正案**: enroll_from_audio に (1) 窓間一貫性チェック（`consist` を流用し、外れ窓を除外。除外率が高ければ登録拒否＋「他の声が混ざっています」）、(2) 既存 `__AI__`/`__PARTNER__` プロファイルとの類似が AI_THRESH 近傍なら拒否、(3) 同名は明示 overwrite フラグ必須（L4解消）を追加。読み上げUI側はAI発話中の録音開始をブロックする。

#### H5. 重なり検出が後発flush側にしか効かない（recent_segs 追加が classify の後）
- **該当**: `_recv_loop.py:58-62, 140, 220-222`
- **問題**: S2のとおり。先にflushされた混合音声がフル照合され、誤帰属・own_sims/pool 汚染の起点になる。重なりはまさに「同時に話す」場面で系統的に発生するため、ランダムノイズでなくバイアスとして効く。
- **修正案**: (1) flushの冒頭で「現在バッファ中の別ラベル partial／未flushの cur_* 区間」も重なり判定に含める。(2) もしくは classify を遅延評価にし、後続 flush で重なりが判明したら直近レコードの overlapped を遡及訂正（own_sims から当該simを取り消す）。(1)が安価で効果的。

### Medium

#### M1. `__AI__`/`__PARTNER__` が roster・プロファイルUIに漏れ、UIから deactivate するとエコー防御が死ぬ
- **該当**: `_voice_profiles.py:571-573`（`active_profile_names` は ANON しか除外しない）、`:575-577`（`all_profile_names` 同様）、`:565-569`（`deactivate` は ANON 以外何でも外せる）、露出先 `_session_state.py:490-491`（api_snapshot の vp.roster）・`_session_state.py:934-947`（レガシーHTMLのプロファイル一覧）、`/activate` ハンドラ `_ui.py:220-244`
- **問題**: AI声紋登録後、UIの roster/プロファイル欄に `__AI__` が表示される。不審に思ったユーザーがトグルをOFFにすると `_active_keys` から外れ、`_ai_echo`（`_voice_profiles.py:228-229` は `_active_keys` を見る）が無効化されて回り込みが再発する。
- **修正案**: `active_profile_names`/`all_profile_names`/`deactivate` の3箇所で `k.startswith("__")` を除外する（`_active_human` は既にやっている。除外述語をヘルパー1つに統一するのが望ましい）。

#### M2. person_th の生存者バイアスで、席移動・声質変化からの復帰経路がない
- **該当**: `_voice_profiles.py:334-339`（中央値−0.12）、`:396-399`（受理simのみ履歴化）、`:408-417`（named prev は person_th 未満で即切断）、open時の二重登録 `:294-318`
- **問題**: S4/S8のとおり。閾値が「良い日の声」に固着し、下方適応しない。closed roster では未確定ブラックホール（C2の症状）へ直結。
- **修正案**: (1) own_sims に「照合を試みたが不受理だったsim」も別系列で記録し、不受理simの中央値が受理帯に近づいてきたら person_th を base 側へ緩める（両側適応）。(2) closed roster 限定で「roster 内の最良候補が margin 付きで首位なら、simが base をやや下回っても『低信頼として本人扱い』（is_reliable からは除外）」の階段を設け、内容だけは失わないようにする（`_speaker_policy` の二層設計と整合する）。

#### M3. 外部 diarization 併用時、確定名の発話が `@diar:N` に劣化リキーされ identity が4名前空間に分裂する
- **該当**: `_recv_loop.py:212-219`（stt_fallback リキー。`resolved.speaker` は名前キーそのものなのに `@diar:` へ変換）、`_session_state.py:288-303`（sp_map/profiles を一切参照しない）
- **問題**: S6のとおり。相槌追従・低信頼追従など「kindは弱いがキー自体は確定名」の発話まで匿名化され、参加度・rekey・invite が分裂した人格を数える。
- **修正案**: stt_fallback リキーは `sp_id` が `#` 始まりの時だけに限定する（確定名・人物Nはそのまま残し、`speaker_source="stt_fallback"` の注記だけ付ける）。`key_for_diarization_speaker` も、声紋確定名との時間重なり多数決で `@diar:N → 人物/名前` の橋渡しマップを学習すると分裂が収束する。

#### M4. 遡及リネームが commit を起こした1ラベルに限られ、登録前の他ラベル発話が幽霊参加者として残る
- **該当**: `_voice_profiles.py:314`（`("#"+sp, target)` のみ）、`_recv_loop.py:161-162`
- **問題**: S1/S6のとおり。pool は「ラベルで仕切らない」設計（複数ラベルのサンプルを束ねて1人物にする）なのに、リネームは最後のラベルしか救わない。
- **修正案**: `_enroll_accumulate` で pool サンプルに由来ラベルを持たせ、commit 時に採用サンプルの由来ラベル全てについて `sp_map[label]` が未確定/#なら target へ張り替え、rename リストとして返して `rekey` を複数回適用する。

#### M5. 割り込み・沈黙の反応が final トークン駆動で、barge-in 遅延が STT エンドポイントに律速される
- **該当**: `_workers.py:1494-1519`（新レコード観測時のみ interrupt）。D3（`_effective_silence`, `_workers.py:81-96`）は「被せない」方向のみ対処済みで、「引く」方向は未対処。
- **問題**: S3のとおり、人間が話し始めてからAIが黙るまで1秒超。対面ではっきり体感される不自然さ。
- **修正案**: `show_partial`（`_session_state.py:834-857`）で非空 partial が一定文字数を超えて更新され続けたら `agent.interrupt()` を発火する早期割り込みレーンを足す（partial はエコー除去前なので、AI自声 partial での自爆を防ぐため `_ai_active` 中は partial-interrupt を無効にする、または partial テキストにもテキスト網を先に当てる）。

### Low

#### L1. reset_session の残骸: 旧「人物N」プロファイルの残留と n_anon=0 による黙った上書き、same_sims/diff_sims 未クリア
- **該当**: `_voice_profiles.py:191-208`（pool/sp_map等はクリアするが `profiles` の人物Nと same_sims/diff_sims は残る）、`:308-310`（新会議の人物1が旧人物1を上書き）
- **問題/修正**: 実害は小さい（旧人物Nは非活性）が、プロファイル辞書が会議を跨いで無限に太り、診断表示（`stats` の部屋分布, `n_all`）が会議を跨いだ混合値になる。reset_session で `ANON` プロファイルの削除と分布リストのクリアを行う。

#### L2. 既知 L4（enroll の同名重複ポリシー矛盾）は未修正のまま、S7の混入リスクと重なって危険度が上がっている
- **該当**: `_voice_profiles.py:483-499`（/rename は duplicate 拒否）vs `:600-601`（/api/enroll は無条件上書き）
- H4 の修正案に含めて解消するのが良い。

---

## 3. 「自然で本質的な動き」のための優先改善トップ3

1. **未確定話者を“いない人”ではなく“名前の分からない人”として扱う（C1・C2・S1系の根治）。**
   `is_intervention_signal` の未確定除外を「個人帰属が要る用途（invite/名指し）」に限定し、内容系（feed/topic/drift/fact/interrupt）には汎用名（`intervention_speaker_name` の既存機構）で通す。割り込み検出は帰属と完全に独立させる。closed roster は「未登録者を登録しない」ためのモードであって「未登録者の声を聞かない」モードになってはいけない — ここが今のチェーンで最も“不自然さ”を生んでいる。合わせて reset_session の実名プロファイル維持（C2）を直せば、対面実験の運用が成立する。

2. **エコー判定の時間軸を壁時計から発話区間（ms）に移す（H2、D2/D3路線の完成）。**
   AI再生区間ログとの重なりで `_ai_active`／テキスト網を判定すれば、「窓を過ぎてflushされた回り込みの素通り」と「窓中にflushされた正当発話の巻き添え（H1のcount=False過剰適用）」が同時に消える。D3 が partial で採った「タイマー間接表現→直接表現」への置換と同じ思想で、修正3部作の自然な次の一手。

3. **事前登録の品質ゲート（H4）— closed roster 戦略の土台固め。**
   窓間一貫性チェック・AI声紋との衝突検査・明示上書き・AI発話中の録音ブロック。closed roster は「閾値ホイッスル問題（既知M9）から実験を切り離す」正しい賭けだが、その成立条件は登録声紋の品質だけなので、ここに classify 側と同等の防御を移植する投資対効果が最も高い。
