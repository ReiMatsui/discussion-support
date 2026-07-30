# コードレビュー: src/das/asr/live/（ライブ介入パイプライン）

対象: `src/das/asr/live/` 全体（_workers.py 2153行 / _session_state.py 1126行 / _bootstrap.py 770行 / replay.py 833行 / agents/ / stt/ ほか）
参照: README.md, docs/research/RESEARCH.md, docs/design/ 配下設計文書, git log, tests/unit/live/

---

## 全体所見

**アーキテクチャの現在地。** live パッケージは「STT受信（RecvLoop）→ 話者確定（VoiceProfiles + SpeakerResolver）→ 共有状態（SessionState）→ 監視ワーカー群（topic/drift/fact/participation checker）→ 候補キュー（_PendingInterventions）→ 採否裁定（FacilitationController）→ 発話（RealtimeAgent）」という多段パイプラインで、責務の切り出し自体は段階的リファクタ（Phase0〜4）でかなり進んでいる。特に `_facilitation.py` の FacilitationController（決定的・LLMなし・候補集合→採否のみ）、`_speaker_policy.py` / `_participation.py` の純粋関数群、`_diarization.py` の SpeakerResolver は責務が明確で品質が高い。介入の trigger/delivery/review を JSONL に落とす観測性の作り込みも研究プロトタイプとして優れており、「なぜ話したか／黙ったか」を事後検証できる設計思想は正しい。

**しかし最大の問題は、この live パイプラインが研究の核（統合議論グラフ）と完全に断絶していることである。** RESEARCH.md は貢献③として「ファシリテータは AF 状態だけで介入を判断する（`decide_intervention(history, store)`）」を掲げるが、`src/das/asr/live/` は `das.graph` / `das.agents.facilitation` を一切 import しない（grep で確認）。live の介入は drift/fact/invite/manual という**AFと無関係のルール＋軽量LLM判定**であり、関係ラベル付き情報提示（貢献②）も存在しない。AFベースの介入は `src/das/cli/_listen.py` の listen-soniox 経路で `ON_UTTERANCE` フック越しに別系統としてぶら下がっているだけで、そこでは音声介入も Controller 裁定も使われない。つまり「AF で判断する脳」と「音声でタイミングよく喋る体」が別々のプログラムとして育っており、対面実験（Phase 2）で研究の主張を検証するには、どこかで必ず統合し直す必要がある。幸い `InterventionCandidate` 抽象は AF 由来の候補（例: 「未回答の攻撃がある」「支持偏重」）をそのまま載せられる形をしているため、統合コストは設計次第で小さくできる（後述の再設計提案）。

**並行処理は asyncio ではなく「デーモンスレッド＋ポーリング」の集合体である。** live 側に asyncio は存在せず、topic(3s)/drift(1s)/fact(0.25s)/participation(1s)/agent_worker(0.25s)/sender/mic/受信/再生/SSE と 10 本前後のスレッドが SessionState を共有する。ロック規約（state_lock/topics_lock/buf_lock、RealtimeAgent._state_lock の「保持したまま ws.send しない」鉄則）は明文化されておりレースの多くは潰されているが、カーソル類（agent_cursor 等）や `rev` は無ロックで複数スレッドから触られており、「新しい会議」リセットとの競合で新会議の発話がエージェントに届かなくなる実レースが残っている。また `save()` が state_lock を保持したまま議事録全文を毎発話ディスクに書き直す構造は、確定発話→介入判断のクリティカルパスを I/O でブロックする。

**「fix の積み重ね」は3箇所に集中しており、いずれも根本原因が明確である。** (1) fact prefilter の正規表現群（`_FACT_CREATIVE_EXPRESSION_RE` に「二丁拳銃」「ビビ弾」等、特定のテスト会話由来と思しき語彙が直書き）、(2) 音声呼びかけ検出（呼称regex＋依頼表現ホワイトリスト＋話題化ブラックリストの三段パッチ。しかもデモ議題が「AIツール導入の是非」で呼称が "AI"）、(3) VoiceProfiles の閾値群（モデル別3値＋人物別閾値＋短発話厳格路＋連続性閾値＋日付入りチューニングコメント）。これらは「連続的・確率的な判定問題をキーワード／固定閾値で解こうとしている」共通根because を持つ。詳細は「対症療法パターンの分析」で述べる。

**介入タイミングの設計はおおむね目的に適う。** 「barge-in レーン（fact/manual/drift/retry はエコー・パートナーガードを越えて短い間に差し込む）」と「通常レーン（count/silence/invite はフロア返却後）」の2レーン構造、種別ごとの pause/cooldown、trigger→発話開始レイテンシの計測は、対面会議で人間を遮らないための設計として妥当。ただし (a) hold の意味論が「少し待つ」と「全介入レーンを止める」を混同しており、確認待ち drift 1件で invite/silence が無期限に飢餓する Critical バグがある、(b) 設計文書 §8.5/§3.5 の「古い判断の破棄」（epoch/deadline_ms）はコード上ほぼ機能しておらず、安全装置が見かけ倒しになっている、(c) 「沈黙」を音声ではなく STT 確定テキストの到着時刻から測っているため、pause 閾値（0.9s 等）が STT レイテンシと混ざって実際の会話の間とずれる、という問題がある。

---

## 指摘一覧

### Critical

#### C1. 確認待ち drift 候補が全介入レーンを無期限に飢餓させる（hold の意味論バグ）
- **該当箇所**: `src/das/asr/live/_workers.py:1710-1711`（`if decision.reason == "hold": continue`）、`_workers.py:952-954`（候補ありだが不採択→hold 返却）、`src/das/asr/live/_facilitation.py:251-256`（drift の確認回数ゲート）、`_workers.py:298-390`（`_PendingInterventions` に drift の TTL がない。`drop_stale_facts`/`drop_stale_manual` はあるが drift 版が無い）
- **問題**: barge-in レーンで候補が1つでも存在し、かつ全候補が抑制された場合、Controller は candidate_id=None を返し worker は `_BargeInDecision("hold")` → `continue` でループ先頭に戻る。このとき**通常レーン（count/silence/invite/conversation）の評価そのものがスキップされる**。drift 候補は `drift_confirmations`（standard/controlled の既定=2）に達するまで抑制され続けるが、`pending.drift_reason` には有効期限がない。drift_count は 20 秒空くと 0 にリセットされる（`_workers.py:316-318`）ため、「脱線が1回だけ検出され、その後会話が自然に本題へ戻った」という**ごく普通のシナリオ**で、drift 候補が永久に pending に残り、以後の沈黙要約・声かけ・発話数トリガーが会議終了まで一切発火しなくなる。drift がクリアされるのは採択時・cooldown 抑制時（`_workers.py:946-951`）・介入オフ時のみ。
- **なぜ問題か**: 「発言の少ない人に声をかける」「停滞時に整理する」という本システムの主要介入が、ユーザーから見て理由なく沈黙する。レビューログ（dispatched=False の hold 記録）には残るが、実運用では「介入が来ない」故障として現れ、再現も難しい。legacy 経路（`_workers.py:485-494`）も同じ挙動なので Controller 導入以前からの潜在バグ。
- **改善案**: ①drift 候補にも `expires_at` を付与する（`InterventionCandidate` は既にフィールドを持つ。`_build_candidates` の drift 生成 `_workers.py:620-629` で `created_at + 60s` 程度を設定し、Controller の期限切れ判定に乗せる。期限切れ時に `pending.clear_drift()`）。②hold の意味論を「barge-in レーンに今すぐ話すものが無い」に限定し、`continue` せず通常レーンの評価へフォールスルーする（barge-in 候補の優先は「両レーンで採択が出た場合に barge-in を先に使う」で表現できる）。本質的には2レーンを分けず、全候補を1回の `arbitrate()` に渡して優先度で解決すべき（Controller は既にそれができる設計になっている）。

### High

#### H1. live 介入パイプラインが研究の核（統合 AF）から完全に切り離されている
- **該当箇所**: `src/das/asr/live/` 全体（`das.graph`/`das.agents` への import ゼロ。grep 確認）。AF 側の介入は `src/das/cli/_listen.py:218-262`（`FacilitationAgent.decide_intervention` を 3 秒周期で呼び `post_system` でテキスト行を流すだけ）
- **問題**: RESEARCH.md の貢献②（関係ラベル付き提示）③（AF 状態のみで介入判断）を担う `decide_intervention(history, store)` は、音声・話者特定・介入タイミング制御を持つ本パッケージと接続されていない。live の drift/fact/invite は AF を見ない汎用会議ファシリテーションであり、listen-soniox 経路の AF 介入は Controller の採否・pause 制御・barge-in・観測ログを一切通らない。
- **なぜ問題か**: 対面実験（研究計画 Phase 2 / RQ1-RQ4）で測りたいのは「AF に基づく介入」の効果だが、実際に対面で動くプロトタイプは AF を使っていない。2つの介入脳が別々に進化しており、fix が積み重なるほど統合コストが増える。研究成果物としての一貫性（「本システムは AF で介入判断する」という主張）が実装と乖離している。
- **改善案**: AF を「もう1つの checker」として live に接続する。すなわち `Orchestrator.run_live` + `NetworkXGraphStore` をライブセッションに常駐させ（ON_UTTERANCE 相当を内部化）、`decide_intervention` の出力を `InterventionCandidate(kind="af_l1"/"af_l2", brief=関係ラベル付き提示文, payload=decision)` に変換して `_PendingInterventions` に積む。タイミング・採否・発話は既存の Controller + RealtimeAgent がそのまま担う。これで「AF が what を、Controller が when を、Realtime が how を決める」という研究主張どおりの1本のパイプラインになる。

#### H2. 「新しい会議」リセットとワーカーのカーソル競合で、新会議の発話がエージェントに届かなくなる
- **該当箇所**: `src/das/asr/live/_workers.py:1533-1539`（records スナップショット→ `n` 算出）、`_workers.py:1575`（`state.agent_cursor = n` を無ロックで書く）、`src/das/asr/live/_session_state.py:600-608`（`reset_for_new_meeting` が state_lock 下で `records=[]`, `agent_cursor=0`）
- **問題**: agent_worker が旧会議の records から `n`（例: 120）を計算した直後に HTTP スレッドが reset を実行すると、worker は 1575 行で `agent_cursor=120` を書き戻す。以後 `n > state.agent_cursor` が成立するのは新会議の発話が 120 件を超えてからで、それまで feed・音声呼びかけ検出・沈黙タイマー更新が全て止まる。`topic_cursor`/`drift_cursor`/`fact_cursor` も同型（各 checker スレッドが無ロックで書き、reset が 0 に戻す）。
- **なぜ問題か**: 「新しい会議」はデモ・実験の区切りで必ず使う操作であり、0.25〜3 秒のレース窓は現実的に踏む。踏んだ場合の症状（AI が一切反応しない）は再現困難で、また「fix」が積まれる典型パターンになる。
- **改善案**: カーソルを SessionState 側の meeting 世代（epoch int）とペアで持ち、worker はループ先頭で「自分が知っている世代 != 現在の世代なら自カーソルを 0 に戻す」チェックを入れる。もしくはカーソルを SessionState から追い出し、各ワーカーのローカル変数＋`meeting_id` 比較にする（共有状態を減らす方向が本筋）。

#### H3. 毎発話で議事録全文を state_lock 保持のままディスクに書き直す
- **該当箇所**: `src/das/asr/live/_session_state.py:1121-1126`（`save()` → write_md + write_turns、毎回 rev+1）、`_session_state.py:837-857`（write_md が `with self.state_lock:` 内で open/write/replace）、`_session_state.py:1041-1059`（write_turns 同様）、呼び出し元 `_recv_loop.py:221`（flush ごと）、`_workers.py:1353`（ファシリテーター発話ごと）
- **問題**: 発話が確定するたびに、(a) 全 records を走査して MD/JSONL 全文を再生成し、(b) それを state_lock を握ったままファイル I/O する。会議が長くなるほど O(n²) の書き込みになり、かつ recv スレッドの flush（=次の発話処理）、agent_worker のスナップショット取得、SSE の api_snapshot が全てこのロックで待たされる。
- **なぜ問題か**: 「話者確定→介入判断」のレイテンシ予算（設計文書 §3.5 で計測までしている）を、ログ書き出しという非本質処理が食う。macOS のディスクが遅い瞬間に介入の間（0.9〜2.4s の pause 判定）を外す原因にもなる。
- **改善案**: ①ロック内はスナップショット取得のみ、シリアライズと I/O はロック外に出す。②turns.jsonl は追記（append）に変え、MD/HTML は「終了時＋数秒間隔のデバウンス」で書く。③保存は専用ライタースレッド（queue 渡し）に寄せ、flush パスから I/O を排除する。

#### H4. 抑制理由の日本語文字列が制御フローのプロトコルになっている
- **該当箇所**: `src/das/asr/live/_workers.py:861-873`（`_suppressed_for(..., reason_part=...)` 部分文字列マッチ）、`_workers.py:946-951`（`"直前の介入から間隔不足"` を含むかで `pending.clear_drift()`）、`_workers.py:1010-1015`（`"直前と同じ" in s["reason"]` で skip_invite に変換）、生成側 `src/das/asr/live/_facilitation.py:258-277`
- **問題**: Controller が返す suppressed[].reason は UI/ログ用の日本語説明文だが、worker 側がその**部分文字列**を見て状態遷移（drift 破棄、invite 消費）を分岐している。
- **なぜ問題か**: 文言を1文字直すだけで挙動が変わる（例: cooldown メッセージの推敲で drift が破棄されなくなり C1 の飢餓が悪化する）。テストも文言に結合する。設計上「Controller は採否だけ、後処理は worker」という分離を守った結果、構造化されるべき情報が自然文に落ちてしまった。
- **改善案**: suppressed に機械可読コードを追加する（`{"candidate_id", "code": "cooldown_global" | "same_as_last_invited" | "awaiting_confirmation" | ..., "reason": 表示文}`）。worker は code のみで分岐し、表示文は UI/ログ専用とする。`_facilitation._eligible` の返り値を `(bool, code, message)` にすれば生成側の変更は小さい。

#### H5. stale 判断破棄（epoch）と deadline_ms が実質機能していない（安全装置の見かけ倒し）
- **該当箇所**: `src/das/asr/live/_workers.py:1664-1669` と `1884-1889`（`valid_for_epoch != state.agent_cursor` チェック）、`src/das/asr/live/_facilitation.py:146,163-174`（`deadline_ms` 定義）— `deadline_ms`/`urgency` は _workers.py 中で一切参照されない（grep 確認）
- **問題**: epoch チェックは「arbitrate に渡した `epoch=state.agent_cursor`」と「同一ループ内・同一スレッドの `state.agent_cursor`」の比較であり、arbitrate は同期・決定的（LLM を挟まない）なので、この間に agent_cursor が変わる経路は reset レース（H2）以外に存在しない。つまり設計文書 §8.5 の「LLM 往復中に会話が進んだら判断を捨てる」は実装されていない。同様に §3.5 の「正しいが遅い介入を deadline_ms で破棄」も、deadline_ms を誰も見ていないため機能していない。実際の遅延源は Controller ではなく `agent.trigger()` 後の Realtime API 生成（`_speak_trigger_at` で計測している 1〜3 秒）だが、その間に新しい発話が来ても応答はキャンセルされない（人間の割り込み `interrupt()` 頼み）。
- **なぜ問題か**: 「古い介入は捨てられるから安全」という前提でコードとログ（valid_for_epoch）が読まれるが、実際には trigger 送信後に文脈が変わっても喋り出す。fact 補正（deadline 1500ms 設定）のような鮮度依存の介入で「もう話題が変わったのに訂正する」不自然さとして表面化しうる。死んでいる安全装置は削るか生かすかしないと、次の改修者が誤読する。
- **改善案**: ①本当に守りたい区間は「trigger 送信 → 最初の音声」なので、`response.output_audio.delta` の初回受信時に「trigger 時の epoch と現在の発話数の差」を見て、進んでいたら `interrupt()`（response.cancel）して候補を retry に戻す実装にする（`_realtime.py:238-251` の `_speech_started` 分岐に挿せる）。②やらないと決めるなら epoch チェックと deadline_ms をコードから撤去し、設計文書側に「未実装」と明記する。

#### H6. fact prefilter が正規表現の積層になっており、特定会話への過学習を含む
- **該当箇所**: `src/das/asr/live/_workers.py:80-124`（8 本の正規表現）、特に `_workers.py:100-104`（`_FACT_CREATIVE_EXPRESSION_RE` = 「奴|襲い|跳弾|二丁拳銃|ビビ弾|凶悪だぜ|お釣りだ…」）、`_workers.py:127-167`（`_looks_like_fact_claim`）
- **問題**: LLM 事実判定の前段ゲートが「電話番号・疑問・不確実・好み・創作・メタ・指示語」のネガティブリスト＋「強アンカー・含有・断定文」のポジティブリストという regex 8 本で構成される。git log（`5829cf4 fix(live): fact prefilter を除外中心にし…`, `fd00911 Simplify fact correction prefilter`, `d942fd7 suppress style-focused…`）が示す通り、誤爆のたびに語彙を足す運用になっている。創作表現リストは明らかに特定のテスト（ロールプレイ音声）由来で、一般の会議には無意味な語彙がハードコードされている。
- **なぜ問題か**: 日本語の断定/意見/引用の区別は正規表現では原理的に閉じない。今後も誤爆・取りこぼしのたびに regex が伸び、挙動が説明不能になる（研究上「介入トリガー条件」を論文に書けない）。また `_FACT_QUESTION_RE` が「何|どこ|誰|いつ」を含むため、「首都はどこだっけ、パリだよね」のような**誤り訂正が最も価値を持つ発話**を前段で落とす、という目的逆行も起きる。
- **改善案**: prefilter の役割を「コスト制御」に限定し直す。①機械的に安全なゲートだけ残す（最小文字数・相槌・重複・cooldown）。②「事実断定か否か」の判定は、既に 0.5 秒間隔で呼んでいる軽量 LLM（gpt-5-mini, structured output）に「is_checkable_factual_claim」を 1 フィールド足して一本化する（`_bootstrap.check_fact_correction` は既に見送り判定を持つ。`_bootstrap.py:396-406`）。呼び出し回数が問題なら、発話のバッチ判定（直近 3 発話を1回で分類）にする。③regex を残すなら、replay ハーネスで precision/recall を計測するフィクスチャを用意し、変更のたびに数値で回帰確認する（現状 replay は候補抽出を通すので土台はある）。

### Medium

#### M1. Controller 導入後も legacy 選択ロジックが「例外時 fallback」として二重管理されている
- **該当箇所**: `src/das/asr/live/_workers.py:447-523`（`_select_barge_in_decision`）、`_workers.py:526-568`（`_select_normal_trigger_decision`）、fallback 呼び出し `_workers.py:1651-1663`, `1872-1883`
- **問題**: Controller は決定的で入力も同プロセス生成の dataclass のみ。「想定外失敗時は実績のある従来選択に fallback」とあるが、Controller が例外を投げる経路はバグ以外に無く、その場合 legacy 側は Controller と微妙に違う判定（例: legacy は fact の deque を while で複数消費、drift の hold 診断間隔が違う）で動くため、障害時に**挙動が静かに変わる**。約 120 行の重複が保守対象として残り続ける。
- **なぜ問題か**: 「fallback があるから Controller の品質保証を緩くできる」という逆インセンティブが働く。二系統の判定は将来必ず乖離する（すでに fact 複数消費の差がある）。
- **改善案**: Phase2 完了の宣言として legacy selector を削除し、Controller 例外は「握りつぶして今回 tick は何もしない＋エラーログ」に単純化する。判定の後方互換はテスト（test_facilitation_controller.py）で担保する。

#### M2. 音声呼びかけ検出が呼称キーワード方式で、議題「AI」と本質的に衝突する
- **該当箇所**: `src/das/asr/live/_workers.py:170-207`（`_FACIL_NAMES = "ファシリテーター|進行役|エーアイ|ＡＩ|AI"`、`_CALL_VOCATIVE_RE`/`_CALL_REQUEST_RE`/`_CALL_META_PREFIX_RE`）
- **問題**: 冒頭呼称＋依頼表現ホワイトリスト＋話題化プレフィックスブラックリストの三段構え（git log: `67cde3c feat` → `900e193 fix: harden` の典型的パッチ連鎖）。README のデモ議題が「AIツール導入の是非」であり、「AIは便利だよね」「AIに聞いてみようか」等の文頭 AI 言及が常時発生する。依頼表現リスト（「まとめ|整理|確認|聞いて…」）は取りこぼし側の穴も大きい（「AIさん、今の論点どう思う？」は no_request で落ちる）。STT が句読点を落とす前提の緩め regex も誤爆源。
- **なぜ問題か**: wake-word 検出を語彙 regex で解くのは fact prefilter と同根の袋小路で、診断ログ（voice_call_diag）を足しても regex を足す往復から抜けられない。
- **改善案**: ①呼称を衝突しない固有名にする（設定可能な wake word。例:「ファシリさん」）— 最小コストで精度が跳ねる。②本命は LLM 判定への統合: agent worker は全発話を既に feed しており、drift/fact checker も直近発話を LLM に送っている。そこに `addressed_to_facilitator: bool, request: str` を structured output で足せば、regex 3 本と診断ログの大半を削除できる（レイテンシ要件が厳しければ regex を「候補ゲート」、LLM を「確定」にする2段でもよい）。③UI ボタン（既実装の manual call）を一次手段として明示し、音声呼びかけは実験機能とスコープを切る。

#### M3. 話者確定ポリシーが4箇所に分散し、closed roster が二重実装されている
- **該当箇所**: `src/das/asr/live/_recv_loop.py:74-215`（flush 内の声紋→resolver→fallback→相槌→constrain の逐次判定 140 行）、`src/das/asr/live/_voice_profiles.py:455-462`（`_classify` 末尾の closed roster 落とし）、`src/das/asr/live/_session_state.py:246-278`（`constrain_human_speaker_key` が tracker.auto を見て再度 roster を強制）、`src/das/asr/live/_speaker_policy.py:26-37`（介入用の信頼判定）
- **問題**: 「この発話は誰か」の最終決定が、VoiceProfiles（声紋＋roster）、SpeakerResolver（声紋 vs diarization vs STT）、RecvLoop（相槌・エコー・fallback キー化）、SessionState（人数上限＋roster 再強制）に分散している。closed roster は `13b138d` `a81cea8` の fix で VoiceProfiles 側と SessionState 側の両方に入っており、コメント自体が「声紋以外の経路で作られた匿名キーも含めて未確定に落とす」と二重防御を認めている。
- **なぜ問題か**: 同じポリシーが2層にあると、片方だけ直して片方が古いままになる（まさに closed roster の fix 連鎖がそれ）。「開いた名簿/閉じた名簿 × 声紋あり/なし × 外部diarizationあり/なし」の組合せ挙動を誰も一望できない。
- **改善案**: SpeakerResolver を唯一の決定点に昇格させる。入力 =（STT ラベル, 声紋判定結果, diarization 重なり, roster ポリシー, 人数上限, 相槌フラグ）、出力 =（最終キー, source, confidence, reason）。VoiceProfiles は「埋め込み照合の結果を返す」だけ、SessionState.constrain_human_speaker_key と RecvLoop 内の分岐は Resolver へ移して削除する。テストも Resolver 1 点に集約できる。

#### M4. SessionState が God object 化している（状態＋レンダリング＋永続化＋状態機械）
- **該当箇所**: `src/das/asr/live/_session_state.py:41-166`（初期化だけで 27 種の責務混在）、`859-1039`（write_html: HTML 文字列組み立て 180 行）、`406-512`（api_snapshot）、`741-767`（manual call の状態機械）、`517-574`（WAV ヘッダ操作と PCM オフセット管理）
- **問題**: 発話記録・声紋・diarization・介入キュー・積極性・PCM バッファ・WAV・出力パス・SSE リビジョン・手動呼び出しステータス・HTML レンダリングが 1 クラスに同居する。1126 行のうち約 300 行は表示（HTML/MD）生成。
- **なぜ問題か**: どのフィールドがどのロックで守られるかの対応が読めず、H2/H3 のような競合・ロック内 I/O を誘発している。テストで FakeState を作る際に「必要な属性だけ持つ偽物」になり、本体側に `getattr(..., None)` ガードが増える（M7）。
- **改善案**: 責務で 5 分割する。`MeetingLog`（records/topics/rekey/disp_name、state_lock を内包）、`AudioBuffers`（pcm/asr_pcm/wav、buf_lock）、`InterventionState`（キュー・イベント・manual状態機械・review 書き出し）、`TranscriptWriter`（md/html/turns、専用スレッド）、`SessionState` はそれらの束＋ライフサイクル（reset/stop/rev）だけにする。既にメソッド境界は責務ごとに揃っているため、機械的な移動で済む部分が多い。

#### M5. replay が live の採否・タイミングを再現しない「別物のシミュレータ」になっている
- **該当箇所**: `src/das/asr/live/replay.py:583-599`（drift を `n % _DRIFT_CHECK_INTERVAL == 0` で駆動 — live は cursor 差分駆動 `_workers.py:1161`）、`replay.py:602-611`（invite を `% _INVITE_WARMUP == 0` で駆動 — live は 8 秒間隔の時間駆動 `_workers.py:1291`）、Controller/pause/cooldown は一切通らない
- **問題**: replay は checker（候補生成）だけを、live と異なる周期規則で回す。FacilitationController の採否、pause/cooldown、hold、barge-in は再現されないため、「replay で介入候補が出た/出ない」と「live で実際に喋る/黙る」が一致しない。
- **なぜ問題か**: replay は「facilitator tuning を再現可能にする」ためのハーネス（モジュール docstring）だが、C1 のような採否層のバグ・チューニング課題はこのハーネスでは絶対に見つからない。閾値調整の結論が live に転移しない。
- **改善案**: turns.jsonl は ms を持っているので、記録された時刻で仮想クロックを進めながら `_PendingInterventions` + `FacilitationController` を通す「採否込みリプレイ」を第2モードとして足す（LLM 部分は保存済み interventions.jsonl の判定を再生すれば no-api で回る）。checker 単体検証モードは残してよいが、live と同じ駆動条件（cursor 差分・時間間隔）を共有関数に括り出して使うこと。

#### M6. 「沈黙」を音声でなく STT 確定イベントの到着時刻で測っている
- **該当箇所**: `src/das/asr/live/_workers.py:1541-1543`（新 record 観測時に `_last_utt_time[0] = time.monotonic()`、0.25s tick）、`_workers.py:1624`（`_silence_elapsed = now - _last_utt_time[0]`）、pause 閾値 `src/das/asr/live/_constants.py:367-372`（fact 0.9s / drift 1.8s / retry 2.4s …）
- **問題**: `_last_utt_time` は「STT が発話を確定し、worker がそれを見つけた時刻」であり、実際に人が話し終えた時刻ではない。Soniox の endpoint 検出遅延＋flush＋tick で数百 ms〜1 秒級のオフセットが乗る。逆に、長い発話の途中でも token 未確定なら records が増えず、「まだ喋っているのに沈黙が伸びる」ケースもある（partial は見ていない）。
- **なぜ問題か**: 0.9〜2.4 秒という繊細な pause 設計が、実会話の間ではなく STT の挙動を測ってしまう。介入が早すぎて被る／遅すぎて間を逃す、の両方向のずれが STT プロバイダ・ネットワーク状況依存になる。対面会議介入という目的に対して、フロア判定の物理的根拠が弱い。
- **改善案**: PCM は手元にある（audio_q/sender）ので、簡易 VAD（エネルギー閾値か webrtcvad）で「最後に音声があった時刻」を SessionState に持たせ、フロア判定はそれを使う。少なくとも partial トークン受信時刻（`show_partial` 経由で既に持っている）で `_last_utt_time` を更新し、「喋っている最中の誤介入」を塞ぐべき。

#### M7. テスト用 FakeState への配慮が本体コードに漏れている（暗黙インターフェース）
- **該当箇所**: `src/das/asr/live/_workers.py:224-226`（「state が未対応（テストの FakeState 等）なら no-op」）、`_workers.py:281-283`, `290-296`, `757-759`（`callable(getattr(state, ...))` ガード群）
- **問題**: SessionState の必須メソッドが「あれば呼ぶ、なければ黙って何もしない」で扱われており、テストの偽物を通すための穴が本番でも開いている。本番で属性名を typo しても no-op で沈黙する。
- **なぜ問題か**: 観測ログ（intervention events / manual status）は研究の一次データであり、「静かに記録されない」故障モードは最悪。またインターフェースが暗黙なので、SessionState 分割（M4）の際に何が契約なのか分からない。
- **改善案**: worker が依存する操作を `Protocol`（例: `InterventionStatePort`: write_intervention_event / add_intervention_event / set_manual_call_status / add_intervention_review / proactivity / stop ...）として `_speaker_policy.py` 並みの小モジュールに定義し、getattr ガードを全廃する。テストは Protocol を満たす Fake を書く。

#### M8. 介入ポリシー（pause/cooldown/優先度）が3箇所に重複エンコードされている
- **該当箇所**: `src/das/asr/live/_constants.py:367-377`（`_INTERVENTION_PAUSE_*` ほか）、`src/das/asr/live/_facilitation.py:162-174`（`_KIND_POLICY` に再掲）、`src/das/asr/live/_workers.py:1712-1727` ほか各 dispatch 節（timing metadata 生成に同じ定数を再度個別参照）、legacy selector（M1）にも同閾値
- **問題**: 「fact は 0.9 秒の間で cooldown 2 秒」という1つの事実が、Controller のポリシー表・dispatch のログ生成・legacy 判定の3系統でそれぞれ定数参照される。kind→pause の対応を変えるとき3箇所の同期が必要で、実際 `_controller_normal_decision` の silence は候補 payload 経由（`pause_required`）という第4の経路まである。
- **なぜ問題か**: 積極性プロファイル（S5）の cooldown は Controller に注入されるのに pause は `_KIND_POLICY` 固定、という非対称も生まれており、「積極性を上げたのに間が変わらない」類の分かりにくさに繋がる。
- **改善案**: `_KIND_POLICY` を唯一の出所とし、dispatch のログは `decision` に採択時ポリシーを含めて返す（Controller が「採択候補・適用 pause・適用 cooldown」を decision に同梱すればログ側の再計算が消える）。proactivity プロファイルは `_KIND_POLICY` への差分（pause/cooldown の係数）として定義する。

#### M9. VoiceProfiles の閾値群が実験ログ駆動の手調整で膨張している
- **該当箇所**: `src/das/asr/live/_voice_profiles.py:102-105`（モデル別3値＋AI閾値）、`137-154`（margin/short_floor/short_bonus/short_margin_mult/enroll_min_total_chars=45/enroll_win_sec/enroll_consist_bonus/_POOL_CAP=24）、`334-339`（人物別閾値 = 中央値-0.12）、`407-417`（continuity_th = max(0.25, person_th-0.12)）、コメント `101`「2026-06-11夜: 0.30→巻き取り復活/個人別→本人分裂のため固定の中庸値に」、`174-181`（撤去した自動校正の墓標コメント）
- **問題**: 15 個前後のマジック閾値が相互依存しており、チューニング履歴が日付入りコメントとして残っている（= 回帰の判定基準がコードコメントにしかない）。一方でクラス設計自体（凍結プロファイル・累積文字数登録・pool クラスタリング・遡及リネームは #ラベル昇格のみ）は筋が良い。
- **なぜ問題か**: 閾値の妥当性が「その日の会議でうまくいったか」でしか検証されておらず、モデル変更（redimnet→次世代）や環境変更（マイク・部屋）のたびに同じ手調整を繰り返すことになる。研究として話者特定精度を主張するにも根拠が出せない。
- **改善案**: 既に diag.jsonl に全判定（sim/second/kind/label）を落としているので、①ラベル付き音声フィクスチャ（scripts/make_overlap_testset.py がある）で「閾値セット→混同行列」を出す校正スクリプトを常設し、DEFAULTS の出所をコードコメントから測定結果ファイルに移す。②short_* 系の特殊経路は、person_th と margin をパラメタ化した単一の受理関数（`accept(sim, second, person, duration)`）に統合し、経路分岐を減らす。

#### M10. interrupt() と再生スレッドの音声キュー操作が競合しうる／graceful yield の複雑さ
- **該当箇所**: `src/das/asr/live/agents/_realtime.py:542-563`（interrupt が `_audio_q` を drain→300ms 分だけ積み直し→終端マーカー→`ai_speaking = bool(kept_items)`）、消費側 `src/das/asr/live/agents/_base.py:112-133`（playback スレッドが並行に get）
- **問題**: interrupt スレッドが get_nowait で吸い出している最中も playback スレッドは get で消費し続けるため、「残す 300ms」の選別と再 put の間に順序逆転・二重終端が起こりうる。また `ai_speaking`/`_responding` は複数スレッドから無ロックで読み書きされる（epoch 機構で終端の誤爆だけは防いでいる）。
- **なぜ問題か**: 症状は「割り込み時に音声の断片が遅れて再生される」「エコーウィンドウの終了時刻が僅かにずれる」程度で致命ではないが、Bug 2/4/6 と番号が振られた修正史が示す通り、この层は既に何度も踏み抜いている。キュー＋フラグ＋epoch の3点併用が原因。
- **改善案**: 再生を「現在再生を許可する epoch」を単一の Atomic 値として持つ設計にする: interrupt は `allowed_epoch` を進めるだけ、playback は要素の epoch < allowed_epoch なら捨てる。drain/再put が不要になり、graceful yield は「切替後 300ms は捨てずに再生」を playback 側の判断にできる。

#### M11. 手動呼び出しの状態機械が4ファイルに分散している
- **該当箇所**: `src/das/asr/live/_session_state.py:715-767`（queued/連打上書き）、`_workers.py:938-945`（waiting）、`_workers.py:1755-1756`（dispatched）、`_workers.py:1595`（cancelled）、`_workers.py:914-915`（expired）、`_session_state.py:1117-1119`（delivered への遷移が「直近 trigger 理由が manual_call かつ status==dispatched」という間接条件）
- **問題**: queued→waiting→dispatched→delivered/expired/cancelled という遷移が、HTTP ハンドラ・worker の3箇所・delivery フックに散らばり、遷移の正当性（例: expired 後に delivered が来ない保証）をどこも検査しない。delivered 判定は `_last_intervention_event_reason` という別目的のフィールド越しで、manual 直後に別介入が挟まると誤遷移しうる。
- **なぜ問題か**: git log（`bac1180 fix: refresh manual call status on new request`, `0794a37 fix: clear pending manual calls when disabled`）の通り、状態の置き場が散っているため fix が散発する構造。
- **改善案**: `ManualCallSession` クラス（id, request, source, created_at, status）を1つ作り、遷移メソッド（accept/hold/dispatch/delivered/expire/cancel）に正当な遷移表を持たせる。delivery との紐付けは `_last_intervention_event_reason` でなく、trigger 時に manual の id を intervention event に埋めて delivery で照合する（event_id の仕組みは既にある）。

### Low

#### L1. クロージャ時代の可変セルハックが残っている
- **該当箇所**: `src/das/asr/live/_session_state.py:164-165`（`_last_utt_time = [time.monotonic()]`, `_was_in_echo = [False]`）、利用側 `_workers.py:1488-1489`
- **問題/改善**: main() クロージャから移した名残の 1 要素リスト。SessionState の通常属性（float / bool）にすれば済む（worker はどのみち state 参照を持つ）。

#### L2. `_constants.py` が定数・LLMプロンプト・HTMLテンプレート・ANSIコードの寄せ集め
- **該当箇所**: `src/das/asr/live/_constants.py:15-173`（レガシー HTML_TMPL＋JS 160 行）、`177-311`（プロンプト5本）、`323-410`（タイミング定数）
- **問題/改善**: プロンプトは `_prompts.py` へ、HTML_TMPL は file:// 用レガシー表示（`write_html`）とセットで削除候補（サーバー UI が既定の今、二重フロントエンドを保守する価値は低い）。タイミング定数は M8 の `_KIND_POLICY` 集約と併せて整理。

#### L3. モジュール間で私有属性へ手を突っ込んでいる
- **該当箇所**: `_workers.py:511`, `631`（`agent._pending_intervention`）、`_workers.py:1383`（`p._connected`, `p._responding`）、`_workers.py:1348`（`state.agent._last_speak_latency_ms`）、`_recv_loop.py:186`（`_src._best_similarity`）
- **問題/改善**: RealtimeAgent 側にロック付き公開プロパティ（`has_pending_intervention`, `is_busy`, `last_speak_latency_ms`）を生やして参照を置換する。現状は _state_lock で守っているはずの状態を外から無ロックで覗いており、規約が破れている。

#### L4. 声紋の同名登録ポリシーが経路によって矛盾する
- **該当箇所**: `src/das/asr/live/_voice_profiles.py:483-499`（`_enroll` は duplicate_name を拒否 — fix `ec98941`）と `579-604`（`enroll_from_audio` は「既に同名があれば上書き」）
- **問題/改善**: UI の /rename は重複拒否、開始前の /api/enroll は黙って上書き。事前登録で他人の声紋を潰せる。ポリシーを片方（明示 overwrite フラグ付き拒否既定）に統一する。

#### L5. `_run_sender` の送信失敗が完全に無音で握り潰される
- **該当箇所**: `src/das/asr/live/_workers.py:2114-2118`（`except Exception: pass`）
- **問題/改善**: 音声を捨てる設計自体は接続リセット中の仕様として妥当だが、連続失敗のカウントとレート付きログ（「直近5秒で N チャンク破棄」）くらいは出さないと、STT 側の無応答と切り分けられない。

#### L6. drift 確認カウントのリセット窓 20 秒がインラインのマジックナンバー
- **該当箇所**: `src/das/asr/live/_workers.py:316-318`（`if now - self.last_drift_request_at > 20.0`）
- **問題/改善**: drift_confirmations の意味（「20 秒以内に2回」）を決める重要パラメータが `_constants.py` の体系外にある。`_DRIFT_CONFIRM_WINDOW` として定数化し、proactivity プロファイルの説明に含める。

#### L7. invite の対象同定が表示名文字列で行われている
- **該当箇所**: `src/das/asr/live/_workers.py:1311-1329`（`state.disp_name(sp)` を valid_invite_targets に採用し、LLM の返す名前と文字列一致）、`_workers.py:1963`（`_last_invited` も表示名）
- **問題/改善**: 会議中のリネーム（rekey）で名前が変わると連続声かけ抑制やターゲット検証が外れる。内部キーで持ち、LLM とのやり取り境界でのみ表示名へ変換する。

---

## 対症療法パターンの分析

git log の fix 集中箇所と対応するコードから、パッチが積み重なっている領域は次の4つに整理できる。いずれも「連続的・確率的な判定を、離散的なルール（regex・固定閾値・フラグ）で解こうとした」ことが根本原因で、誤爆・取りこぼしのたびにルールを1本足す構造になっている。

### 1. テキスト表層ルールの積層（fact prefilter / 音声呼びかけ / 相槌）
- **兆候**: `_looks_like_fact_claim` の regex 8 本（H6）、`_detect_facilitator_call_ex` の三段 regex（M2）、`_BACKCHANNEL_RE`（_constants.py:399-409。相槌語彙の列挙で、末尾文字クラスに「うんはいええそっか…」を詰め込む力技）。fix 履歴: `5829cf4`, `d942fd7`, `fd00911`, `900e193`, `67cde3c`。
- **根本原因**: 「LLM を呼ぶ回数を抑えたい」というコスト制約から、意味判定をローカル regex に肩代わりさせた。しかし判定対象（断定か意見か、呼びかけか話題か、相槌か発言か）はすべて文脈依存で、表層語彙では原理的に閉じない。ルールが会話コーパス（しかも特定のデモ音声）に過学習していく。
- **本質的な再設計**: 「発話ごとの表層分類」を1本の軽量 LLM 呼び出しに統合する。現在 drift(1s)/fact(0.25s)/participation(1s) が別々に直近発話を LLM へ送っているので、**発話確定ごとに1回だけ**「この発話の分類（相槌/断定/質問/ファシリテーター宛/…）」を structured output で取り、その結果を全 checker が共有する Utterance アノテーションにする。regex は「明白な短文（<8 文字等）を無料で落とす」だけに縮退させる。LLM 呼び出し回数はむしろ減り、判定根拠がログに残るため研究上も説明可能になる。

### 2. 話者同定の閾値ホイッスル（voice profiles / closed roster）
- **兆候**: M9 の閾値群、M3 の closed roster 二重実装。fix 履歴: `13b138d`, `a81cea8`, `bba6c05`（撤回コミット）, `c64f1e0`, `49ab807`, `c3e21cb`, `e0df171`, `9268ded`。「巻き取り復活/本人分裂」のコメント（_voice_profiles.py:101）が示す通り、閾値を上げると分裂・下げると誤併合のトレードオフを行き来している。
- **根本原因**: (a) オンライン・オープンセット話者同定という本質的に不確実な問題に対して、「その場で1回で確定し、確定は覆さない（凍結）」という不可逆コミットを課している。(b) Soniox ラベルが「新しい声を既存ラベルに混ぜる」ため、前提（ラベル≒人物）が崩れた状態で閾値を調整している。(c) 判定ポリシーが4箇所（M3）に分散し、fix がその都度いちばん近い層に入る。
- **本質的な再設計**: 2方向ある。①**実験用途に振り切る**: 対面実験は参加者が事前に分かるのだから、closed roster＋事前登録（すでに実装済み: enroll_from_audio＋読み上げ台本 UI）を「サポートされる唯一のモード」と宣言し、オープンセット自動登録系（pool/累積文字数/人物別閾値/short 経路）を実験コードから外す。閾値問題の大半が消える。②**遡及修正を一級市民にする**: records の話者を「暫定」とみなし、数十秒ごとにオフラインで再クラスタリングして過去ラベルを修正する（rekey 機構・diag ログ・UI の「声紋補正」表示は既にあるので土台はある）。リアルタイム介入に使う話者情報は `_speaker_policy.is_reliable_human_speaker` が既に低信頼を落とす設計なので、暫定ラベルの揺れと共存できる。

### 3. 採否レイヤの移行残骸（legacy fallback / shadow 語彙 / 文字列プロトコル）
- **兆候**: legacy selector の温存（M1）、`legacy_decision`・`type: "shadow_decision"` という歴史的キー名の据え置き（_workers.py:710-721, _session_state.py:1078-1092）、抑制理由の文字列マッチ（H4）、機能しない epoch/deadline（H5）。fix 履歴: `27ab1e5`(shadow) → `01f5776`(Phase2) → `ea527db`(Phase3) → `88f308e`(dead code purge) と段階移行の途中。
- **根本原因**: shadow 運用→実運用の段階移行自体は正しい手順だったが、「Phase2 完了後に旧経路とログ互換を畳む」最終ステップが実行されず、両方が生き続けている。移行のための一時構造（legacy 逆変換 `_BargeInDecision`/`_NormalTriggerDecision`、reason 文字列での後処理通知）が恒久コードになった。
- **本質的な再設計**: Controller の出力（FacilitationDecision）を dispatch が直接消費する形に一本化する。具体的には decision に「採択候補への参照＋採択時ポリシー＋後処理指示（drift クリア/invite 消費）」を構造化して含め、`_BargeInDecision` への逆変換・suppressed 文字列マッチ・legacy selector・タイミング再計算（M8）をまとめて削除する。ログスキーマは v2 として `decision` キーに改名し、replay 側ローダーで旧キーを読み替える。これで _workers.py の 500 行前後が消える。

### 4. hold・保留状態の場当たり管理（C1 / manual TTL / fact TTL / retry TTL）
- **兆候**: fact は TTL30s＋drop_stale、manual は TTL30s＋expired ステータス、retry は TTL60s＋回数上限、drift は TTL なし（C1）と、候補種別ごとに保留・破棄の実装がバラバラ。hold が「全レーン停止」を意味してしまう制御構造。fix 履歴: `0794a37`, `09968bf`, `7ddb219`, `211caa9`, `d502ce8`。
- **根本原因**: `_PendingInterventions` が「種別ごとの専用フィールド＋専用 drop メソッド」の集合であり、「候補には寿命と再評価タイミングがある」という共通概念がデータ構造に無い。Controller は expires_at を見られるのに、生成側が種別ごとに設定したりしなかったりする。
- **本質的な再設計**: 保留候補を単一の `list[InterventionCandidate]`（全種別、必ず expires_at 付き）にし、drain は「キュー→候補リストへの変換＋期限切れの一括掃除＋期限切れイベントの一括ログ」だけを行う。worker のループは「候補リスト全体を1回 arbitrate → 採択があれば dispatch、なければ次 tick」に単純化され、hold という中間概念と C1 の飢餓が構造的に消える。

---

## 根本的な再設計提案

### A. 「AF 脳」と「音声の体」の統合（最重要・研究整合性）
現状は (i) live の rule-based 介入と (ii) `das.agents.facilitation` の AF-based 介入が並立している（H1）。research の主張（貢献②③）を対面プロトタイプで成立させるには、次の統合が最短:

1. ライブセッションに Orchestrator + GraphStore を常駐させる（確定発話 → extraction/linking は既に非同期設計）。
2. 新しい checker `_run_af_checker` が数秒周期で `decide_intervention(history, store)` を呼び、SKIP 以外を `InterventionCandidate(kind="af_l1"|"af_l2", brief=関係ラベル付き提示文, target_speaker=addressed_to, expires_at=…)` として積む。
3. FacilitationController の `_KIND_POLICY` に af_l1/af_l2 のポリシー（L2 は global cooldown 長め等）を追加。発話は既存 RealtimeAgent が担う（trigger に af 用コンテキスト節を1つ足すだけ）。

これにより drift/fact/invite は「AF がまだ薄い序盤の補助介入」、AF 介入は「本命」という位置付けが1本のパイプラインで表現でき、interventions.jsonl の観測資産（timing/review）が研究データとしてそのまま使える。

### B. イベント駆動ディスパッチャへの置き換え（tick ポーリングの限界）
0.25s/1s tick のポーリング worker 群は、C1（hold 飢餓）や H2（カーソル競合）のような「ループ構造由来」のバグを生みやすく、反応も tick 量子化される。発話確定・候補追加・沈黙タイマー・エージェント状態変化を単一の優先度付きイベントキューに流し、1本のディスパッチャスレッドが「イベント→候補更新→arbitrate→dispatch」を処理する形にすれば、(a) 共有カーソルが消える、(b) hold は「次のイベントまで待つ」に自然に置き換わる、(c) 沈黙は monotonic デッドラインのタイマーイベントで表現できる。既存の Controller・candidate 抽象はそのまま使えるため、書き換え範囲は `_run_agent_worker`（約 500 行）に閉じる。asyncio 化までは必須ではないが、するならこのディスパッチャ1本を async にするのが自然な単位。

### C. ファイル分割の具体案
- `_workers.py`（2153 行）→ 5 分割:
  - `_fact_prefilter.py`（regex 群＋`_looks_like_fact_claim`、将来 LLM 分類に置換される単位）
  - `_voice_call.py`（呼びかけ検出＋診断ログ）
  - `_checkers.py`（agenda/topic/drift/fact/participation の監視スレッド）
  - `_dispatch.py`（`_PendingInterventions`、Controller アダプタ、`_run_agent_worker`、review recorder）
  - `_audio_io.py`（`_run_from_mic`/`_run_from_wav`/`_run_sender`/`_cleanup`）＋ `set_session_mode`/partner attach は `_modes.py`
- `_session_state.py`（1126 行）→ M4 の 5 分割（MeetingLog / AudioBuffers / InterventionState / TranscriptWriter / SessionState 本体）。
- `_constants.py` → `_prompts.py` を分離、レガシー `HTML_TMPL`+`write_html` は削除を検討（サーバー UI に一本化）。

### D. 「間」の物理的根拠の強化（対面介入の品質）
M6 の VAD 導入に加え、trigger→初回音声のレイテンシ計測（既にある `_last_speak_latency_ms`）を使い、「pause 判定時点＋発話開始レイテンシの p50」を見込んだ先読みトリガー（pause が 0.9s 必要で発話開始まで 1.2s かかるなら、0.9-1.2s ではなく沈黙 0.3s 時点で生成を開始し、再生開始をゲートする）を検討する価値がある。Realtime API は生成と再生を分離できる（audio delta をキューに貯めてから再生）ため、「生成は早めに・再生はフロアが空いてから」という2段ゲートは現アーキテクチャの小改修で実現でき、対面での介入の自然さに直結する。

---

## 補足（良い点）

- `_facilitation.py` は決定的・入出力 dataclass・ポリシー表駆動で、この規模のプロトタイプとしては模範的。
- 介入の trigger/delivery/review/timing の JSONL 記録と replay UI は、研究の「介入根拠の透明性」（C3）をエンジニアリングで裏打ちしており価値が高い。
- `_speaker_policy.py` の「内容としては使うが個人としては使わない」という信頼度の二層扱いは、誤介入リスクの下げ方として本質的で良い設計。
- RealtimeAgent の trigger の test-and-set、送信成功までスナップショットをクリアしない設計（Bug 2/4 対応）、エコー除去の3層防御は、音声エージェント特有の落とし穴を丁寧に潰している。
