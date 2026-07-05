# AF×ライブ統合 新規コードレビュー（直近60コミット分）

対象HEAD: `4bf762e`（2026-07-04 15:37 コミット時点）。設計書 `docs/design/af_live_integration_2026-07.md` および `docs/review-2026-07-02/{01,07}` を前提知識として、既知指摘（af_l2連発の根本修正、生成先行ゲートのController駆動化）は再確認のみで再指摘しない。

---

## 1. シナリオ別トレース結果

- **S1（取り込み遅延と価値ゲート/鮮度TTLの整合）**: 問題なし。`_af_checker_tick`（`src/das/asr/live/_workers.py:1489-1587`）は L1 なら必ず `apply_l1_value_gate`（`src/das/agents/facilitation.py:579-634`）を通し、`presented` 集合更新も同一箇所で行う。`_PendingInterventions.drop_stale_af`（TTL 45s/90s, `_workers.py:282-290`）と `InterventionCandidate.expires_at`（`_workers.py:475-486`）が二重にTTLを効かせており、古い介入が無期限に生き残って喋られる経路は見当たらない。
- **S2（hold中の通常経路によるaf以外の採択とtrigger）**: **Critical**。下記指摘1参照。hold中（`_responding=True`, `ai_speaking=False`）でも `_run_agent_worker` の通常レーンは agent のbusy状態を一切見ずに評価・trigger試行する。`RealtimeAgent.trigger()` 側は `_responding` のtest-and-setで実送信を防ぐため二重発話にはならないが、**採択・消費・ログ記録だけが空振りする「介入の握りつぶし」**が起きうる。
- **S3（release/cancelとinterrupt/barge-inの競合）**: Medium。下記指摘2参照。`agent.interrupt()` の呼び出し経路（`_workers.py:2028-2029`）が `agent.ai_speaking` 条件付きのため、hold中（`ai_speaking=False`）は現状の呼び出し経路では発火せず、当初懸念した二重 `response.cancel` は**通常の自動割り込みフローでは発現しない**。ただし `RealtimeAgent.interrupt()`（`_realtime.py:581`）自体には hold 状態を考慮したガードがなく、将来の呼び出し経路追加（手動interrupt等）で問題化しうる潜在的な脆弱性として記録。
- **S4（meeting epochリセットとAF/ゲート/バックオフの整合）**: **High**。下記指摘3参照。会議リセット時に `af_requests` キューがクリア対象から漏れている。加えて `_run_agent_worker` 内の `_af_gate`（`_AfEarlyGenGate`インスタンス）・`_af_held_text`/`_af_held_kind`・`_pending.af` を epoch 変化で明示的にリセットする経路が存在しない。
- **S5（--afなしでの副作用ゼロ確認）**: 問題なし。`_run_af_checker`は`state.af_runtime is None`で即return（`_workers.py:1505-1507`）、`_af_gate_status`は`pending.af`が無ければ`("none", None)`即return（`_workers.py:1721-1722`）、`_build_candidates`のaf分岐は`pending.af`がNoneなら候補を追加しない（`_workers.py:471`）。af_requestsキュー自体、AF無効時は何も積まれないため既存経路への影響は無い。
- **S6（facilitationの境界条件）**: 問題なし。窓境界（`turn_index >= window_start`、境界含む: `facilitation.py:199,330,604,648`）、transcript空/ノード0（`decide_intervention`冒頭の `n_utts == 0` skip、`facilitation.py:361-364`）、全提示済み（`apply_l1_value_gate`が`kept`空なら`skip`を返す、`facilitation.py:623-626`）、cluster_id未割当ノードでも`detect_bias`/`apply_l1_value_gate`はノード単位で動作し例外にはならない、をそれぞれ確認。
- **S7（note_intervention/応答エッジのスレッド安全性）**: Medium（潜在リスク）。`_detect_responds_to`（`_af_runtime.py:171-200`）で `_intervention_lock` 保持中にコピーした `interventions` リストの要素（dict）に対し、ロック解放後に `iv["embedding"] = ...` のインプレース変更を行っている。現状は同一スレッド（AFRuntime専属スレッド）内でしか`_detect_responds_to`/`ingest_utterance`が呼ばれないため実害はないが、ロック規律が崩れている。
- **S8（eval側decide_intervention共用への波及）**: 問題なし。`src/das/eval/conditions.py:402-492`は`FacilitationAgent(llm=..., max_items=...)`のみを使い、`active_window`は新規デフォルト値(12)のまま、`apply_l1_value_gate`は呼ばない。`turn_id`はシミュレーション側で連番付与されるため`turn_index`ベースの窓判定も従来どおり機能する。

---

## 2. 指摘一覧

### Critical

**C-1. hold中／agent応答生成中に通常レーンがaf以外の介入を「採択したことにして」握りつぶす**
- file:line: `src/das/asr/live/_workers.py:2151`（バージインレーンのガード条件）, `src/das/asr/live/_workers.py:2369-2499`（通常レーンの評価・dispatch）, `src/das/asr/live/agents/_realtime.py:419-426`（`trigger()`のtest-and-set早期return）
- 問題: `_run_agent_worker`のバージインレーン（fact/manual/drift/retry）は`if not agent._responding and not agent.ai_speaking:`（`_workers.py:2151`）でガードされているが、このifがFalseの場合に「何もしない」処理が無く素通りする。その後2313/2332行のecho/partnerチェックを経て、agentのbusy状態を一切参照しない通常レーン（`_controller_normal_decision`, `_workers.py:2369-2410`）に到達し、`summarize`/`silence`/`invite`/`af_l1`/`af_l2`いずれかを「採択」して`agent.trigger(...)`を呼ぶ（例: `_workers.py:2440`, `2489`）。`RealtimeAgent.trigger()`は`_responding`のtest-and-set（`_realtime.py:419-421`: `if self._responding: return`）で実送信こそ防ぐが、呼び出し元は戻り値を見ておらず、`_pending.summarize = None`（`_workers.py:2443`）・`_pending.clear_af()`（`_workers.py:2498`）・`_note_intervention`によるcooldown履歴記録・`_log_intervention_event`によるログ記録を無条件に実行する。
- なぜ問題か: af_l1/af_l2のhold_playbackは最大`MAX_HOLD_SEC=8.0`秒（`_workers.py:1612`）持続する。この間、`_responding=True`が続くため、通常レーンで一度「採択」された介入候補（summarize等）は実際には音声化されずに消費・cooldown登録だけがされ、機会そのものが失われる。ログ上は「summarize: ...」とtrigger成功したかのように記録される（`_workers.py:2437-2439`）ため、observability（`intervention_review.jsonl`, `.interventions.jsonl`）と実際の発話が乖離し、研究データとしての信頼性を損なう。af_l1/af_l2自身が通常レーンで採択された場合も同様に空振りしうる。
- 修正案: 通常レーン評価の前段（もしくは`_controller_normal_decision`呼び出し前）で`agent._responding or agent.ai_speaking`をチェックし、busyならControllerを呼ばず`_review.evaluate`のみ行うガードを追加する（バージインレーン同様の構造）。あるいは`trigger()`に「送信できたか」を示す戻り値を持たせ、呼び出し側がFalseの場合は候補を消費せず・cooldown登録もしないようにする。

### High

**H-1. 会議リセット時に `af_requests` キューがクリアされない**
- file:line: `src/das/asr/live/_session_state.py:775-781`（会議リセットのキュークリア処理）, `src/das/asr/live/_session_state.py:126`（`af_requests`定義）
- 問題: 会議リセット処理は`for q in (self.drift_requests, self.invite_requests, self.factcheck_requests, self.manual_call_requests, self.summarize_requests): ...`（`_session_state.py:775-776`）でキューを空にするが、この対象タプルに`self.af_requests`（`_session_state.py:126`で定義）が含まれていない。
- なぜ問題か: リセット直前に`_run_af_checker`が積んだ古い会議のAF介入候補が`af_requests`キューに残っていた場合、新しい会議（新epoch）の`_pending.drain()`（`_workers.py:250-258`）でそのままpending.afに取り込まれ、TTL（45s/90s）が切れるまでの数十秒間、**別会議・別参加者の内容に基づく介入テキストが新会議のControllerに渡り、採択されうる**。対面会議という文脈では「前の会議の内容を話してしまう」という研究的・実務的に重大な不整合になる。
- 修正案: リセット対象キューのタプルに`self.af_requests`を追加する。

**H-2. `_run_agent_worker`内のaf関連ローカル状態（`_pending.af`, `_af_gate`, `_af_held_text/_af_held_kind`）が epoch リセットで明示的にクリアされない**
- file:line: `src/das/asr/live/_workers.py:1945`（`_pending = _PendingInterventions()`をループ外で1回生成）, `_workers.py:1954-1956`（`_af_gate`, `_af_held_text`, `_af_held_kind`をループ外で1回生成）, `_workers.py:1992,2006,2019`（`meeting_epoch`比較はこの3箇所のみで発話供給ガード用途に限定）
- 問題: `_run_af_checker`側は`state.meeting_epoch != epoch`で`presented`/`af_gate`/`facil`をリセットする経路を持つ（`_workers.py:1765-1769`）が、`_run_agent_worker`側は`meeting_epoch`変化を検知して`_pending`, `_af_gate`, `_af_held_text/_af_held_kind`をリセットする専用コードパスが存在しない。
- なぜ問題か: H-1と合わせて、会議リセット直後も旧会議の`_pending.af`やhold中の`_af_gate`状態（`is_holding=True`のまま）が持ち越され、新会議の最初の数tickで古い介入が`release`/`deliver`される余地がある。三層分離の原則（when=Controller/pause, epoch整合=H2パターン）が、既存のdrift/fact系には及んでいるがAF系には及んでいない非対称な実装になっている。
- 修正案: `_run_agent_worker`のメインループ冒頭で`meeting_epoch`変化を検知し、`_pending.clear_all()`（既存メソッド、`af`も含めてクリアされる）に加え`_af_gate.reset()`・`_af_held_text=""`・`_af_held_kind="af_l1"`をリセットする処理を追加する。

### Medium

**M-1. `RealtimeAgent.interrupt()` が hold_playback 状態を考慮しておらず、将来の呼び出し経路変更に対して脆弱**
- file:line: `src/das/asr/live/agents/_realtime.py:581`（`if not self.ai_speaking and not self._responding: return`）, `_realtime.py:640-650`（`_responding=False`設定、`_hold_playback`/`_held_audio`は未変更）, `_workers.py:2028-2029`（`agent.interrupt()`の唯一の呼び出しが`agent.ai_speaking`条件付き）
- 問題: 現状`agent.interrupt()`の呼び出し元（`_workers.py:2029`）は`agent.ai_speaking`を前提条件にしているため、hold中（`ai_speaking=False`）は到達しない。しかし`interrupt()`自体のガード（`_realtime.py:581`）は`_responding`もOR条件に含むため、hold中に呼ばれれば通過してしまう。通過した場合、`_hold_playback`/`_held_audio`はクリアされないまま`_responding=False`・`response.cancel`送信のみが行われ、次tickで`_af_gate.tick()`が`cancel_held()`を呼ぶと二重の`response.cancel`送信が発生する（実害は`_BENIGN_ERROR_SUBSTRINGS`で吸収される可能性が高いが、状態機械としては不整合）。
- なぜ問題か: 現時点のコードパスでは発現しないが、hold_playbackの保持時間が最大8秒と長く、この間に人間が発話する状況は十分あり得るシナリオである。将来UIからの明示的なinterrupt機能や別経路が追加された場合に静かに壊れる。
- 修正案: `interrupt()`冒頭で`_hold_playback`ならまず`cancel_held()`相当の処理（held_audioクリア・hold解除）を行ってから通常のinterrupt処理に進むようにするか、`_af_gate`側で「holding中はagent.interrupt()を呼ばずcancel_heldのみで統一する」契約を明文化する。

**M-2. `_detect_responds_to`でロック外の dict インプレース変更**
- file:line: `src/das/asr/live/_af_runtime.py:178-200`（`with self._intervention_lock:`でコピー取得後、ロック外で`iv["embedding"] = await self._llm.embed_one(...)`）, `_af_runtime.py:254-269`（`save_snapshot`が同じ`_intervention_lock`で`self._interventions`を読む）
- 問題: `_detect_responds_to`は`_intervention_lock`保持中に`list(self._interventions)`でリストをコピーするが、これは浅いコピーであり、要素のdictそのものは`self._interventions`内のオブジェクトと共有されている。ロック解放後に`iv["embedding"] = ...`で该当dictを直接変更している。
- なぜ問題か: 現状の呼び出し関係（`ingest_utterance`→`_detect_responds_to`は`AFRuntime`専属スレッド内でのみ呼ばれ、`note_intervention`は別スレッド=agent workerから呼ばれるが新規appendのみで既存dictを触らない）では競合は起きないが、ロック規律としては「ロックが保護すべき区間」の境界があいまいで、将来`poll_interval`短縮や並列化が入ると容易に破綻する。`save_snapshot`が`_intervention_lock`を取得している最中に、別スレッドで`_detect_responds_to`のインプレース変更が走ると理論上は競合しうる（現状シングルスレッド運用なので発現しないだけ）。
- 修正案: `iv["embedding"]`の計算をロック内で行うか、embeddingの遅延計算結果を別の辞書（`{iv_id: vector}`）にロック保護して格納し、`iv`本体を直接変更しない設計に変える。

### Low

**L-1. af_l1候補のconfidenceが常に0.0で、summarizeと同priority帯(4)での競合時に構造的に不利**
- file:line: `src/das/asr/live/_workers.py:476-486`（af候補生成、`confidence`フィールドを明示的に設定していないため`InterventionCandidate`の既定値0.0のまま）, `src/das/asr/live/_facilitation.py:188,194`（`summarize`と`af_l1`はともに`priority=4`）, `_facilitation.py:243`（`eligible.sort(key=lambda pc: (pc[0], -pc[1].confidence))`で同priority時はconfidence降順）
- 問題: `af_l1`/`af_l2`候補は`InterventionCandidate(...)`生成時に`confidence`引数を渡していないため常に0.0になる。`summarize`候補も同様に`confidence`未設定（0.0）だが、他のfact等confidence付き候補と同priority帯で競合する場面では常に後者が有利。
- なぜ問題か: 現状`summarize`と`af_l1`が同時に候補化されるケースは、af_l2 pending時はsummarize自体を抑止する規則（設計88f9a78）があるため限定的だが、af_l1 pending時はsummarize候補も生成されうる（`_workers.py:456`の`not _af_l2_pending`条件はaf_l2のみ対象）。この場合confidence同点(0.0)でのタイブレークがsort実装の安定性に依存し、意図的な優先順位付けになっていない。
- 修正案: af_l1候補に`decide_intervention`が返すitemsの最大confidenceなど意味のある値を設定するか、意図的に同点として扱うなら「同点時はkindのタイブレークルールを明文化する」ことをコメントで残す。

---

## 3. 品質総評

フェーズ1〜6の設計書に沿った実装は概ね忠実で、三層分離（what/whom=AF、when=Controller、how=RealtimeAgent）の原則は`_af_gate_status`が「WHENをControllerに委譲する」薄いアダプタとして機能しており、既知修正済み事象（af_l2連発、生成先行ゲートのController駆動化）も設計通りコード上で確認できた。af_l2の再発火ガードとバックオフ（4→8→16倍化、理由タイプ別リセット）はテストも充実しており実装の意図が読み取りやすい。`apply_l1_value_gate`の緊張・新規性・鮮度の三条件も、既存の`decide_intervention`のロジックを壊さない後置フィルタとして綺麗に分離されている。

一方で、今回の統合で新設された「生成先行・再生ゲート」（hold_playback）は、RealtimeAgent単体・ゲート状態機械単体では丁寧にテストされている（`test_af_checker.py`のフェイクMagicMockベースのユニットテスト群）ものの、**実際の`_run_agent_worker`ループ全体を通した統合的な振る舞い**、特に「hold中に他の候補が通常レーンで採択を試みたらどうなるか」「会議リセットがhold中に起きたらどうなるか」という横断シナリオのテストが手薄だった。これはテスト側の`FakeAgent`（`tests/unit/live/test_agent_worker.py`）が`trigger()`のtest-and-setガードや`hold_playback`引数を再現していないため、実装のこの種の不整合をテストで検出できない構造になっていたことが根本にある。C-1とH-1/H-2はいずれも「単体では正しいが、既存の物理レーン構造・epoch管理パターンとの継ぎ目で見落とされた」という共通点があり、個別モジュールの品質そのものよりも、新旧コードの結合面のレビュー・テストが手薄だったことが主因と評価する。

対症療法ではなく構造的な修正（busy状態を通常レーンの入口でチェックする、epoch対象キューにaf_requestsを追加する）で対応可能な指摘が中心であり、恒久的なアーキテクチャ変更を要するものはない。次のコミットで着手すべき優先順位はC-1、H-1、H-2の順。
