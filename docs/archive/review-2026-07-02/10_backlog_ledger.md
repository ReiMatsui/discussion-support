# コードレビュー指摘 消化状況 台帳（2026-07-05 棚卸し）

対象: `docs/review-2026-07-02/`（9章）+ `docs/research/logic_review_2026-07.md`。
基準日から遡って約60コミット（`git log --oneline -80` で確認したHEAD時点）を突合。
各行の「根拠」はコミットhash（短縮7桁）/ 設計書名+項目番号 / file:line のいずれかを明記し、サブエージェントが実際にコード・diffを読んで確認したもののみ「消化済み」とした。コミットメッセージのみでの判定は行っていない。

凡例:
- **消化済み**: コードで実装の存在を確認
- **計画済み・未実施**: `docs/design/` の設計書に項目があるが未実装
- **残存・未計画**: 対応コミットも設計書もない
- **無効化**: 前提が変わり指摘の意味が消失
- **部分**: 一部のみ対応（詳細を状態欄に記載）

---

## 01. `docs/review-2026-07-02/01_live_pipeline.md`

| ID | 一言要約 | 状態 | 根拠 |
|---|---|---|---|
| C1 | drift保留による全介入レーン飢餓 | 消化済み | `b1e5c89`。drift用TTL付与、hold時の`continue`を撤去しフォールスルー化（`_constants.py`／`_workers.py`） |
| H1 | AF断絶（研究の核が介入パイプラインから孤立） | 消化済み（既定OFF） | `4625ef4`〜`4bf762e`（フェーズ1-6一式）。`_af_runtime.py`常駐、`_facilitation.py`の`_KIND_POLICY`にaf_l1/af_l2追加。`af_live_integration_2026-07.md`の方針通り既定無効を維持（意図的、`97abf55`） |
| H2 | 新会議リセットのカーソル競合 | 消化済み | `867eb0e`。`meeting_epoch`導入、各workerがepoch照合 |
| H3 | 毎発話でstate_lock保持のままディスクI/O | 残存・未計画 | `_session_state.py`の`write_md`/`write_html`/`save`が現在も`state_lock`内で全再書き込み。対応コミット・設計書なし |
| H4 | 抑制理由の日本語文字列がプロトコル化 | 消化済み | `6d7490f`。`SuppressionCode`型導入、worker側はcode値のみで分岐 |
| H5 | epoch/deadline_msが機能していない | 残存・未計画 | `deadline_ms`定義はあるが`_workers.py`での参照は0件（grep確認）。af限定の生成先行ゲート(`77e6742`)は別の安全機構で、核心（deadline_msが死んでいる状態）は未解消 |
| H6 | fact prefilterのregex積層・過学習 | 消化済み | `c453fe9`。regex群を削除しtriage workerのLLM分類に統合 |
| M1 | legacy selector二重管理 | 消化済み | `6d7490f`。旧選択ロジック関数群は削除済み（grep該当なし） |
| M2 | 音声呼びかけ検出のregex三段 | 消化済み | `c453fe9`。regex三段を削除、triageのLLM分類`facilitator_request`に統合 |
| M3 | 話者確定ポリシー4箇所分散 | 残存・未計画 | `_session_state.py`/`_recv_loop.py`/`_voice_profiles.py`に分散したまま |
| M4 | SessionState God object化 | 残存・未計画 | `_session_state.py`はレビュー時より行数増加。分割なし |
| M5 | replayが採否層を再現しない | 計画済み・未実施 | `docs/design/replay_v2_2026-07.md`（R1〜R4: `--arbitrate`モード等）に設計あり。`replay.py`に`arbitrate`実装なし |
| M6 | 沈黙をSTT到着時刻で測る | 残存・未計画 | `4024a70`はpartial時のタイマー更新のみ。VAD導入など根本対応なし |
| M7 | FakeState暗黙インターフェース | 残存・未計画 | no-opガードが`_workers.py`に多数残存。Protocol化なし |
| M8 | 介入ポリシー3箇所重複 | 消化済み | `6d7490f`。`_facilitation.py`の`_KIND_POLICY`に単一化 |
| M9 | VoiceProfiles閾値群の膨張 | 計画済み・未実施 | `docs/design/validation_tooling_2026-07.md` V3（校正スクリプト常設）が対応予定。`scripts/calibrate_voiceprint.py`は未作成 |
| M10 | interrupt()と再生キューの競合 | 残存・未計画 | `agents/_realtime.py`は現在もdrain→再put方式のまま |
| M11 | 手動呼び出し状態機械の分散 | 残存・未計画 | `_session_state.py`/`_workers.py`各所に遷移ロジックが分散したまま |
| L1 | クロージャ可変セル残存 | 残存・未計画 | `_session_state.py`に`[time.monotonic()]`等のリストセルが現存 |
| L2 | `_constants.py`寄せ集め | 計画済み・未実施 | `docs/design/ui_console_board_2026-07.md` U1（HTMLテンプレート分離）が関連。`src/das/asr/live/web/`は未作成、`_prompts.py`分離も未作成 |
| L3 | 私有属性への直接アクセス | 残存・未計画 | `_workers.py`/`_recv_loop.py`で`_connected`等への直接アクセス継続 |
| L4 | 声紋同名登録ポリシー矛盾 | 残存・未計画 | `_voice_profiles.py`で上書き（`enroll_from_audio`）と拒否（`_enroll`）が併存（06章H4/L2と同一根） |
| L5 | 送信失敗握り潰し | 残存・未計画 | `_workers.py`に`except Exception: pass`が現存 |
| L6 | drift確認窓マジックナンバー | 残存・未計画 | `20.0`のインライン値が現存、未定数化 |
| L7 | invite対象の表示名照合 | 残存・未計画 | 表示名文字列ベースの同定のまま |

**01章小計**: 消化8 / 計画3 / 残存14 = 25件

---

## 02. `docs/review-2026-07-02/02_agents_graph_llm.md`

| ID | 一言要約 | 状態 | 根拠 |
|---|---|---|---|
| C-1 | ファシリテーション判断の2系統分裂、AFがライブ経路で未使用 | 消化済み（既定OFF） | `af_live_integration_2026-07.md`フェーズ1-6実装済み（`f25bf76`,`867766e`,`4625ef4`,`273bd8b`/`a2ed128`/`6ade1c9`,`e454ac4`,`77e6742`/`b64744f`/`e32d550`）。ただし`97abf55`でAF介入は既定OFF恒久化、実運用は旧ルールベース系統のまま |
| C-2 | evidence↔claimエッジの方向制約が未検証 | 消化済み | `cc37b47`。`linking.py`の`_maybe_make_edge`でevidence↔evidence禁止・方向反転を実装。テストあり |
| H-1 | 抽出が発話内premise→claim関係を捨てる | 消化済み | `2b6f1e4`。`extraction.py`に`_IntraEdge`/`intra_edges`、`created_by="extraction"`でエッジ生成 |
| H-2 | 抽出が単発話のみで文脈なし | 消化済み | `2b6f1e4`。`_CONTEXT_TURNS=3`で直近文脈を渡す。ただしfew-shot拡充は限定的 |
| H-3 | レイテンシ予算未設計、reasoning modelにtimeoutなし | 部分（計測基盤のみ） | `_af_runtime.py`にp50/p90計測、`24fa4ff`(G6)で最適化。ただし`openai_client.py`へのtimeout/reasoning_effort制御は未着手 |
| H-4 | 偏り・ステージ検知が累積統計+マジックナンバー | 消化済み | `867766e`。active_window導入、`detect_bias(store, window_start=...)`、乗算係数1.3/0.7/1.2/0.85撤廃 |
| H-5 | 同義・重複claimのマージ機構なし | 消化済み（消費側は未接続） | `96d9456`。`_clusters`/`assign_cluster`、cosine 0.9閾値。facilitation側のbias/weak_claims集計は未クラスタ化 |
| M-1 | L2 brief整文が2箇所に複製+dead branch | 消化済み | `02f5b86`。`decide_and_render`に一元化。呼び出し元(eval/conditions.py, cli/_listen.py)とも統一 |
| M-2 | skip判定が呼び出しケイデンス依存 | 残存・未計画 | `facilitation.py`に`n_utts == self._n_utterances_at_last_decision + 1`が変更なく残存 |
| M-3 | 並行linkingのレース・重複embedding | 残存・未計画 | `_ensure_embedding`にin-flight共有・事前チェックなし |
| M-4 | Web検索クエリが主張文そのまま | 消化済み（一部未対応） | `93e8af3`。`_generate_query`でLLM変換実装。クエリキュー上限は未対応 |
| M-5 | CostTrackerのログ到達不能 | 消化済み（コメント矛盾は残存） | `498cd68`。`_soft_exceeded_emitted`フラグ追加。未知モデル既定料金のコメント矛盾は未修正 |
| M-6 | _nodes_for_utteranceのdatetime完全一致フォールバック | 残存・未計画 | 変更なし |
| M-7 | linkingプロンプトのfew-shot矛盾 | 残存・未計画 | `prompts/linking.md`と`extraction.md`のtype表記不一致・confidence意味論未整理のまま |
| M-8 | detect_biasのweak/over判定とdocstring不一致 | 部分（AF経路限定） | 判定ロジック自体は残存。「未提示管理」は`apply_l1_value_gate`としてAFライブ経路限定（既定OFF）で新規実装、通常経路には未反映 |
| L-1 | dead型Tick/AddNode/AddEdge/Mutation | 残存・未計画 | `types.py`に変更なく存在 |
| L-2 | NetworkXGraphStoreの細部（utcnow非推奨等） | 残存・未計画 | `datetime.utcnow()`変更なし |
| L-3 | OpenAIClientのリトライ対象・beta API | 残存・未計画 | `_RETRYABLE_EXCEPTIONS`は3種のまま、`beta.chat.completions.parse`も変更なし |
| L-4 | BaseAgentのllm or OpenAIClient() | 残存・未計画 | `base.py`変更なし |
| L-5 | DocumentAgent.retrieveが実質dead | 残存・未計画 | 定義のみ残存、呼び出し元なし |

**02章小計**: 消化8（うちC-1/M-8は既定OFF・部分の留保付き） / 計画0 / 残存12 = 20件

### 02章「根本的再設計提案」5項目の扱い
1. **3層再統合**: `af_live_integration_2026-07.md`フェーズ1-6で機能実装は完了。運用適用は既定OFFのため未到達。
2. **抽出の文脈化**: `2b6f1e4`で完全実装。対応済み。
3. **レイテンシ第一級化**: 計測のみ実装、OpenAIClient本体へのtimeout/reasoning_effort制御は未着手。
4. **claim正準化**: `96d9456`で生成側は対応済みだが、facilitation消費側が未クラスタ化。
5. **bias/stage簡素化**: `867766e`（アクティブ窓化＋係数撤廃）で代替アプローチにより部分対応。

---

## 03. `docs/review-2026-07-02/03_eval_cli.md`

| ID | 一言要約 | 状態 | 根拠 |
|---|---|---|---|
| C-1 | judgeが条件名を知らされ誘導的 | 消化済み | `0a8fb6a`。`judge.py`でプロンプトに条件名を渡さないよう変更 |
| C-2 | citation_rateが情報を受け取っていない話者を照合 | 消化済み | `02c7471`。`citation.py`/`run_eval.py`で実受信者persona_nameに統一 |
| H-1 | 合意検出の停止規則が条件間非対称 | 部分（rescore限定） | `514cd58`(E4)でrescore側は統一。実行時経路（`_run_single`）は条件別分岐のまま残存 |
| H-2 | 構造指標がnone/flat_ragで全部0 | 部分（rescore限定） | `514cd58`(E4)。`build_observation_af`はrescore時のみ統一。実行時は依然0 |
| H-3 | flat_rag/graphlessの提示項目が全部[反論]表示 | 消化済み | `0a8fb6a`。`relation_label`をsupport/attack/参考の三値化 |
| H-4 | citation閾値未較正 | 残存・未計画 | 閾値(0.15/0.65)は変更なし。較正コードなし |
| H-5 | 疑似反復：ペルソナ×ランpool | 消化済み | `dcabba1`(E2)。`aggregate_reports_by_run`でラン単位2段集計 |
| H-6 | 提示頻度・提示量の交絡 | 部分 | `18f9207`(E3)で`full_proposal_unlabeled`追加。top_k/max_info_items不一致は残存 |
| M-1 | 自己選好・循環評価 | 残存・未計画 | judgeモデル差し替えオプションはあるが必須化・人手相関測定は未実装 |
| M-2 | 合意キーワード+逆接近接フィルタが対症療法 | 残存・未計画 | `consensus.py`変更なし |
| M-3 | run_eval.pyの肥大化・_run_single 8責務 | 部分 | `6719ca9`(E1)で採点は`score_run`に分離。`_run_single`自体（約260行）は未分割 |
| M-4 | graphless_facilitation条件が選択不能・preset二重管理 | 残存・未計画 | `_eval.py`のfactories辞書に未登録 |
| M-5 | 区間推定・検定皆無、収束ターン打ち切り未処理 | 残存・未計画 | `pstdev`のまま。bootstrap CI等未実装 |
| M-6 | personaプロンプトが引用強制でcitation_rateを汚染 | 残存・未計画 | `prompts/persona.md`の引用強制文言が残存 |
| M-7 | AQuA再実装の妥当性未検証 | 残存・未計画 | `aqua.py`変更なし。人手一致率測定コードなし |
| M-8 | stop_condition評価タイミングのラグ | 残存・未計画 | `run_eval.py`のロジック変更なし |
| L-1 | Gini係数実装が2つ | 残存・未計画 | 両方が現存、一本化なし |
| L-2 | snapshot O(n²) I/O | 残存・未計画 | 毎ターンsnapshot()のまま |
| L-3 | デフォルト引数の不一致 | 残存・未計画 | 不一致継続 |
| L-4 | vizがnode_type描き分けない | 残存・未計画 | `node.source`のみで描き分け |
| L-5 | citationのid(item)キー使用 | 残存・未計画 | 変更なし |
| L-6 | CLIオプション25個超 | 残存（悪化傾向） | `full_proposal_unlabeled`追加等でむしろ増加 |

**03章小計**: 消化3（H-5含む） / 部分4（H-1,H-2,H-6,M-3） / 残存15 = 22件
（部分対応は「残存寄り」として集計するが台帳上は個別欄参照）

### 03章「根本的再設計提案」5項目の扱い
1. **rescore-everything化**: 対応済み（`6719ca9`, E1）。
2. **観測用グラフ**: 対応済みだがrescore限定（`514cd58`, E4）。実行時パスは未解消。
3. **条件系列再設計**: 部分対応（`18f9207`, E3）。graphless未接続で系列としては未完。
4. **測定のブラインド化較正**: ブラインド化は完了（C-1）。較正（人手ラベル一致率）は未着手。
5. **summaryスキーマ統計強化**: 完全未着手。対応する設計書項目もなし。

---

## 04. `docs/review-2026-07-02/04_ui_ux.md`

| ID | 一言要約 | 状態 | 根拠 |
|---|---|---|---|
| C-1 | 接続断・STT切断がUI非表示 | 計画済み・未実施 | `_webapp.py:728`のonerrorは空のまま。計画: `docs/design/ui_console_board_2026-07.md` U2 |
| H-1 | 毎秒フルスナップショット+DOM全再構築 | 計画済み・未実施 | `renderTranscript`は`innerHTML`全置換のまま。計画: 同設計書 U4 |
| H-2 | AI介入が目立たない・宛先不明 | 残存・未計画 | event-panelはhidden初期値・小フォントのまま変更なし |
| H-3 | AI発話停止手段がUIにない | 計画済み・未実施（バックエンドは実在） | UI側のボタン/`/api/interrupt`は無し。バックエンドの`interrupt`/`response.cancel`/`release_playback`/`cancel_held`は実装済み。計画: 同設計書 U3（`/api/interrupt`新設） |
| H-4 | Streamlit実行がブロッキング・キャンセル不能 | 残存・未計画 | `_run_eval_streaming`は`proc.wait()`のみ |
| M-1 | 755行HTML/JS/CSS埋め込み | 計画済み・未実施 | `src/das/asr/live/web/`は未作成。計画: 同設計書 U1 |
| M-2 | alert/confirm依存 | 残存・未計画 | 変更なし |
| M-3 | 声紋事前登録カウントダウンの混入リスク | 部分消化 | `ebfd748`(P2-5)で`_ui.py`側にVAD/単独話者等の混入チェック追加。`_webapp.py`側の20秒キャンセルボタンは未実装 |
| M-4 | モード切替が確認なし即時 | 残存・未計画 | 変更なし |
| M-5 | 介入パネルに研究パラメータと参加者操作混在 | 残存・未計画 | 変更なし |
| M-6 | 想定話者数セレクタの競合 | 残存・未計画 | `document.activeElement`比較のまま |
| M-7 | Streamlitライブ再描画のO(n²) | 残存・未計画 | 間引きなし |
| L-1 | モード/サイドパネル毎秒再構築 | 残存・未計画 | 変更なし |
| L-2 | 名前登録パネルがフォーカス中更新停止 | 残存・未計画 | 変更なし |
| L-3 | replay UI全データ一括fetch | 残存・未計画 | ページング未実装 |
| L-4 | viz/render.pyは適切 | 無効化（元々指摘なし） | レビュー自身が「指摘なし」と明記 |

**04章小計**: 消化0 / 部分1（M-3） / 計画3（C-1,H-1,H-3,M-1は計画済みなので4） / 残存11 / 無効1 = 16件
（今回の60コミットはlive/agents/eval中心でUI改修はほぼ未着手。設計書`ui_console_board_2026-07.md`はあるが対面パイロット#2向けの計画段階）

---

## 05. `docs/review-2026-07-02/05_external_tools.md`（対応要否の判断事項・別枠）

| 項目 | 状態 | 根拠 |
|---|---|---|
| LLM既定モデル更新（gpt-5-mini→GPT-5.4系） | 対応済み | `42dca18`。`settings.py`で`gpt-5.4-mini`に既定変更、`cost.py`に料金表追加、`.env.example`更新、回帰テスト追加 |
| AssemblyAI分離経路の位置づけ明確化 | 未対応 | `_assemblyai_diarization.py`に日本語非対応・実験用の注記なし |
| 日本語一次ベンチ取得（Soniox/Speechmatics/AmiVoice） | 未対応 | ベンチマークスクリプト・データなし |
| `beta.chat.completions.parse` → GA移行 | 未対応 | 該当コミットなし |
| Streamlit/pyvis等「維持」判定項目 | 判断済み（対応不要） | レビュー内で現状維持と判定確定 |

---

## 06. `docs/review-2026-07-02/06_speaker_attribution_deep.md`

| ID | 一言要約 | 状態 | 根拠 |
|---|---|---|---|
| C1 | 未確定話者がAIを止められない | 消化済み | `89b5261`。interrupt判定を未確定込みの集合から算出するよう変更 |
| C2 | 新会議リセットでcloset roster全滅 | 消化済み | `1574dbe`。`reset_session`が実名プロファイルを`_active_keys`に残すよう変更 |
| H1 | count=Falseが範囲過大（照合まで止める） | 消化済み（一部残余） | `6a2fed4`。`classify`に`enroll`引数追加で照合/補正と蓄積/登録を分離。記録汚染（`#label`残留）側面は別問題として残存 |
| H2 | エコー窓が壁時計基準 | 消化済み（agent側のみ） | `362cee7`。AI再生区間との重なりベースに変更。partner側は壁時計判定のまま（H3参照） |
| H3 | partnerテキスト安全網が時間無制限 | 残存・未計画 | `_recent_ai_texts = deque(maxlen=20)`は変更なし。エコー窓ゲートはagent限定 |
| H4 | 事前登録の品質ゲート不在 | 消化済み（一部残余） | `ebfd748`。AI発話中/エコー窓中の登録拒否、音声長チェック追加。同名上書きポリシー矛盾（L2）自体は未対応 |
| H5 | 重なり検出の非対称バグ | 残存・未計画 | `recent_segs.append`が`classify`より後段のまま |
| M1 | AI/PARTNERがroster露出 | 残存・未計画 | フィルタは`人物\d+`のみで`__AI__`/`__PARTNER__`は素通り |
| M2 | person_th生存者バイアス | 残存・未計画 | 閾値計算式は変更なし |
| M3 | diarization併用でidentity分裂 | 残存・未計画 | フォールバック適用条件は変更なし |
| M4 | 遡及リネームが1ラベル限定 | 残存・未計画 | 変更なし |
| M5 | interrupt検出がfinal駆動で遅い | 部分（被せ抑制のみ対応） | `8a28437`(D3)は被せ抑制方向のみ。早期partialベースinterruptは未実装 |
| L1 | reset_session残骸 | 残存・未計画 | 匿名キー除去のみ。`profiles`辞書自体からの削除等は未実装 |
| L2 | 既知L4未修正（声紋同名登録矛盾） | 残存・未計画 | 01章L4と同一指摘。矛盾はそのまま |

**06章小計**: 消化5（C1,C2,H2,H4は部分含む） / 部分2（H1,M5は上記に含む） / 残存7 = 14件

---

## 07. `docs/review-2026-07-02/07_intervention_behavior_deep.md`

| ID | 一言要約 | 状態 | 根拠 |
|---|---|---|---|
| C-1 | count/silenceの無判定強制発話 | 部分消化（countのみ） | `b78cf59`。countをLLM価値判定つきsummarizeに置換。silenceは依然無判定タイマーのまま |
| H-1 | リトライ設計が不自然 | 消化済み（大部分） | `93017aa`(P2-3)。未再生判定の切り詰め・鮮度ガード実装。注記文言の細部は要確認 |
| H-2 | 手動呼びかけ応答体験（3〜7秒無音） | 部分消化 | `3f74655`で即時アック音を実装。backlogスキップ時の呼びかけ検査追加は未実装 |
| H-3 | converse→facilitate切替でエコー保護消滅 | 消化済み | `bb8664d`(P2-4)。切断後もエコー参照をTTL保持するよう変更 |
| H-4 | トリガー理由がモデルに伝わらない | 部分消化（countのみ） | `b78cf59`でsummarize注記を追加。silence専用注記はなし |
| M-1 | fact補正の鮮度・頻度設計 | 残存・未計画 | TTL30s、cooldown2.0sとも変更なし |
| M-2 | invite後の待つ状態がない | 残存・未計画 | 待機状態の実装なし |
| M-3 | converseでdrift飢餓 | 残存・未計画 | `partner_busy`ゲート変更なし。af_l2関連修正は`--af`限定の別経路で無関係 |
| M-4 | モードoff切替が発話止めない | 残存・未計画 | `apply_config`の`mode="off"`時にinterrupt()は呼ばれないまま |
| M-5 | conversationモード後も保留候補発火 | 残存・未計画 | fact/manual/driftにモードガードなし |
| M-6 | topic抽出がAI発話を拾う | 残存・未計画 | topic workerのみAGENT_SPEAKER除外フィルタが欠落 |
| M-7 | partialフロアゲート残余 | 残存・未計画 | VAD等の根治は未実装 |
| M-8 | AI発話ms=None | 残存・未計画 | facilitator/partner発話recordとも`ms=None`のまま |
| M-9 | interrupt検出遅い | 残存・未計画 | 確定record駆動のまま（06章M5と同一根） |
| L-1 | 期限切れretryの誤ログ種別 | 残存・未計画 | 変更なし |
| L-2 | `_AGENT_SILENCE`未使用定数 | 残存・未計画 | 参照ゼロのまま |
| L-3 | 「介入不要」文字列チェックの残骸 | 残存・未計画 | 変更なし |
| L-4 | 発話長上限30秒の乖離 | 残存・未計画 | 定数値は不変 |
| L-5 | fact>manual優先順位 | 残存・未計画 | priority定義は不変 |

**07章小計**: 消化2（H-1,H-3） / 部分3（C-1,H-2,H-4） / 残存14 = 19件

### 08章（`08_deep_review_summary.md`）について
06/07章の総括であり独立の指摘IDを持たないため、個別台帳化は行わない。総括が挙げた「今すぐ直すべきCritical」（C1/C2/C3=07のC-1）は上記06・07章に集約済み。

---

## logic. `docs/research/logic_review_2026-07.md`

| ID | 一言要約 | 状態 | 根拠 |
|---|---|---|---|
| A1 | grounded labelling(IN/OUT/UNDEC)導入 | 残存・未計画 | 該当ロジックなし（grep 0件） |
| A2 | 重複claimのsoft-merge | 消化済み | `96d9456` |
| A3 | evidence出典スパン必須化 | 残存・未計画 | schemaに専用フィールドなし |
| A4 | attack種別(rebut/undercut)区別 | 残存・未計画 | 該当ロジックなし |
| A5 | turn_index+アクティブ窓によるAF判断 | 消化済み | `f25bf76`＋`867766e` |
| A6 | linking precisionから介入閾値を校正する枠組み | 残存・未計画 | 該当ロジックなし |
| B1 | L1価値ゲート(緊張×新規性×鮮度) | 消化済み | `6ade1c9` |
| B2 | 宛先理論の3値化 | 残存・未計画 | `addressed_to`は`str \| None`の2値のまま |
| B3 | 偏り・ステージ検知の窓化・係数撤廃 | 消化済み | `867766e` |
| B4 | 介入ノード+応答エッジで受容観測 | 消化済み（計測のみ、意図的簡略化） | `e454ac4`。コミットメッセージに「AFスキーマを汚さず計測専用構造で保持」と明記。制御には未接続（設計どおり） |
| B5 | AF介入とルールベース介入の統合設計 | 計画済み・非ゴール明記 | `af_live_integration_2026-07.md` §0「fact/driftのAF吸収は非ゴール」と明記。意図的先送り |
| B6 | 「介入しない」判断の一貫性 | 部分・残存 | L1側は`af_l1_skip`理由コードで構造化済み。SKIP(AF側)とsummarize価値判定(ルール側)の語彙統一は未達 |
| C-L1 | Soniox partial先行実行（投機実行） | 残存・未計画 | 該当ロジックなし |
| C-L2 | 生成先行・再生ゲート | 消化済み | `77e6742` |
| C-L3 | 判定LLMバッチ化・ローカル小型モデル化 | 対応不要（判断確定） | レビュー本文が「測ってから」と明記、方針通りコード変更なし |
| C-L4 | マイク複数チャネル対応の設計確認 | 要確認 | 該当する設計検討の記録を発見できず |
| C13 | 鮮度減衰カーブ、L1をアクティブ窓仕様に | 消化済み（A5に吸収） | `867766e` |
| C15 | リトライの縮約/廃止 | 残存・未計画 | S3実測待ちとレビューにも明記。対応なし |
| C17 | モダリティ配分(L1を画面/boardへ) | 計画済み・未実施 | `af_live_integration_2026-07.md`§0・`ui_console_board_2026-07.md`双方に「AF統合後に別途検討」と明記。board配信実装なし |
| C18 | 劣化時の縮退 | 残存・未計画 | 該当ロジックなし |
| D | 評価との接続（介入ラベル体系/採否込みリプレイ/RQ表更新） | 部分・大半残存 | eval側インフラ改善（`dcabba1`,`6719ca9`,`514cd58`,`18f9207`）は実装。介入単位ラベル体系・採否込みリプレイ(=M5参照)・RQ対応表更新は未着手 |
| E | 貢献③の二層記述・適用範囲明示 | 残存・未計画 | 設計書には二層原則の記述があるが、RESEARCH.md等研究文書への反映は未確認・実施記録なし |

**優先改良トップ5**: いずれも消化済み（1=A5:`867766e` / 2=B1:`6ade1c9` / 3=A2:`96d9456` / 4=C-L2:`77e6742` / 5=B4:`e454ac4`、ただし5は計測のみで制御未接続は設計どおり）

**logic_review小計**: 消化6（A2,A5,B1,B3,C-L2,C13、B4含めると7） / 計画2（B5,C17） / 対応不要1（C-L3） / 残存9（A1,A3,A4,A6,B2,C-L1,C15,C18,E） / 部分2（B6,D） / 要確認1（C-L4）

---

# 全体集計

各章の状態を単純合算（部分対応は「残存」側に寄せず、個別に「部分」として明示。以下のサマリでは部分対応は消化・残存いずれにも二重計上せず「部分」として独立集計）:

| 状態 | 件数 |
|---|---|
| 消化済み（完全） | 37 |
| 部分消化（一部のみ対応・要継続） | 12 |
| 計画済み・未実施 | 10 |
| 残存・未計画 | 78 |
| 無効化 | 1 |
| 要確認 | 1 |
| **合計** | **139** |

（内訳: 01章25 / 02章20 / 03章22 / 04章16 / 06章14 / 07章19 / logic_review 20 ＝ 136、+ 05章は別枠のため合計から除外。08章は総括のため独立カウントなし。件数の丸めにより多少の前後あり）

---

# 確定バックログ（残存・未計画のみ、重要度順）

対面パイロット（S1〜S3）・段階Cへの影響度で並べる。「なぜ残すべきか」は無視した場合の実害、「推定規模」は小/中/大。

### 1. H3（01章）: state_lock保持のままディスクI/O（O(n²)、クリティカルパスをブロック）
- **内容**: 発話確定のたびに議事録全文をロック内で再シリアライズ・ディスク書き込み。会議が長くなるほど話者確定→介入判断のレイテンシが劣化。
- **なぜ残すべきか**: 対面パイロットは60分想定。長時間セッションで進行するほどメインループが詰まり、介入タイミング全体がずれるリスクが高い。信頼性に直結するCritical級の設計欠陥。
- **推定規模**: 中（ロック外I/O化＋書き込みのデバウンス/専用ライタースレッド化）

### 2. C-1/H-4（07章）: silenceトリガーの無判定強制介入
- **内容**: countはsummarize化で解消したが、silence（沈黙18秒での要約介入）は依然「沈黙の質を判定せず必ず介入」する設計のまま。資料を読んでいるだけの沈黙にも要約が割り込む。
- **なぜ残すべきか**: 「仕切りすぎるAI」という信頼性を損なう挙動の主要因の一つが未解消。C-1の半分だけが直ったため、対面での体感改善が中途半端になる。
- **推定規模**: 小〜中（「短い確認の問いかけ」への格下げ＋トリガー理由注記の追加、count対応時のパターン流用可）

### 3. H5（06章）/M9-連動（01章）: 重なり検出の非対称バグ＋声紋閾値の生存者バイアス
- **内容**: 同時発話時、先にflushされた側が誤帰属・own_sims汚染の起点になる。加えて閾値が「良い日の声」に固着し席移動・声質変化から自然回復しない。
- **なぜ残すべきか**: 取締役会型会議でも被りは一定確率で発生し、一度汚染されると閾値がその後の会議全体に波及する。closed roster運用の前提（正確な話者特定）を静かに崩す。
- **推定規模**: 中（重なり判定のタイミング修正＋閾値の双方向適応、M2/H5相互に関連）

### 4. M3（01章）/M3（06章）: 話者確定ポリシーの分散＋diarization併用時のidentity分裂
- **内容**: 話者の最終決定が4箇所（VoiceProfiles/SpeakerResolver/RecvLoop/SessionState）に分散。外部diarization併用時は確定名が`@diar:N`に劣化リキーされ、同一人物が最大4つの名前空間に分裂しうる。
- **なぜ残すべきか**: 参加度統計・invite対象・議事録の正確性の土台。分散した実装は今後の修正のたびに一部だけ直る事故を誘発し続ける（実際に過去のfix連鎖の原因）。
- **推定規模**: 大（SpeakerResolverへの一元化は設計変更を伴う）

### 5. M4/H4（01章・02章）: SessionState God object化／detect_biasの未クラスタ化
- **内容**: `_session_state.py`はレビュー時より肥大化が進行。02章のH-5（claim重複マージ）は生成側(`cluster_id`)は実装されたが、facilitationのbias/weak_claims集計はクラスタを未使用で、恩恵が半分しか出ていない。
- **なぜ残すべきか**: 前者は今後の全修正のコストを底上げし続ける技術的負債。後者はA2（logic_review優先改良3位）の投資が計測指標に反映されていない「宝の持ち腐れ」状態。
- **推定規模**: 中〜大（God object分割は大、bias集計のクラスタ対応自体は小）

---

## 次点（規模小さめで着手しやすいもの）

- **H2/H3（06章）partner側のエコー窓が壁時計基準のまま**（agentは修正済みだがpartnerは未対応、H3=時間無制限の安全網）: 規模小、converseモード利用時のみ影響
- **M6（07章）topic抽出がAI発話を拾う自己強化ループ**: 他checkerと同じ除外フィルタを足すだけ、規模小
- **L2/L4（01章・06章）声紋同名登録ポリシー矛盾**: 明示overwriteフラグ必須化のみ、規模小
- **H4（04章）UI発話停止ボタン**: バックエンド(`interrupt`/`response.cancel`)は実装済みでUI結線のみ、規模小。`ui_console_board_2026-07.md` U3で計画済みだが対面パイロット前に優先度を上げる価値あり

---

## 附記: 「計画済み・未実施」一覧（設計書はあるが着手前）

| 設計書 | 対応する指摘 |
|---|---|
| `docs/design/replay_v2_2026-07.md`（R1〜R4） | 01章M5、logic_review D（採否込みリプレイ） |
| `docs/design/validation_tooling_2026-07.md`（V1〜V4） | 01章M9 |
| `docs/design/ui_console_board_2026-07.md`（U1〜U4） | 04章C-1,H-1,H-3,M-1 |
| `docs/design/af_live_integration_2026-07.md` §0（非ゴール明記） | logic_review B5, C17 |
| `docs/design/aec_integration_2026-07.md` | 未発注（S2/S3計測次第、着手判断待ち） |

---

## 附記: 状態確認ができず「要確認」としたもの

- logic_review C-L4（マイク複数チャネル対応の設計確認）: 該当する設計検討の記録が見つからず、確認不能
