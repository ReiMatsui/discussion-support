# コードレビュー: agents / graph / llm / runtime / presentation / types / settings

対象: `src/das/agents/`（base, extraction, document, web_search, linking, facilitation, consensus_agent, stance_agent, prompts/）、`src/das/graph/`、`src/das/llm/`、`src/das/runtime/`、`src/das/presentation/`、`src/das/types.py`、`src/das/settings.py`
観点: コード品質 + 「対面会議へのリアルタイム介入」という研究目的に対するロジックの適切さ

---

## 全体所見

**アーキテクチャは研究プロトタイプとして筋が良い。** claim/premise（議論側・立場あり）と evidence（知識側・中立）の2系統ノード + 「スタンスは対象主張ごとのエッジに置く」という設計は、Toulmin / FEVER の裏づけどおり理論的に妥当で、コード（`graph/schema.py`）・プロンプト（`prompts/linking.md` の relativity の例示）・README の三者が一貫している。EventBus + Orchestrator の pub/sub、GraphStore の Protocol 化、LLM クライアントの DI・リトライ・コスト計測、structured output の全面採用、決定的 fallback の徹底（L2 brief / L3 summary / L4 retrospective）など、テスト容易性と再現性への配慮は水準が高い。ドキュメンテーション（docstring に設計判断の理由を書く習慣）も優秀で、研究コードとしては上位の品質。

**しかし「統合議論グラフ」が介入判断でほとんど活用されていない。** FacilitationAgent が実際に見ているのは (a) 最新発話ノードの 1-hop 入エッジ、(b) セッション全体の support/attack 件数比、(c) 直近窓のノード/エッジ追加数、の3つだけである。Dung 意味論・受容可能性・攻撃チェーンの深さといった「グラフだからこそ計算できる構造」は一切使われておらず、premise/claim の区別すら facilitation 側では参照されない（`detect_bias` は `node.source` しか見ない）。現状のグラフは実質「関係ラベル付き検索インデックス」であり、それ自体はライブ介入という目的に対して合理的だが、RESEARCH.md が掲げる貢献③（AF 状態だけで判断）の説得力は、予備実験で全介入が `adjacent`（=単なる隣接エッジ提示）だった事実と併せて弱い。bias/stage 検知系のコード量（facilitation.py の約半分）に対して、実際の介入への寄与がほぼゼロという不均衡がある。

**最も深刻な構造問題は、ファシリテーション判断ロジックが2系統に分裂していること。** 研究の中核である `FacilitationAgent.decide_intervention`（AF 状態ベース）は eval シミュレーションと `das listen-soniox`（`cli/_listen.py:248`）でのみ使われ、README がメイン導線として掲げる対面ライブ UI（`python -m das.asr.live`）は別物の `FacilitationController`（`asr/live/_facilitation.py`、fact/drift/invite 等のヒューリスティック調停）で動いている。つまり「対面会議に AF ベースで介入する」という研究主張が、肝心の対面モダリティで exercise されていない。Phase 2（対面実験）で実装をそのまま使うという RESEARCH.md §4 の transferability 主張と矛盾するため、早期の統合が必要である。

**LLM 呼び出しの分業は概ね自然だが、境界の引き方で1箇所大きく損をしている。** 発話→抽出→連結のパイプラインで、抽出 LLM は「同一発話内の premise がどの claim を支えるか」を知っているのに、それをエッジとして出力せず捨てている（`EdgeCreator` に `"extraction"` が定義されているのに未使用）。その結果、自明な発話内 support 関係を embedding 検索 + 別の LLM 呼び出しで再発見するという無駄が発生している。逆に、抽出は文脈（直近発話）を一切受け取らないため、日本語会話に頻出する指示語・省略（「それは違うと思います」）を自己完結した claim に解決できない。これは対面日本語議論という目的に対する最大の精度リスクである。

**レイテンシ面は「判断は速いが、素材が遅い」。** `decide_intervention` 自体は LLM 0回の決定的計算で、ライブ向けに正しい設計。しかし L1 で提示する素材はextraction（1 call）→ embedding → linking batch judge（ノードごと 1 call）を経てからしか存在せず、既定モデルが gpt-5-mini（reasoning モデル）で `reasoning_effort` の制御も per-call timeout もないため、発話から介入可能になるまで実測で数秒〜十数秒の遅延が構造的に入る。会議の1ターンが 10〜30 秒であることを考えると成立ぎりぎりのラインで、レイテンシ予算の明示的な設計・計測が未着手なのは目的に照らして弱点である。

---

## 指摘一覧

### Critical

#### C-1. ファシリテーション判断が2系統に分裂し、研究の中核ロジックが対面ライブ経路で使われていない
- **file:line**: `src/das/agents/facilitation.py:301`（`decide_intervention`）/ `src/das/asr/live/_facilitation.py:1-223`（`FacilitationController.arbitrate`、fact/manual/drift/retry/count/invite の `_KindPolicy` テーブル）/ 利用箇所は `src/das/cli/_listen.py:248` と `src/das/eval/conditions.py:477` のみ
- **問題**: AF 状態ベースの `FacilitationAgent` は eval と `listen-soniox` 経路でしか呼ばれず、README がメイン導線とする対面ライブ UI は AF を参照しない別実装の調停器で介入を決めている。
- **なぜ問題か**: 研究の貢献③「ファシリテータは AF 状態だけで判断（モダリティ非依存）」が、対面という本命モダリティで検証されない。Phase 2 で「実装を変えずそのまま適用」という RESEARCH.md の transferability 主張が現状のコードでは成立しない。また同種の cooldown / 連続介入抑制ロジックが両系統に別々に存在し、片方だけ改修される事故が起きやすい。
- **改善案**: 「候補生成（AF から決定的に）→ 調停（cooldown・優先度ポリシー）→ 整文・配信」の3層に分離し、`FacilitationAgent` を候補生成器、`FacilitationController` を唯一の調停層として統合する。少なくとも `decide_intervention` の出力を `InterventionCandidate` としてライブ調停器に流し込む接続を先に作るべき。

#### C-2. evidence↔claim エッジの方向制約をコードで検証していない（プロンプト任せ）
- **file:line**: `src/das/agents/linking.py:594-620`（`_maybe_make_edge`）/ 方向指示は `src/das/agents/prompts/linking.md:25`（「向きは常に『事実 → 主張』(evidence が src、claim/premise が dst)」）のみ
- **問題**: LLM が `a_supports_b` / `b_supports_a` 等の5値で返す方向を、コードはそのままエッジ化する。片方が evidence の場合に「evidence が src でなければならない」という設計上の不変条件を `_maybe_make_edge` は一切チェックしない。
- **なぜ問題か**: LLM が向きを取り違えると claim→evidence エッジが生まれる。すると (1) `FacilitationAgent._select_for_target`（facilitation.py:614、`direction="in"` の入エッジのみ走査）から漏れて L1 提示されない、(2) `detect_bias` の per-node 集計（facilitation.py:176-189、`edge.dst_id` 基準）が evidence ノードを weak/over 判定対象から誤って除外/包含する、(3) RQ4 指標（異種ソース間エッジ密度）が方向誤りぶんノイズを含む。核心的貢献①の計測を静かに劣化させるバグ源。
- **改善案**: `_maybe_make_edge` で `src.node_type == "evidence" and dst.node_type == "evidence"` の禁止と、「片方が evidence なら evidence を src に正規化（逆向き判定は向きを反転して採用 or 破棄してログ）」を実装する。ユニットテストで LLM が逆向きを返したケースを固定する。

### High

#### H-1. 抽出エージェントが発話内の premise→claim 関係を捨て、連結エージェントが LLM で再発見している
- **file:line**: `src/das/agents/extraction.py:49-87`（ノードのみ返す）/ `src/das/graph/schema.py:31`（`EdgeCreator = Literal["extraction", "linking", "manual"]` — `"extraction"` はコードベースで未使用）
- **問題**: 抽出 LLM は「この premise はこの claim の根拠」という関係を発話内で自明に把握しているのに、出力スキーマ（`_ExtractionResult`）にエッジがなく捨てられる。その関係は後段で embedding top-k に乗った場合のみ、別の LLM 呼び出し（batch judge）で再判定される。
- **なぜ問題か**: (1) 1回で済む判定を2回の LLM 呼び出しに分割している（コスト・レイテンシ）。(2) top-k から漏れれば発話内の support 関係すら張られず、`avg_premises_per_claim` / `pct_unsupported_claims` という主要評価指標が retrieval 品質に汚染される。(3) スキーマに `"extraction"` を用意した設計意図と実装が乖離している。
- **改善案**: `_ExtractionResult` に `supports: list[tuple[int, int]]`（unit index ペア）を追加し、発話内エッジを `created_by="extraction"`, `confidence=1.0` で直接張る。LinkingAgent は発話間・evidence↔claim に専念させる。

#### H-2. 抽出が単発話のみを入力とし、会話文脈（指示語・省略・応答関係）を解決できない
- **file:line**: `src/das/agents/extraction.py:52-58`（user content は turn_id/話者/発話のみ）/ `src/das/agents/prompts/extraction.md`（文脈・指示語・疑問文・ASR ノイズへの言及なし、few-shot 1件のみ）
- **問題**: 日本語の対面発話は「それは違うと思います」「さっきの話ですけど」のような文脈依存表現が支配的だが、抽出は直近発話を一切受け取らない。プロンプトにも照応解決の指示がなく、few-shot は書き言葉的な自己完結発話1例だけ。疑問文・ASR 誤認識断片の扱いも未定義。
- **なぜ問題か**: 文脈なしでは「それは違う」から意味のある claim ノードを作れず、ノード文が後段の embedding 検索・linking 判定・L1 提示文の全てで劣化する。対面リアルタイムという研究目的に対して、パイプラインの最上流で最も大きな精度損失が起きる箇所。RESEARCH.md 段階 A（extraction F1 の手動アノテーション測定）が未実施のまま放置されているのもここのリスクを不可視にしている。
- **改善案**: 直近 2-4 発話を文脈として渡し、「指示語は文脈から解決して自己完結文に書き換える（ただし新情報は補完しない）」ルールと日本語話し言葉の few-shot（照応、譲歩→反論、疑問文は除外、ASR ノイズ）を 3-4 例追加。ついでに「どの過去発話への応答か」も返させれば H-1 の発話間エッジと response_rate 計測が同じ1回で取れる。

#### H-3. レイテンシ予算が未設計: reasoning モデル既定 + per-call timeout なし
- **file:line**: `src/das/settings.py:26-27`（fast/smart とも既定 `gpt-5-mini`）/ `src/das/llm/openai_client.py:99-158`（`chat`/`chat_structured` にタイムアウト・`reasoning_effort` 指定なし。temperature 互換レイヤ（40-51, 193-211）はあるが reasoning 制御はない）
- **問題**: gpt-5-mini は reasoning モデルで、`reasoning_effort` 未指定だと既定推論量ぶんのレイテンシが毎回乗る。API 呼び出し自体のタイムアウトは SDK 既定（10分）任せで、`asyncio.wait_for` 保護があるのは linking の judge だけ（linking.py:363-421）。extraction が hang すると `run_live` のパイプライン全体が詰まる。
- **なぜ問題か**: 発話→L1 提示可能までのクリティカルパス（extraction 1 call + embed + batch judge 1 call、直列）に数秒〜十数秒かかる構造で、「ライブ介入に耐えるか」という設計要件が計測もされていない。研究計画がリアルタイム前提を明言している以上、これは機能要件。
- **改善案**: (1) `OpenAIClient` に `timeout` と `reasoning_effort`（GPT-5 系なら `"minimal"`/`"low"`）を設定可能にし、抽出・連結は minimal を既定にする。(2) 発話タイムスタンプ→エッジ確定までの経過時間を構造化ログに出し、p50/p95 レイテンシを eval で常時計測する。(3) 抽出と埋め込みの並行化（現在は直列）。

#### H-4. 偏り・ステージ検知が「累積グローバル統計 + マジックナンバー乗算」で、予備実験でも機能していない
- **file:line**: `src/das/agents/facilitation.py:169-208`（`detect_bias` は全セッション累積のエッジ比）/ `facilitation.py:629-642`（priority への 1.3 / 0.7 / 1.2 / 0.85 乗算補正）/ RESEARCH.md §6（`balance_correction` / `stage_alignment` が一度も発火せず全介入 `adjacent`）
- **問題**: `imbalance_ratio` はセッション開始からの全エッジ累積比なので、議論が長くなるほど動かなくなる（linking が構造的に support 過剰を出す傾向は予備実験の 73:26 で確認済み）。それを補正する priority 乗算は根拠のない係数の重ね掛けで、`reason` ラベルの付き方も分岐順序依存（バイアス補正が付くとステージ補正のラベルは付かない、637-640行）。典型的な対症療法的条件分岐の積み重ねになっている。
- **なぜ問題か**: 「支持・攻撃の構造的バランス提示」は RQ2 の核なのに、その発火機構が実験で一度も動いておらず、動かない理由（累積統計の鈍感さ + 発火条件の AND）がアーキテクチャに埋まっている。係数 1.3/0.7 等は評価で ablation できる形になっておらず、調整の根拠も残らない。
- **改善案**: bias を直近窓（エッジの `timestamp` で窓掛け）で計算する。priority 補正は「乗算係数」ではなく「ソートキーのタプル（reason 種別の優先順位, confidence）」のような説明可能な形にする。まず `balance_correction` が発火する条件を単体テストで固定してから閾値を調整する。

#### H-5. 同義・重複 claim のマージ機構がなく、会議のグラフが構造的に濁る
- **file:line**: `src/das/graph/store/networkx_store.py:65-78`（`add_node` は id 重複のみ排除）/ `src/das/agents/linking.py`（同値関係・言い換え検出なし。5値に equivalence 相当なし）
- **問題**: 実会議では同じ主張が言い換えで何度も出るが、毎回独立ノードになる。既に embedding を全ノードで持っている（linking.py:270, `_embeddings`）のに、同一視には使っていない。
- **なぜ問題か**: 重複 claim はエッジを分散させ、`weak_claims`（攻撃≥2 かつ支持0）や `pct_unsupported_claims`、`n_isolated_claims` を系統的に歪める。L1 も「さっき提示したのと同じ内容」を別ノード経由で再提示しうる。60分・5-8名という Phase 2 の規模ではノイズが支配的になるリスクが高い。
- **改善案**: linking の候補選定時に cosine > 0.9 程度の claim 同士を `equivalent`（または既存ノードへの吸収）として扱う軽量パスを追加。少なくとも facilitation の集計側でクラスタ単位に畳む。

### Medium

#### M-1. L2 brief の LLM 整文が本体から漏れ、呼び出し側2箇所に同じ後置パッチが複製されている + dead branch
- **file:line**: `src/das/agents/facilitation.py:494-509`（decide 経路は deterministic 固定）/ `facilitation.py:484, 565`（`if self.llm is None` — `BaseAgent.__init__`（base.py:20）が常に `llm or OpenAIClient()` するため到達不能）/ 複製: `src/das/eval/conditions.py:481-495` と `src/das/cli/_listen.py:249-259`（ほぼ同一の「L2 なら compose_l2_brief で作り直して InterventionDecision を組み直す」コード）
- **問題**: `decide_intervention` を同期 API に保つ判断の副作用として、「L2 判定→LLM 整文」の合成責務が呼び出し側に漏れ、同じ8行が2箇所にコピーされている。モジュール docstring（15-16行）の「LLM が居ない場合の fallback」も、llm が None になり得ない実装と矛盾。
- **なぜ問題か**: 新しい配信経路（ライブ UI）を足すたびに整文パッチを写経する必要があり、既に2系統で微妙に違う（片方は例外を warning ログ、片方は suppress）。dead branch は「LLM なし運用」ができるという誤解を生む。
- **改善案**: `async def decide_and_render(...)` を FacilitationAgent に用意して整文込みの決定を一箇所化する（同期 decide は内部関数に降格）。`BaseAgent` は `llm: OpenAIClient | None` を素直に保持し、None 許容エージェントを型で表現する。

#### M-2. skip 判定が呼び出しケイデンス依存の内部状態を持ち、「グラフ状態のみで判断」という主張と食い違う
- **file:line**: `src/das/agents/facilitation.py:329-339`（`n_utts == self._n_utterances_at_last_decision + 1` のときだけ連続介入抑制）/ 154-157（ミュータブルな判断履歴状態）
- **問題**: 連続介入抑制は「前回判断からちょうど1発話進んだ」場合しか効かない。`_listen.py:242-248` のように周期タイマーで呼ぶ経路では、発話が2件進む・同じ状態で2回呼ばれる等で抑制がすり抜け/誤発火する。判断が (transcript, store) の純関数でなく、呼び出し履歴に依存する。
- **なぜ問題か**: RESEARCH.md 貢献③「AF 状態だけで意思決定」の実装が、実際には呼び出しタイミングというモダリティ依存の暗黙状態を持っている。replay・テスト・対面移行時の再現性を損なう。
- **改善案**: 「最後に介入した時点のエッジ数/発話数」ではなく「前回介入以降に追加されたエッジ集合が空か」をグラフから直接判定する（エッジ timestamp で計算可能）。介入履歴自体を store 外の明示的な `InterventionHistory` として引数化する。

#### M-3. 同一発話由来ノードの並行 linking によるレースと重複 embedding 呼び出し
- **file:line**: `src/das/runtime/orchestrator.py:114-118`（ノードごとに `NodeAdded` publish）+ `src/das/runtime/bus.py:40-49`（handler ごとに `create_task`）+ `src/das/agents/linking.py:435-440`（未キャッシュ分を await 中にもう一方のタスクが同じノードを embed）
- **問題**: 1発話から3ノード抽出されると3つの linking タスクが並行し、(1) `_embeddings` キャッシュ未反映の同一候補群を重複 embed（コスト増）、(2) 兄弟ノード A→B と B→A の関係判定が両タスクで独立に走り、重複・矛盾エッジ（A supports B と B attacks A の両方）が張られ得る。
- **なぜ問題か**: エッジ重複は bias 集計・L1 の dedup（facilitation.py:392-397 はテキストベース）で部分的にしか吸収されず、評価指標（エッジ数系）を直接歪める。
- **改善案**: 同一 (src,dst) ペアの既存エッジチェックを `add_edge` 前に入れる（store に `has_edge_between` を追加）。発話内の兄弟ノードは H-1 の抽出時エッジで処理し、linking の候補から互いを除外する。embedding は `_ensure_embedding` に in-flight future の共有（`dict[UUID, asyncio.Future]`）を入れる。

#### M-4. Web 検索クエリが日本語の主張文そのままで、クエリ生成ステップがない
- **file:line**: `src/das/agents/web_search.py:267`（`await self.search(node.text)`）/ lazy キューの無限成長: `web_search.py:241-249`（`_pending_queries` に上限なし、flush 時（130-144行）に古くなったノードも全件検索）
- **問題**: 「プラ容器を廃止すべき」のような主張文・当為文は検索クエリとして品質が低い（Tavily でも命題の裏取りには否定形・キーワード化が必要）。また lazy モードは stalled まで無制限にキューが溜まり、flush で一斉に古い claim を検索する。
- **なぜ問題か**: 貢献④（C4: リアルタイム検索で事前資料の限界を補う）の実効性がクエリ品質で頭打ちになる。「主張への反証を探す」目的なら、賛否両面のクエリ（例: 「紙容器 コスト 増加 事例」「紙容器 コスト 削減 事例」）を明示的に生成すべき局面。
- **改善案**: 抽出済み claim から検索クエリを生成する軽量 LLM ステップ（nano で1 call、supports/refutes の2クエリ）を挟む。lazy キューに maxlen と「最新 claim 優先」の間引きを入れる。

#### M-5. CostTracker の soft budget 超過ログが事実上出ない + 未知モデル既定料金のコメントと実装の矛盾
- **file:line**: `src/das/llm/cost.py:245`（`if self.is_over_budget() and not self._warning_emitted:` — 80% 到達時の warning（229-242行）が先に `_warning_emitted=True` にするため、この info ログはほぼ到達不能）/ `cost.py:67-69`（「不明モデルで cost が『過小』評価されないようにする」とコメントしつつ、既定は最安クラスの gpt-5-mini 料金 = 未知の高価モデルは過小評価される）
- **問題**: 2つの小さなロジック/ドキュメント矛盾。
- **なぜ問題か**: budget enforcement は eval の暴走防止装置であり、「超過したのに通知が出ない」「新モデルで実コストが推定の10倍」は実害に直結する。
- **改善案**: soft 超過ログ用に別フラグを持つ。既定料金は gpt-5（高い方）にするか、未知モデル検出時に warning を1回出す。

#### M-6. `_nodes_for_utterance` の datetime 完全一致フォールバックと全ノード線形走査
- **file:line**: `src/das/agents/facilitation.py:452-462`（毎回 `store.nodes()` 全走査、フォールバックは `n.timestamp == utt.timestamp` の等値比較）
- **問題**: 対面経路（turn_id が無い想定と docstring に明記）では datetime の完全一致が唯一のキーになるが、シリアライズ往復・マイクロ秒精度・再構築で簡単に壊れる。また `decide_intervention` 呼び出しごとに O(N) 走査が3箇所（`_nodes_for_utterance` / `detect_bias` / `_count_recent_additions`）走る。
- **なぜ問題か**: 一致に失敗すると常に「ノード化が未完了」で skip し続け、介入が黙って死ぬ（気づきにくい故障モード）。
- **改善案**: Node.metadata に `utterance_id`（UUID）を必ず入れ、Utterance 側にも同じ ID を持たせて等値キーにする。store に `nodes_by_utterance` の索引を追加。

#### M-7. linking プロンプトの few-shot 型表記と confidence 意味論の緩さ
- **file:line**: `src/das/agents/prompts/linking.md:50`（「プラ容器は年間 2 トンのゴミを出している」を `type=claim` と表記 — extraction.md:28 では同一文を premise と教えており、few-shot 間で node_type の教え方が矛盾）/ `linking.md:42`（none の confidence は「1.0 でも低い値でもどちらでも構いません」）
- **問題**: 型の使い分けを judge に混乱させる例示。confidence は閾値 0.6（settings.py:35）でエッジ採否を決める重要スカラーなのに、「関係が存在する確率」なのか「支持の強さ」なのか定義がない。
- **なぜ問題か**: 閾値フィルタの意味が判定ごとにぶれ、threshold 調整（予備実験後の再調整項目）が安定しない。
- **改善案**: few-shot の type を extraction と揃える。confidence を「その関係が成立していると判断する確率」と明示し、0.9/0.6/0.3 のアンカー例を添える。

#### M-8. `detect_bias` の weak/over 判定が docstring と異なり premise も含む + L1 の「未提示」管理が存在しない
- **file:line**: `src/das/agents/facilitation.py:183-189`（`node.source != "utterance"` のみ除外、`node_type` は見ない — 45行の docstring「発話 claim ノード」と不一致）/ `facilitation.py:404`（skip reason 文言「未提示の隣接エッジなし」だが、跨ターンの提示済み集合はどこにも保持されない。dedup（392-397行）は同一呼び出し内のみ）
- **問題**: 名称・文言と実装の乖離。同じ evidence が別々のターンで繰り返し L1 提示されることを防ぐ機構がない。
- **なぜ問題か**: 対面で同じ情報を2回通知されるのは介入の信頼を直接損なう（RQ3 の納得感に逆効果）。
- **改善案**: `presented_edge_ids: set[UUID]` を介入履歴として保持し、選定時に除外。weak/over 判定は `node_type == "claim"` に限定するか docstring を実装に合わせる。

### Low

#### L-1. dead 型: `Tick` / `AddNode` / `AddEdge` / `Mutation`
- **file:line**: `src/das/types.py:38-70`（コードベース内で定義と `__all__` 以外に参照なし。docstring は「Mutation はエージェントが返す差分」と謳うが、実際はエージェントが store に直接書き込む）
- **問題/改善**: 差分ベース設計の意図と実装が乖離。使わないなら削除、使うなら Orchestrator を Mutation 適用者に寄せる（監査ログ・undo が欲しくなる Phase 2 では有用な設計なので、方針を決めて片付けるべき)。

#### L-2. `NetworkXGraphStore` の細部
- **file:line**: `src/das/graph/store/networkx_store.py:75, 99`（`datetime.utcnow()` は deprecated、他モジュールは `datetime.now(UTC)` で不統一）/ 165-176（replay の `ORDER BY written_at` は同一秒内の順序が未定義）/ 78, 102（insert ごとに commit — ライブで発話毎に fsync）
- **改善案**: `datetime.now(UTC)` に統一、`ORDER BY rowid`、commit のバッチ化（またはWALモード）。

#### L-3. OpenAIClient のリトライ対象と API 面
- **file:line**: `src/das/llm/openai_client.py:33-37`（リトライは RateLimit/Connection/Timeout のみ — 5xx `InternalServerError` が対象外）/ 146（`beta.chat.completions.parse` — 現行 SDK では `chat.completions.parse` が正で beta は将来削除見込み）/ structured output の `message.refusal` 未処理（155-157行は parsed None を一括 RuntimeError）
- **改善案**: `APIStatusError` の 5xx をリトライ対象に追加、beta 名前空間から移行、refusal はメッセージ付きで区別。

#### L-4. `BaseAgent` の `llm or OpenAIClient()`
- **file:line**: `src/das/agents/base.py:19-21`
- **問題**: API キー未設定でも黙って実クライアントを生成する truthiness ベースの DI。基底抽象としては最小で妥当（過剰な共通化をしていない点は良い）が、`llm is None` を許すエージェント（facilitation の deterministic-only 運用）が型として表現できない。
- **改善案**: `llm if llm is not None else OpenAIClient()` にし、None 許容が必要なら基底で `llm: OpenAIClient | None` を認める。

#### L-5. `DocumentAgent.retrieve` が実質 dead（M1 stub のまま）
- **file:line**: `src/das/agents/document.py:125-140`（全 document ノードを返すだけで、パイプライン中の呼び出し元が存在しない — 連結は LinkingAgent の retrieval が担っている）
- **改善案**: 削除して DocumentAgent を「ingest 専用」と明確化するか、FlatRAG 条件と共有する retrieval 実装として活かす。

---

## グラフ設計・エージェント分業の本質性評価

**2系統ノード + 対象主張ごとエッジは、目的に対して「過剰でも過少でもなく、ただし半分しか使われていない」。** 外部知識を主張に分解せず evidence として中立に置き、スタンスをエッジ側に持たせる設計は、C1（RAG の同調バイアス）への解として本質的で、FlatRAG 条件との差分を作る最小の構造でもある。問題は活用側で: (1) premise/claim の区別が facilitation・bias 検知・L1 選定のどこでも参照されず、区別の運用コスト（抽出誤り・プロンプト複雑化）に見合うリターンが現状ない。(2) 「AF」と呼びつつ受容可能性計算も攻撃チェーンも使わず、1-hop 隣接照会に留まる — リアルタイム介入には 1-hop で十分という判断自体は妥当だが、それなら `weak_claims`（攻撃≥2・支持0）のようなアドホック規則ではなく、grounded extension（多項式時間で計算可能）による「現在守られていない主張」の方が理論とも接続し説明も一貫する。(3) 中立=エッジなしの設計は綺麗だが、「判定したが none だった」と「まだ判定していない」が区別できず、未反映外部知識率（RQ4 指標）の解釈が濁る（RetrievalQualityLog が部分的に補っているのは良い）。

**5エージェント分業は概ね自然だが、「エージェント」と呼べる自律性を持つのは linking と facilitation だけ。** extraction/document は実質ステートレスな変換関数、web_search はキャッシュ付きフェッチャで、この構成自体は健全（過剰なマルチエージェント化をしていない）。分業の境界で不適切なのは前述の1点 — 抽出と連結の間で発話内関係が落ちる（H-1）— と、抽出に文脈を渡さない（H-2）こと。逆に linking の batch judge 化（O(top_k)→O(1)）、consensus の2段ハイブリッド（構造シグナル→LLM）はコスト分業として的確で、「1回で済むものを分割していないか」という問いには全体としてよく答えている。ConsensusAgent / StanceAgent は eval 専用として base を共有する立て付けも妥当。

**facilitation.py（669行）の内訳評価**: 決定的・LLM 0回の `decide_intervention` はライブ要件に対して本質的な設計選択で、SKIP→L2→L1 の3段も研究計画の記述と正確に対応している。ただし中身の約半分（bias/stage 検知 + priority 乗算補正）は予備実験で一度も介入に寄与しておらず（全介入 `adjacent`）、累積統計・係数チューニング・呼び出しケイデンス依存の内部状態という3点で対症療法の兆候が出ている（H-4, M-2）。「いつ黙るか」に比べ「同じことを二度言わない」機構（提示履歴）が欠けているのも、対面介入の受容性という目的からは優先度の逆転。

---

## 根本的な再設計提案

1. **ファシリテーションを3層に再統合する（最優先）**: ①候補生成（AF から決定的に: 隣接エッジ、未応答攻撃、grounded extension 外の主張）→ ②調停（cooldown・優先度・提示履歴を持つ唯一の `FacilitationController`、eval とライブで共有）→ ③整文・配信（LLM 整文はここに一元化、チャネル別アダプタ）。これで C-1（2系統分裂）、M-1（整文パッチ複製）、M-2（状態依存）、M-8（再提示）が一括で解消し、貢献③がライブで実際に検証可能になる。

2. **抽出を「文脈付き・関係込み」の1呼び出しに拡張する**: 直近発話を入力に含め、units + 発話内 supports + reply-to（どの過去主張への応答か）を単一 structured output で返す。LLM 呼び出し回数は変えずに、発話内エッジ（`created_by="extraction"`）、照応解決済みノード文、response_rate の3つが同時に手に入る。linking は発話間・evidence↔claim 専任になり、判定数も減る。

3. **レイテンシを第一級の設計変数にする**: `OpenAIClient` に timeout / `reasoning_effort` を追加し、発話→エッジ確定→介入可能の各段の経過時間を構造化ログ + eval 指標（p50/p95）にする。「対面会議で使える」の定量的根拠は現状ゼロであり、Phase 2 の実験計画（60分セッション）を守れるかはここで決まる。

4. **claim の正準化（canonicalization）を linking に足す**: 既に全ノードの embedding を保持しているのだから、高類似 claim の同一視（吸収 or equivalent クラスタ）は追加コストほぼゼロで入る。会議データでのグラフ指標の信頼性はこれ無しでは確保できない。

5. **（採用可否の検討として）bias/stage 検知の簡素化**: 予備実験の証拠に基づき、動いていない補正系をいったん削って「隣接提示 + 提示履歴 + 未応答攻撃の L2」だけの最小構成に戻し、そこから ablation で足す方が、研究としても「どの機構が効いたか」を主張しやすい。
