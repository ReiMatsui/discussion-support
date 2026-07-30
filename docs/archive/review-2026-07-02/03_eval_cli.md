# コードレビュー: 評価系 (src/das/eval/)・CLI (src/das/cli/)・可視化 (src/das/viz/)

対象コミット時点のファイル実体を直接確認した上でのレビュー。行番号はすべて実ファイルで確認済み。

- 対象: `src/das/eval/` (run_eval.py 906行, conditions.py, aqua.py, citation.py, structural_metrics.py, consensus.py, controller.py, judge.py, metrics.py, persona.py, presets.py, prompts/)、`src/das/cli/` (_eval.py, _listen.py, _session.py, __init__.py)、`src/das/viz/render.py`
- 参照した研究文書: `README.md`, `docs/research/RESEARCH.md`, `docs/design/comparison_execution_plan.md`

---

## 全体所見

**工学的な完成度は研究プロトタイプとして高い水準にある。** インクリメンタル保存（途中クラッシュでも部分結果が残る）、soft/hard の二段予算ゲート、条件を横断する stratified round-robin スケジューリング（予算切れ時にも全条件のデータが残る設計、run_eval.py:815-821）、cost snapshot の保存など、「LLM API を使った多数回実験を安全に回す」ための配慮が随所にあり、これは多くの研究コードに欠けている美点である。指標も主観 5 軸・構造 8 指標・citation・stance polling・AQuA・介入の負の影響（発話萎縮）と多層的で、RQ との対応付けが文書化されている点も良い。

**しかし、研究の中核的な問い「AI 介入は会議を良くするか」に answer する測定系として見ると、結論を無効化しうる欠陥が複数ある。** 最も重大なのは (1) LLM judge が評価対象の条件名を知らされており（unblinded）、さらに judge プロンプトが「提案手法なら透明性を高く付けよ」と事実上指示していること、(2) RQ4 の主要 evidence である citation_rate が、full_proposal 条件では「情報を実際に受け取っていない話者」の発話を照合対象にしており、flat_rag 条件（受け取った話者を照合）と測っているものが異なること、の 2 点である。前者は RQ3（透明性）の条件差を、後者は RQ4（外部知識活用）の条件差を、それぞれ実験結果ではなく実装の作り込みで決めてしまう。現状の summary.json の条件間比較は、この 2 点を直すまで論文のエビデンスとして使えない。

**比較条件の設計にも系統的な交絡がある。** flat_rag は毎ターン必ず 3 チャンク提示するのに対し、full_proposal は skip ロジック付きで最大 2 件。つまり「関係ラベルの有無」（本来動かしたい独立変数）と「提示頻度・提示量」（交絡変数）が同時に動いている。また合意検出は full_proposal だけがグラフ由来の構造シグナルを使えるため、`--until-consensus` 時の停止規則そのものが条件間で異なり、収束率・ターン数はもちろん、ターン数に依存する全下流指標（judge スコア、AQuA、構造指標）に条件差が波及する。構造指標（response_rate 等）に至っては、グラフを構築しない none / flat_rag では常に 0 が記録されるため、条件比較が原理的に成立していない。

**コード品質面では、run_eval.py の肥大化が最大の懸念。** `_run_single` (run_eval.py:385-682) は「セッション実行・イベント通知・インクリメンタル保存・stance 測定・合意検出・judge・構造指標・citation 計算」の 8 責務を 1 関数に抱えており、約 300 行ある。また CLI (`_eval.py`) にはトピック preset 定義が `eval/presets.py` と二重管理されており、`ConditionGraphlessFacilitation`（ablation 条件）は実装されているのに CLI から選択できないなど、実験系としての一貫性に穴がある。統計処理は「ペルソナ×ラン を独立サンプルとして pool する」疑似反復があり、n を過大申告している。

総じて「実験を回すインフラ」は A 評価、「実験から結論を引き出す測定設計」は現状 C 評価。以下の Critical / High を潰せば、Phase 1（政策トピック n=5〜10 の本ラン）を回す価値のある測定系になる。逆に言えば、**本ランを回す前に必ず直すべき**項目が Critical / High に集中している。

---

## 指摘一覧

### Critical

#### C-1. LLM judge が条件名を知らされており（unblinded）、プロンプトが期待する結果を誘導している

- **file:line**: `src/das/eval/judge.py:130-134`（user メッセージに `## 条件\n{condition_name}` を明示的に埋め込み）、`src/das/eval/prompts/judge.md:24`
- **問題**: judge への入力に評価対象の条件名（`none` / `flat_rag` / `full_proposal`）がそのまま渡される。さらに judge.md の intervention_transparency の定義文が「**情報提供がない条件では 1 (低い)、提案手法のように構造化された通知があるなら高めに**」と、条件と期待スコアの対応を直接指示している。加えて `info_log=None` のとき `"(情報提供なし条件)"` という文字列も渡される（judge.py:126-129）。
- **なぜ問題か**: これは人間実験でいう「実験者が被験者に仮説を教えてから回答させる」状態（demand characteristics）。RESEARCH.md §6 で「介入透明性は提案手法が最高 ✓ (RQ3 を支持)」と予備実験結果を解釈しているが、この結果はプロンプトの指示をそのまま反映しただけの可能性が高く、RQ3 のエビデンスとして無効。LLM judge は指示への迎合（sycophancy）が強いことが知られており、他の 4 軸（満足度等）にも条件名の開示がハロー効果として波及しうる。
- **改善案**:
  1. `_build_messages` から `condition_name` を除去する（`JudgeReport` のメタデータとして保持するのは問題ない。judge の入力に入れないことが本質）。
  2. judge.md:24 の条件別スコア指示を削除し、「提示情報の出所・理由がその場で理解できたか」という条件非依存の操作的定義に書き換える。none 条件では「該当なし」を許す別スケール（または N/A）にする。
  3. 可能なら判定を条件ブラインドの pairwise 比較（2 transcript を A/B 提示、提示順をランダム反転して位置バイアス相殺）に変え、絶対 Likert と併用する。

#### C-2. citation_rate (RQ4 主要指標) が full_proposal では「情報を受け取っていない話者」を照合しており、条件間で測定対象が異なる

- **file:line**: `src/das/eval/citation.py:193-199`（L1 の target を `addressed_to` 話者の次発話に取る）、`src/das/agents/facilitation.py:409-417`（`addressed_to=last_utt.speaker` = **直前の発話者**）、`src/das/eval/conditions.py:453-521`（info は **次話者** のプロンプトにのみ注入され、`addressed_to != persona.name` のため常に三人称整形になる）
- **問題**: シミュレーションは round-robin なので、`decide_intervention` が返す `addressed_to`（直前の発話者）と、実際に参考情報を見る persona（次話者）は**必ず別人**になる。citation.py の L1 ロジックは `addressed_to` 話者が次に発言するターン（N-1 ターン後）を照合対象にするが、その話者は当該情報を一度もプロンプトで受け取っていない。一方 flat_rag / graphless はログの `addressed_to` に受信者（次話者）を記録している（conditions.py:206, 317, 330）ため、「受け取った直後の発話」を正しく照合する。
- **なぜ問題か**: RQ4 の直接 evidence と位置付けた指標が、full_proposal では「提示→引用」の因果連鎖を測っておらず、「情報が届いていない人が偶然似たことを言った率」を測っている。条件間比較（関係ラベルあり vs なし）が同じ量の比較になっていないため、citation_rate の条件差から RQ4 に関する結論は一切引き出せない。方向としては full_proposal に不利に働く可能性が高く、「提案手法の効果が出ない」という偽陰性を作り込む。
- **改善案**: `InterventionLogEntry` に「実際に情報が注入された話者」（= `info_provider` の `persona.name`、現在 `persona_name` フィールドに入っている）と「グラフ上の宛先」（`addressed_to`）を明確に区別し、citation の照合は前者の直後の発話（= trigger_turn+1 のターン）に統一する。conditions.py:499-510 では既に `persona_name=persona.name` を記録しているので、citation.py 側を `addressed_to` ではなく `persona_name` ベースに変えるだけで直る。

### High

#### H-1. 合意検出の停止規則が条件間で非対称（構造シグナルは full_proposal のみ）

- **file:line**: `src/das/eval/run_eval.py:461-503`（`store_ref = _store_for_condition(condition)` は full_proposal 以外 None）、`src/das/eval/consensus.py:204-230`（store があるときだけ `new_claim_stalled` / `no_new_attacks` が発火し、「構造シグナル 2 つ」で explicit キーワードなしでも合意成立）
- **問題**: `--until-consensus` 時、none / flat_rag はキーワードベースのみで停止判定されるが、full_proposal はキーワード + 構造シグナル（+ LLM 判定への前段トリガーも構造シグナルで多く開く）で判定される。つまり**実験の停止規則そのものが処置条件と共変**している。
- **なぜ問題か**: 収束率・平均ターン数・合意到達ターン（CLI が表示する主要アウトカム、_eval.py:624-642）の条件差が「介入の効果」なのか「検出器の感度差」なのか識別不能。さらに早期終了はターン数を変えるので、judge スコア・AQuA・structural・citation のすべてに間接的に波及する（短い議論は満足度も構造も変わる）。
- **改善案**: (a) 停止判定は全条件で同一の検出器（transcript のみを入力とする LLM 判定 or キーワード）に統一する。(b) 構造シグナルをどうしても使うなら、none / flat_rag の transcript にも同じ extraction+linking をバックグラウンドで走らせて「観測専用グラフ」を作る（処置には使わず検出のみに使う）。(c) 少なくとも Phase 1 本ランは `--until-consensus` を切り、固定ターンで回して time-to-consensus は事後解析にする。

#### H-2. 構造指標 (DQI 風) が none / flat_rag では全て 0 になり、条件比較が成立していない

- **file:line**: `src/das/eval/run_eval.py:624`（`compute_structural_metrics(transcript, final_store)`、final_store は full_proposal 以外 None）、`src/das/eval/structural_metrics.py:168-169`（store None なら gini 以外 0 のまま early return）
- **問題**: `response_rate`, `avg_premises_per_claim`, `pct_attacks_answered` 等のグラフ由来指標は、グラフを構築する full_proposal でしか計算されず、他条件では 0 が `summary.json` の `structural` 集計に入る。RESEARCH.md §4 はこれらを「3 条件比較の客観指標」と明記しているが、実装上は比較不能。
- **なぜ問題か**: summary の表を条件間で並べると「none は response_rate 0.0」のような無意味な比較が生まれ、誤読・誤引用のリスクが高い。また「LLM 不要で対面にも同じ意味で使える」という設計意図（structural_metrics.py:1-20）にも反する（対面では全条件で録音から AF を作る前提のはず）。
- **改善案**: 評価フェーズで、**全条件の transcript に対して同一の extraction+linking パイプラインを post-hoc に適用**して観測用 AF を構築し、構造指標はそこから計算する（処置用グラフと観測用グラフの分離）。これは H-1(b) と同じ仕組みで実現できる。コストが問題なら `das eval` 本体から切り離した `das structural-rescore` サブコマンド（aqua-rescore と同型）にする。当面は none / flat_rag の structural を 0 でなく null にして集計から除外するだけでも誤読は防げる。

#### H-3. flat_rag / graphless の提示項目が judge には全部「[反論]」と表示される

- **file:line**: `src/das/eval/judge.py:88`（`tag = "[支持]" if item.get("relation") == "support" else "[反論]"`）、`src/das/eval/conditions.py:200`（flat_rag は `relation: ""`）、`conditions.py:328`（graphless も `relation: ""`）
- **問題**: relation が空文字（= ラベルなし提示）のアイテムが、judge への info ログ整形時に一律「[反論]」とラベル付けされる。
- **なぜ問題か**: flat_rag は「関係ラベル**なし**の提示」を表す統制条件なのに、judge の目には「全部反論ラベル付きで提示された」ように見える。条件の操作的定義を評価段階で破壊しており、information_usefulness / opposition_understanding / transparency の条件差に未知の方向のバイアスを注入する。1 行の条件分岐ミスだが、実験の妥当性への影響は大きい。
- **改善案**: `relation` が support/attack 以外のときは「[参考]」等の中立タグにする。`_format_l1_self` / `_format_l1_third_person`（conditions.py:523-538）も同じ三値分岐（support/attack/その他）に統一する。

#### H-4. citation 判定の閾値が未較正で、系統的な長さバイアスと天井効果を持つ

- **file:line**: `src/das/eval/citation.py:63-82`（n-gram coverage 0.15 / embedding cosine 0.65 の OR 判定）、`src/das/eval/run_eval.py:659-666`（常に embedding 併用パスを既定使用）
- **問題**: (1) n-gram coverage は「source の n-gram のうち target に現れる割合」なので、flat_rag の長い文書段落（数百字）は 2〜4 文の発話に 15% 含まれることがほぼなく、full_proposal の短い claim 引用（数十字）は容易に超える。**source の長さが判定確率を直接支配**する。(2) embedding 類似度 0.65 は、同一トピックの日本語文同士なら引用がなくても超えやすい水準で、天井効果（全条件で citation 率が飽和）を起こしうる。どちらの閾値も人手アノテーションとの照合による較正が行われていない。
- **なぜ問題か**: RQ4 の主要指標が「引用の有無」ではなく「提示テキストの長さ分布の条件差」や「トピック一致度」を測ってしまう。C-2 と併せ、citation_rate は現状二重に壊れている。
- **改善案**: (a) 少数の transcript で「引用あり/なし」を人手ラベルし、ROC で閾値を較正する（comparison_execution_plan の Week 1 保険と同思想）。(b) 長さバイアスには source 側を文単位に分割して max coverage を取る、または target 側基準の coverage を併用する。(c) embedding 判定には null 分布（提示されていないチャンクとの類似度）を対照にした相対判定（z-score / rank）を使う。(d) 感度分析として閾値を振った結果を supplementary に出す。

#### H-5. 疑似反復: ペルソナ×ランを独立サンプルとして pool して集計している

- **file:line**: `src/das/eval/run_eval.py:129-136`（`EvalResult.aggregate` が全ラン全ペルソナの report を flat に結合）、`src/das/eval/judge.py:212-252`（`aggregate_reports` が flat な mean/std）、`run_eval.py:333`（summary の `n_judge_reports` がそのまま n として出る）
- **問題**: 同一ラン内の 3〜4 ペルソナのスコアは同じ transcript を見た評価であり独立でない（クラスタ構造）。それを pool して n=ラン数×ペルソナ数 として mean±std を出している。stance 集計（run_eval.py:274-321）も同様に persona-phase を pool し、`mean_public_shift` を paired difference ではなく「post 平均 − pre 平均」で計算している（pre/post の欠損があると系統的にずれる）。
- **なぜ問題か**: 実効サンプルサイズの過大評価。n=5 ラン × 4 ペルソナを n=20 と扱えば、条件差の std が実際より小さく見え、偽陽性の結論（「提案手法が有意に良い」）を導きやすい。ペルソナには立場（pro/con/neutral）という強い系統因子もあり、予備実験でも「B (con) だけ full_proposal で低下」という交互作用が観測済み（RESEARCH.md §6）なのに、集計はそれを平均で潰している。
- **改善案**: ラン単位で「ペルソナ平均」を先に取り、ラン間で集計する（クラスタ平均）。加えて stance 別（pro/con/neutral）の層別集計を summary に常設する。将来的には run を単位とした mixed-effects（persona をランダム効果）が正攻法。stance shift は persona ごとの paired diff の平均に修正する。

#### H-6. 比較条件間で「提示頻度・提示量」が交絡している

- **file:line**: `src/das/eval/conditions.py:129`（flat_rag `top_k=3`、毎ターン必ず提示）、`conditions.py:353`（full_proposal `max_info_items=2`）、`src/das/agents/facilitation.py:312-419`（full_proposal は skip 判定付きで提示しないターンがある）
- **問題**: 条件間で動いているのは「関係ラベルの有無・グラフの有無」だけではなく、(1) 1 回あたりの提示件数（3 vs ≤2）、(2) 提示タイミング（毎ターン vs トリガー時のみ）、(3) 提示ソース構成（文書のみ vs 発話+文書混合）が同時に動く。
- **なぜ問題か**: RQ1/RQ4 で条件差が出ても「関係ラベルの効果」なのか「提示量・頻度の効果」なのか識別できない。comparison_execution_plan.md の Phase 2 は presentation 次元を 1 つずつ動かす計画であり方向は正しいが、Phase 1 の主比較自体が既に多重交絡している。
- **改善案**: 最低限、flat_rag の `top_k` を full_proposal の `max_info_items` と揃える（2 に）。理想は「dose を揃えた条件系列」: (i) none、(ii) flat_rag（提示タイミングを full_proposal の介入ログに yoke して同頻度化）、(iii) full_proposal-無ラベル（同じ item から relation タグだけ除去）、(iv) full_proposal。(iii)–(iv) 比較が RQ4（関係ラベルの寄与）の最もクリーンな検定になる。

### Medium

#### M-1. persona・facilitator・judge・AQuA・consensus がすべて同一 LLM ファミリで閉じている（自己選好・循環評価）

- **file:line**: `src/das/cli/_eval.py:492-528`（単一の `OpenAIClient` を persona / judge / consensus / stance で共有）、`src/das/eval/judge.py:157`（judge モデル既定 = `smart_model`）
- **問題**: 発話を生成するモデルと採点するモデルが同一。LLM-as-judge の自己選好バイアス（自分の文体・自分に与えられた指示に沿う出力を高く評価する）への対策（別プロバイダ judge、複数 judge の合議、人手サンプル検証）がコード上どこにもない。RESEARCH.md §7 で ChatEval や Mirzakhmedova の κ に言及しているのに実装されていない。
- **改善案**: judge / AQuA だけ別系統モデル（例: 別プロバイダ）を指定できる引数は既にある（`JudgeAgent(model=...)`, `aqua-rescore --model`）ので、実験プロトコルとして「judge は persona と別モデル」を必須化し meta.json に記録する。少数 transcript の人手採点との相関（κ or Spearman）を段階 A に追加する。

#### M-2. 合意キーワード + 逆接近接フィルタは対症療法であり、貧弱な前段が LLM 判定の呼び出し分布を歪める

- **file:line**: `src/das/eval/consensus.py:31-93`（ハードコードされた日本語キーワード 11 語 + 逆接 13 語 + 30 文字近接ヒューリスティック）、`consensus.py:280-281`（前段シグナルが立たない限り LLM 判定は呼ばれない）
- **問題**: 「確かに〜が、」誤検出への 30 文字近接ルールは典型的な場当たり補正で、少し言い回しが変わる（「一理あるとは思いつつも…」等）とすり抜ける/過剰に弾く。さらに二段構成のため、前段キーワードの再現率が低い条件では LLM 判定自体が呼ばれず、**検出漏れが前段の癖に支配される**。トピックや言語を変えると挙動が変わり、条件間・トピック間比較の再現性を損なう。
- **改善案**: 前段は「安価な LLM 分類器（nano モデル、数トークン出力）を k ターンごとに定期実行」に置き換えるのが最も素直（コストはキーワードとほぼ同水準に抑えられる）。キーワード方式を残すなら、較正データ（人手で合意ターンをラベルした transcript）に対する再現率を測って閾値を決め、その較正結果を残す。

#### M-3. run_eval.py の肥大化: `_run_single` が 8 責務・約 300 行

- **file:line**: `src/das/eval/run_eval.py:385-682`（セッション実行 / イベント emit / incremental save / stance pre-post / stop_condition クロージャ / consensus / judge / structural / citation を 1 関数で実施）、`run_eval.py:459-503`（クロージャ内に mutable dict でステートを持つ stop_condition 組み立て）
- **問題**: 新しい指標（AQuA は既に別 CLI、stance は引数追加…）を足すたびに `_run_single` と `SingleRunResult` と `_save_run` と `_save_eval_result` の 4 箇所を同時に触る構造。stop_condition の `last_llm_check_at` / `last_report` のような dict-as-cell ステートは読みにくくテストしにくい。
- **改善案**: (1) 「セッション実行（transcript+ログ生成）」と「スコアリング（judge/structural/citation/consensus/stance）」を分離し、スコアラを `Scorer` プロトコル（`score(run_artifacts) -> dict`）の登録リストにする。aqua-rescore が既に post-hoc 型なので、全指標を post-hoc rescore 型に寄せると再採点・指標追加が自由になる。(2) stop_condition は小さなクラス（`ConsensusStopper`）に切り出す。(3) 保存系（`_save_run` 等）は `eval/io.py` へ。

#### M-4. CLI: `graphless_facilitation` 条件が選択不能・preset 定義の二重管理

- **file:line**: `src/das/cli/_eval.py:494-512`（factories は none / flat_rag / full_proposal のみ、他は Exit(1)）、`_eval.py:453-464`（preset→persona/topic/docs のマップを CLI 内にハードコード。`src/das/eval/presets.py` と二重管理）
- **問題**: ablation 用に実装した `ConditionGraphlessFacilitation`（conditions.py:218-334）が CLI から走らせられない。実験で使うには ad-hoc スクリプトが必要になり、「グラフの寄与の切り分け」という ablation の目的が実行系に接続されていない。preset のトピック文字列・docs サブディレクトリが CLI に埋まっており、presets.py（ペルソナのみ）と分裂している。
- **改善案**: 条件レジストリ（`dict[str, ConditionFactory]`）を `eval/conditions.py` 側に置いて CLI はそれを引く。preset は `presets.py` に `TopicPreset(name, topic, personas, docs_subdir)` として統合。実験設定は最終的に 1 つの TOML/JSON（meta.json と同型）で宣言できると再現性が上がる。

#### M-5. 集計に区間推定・検定が皆無で、収束ターンの打ち切りも未処理

- **file:line**: `src/das/eval/run_eval.py:324-371`（summary は mean/std のみ）、`run_eval.py:235-271`（`mean_turns_to_consensus` は収束ランのみの平均 = 生存バイアス）、`src/das/eval/judge.py:232`(pstdev=母標準偏差)
- **問題**: n=5〜10 のラン数で条件差を主張するには CI か検定が要るが、summary.json には mean/std しかない。time-to-consensus は「収束しなかったラン」（右打ち切り）を単に除外して平均しており、収束率の異なる条件間で比較すると系統的に歪む（収束しにくい条件ほど「早く収束した少数ラン」だけが平均に入る）。std も母標準偏差（pstdev）で、小 n では過小。
- **改善案**: ラン単位（H-5 修正後）の bootstrap CI を summary に追加。time-to-consensus は Kaplan-Meier / 打ち切り明記（RESEARCH.md 参照文献の Adaptive Stability Detection と同じ土俵）。pstdev → stdev（標本）に統一。

#### M-6. persona プロンプトが「提示情報を必ず引用または反論せよ」と命じており、citation_rate が処置効果ではなく指示遵守を測る

- **file:line**: `src/das/eval/prompts/persona.md:44-48`（「それを**そのまま無視せず**、自分の発言の根拠として明示的に引用するか、または反論してください」）
- **問題**: 提示情報への反応がプロンプトで強制されているため、citation_rate の絶対値は「LLM の指示遵守率」に近い。条件間の相対比較は同一プロンプトなので一応成立するが、人間への transferability（RESEARCH.md §4 の売り）は崩れる — 人間は引用を命じられていない。また「関係ラベルがあると引用したくなるか」という RQ4 の心理的メカニズムを、指示が上書きしてしまう。
- **改善案**: 引用強制を外した（または「無視してよい」と明示した）persona プロンプトでの感度分析を 1 回行い、citation_rate の条件差が指示に依存していないことを確認する。本ランでは引用指示を弱い表現（「役立つと思えば使ってよい」）に緩める。

#### M-7. AQuA 再実装の妥当性検証（原論文アダプタ/人手との照合）が未実施のまま条件比較に使われる

- **file:line**: `src/das/eval/aqua.py:1-25`（設計メモに「完全互換ではない」と明記はある）、`aqua.py:59-190`（重みは原論文 Table 1 転記、0-3 スケール、集計式）
- **問題**: 原論文の重みは「訓練済みアダプタの出力分布」を前提に導かれた回帰係数であり、LLM-judge の 0-3 スコア分布に同じ重みを掛けて 0-5 リスケールした値が同じ意味を持つ保証はない。特に負重み指標（opinion, question 等）は LLM の採点の癖に敏感。ドキュメント上の自覚はあるが、検証タスクがどこにも積まれていない。
- **改善案**: 段階 A に「AQuA LLM-judge vs 人手アノテーション（20 indicator のうち主要 5 つで可）の一致率」を追加。また重み付き合成値だけでなく `per_indicator_mean` を主報告にする（既に保存はされている。aqua.py:375-378 — これは良い設計）。

#### M-8. 判定タイミングの結合: stop_condition 評価時に最新発話の extraction が未完了になりうる

- **file:line**: `src/das/eval/run_eval.py:505-566`（発話 yield 後に stop_condition が呼ばれるが、full_proposal のグラフ反映は**次ターンの** `info_provider` 内 `bus.drain()`（conditions.py:466-471）で行われる）、`src/das/eval/consensus.py:204-219`（直近窓の新規 claim/attack ゼロ判定）
- **問題**: 最後の発話がまだグラフに載っていない時点で `new_claim_stalled` / `no_new_attacks` を判定するため、停滞シグナルが構造的に立ちやすい（1 ターン分のラグ）。構造シグナル 2 つで合意成立しうる（consensus.py:230）ので、full_proposal の早期終了を過剰に引き起こす方向のバイアス。
- **改善案**: stop_condition の前に条件の store を明示的に drain する（`_run_single` は condition を知っているので可能）。あるいは構造シグナルの窓から最新ターンを除外する。

### Low

#### L-1. Gini 係数の実装が 2 つある

- **file:line**: `src/das/eval/metrics.py:45-63`（ソート式）と `src/das/eval/structural_metrics.py:87-99`（O(n²) 全対差分式）
- **問題**: 同じ指標の実装が 2 箇所にあり、将来片方だけ修正されるリスク。値は一致するはずだが保証がない。
- **改善案**: `metrics.gini_coefficient` に一本化して structural_metrics から import。

#### L-2. インクリメンタル snapshot が毎ターン全書き換えで O(n²) I/O、例外は広く握り潰し

- **file:line**: `src/das/eval/run_eval.py:549-566`（毎ターン `store_now.snapshot()` を indent=2 で全量書き出し、`contextlib.suppress(Exception)`）
- **問題**: プロトタイプ規模では実害が小さいが、長セッション・高並列時にディスク I/O が支配的になる。suppress(Exception) はデータ破損時に無音になる。
- **改善案**: snapshot は k ターンごと + 最終時に。suppress する場合も warning ログは出す。

#### L-3. `_run_eval_cli` のデフォルト引数が CLI オプションのデフォルトと不一致

- **file:line**: `src/das/cli/_eval.py:424-425`（`agreement_threshold=0.6, min_turns_before_consensus=4`）vs `_eval.py:51-60`（CLI 側は 0.67 / 6）
- **問題**: CLI 経由では常に上書きされるため実害はないが、この関数を直接呼ぶテスト・スクリプトが CLI と異なる挙動になる罠。
- **改善案**: 内部関数のデフォルトを削除して必須引数化するか、定数を 1 箇所に。

#### L-4. viz が node_type（claim/premise/evidence）を描き分けない

- **file:line**: `src/das/viz/render.py:25-35`（色・形は `source` のみで決定）、`render.py:83-90`
- **問題**: README・RESEARCH.md が強調する「claim / premise / evidence の 2 系統ノード設計」が、デバッグ用ビューアで視認できない（hover の title には出るが一覧性がない）。linking / extraction の品質デバッグ（段階 A）で claim と premise の区別は重要。
- **改善案**: node_type で枠線色 or 形状を追加分岐（claim=太枠、premise=細枠など）。

#### L-5. `citation` の embedding 対応付けに `id(item)` を辞書キーとして使用

- **file:line**: `src/das/eval/citation.py:221`, `citation.py:304`
- **問題**: 現状は同一 list を通すため動くが、途中で dict を再構築する呼び方をすると無警告で embedding 経路が無効化する脆い契約。
- **改善案**: item に安定キー（`(entry_index, item_index)`）を振る。

#### L-6. `das eval` のオプションが 25 個超に肥大

- **file:line**: `src/das/cli/_eval.py:17-139`
- **問題**: linking のコストチューニング系（top-k / per-source / model）と実験デザイン系（conditions / n_runs / consensus）と運用系（budget / concurrency / emit-events）が同列に並び、実験の再現条件を CLI 引数の組合せとして口伝することになる。
- **改善案**: 実験デザイン系を preset/設定ファイルへ吸い上げ（M-4 と同件）。meta.json に全 CLI 引数を保存しているか確認し、していなければ保存する（現状 meta には linking 系が入っていない: run_eval.py:860-879）。

---

## 評価設計の研究的妥当性の分析

### 「AI 介入は会議を良くするか」に answer できる構造になっているか

**枠組み自体は正しい方向にある。** (1) 主観（judge 5 軸）× 客観（構造指標）× 行動（citation, stance shift）の三角測量、(2) 外部枠組み（AQuA/DQI）による外部基準の導入、(3) 介入の負の効果（発話萎縮、dropout）まで測る対称性、(4) proxy（LLM シミュレーション）→ 対面実験へ同一指標で transferability を確保する二段戦略 — これらは研究計画として筋が良く、The Social Laboratory / DEBATE benchmark 系の先行と比較可能な形を意識している点も評価できる。

**しかし現状の実装は、条件差の主要な読み取り口 3 つすべてに実装由来バイアスがある:**

| RQ | 主要指標 | 現状の問題 |
|---|---|---|
| RQ1 (議論の質) | judge 5 軸 / AQuA / 構造指標 | judge unblinded (C-1)、構造指標は条件比較不能 (H-2)、疑似反復 (H-5) |
| RQ3 (透明性) | intervention_transparency | プロンプトが期待方向を明示指示 (C-1) — 現状ほぼ無効 |
| RQ4 (外部知識活用) | citation_rate | 照合対象の取り違え (C-2) + 閾値未較正 (H-4) + 指示遵守の混入 (M-6) |

つまり「仮説に沿った結果（透明性↑）」は誘導の産物である疑いが強く、「仮説に反した結果（満足度↓）」の方がむしろ信頼できる、という逆説的な状態にある。予備実験の解釈（RESEARCH.md §6）は C-1 修正後に再取得すべき。

### LLM ペルソナ・シミュレーションの proxy としての妥当性

- ペルソナ設計（stance × focus × personality、pro/con/neutral の 3-4 名）は最小限だが目的に足る。ただし persona プロンプトの「(a)〜(d) を必ず行え」「同じ言い回しの繰り返し禁止」（persona.md:19-42）は、議論の質を**プロンプトで底上げ**しており、介入が改善余地を持ちにくい高ベースラインを人工的に作る。介入効果の検出力を下げる方向の設計であることは自覚しておくべき（悪いとまでは言わないが、論文の limitation に明記が必要)。
- round-robin 固定順は「発言量の偏り」「割り込み」という対面会議の主要な現象を消してしまうため、participation_gini / speaker_dropout / longest_silence はシミュレーションではほぼ縮退する（gini は構造的に ~0）。これらの指標は対面 Phase 2 専用と割り切って、シミュレーション summary からは外すか「参考値」と明記した方が誠実。
- LLM judge の既知バイアスへの対策状況: 位置バイアス — 絶対評価なので直接は該当しないが、pairwise 化する場合は順序ランダム化が必須。**自己選好** — 未対策 (M-1)。**verbosity バイアス** — 条件によって transcript 長が変わる（until_consensus, H-1）ため間接的に混入しうる。**スコア圧縮**（Likert が 4-6 に集中） — 温度 0 の単発採点で分散推定なし。複数 judge 合議（ChatEval 型）か、最低でも異種モデル 2 judge の平均を推奨。

### 統計面の総括

n=2〜5 の予備段階でこれを咎めるのは酷だが、Phase 1 本ラン（n=5〜10）に進む前に: (1) クラスタ構造を尊重した集計（H-5）、(2) 打ち切りを考慮した time-to-consensus（M-5）、(3) bootstrap CI、(4) 主要比較（full_proposal vs flat_rag の citation_rate / 透明性）の事前登録的な固定 — の 4 点は必須。多重比較（5 軸 × 20 indicator × 構造 10 指標超）に対する言及もどこにもないため、探索的指標と確証的指標を区別する宣言が要る。

### 対症療法的ロジックの棚卸し

- 合意キーワード + 逆接 30 文字近接（consensus.py:47-93）: 典型的な場当たり補正 (M-2)。
- judge.md の条件別スコア指示（judge.md:24）: 「透明性スコアが期待と逆に出た」ことへの補正として入った疑いが濃い。測定の破壊 (C-1)。
- `stall_max_new_claims=1`（facilitation.py:126-127, RESEARCH.md「しきい値再調整」）: 予備実験で発火しなかったことへの後付け調整。調整自体は許容範囲だが、しきい値を動かした場合は予備実験データでの再検証ログを残すべき。
- flat_rag → citation 計算のための intervention_log 形式合わせ（conditions.py:189-211）: これは対症療法ではなく妥当な統一化。ただし relation="" の下流処理 (H-3) が抜けた。

---

## 根本的な再設計提案

### 1. 「実行」と「採点」の完全分離（rescore-everything アーキテクチャ）

現状 aqua-rescore だけが post-hoc 再採点になっているが、これを全指標に一般化する。

```
das eval          → transcript / interventions / snapshot / meta のみ生成（LLM は persona+facilitator のみ）
das score <dir>   → judge / structural / citation / consensus / stance / aqua を
                    登録済み Scorer として一括 or 個別に post-hoc 実行
```

利点: (a) C-1/H-3 のような judge 側バグの修正が**再ラン不要**（既存 transcript の再採点で済む = API コスト数十分の一）、(b) judge モデルの差し替え・複数 judge 合議・人手採点の差し込みが自由、(c) run_eval.py の肥大 (M-3) が構造的に解消。`SingleRunResult` はディスク上の run ディレクトリを正とし、summary はスコア成果物の集約に徹する。

### 2. 「観測用グラフ」の全条件適用

処置（介入）に使うグラフと、測定（構造指標・合意検出）に使うグラフを分離し、後者は**全条件の transcript に同一パイプラインを post-hoc 適用**して構築する。これで H-1（停止規則の対称性）と H-2（構造指標の比較可能性)が同時に解決し、「介入が議論の argumentation 構造をどう変えたか」という本来の RQ1 の客観測定が初めて可能になる。段階 A の linking F1 測定（comparison_execution_plan Week 1）は、この観測用グラフの信頼性の裏付けとしてそのまま機能する。

### 3. 条件系列の再設計（dose を固定して 1 要因ずつ動かす）

`none → flat_rag(頻度を full_proposal に yoke, 件数 2) → full_proposal_unlabeled(関係ラベル除去) → full_proposal` の 4 条件に組み直す。特に `full_proposal_unlabeled` は既存の `_format_l1_*` からタグを外すだけで実装でき、RQ4（関係ラベルの寄与）を交絡なしで検定できる最重要 ablation。graphless_facilitation は「グラフの寄与」用として CLI に接続する (M-4)。

### 4. 測定のブラインド化と較正を実験プロトコルに組み込む

- judge / AQuA への入力から条件識別可能な情報を排除（条件名・「情報提供なし条件」表記・提示ログの体裁差を最小化）。
- citation / consensus / AQuA の 3 つの自動測定について、それぞれ 30〜50 サンプルの人手ラベルとの一致率（κ）を測る「較正ステップ」を段階 A に正式に追加。これがないと、いくら n を増やしても「測定器が何を測っているのか不明」という批判に耐えられない。査読でも最初に突かれる箇所。

### 5. summary スキーマの統計強化

summary.json の各指標を `{mean, ci95_low, ci95_high, n_runs, n_clusters}`（ラン単位 bootstrap）に統一し、ペルソナ stance 別の層別テーブルを常設。確証的指標（事前指定: citation_rate by source, opposition_understanding, intervention_transparency）と探索的指標（それ以外全部）をスキーマ上で区別するフィールドを持たせる。

---

*レビュー実施日: 2026-07-02*
