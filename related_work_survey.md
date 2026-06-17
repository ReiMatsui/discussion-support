# 関連研究サーベイ — 議論グラフ統合型 マルチエージェント議論支援

> 目的：本研究（議論ログと外部知識を単一の論証グラフ＝AFとして統合し、支持/攻撃エッジで連結、5エージェントで構築・運用、対面議論の確証バイアス緩和）の関連研究を網羅的に整理し、最近接研究との差分（ギャップ）と位置づけを明確化する。
> 注：掲載文献はWeb検索で実在確認済み。末尾の「要確認」節に、検索で出たが書誌情報が未確定の文献を分離した。

---

## 0. 位置づけ（一文）

> 先行研究は本研究の構成要素を**個別には**カバーしている（AF理論／LLM論証マイニング／LLMによるAF構築／AIによる熟議メディエーション／確証バイアス対策のLLMディベート）。しかし、**ライブの対面議論ログと外部知識を一つのLLM構築型AFグラフに融合し、その支持/攻撃エッジの構造を使ってリアルタイムにバイアス配慮のファシリテーションを行う**システムは見当たらない。これが本研究のギャップである。

---

## 1. テーマ別マップ

### A. 議論支援・自動ファシリテーション（伊藤研の系譜＝直接の母体）

- **Ito, Imi, Hideshima ほか「COLLAGREE」(2014–2018)** — ファシリテータ媒介の大規模合意形成プラットフォーム。名古屋市の実証で実運用。LLM以前・グラフ自動構築なし。
- **Ito, Hadfi, Suzuki (2021) "An Agent that Facilitates Crowd Discussion." *Group Decision and Negotiation* 31:621–646.** — IBIS構造を抽出し対象を絞った介入を投稿する自動ファシリテーションエージェント。**本研究の直接の前身**。グラフは内部的・外部知識と非融合・単一パイプライン。
- **Ito ほか (2020) "D-Agree: Crowd Discussion Support System Based on Automated Facilitation Agent." AAAI 2020 (Demo).** — 実運用システム。IBISベースの構造抽出＋介入。本研究はこれをLLM化＋知識連結グラフ化して刷新する位置づけ。
- **(伊藤研系) "Extraction of Online Discussion Structures for Automated Facilitation Agent" (LNCS 2020) / GAT等によるIBISリンク抽出** — グラフ構築要素の技術的隣接研究。外部知識・AF意味論・多エージェントは無し。
- **★ Dong, Ding, Ito (2024) "An Automated Multi-Phase Facilitation Agent Based on LLM." *IEICE Trans. Inf. & Syst.* E107.D(4).** — 伊藤研系列で**初のLLM統合**。多段階で自然な介入文を生成。**単一エージェントの最近接ベースライン**。持続的な知識連結AFや支持/攻撃エッジ分類は無い。
- **Hadfi, Haqbeen, Sahab, Ito (2023) "Conversational agents enhance women's contribution in online debates." *Scientific Reports* 13:14534.** — ファシリテーション介入が参加・包摂を高めるフィールド実証。「ファシリテーションが熟議を改善する」前提の裏づけ。

### B. LLMによる合意形成・熟議メディエーション（目的の双子）

- **★ Tessler ほか (2024) "AI can help humans find common ground in democratic deliberation." *Science* 386 (DeepMind, Habermas Machine).** — LLM「調停者」が集団の合意文を反復生成、人間調停者を上回る。**最重要ベースライン**。出力は**合意テキスト**で、AFグラフ・外部知識融合・支持/攻撃構造・事実確認は持たない（論文自身が限界として明記）。本研究はこの空白を直接狙う。
- **★ Gu, Hadfi, Ito ほか (2025) "PTFA: An LLM-Based Agent that Facilitates Online Consensus Building Through Parallel Thinking." PRICAI 2025 (LNAI).** — Six Thinking Hatsの6役割をLLMが並行。役割は**ファシリテーション様式**であり、知識連結AFの共同構築ではない。
- **★ Fulay & Roy (2025) "The Empty Chair: Using LLMs to Raise Missing Perspectives in Policy Deliberations." CSCW 2025.** — 不在ステークホルダーをLLMペルソナで補完。**対面設定・確証バイアス/視点拡張の目的を共有**。機構はペルソナ注入で、持続的AFグラフ・外部知識グラフ・エッジ分類は無い。
- **Konya ほか (2025) "Using Collective Dialogues and AI to Find Common Ground..." FAccT 2025.** — 実地での合意形成（橋渡しランキング＋LLM文生成）。グラフ・エッジ・知識融合なし。
- **★ (2025) "Can AI Truly Represent Your Voice...?" / DeliberationBank (arXiv:2510.05154).** — フラット要約が**少数意見を過小表現**し入力順バイアスを持つことを実証。**「平坦な要約」の失敗を定量化**＝本研究の構造的AF（各主張をノード保持）の動機づけに使える批判的引用。
- **Small ほか (2021) "Polis: Scaling Deliberation by Mapping High Dimensional Opinion Spaces."** — 賛否投票＋クラスタリングで意見空間を地図化（vTaiwan等で実運用）。**意見/投票空間**であり論証グラフではない（主張間の支持/攻撃なし）。

### C. 計算論的議論 × LLM（AF構築・論証マイニング）

**基礎**
- **★ Dung (1995) "On the acceptability of arguments..." *Artificial Intelligence* 77(2):321–357.** — 抽象論証フレームワーク（AF）。本研究グラフの形式的土台。
- **★ Cayrol & Lagasquie-Schiex (2005) Bipolar Argumentation Frameworks (BAF).** — 攻撃に加え**支持**関係を明示的に導入。本研究が支持/攻撃両エッジを持つ根拠。
- **Baroni, Caminada, Giacomin (2011) "An introduction to argumentation semantics." *Knowledge Eng. Review* 26(4).** — grounded/preferred/stable等の意味論サーベイ。受理可能性計算の参照。
- **Modgil & Prakken (2014) ASPIC+ tutorial.** — 構造化論証（規則・前提から主張を構築）。発話/根拠を整形する層の指針。

**論証マイニング**
- **★ Lawrence & Reed (2019) "Argument Mining: A Survey." *Computational Linguistics* 45(4).** — 論証マイニングの定番サーベイ。本研究の抽出能力の定義。
- **Stede & Schneider (2018) *Argumentation Mining* (Synthesis Lectures).** — 教科書的体系。対話の論証マイニングを含む。
- **Stab & Gurevych (2017) "Parsing Argumentation Structures in Persuasive Essays." *CL* 43(3).** — 古典的エンドツーエンド論証構造解析。
- **★ H. Li ほか (2025) "Large Language Models in Argument Mining: A Survey." (arXiv:2506.16383).** — LLM時代の論証マイニング最新サーベイ。抽出要素の現状アンカー。

**LLM × AF（最近接クラスタ）**
- **★ Freedman, Dejl, Gorur, Yin, Rago, Toni (2025) "Argumentative LLMs for Explainable and Contestable Claim Verification." AAAI 2025 (arXiv:2405.02079).** — LLMがAFを構築し**形式的勾配意味論**で主張検証。**「LLMがAFを構築して推論」最近接**。単一主張検証・単一エージェント・ライブ議論ログ無し。
- **★ Gorur, Rago, Toni (2025) "Can LLMs perform Relation-based Argument Mining?" COLING 2025 (arXiv:2402.11243).** — LLMが主張対の**支持/攻撃を分類**できることを実証。**本研究のエッジ構築の実現可能性の裏づけ**。
- **Chen, Cheng, Luu, Bing (2024) "Exploring the Potential of LLMs in Computational Argumentation." ACL 2024 (arXiv:2311.09022).** — 主張検出・スタンス・関係分類・生成のベンチマーク。各サブタスクのLLM実力。
- **★ Hong, Xiao ほか (2024) "ArgMed-Agents: Explainable Clinical Decision Reasoning with LLM Discussion via Argumentation Schemes." IEEE BIBM 2024 (arXiv:2403.06294).** — LLMエージェントが議論スキームで自己論争→**攻撃グラフ（AF）構築→記号ソルバで非矛盾集合選択**。**多エージェント＋AF構築の最近接アーキテクチャ**。臨床・エージェント同士の論争で、ライブ人間議論ログ・外部知識融合は目的外。
- **Sanayei, Vesic, Blanco, Surdeanu (2025) "Can LLMs Judge Debates? ... Argumentation Theory Semantics." Findings of EMNLP 2025 (arXiv:2509.15739).** — 実ディベート→論証グラフ→意味論評価。本研究の表現と同型（評価視点）。

**ツール・標準・対話**
- **★ Slonim ほか (2021) "An autonomous debating system." *Nature* 591 (IBM Project Debater).** — 人間と討論する end-to-end システム。**競争的1対1ディベート**で、共有AFグラフによる人間熟議の支援ではない。旗艦先行システムとして対比。
- **Argument Interchange Format (AIF; Chesñevar/Rahwan ほか 2006) / Araucaria・OVA・Kialo** — 論証グラフの表現標準・可視化・構造化討論UX。手動構築が主で、ライブ会話からのLLM自動構築・外部知識統合は無い。
- **Prakken (2006) "Formal systems for persuasion dialogue." *KER* 21(2).** — 説得対話の形式系。ファシリテーションを対話プロトコルとして枠づける背景。

### D. RAGの限界とスタンス/論証認識型検索（課題C1の裏づけ）

- **★ Lewis ほか (2020) "Retrieval-Augmented Generation..." NeurIPS 2020.** — RAGの原典。**類似度駆動・スタンス無視**で、支持/攻撃構造を持たない＝本研究が批判する基準線。
- **Gao ほか (2024) "Retrieval-Augmented Generation for LLMs: A Survey." (arXiv:2312.10997).** — RAG限界の全体像。スタンス均衡は対象外。
- **Chen ほか (2024) "Benchmarking LLMs in RAG (RGB)." AAAI 2024.** — ノイズ頑健性・統合等4能力。**視点バイアスは含まれない**（=本研究の隙間）。
- **★ Sharma ほか (2024) "Towards Understanding Sycophancy in Language Models." ICLR 2024.** — LLMがユーザの信念に**追従（sycophancy）**することをRLHFに帰す。確証バイアス助長の中心的根拠（生成側）。
- **Liu ほか (2024) "Lost in the Middle." TACL** / **Yu ほか (2024) "Chain-of-Note." EMNLP 2024** / **Cuconasu ほか (2024) "The Power of Noise." SIGIR 2024.** — 文脈位置・ノイズ頑健性。検索文書を「関連/ノイズ」の均質物として扱い、ライブ主張に対する**支持/攻撃で型付けしない**。
- **★ Zhao ほか (2024) "Beyond Relevance: ... Perspective Awareness (PIR)." (arXiv:2405.02714).** — 主張に対し**支持/反対文書を区別**する検索器。**検索側の最近接**。ただし一発検索に留まり、ライブ議論に紐づく持続的論証グラフは持たない。
- **★ Chen ほか (2024) "Open-World Evaluation for Retrieving Diverse Perspectives (BeRDS)." (arXiv:2409.18110).** — 標準検索器が全視点を被覆できるのは約33.7%のみと定量化。類似度検索の多様性不足の証拠。
- **Bondarenko ほか (2020+) "Touché: Argument Retrieval." ECIR/CLEF.** — 賛否を扱う論証検索タスクの基盤（IR系譜）。
- **★ Thorne ほか (2018) "FEVER." NAACL 2018.** — SUPPORTED/REFUTED/NEI の三値。**本研究の支持/攻撃/中立ラベルの源流**。
- **★ Edge ほか (2024, Microsoft) "From Local to Global: A Graph RAG Approach." (arXiv:2404.16130).** — 旗艦GraphRAG。ただし**エンティティ知識グラフ**で、**論証的な支持/攻撃ではない**。「GraphRAG（エンティティ）」と「論証グラフRAG（スタンス）」の区別が本研究の差分。
- **Chen ほか (2024) "Detecting Hallucination and Coverage Errors in RAG for Controversial Topics." LREC-COLING 2024.** — 「被覆誤り（視点欠落）」を定義＝本研究の均衡評価指標に直結。

### E. マルチエージェントLLM・ディベート（アーキテクチャの系譜）

- **★ Du ほか (2024) "Improving Factuality and Reasoning ... through Multiagent Debate." ICML 2024.** — マルチエージェントディベートの定番。**均質コピー**で、機械の正答が目的（人間熟議支援ではない）。状態は自由文の transcript。
- **Liang ほか (2024) "Encouraging Divergent Thinking ... (MAD)." EMNLP 2024.** — 「Degeneration-of-Thought」を提起し、早期収束を抑える。本研究の早期収束抑制と概念的に近い。
- **Wu ほか (2023) AutoGen** / **Li ほか (2023) CAMEL** / **★ Hong ほか (2024) MetaGPT (ICLR)** / **Qian ほか (2024) ChatDev (ACL)** — 役割特化エージェントの分業と**構造化共有成果物**。MetaGPTが**役割分業＋構造化共有**の最近接先行。ただし収束的なSWタスクで、論証グラフ・熟議品質目的は無い。
- **Chan ほか (2024) "ChatEval." ICLR 2024.** — 多役割の討論が評価を改善。パネル＞単体の傍証。
- **★ Smit ほか (2024) "Should we be going MAD? ..." ICML 2024.** — ディベートが自己無撞着/アンサンブルを安定して上回らないことを示す**反証**。本研究は「機械の正答」でなく「人間熟議の質」を目的とし、構造（グラフ）を外在化する点で答える必要がある。
- **(2025) Blackboard型 LLM-MAS (arXiv:2507.01701 ほか)** — 共有黒板に全情報を載せる**最近接アーキテクチャ**。ただし共有ストアは非構造のメッセージ記憶で、型付き論証グラフではない（=本研究の貢献点）。※2025プレプリント、査読状況要確認。

### F. 確証バイアス・群極化と介入（認知/社会科学＋HCI）

- **★ Nickerson (1998) "Confirmation Bias: A Ubiquitous Phenomenon..." *Review of General Psychology* 2(2).** — 確証バイアスの古典的定義。
- **★ Sunstein (2002) "The Law of Group Polarization." *J. of Political Philosophy* 10(2).** — 同質集団の極化。
- **★ Mercier & Sperber (2011) "Why Do Humans Reason? ... Argumentative Theory." *BBS* 34(2).** — 推論は論証のために進化し、**集団での論証交換は良く機能する**。分散・論証ベース設計の理論的正当化。
- **Stasser & Titus (1985) hidden-profile / 情報共有バイアス** — 集団は共有情報を過剰に議論し独自情報を出さない。過小表現論点を能動的に提示する動機。
- **Bakshy, Messing, Adamic (2015) Science / Flaxman ほか (2016) POQ** — エコーチェンバー/フィルターバブルの大規模実証（過度な主張を避けバランス良く動機づけ）。
- **★ Kriplean ほか (2012) "ConsiderIt." CSCW 2012** / **"Reflect." CHI 2012** — 賛否ポイントで反省的熟議を支援する**最近接HCI先行**。本研究は人手の賛否リストをLLM構築の完全な論証グラフへ一般化。
- **Munson & Resnick (2010) "Presenting Diverse Political Opinions." CHI 2010.** — 反対意見の提示は挑戦回避型ユーザに逆効果になり得る＝**提示設計の重要な注意点**。
- **★ Shi ほか (2024) "Argumentative Experience: Reducing Confirmation Bias ... LLM-Generated Multi-Persona Debates." (arXiv:2412.04629).** — **確証バイアス目的・LLM多エージェントの最近接**。アイトラッキングで緩衝効果。機構は一過性のペルソナ討論で、**持続的AFグラフ・5役割の機能特化が無い**。本研究の差分の中心。
- **Chiang ほか (2024) "LLM-Powered Devil's Advocate." IUI 2024** / **(2024) 臨床多エージェントでの認知バイアス緩和 JMIR e59439** — 「悪魔の代弁者」役の有効性。役割特化でバイアスに対処する本研究前提の傍証（グラフ無し）。

---

## 2. 最近接研究（must-engage）— 必ず本文で扱うべき10本

1. **Tessler ほか 2024 (Habermas Machine, Science)** — AI熟議メディエーションの旗艦。合意文 vs 論証グラフ、外部知識/事実確認なし。
2. **Ito/Hadfi 2021 + D-Agree (AAAI 2020)** — IBIS自動ファシリテーションの母体。
3. **Dong, Ding, Ito 2024 (LLM多段ファシリテーション)** — 系譜内の単一エージェントLLMベースライン。
4. **Gu ほか 2025 (PTFA, PRICAI)** — 多役割LLMファシリテーションの最近接（役割 vs グラフ）。
5. **Fulay & Roy 2025 (Empty Chair, CSCW)** — 確証バイアス目的・対面設定（ペルソナ vs グラフ）。
6. **Freedman ほか 2025 (ArgLLMs, AAAI)** — 「LLMがAF構築＋形式意味論」の最近接。
7. **Hong ほか 2024 (ArgMed-Agents)** — 「多エージェント→AF→ソルバ」の最近接アーキテクチャ。
8. **Gorur, Rago, Toni 2025 (RbAM with LLMs, COLING)** — LLMによる支持/攻撃エッジ構築の実現可能性。
9. **Shi ほか 2024 (Argumentative Experience)** — 確証バイアス×LLM多エージェントの最近接（ペルソナ討論 vs 持続グラフ）。
10. **Zhao ほか 2024 (Beyond Relevance / PIR) ＋ Edge ほか 2024 (GraphRAG)** — 検索側最近接（スタンス検索）／知識グラフRAGとの区別。

（基礎の必須引用：Dung 1995, Bipolar AF (Cayrol & Lagasquie-Schiex 2005), Lawrence & Reed 2019, FEVER 2018, Nickerson 1998, Sunstein 2002, Mercier & Sperber 2011）

---

## 3. 位置づけ表（システム × 特徴）

凡例：✓=有 / △=部分的 / ✗=無

| システム | 持続的AFグラフ | 議論ログ＋外部知識の融合 | 支持/攻撃エッジ分類 | 多エージェント分業 | リアルタイム対面 | 確証バイアス抑制が目的 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **本研究** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| D-Agree / IBIS (Ito 2020–21) | △(IBIS・内部) | ✗ | △(支持/反対) | ✗ | ✗(オンライン) | △ |
| Dong, Ding, Ito 2024 (LLM) | ✗ | ✗ | ✗ | ✗ | ✗ | △ |
| Habermas Machine (2024) | ✗ | ✗ | ✗ | ✗ | ✗(非同期) | △(分断低減) |
| PTFA (2025) | ✗ | ✗ | ✗ | △(役割) | ✗ | △ |
| The Empty Chair (2025) | ✗ | ✗ | ✗ | △(ペルソナ) | ✓ | ✓ |
| MeetMap (2025) | △(対話マップ) | ✗ | ✗ | ✗ | △(オンライン) | ✗ |
| Polis (2021) | ✗(意見空間) | ✗ | ✗ | ✗ | ✗ | △ |
| ConsiderIt (2012) | △(賛否・人手) | ✗ | △ | ✗ | ✗ | ✓ |
| ArgLLMs (2025) | ✓(主張単位) | △(LLM内部知識) | ✓ | ✗ | ✗ | ✗ |
| ArgMed-Agents (2024) | ✓(攻撃グラフ) | ✗ | △(攻撃) | ✓ | ✗ | ✗ |
| Argumentative Experience (2024) | ✗ | △(検索比較) | ✗ | △(ペルソナ) | ✗ | ✓ |
| GraphRAG (2024) | ✗(エンティティ) | △ | ✗ | ✗ | ✗ | ✗ |

> この表が示す通り、各特徴は個別には存在するが、**全列を同時に満たすのは本研究のみ**。

---

## 4. ギャップと新規性の主張（本文用ドラフト）

4クラスタ70本超を通じて、以下を**同時に**満たす研究は確認できなかった。

1. **共有表現としての単一の持続的AFグラフ**（支持＋攻撃、Dung/双極）。合意文（Habermas）・ディベート発話（Project Debater）・賛否木（Kialo）・単一主張検証（ArgLLMs）のいずれでもない。
2. **ライブ対面議論ログ＋外部知識の単一グラフへの融合**。既存は会話のみ（Habermas）か外部証拠のみ（Hua&Wang等）かエージェント同士の論争（ArgMed-Agents）で、**部屋のライブ熟議と検索知識を一つのAFに統合しない**。本研究では議論側を立場を持つ主張（claim/premise）、外部知識を中立な事実（evidence）としてノード化し、事実が各主張を支持/攻撃するかを**対象主張ごとのエッジ**で型付けする（FEVER の SUPPORTS/REFUTES/NEI に対応）。同じ事実が一方の主張を支持し他方を攻撃しうる相対性を、ノードではなくエッジで表す点が要。
3. **その構築を担うマルチエージェントLLM分業**（最近接：ArgMed-Agentsはエージェント間・臨床、ArgLLMsは単一・単一主張）。
4. **人間熟議のリアルタイム・ファシリテーション**（多くはオフライン/非同期/評価のみ/競争的）。
5. **確証バイアス抑制を目的とし、AF構造（未反論/未支持の検出）として作動**。Shiらや「認知資源」論文はバイアスを扱うがAFグラフ・ライブ集団を欠き、Habermasは分断/合意が主目的。

**一文の位置づけ：** 先行研究は構成要素を個別にカバーするが、**ライブ多者議論ログと外部知識を一つのLLM構築型AFグラフに統合し、その意味論をリアルタイムのバイアス配慮ファシリテーションに用いる**のは、現時点で本研究が初と考えられる。最も新規性を脅かすのは **PIR (検索)・GraphRAG (グラフ)・Argumentative Experience (バイアス×多エージェント)・ArgMed-Agents (多エージェント×AF)** であり、本文でこれらが「持続的・議論連結の支持/攻撃グラフを持たない」ことを明示して差別化する。

---

## 5. 関連研究スライドの改善案

現状は2本（PTFA・Empty Chair）のみで薄い。次のいずれかを推奨：

- **案A（推奨）：位置づけ比較表1枚**＋関連研究2枚（テーマ別）。比較表は§3を簡略化（行を主要6システム、列を6特徴）。新規性が一目で伝わり、審査の「何が新しいか」に即答できる。
- **案B：テーマ別マップ2枚**（①議論支援・合意形成系 ②計算論的議論＋RAG＋マルチエージェント系）＋各テーマ2–3本。
- **案C：最近接5本を「目的が近い／機構が近い」で対比する1枚**＋比較表1枚。

→ いずれも、最後に「本研究＝唯一全特徴を満たす」を比較表で締める構成が効く。

---

## 6. 要確認・未検証（引用前に各自で確認）

検索で概念は妥当だが書誌IDが未来日付等で未確定のため**本サーベイの本体からは除外**したもの：
- 「Retrieval Sycophancy」系（FVA-RAG 2512.07015, CoRM-RAG 2605.01302）→ 概念は動機づけに使えるが、引用は要ID確認。
- ARGORA (2601.21533), ArgBench (2604.17366), QBAF集約意味論 (2603.06067) 等 → 実在すれば論証グラフ系で高関連。手動確認推奨。
- 書誌の細部要確認：COLLAGREE の確定版（2014–2018のどれを引くか）、Project Debater の Nature 巻号頁、Wachsmuth の品質次元数（15が標準）、Chiang ほか Devil's Advocate の正確な会議/年、Stasser & Titus 1985 の頁、Bench-Capon 2003 の書誌。
