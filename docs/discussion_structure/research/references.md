# 文献一覧

最終更新: 2026-10-06
対応する文書: `docs/discussion_structure/`（分野ごとのまとめは [related_work.md](related_work.md)）

議論構造の設計検討で参照した重要文献の一覧。「確認範囲」は、設計判断の根拠にした
内容をどこまで原文で確認したかを示す（本文 / 抄録・二次情報）。抄録・二次情報のものは、
論文に引用する前に本文を確認する。

## 1. リアルタイムの議論可視化（最も近い先行研究）

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Chen, W., Yu, C., Wang, Y., Chen, M., Xu, Y., Shi, Y. (2025). **EchoMind: Supporting Real-time Complex Problem Discussions through Human-AI Collaborative Facilitation.** PACM HCI 9(7), CSCW406. https://doi.org/10.1145/3757587 | 本文 | 対面・共有画面での問い／案の2層の木。論拠ノードを削除した理由。焦点は人が手動設定し、LLM による自動切替は適合率 28%・再現率 58%。遅延 4〜14 秒。焦点の部分木だけを更新する方式 |
| Chen, X., Yap, N., Lu, X., Gunal, A., Wang, X. (2025). **MeetMap: Real-Time Collaborative Dialogue Mapping with LLMs in Online Meetings.** PACM HCI (CSCW 2025). https://arxiv.org/abs/2502.01564 | 本文（要約経由） | オンライン2人組で IBIS 4種。AI が図を作る版と人が組む版の比較。賛否の取り違え・重複ノード・粒度への不満 |
| Chandrasegaran, S. et al. (2019). **TalkTraces: Real-Time Capture and Visualization of Verbal Content in Meetings.** CHI 2019. https://dl.acm.org/doi/10.1145/3290605.3300807 | 抄録 | 音声認識から話題を議題に対応づけるリアルタイム可視化 |
| **Shared Gaze on AI Node Maps: Mitigating Empathy Fog in AI-Mediated Dyadic Discussion.** https://dl.acm.org/doi/10.1145/3795011.3795058 | 抄録・二次情報 | 対面で AI のノードマップだけを見せると、注意が相手から画面へ移る |

## 2. 対話マッピングと IBIS

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Kunz, W., Rittel, H. (1970). **Issues as Elements of Information Systems.** Working Paper 131, UC Berkeley. | 二次情報 | IBIS（問い・案・論拠）の原典 |
| Conklin, J. (2005). **Dialogue Mapping: Building Shared Understanding of Wicked Problems.** Wiley. https://dl.acm.org/doi/10.5555/1237968 | 二次情報 | 書記が会議中に IBIS で図を描く手法 |
| Conklin, J. (2008). **Dialogue Mapping Demonstration.** DIAC-08. https://www.publicsphereproject.org/events/diac08/proceedings/33.Dialogue_Mapping.Conklin.pdf | 本文 | 要点は「暗黙の問いの明示」「図を参加者に確認する」「小さな図への分割」。会話の深い構造は問いで組織される。作業中の人に構造化を任せると負担が大きい |

## 3. 会議の議論構造の注釈と自動推定

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Rienks, R., Heylen, D., van der Weijden, E. (2005). **Argument Diagramming of Meeting Conversations.**（掲載先は要確認）https://research.utwente.nl/en/publications/argument-diagramming-of-meeting-conversations/ | 本文 | Twente Argument Schema（TAS）。問いを自由回答型・二択型・是非型に分ける |
| Verbree, D., Rienks, R., Heylen, D. (2006). **First Steps Towards the Automatic Construction of Argument-Diagrams from Real Discussions.** COMMA 2006. https://www.researchgate.net/publication/221650971 | 本文 | AMI コーパスの単位数（是非型 460・自由回答型 244・二択型 72、議論外の「その他」3061）。単位種類の自動判定 78.5%、注釈者一致 κ 0.50 → 0.87 |
| Rienks, R., Verbree, D. **Twente Argument Schema Annotation Manual v0.99b.** https://groups.inf.ed.ac.uk/ami/corpus/Guidelines/TAS-annotation-manual.pdf | 未読 | TAS の注釈手順（注釈設計の際に参照予定） |
| Hautli-Janisz, A. et al. (2022). **QT30: A Corpus of Argument and Conflict in Broadcast Debate.** LREC 2022. https://aclanthology.org/2022.lrec-1.352.pdf | 抄録 | 対話の議論構造コーパス（Inference Anchoring Theory） |
| Ruiz-Dolz, R. et al. (2024). **Overview of DialAM-2024: Argument Mining in Natural Language Dialogues.** ArgMining 2024. https://aclanthology.org/2024.argmining-1.8/ | 抄録・二次情報 | 対話の支持・攻撃関係の自動推定は最高でも F1 48（関係に絞った評価） |
| Lippi, M., Torroni, P. (2016). **Argument Mining from Speech: Detecting Claims in Political Debates.** AAAI 2016. https://ojs.aaai.org/index.php/AAAI/article/view/10384 | 抄録 | 音声からの主張検出 |
| Pallotta, V. et al. (2007). **User Requirements Analysis for Meeting Information Retrieval Based on Query Elicitation.** ACL 2007. https://aclanthology.org/P07-1127.pdf | 抄録・二次情報 | 会議記録への問い合わせの約6割が議論の過程と結果（何が論点で何が決まったか） |

## 4. 問い（QUD）の理論と解析

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Roberts, C. (1996/2012). **Information Structure in Discourse: Towards an Integrated Formal Theory of Pragmatics.** Semantics & Pragmatics 5. | 二次情報 | QUD の理論的基盤。問いを答えの集合として扱う |
| Ko, W.-J. et al. (2023). **Discourse Analysis via Questions and Answers: Parsing Dependency Structures of Questions Under Discussion.** Findings of ACL 2023. https://aclanthology.org/2023.findings-acl.710/ | 抄録 | 書き言葉の QUD 依存木の自動解析 |
| Wu, Y. et al. (2023). **QUDeval: The Evaluation of Questions Under Discussion Discourse Parsing.** EMNLP 2023. https://aclanthology.org/2023.emnlp-main.325/ | 抄録 | QUD 解析の評価方法 |

## 5. AI ファシリテーターと議論構造

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Ito, T. et al. (2020). **D-Agree: Crowd Discussion Support System Based on Automated Facilitation Agent.** AAAI 2020 Demo. https://aaai.org/papers/13614-d-agree-crowd-discussion-support-system-based-on-automated-facilitation-agent/ | 抄録・二次情報 | オンラインのテキスト議論から IBIS 構造を抽出し、エージェントの介入に使う |

## 6. 共有表示が議論に与える影響と、立場の可視化

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Suthers, D. (2003). **Representational Guidance for Collaborative Inquiry.** In *Arguing to Learn*, Springer. https://link.springer.com/chapter/10.1007/978-94-017-0781-7_2 | 抄録・二次情報 | 共有表現の形式が議論の内容と進み方を変える |
| DiMicco, J. M., Pandolfo, A., Bender, W. (2004). **Influencing Group Participation with a Shared Display.** CSCW 2004. | 抄録・二次情報 | 対面会議の共有画面で発言量を見せると、最も多く話す人の発言が減る（最も少ない人には効果なし） |
| Leshed, G. et al. (2009). **Visualizing Real-Time Language-Based Feedback on Teamwork Behavior in Computer-Mediated Groups.** CHI 2009. https://www.cs.cornell.edu/~gl/paper0382-leshed.pdf | 二次情報（Tausczik & Pennebaker 2013 での引用） | 合意の度合いを見せると合意が増えるが、受け身の同調になり成果の質が下がった |
| Tausczik, Y. R., Pennebaker, J. W. (2013). **Improving Teamwork Using Real-Time Language Feedback.** CHI 2013. https://dl.acm.org/doi/10.1145/2470654.2470720 | 本文の一部 | 上記 Leshed らの知見の引用元 |
| Mahyar, N. et al. (2017). **ConsensUs: Visualizing Points of Disagreement for Multi-Criteria Collaborative Decision Making.** CSCW 2017 Companion. https://groups.cs.umass.edu/wp-content/uploads/sites/8/2017/02/2017_CSCW_Demo.pdf | 本文の一部 | 不一致点の可視化で不一致の特定が向上。一方で、可視化により意見が集団寄りに変化 |
| Kriplean, T. et al. (2012). **Supporting Reflective Public Thought with ConsiderIt.** CSCW 2012. https://www.researchgate.net/publication/220879478 | 抄録 | 賛否リストで熟議を構造化（オンライン・非同期） |
| Faridani, S., Bitton, E., Ryokai, K., Goldberg, K. (2010). **Opinion Space: A Scalable Tool for Browsing Online Comments.** CHI 2010. | 抄録 | 意見の分布を2次元で可視化（オンライン・非同期） |

## 7. 知覚・可視化の基礎

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Cleveland, W. S., McGill, R. (1984). **Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods.** JASA 79(387). | 二次情報 | 量の読み取り精度は「位置・長さ」が高く、「色の濃淡・彩度」は低い |
| Kaufman, E. L. et al. (1949). **The Discrimination of Visual Number.** American Journal of Psychology 62(4). | 二次情報 | 4個程度までは数えずに瞬時に個数が分かる（subitizing） |

## 8. 焦点（話題）の推移の判定

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Grosz, B. J., Sidner, C. L. (1986). **Attention, Intentions, and the Structure of Discourse.** Computational Linguistics 12(3). https://www.let.rug.nl/nerbonne/teach/ling-tech/literature/Grosz-Sidner-1986.pdf | 二次情報 | 注意の状態を焦点空間のスタック（push / pop）で表す。合図の語句が移動を示す |
| Sacks, H. (1992). **Lectures on Conversation.** Blackwell. / Jefferson, G. (1984). On Stepwise Transition from Talk about a Trouble to Inappropriately Next-Positioned Matters. In *Structures of Social Action*. | 二次情報 | 区切りを明示する話題移行と、段階的な移行（stepwise）の区別 |
| Galley, M., McKeown, K., Fosler-Lussier, E., Jing, H. (2003). **Discourse Segmentation of Multi-Party Conversation.** ACL 2003. | 二次情報 | ICSI 会議コーパスの話題区切り（LCSeg）。細かい粒度では注釈者が一致しにくい |
| Hsueh, P.-Y., Moore, J. D., Renals, S. (2006). **Automatic Segmentation of Multiparty Dialogue.** EACL 2006. https://aclanthology.org/E06-1035.pdf | 抄録・二次情報 | AMI の話題境界の注釈者一致（上位 κ 0.79、下位 κ 0.73） |
| Pevzner, L., Hearst, M. A. (2002). **A Critique and Improvement of an Evaluation Metric for Text Segmentation.** Computational Linguistics 28(1). | 二次情報 | WindowDiff。境界の位置ずれを許容する評価指標 |
| Coen, M. H. (2025). **When F1 Fails: Granularity-Aware Evaluation for Dialogue Topic Segmentation.** arXiv:2512.17083. https://arxiv.org/abs/2512.17083 | 抄録 | 厳密一致の F1 は粒度の違いに左右される。許容幅つき F1 と粒度の診断を推奨 |
| Williams, J. D., Raux, A., Henderson, M. (2016). **The Dialog State Tracking Challenge Series: A Review.** Dialogue & Discourse 7(3). | 二次情報 | 発話ごとの観測から状態の信念を更新する対話状態追跡の考え方 |
| **Two Social Functions of Stepwise Transitions When Discussing Ideas in Workplace Meetings.**（2019）https://www.researchgate.net/publication/330024487 | 未読 | 職場会議でのアイデア検討中の段階的な話題移行（本文確認後に活用を判断） |

## 9. 判定専用モデル（2026年9月時点の製品情報）

学術文献ではなく製品情報。仕様は変わり得るため、利用時に公式文書で再確認する。

| 情報源 | 確認範囲 | 内容 |
|---|---|---|
| TypeSafe AI. **Jev（jev-1.13.0）モデル文書.** https://docs.typesafe.ai/models | 公式文書 | 文章を生成せず、是非・選択・スコアの判定を較正された確率で返す。入力はテキストのみ、64k トークン（状態 32k）。英語が主で、CJK は精度が下がると明記。顧客データで学習しない |
| DataCamp. **Jev: TypeSafe's System One Model That Never Hallucinates.** https://www.datacamp.com/blog/system-one-models-jev | 二次情報 | 遅延 70〜500 ms。複数の問いを1回で並列に判定。難しい事例だけ LLM に回す組み合わせ方 |
| TechCrunch (2026-09-18). **A new kind of AI model from a ChatGPT inventor is thrilling developers.** https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/ | 報道 | Jev の開発元と位置づけ |
| TechCrunch (2026-09-30). **OpenAI's Jev clone could help the frontier lab stop its swarming agents.** https://techcrunch.com/2026/09/30/openais-jev-clone-could-help-the-frontier-lab-stop-its-swarming-agents/ | 報道 | OpenAI の Decisions API（GPT-6 Luna、限定プレビュー） |

## 10. 遅延の基準

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Stivers, T. et al. (2009). **Universals and Cultural Variation in Turn-Taking in Conversation.** PNAS 106(26). https://doi.org/10.1073/pnas.0903616106 | 抄録・二次情報 | 順番交替の間は言語を問わず約 200 ms 前後（日本語を含む10言語） |
| Miller, R. B. (1968). **Response Time in Man-Computer Conversational Transactions.** AFIPS Fall Joint Computer Conference. / Card, S. K., Robertson, G. G., Mackinlay, J. D. (1991). The Information Visualizer. CHI 1991. / Nielsen, J. (1993). *Usability Engineering*. | 二次情報 | 応答時間の目安（0.1 秒・1 秒・10 秒） |
| Ofcom (2013–2015). **Measuring the Quality of Live Subtitling**（統計報告と声明）. https://www.ofcom.org.uk/__data/assets/pdf_file/0017/51731/qos-statement.pdf | 二次情報 | 生放送字幕の遅延の指針は最大 3 秒、実測平均 5〜6 秒。後に平均 4.5 秒を目標とする案 |
| Soniox. **Endpoint detection（文書）** https://soniox.com/docs/stt/rt/endpoint-detection / **Soniox v4 real-time（ブログ）** https://soniox.com/blog/2026-02-05-soniox-v4-real-time | 製品情報 | 終話検出の最大遅延は 0.5〜3 秒で設定可能。確定までの中央値約 250 ms の構成例 |

## 11. 議論と無関係な話（脱線）の検知

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Yoon, S.-Y. et al. (2017). **Off-topic Spoken Response Detection with Word Embeddings.** Interspeech 2017. https://www.isca-archive.org/interspeech_2017/yoon17_interspeech.pdf / Wang, X. et al. (2019). **Automatic Detection of Off-Topic Spoken Responses Using Very Deep Convolutional Neural Networks.** Interspeech 2019. | 抄録・二次情報 | 話題が明確に決まった試験の回答では、話題外の検出は F1 90% 前後。自由な議論より条件が易しい |
| **Navigating Wanderland: Highlighting Off-Task Discussions in Classrooms.** AIED 2023. https://link.springer.com/chapter/10.1007/978-3-031-36272-9_63 | 抄録・二次情報 | 協調学習での課題外の会話の自動検出。課題外の会話には退屈を和らげ関係を強める役割もある |
| Konigari, R. et al. (2021). **Topic Shift Detection for Mixed Initiative Response.** SIGDIAL 2021. https://aclanthology.org/2021.sigdial-1.17.pdf | 抄録・二次情報 | 会話の主要な話題に属する発話の判定（適合率 84%） |
| （内部）Rei. `origin/feat/live-native` の `src/das/asr/live/_drift.py` と `_DRIFT_LABEL_PROMPT`（2026-09-14） | 実装を確認 | 発話ごとの on / aside / off 判定と、off が続いた時間で脱線を測る状態機械 |

## 12. ノードの作成（問い・提案・決定の検出）

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Fernández, R., Frampton, M., Ehlen, P., Purver, M., Peters, S. (2008). **Modelling and Detecting Decisions in Multi-party Dialogue.** SIGdial 2008. https://aclanthology.org/W08-0125/ | 本文 | 決定に関わる発話の分類（問い・解決案・解決の言い直し・同意）。AMI で決定関連の発話は約 4%（問い 1% 未満、解決案約 1%、同意約 2%）。注釈者一致 κ 0.63〜0.73。当時の分類器の F1 は 0.25〜0.39 |
| Frampton, M., Huang, J., Bui, T. H., Peters, S. (2009). **Real-time Decision Detection in Multi-party Dialogue.** EMNLP 2009. https://aclanthology.org/D09-1118.pdf | 本文の一部 | 決定の検出をリアルタイムで行う構成（窓をずらしながら判定）。当時のリアルタイム音声認識の遅延は 5〜15 秒 |
| Stevanovic, M. (2012). **Establishing Joint Decisions in a Dyad.** Discourse Studies 14(6). https://journals.sagepub.com/doi/abs/10.1177/1461445612456654 | 抄録 | 提案が共同の決定になるには、聞き手の応答に理解・同意・拘束の受け入れが必要 |
| Huisman, M. (2001). **Decision-Making in Meetings as Talk-in-Interaction.** International Studies of Management & Organization 31(3). | 二次情報 | 会議の決定は明示されず、やり取りの中で成立することが多い |

## 13. 提示の時機と注意（構造を見せるか・いつ見せるか）

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Yantis, S., Jonides, J. (1984). **Abrupt Visual Onsets and Selective Attention: Evidence from Visual Search.** JEP: Human Perception and Performance 10(5). | 二次情報 | 新しく現れる視覚対象は注意を自動的に引く。画面の存在より変化が注意を奪う |
| Adamczyk, P. D., Bailey, B. P. (2004). **If Not Now, When? The Effects of Interruption at Different Moments within Task Execution.** CHI 2004. https://www.researchgate.net/publication/221518227 | 抄録・二次情報 | 作業の大きな区切りでの割り込みは、再開の負担が小さい |
| Iqbal, S. T., Bailey, B. P. (2008). **Effects of Intelligent Notification Management on Users and Their Tasks.** CHI 2008. https://interruptions.net/literature/Iqbal-CHI08.pdf | 抄録・二次情報 | 通知を区切りまで遅らせると、不満と反応時間が減る |
| Heritage, J., Watson, R. (1979). **Formulations as Conversational Objects.** In G. Psathas (Ed.), *Everyday Language: Studies in Ethnomethodology*. Irvington. | 二次情報 | それまでの話の要点をまとめる発話（gist）と、そこからの帰結を引き出す発話（upshot） |
| **Examining Group Facilitation In Situ: The Use of Formulations in Facilitation Practice.** Group Decision and Negotiation (2018). https://link.springer.com/article/10.1007/s10726-018-9577-7 | 抄録・二次情報 | ファシリテーターは要約の発話を使って、活動の開始・整理・締めくくりを行う |
| Weiser, M., Brown, J. S. (1996). **Designing Calm Technology.** PowerGrid Journal. / Mankoff, J. et al. (2003). **Heuristic Evaluation of Ambient Displays.** CHI 2003. | 二次情報 | 注意の中心と周辺を行き来できる、注意を奪わない情報提示の考え方 |
| Simons, D. J., Franconeri, S. L., Reimer, R. L. (2000). **Change Blindness in the Absence of a Visual Disruption.** Perception 29(10). | 抄録・二次情報 | ゆっくりした変化は、目の前で起きていても気づかれにくい（急な変化だけが注意を引く） |
| Matthews, T., Dey, A. K., Mankoff, J., Carter, S., Rattenbury, T. (2004). **A Toolkit for Managing User Attention in Peripheral Displays.** UIST 2004. | 二次情報 | 周辺表示での通知の強さを段階に分けて設計する（気づかれなくてよい変化から、注意を要求する通知まで） |

## 14. 立場（同意・不同意）の判定

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Galley, M., McKeown, K., Hirschberg, J., Shriberg, E. (2004). **Identifying Agreement and Disagreement in Conversational Speech: Use of Bayesian Networks to Model Pragmatic Dependencies.** ACL 2004. https://aclanthology.org/P04-1085/ | 抄録・二次情報 | ICSI 会議で同意・不同意・相槌・その他を分類し、正解率 86.9%。発話の対（隣接対）と過去の同意関係を手がかりにする |
| Hillard, D., Ostendorf, M., Shriberg, E. (2003). **Detection of Agreement vs. Disagreement in Meetings: Training with Unlabeled Data.** HLT-NAACL 2003. https://aclanthology.org/N03-2012/ | 抄録 | 語と韻律の手がかりによる会議の同意・不同意の分類 |
| Germesin, S., Wilson, T. (2009). **Agreement Detection in Multiparty Conversation.** ICMI-MLMI 2009. https://dl.acm.org/doi/10.1145/1647314.1647319 | 未読 | AMI での同意検出 |
| Pomerantz, A. (1984). **Agreeing and Disagreeing with Assessments: Some Features of Preferred/Dispreferred Turn Shapes.** In *Structures of Social Action*. Cambridge University Press. | 二次情報 | 同意は短く即座に、不同意は前置き・言いよどみを伴って遅れて言われやすい |
| Maynard, S. K. (1986). **On Back-Channel Behavior in Japanese and English Casual Conversation.** Linguistics 24. / Kita, S., Ide, S. (2007). **Nodding, Aizuchi, and Final Particles in Japanese Conversation.** Journal of Pragmatics 39(7). | 二次情報 | 日本語の相槌は継続の合図・聞いていることの表示であり、同意とは限らない |
| **MT²-CSD: A New Dataset and Multi-Semantic Knowledge Fusion Method for Conversational Stance Detection.** arXiv:2506.21053 (2025). https://arxiv.org/abs/2506.21053 | 抄録 | 多段の会話での立場判定。LLM を用いた手法でも平均 F 値 54% |
| **C-MTCSD: A Chinese Multi-Turn Conversational Stance Detection Dataset.** arXiv:2504.09958 (2025). https://arxiv.org/pdf/2504.09958 | 抄録・二次情報 | LLM でも暗黙の文脈の手がかりを捉えにくく、ゼロショットで F1 最高 64% |

## 15. 決定の判定と合意の確認

| 文献 | 確認範囲 | 本設計で使った内容 |
|---|---|---|
| Kaner, S. et al. (2014). **Facilitator's Guide to Participatory Decision-Making** (3rd ed.). Jossey-Bass. | 二次情報 | 合意の確認と決定ルールの明確化。賛否を段階で表す Gradients of Agreement（留保つきの支持などを含む） |
| Inoue, S. et al. (2022). **Meeting Decision Tracker: Making Meeting Minutes with De-Contextualized Utterances.** AACL-IJCNLP 2022 Demo. https://arxiv.org/abs/2210.11374 | 抄録 | 決定発話の検出と、文脈なしで読めるように書き換える処理 |
| **GADR: Gathering Architecture Decision Records from Meeting Transcriptions.** arXiv:2608.17694 (2026). https://arxiv.org/html/2608.17694v1 | 抄録・二次情報 | 会議の決定は暗黙・断片的で、関係ない会話と混ざっている。1回の LLM 呼び出しでは精度が落ちる |
| **Towards Group Decision Support with LLM-based Meeting Analysis.** ACM (2025). https://dl.acm.org/doi/10.1145/3708319.3733646 | 抄録・二次情報 | LLM で議論の選択肢・結果・決定の推移を追跡する試み |
