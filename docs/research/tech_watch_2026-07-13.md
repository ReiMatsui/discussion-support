# Tech Watch 2026-07-13

> 作成: Claude Fable 5 (claude-fable-5) 配下の調査エージェント, 2026-07-13
> 作成者注記: 本ファイルはエージェント（技術調査員）が2026-07-13時点のWeb情報をもとに新規作成した。以降のセクションは対象領域ごとに追記される想定。

## A. 音声系

対象システム: 対面グループ議論をライブ文字起こしし（現在Soniox STT＋声紋照合ECAPA/ReDimNet＋pyannote Live-1検証中）、AIファシリテーターが音声で介入する（OpenAI Realtime API gpt-realtime-2.1使用）研究プロトタイプ。日本語対応が必須という前提で調査した。

### A-1. リアルタイムSTT

#### Soniox（現状: stt-rt-v5 使用中）
- 提供元: Soniox
- 公開時期: v5 Real-Timeは2026年にリリース済み。旧stt-rt-v4は2026年6月30日に廃止予定（v5へ自動ルーティング、API変更不要）
- 要点: 「reinvented speaker separation」を謳い、会話の流れ・音響情報・文脈を使ってリアルタイムで話者変化を検出する話者分離内蔵機能を強化。60言語以上対応、雑音・電話音声・遠隔マイク・訛り・重複発話に対する頑健性向上、構造化データ（数字・日付・メール・氏名等）の認識精度も改善
- 価格: pricingページに詳細あり（具体額は要フォロー）
- 日本語対応: 60言語以上に含まれる想定（個別WER未確認）
- 評価: **本質改善**。現在使用中のv5がまさに最新であり、話者分離の内蔵強化は本システムの声紋照合パイプラインと直接競合・補完しうる。v4廃止スケジュールに注意し、v5の話者分離機能とpyannote Live-1の役割分担を再設計する価値あり
- 出典: https://soniox.com/docs/stt/models , https://soniox.com/blog/soniox-v5-real-time , https://soniox.com/pricing , https://soniox.com/docs/stt/rt/real-time-transcription

#### Deepgram (Nova-3)
- 提供元: Deepgram
- 公開時期: Nova-3は既存モデル、2026年時点で最新の主力ストリーミングモデル
- 要点: 日本語を含む10言語のコードスイッチング（言語切替）をリアルタイム認識。WebSocketストリーミングでE2Eレイテンシ200-300ms。ストリーミングでも diarize_model=latest / v1 でオンライン話者分離が利用可能（話者数の事前指定不要）。競合比でWER大幅改善（ストリーミング54.2%減、バッチ47.4%減、との自社主張）
- 価格: brasstranscriptsの試算でNova-3は約$0.46/hr（実勢は要件次第）
- 日本語対応: あり（コードスイッチング含む主要10言語の一つ）
- 評価: **様子見〜本質改善**。ストリーミング内蔵diarizationは魅力的だが、日本語での多話者・重複発話精度の実測データが乏しい。Sonioxとの直接比較ベンチマークを実施する価値はある
- 出典: https://deepgram.com/product/speech-to-text , https://deepgram.com/learn/introducing-nova-3-speech-to-text-api , https://developers.deepgram.com/docs/diarization , https://brasstranscripts.com/blog/deepgram-pricing-per-minute-2025-real-time-vs-batch

#### AssemblyAI Universal-Streaming
- 提供元: AssemblyAI
- 公開時期: Multilingual Universal-Streaming発表済み、2026年にstreaming diarizationの「major upgrade」実施
- 要点: **重要な制約** — Multilingual Universal-Streamingモデルは英・西・仏・独・伊・葡の6言語のみで、**日本語は現時点で非対応**。日本語が必要な場合は別モデル（Universal-3 Pro Streamingなど単一言語向け）を使う必要がある。ストリーミング話者分離は同一WebSocket上でspeaker_labels:trueを付けるだけで利用可能、ミリ秒単位でSPEAKER_A/B等を即時付与。2026年のアップグレードで「attribution精度を大幅改善」とアナウンス
- 価格: Universal-Streamingは$0.15/hr（全言語共通）、diarizationアドオンが$0.06/hr
- 日本語対応: Multilingual Universal-Streamingでは**非対応**。別モデル経由なら対応の可能性あるが要確認
- 評価: **不要（現時点）**。日本語がマルチリンガル最新モデルの対象外である点が本システムの必須要件と相容れない。ただし単一言語streaming diarizationの精度改善技術（attribution方式）は参考にする価値あり
- 出典: https://www.assemblyai.com/blog/introducing-multilingual-universal-streaming , https://www.assemblyai.com/pricing , https://www.assemblyai.com/blog/streaming-diarization-major-upgrade , https://www.assemblyai.com/docs/faq/can-i-use-speaker-diarization-with-live-audio-transcription

#### Speechmatics
- 提供元: Speechmatics
- 公開時期: 継続更新、2026年時点でJapanese STTモデルは96%語単位精度を主張
- 要点: リアルタイム・バッチ両対応、日本語専用モデルで高精度（Whisper・Deepgramを上回るとの自社主張）。Flowプラットフォーム上でリアルタイム話者分離、単語・文字レベルタイムスタンプ、オーディオイベントタギングに対応。雑音・重複発話・マイク品質変動への頑健性を強調
- 価格: 未確認（要問い合わせ）
- 日本語対応: 強み。専用モデルで高精度を主張
- 評価: **本質改善の可能性あり（要検証）**。日本語特化かつリアルタイム話者分離を統合提供する点がSonioxの代替・補完候補になりうる。自社ベンチマークのみのため第三者評価・自前ベンチマークが必要
- 出典: https://www.speechmatics.com/speech-to-text/japanese , https://docs.speechmatics.com/speech-to-text/realtime/realtime-diarization , https://docs.speechmatics.com/speech-to-text/features/diarization

#### Google Gemini Live API / Gemini系
- 提供元: Google
- 公開時期: Gemini 2.5 Flash Native AudioがVertex AIでGA、Gemini APIでプレビュー。gemini-3.1-flash-live-preview等も存在
- 要点: 音声→音声のE2Eネイティブオーディオアーキテクチャ（単一モデルでaudio in/out）。70言語対応、翻訳文脈では97言語。日本語の音声例が公式ドキュメントに明記
- 価格: 未確認
- 日本語対応: あり（native audioでプロソディ維持）
- 評価: **様子見**。STT単体というより音声対話モデルに近く、本システムのSTTパイプラインへの直接組み込みには不向き。ただしAIファシリテーター音声介入部分（gpt-realtime-2.1の代替候補）としては要検討
- 出典: https://ai.google.dev/gemini-api/docs/live-api , https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai , https://blog.google/products-and-platforms/products/gemini/gemini-audio-model-updates/

#### Whisper系ストリーミングOSS
- 提供元: OSS各種（collabora/WhisperLive, ufal/whisper_streaming, whisper.cpp, faster-whisper等）
- 公開時期: 継続更新。whisper_streamingは2025年時点で「SimulStreamingに置き換えられつつある」とされ、やや陳腐化傾向
- 要点: whisper.cpp・faster-whisperで実用的なリアルタイム化が可能（レイテンシ0.5〜2秒、モデルサイズ依存）。Apple Silicon上ではwhisper.cppがCore ML/Metal最適化でlarge-v3が約10倍速（M5 Pro）。VAD内蔵でストリーミング風運用が可能
- 価格: 無料（自前ホスティングコストのみ）
- 日本語対応: Whisperモデル自体は多言語対応だが、リアルタイム系OSSラッパーは精度・レイテンシのトレードオフがあり、商用STT（Soniox/Speechmatics等）に劣る場合が多い
- 評価: **不要（本番用途としては）**。オフライン・プライバシー重視や低コスト実験用途では価値があるが、現行の商用STT（Soniox）に対する精度・話者分離統合面での優位性は薄い。研究のバックアップ/比較ベースラインとしての価値はある
- 出典: https://github.com/collabora/WhisperLive , https://github.com/ufal/whisper_streaming , https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026

### A-2. 話者分離・声紋

#### pyannote Live-1（現状: 検証中）
- 提供元: pyannoteAI
- 公開時期: 2026年にGA（ベータ期間中は無料提供）
- 要点: ライブ音声パイプライン専用に設計されたストリーミング話者分離モデル。WebSocket経由で16kHzモノラル音声を100msチャンクで送信、diarization_speaker_start/end イベントをタイムスタンプ付きで返す。最大8話者・最大5時間/ストリームに対応。精度はバッチ版Precision-2相当を謳う
- 価格: ベータは無料、GA後の価格体系は要確認（pricingページ参照）
- 日本語対応: 言語非依存の音響ベース話者分離のため理論上対応可能だが、日本語特有の重複発話・相槌への実測評価は未確認
- 評価: **本質改善（継続検証中のため妥当な選択）**。本システムで既に検証を進めている方向性は最新の業界動向と一致。8話者上限・5時間制限が実運用要件（グループ議論の人数・長さ）に合うか要確認
- 出典: https://www.pyannote.ai/changelog/streaming-diarization-beta , https://www.pyannote.ai/changelog , https://www.pyannote.ai/pricing

#### AssemblyAI Streaming Diarization（競合として）
- 上記A-1参照。2026年のmajor upgradeで「attribution精度改善」を主張。ただし日本語マルチリンガルモデル非対応が障壁
- 評価: **不要（日本語制約のため）**

#### Deepgram Streaming Diarization（競合として）
- 上記A-1参照。diarize_model=latest/v1でストリーミング話者分離、話者数事前指定不要
- 評価: **様子見**。Soniox/pyannoteとの精度比較ベンチマークの価値はあるが、乗り換えの決定打となる情報は未確認

#### NVIDIA NeMo / Streaming Sortformer
- 提供元: NVIDIA
- 公開時期: 2025年8月にStreaming Sortformerとして発表、NeMo/Rivaに統合済み
- 要点: Arrival-Order Speaker Cache（AOSC）により到着順で話者ラベルを維持しながらフレームレベルでリアルタイム話者分離。2〜4話者以上のトラッキングに対応、オープンソースかつ本番グレード。英語最適化だが中国語(Mandarin)・CALLHOME非英語セットでもテスト済み
- 価格: オープンソース（NeMo/Rivaのインフラコストのみ）
- 日本語対応: **未検証**。英語最適化モデルであり日本語での性能は不明、自前ファインチューニングが前提になる可能性が高い
- 評価: **様子見**。セルフホスト・オープンソースという特性はコスト面で魅力的だが、日本語での精度実証がない状態でpyannote Live-1から乗り換える根拠は薄い。GPU運用コストも考慮要
- 出典: https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/ , https://arxiv.org/pdf/2507.18446 , https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2

#### 話者埋め込みモデル（ECAPA/ReDimNetより新しいもの）
- 提供元: 各研究機関（学術ベース）
- 公開時期: 継続研究、2026年時点でも決定的な後継の商用主流化は未確認
- 要点: WavLM-ECAPA（自己教師あり表現WavLM-Large＋ECAPA-TDNNのカスケード、約101.1Mパラメータ）が頑健性・汎化性能で優位という報告あり。ECAPA2（ECAPA-TDNNの改良版）も存在。ReDimNet（約15Mパラメータ、次元再構成による軽量高精度）は既知。ただし「決定的に置き換える新モデル」というよりは、自己教師あり表現とECAPA系のハイブリッドが2026年も主流という状況
- 価格: 研究モデルのためOSS前提
- 日本語対応: 音響ベースのため言語非依存だが、日本語話者データでのファインチューニング・評価は個別に必要
- 評価: **様子見**。ECAPA/ReDimNetから明確に置き換えるべき決定版は現時点で見当たらない。WavLM-ECAPAは頑健性向上の候補として実験対象にする価値はあるが、パラメータ数が大きくレイテンシ面での検証が必要
- 出典: https://www.emergentmind.com/topics/wavlm-ecapa-tdnn-architecture , https://arxiv.org/pdf/2407.11365 , https://www.isca-archive.org/interspeech_2025/ferrofilho25_interspeech.pdf

### A-3. リアルタイム音声対話モデル

#### OpenAI Realtime API — gpt-realtime-2.1（現状: 使用中）
- 提供元: OpenAI
- 公開時期: 2026年7月6日にgpt-realtime-2.1 / gpt-realtime-2.1-miniをリリース（本調査日の1週間前、まさに最新）
- 要点: GPT-Realtime-2からの更新で、アルファニューメリック認識・無音/ノイズ処理・割り込み挙動を改善。p95レイテンシを改善キャッシュにより25%以上削減。reasoning effort（minimal/low/medium/high/xhigh）を設定可能、tool useも強化。miniは低コスト・低レイテンシ、通常版は高性能
- 価格: 2.1-mini: 音声入力$10.00 / 出力$20.00、2.1（通常）: 音声入力$32.00 / 出力$64.00（100万トークンあたり想定、要確認）
- 日本語対応: Realtime API自体は多言語対応（具体的な日本語精度の第三者評価は未確認）
- 評価: **本質改善（既に採用中で妥当）**。本システムが使用中のバージョンがまさに最新リリース。割り込み処理・レイテンシ改善は「AIファシリテーターが自然に介入する」という要件に直結するため、アップデートの効果を実測評価する価値がある
- 出典: https://www.marktechpost.com/2026/07/06/openai-gpt-realtime-2-1-mini-reasoning-realtime-api/ , https://developers.openai.com/api/docs/models/gpt-realtime-2.1 , https://community.openai.com/t/new-realtime-models-on-the-api-gpt-realtime-2-1-and-gpt-realtime-2-1-mini/1385896

#### GPT-Live / GPT-Live-1
- 提供元: OpenAI
- 公開時期: 2026年7月8日発表（gpt-realtime-2.1のわずか2日後）。ChatGPTコンシューマー向け（iOS/Android/ChatGPT.com）でのみ提供開始
- 要点: フルデュプレックス（全二重）音声モデルファミリー。GPT-5.6ロールアウトと同時展開。API提供は「計画中」で、開発者は openai.com/form/gpt-live-1-in-the-api で通知登録可能な段階。現時点でAPIはGA（一般提供）されていない
- 価格: 未定（API未提供のため）
- 日本語対応: 詳細未確認
- 評価: **様子見（要フォロー）**。全二重対話という本システムのファシリテーター介入要件に理論上非常に合致するが、API未提供のため現時点で採用不可。API公開時期の継続監視が必要
- 出典: https://openai.com/index/introducing-gpt-live/ , https://openai.com/form/gpt-live-1-in-the-api/ , https://siliconangle.com/2026/07/08/openai-launches-gpt-live-voice-model-series-ahead-broad-gpt-5-6-release/ , https://www.buildfastwithai.com/blogs/gpt-live-review-openai-voice-model-july-2026

#### Google Gemini Live API
- 提供元: Google
- 公開時期: Gemini 2.5 Flash Native AudioがVertex AIでGA、Gemini APIでプレビュー中。gemini-3.1-flash-live-preview等も展開
- 要点: 音声のみのネイティブE2Eアーキテクチャ、70言語対応・翻訳文脈で97言語。日本語の使用例あり。全二重・割り込み対応は明記されているがOpenAI Realtimeとの直接比較データは限定的
- 価格: 未確認
- 日本語対応: あり
- 評価: **様子見**。gpt-realtime-2.1からの乗り換えを正当化する決定的優位性（日本語精度・割り込み自然さ）の実証データが不足。並行検証の価値はある
- 出典: https://ai.google.dev/gemini-api/docs/live-api , https://cloud.google.com/blog/topics/developers-practitioners/how-to-use-gemini-live-api-native-audio-in-vertex-ai

#### Amazon Nova Sonic（Nova 2 Sonic）
- 提供元: AWS
- 公開時期: Nova SonicからNova 2 Sonicへ進化、Bedrock経由で提供
- 要点: 音声→音声基盤モデル、ポリグロット音声、モーダル切替（音声/テキスト）、非同期tool use、最大1Mトークンのコンテキスト
- 価格: Bedrock課金体系
- 日本語対応: **非対応**。公式には英・西・独・仏・伊のみサポート、日本語は「AWS担当者にロードマップを問い合わせ」というステータス
- 評価: **不要**。日本語必須という本システムの要件を満たさないため現時点で検討外
- 出典: https://aws.amazon.com/blogs/aws/introducing-amazon-nova-2-sonic-next-generation-speech-to-speech-model-for-conversational-ai/ , https://docs.aws.amazon.com/nova/latest/userguide/speech.html

#### OSS全二重音声対話（Moshi / Kyutai）
- 提供元: Kyutai Labs
- 公開時期: 継続更新（元論文2024年10月、GitHubで継続メンテナンス）
- 要点: テキスト-音声基盤モデルによる真の全二重対話。Mimiという音声コーデックによりレイテンシ理論値160ms・実測200ms。Inner Monologue方式（音声トークンの前にテキストトークンを予測）でストリーミングASR/TTSも同時実現。CC-BY 4.0（モデル）/ Apache 2（推論コード）
- 価格: OSSで無料（自前ホスティングのGPUコストのみ）
- 日本語対応: **英語のみ**（2026年半ば時点）。多言語版は開発中とアナウンスされているが未リリース
- 評価: **不要（現時点）**。全二重・低レイテンシという設計思想は本システムの理想に近く技術的には注目に値するが、日本語非対応のため実運用投入は不可。多言語版のリリースを継続監視する価値はある
- 出典: https://github.com/kyutai-labs/moshi , https://arxiv.org/html/2410.00037v2 , https://localaimaster.com/blog/moshi-realtime-speech-guide

### A-まとめ: 導入検討に値するトップ3

1. **gpt-realtime-2.1の更新内容の実測評価（継続採用・チューニング）** — 2026年7月6日リリースの最新版であり、既に採用中。割り込み挙動・レイテンシ改善（p95で25%以上）がファシリテーター介入の自然さに直結するため、アップデート後の主観評価・遅延計測を優先実施すべき。追加コスト・移行コストがほぼゼロで即効性がある。

2. **Speechmatics 日本語STT＋リアルタイム話者分離のベンチマーク**（Soniox併用/代替候補） — 日本語専用モデルで96%語単位精度を主張し、Flow上でリアルタイム話者分離・単語/文字レベルタイムスタンプまで統合提供。Sonioxの声紋照合パイプラインと役割が重なるため、コスト・精度両面での直接比較を行う価値が高い。

3. **Soniox v5 Real-Timeの内蔵話者分離機能とpyannote Live-1/ECAPA-ReDimNet構成の役割再設計** — 現在使用中のSonioxが「reinvented speaker separation」を謳う新世代へ移行済み（v4は2026/6/30廃止）。STT側で話者分離が強化されたことで、pyannote Live-1＋声紋照合との二重構成が冗長になっていないか、あるいは相互補完（Sonioxの発話区間検出＋pyannote/ECAPAの声紋確定）として最適化できるかを検証する価値が高い。

**様子見だが継続監視すべき項目**: OpenAI GPT-Live-1のAPI提供開始（全二重ファシリテーターの本命候補）、Kyutai Moshiの多言語（日本語）版リリース、Google Gemini Liveの日本語割り込み性能の実測比較。

**現時点で不要と判断した項目**: AssemblyAI Universal-Streaming（日本語非対応）、Amazon Nova Sonic（日本語非対応）、Moshi（英語のみ）、Whisper系OSSストリーミング（商用STTに対する精度・統合面での優位性なし）。

---

## B. LLM・知識・評価系

対象システム: 対面グループ議論の発話をLLMでリアルタイムに議論グラフ（AF: claim/premise/evidence＋支持/攻撃エッジ）へ変換し、AIファシリテーターが介入する研究プロトタイプ（修士研究）。LLMはOpenAI API（gpt-5.4-mini等）、extraction/linking/facilitation/web_searchの多エージェント構成、評価はLLM-as-judge＋決定的指標。

### B-1. LLM API の新機能・新モデル

#### OpenAI GPT-5.5 / GPT-5.6（Sol/Terra/Luna）
- 提供元: OpenAI
- 公開時期: GPT-5.5は2026年前半、GPT-5.6は2026年6月（Sol/Terra/Lunaの3ティア構成）
- 要点: GPT-5.5は$5/$30（入力/出力、100万トークンあたり、キャッシュ入力$0.50）。GPT-5.6は3ティア化され、Sol（$5/$30）、Terra（$2.5/$15）、Luna（$1/$6）。いずれも272Kトークン超で価格が上振れするブレークポイント制。structured outputs継続サポート。
- 価格: 上記の通り。GPT-5.4-mini（現行使用モデル系列）は$0.75/$4.50で、高頻度・低単価タスク向けに位置づけられている。
- 本システムへの評価: **本質改善**。extraction/linkingのような高頻度低単価タスクには、Luna（$1/$6）やGPT-5.4-mini系が実質的なコスト半減〜1/5を実現できる可能性がある。ただし抽出精度の劣化リスクがあるため、gpt-5.4-mini→Lunaへの置換はA/Bでの精度検証が必須。facilitation（介入判断）は品質要求が高いためSol/Terra級を維持すべき。
- 出典: https://www.aipricing.guru/openai-pricing/ , https://www.finout.io/blog/gpt-5.6-pricing-2026-sol-terra-and-luna-tiers-explained , https://developers.openai.com/api/docs/models/gpt-5.5

#### OpenAI Batch API / Prompt Caching
- 提供元: OpenAI
- 公開時期: 継続提供、2026-07-02時点でo3等の価格改定あり
- 要点: Batch APIは全モデル一律で入出力とも50%割引（24時間以内の非同期処理）。Prompt Cachingはキャッシュ入力が通常価格の10%（GPT-5.4は$0.25/M）。両者併用でキャッシュ入力トークンは通常比75%減の実績あり。
- 価格: 上記の通り。2026-07-02にo3が$10/$40→$2/$8に大幅値下げ。
- 本システムへの評価: **本質改善**。修論のシミュレーション実験（大量ラン）はリアルタイム性を要求しないため、Batch APIで実験コストを半減できる。またextraction/linkingは議論の文脈（直前の発話・グラフ状態）を繰り返しプロンプトに含める設計であれば、Prompt Cachingの適用余地が大きい。プロンプト構造を「固定プレフィックス（システム指示・グラフスキーマ）＋可変部（新規発話）」に再設計すればキャッシュヒット率が上がり大幅減額が見込める。
- 出典: https://benchlm.ai/blog/posts/openai-api-pricing , https://tokenmix.ai/blog/openai-batch-api-pricing , https://devtoollab.com/blog/prompt-caching-guide

#### Anthropic Claude（Sonnet 5 / Haiku 4.5 / Opus 4.8 / Fable 5）
- 提供元: Anthropic
- 公開時期: Claude Sonnet 5は2026-06-30公開（キャンペーン価格$2/$10、2026-09-01より$3/$15に復帰）。Haiku 4.5は$1/$5で低価格帯の主力。
- 要点: Haiku 4.5はプロンプトキャッシュで最大90%減、Batchで50%減が可能。Sonnet 5は価格性能比を訴求。
- 価格: 上記の通り。
- 本システムへの評価: **様子見〜部分導入検討**。現行はOpenAI一本化構成だが、extraction/linkingのような定型タスクをHaiku 4.5に切り出せば、OpenAIより低単価になりうる（$1/$5 vs Luna $1/$6は僅差だがキャッシュ割引率がHaikuの方が大きい）。ただしAPI混在は運用複雑性・エージェント間整合性（構造化出力フォーマット差異）のコストを伴うため、修論の期間内では「将来課題」として言及するに留め、本実装は現行API単一化を維持するのが妥当。
- 出典: https://www.aipricing.guru/anthropic-pricing/ , https://www.anthropic.com/claude/haiku , https://platform.claude.com/docs/en/about-claude/pricing

#### Google Gemini Flash系
- 提供元: Google
- 公開時期: Gemini 3.5 Flash（2026-05-19公開）、Gemini 3 Flash Preview、Gemini 3.1 Flash Lite Preview
- 要点: Gemini 3.5 Flashは$1.50/$9.00で、コーディング・エージェント系ベンチマークでGemini 3.1 Proを上回る性能かつ高速（214 tok/s）。Gemini 3 Flash Previewは$0.50/$3.00、3.1 Flash Lite Previewは$0.25/$1.50とさらに安価。
- 価格: 上記の通り。
- 本システムへの評価: **様子見**。Flash Lite Preview（$0.25/$1.50）は現行のgpt-5.4-mini（$0.75/$4.50）より大幅に安く、extraction/linkingのバッチ実験用途では有力候補。ただしPreview版は安定性・API仕様変更リスクがあり、修論の再現性要件（評価の一貫性）と相性が悪い。本実装への即時採用は非推奨、将来の代替候補として記録に留める。
- 出典: https://pricepertoken.com/pricing-page/model/google-gemini-3.5-flash , https://pricepertoken.com/pricing-page/model/google-gemini-3.1-flash-lite-preview , https://www.tldl.io/resources/google-gemini-api-pricing

### B-2. リアルタイム知識検索（web_searchエージェント改善候補）

#### OpenAI Web Search Tool（Responses API）
- 提供元: OpenAI
- 公開時期: 継続更新。`return_token_budget`パラメータ追加（長時間の高精度リサーチ向け）
- 要点: 新規統合では`{"type": "web_search"}`を使用すべきで、旧`web_search_preview`は`filters`・`external_web_access`・`return_token_budget`等の新機能が使えない。ChatGPT検索と同じモデルで駆動。
- 本システムへの評価: **本質改善（軽微だが対応必須）**。現行のweb_searchエージェントが`web_search_preview`を使用している場合、`web_search`への切替でfilters機能（ドメイン絞り込み等）が使えるようになり、ファクトチェック精度・レイテンシ管理が改善する可能性。まず現行実装のツール指定を確認すべき。
- 出典: https://developers.openai.com/api/docs/guides/tools-web-search , https://developers.openai.com/api/docs/changelog

#### Perplexity API / Exa / Tavily 比較
- 提供元: Perplexity、Exa、Tavily（Nebius傘下）
- 公開時期: 各社継続運用、2026年時点の比較記事複数
- 要点: 小規模ベンチマーク（8問）ではPerplexity API＞Exa＞Gemini＞Tavilyの精度順。Exaは神経埋め込みによる意味検索が特徴で「概念検索」に強い。価格は月10万クエリでTavily約$800、Exa約$450-500、Perplexity Sonar約$500。Perplexity APIはレート制限50 req/minで本番運用に課題との指摘あり。
- 本システムへの評価: **様子見**。現行web_searchエージェントの実装内容（OpenAI組込みtool利用か外部API利用か）を要確認。もし外部APIを使うなら、議論のファクトチェック用途では「概念検索」に強いExaが根拠資料探索に向く可能性があるが、レート制限やコストを考慮すると、リアルタイム対面議論（発話頻度が高い）用途にはPerplexity APIのレート制限がボトルネックになりうる。当面はOpenAI組込みweb_search toolの継続利用が無難。
- 出典: https://www.humai.blog/perplexity-vs-tavily-vs-exa-vs-you-com-the-complete-ai-search-engine-comparison-2026/ , https://serp.fast/guides/ai-search-apis-compared

### B-3. 議論マイニング・AF系

#### LLMベース議論マイニングのパラダイムシフト（サーベイ）
- 提供元: 学術（arXiv、複数著者）
- 公開時期: 2025-06〜2026（"Large Language Models in Argument Mining: A Survey" arXiv:2506.16383、"An LLM-Based System for Argument Mining" arXiv:2605.13793、"Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection" arXiv:2603.03095）
- 要点: 議論マイニングは教師ありの専用分類器パイプラインから、プロンプト駆動・RAG・推論志向のLLMベース手法へ移行中。長文脈推論、多言語・マルチモーダル頑健性、解釈可能性、低コスト運用が今後の課題として指摘されている。LLaMA-3/Gemma-2/Mistral/Phi-3/Qwen-2等の量子化モデルをPE/AbstRCT/CDCPデータセットでファインチューニングする研究も進行。
- 本システムへの評価: **本質改善（先行研究としての位置づけを強化）**。本研究のextraction/linkingエージェント設計（LLMによるclaim/premise/evidence抽出＋支持/攻撃エッジ）は、まさにこのサーベイが指摘する「プロンプト駆動パラダイム」の実例。関連研究セクションでarXiv:2506.16383やarXiv:2605.13793を引用することで研究の位置づけが強化できる。特にCompact Prompting論文（軽量プロンプトでの構成要素検出）は、gpt-5.4-miniでの低コスト抽出プロンプト設計に直接応用できる可能性があり、要精読。
- 出典: https://arxiv.org/abs/2506.16383 , https://arxiv.org/html/2605.13793 , https://arxiv.org/pdf/2603.03095

#### AF-Xray（法的議論フレームワークの可視化・explainability）
- 提供元: 学術（arXiv:2507.10831、PyArg基盤）
- 公開時期: 2025年7月
- 要点: AFの解（extension）の探索・分析・可視化を行うプラットフォームで、オープンソースのPyArg上に構築。曖昧性の視覚的説明・解消を目的とする。
- 本システムへの評価: **不要（直接導入は非該当だが概念参照は有用）**。本研究はリアルタイム対面議論向けの軽量可視化が目的であり、法的議論フレームワークの厳密なextension計算基盤（PyArg）は要件が異なる（形式的AF意味論の厳密実装よりも、LLMが生成する近似的グラフのリアルタイム更新・UI表示が優先）。ただし「なぜこの発話が攻撃/支持と判定されたか」の説明可能性UIのデザイン参考にはなりうる。
- 出典: https://arxiv.org/pdf/2507.10831

#### 汎用グラフ可視化ライブラリ（Cytoscape.js等）
- 提供元: OSSコミュニティ
- 公開時期: Cytoscape.js継続メンテナンス（2023年更新論文が引き続き参照される）
- 要点: 専用AF可視化ツールは少なく、実務ではD3.js/GraphViz/Cytoscape.js/Vis.jsといった汎用グラフライブラリを転用するのが一般的。
- 本システムへの評価: **様子見**。現行の可視化実装（webapp）が何を使っているか要確認だが、専用AFツールへの乗り換えは不要。汎用ライブラリの継続利用で十分。
- 出典: https://cambridge-intelligence.com/blog/open-source-data-visualization/ , https://github.com/topics/argumentation-frameworks

### B-4. LLM評価（LLM-as-judgeバイアス対策・マルチエージェント議論シミュレーション）

#### LLM-as-Judgeバイアス体系化とアンサンブル評価
- 提供元: 学術＋実務（arXiv:2604.23178 "Judging the Judges"、FutureAGIブログ2026）
- 公開時期: 2026年前半
- 要点: 5種のバイアス（位置バイアス、冗長性バイアス、自己選好バイアス、フォーマットバイアス、キャリブレーションドリフト）が体系化され、それぞれに測定・緩和策が提示されている。ペア比較では順序ランダム化＋多数回集計、複数judge（例: Claude Sonnet 4.5・GPT-5.1・Gemini 2.5 Pro）のアンサンブル多数決が2026年5月時点のデフォルトプラクティスとして提案。人間との月次キャリブレーションを推奨。judgeコストは本番LLMコストの10-15%以内に抑えるべきとされる。
- 本システムへの評価: **本質改善**。現行のLLM-as-judge評価が単一モデル・単一プロンプトである場合、(1) 独立スコアリング方式への変更（ペア比較でなく各出力を独立にルーブリック採点）、(2) 複数モデル（例: gpt-5.4-mini系とは別系統のAnthropic/Geminiモデル）による簡易アンサンブルの追加、を修論の評価妥当性の脅威への対処として明記する価値が高い。特に自己選好バイアス（OpenAIモデルで生成した議論をOpenAIモデルで評価する構造的リスク）は本システムの評価設計に直結する論点であり、限界セクションでの言及が必須。
- 出典: https://arxiv.org/html/2604.23178v2 , https://futureagi.com/blog/llm-as-judge-best-practices-2026 , https://futureagi.com/blog/evaluating-llm-judge-bias-mitigation-2026/

#### マルチエージェント議論シミュレーションの新フレームワーク・ベンチマーク
- 提供元: 学術（DEBATE: arXiv:2510.25110、ARGORA: arXiv:2601.21533、"The Confident Liar": arXiv:2606.10296）
- 公開時期: 2025年10月〜2026年6月
- 要点: DEBATEは2,832名の実participant・107論題の実討論データから構築された、ロールプレイLLMエージェントの意見動態の「本物らしさ」を評価する大規模ベンチマーク。ARGORAは因果的に根拠づけられたLLM推論のための議論オーケストレーション。"The Confident Liar"はログ確率とLLM-as-judgeを組み合わせて多エージェント討論の診断（過信・欺瞞的主張の検出）を行う手法。
- 本システムへの評価: **本質改善**。本研究はシミュレーション実験（合成議論データでの多エージェント評価）を含むため、DEBATEベンチマークの手法論（実データとの整合性検証手法）は評価妥当性の参考になる。特に「LLMで生成した議論参加者の発話が実際の人間討論とどれだけ乖離するか」という論点は、本研究の評価章における妥当性の脅威（threats to validity）としてそのまま引用・議論可能。"The Confident Liar"のログ確率併用診断は、facilitationエージェントの介入判断の信頼度較正に応用できる可能性がある。
- 出典: https://arxiv.org/html/2510.25110 , https://arxiv.org/pdf/2601.21533 , https://arxiv.org/pdf/2606.10296

---

## B-まとめ: 導入検討に値するトップ3

1. **Prompt Caching対応のプロンプト再設計＋Batch APIによる実験コスト削減**（B-1）: extraction/linkingプロンプトを「固定プレフィックス＋可変部」に分離しキャッシュヒット率を上げる、かつシミュレーション大量ランはBatch APIで50%減。実装コストが低く即座に着手可能で、修論のコスト制約下での実験規模拡大に直結する。

2. **LLM-as-judge複数バイアス対策（独立スコアリング＋アンサンブル＋自己選好バイアスの明記）**（B-4）: 現行のLLM-as-judge評価設計に対し、arXiv:2604.23178の枠組みに沿った独立スコアリング化・複数モデルアンサンブルの部分導入、および評価の妥当性の脅威としての自己選好バイアスの明示的記述。研究の学術的信頼性を直接高める。

3. **議論マイニングサーベイ（arXiv:2506.16383, 2605.13793, 2603.03095）を関連研究として精読・引用**（B-3）: 本研究のextraction/linkingエージェント設計をLLMベース議論マイニングの最新パラダイムの中に明確に位置づけられ、かつCompact Prompting論文の軽量プロンプト手法はgpt-5.4-mini低コスト運用に直接転用できる可能性がある。
