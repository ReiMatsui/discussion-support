# 外部ツール選定レビュー（2026年7月時点）

対象リポジトリ: `/Users/matsuirei/discussion-support`
目的: 日本語対面会議のリアルタイム文字起こし＋話者特定＋LLMマルチエージェント介入という研究プロトタイプにおける外部ツール選定の妥当性評価。

---

## 1. 使用中ツール一覧

| ツール / モデル | 使用箇所 | 用途 | バージョン・設定 |
|---|---|---|---|
| **Soniox Real-Time STT** (`stt-rt-v5`) | `src/das/asr/live/stt/_soniox.py`, `_bootstrap.py`（既定STT） | メインのリアルタイム文字起こし＋話者分離＋セマンティックエンドポイント検出（WebSocket, PCM16 16kHz） | `--model stt-rt-v5` 既定、`enable_speaker_diarization: true` |
| **Speechmatics RT** | `src/das/asr/live/stt/_speechmatics.py` | 代替STTバックエンド（`--stt speechmatics`）。enhanced運用点＋話者分離＋partials | `eu.rt.speechmatics.com/v2`, `max_delay 1.2s` |
| **pyannoteAI live API** | `src/das/asr/live/_pyannote_diarization.py` | 外部ストリーミング話者分離プロバイダ（`--diarization pyannote`） | `api.pyannote.ai/v1/live` |
| **AssemblyAI Streaming v3** | `src/das/asr/live/_assemblyai_diarization.py` | 話者分離のみを外部プロバイダとして利用（`--diarization assemblyai`） | `speech_model=universal-3-5-pro`, `speaker_labels=true` |
| **声紋モデル: ReDimNet（既定）/ ECAPA / Resemblyzer** | `src/das/asr/live/_voice_profiles.py`, `_bootstrap.py` | 登録済み声紋プロファイルとの照合による話者特定 | ReDimNet は torch.hub、ECAPA は speechbrain 経由 |
| **WhisperLiveKit + mlx-whisper / faster-whisper** | `das listen` 系（`asr` extras）、`settings.py` | ローカル（オフライン）リアルタイム文字起こし | `DAS_ASR_BACKEND=mlx-whisper`, `DAS_ASR_MODEL=large-v3`, `ja` |
| **OpenAI `gpt-5-mini`** | `settings.py`（`OPENAI_MODEL_FAST`/`SMART` 既定）、`llm/openai_client.py` | 論証抽出・連結・ファシリテーションの全LLM呼び出し | Chat Completions + `beta.chat.completions.parse`（structured output） |
| **OpenAI `gpt-5-nano`** | `agents/linking.py`, `runtime/orchestrator.py` | Linking の判定呼び出しのコスト削減用 | オプション指定 |
| **OpenAI `gpt-realtime-2`** | `asr/live/_constants.py`, `agents/_realtime.py`, `_partner.py` | Realtime API（音声対話するAIファシリテーター/議論パートナー） | `OPENAI_REALTIME_MODEL` 既定 `gpt-realtime-2` |
| **OpenAI `text-embedding-3-small`** | `llm/openai_client.py` | 埋め込み（類似判定） | 固定値 |
| **Tavily** | `web` extras、WebSearchエージェント | リアルタイムWeb検索 → evidenceノード化 | `tavily-python` |
| **Streamlit** | `src/das/ui/streamlit_app.py`（`das ui`） | 議論グラフのビューア | `>=1.40` |
| **pyvis + networkx** | `src/das/viz/render.py` | 議論グラフのHTML可視化 | `pyvis>=0.3.2` |
| **websockets / sounddevice / http.server** | `asr/live` 全般、`_ui.py`, `_webapp.py` | WS通信は`websockets`直叩き、ライブUIは標準ライブラリ`http.server`（aiohttp等は不使用） | — |
| その他 | pydantic(-settings), tenacity, structlog, typer, tiktoken, opencv/mediapipe | 設定・リトライ・ログ・CLI・（映像系） | — |

---

## 2. 分野別: 2026年7月の状況と評価

### 2.1 日本語リアルタイムSTT＋話者ダイアライゼーション

**Soniox（現行のメイン）— 妥当**

- コード既定の `stt-rt-v5` は **2026年6月16日リリースの最新モデル**。`stt-rt-v4` は2026年6月30日に廃止済み（自動で v5 にルーティング）なので、既定値は最新に追随できている。話者分離の再設計・セマンティックエンドポイント高速化・日英コードスイッチ対応が v5 の売り。出典: [Soniox Models/Changelog](https://soniox.com/docs/stt/models), [Soniox v5 Real-Time blog](https://soniox.com/blog/soniox-v5-real-time)
- 価格はストリーミング **$0.12/時（話者分離・言語識別込み）**。Deepgram（分離が約$0.12/h追加）や Azure（約$0.30/h追加）より安価にオールインワン。出典: [Soniox Pricing](https://soniox.com/pricing), [Soniox Compare](https://soniox.com/compare)
- 第三者比較でも2026年時点の精度上位グループ（英語WERでSoniox 1.25% vs Deepgram 1.71% / AssemblyAI 1.74% とするベンダー系ベンチあり）。**日本語単体の独立ベンチマークは未確認**（「日本語を含む60+言語で高精度」「日英コードスイッチ対応」はベンダー主張）。出典: [futureagi ベンチまとめ](https://futureagi.substack.com/p/speech-to-text-apis-in-2026-benchmarks)
- 留意点: モデル世代交代が約4か月周期（v3→v4→v5）と速く、旧モデルは数か月で廃止される。既定値のハードコード（`_bootstrap.py:63` / `__init__.py:59`）は定期的な追随が必要。

**代替候補の2026年状況**

| 候補 | 日本語リアルタイム | リアルタイム話者分離 | 所見 |
|---|---|---|---|
| **AssemblyAI Universal-Streaming** | 多言語ストリーミングは **en/es/fr/de/it/pt の6言語のみで日本語非対応**（asyncのUniversal-3 Proは99言語で日本語対応）。出典: [AssemblyAI multilingual streaming](https://www.assemblyai.com/docs/streaming/universal-streaming/multilingual-transcription), [発表blog](https://www.assemblyai.com/blog/introducing-multilingual-universal-streaming) | あり（話者数誤り率2.9%を主張、Universal-3 Pro Streaming P50 307ms） | 英語では最有力級だが日本語ストリーミングが壁 |
| **Deepgram (Nova-3 / Flux)** | Nova-3 multilingualに日本語はあるが数値フォーマット非対応等の制限。Fluxは音声エージェント向けで最速のEOS検出 | 「リアルタイム話者分離は非英語で大きな問題がある」との第三者評価あり。出典: [softcery 14 STT/TTS比較](https://softcery.com/lab/how-to-choose-stt-tts-for-ai-voice-agents-in-2025-a-comprehensive-guide), [Coval独立ベンチ](https://www.coval.ai/blog/best-speech-to-text-providers-in-2026-independent-benchmarks-and-how-to-choose/) | 日本語×分離の組合せでは推奨しない |
| **ElevenLabs Scribe v2 Realtime** | 90+言語・初回パーシャル約150msで最速級 | **リアルタイム版は話者分離を意図的に省略**（バッチ版Scribe v2のみ32話者まで対応）。出典: [Scribe v2 Realtime](https://elevenlabs.io/realtime-speech-to-text), [Scribe v2ガイド](https://aividpipeline.com/blog/elevenlabs-scribe-v2-guide-2026) | 分離必須の本用途では単独採用不可 |
| **Speechmatics (Ursa 2)** | 日本語対応、リアルタイム話者分離あり（句読点で分離補正）。Ursa 2で55言語WER 18%改善を主張。出典: [Realtime diarization docs](https://docs.speechmatics.com/speech-to-text/realtime/realtime-diarization), [Ursa紹介](https://www.speechmatics.com/company/articles-and-news/introducing-ursa-the-worlds-most-accurate-speech-to-text) | 既にフォールバック実装済み。妥当な保険 |
| **国産（AmiVoice等）** | AmiVoice APIはストリーミング書き起こし＋業界別辞書＋話者分離対応。日本語特化では Rimo Voice 等も。出典: [AmiVoice API](https://acp.amivoice.com/en/amivoice_api/), [AmiVoiceストリーミング docs](https://docs.amivoice.com/en/amivoice-api/manual/tutorial-streaming-transcription/), [話者分離ツール比較(LINE WORKS)](https://line-works.com/ainote/column/speaker-diarization-tools-recommend/) | 日本語の同音異義語・専門用語には国産特化が強いという国内評あり。**リアルタイム話者分離の精度・レイテンシの独立比較は未確認**。研究の対照条件としてベンチする価値あり |
| Google / Azure / AWS | 日本語ストリーミング自体は対応 | 分離は追加課金・精度面で2024–2026世代の専業勢に劣後という比較が多い。出典: [Soniox vs Google](https://soniox.com/compare/soniox-vs-google)（ベンダー系）, [deepgram比較記事](https://deepgram.com/learn/best-speech-to-text-apis-2026) | 積極的に乗り換える理由は見当たらない |

**判定: Soniox 維持（妥当）**。日本語×リアルタイム×話者分離×低価格の同時要件を満たす競合が2026年7月時点でも少ない。Speechmaticsフォールバック併存も妥当。日本語WERの一次データが乏しいため、研究上は自前ベンチ（AmiVoice・Speechmaticsとの3点比較）を一度取っておくと選定根拠が論文に書ける。

### 2.2 話者分離プロバイダ（STT外付け）

- **pyannoteAI live API — 妥当**。pyannote系は2026年時点もOSS/ホスティング両面でSOTA圏: `pyannote.audio 4.0` + 新OSSモデル **Community-1**（2025末〜2026初）、商用APIは **Precision-2**（OSS 3.1比+28%精度）で、ホスティッドAPIはサブ150msレイテンシを主張。現在使用中の `api.pyannote.ai/v1/live` はその公式リアルタイムAPIで、選定は最新に沿っている。出典: [pyannoteAI](https://www.pyannote.ai/), [Community-1発表](https://www.pyannote.ai/blog/community-1), [pyannote/speaker-diarization-community-1 (HF)](https://huggingface.co/pyannote/speaker-diarization-community-1), [changelog](https://www.pyannote.ai/changelog)
- **AssemblyAI 分離専用プロバイダ — 再検討推奨**。実装は `universal-3-5-pro` + `speaker_labels` をストリーミングで叩くが、上記の通り **AssemblyAIの多言語ストリーミングは日本語非対応**（6言語のみ）。日本語音声を流した際に話者分離イベントが安定して得られるかは**未確認**（言語非依存に動く可能性はあるが公式保証を確認できず）。`universal-3-5-pro` というモデル名の公式ドキュメント上の日本語ストリーミング対応も**未確認**。日本語会議が主対象なら pyannoteAI 側を正とし、AssemblyAI 経路は実験用と割り切るべき。
- ローカル代替: WhisperLiveKit が採用する **Streaming Sortformer**（NVIDIA, 2025 SOTA）系のストリーミング分離もOSSで利用可能になっており、完全ローカル要件が出た場合の受け皿はある。出典: [WhisperLiveKit README](https://github.com/QuentinFuxa/WhisperLiveKit)

### 2.3 声紋（話者認識・スピーカー埋め込み）

- 既定の **ReDimNet**（Interspeech 2024）は2026年でも「軽量×高精度」のリファレンス的存在（1–15Mパラメータ、VoxCeleb系でSOTA級）。ECAPA-TDNN（speechbrain）併用のフォールバック設計も標準的。出典: [ReDimNet arXiv系まとめ](https://arxiv.org/pdf/2407.11365), [ECAPA-TDNN](https://arxiv.org/abs/2104.01466)
- 2026年時点でReDimNetを明確に置き換える決定版の新埋め込みモデルは**未確認**。**判定: 維持**。
- 補足: pyannoteAIの商用プラットフォームは「speaker identification（登録話者の同定）」もAPIで提供し始めており（出典: [Speaker Platform](https://www.pyannote.ai/speaker-platform)）、自前の声紋照合（`_voice_profiles.py` の閾値チューニング含む）を外部化できる可能性がある。研究の統制を自前に置きたいなら現状維持でよい。

### 2.4 ローカルASR（WhisperLiveKit / mlx-whisper）

- WhisperLiveKit は2026年も活発に開発中: SimulStreaming（2025 SOTA方針）、Streaming Sortformer分離、Voxtral Mini・Qwen3-ASR等の新バックエンド追加。mlx-whisper（Apple Silicon）経路も維持されている。出典: [GitHub](https://github.com/QuentinFuxa/WhisperLiveKit), [PyPI](https://pypi.org/project/whisperlivekit/), [解説記事(2026-05)](https://www.blog.brightcoding.dev/2026/05/30/whisperlivekit-self-hosted-speech-to-text-that-actually-works-in-real-time)
- 固定の `whisperlivekit>=0.2.20` は古い可能性が高い（依存漏れ回避の `python-multipart` 手当も上流で解消済みか要確認—**未確認**）。**判定: 維持（バージョン追随とバックエンド見直しを推奨）**。オフライン/プライバシー要件の受け皿として価値が高い。

### 2.5 LLM（OpenAI）

2026年7月時点のOpenAIラインアップ（出典: [OpenAI Models](https://developers.openai.com/api/docs/models), [aipricing.guru 2026-07-01集計](https://www.aipricing.guru/openai-pricing/), [pricepertoken GPT-5.4 mini](https://pricepertoken.com/pricing-page/model/openai-gpt-5.4-mini), [GPT-5.5発表](https://openai.com/index/introducing-gpt-5-5/), [GPT-5.6 Solプレビュー](https://openai.com/index/previewing-gpt-5-6-sol/)):

| 階層 | 2026年7月の現行 | 価格($/1M in/out) | 備考 |
|---|---|---|---|
| フラッグシップ | GPT-5.5（推奨）、GPT-5.6 Sol/Terra/Luna（限定プレビュー） | 5.5: $5/$30、5.6 Luna: $1/$6 | 5.6は一般提供前（プレビューのみ） |
| mini | **GPT-5.4 mini**（2026年3月〜） | $0.75/$4.50 | 「gpt-5.5-mini」「gpt-5.2-mini」は存在しない |
| nano | GPT-5.4 nano | $0.20/$1.25 | ルーティング・抽出向け |

- 使用中の **`gpt-5-mini` は2世代以上前**（GPT-5 → 5.1 → 5.2 → 5.4 → 5.5系列）。廃止情報は**未確認**（現時点でも呼べる可能性が高い）が、抽出・連結の精度/コスト比では GPT-5.4 mini / nano が現行水準。設定は環境変数（`OPENAI_MODEL_FAST`/`SMART`）で差し替え可能なので**移行コストはほぼゼロ**。ただし `llm/cost.py` の料金表と `_supports_custom_temperature()` のモデル名プレフィックス判定（`gpt-5` で一括）は新モデル名に合わせた更新が必要。
- **Realtime API: `gpt-realtime-2`（2026年5月7日リリース）を既に採用しており最新**。価格は音声$32/1M入力・$64/1M出力トークン。同時リリースの **gpt-realtime-whisper（$0.017/分の低遅延文字起こし）/ gpt-realtime-translate** は、STT一本化の代替候補になり得るが話者分離がない点でSonioxを置き換えない。出典: [OpenAI voice intelligence発表](https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/), [GPT-Realtime-2 model docs](https://developers.openai.com/api/docs/models/gpt-realtime-2), [9to5Mac報道](https://9to5mac.com/2026/05/07/openai-has-new-voice-models-that-reason-translate-and-transcribe-as-you-speak/)
- `text-embedding-3-small`: これを置き換える新埋め込みモデルの発表は**未確認**。維持で問題なし。
- SDK実装面: `client.beta.chat.completions.parse` の beta 名前空間は非推奨方向（`chat.completions.parse` がGA、さらにOpenAIはResponses APIへの移行を推奨）。動作はするが、SDKメジャーアップデートで壊れやすい箇所。

### 2.6 structured output / エージェント基盤

- 現状は自前の薄いラッパ（pydantic + `parse` + tenacity）。2026年の主要選択肢は **Pydantic AI**（型安全なstructured output特化）、**OpenAI Agents SDK**（セッション/ハンドオフ内蔵）、**LangGraph**（永続化・承認フロー付きグラフオーケストレーション）。出典: [2026フレームワーク比較](https://open-techstack.com/blog/langgraph-vs-openai-agents-sdk-vs-pydanticai-2026/), [langchain資料](https://www.langchain.com/resources/ai-agent-frameworks), [morphllm比較](https://www.morphllm.com/ai-agent-framework)
- 本プロジェクトはエージェント間の制御フローが自前オーケストレーター（研究の主題そのもの）なので、フレームワーク導入はブラックボックス化のデメリットが勝つ。**判定: 自前実装の維持が妥当**。将来プロバイダ非依存にしたければ Pydantic AI が最も移行親和的。

### 2.7 可視化（Streamlit / pyvis）

- Streamlit・pyvisともに研究プロトタイプのビューア用途としては2026年も標準的で問題なし。代替は Plotly **Dash + Dash Cytoscape**（本格的なグラフUI・リアルタイム更新）、**Gradio**（AIデモ特化）、Reflex/Panel など。出典: [Deepnote Streamlit代替2026](https://deepnote.com/compare/alternatives/streamlit), [Reflex比較](https://reflex.dev/blog/streamlit-vs-dash-python-dashboards/), [Plotly blog](https://plotly.com/blog/best-streamlit-alternatives-production-data-apps/)
- pyvis は開発が低頻度（0.3.x が長い）だが静的HTML出力用途では実害なし。ライブ介入UIは既に自前 `http.server` + WebSocket で別実装しており、Streamlitへのリアルタイム要件はない。**判定: 維持**。大規模グラフの操作性が課題になったら Dash Cytoscape / Cytoscape.js への移行を検討。

### 2.8 その他

- **Tavily**（Web検索→evidence化）: LLM向け検索APIとして2026年も一般的。深掘り調査は未実施（本用途で問題の兆候なし）。**維持**。
- **websockets / sounddevice / http.server**: 外部サービス依存なし・軽量で研究用途に適切。**維持**。

---

## 3. 推奨アクション一覧

| ツール | 判定 | 理由 / アクション | 移行コスト |
|---|---|---|---|
| Soniox `stt-rt-v5` | **維持** | 2026年6月の最新モデルを既に既定化。日本語×RT×分離×$0.12/hの組合せで代替優位なし。モデル廃止サイクル（約4か月）をウォッチし、既定値のハードコード2箇所を追随 | — |
| Speechmatics バックエンド | **維持** | Ursa 2で日本語RT分離対応の有力フォールバック。実装済み | — |
| pyannoteAI live | **維持** | Precision-2系ホスティッドAPIはサブ150msで現行SOTA圏 | — |
| AssemblyAI 分離プロバイダ | **再検討** | 日本語ストリーミング非対応（多言語streamingは6言語のみ）。日本語音声での分離動作は未確認。実験用と明記するか削除 | 削除は小（オプション経路） |
| 声紋 ReDimNet / ECAPA | **維持** | 2026年でも軽量SOTA級。決定的な後継は未確認 | — |
| WhisperLiveKit + mlx-whisper | **維持（更新推奨）** | プロジェクトは活発。`>=0.2.20` 固定を最新へ、`python-multipart` 手当の要否を再確認（未確認） | 小 |
| OpenAI `gpt-5-mini` / `gpt-5-nano` | **乗り換え推奨** | 2世代前。現行は GPT-5.4 mini（$0.75/$4.50）/ nano（$0.20/$1.25）。env切替のみだが、`cost.py` 料金表と temperature 判定ロジックの更新、評価スイート（`das eval`）での回帰確認が必要 | 小（半日〜1日） |
| OpenAI `gpt-realtime-2` | **維持** | 2026年5月リリースの最新Realtimeモデルを既に採用 | — |
| `text-embedding-3-small` | **維持** | 後継モデル未確認 | — |
| `beta.chat.completions.parse` | **検討** | beta名前空間は非推奨方向。`chat.completions.parse`（または Responses API）へ移行しSDK更新耐性を確保 | 小 |
| エージェント基盤（自前） | **維持** | オーケストレーションが研究主題のためフレームワーク導入は不利。必要なら Pydantic AI が親和的 | — |
| Streamlit / pyvis | **維持** | 研究ビューア用途では十分。大規模グラフ操作が必要になったら Dash Cytoscape を検討 | — |
| Tavily | **維持** | 問題の兆候なし（深掘り未実施） | — |
| （追加提案）日本語STTベンチ | **実施推奨** | Soniox vs Speechmatics vs AmiVoice の日本語会議音声での自前比較。日本語WER・分離精度の独立公開データが乏しく、選定根拠を論文に書くための一次データになる | 中（録音データ＋評価スクリプト） |

### 特に優先度の高い3点

1. **LLM既定モデルの更新**（`gpt-5-mini` → GPT-5.4 mini、nano同様）: env変更＋`cost.py`更新＋eval回帰のみで効果大。
2. **AssemblyAI分離経路の位置づけ明確化**: 日本語ストリーミング非対応が公式情報。日本語実験で使うなら動作検証と注記、そうでなければ削除。
3. **日本語一次ベンチの取得**: Soniox採用の根拠が現状ベンダー主張中心。研究として国産（AmiVoice）含む3点比較を一度取る。

---
*確認できなかった事項（推測を避け「未確認」とした点): Sonioxの日本語単体WERの独立ベンチ、AssemblyAI `universal-3-5-pro` の日本語ストリーミング対応、`gpt-5-mini` の廃止予定、`text-embedding-3-small` 後継、WhisperLiveKit上流での `python-multipart` 依存修正、AmiVoiceリアルタイム話者分離の精度・レイテンシ実測。*
