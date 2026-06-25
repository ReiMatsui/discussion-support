# リアルタイム話者識別の改善 — 技術調査レポート（2026-06）

対象: 日本語ライブ会議/議論支援アプリ。完全ストリーミングで「誰が話しているか」を表示。
課題: 雑談レベルで複数人が速く応酬（短い発話・速いターンテイク・重なり）すると話者特定が崩れる。
現状: Soniox リアルタイムSTT（話者ラベル付き）＋ 自前の声紋補正（ReDimNet / Resemblyzer / ECAPA）。

---

## 0. 結論（率直に）

**完全リアルタイム × 雑談レベルの多人数では、「人間並みに誰が話したか分かる」は今の技術では構造的に到達困難です。** ただし、やり方を変えれば現状よりはっきり良くできます。要点は3つ。

1. **2〜3人の落ち着いた会話なら、ほぼ人間並み（DER 5〜12%程度）は到達可能。** 崩れるのは「重なり」と「速い多人数」のとき。
2. リアルタイム話者分離の誤りは現実の雑談で **DER 20〜40%、重なりが多いと50%超** になり得る。標準指標(DER)は重なりを十分採点しないので、体感はさらに悪い。これは特定ベンダーの問題ではなく、業界共通の天井。([roamingpigs](https://roamingpigs.com/field-manual/speaker-diarization-hardest/))
3. **最も効くレバーは「ブラインドな話者分離の改良」ではなく「登録済み話者のターゲット話者モデリング」。** 参加者が既知/常連なら、各人の声を登録して“その人がいま話しているか”を直接当てにいく方式（TS-VAD系）が、重なりに強く、クラスタリングが苦手な短い発話の天井を上げる。TS-VADは従来のクラスタリングに対し **相対DER 30%超の改善** 実績。([arXiv:2005.07272](https://arxiv.org/abs/2005.07272))

つまり方針は「人間並みの万能ブラインド分離を狙う」のではなく、**「既知の参加者を登録して狙い撃つ」＋「自信が無い短い発話は無理に当てず“未確定”に倒す」** という、すでに着手している路線を強化するのが本筋です。

---

## 1. なぜ難しいか（構造的な3つの壁）

- **重なり(overlap)が誤りの主因。** 2人同時発話の区間は、多くの実装で「最初の話者」に丸ごと割り当てられる既知の近道があり、雑談の被りで崩れる。([arXiv:2501.16641](https://arxiv.org/pdf/2501.16641))
- **ストリーミングは「即時に確定」せざるを得ず、後から直せない。** 文脈が部分的なまま各フレームのラベルを確定するため、非同期(async)より必ず精度が落ちる。各ベンダーも明言。([AssemblyAI](https://www.assemblyai.com/blog/top-speaker-diarization-libraries-and-apis))
- **〜1秒未満の短い発話はラベルが揺れる。** 短い区間は声紋が不安定で、クラスタリングが苦手。これは学習データを増やしても消えない構造的限界。([Gladia](https://www.gladia.io/blog/what-is-diarization), [arXiv:2106.05792](https://arxiv.org/pdf/2106.05792))
- **日本語の自発音声はさらに難しい。** 同系手法でも英語CALLHOME 12.84% に対し、日本語CSJ(話し言葉)は **21.64% DER**（1秒遅延）という報告。日本語のリアルタイムDER公開値はほぼ存在せず、自前ベンチが要る。([arXiv:2101.08473](https://arxiv.org/abs/2101.08473))

---

## 2. 最大のレバー: 登録（enrollment）＋ターゲット話者モデリング

- **TS-VAD (Target-Speaker VAD):** 各参加者の登録声紋を入力に、フレームごとに「その人が話しているか」を全員ぶん同時推定。**重なりを正面から扱える**（クラスタリングが破綻する所で効く）。CHiME-6で x-vectorクラスタリングに対し相対DER 30%超改善。オンライン/ストリーミング版もあり。([arXiv:2005.07272](https://arxiv.org/abs/2005.07272), [online TS-VAD](https://arxiv.org/pdf/2310.08696))
- **登録の効果は定量的にも大きい:** 最初の1人を登録するだけで **DER 3〜5%(絶対)改善**、2人ぶんで頭打ち。([arXiv:2509.18377](https://arxiv.org/pdf/2509.18377))
- 示唆: あなたの「声紋登録レイヤ」はこの路線の入口。**登録を主役に据え、登録済みの人は狙い撃ち、未登録/自信なしは未確定** にするのが、現状資産を活かしつつ天井を上げる道。

---

## 3. 選択肢と trade-off

### 案A: 現状スタック（Soniox＋自前声紋）を強化 — 低コスト・即効
- **やること:**
  - 埋め込みモデルを更新（§4）。Resemblyzerは捨て、ReDimNet2-B3/B6 か ERes2NetV2(多言語・短時間最適)へ。
  - スコア較正（品質指標で補正）、品質重み付きの逐次セントロイド更新、低自信の短い発話は“未確定”に倒して後で再割当（§5）。
  - **Sonioxの使い方の見直し（重要）:** Soniox公式は「**エンドポイント検出や手動確定(finalize)は話者分離の精度を下げる**、asyncの方が大幅に高精度」と明記。今のアプリは手動flush/確定をしているので、**自分の確定ロジックがSonioxの話者分離を弱めている可能性**がある。ここは要検証。([Soniox docs](https://soniox.com/docs/stt/concepts/speaker-diarization))
- **精度:** 中（現状改善）／**レイテンシ:** 維持／**コスト:** 低／**実装難:** 低〜中。

### 案B: 専用ストリーミングdiarizerを追加（Sonioxは文字起こし、話者はSortformer）
- **NVIDIA Streaming Sortformer (Interspeech 2025):** オープン(CC-BY-4.0)、重なりネイティブ、**0.32〜1.04秒の低遅延でも精度を保つ**（2話者CALLHOME 6.57% DER、超低遅延0.32sでも13.4%前後）。チャンクサイズで遅延↔精度を実行時調整可。NeMo/Rivaで運用。([HF](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2), [paper](https://arxiv.org/abs/2507.18446))
- **弱点:** **最大4話者**（5人以上は急悪化）。英語/中国語中心で、NVIDIA自身「他言語はfine-tune推奨」。**日本語は要追加学習**だが、同系のNEST日本語エンコーダ `nest-ja-0.1b` があり微調整の道はある。NVIDIA自身「**非常に速いターンテイクや激しいクロストークは依然challenging**」と明言（=雑談の核心は完全には解けない）。([blog](https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/), [nest-ja](https://huggingface.co/sbintuitions/nest-ja-0.1b))
- **精度:** 高（特に2〜4人）／**レイテンシ:** 低（0.3〜1s）／**コスト:** 中（GPU）／**実装難:** 中〜高（日本語fine-tune込みで高）。

### 案C: ターゲット話者モデリング(TS-VAD系)を自前構築 — 天井は最も高い
- 登録済み参加者に特化。重なりに最強。研究的裏付けも最も強い（§2）。
- **精度:** 最高（既知話者・重なり時）／**レイテンシ:** 低〜中／**コスト:** 中／**実装難:** 高（研究実装の自前運用）。

### 案D: 他のクラウドAPIへ乗り換え/比較 — ただし日本語リアルタイム分離は限定的
- **Google:** Chirp3の話者分離は**バッチ専用**（StreamingRecognize不可）→ライブ用途は除外。([docs](https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3))
- **Gladia / OpenAI(gpt-4o-transcribe-diarize):** 話者分離は**バッチ/非Realtime専用**→除外。([Gladia](https://docs.gladia.io/chapters/audio-intelligence/speaker-diarization), [OpenAI](https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize))
- **AssemblyAI:** ストリーミング話者分離は強いが**日本語は対象外**（多言語ストリーミングにJAなし）。話者“識別(名前)”はasync専用。([docs](https://www.assemblyai.com/docs/streaming/universal-streaming/multilingual-transcription))
- **AWS Transcribe:** ja-JPストリーミング＋分離可だが、**話者ラベルは確定済みセグメントにしか付かず遅れて出る**（速い雑談のライブUXに不利）。([docs](https://docs.aws.amazon.com/transcribe/latest/dg/diarization-streaming.html))
- **Deepgram / Speechmatics / Azure:** リアルタイム分離は可能。日本語×リアルタイム分離の明言は弱く**要POC検証**。Deepgramはストリーミングは v1 diarizer のみ／最大12話者。Azureは最大35話者。Speechmaticsは `prefer_current_speaker` 等の調整可。([Deepgram](https://developers.deepgram.com/docs/diarization), [Azure](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization), [Speechmatics](https://docs.speechmatics.com/speech-to-text/realtime/realtime-diarization))
- **Soniox(現状):** **日本語のリアルタイム分離を明示的に唱う唯一格**（速い/重なり対応を主張、最大15話者）。ただし全てベンダー自己申告。([JA](https://soniox.com/speech-to-text/japanese))
- **精度:** ベンダー次第／**レイテンシ:** 低／**コスト:** 中（従量）／**実装難:** 低（ただし日本語分離の実力はPOC必須）。

---

## 4. 埋め込みモデルの更新（どの案でも効く安い改善）

VoxCeleb1-O EER（低いほど良い／2秒列が今回重要）:

| モデル | 規模 | フルEER | 2秒EER | 備考 |
|---|---|---|---|---|
| **ReDimNet2-B6** | 12.5M | **0.17%** | 未公表 | 現状最高クラス。MIT、torch.hub。 |
| ReDimNet2-B3 | 4.1M | 0.42% | 未公表 | 軽量で高精度のスイートスポット。 |
| **ERes2NetV2** | 17.8M | 0.61% | **1.48%** | **短時間最適化**＋多言語20万話者チェックポイント（日本語に有利）。 |
| CAM++ | 7M | 0.73% | 競争力 | ECAPA比 約半分の規模・約2倍速。 |
| ECAPA-TDNN(現状) | 20M | 0.82% | 1.95% | 2秒で意外と粘る。フォールバックに残す価値あり。 |
| Resemblyzer(現状) | — | 低品質 | — | **ensembleから外す推奨。** |

出典: [ReDimNet2](https://github.com/PalabraAI/redimnet2), [ERes2NetV2](https://arxiv.org/html/2406.02167v1), [CAM++](https://arxiv.org/html/2303.00332v3), [3D-Speaker(多言語)](https://github.com/modelscope/3D-Speaker)

注意: 公表EERは英語。**日本語/会話ドメインではEERが10〜40倍悪化し得る**ので、頭の数字はそのまま当てにせず**自前の日本語短尺クリップで実測**が必須。「Large-margin(LM)版」は3秒超向けで、**短い発話では非LM版が有利な可能性**あり（要A/B）。([WeSpeaker](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md))

---

## 5. 短い発話の識別を底上げする実装テクニック（バックボーン差し替えより高レバレッジ）

1. **品質ベースのスコア較正(QMF):** コサイン類似度を、enroll/testの長さ・**埋め込みのノルム(L2)**・SNR で補正（小さなロジスティック回帰）。短い/雑音発話はノルムが小さくなるので、ほぼ無料の信頼度指標になる。([xi+/品質](https://arxiv.org/pdf/2407.11365))
2. **不確かさ考慮プーリング(xi-vector):** フレームを不確かさで重み付け。セグメント単位の信頼度が得られ閾値化できる。
3. **品質重み付きの逐次セントロイド更新:** いまの累積登録を拡張し、各埋め込みを**ノルム/品質で重み付け**（雑音短尺はセントロイドを動かしにくく）。
4. **低自信は確定せず後で再割当:** 短すぎ/低ノルムは即断せず“未確定”にし、その人のセントロイドが安定してから遡及で割当（=いまの未確定方針の強化）。先端diarizer(DiariZen/cVBx)も「短尺をクラスタから外し後で再割当」。([DiariZen](https://arxiv.org/html/2509.26177v1))
5. **AS-norm(適応スコア正規化):** セッション参加者のセントロイド群をコホートに正規化。短尺で効く。
6. **マルチクロップ平均:** 1ターンから複数の小窓で埋め込みを取り平均して安定化（窓分割の延長）。
7. **VAD/エンドポイントの設計:** Silero VAD v5(32ms/CPU/状態あり)で軽くゲート。速い「ぽんぽん」を早切り/併合しないよう、**意味的エンドポイント(部分文字起こし)と併用**。セッション間で `reset_states()`。([VAD](https://soniqo.audio/guides/vad))

---

## 6. 日本語の注意点

- 公開の**日本語リアルタイムDERは事実上なし**。多言語diarizationの良い数字は全部バッチ(PyannoteAI 11.2%/DiariZen 13.3%)。→ **自前の日本語ベンチが必須。** ([DiariZen](https://arxiv.org/pdf/2509.26177))
- 日本語ベンチは **CSJ(話し言葉コーパス)**。LLMによる日本語ASR誤り訂正(GER)研究もあり、**事後テキスト補正レイヤ**としては有望（声紋補正と相補）。([CSJ-GER](https://arxiv.org/abs/2408.16180))
- 日本語NESTエンコーダ `nest-ja-0.1b` があり、Sortformer系の日本語fine-tuneの足場になる。([nest-ja](https://huggingface.co/sbintuitions/nest-ja-0.1b))

---

## 7. 推奨ロードマップ（段階的・低リスク順）

**フェーズ1（数日〜1週、低コスト・現状スタック）**
1. Sonioxの使い方検証: 手動flush/確定が分離を弱めていないか切り分け（async比較・finalizeの間引き）。
2. 埋め込みを ReDimNet2-B3 か ERes2NetV2(多言語) に差し替え、Resemblyzerを外す。**自前の日本語短尺で実測。**
3. 既存の登録レイヤに「品質重み付きセントロイド＋スコア較正＋低自信は未確定で後追い割当」を追加。

**フェーズ2（評価）**
4. 同一日本語音声で **Soniox vs Deepgram/Speechmatics/Azure のリアルタイム分離POC** を実測比較。
5. **Streaming Sortformer(4話者)** を日本語データで評価（必要なら nest-ja で微調整）。

**フェーズ3（本命・天井上げ）**
6. 既知参加者向けに **TS-VAD系のターゲット話者モデリング**（登録前提）を試作。重なり・速い応酬への最有力。

**期待値の正直な握り:** 2〜3人なら人間並みを狙える。4人以上の速い被りは「良い」止まりで「完璧」は現実的でない。UX側（未確定表示・ラベル安定性・後追い確定）で“間違って言い切らない”ことが体感品質を最も上げる。

---

## 8. 主要出典
- Streaming Sortformer: https://arxiv.org/abs/2507.18446 / https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2 / https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/
- LS-EEND(最大8話者・オンライン): https://arxiv.org/abs/2410.06670 / https://github.com/Audio-WestlakeU/FS-EEND
- diart(オンライン・重なり対応): https://github.com/juanmc2005/diart / https://arxiv.org/abs/2407.04293
- pyannote 3.1 / community-1: https://huggingface.co/pyannote/speaker-diarization-3.1 / https://www.pyannote.ai/blog/community-1
- TS-VAD: https://arxiv.org/abs/2005.07272 / online: https://arxiv.org/pdf/2310.08696
- 登録効果: https://arxiv.org/pdf/2509.18377
- 埋め込み: https://github.com/PalabraAI/redimnet2 / https://arxiv.org/html/2406.02167v1 (ERes2NetV2) / https://arxiv.org/html/2303.00332v3 (CAM++) / https://github.com/modelscope/3D-Speaker
- 短尺較正/品質: https://arxiv.org/pdf/2407.11365 / https://arxiv.org/html/2509.26177v1 (DiariZen/cVBx)
- API: https://soniox.com/docs/stt/concepts/speaker-diarization / https://soniox.com/speech-to-text/japanese / https://developers.deepgram.com/docs/diarization / https://docs.speechmatics.com/speech-to-text/realtime/realtime-diarization / https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization / https://docs.cloud.google.com/speech-to-text/docs/models/chirp-3 / https://docs.aws.amazon.com/transcribe/latest/dg/diarization-streaming.html
- 現実の天井: https://roamingpigs.com/field-manual/speaker-diarization-hardest/
- 日本語: CSJ-GER https://arxiv.org/abs/2408.16180 / nest-ja https://huggingface.co/sbintuitions/nest-ja-0.1b
</content>
