# 視覚話者検証 + ローカル音声処理: 実現可能性調査

> 目的: 議論支援システムにおいて、マイクが拾った音声が **カメラに映る対象人物
> の発話** であることを保証したい。隣席や別話者の音声が混入しても、口の動き
> が伴わなければ却下する。さらに可能なら iPhone 側で音声処理まで完結させたい。
>
> 結論を先に: **Mac での PoC は問題なく実装可能。iPhone 側でのフル
> ローカル処理 (Gemma 3n マルチモーダルで音声+映像) も技術的には可能だが、
> リアルタイム性とメモリ制約から「カメラ前処理は iPhone、Whisper による ASR
> 本体は Mac」が現実解。**

---

## 1. アーキテクチャの選択肢

| # | 構成 | リアルタイム性 | プライバシ | 実装コスト | 採否 |
|---|------|----------------|------------|------------|------|
| A | iPhone をマイク&カメラとしてのみ使い、Mac で全処理 (MediaPipe + Whisper) | ◎ | ○ (端末内) | 低 | **採用 (PoC)** |
| B | iPhone 側で口パク検出 → 検出区間だけ Mac に転送 → Mac で Whisper | ○ | ◎ | 中 | 将来 |
| C | iPhone 側で Gemma 3n により口パク検出 + ASR まで完結 | △ | ◎ | 高 | 調査のみ |
| D | iPhone 側で USM/Whisper-tiny を ANE 経由で実行 + Mac で集約 | ○ | ◎ | 中 | 候補 |

PoC は **A** に振る。理由:

- Mac (Apple Silicon) なら Whisper large-v3 でも余裕で動く。既存
  `das listen` も `mlx-whisper` バックエンドで動作している。
- iPhone を Continuity Camera 経由で接続すれば、USB / Wi-Fi ケーブルなしで
  カメラとマイクの両方を Mac から扱える。Python の OpenCV から普通の
  `cv2.VideoCapture(index)` で開ける (macOS Ventura+ / iOS 16+)。
- 視覚 VAD と音声 VAD の AND ロジックという、本タスクの核となる部分の検証は
  どの構成でも共通なので、まず A で「コンセプトが妥当か」を確かめるのが速い。

## 2. Gemma 3n を iPhone でローカル実行できるか

**結論: できる。ただし用途は限定的。**

- Gemma 3n は Google が「日常デバイス (スマホ・ノート PC・タブレット) 向けに
  最適化したマルチモーダルモデル」と明示している。テキスト + 画像 + 音声 +
  (限定的に) 動画を同時入力できる。
  ([overview](https://ai.google.dev/gemma/docs/gemma-3n))
- 音声エンコーダは Universal Speech Model (USM) 系で、160ms ごとに 1 トークンを
  生成する。日本語を含む多言語に対応。([overview](https://ai.google.dev/gemma/docs/gemma-3n))
- iOS への実行経路:
  1. **Google AI Edge Gallery (iOS 版, 2026/04 公開)**: A14 Bionic 以降のチップ
     で Gemma 3n をオフライン実行可能。試運転には十分。
     ([MindStudio記事](https://www.mindstudio.ai/blog/google-ai-edge-gallery-offline-llm-ios))
  2. **LiteRT-LM (旧 MediaPipe LLM Inference iOS の後継)**: Google AI Edge が
     推奨する新フレームワーク。MediaPipe LLM Inference の iOS 実装は
     **deprecated** になっているので、新規開発は LiteRT-LM を使う。
     ([iOS guide](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios))
  3. **コミュニティ製 Swift/MLX 実装**: `gemma3n-ios`, `gemma-4-swift-mlx` 等。
     Apple Silicon ネイティブ。
     ([gemma3n-ios](https://github.com/sid9102/gemma3n-ios),
      [gemma-4-swift-mlx](https://github.com/VincentGourbin/gemma-4-swift-mlx))

### 制約

- **モデルサイズ**: Gemma 3n の E2B (effective 2B) / E4B (effective 4B) は
  量子化後でそれぞれ ~1.5GB / ~3GB。iPhone のメモリ帯では E2B でも稼働中に
  4GB 強の RAM を要する。**iPhone 12 以降推奨、Pro モデルだと余裕**。
- **リアルタイム性**: 「30 fps の映像 + 連続音声」を毎フレーム LLM に流すの
  は非現実的。Gemma 3n を「音声断片の話者帰属判定」「短い発話の認識」用に
  on-demand で呼ぶならば実用速度。
- **連続 ASR 用途**: ストリーミング ASR を Gemma 3n 単体で実装するのは
  プラクティスとして確立していない。Whisper-tiny を Core ML 化して
  Apple Neural Engine で走らせるほうが、目的が ASR ならば素直。

### 推奨

- 視覚話者検証 (口の開閉) **だけ** iPhone 側で行うのは軽量で実用的。
  Apple Vision Framework の `VNDetectFaceLandmarksRequest` が
  Core ML 経由で ANE 上で動き、~5ms/frame。Swift 製の小さな iOS アプリで
  「対象人物が話しているフラグ + 音声」を WebSocket で Mac に送るのが、
  プライバシーと性能のバランスが良い。
- Gemma 3n は、認識結果の **意味的フィルタリング** (「これは隣の人の独り言
  だろうか、参加者の発言だろうか」を文脈で判定) に使うのが向く。これは現行
  パイプラインの GraphAgent 直前に挟むレイヤとして整合する。

## 3. PoC (構成 A) の中身

`scripts/visual_asr_poc.py` 参照。要点:

```
[カメラ] →[MediaPipe FaceMesh]→ MAR (mouth aspect ratio)
                                  ↓ 直近 500ms の分散
                                  → visually_speaking フラグ
                                                ┐
[マイク (sounddevice 16kHz mono)]→ 30ms ブロックごとに RMS
                                  → audio_active フラグ
                                                ┘
                                      AND
                                       ↓
                          speaker_verified == True が連続する区間を
                          ひとつの発話として切り出し、
                          0.5s 無音で終端 → Whisper (faster-whisper / mlx-whisper)
                                            に投入 → 日本語テキストを出力
```

ポイント:

- **MAR の絶対値ではなく分散を使う** (口を開けっぱなしの人を「話している」
  と誤判定しない)。`mouth_aspect_ratio` を 0.5 秒分蓄積し、分散が閾値を
  超えたら "visually speaking" とする。
- **音声 VAD は RMS ベースの簡易版**。本実装に進める段階で WebRTC VAD や
  Silero VAD に差し替える前提。
- **同期は厳密でなくて良い**: 唇の動きと音声には数十 ms のオフセットがあるが、
  発話の前後 500ms 程度のマージンで充分。SyncNet 等の厳密な phoneme-viseme
  同期は本タスクの優先度では過剰。

## 4. 評価方法 (案)

検証セッションでは、以下の 4 シナリオで誤判定率を測る:

| シナリオ | 期待挙動 |
|----------|----------|
| 対象人物だけが話す | すべて Whisper に流れる |
| 対象人物が黙っている時に隣で会話 | Whisper が呼ばれない |
| 対象人物が「あー」と口を開けたまま無発声 | Whisper が呼ばれない (MAR 分散が小さい) |
| 対象人物と隣が同時に発話 | 対象人物分のみが切り出される (誤って隣の音声が混入するのは諦める。本 PoC のスコープ外) |

定量指標案: precision (出力された発話のうち対象人物のもの) と recall
(対象人物の発話のうち拾えたもの) を、~5 分のサンプルセッションで集計。

## 5. discussion-support 本体への組み込み方針

現行の `das listen` (WhisperLiveKit + sounddevice) の前段に、本 PoC の
「視覚 VAD ゲート」を挟む形が綺麗:

1. sounddevice の入力を分岐させ、(a) 視覚 VAD ゲート (b) raw passthrough の
   2 系統に。
2. (a) で speaker_verified == True の区間にだけマスクを立てる。
3. WhisperLiveKit に渡す PCM ストリームを、マスクが立っている区間 + 余韻
   500ms に限定。残りはゼロ埋めまたは無音バッファに置き換え。
4. WhisperLiveKit 側で発話単位の確定が走る。

将来的に複数話者対応 (`speaker_1`, `speaker_2`...) を導入する場合は、各話者
に対応するカメラビュー (or バウンディングボックス) を持たせ、フェイス
ランドマークごとに独立した speaker_verified を出力するように拡張できる。

## 6. 参考

- [Gemma 3n model overview - Google AI for Developers](https://ai.google.dev/gemma/docs/gemma-3n)
- [Deploy Gemma on mobile devices](https://ai.google.dev/gemma/docs/integrations/mobile)
- [LLM Inference guide for iOS (MediaPipe → LiteRT-LM)](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios)
- [sid9102/gemma3n-ios (community iOS app)](https://github.com/sid9102/gemma3n-ios)
- [VincentGourbin/gemma-4-swift-mlx (Apple Silicon MLX)](https://github.com/VincentGourbin/gemma-4-swift-mlx)
- [Google AI Edge Gallery on iPhone (MindStudio)](https://www.mindstudio.ai/blog/google-ai-edge-gallery-offline-llm-ios)
- [Apple Continuity Camera support](https://support.apple.com/en-us/102546)
- [MediaPipe Face Mesh / mouth landmarks 解説 (Mert)](https://medium.com/@Mert.A/detect-eyes-nose-and-mouth-with-mediapipe-bbfdf7a61f21)
- [MAR を用いた口の開閉検出 (Drowsiness 論文)](https://www.researchgate.net/publication/396219261_Development_of_a_Real-time_Driver's_Drowsiness_Detection_System_Using_MediaPipe_Face_Mesh)
