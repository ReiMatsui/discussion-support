# visual_asr_poc.py — 視覚話者検証つき音声認識 PoC

カメラで口の動きを見て「対象人物が今しゃべっている」ことを確認できた区間
だけ Whisper に流す検証スクリプト。隣席の声などのマイク漏れを弾く。

詳しい設計と Gemma 3n の iPhone 実行可否の調査は
[`docs/visual_speaker_verification_feasibility.md`](../docs/visual_speaker_verification_feasibility.md)
にまとめてある。

## セットアップ (Mac, Apple Silicon)

```bash
# 既存プロジェクトの venv を流用する場合
uv pip install opencv-python mediapipe sounddevice faster-whisper

# 速度重視 (Apple Silicon): faster-whisper の代わりに mlx-whisper を使う
uv pip install mlx-whisper
```

> **Note** Python 3.11 を想定。`mediapipe` の wheel が 3.12/3.13 にまだ提供
> されていないバージョンに当たることがあるので、本プロジェクトの
> `requires-python = ">=3.11,<3.13"` に揃えるのが安全。

### iPhone をマイク&カメラとして使う (Continuity Camera)

1. iPhone (iOS 16+) と Mac (macOS Ventura+) を同じ Apple ID にサインインし、
   両方で Wi-Fi と Bluetooth を ON。
2. iPhone を Mac の近くに置き、ロックを解除。
3. スクリプトを起動する直前に **Mac 側のシステム設定 → カメラ / マイク** で
   ターミナル (or 使っている Python 実行環境) に権限を付与しておく。
4. `python scripts/visual_asr_poc.py --list-devices` で iPhone Camera /
   iPhone Microphone が見えていることを確認。

> macOS Sonoma 以降の OpenCV では `AVCaptureDeviceTypeContinuityCamera` への
> 移行が必要だが、`opencv-python` 4.10 以降は通常 `VideoCapture(index)` で
> 開ける。開けない場合は `--camera-index` を 0/1/2 と順に試す。

## 使い方

```bash
# 既定 (内蔵カメラ + 既定マイク + faster-whisper "small")
python scripts/visual_asr_poc.py

# デバイス一覧
python scripts/visual_asr_poc.py --list-devices

# iPhone を Continuity Camera で接続
python scripts/visual_asr_poc.py \
    --camera-index 1 \
    --audio-device "iPhone Microphone"

# Apple Silicon で高速化したい (mlx-whisper)
python scripts/visual_asr_poc.py \
    --asr-backend mlx \
    --asr-model mlx-community/whisper-large-v3-turbo

# 視覚 VAD だけ動作確認 (ASR を呼ばない)
python scripts/visual_asr_poc.py --asr-backend none

# しきい値調整 (プレビュー画面で MAR var / RMS を見ながら詰める)
python scripts/visual_asr_poc.py \
    --mar-var-threshold 0.0008 \
    --audio-rms-threshold 0.012
```

実行中はカメラ映像のプレビューに以下が表示される:

- `face: OK/NO` — 顔ランドマークが検出できているか
- `MAR` — Mouth Aspect Ratio (口の縦/横比)
- `MAR var` — 直近 500ms の MAR 分散 (これがしゃべってる時に跳ねる)
- `visual speak` — MAR var がしきい値を超えたら True
- `audio RMS` — マイク入力の音量

枠が **緑** のときが視覚的に「話している」と判定された状態。

## チューニングのコツ

しきい値 (`--mar-var-threshold`, `--audio-rms-threshold`) は環境依存。最初
は `--asr-backend none` で起動し、プレビューに出る `MAR var` と `audio RMS`
を見ながら以下を決める:

1. **無発話時のノイズフロア** を 5 秒ほど観察。MAR var の最大値と RMS の
   平均が「無音」の基準。
2. **発話時のピーク** を観察。MAR var は 0.001〜0.005 のオーダになることが
   多い。RMS は 0.02〜0.1 程度。
3. しきい値はノイズフロアの ~3 倍を狙う。

照明条件・口紅の有無・マスクの有無で大きく変わる点に注意。

## 既知の制限

- 単一話者前提 (`max_num_faces=1`)。複数話者の verification には未対応。
- MediaPipe FaceMesh は **マスクや手で口を覆っている顔** に弱い。
- 簡易 RMS VAD なので、大きな環境ノイズ (空調・タイピング音) があると
  audio_active が立ちっぱなしになる可能性がある。本実装では WebRTC VAD か
  Silero VAD に置き換える前提。
- Whisper は短すぎる発話 (~0.4 秒未満) を捨てている。`--min-utterance-s` で
  調整可能だが、短い発話を拾い切るには WhisperX や Whisper streaming への
  乗せ替えが必要。

## ファイル構成

- `scripts/visual_asr_poc.py` — 本スクリプト
- `scripts/README_visual_asr.md` — この文書
- `docs/visual_speaker_verification_feasibility.md` — 設計と調査レポート
