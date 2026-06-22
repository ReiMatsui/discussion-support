#!/usr/bin/env python3
"""複数話者の会議音声WAVを生成する。

OpenAI TTS APIで各話者の発言を生成し、適切な間（ま）を入れて1つのWAVに結合する。
生成したWAVは soniox_live.py --wav で再生し、--agent と組み合わせて
ファシリテーターAIの介入テストに使う。

使い方:
  # 組み込みシナリオで生成
  uv run python tools/generate_conversation_wav.py --scenario stalled

  # 全シナリオを生成
  uv run python tools/generate_conversation_wav.py --all

  # 生成したWAVでファシリテーターをテスト（trigger=5で短いシナリオでも発火しやすく）
  uv run python -m das.asr.soniox_live --wav test_wavs/stalled.wav --agent --play --agent-trigger 5

  # 自分も参加（マイクON）
  uv run python -m das.asr.soniox_live --wav test_wavs/stalled.wav --agent --join --agent-trigger 5
"""

from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from das.asr.soniox_live import load_env

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  話者 → TTS音声マッピング
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# OpenAI TTSの利用可能な音声（区別しやすい組み合わせを選定）
VOICE_MAP: dict[str, str] = {
    "松井": "alloy",
    "田中": "echo",
    "佐藤": "nova",
    "鈴木": "onyx",
}

DEFAULT_VOICE = "shimmer"
TARGET_SR = 16000  # soniox_liveのサンプリングレート


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  シナリオ定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIOS: dict[str, list[dict]] = {}


def _reg(name: str, utterances: list[dict]):
    SCENARIOS[name] = utterances


# 停滞: 同じ話題を繰り返す
_reg("stalled", [
    {"speaker": "松井", "text": "やっぱりコストが一番の問題だと思います。"},
    {"speaker": "田中", "text": "そうですね、コストは確かに高いですよね。", "pause": 1.0},
    {"speaker": "松井", "text": "コスト削減をなんとかしないと先に進めないですよね。"},
    {"speaker": "田中", "text": "ええ、コストの問題は大きいです。", "pause": 1.5},
    {"speaker": "松井", "text": "とにかくコストをどうするかですよね。"},
    {"speaker": "田中", "text": "コスト面での対策が必要ですね。"},
    {"speaker": "松井", "text": "何かコストを下げる方法はないですかね。"},
    {"speaker": "田中", "text": "うーん、コストか…難しいですね。", "pause": 6.0},
    # ↑ 6秒の沈黙 → ファシリテーターの介入ポイント（_AGENT_SILENCE=5s）
])

# 偏り: 賛成意見ばかり
_reg("biased", [
    {"speaker": "松井", "text": "新しいAIツールを導入しましょう。業務効率が上がるはずです。"},
    {"speaker": "田中", "text": "賛成です。最近のAIはすごく進化してますからね。"},
    {"speaker": "佐藤", "text": "私も賛成です。競合他社も導入してますし。", "pause": 0.8},
    {"speaker": "松井", "text": "じゃあ早速来月から導入の準備を始めましょう。"},
    {"speaker": "田中", "text": "いいですね。ベンダーに見積もりを取りましょう。"},
    {"speaker": "佐藤", "text": "予算は問題ないと思います。効果を考えれば安いものです。", "pause": 6.0},
])

# 脱線: 本題から逸れる
_reg("derailed", [
    {"speaker": "松井", "text": "では次のスプリントの計画を決めましょう。"},
    {"speaker": "田中", "text": "はい、まずバックログの優先順位を。あ、そういえば昨日のサッカー見ました？"},
    {"speaker": "佐藤", "text": "見ましたよ！すごい試合でしたね。"},
    {"speaker": "田中", "text": "後半のゴールがすごかったですよね。", "pause": 0.5},
    {"speaker": "松井", "text": "確かに。でも審判の判定はどうかと思いましたけど。"},
    {"speaker": "佐藤", "text": "VAR導入してからああいう判定増えましたよね。"},
    {"speaker": "田中", "text": "スポーツとテクノロジーの関係って面白いですよね。", "pause": 6.0},
])

# 合意形成が必要
_reg("consensus_needed", [
    {"speaker": "松井", "text": "リリースは来週金曜日にしましょう。"},
    {"speaker": "田中", "text": "来週は早すぎます。テストが間に合わないかもしれません。"},
    {"speaker": "佐藤", "text": "でも顧客への約束があるので遅らせられないです。", "pause": 1.0},
    {"speaker": "田中", "text": "品質を犠牲にしてまでリリースすべきじゃないと思います。"},
    {"speaker": "松井", "text": "じゃあもう来週金曜で決定ということで。次の議題に移りましょう。", "pause": 6.0},
])

# 正常な議論（介入不要）
_reg("healthy", [
    {"speaker": "松井", "text": "認証方式はOAuth2.0にしたいと思います。"},
    {"speaker": "田中", "text": "いいと思います。ただ、トークンのリフレッシュ戦略はどうしますか？"},
    {"speaker": "佐藤", "text": "サイレントリフレッシュがUX的にはベストですが、セキュリティ面が気になります。"},
    {"speaker": "松井", "text": "確かに。リフレッシュトークンのローテーションを入れれば緩和できるかと。", "pause": 1.0},
    {"speaker": "田中", "text": "それならアクセストークンの有効期限も短めにできますね。15分くらい？"},
    {"speaker": "佐藤", "text": "15分でいいと思います。オフラインアクセスが必要な場合は別途検討しましょう。"},
    {"speaker": "松井", "text": "では認証はOAuth2.0、トークンローテーション、15分有効期限で進めましょう。", "pause": 2.0},
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TTS + WAV生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tts_to_pcm(text: str, voice: str) -> bytes:
    """OpenAI TTS APIで音声生成し、16kHz 16bit mono PCMを返す."""
    import openai
    client = openai.OpenAI()
    resp = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text,
        response_format="pcm",   # 24kHz 16bit mono PCM
    )
    pcm_24k = resp.content

    # 24kHz → 16kHz リサンプル（線形補間）
    import numpy as np
    samples_24k = np.frombuffer(pcm_24k, dtype="<i2").astype(np.float32)
    n_out = int(len(samples_24k) * 16000 / 24000)
    if n_out < 2:
        return b""
    indices = np.linspace(0, len(samples_24k) - 1, n_out)
    idx_floor = indices.astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(samples_24k) - 1)
    frac = indices - idx_floor
    samples_16k = samples_24k[idx_floor] * (1 - frac) + samples_24k[idx_ceil] * frac
    return np.clip(samples_16k, -32768, 32767).astype("<i2").tobytes()


def _silence_pcm(duration_sec: float) -> bytes:
    """指定秒数の無音PCMを返す."""
    n_samples = int(TARGET_SR * duration_sec)
    return b"\x00\x00" * n_samples


def _make_wav(pcm: bytes) -> bytes:
    """生PCMにWAVヘッダを付ける."""
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, TARGET_SR, TARGET_SR * 2, 2, 16) +
            b"data" + struct.pack("<I", n) + pcm)


def generate_scenario_wav(
    scenario_name: str,
    output_dir: str = "test_wavs",
    default_pause: float = 0.7,
    verbose: bool = True,
) -> str:
    """シナリオからWAVファイルを生成."""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"不明なシナリオ: {scenario_name} (利用可能: {', '.join(SCENARIOS)})")

    utterances = SCENARIOS[scenario_name]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scenario_name}.wav")

    if verbose:
        print(f"🎙️  シナリオ: {scenario_name} ({len(utterances)}発話)")

    pcm_parts: list[bytes] = []

    # 冒頭に1秒の無音
    pcm_parts.append(_silence_pcm(1.0))

    for i, utt in enumerate(utterances):
        speaker = utt["speaker"]
        text = utt["text"]
        pause = utt.get("pause", default_pause)
        voice = VOICE_MAP.get(speaker, DEFAULT_VOICE)

        if verbose:
            print(f"  [{i+1}/{len(utterances)}] {speaker}({voice}): {text[:40]}...", end="", flush=True)

        t0 = time.monotonic()
        pcm = _tts_to_pcm(text, voice)
        elapsed = time.monotonic() - t0

        if verbose:
            dur = len(pcm) / (TARGET_SR * 2)
            print(f" ({dur:.1f}s音声, {elapsed:.1f}s生成)")

        pcm_parts.append(pcm)
        pcm_parts.append(_silence_pcm(pause))

    # 末尾に十分な無音（沈黙トリガー発火 + 介入の余地）
    pcm_parts.append(_silence_pcm(10.0))

    all_pcm = b"".join(pcm_parts)
    wav_data = _make_wav(all_pcm)

    with open(out_path, "wb") as f:
        f.write(wav_data)

    total_dur = len(all_pcm) / (TARGET_SR * 2)
    if verbose:
        print(f"✅ 保存: {out_path} ({total_dur:.1f}秒)")

    return out_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description="会議シナリオ音声WAV生成")
    ap.add_argument("--scenario", "-s", help="生成するシナリオ名")
    ap.add_argument("--all", "-a", action="store_true", help="全シナリオを生成")
    ap.add_argument("--list", "-l", action="store_true", help="シナリオ一覧")
    ap.add_argument("--output-dir", "-o", default="test_wavs", help="出力ディレクトリ")
    ap.add_argument("--pause", type=float, default=0.7, help="発話間のデフォルト間隔（秒）")
    args = ap.parse_args()

    if args.list:
        for name, utts in SCENARIOS.items():
            speakers = sorted(set(u["speaker"] for u in utts))
            print(f"  {name:20s} {len(utts)}発話  話者: {', '.join(speakers)}")
        return

    load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY が設定されていません。", file=sys.stderr)
        sys.exit(1)

    if args.all:
        paths = []
        for name in SCENARIOS:
            path = generate_scenario_wav(name, args.output_dir, args.pause)
            paths.append(path)
        print(f"\n🎉 全{len(paths)}シナリオ生成完了")
        print(f"\nテスト実行例:")
        print(f"  uv run python -m das.asr.soniox_live --wav {paths[0]} --agent --play")
    elif args.scenario:
        path = generate_scenario_wav(args.scenario, args.output_dir, args.pause)
        print(f"\nテスト実行:")
        print(f"  uv run python -m das.asr.soniox_live --wav {path} --agent --play")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
