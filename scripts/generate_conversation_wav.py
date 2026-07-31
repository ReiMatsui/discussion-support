#!/usr/bin/env python3
"""複数話者の会議音声WAVを生成する。

OpenAI TTS APIで各話者の発言を生成し、適切な間（ま）を入れて1つのWAVに結合する。
生成したWAVは live.py --wav で再生し、--agent と組み合わせて
ファシリテーターAIの介入テストに使う。

使い方:
  # 組み込みシナリオで生成
  uv run python scripts/generate_conversation_wav.py --scenario stalled

  # 全シナリオを生成
  uv run python scripts/generate_conversation_wav.py --all

  # 生成したWAVでファシリテーターをテスト（trigger=5で短いシナリオでも発火しやすく）
  uv run python -m das.asr.live --wav test_wavs/stalled.wav

  # 自分も参加（マイクON）
  uv run python -m das.asr.live --wav test_wavs/stalled.wav
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

from das.asr.live._bootstrap import load_env  # noqa: E402

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
TARGET_SR = 16000  # liveのサンプリングレート


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  シナリオ定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENARIOS: dict[str, list[dict]] = {}


def _reg(name: str, utterances: list[dict]):
    SCENARIOS[name] = utterances


# 停滞: 同じ話題を繰り返す（12発話）
_reg("stalled", [
    {"speaker": "松井", "text": "やっぱりコストが一番の問題だと思います。"},
    {"speaker": "田中", "text": "そうですね、コストは確かに高いですよね。"},
    {"speaker": "松井", "text": "コスト削減をなんとかしないと先に進めないですよね。"},
    {"speaker": "田中", "text": "ええ、コストの問題は大きいです。", "pause": 1.0},
    {"speaker": "松井", "text": "とにかくコストをどうするかですよね。"},
    {"speaker": "田中", "text": "コスト面での対策が必要ですね。"},
    {"speaker": "松井", "text": "何かコストを下げる方法はないですかね。"},
    {"speaker": "田中", "text": "うーん、コストか…難しいですね。"},
    {"speaker": "松井", "text": "外注を減らせばコストは下がると思うんですけど。"},
    {"speaker": "田中", "text": "でも外注を減らすのも簡単じゃないですよね。コストがかかりますし。"},
    {"speaker": "松井", "text": "結局コストの問題に戻ってきますよね。"},
    {"speaker": "田中", "text": "そうなんですよ。コストをどうにかしないと。", "pause": 6.0},
])

# 偏り: 賛成意見ばかり（12発話）
_reg("biased", [
    {"speaker": "松井", "text": "新しいAIツールを導入しましょう。業務効率が上がるはずです。"},
    {"speaker": "田中", "text": "賛成です。最近のAIはすごく進化してますからね。"},
    {"speaker": "佐藤", "text": "私も賛成です。競合他社も導入してますし。"},
    {"speaker": "松井", "text": "じゃあ早速来月から導入の準備を始めましょう。"},
    {"speaker": "田中", "text": "いいですね。ベンダーに見積もりを取りましょう。"},
    {"speaker": "佐藤", "text": "予算は問題ないと思います。効果を考えれば安いものです。"},
    {"speaker": "松井", "text": "導入は段階的にやるのがいいかな。まず営業部から。"},
    {"speaker": "田中", "text": "営業部がいいと思います。一番効果が出やすいですから。"},
    {"speaker": "佐藤", "text": "研修も簡単で済むと思います。みんなすぐ使いこなせますよ。"},
    {"speaker": "松井", "text": "では来月中に導入完了を目標にしましょう。"},
    {"speaker": "田中", "text": "問題ないです。スケジュール的にも余裕があります。"},
    {"speaker": "佐藤", "text": "早く導入して効果を実感したいですね。", "pause": 6.0},
])

# 脱線: 本題から逸れる（12発話）
_reg("derailed", [
    {"speaker": "松井", "text": "では次のスプリントの計画を決めましょう。"},
    {"speaker": "田中", "text": "はい、まずバックログの優先順位を。あ、そういえば昨日のサッカー見ました？"},
    {"speaker": "佐藤", "text": "見ましたよ！すごい試合でしたね。"},
    {"speaker": "田中", "text": "後半のゴールがすごかったですよね。"},
    {"speaker": "松井", "text": "確かに。でも審判の判定はどうかと思いましたけど。"},
    {"speaker": "佐藤", "text": "VAR導入してからああいう判定増えましたよね。"},
    {"speaker": "田中", "text": "スポーツとテクノロジーの関係って面白いですよね。"},
    {"speaker": "松井", "text": "AIで審判の判定を自動化できたらいいのに。"},
    {"speaker": "佐藤", "text": "それいいですね。画像認識で判定するとか。"},
    {"speaker": "田中", "text": "でもやっぱり人間の判断も大事ですよね。"},
    {"speaker": "松井", "text": "まあスポーツは感情も大事ですからね。"},
    {"speaker": "佐藤", "text": "来週の試合も楽しみですね。", "pause": 6.0},
])

# 合意形成が必要（11発話）
_reg("consensus_needed", [
    {"speaker": "松井", "text": "リリースは来週金曜日にしましょう。"},
    {"speaker": "田中", "text": "来週は早すぎます。テストが間に合わないかもしれません。"},
    {"speaker": "佐藤", "text": "でも顧客への約束があるので遅らせられないです。"},
    {"speaker": "田中", "text": "品質を犠牲にしてまでリリースすべきじゃないと思います。"},
    {"speaker": "松井", "text": "品質は後からパッチで直せばいいでしょう。"},
    {"speaker": "田中", "text": "パッチ前提のリリースは信頼を損ないますよ。"},
    {"speaker": "佐藤", "text": "でも顧客が待ってるんですよ。約束は約束です。"},
    {"speaker": "田中", "text": "約束を守るために壊れた製品を出すのは本末転倒です。"},
    {"speaker": "松井", "text": "堂々巡りになってますね。"},
    {"speaker": "佐藤", "text": "どちらの言い分も分かるんですけどね。"},
    {"speaker": "松井", "text": "じゃあもう来週金曜で決定ということで。次の議題に移りましょう。", "pause": 6.0},
])

# 正常な議論・介入不要（12発話）
_reg("healthy", [
    {"speaker": "松井", "text": "認証方式はOAuth2.0にしたいと思います。"},
    {"speaker": "田中", "text": "いいと思います。ただ、トークンのリフレッシュ戦略はどうしますか？"},
    {"speaker": "佐藤", "text": "サイレントリフレッシュがUX的にはベストですが、セキュリティ面が気になります。"},
    {"speaker": "松井", "text": "確かに。リフレッシュトークンのローテーションを入れれば緩和できるかと。"},
    {"speaker": "田中", "text": "それならアクセストークンの有効期限も短めにできますね。15分くらい？"},
    {"speaker": "佐藤", "text": "15分でいいと思います。"},
    {"speaker": "松井", "text": "オフラインアクセスが必要な場合はどうしましょうか。"},
    {"speaker": "田中", "text": "リフレッシュトークンの長期保存が必要になりますね。暗号化してストレージに保存しましょう。"},
    {"speaker": "佐藤", "text": "デバイス単位でトークンを発行して、紛失時に個別に失効できるようにしたいです。"},
    {"speaker": "松井", "text": "それいいですね。デバイス管理画面も作りましょう。"},
    {"speaker": "田中", "text": "管理画面は次のスプリントでいいですか？認証のコア部分を先にやりたいです。"},
    {"speaker": "松井", "text": "はい、ではOAuth2.0のコア実装を今スプリント、管理画面を次スプリントで。", "pause": 6.0},
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
        print("\nテスト実行例:")
        print(f"  uv run python -m das.asr.live --wav {paths[0]}")
    elif args.scenario:
        path = generate_scenario_wav(args.scenario, args.output_dir, args.pause)
        print("\nテスト実行:")
        print(f"  uv run python -m das.asr.live --wav {path}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
