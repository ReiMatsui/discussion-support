#!/usr/bin/env python3
"""単独で話した音声から声紋を作り、voices.json に事前登録する.

対面実験の「声の登録」と同じ経路（VoiceProfiles.enroll_from_audio）を、
ファイルから一括で通すための道具。作った voices.json は
`das listen-soniox --voices <ファイル> --activate all` で照合対象になる。

    # 任意の音声から（名前=パス。@開始-終了 で秒の範囲を切り出せる）
    uv run python scripts/enroll_voices.py --voices voices_rehacq.json \\
        --add 高橋=clips/takahashi.wav --add 佐藤=clips/sato.wav@30-90

    # 千葉コーパスの話者別ヘッドセット録音から（各話者の発話を先頭から60秒ぶん）
    uv run python scripts/enroll_voices.py --voices data/chiba/voices_chiba0132_e60.json \\
        --chiba chiba0132 --seconds 60

    # 実験当日: その場のマイクで一人ずつ録って登録（30秒。録音は data/voices/<名前>.wav に残す）
    uv run python scripts/enroll_voices.py --voices voices.json --record 田中 --seconds 30
    uv run python scripts/enroll_voices.py --voices voices.json --record 佐藤 --seconds 30

--chiba は Morph の時刻で「その人が話している区間」だけをその人のチャンネルから
切り出して繋ぐので、他人の声の回り込みが少ないクリーンな登録音声になる。
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from das.asr.live._constants import SR  # noqa: E402

CORPUS = ROOT / "data" / "chiba" / "Chiba3Party"
SPEAKERS = ("A", "B", "C")


def parse_add(spec: str) -> tuple[str, str, float | None, float | None]:
    """'名前=パス' または '名前=パス@開始-終了' を分解する."""
    if "=" not in spec:
        raise SystemExit(f"--add は 名前=パス の形で指定してください: {spec}")
    name, rest = spec.split("=", 1)
    start = end = None
    if "@" in rest:
        rest, rng = rest.rsplit("@", 1)
        a, _, b = rng.partition("-")
        start = float(a) if a else None
        end = float(b) if b else None
    return name.strip(), rest.strip(), start, end


def cut(wav: np.ndarray, start: float | None, end: float | None) -> np.ndarray:
    s = int((start or 0.0) * SR)
    e = int(end * SR) if end is not None else wav.size
    return wav[s:e]


def read_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: 16kHz mono 16bit の PCM WAV にしてください"
                             f"（ffmpeg -i in -ar 16000 -ac 1 out.wav）")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0


def speech_segments(conv: str, who: str) -> list[tuple[float, float]]:
    """Morph の時刻から、その話者が話している区間（秒）を先頭から順に返す."""
    segs: list[tuple[float, float]] = []
    with open(CORPUS / "Morph" / f"{conv}.csv", encoding="cp932") as f:
        for r in csv.DictReader(f):
            if (r.get("who") or "").strip() != who:
                continue
            try:
                s, e = float(r["startTime"]), float(r["endTime"])
            except (KeyError, ValueError):
                continue
            if e <= s:
                continue
            if segs and s - segs[-1][1] <= 0.3:
                segs[-1] = (segs[-1][0], max(segs[-1][1], e))
            else:
                segs.append((s, e))
    return segs


def record_from_mic(seconds: float, device: str | None = None) -> np.ndarray:
    """その場のマイクから seconds 秒録る（16kHz mono float32）. 3 秒のカウントダウン付き."""
    import time
    import sounddevice as sd
    dev = None
    if device:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and device.lower() in d["name"].lower():
                dev = i
                break
        else:
            raise SystemExit(f"入力デバイスが見つかりません: {device}")
    info = sd.query_devices(dev if dev is not None else sd.default.device[0])
    print(f"# 入力デバイス: {info['name']}。{seconds:.0f} 秒録ります。"
          "普段の声で、自己紹介や今日の予定など何でも話し続けてください", flush=True)
    for i in (3, 2, 1):
        print(f"  {i}…", flush=True)
        time.sleep(1)
    print("  ● 録音中", flush=True)
    x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=dev)
    sd.wait()
    print("  ■ 終了", flush=True)
    return np.asarray(x, dtype="float32").reshape(-1)


def check_level(wav: np.ndarray) -> str | None:
    """録音が登録に使える音量かを見る。問題があれば理由を返す."""
    if wav.size < SR * 2:
        return f"短すぎます（{wav.size / SR:.1f}s）"
    rms = float(np.sqrt((wav ** 2).mean() + 1e-12))
    db = 20 * np.log10(rms + 1e-9)
    if db < -50:
        return f"ほぼ無音です（{db:.1f} dBFS）。ミュートやデバイス選択を確認"
    if db < -40:
        return f"小さいです（{db:.1f} dBFS）。マイクに近づいて録り直しを推奨"
    if float(np.abs(wav).max()) > 0.98:
        return "クリップしています。入力ゲインを下げて録り直しを推奨"
    return None


def save_wav_16k(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(wav, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())


def build_clip(wav: np.ndarray, segs: list[tuple[float, float]], seconds: float,
               min_seg: float = 0.5, after: float = 0.0) -> tuple[np.ndarray, float]:
    """発話区間を繋いで seconds ぶんの登録音声を作る.

    短すぎる区間（min_seg 未満）は相槌の可能性が高いので使わない。after を
    渡すと、その秒より後に始まる区間だけを使う（採点する先頭 N 分と登録音声を
    重ねないため。例: 先頭 4 分を採点するなら after=240）。
    戻り値は (音声, 使った区間の末尾の秒)。末尾の秒は「会議の何秒目までを
    登録に使ったか」の記録用。
    """
    parts: list[np.ndarray] = []
    got = 0.0
    used_until = 0.0
    for s, e in segs:
        if s < after or e - s < min_seg:
            continue
        piece = wav[int(s * SR):int(e * SR)]
        if piece.size == 0:
            continue
        parts.append(piece)
        got += piece.size / SR
        used_until = e
        if got >= seconds:
            break
    if not parts:
        return np.zeros(0, dtype="float32"), 0.0
    return np.concatenate(parts), used_until


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--voices", required=True, help="書き出す voices.json")
    p.add_argument("--add", action="append", default=[], metavar="名前=パス[@開始-終了]")
    p.add_argument("--chiba", default=None, metavar="CONV", help="千葉コーパスの会話名（例 chiba0132）")
    p.add_argument("--seconds", type=float, default=60.0, help="--chiba で各話者に使う発話の秒数")
    p.add_argument("--speakers", default="A,B,C", help="--chiba で登録する話者（カンマ区切り）")
    p.add_argument("--after", type=float, default=0.0, metavar="SEC",
                   help="--chiba で、会議の SEC 秒より後の発話だけを登録に使う"
                        "（先頭を採点する再生ランと登録音声を重ねないため）")
    p.add_argument("--record", action="append", default=[], metavar="名前",
                   help="その場のマイクで録って登録する（--seconds 秒。複数可）。"
                        "録音は data/voices/<名前>.wav に残す")
    p.add_argument("--device", default=None, help="--record の入力デバイス名の一部")
    a = p.parse_args(argv)

    from das.asr.live._voice_profiles import VoiceProfiles
    vp = VoiceProfiles(path=a.voices, auto=False)

    jobs: list[tuple[str, np.ndarray, str]] = []
    for spec in a.add:
        name, path, s, e = parse_add(spec)
        wav = cut(read_wav_16k(Path(path)), s, e)
        jobs.append((name, wav, f"{path} {wav.size / SR:.1f}s"))
    if a.chiba:
        for who in [x.strip() for x in a.speakers.split(",") if x.strip()]:
            if who not in SPEAKERS:
                raise SystemExit(f"--speakers は A,B,C の中から: {who}")
            wav = read_wav_16k(CORPUS / "Wav1" / f"{a.chiba}-{who}.wav")
            clip, until = build_clip(wav, speech_segments(a.chiba, who), a.seconds, after=a.after)
            jobs.append((who, clip, f"{a.chiba}-{who} 発話 {clip.size / SR:.1f}s"
                         f"（会議の {a.after:.0f}〜{until:.0f} 秒の範囲）"))
    for name in a.record:
        name = name.strip()
        wav = record_from_mic(a.seconds, a.device)
        problem = check_level(wav)
        if problem:
            print(f"# {name}: 録音に問題があります: {problem}", flush=True)
        out = ROOT / "data" / "voices" / f"{name}.wav"
        save_wav_16k(out, wav)
        jobs.append((name, wav, f"マイク録音 {wav.size / SR:.0f}s → {out.relative_to(ROOT)}"))
    if not jobs:
        raise SystemExit("--add か --chiba か --record を指定してください")

    for name, wav, note in jobs:
        if wav.size < SR * 2:
            print(f"# {name}: 音声が短すぎます（{wav.size / SR:.1f}s）— 2 秒以上必要", flush=True)
            continue
        ok = vp.enroll_from_audio(name, wav)
        print(f"# {name}: {'登録' if ok else '失敗'}  {note}", flush=True)
    print(f"# 保存: {a.voices}  登録済み: {', '.join(vp.all_profile_names())}", flush=True)


if __name__ == "__main__":
    main()
