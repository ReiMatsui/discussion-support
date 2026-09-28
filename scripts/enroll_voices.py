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


def build_clip(wav: np.ndarray, segs: list[tuple[float, float]], seconds: float,
               min_seg: float = 0.5) -> tuple[np.ndarray, float]:
    """発話区間を先頭から繋いで seconds ぶんの登録音声を作る.

    短すぎる区間（min_seg 未満）は相槌の可能性が高いので使わない。
    戻り値は (音声, 使った区間の末尾の秒)。末尾の秒は「会議の何秒目までを
    登録に使ったか」の記録用（採点で先頭を除外したいときに使う）。
    """
    parts: list[np.ndarray] = []
    got = 0.0
    used_until = 0.0
    for s, e in segs:
        if e - s < min_seg:
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
            clip, until = build_clip(wav, speech_segments(a.chiba, who), a.seconds)
            jobs.append((who, clip, f"{a.chiba}-{who} 発話 {clip.size / SR:.1f}s（会議の {until:.0f} 秒目まで）"))
    if not jobs:
        raise SystemExit("--add か --chiba を指定してください")

    for name, wav, note in jobs:
        if wav.size < SR * 2:
            print(f"# {name}: 音声が短すぎます（{wav.size / SR:.1f}s）— 2 秒以上必要", flush=True)
            continue
        ok = vp.enroll_from_audio(name, wav)
        print(f"# {name}: {'登録' if ok else '失敗'}  {note}", flush=True)
    print(f"# 保存: {a.voices}  登録済み: {', '.join(vp.all_profile_names())}", flush=True)


if __name__ == "__main__":
    main()
