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

    # 過去の収録セッションの、あるラベルの発話から（名前=セッション/ラベル[@開始-終了 秒]）
    # 別のセッションで登録して本番のセッションを流す＝「事前に声を登録した」状況の再現
    uv run python scripts/enroll_voices.py --voices data/voices_zemi.json --seconds 60 \\
        --from-session 伊藤先生=2026-06-25_140652/伊藤先生 \\
        --from-session 岡田さん=2026-06-25_1520/岡田さん@600-

--chiba は Morph の時刻で「その人が話している区間」だけをその人のチャンネルから
切り出して繋ぐので、他人の声の回り込みが少ないクリーンな登録音声になる。
--from-session は transcripts/<セッション>.turns.jsonl のラベルで区間を選ぶので、
そのランの話者ラベルの誤りがそのまま混ざる。作ったクリップは
data/voices/<名前>.wav に残すので、登録前に一度聞いて確かめられる。
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


def parse_from_session(spec: str) -> tuple[str, str, str, float | None, float | None]:
    """'名前=セッション/ラベル' または '…/ラベル@開始-終了' を分解する."""
    if "=" not in spec or "/" not in spec.split("=", 1)[1]:
        raise SystemExit(f"--from-session は 名前=セッション/ラベル[@開始-終了] の形で: {spec}")
    name, rest = spec.split("=", 1)
    start = end = None
    if "@" in rest:
        rest, rng = rest.rsplit("@", 1)
        a, _, b = rng.partition("-")
        start = float(a) if a else None
        end = float(b) if b else None
    session, label = rest.split("/", 1)
    return name.strip(), session.strip(), label.strip(), start, end


def session_segments(session: str, label: str, *, gap: float = 0.3,
                     transcripts: Path | None = None) -> list[tuple[float, float]]:
    """収録セッションの turns から、そのラベルが付いた発話区間（秒）を順に返す."""
    import json
    tdir = transcripts or (ROOT / "transcripts")
    path = tdir / f"{session}.turns.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} がありません")
    segs: list[tuple[float, float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t.get("speaker") != label or t.get("end_ms") is None:
                continue
            s, e = t["ms"] / 1000.0, t["end_ms"] / 1000.0
            if e <= s:
                continue
            if segs and s - segs[-1][1] <= gap:
                segs[-1] = (segs[-1][0], max(segs[-1][1], e))
            else:
                segs.append((s, e))
    return segs


def parse_from_gt(spec: str) -> tuple[str, str, str, str | None]:
    """'名前=GTのjson/コード' または '…/コード:wavパス' を分解する."""
    if "=" not in spec or "/" not in spec.split("=", 1)[1]:
        raise SystemExit(f"--from-gt は 名前=eval/gt_X.json/S1[:wavパス] の形で: {spec}")
    name, rest = spec.split("=", 1)
    wav = None
    if ":" in rest:
        rest, wav = rest.rsplit(":", 1)
    gt_path, code = rest.rsplit("/", 1)
    return name.strip(), gt_path.strip(), code.strip(), (wav.strip() if wav else None)


def gt_segments(gt_path: Path, code: str, *, gap: float = 0.3,
                root: Path | None = None) -> tuple[list[tuple[float, float]], Path | None]:
    """annotate.py の正解（labels）から、そのコードが付いた区間（秒）と音声の場所を返す.

    区間は、収録セッションなら transcripts/<session>.turns.jsonl の turn_id から、
    任意の音声（無音で自動区切り）なら eval/segments_<name>.json から引く。
    音声は transcripts/<session>.wav か eval/_annot_audio/<name>*.wav を探す（無ければ None）。
    """
    import json
    root = root or ROOT
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    session = gt.get("session") or gt_path.stem.removeprefix("gt_")
    labels = {str(k): v for k, v in (gt.get("labels") or {}).items()}
    raw: list[tuple[float, float]] = []
    turns = root / "transcripts" / f"{session}.turns.jsonl"
    segs_json = root / "eval" / f"segments_{session}.json"
    if turns.exists():
        with open(turns, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if labels.get(str(t["turn_id"])) == code and t.get("end_ms"):
                    raw.append((t["ms"] / 1000.0, t["end_ms"] / 1000.0))
    elif segs_json.exists():
        for sg in json.loads(segs_json.read_text(encoding="utf-8")):
            if labels.get(str(sg["id"])) == code:
                raw.append((float(sg["start"]), float(sg["end"])))
    else:
        raise SystemExit(f"{gt_path}: 区間の元（transcripts/{session}.turns.jsonl か "
                         f"eval/segments_{session}.json）が見つかりません")
    segs: list[tuple[float, float]] = []
    for s, e in sorted(raw):
        if e <= s:
            continue
        if segs and s - segs[-1][1] <= gap:
            segs[-1] = (segs[-1][0], max(segs[-1][1], e))
        else:
            segs.append((s, e))
    wav = root / "transcripts" / f"{session}.wav"
    if not wav.exists():
        cands = sorted((root / "eval" / "_annot_audio").glob(f"{session}*.wav")) \
            if (root / "eval" / "_annot_audio").exists() else []
        wav = cands[0] if cands else None
    return segs, wav


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
    p.add_argument("--from-session", action="append", default=[],
                   metavar="名前=セッション/ラベル[@開始-終了]",
                   help="過去の収録セッション（transcripts/）で、そのラベルが付いた発話を"
                        "--seconds 秒ぶん繋いで登録する。@開始-終了 は使う範囲（秒）。"
                        "クリップは data/voices/<名前>.wav に残す")
    p.add_argument("--from-gt", action="append", default=[],
                   metavar="名前=GTのjson/コード[:wavパス]",
                   help="eval/annotate.py で耳で付けた正解から、そのコード（S1 など）の区間を"
                        "--seconds 秒ぶん繋いで登録する。当時のラベルが信用できない録音向け。"
                        "音声は transcripts/ か eval/_annot_audio/ から探す（:wavパス で明示も可）")
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
    for spec in a.from_session:
        name, session, label, s, e = parse_from_session(spec)
        wav = read_wav_16k(ROOT / "transcripts" / f"{session}.wav")
        segs = [(x, y) for x, y in session_segments(session, label)
                if e is None or y <= e]
        # ラベル付きの区間は他人の声が混ざりやすいので、1 秒未満の区間は使わない
        clip, until = build_clip(wav, segs, a.seconds, min_seg=1.0, after=s or 0.0)
        if clip.size == 0:
            print(f"# {name}: {session} に「{label}」の使える発話がありません"
                  f"（ラベル名は transcripts/{session}.md の話者欄と同じ表記で）", flush=True)
            continue
        out = ROOT / "data" / "voices" / f"{name}.wav"
        save_wav_16k(out, clip)
        jobs.append((name, clip, f"{session} の「{label}」 発話 {clip.size / SR:.1f}s"
                     f"（{s or 0:.0f}〜{until:.0f} 秒の範囲）→ {out.relative_to(ROOT)}"))
    for spec in a.from_gt:
        name, gt_path, code, wav_hint = parse_from_gt(spec)
        segs, wav_path = gt_segments(ROOT / gt_path if not Path(gt_path).is_absolute() else Path(gt_path), code)
        if wav_hint:
            wav_path = Path(wav_hint)
        if wav_path is None or not wav_path.exists():
            raise SystemExit(f"{name}: 音声が見つかりません（名前=GT/コード:wavパス で指定してください）")
        clip, until = build_clip(read_wav_16k(wav_path), segs, a.seconds, min_seg=1.0)
        if clip.size == 0:
            print(f"# {name}: {gt_path} に「{code}」の使える区間がありません", flush=True)
            continue
        out = ROOT / "data" / "voices" / f"{name}.wav"
        save_wav_16k(out, clip)
        jobs.append((name, clip, f"{gt_path} の {code} 区間 {clip.size / SR:.1f}s"
                     f"（〜{until:.0f} 秒）→ {out.relative_to(ROOT)}"))
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
        raise SystemExit("--add か --chiba か --from-session か --from-gt か --record を指定してください")

    # 同じ名前を複数回渡したら（--add 名前=…@10-40 --add 名前=…@90-120 など）繋いで 1 人ぶんにする
    merged: dict[str, tuple[list[np.ndarray], list[str]]] = {}
    for name, wav, note in jobs:
        merged.setdefault(name, ([], []))
        merged[name][0].append(wav)
        merged[name][1].append(note)
    jobs = [(name, np.concatenate(ws), " + ".join(ns)) for name, (ws, ns) in merged.items()]

    for name, wav, note in jobs:
        if wav.size < SR * 2:
            print(f"# {name}: 音声が短すぎます（{wav.size / SR:.1f}s）— 2 秒以上必要", flush=True)
            continue
        ok = vp.enroll_from_audio(name, wav)
        print(f"# {name}: {'登録' if ok else '失敗'}  {note}", flush=True)
    print(f"# 保存: {a.voices}  登録済み: {', '.join(vp.all_profile_names())}", flush=True)


if __name__ == "__main__":
    main()
