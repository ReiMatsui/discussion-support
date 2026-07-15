#!/usr/bin/env python3
"""TalkBank Sakura（対面4人グループ会話）を評価用に整形する.

データ源: CABank Japanese Sakura Corpus (doi:10.21415/T5M90R)。
大学の会議室で収録された4人の実グループ会話×18本。CHAT転記の一部の発話に
ミリ秒タイムスタンプ（\\x15start_end\\x15）が付く（会話あたり60〜118個）。
時間付き発話のみを正解タイムラインのアンカーとして使う（まばらだが採点には十分。
分布の偏りは採点対象が減るだけで、誤った正解にはならない）。
引用義務: Miyata, S. et al. (2009). CABank Japanese Sakura Corpus.

使い方:
    uv run python eval/prep_sakura.py <素材ディレクトリ> sakura01 sakura02 ...
    （素材ディレクトリに sakuraNN.cha と sakuraNN.(mp3|mp4|wav) を置く。
      音声変換に ffmpeg を使用）
出力:
    data/sakura/<name>.wav      16kHz mono PCM（--wav 入力用）
    data/sakura/<name>.gt.json  timeline形式GT（eval_speaker_gt.py 互換）

その後:
    uv run das listen-soniox --skip-docs --max-speakers 4 \
        --soniox-args "--no-agent" --wav data/sakura/sakura01.wav
    uv run python eval/eval_speaker_gt.py data/sakura/sakura01.gt.json <セッション名>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "sakura"
SR = 16000
_TS = re.compile(r"\x15?(\d+)_(\d+)\x15?\s*[.!?]?\s*$")
_EXTS = (".wav", ".mp3", ".mp4", ".mov", ".m4a", ".mpg", ".avi")


def parse_cha(text: str) -> tuple[list[dict], list[str]]:
    """時間付き発話のみ (speaker, start_ms, end_ms) を抜き出す.

    話者 "ALL"（全員同時の笑い等）は単一話者の正解にならないため除外。
    """
    timeline, speakers = [], set()
    cur, buf = None, ""
    for line in text.splitlines():
        if line.startswith("*"):
            cur = line[1:line.index(":")]
            buf = line[line.index(":") + 1:].strip()
        elif line.startswith("\t") and cur:
            buf += " " + line.strip()
        else:
            cur = None
            continue
        m = _TS.search(buf)
        if m and cur and cur != "ALL":
            timeline.append({"speaker": cur,
                             "start_ms": int(m.group(1)),
                             "end_ms": int(m.group(2))})
            speakers.add(cur)
            buf = ""
    return timeline, sorted(speakers)


def convert(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)],
        check=True)


def main(src_dir: str, names: list[str]) -> None:
    src = Path(src_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        cha = (src / f"{name}.cha").read_text(encoding="utf-8")
        tl, speakers = parse_cha(cha)
        if not tl:
            print(f"= {name}: 時間付き発話なし、スキップ")
            continue
        media = next((src / f"{name}{e}" for e in _EXTS
                      if (src / f"{name}{e}").exists()), None)
        if media is None:
            print(f"= {name}: 音声ファイルが見つかりません"
                  f"（{name}.mp3/.mp4 等を {src} に置いてください）")
        else:
            convert(media, OUT / f"{name}.wav")
        gt = {"session": name,
              "source": "CABank Japanese Sakura Corpus (doi:10.21415/T5M90R)",
              "kind": "timeline", "speakers": speakers, "timeline": tl}
        (OUT / f"{name}.gt.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
        total = sum(t["end_ms"] - t["start_ms"] for t in tl) / 1000
        print(f"= {name}: アンカー{len(tl)}区間 / 話者{speakers} / "
              f"ラベル済み{total/60:.1f}分"
              + ("" if media is None else f" / 音声変換OK → data/sakura/{name}.wav"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
